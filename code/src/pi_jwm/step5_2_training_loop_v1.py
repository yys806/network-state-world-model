"""STEP 5.2 CPU training loop for the frozen Definition 05 primitives.

This module connects the already accepted encoder, structured RSSM, future
target teachers and mask-aware loss primitives.  It deliberately remains a
development-only CPU loop: it supports a few optimizer steps, validation and
checkpoint/resume, but it does not claim formal training, GPU performance or a
formal dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import (
    PhysicalTopologyConfig,
    build_typed_dual_graph_batch,
    load_typed_dual_graph_batch,
)
from pi_jwm.step4_3b_dual_graph_encoder_v1 import (
    DualGraphEncoderConfig,
    PIJointGraphEncoder,
    fit_encoder_normalization_stats,
)
from pi_jwm.step4_4_structured_rssm_world_model_v1 import (
    StructuredRSSMConfig,
    StructuredRSSMWorldModel,
)
from pi_jwm.step5_1b_posterior_loss_metric_v1 import (
    Step5_1BConfig,
    TargetEncoder,
    FuturePosterior,
    diagonal_gaussian_kl,
    masked_family_mse,
)


ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922"
TARGET = ROOT / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921"
UPSTREAM_STATS = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/train_normalization_stats.json"
GRAPH_PACKAGE = ROOT / "code/artifacts/protocols/pi_jwm_step5_1d_unified_model_chain_v1_20260922/unified_typed_dual_graph.npz"
TARGET_PACKAGE = TARGET / "tensor.npz"
SCHEMA_VERSION = "PI-JWM-Step-5.2-Training-Loop-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_tree(value: Any, device: torch.device | None = None) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return torch.from_numpy(value.copy()).bool().to(device=device)
        if np.issubdtype(value.dtype, np.integer):
            return torch.from_numpy(value.copy()).long().to(device=device)
        return torch.from_numpy(value.copy()).float().to(device=device)
    if isinstance(value, Mapping):
        return {key: _torch_tree(item, device=device) for key, item in value.items()}
    return value


def _slice_mapping(value: Mapping[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    def take(item: Any) -> Any:
        if isinstance(item, torch.Tensor) and item.ndim and item.shape[0] == batch_size:
            return item[index:index + 1]
        if isinstance(item, Mapping):
            return {key: take(child) for key, child in item.items()}
        return item
    return take(value)


def _cat_mappings(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot concatenate an empty mapping sequence")
    keys = values[0].keys()
    output: dict[str, Any] = {}
    for key in keys:
        items = [value[key] for value in values]
        if isinstance(items[0], torch.Tensor):
            output[key] = torch.cat(items, dim=0)
        elif isinstance(items[0], Mapping):
            output[key] = _cat_mappings(items)  # type: ignore[arg-type]
        else:
            output[key] = items[0]
    return output


def _pad_action_rows(actions: Sequence[Mapping[str, torch.Tensor]], key: str, width: int, value_dim: int | None = None) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for action in actions:
        value = action[key]
        pad = width - value.shape[1]
        if pad:
            shape = (1, pad) if value.ndim == 2 else (1, pad, value.shape[-1])
            fill = -1 if key.endswith("_index") else 0.0
            value = torch.cat((value, torch.full(shape, fill, dtype=value.dtype, device=value.device)), dim=1)
        rows.append(value)
    return torch.cat(rows, dim=0)


def _stack_actions(actions: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not actions:
        raise ValueError("no actions to stack")
    output: dict[str, torch.Tensor] = {}
    for key in ("mobility_entity_index", "comm_relation_index", "comp_agent_index", "comp_task_index", "route_task_index", "route_flow_index"):
        width = max(action[key].shape[1] for action in actions)
        output[key] = _pad_action_rows(actions, key, width)
    for key in ("mobility_values", "comm_values", "comp_values", "route_values"):
        width = max(action[key].shape[1] for action in actions)
        output[key] = _pad_action_rows(actions, key, width)
    output["comm_allocation_mask"] = torch.cat([action["comm_allocation_mask"] for action in actions], dim=0)
    return output


def count_available_validation_windows(motion_mask: torch.Tensor, csi_mask: torch.Tensor) -> int:
    """Count windows with at least one valid target element in both families."""
    if motion_mask.ndim < 2 or csi_mask.ndim < 2 or motion_mask.shape[0] == 0 or motion_mask.shape[0] != csi_mask.shape[0]:
        raise ValueError("Motion and CSI masks must have the same nonempty batch axis")
    motion_available = motion_mask.reshape(motion_mask.shape[0], -1).bool().any(dim=1)
    csi_available = csi_mask.reshape(csi_mask.shape[0], -1).bool().any(dim=1)
    return int((motion_available & csi_available).sum().item())


def aggregate_validation_horizon_rows(horizon_rows: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate validation errors over all valid elements for each horizon.

    Motion and CSI are normalized independently before the frozen 0.5/0.5
    prediction combination.  A sample missing one family therefore does not
    remove the other family's valid elements from the global numerator/count.
    """
    per_horizon: list[dict[str, Any]] = []
    for horizon, rows in sorted(horizon_rows.items()):
        motion_numerator = float(sum(float(row.get("motion_numerator", 0.0)) for row in rows))
        csi_numerator = float(sum(float(row.get("csi_numerator", 0.0)) for row in rows))
        motion_count = int(sum(int(row.get("motion_count", 0)) for row in rows))
        csi_count = int(sum(int(row.get("csi_count", 0)) for row in rows))
        if motion_count <= 0 or csi_count <= 0:
            continue
        l_mot = motion_numerator / motion_count
        l_csi = csi_numerator / csi_count
        per_horizon.append({
            "horizon": int(horizon),
            "motion_numerator": motion_numerator,
            "motion_count": motion_count,
            "csi_numerator": csi_numerator,
            "csi_count": csi_count,
            "L_Mot": l_mot,
            "L_CSI": l_csi,
            "L_Pred": 0.5 * l_mot + 0.5 * l_csi,
            "available_sample_count": sum(
                int(row["available_sample_count"]) if "available_sample_count" in row
                else int(bool(row.get("available", False))) for row in rows
            ),
        })
    return per_horizon


@dataclass(frozen=True)
class CurriculumConfig:
    """Explicit horizon schedule; ``start_steps`` are global optimizer steps."""

    horizons: tuple[int, ...] = (1, 2, 4)
    start_steps: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not self.horizons or len(self.horizons) != len(self.start_steps):
            raise ValueError("curriculum horizons and start_steps must have equal non-zero length")
        if any(int(h) <= 0 for h in self.horizons) or any(int(s) < 0 for s in self.start_steps):
            raise ValueError("curriculum values must be positive horizons and non-negative steps")
        if tuple(self.start_steps) != tuple(sorted(self.start_steps)):
            raise ValueError("curriculum start_steps must be sorted")

    def horizon_for_step(self, global_step: int, max_horizon: int) -> int:
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        selected = self.horizons[0]
        for start, horizon in zip(self.start_steps, self.horizons):
            if global_step >= start:
                selected = horizon
        return min(int(selected), int(max_horizon))


@dataclass(frozen=True)
class KLSchedule:
    target_beta: float = 1.0
    warmup_steps: int = 10
    free_bits: float = 0.1

    def __post_init__(self) -> None:
        if self.target_beta < 0 or self.warmup_steps < 0 or self.free_bits < 0:
            raise ValueError("KL schedule values must be non-negative")

    def beta_at(self, global_step: int) -> float:
        if self.warmup_steps == 0:
            return float(self.target_beta)
        progress = min(max(float(global_step), 0.0) / float(self.warmup_steps), 1.0)
        return float(self.target_beta) * progress


@dataclass(frozen=True)
class Step52TrainingConfig:
    seed: int = 5201
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1
    gradient_clip_norm: float = 1.0
    max_epochs: int = 1
    max_steps: int = 2
    stage1_steps: int = 1
    checkpoint_patience: int = 3
    max_horizon: int = 2
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    kl_schedule: KLSchedule = field(default_factory=KLSchedule)
    rssm: StructuredRSSMConfig = field(default_factory=StructuredRSSMConfig)
    encoder: DualGraphEncoderConfig = field(default_factory=DualGraphEncoderConfig)
    csi_decoder_output_space: str = "raw_db"
    csi_decoder_bias_init_policy: str = "train_only_csi_mean"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.batch_size <= 0:
            raise ValueError("invalid optimizer configuration")
        if self.gradient_clip_norm < 0 or self.max_epochs <= 0 or self.max_steps <= 0:
            raise ValueError("invalid training limits")
        if self.stage1_steps < 0 or self.checkpoint_patience < 0 or self.max_horizon <= 0:
            raise ValueError("invalid stage/curriculum configuration")
        if self.csi_decoder_output_space != "raw_db":
            raise ValueError("STEP 5.3E v1 requires raw_db CSI decoder output")
        if self.csi_decoder_bias_init_policy != "train_only_csi_mean":
            raise ValueError("STEP 5.3E v1 requires train_only_csi_mean bias initialization")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")


@dataclass
class DevelopmentBundle:
    samples: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    tensor: dict[str, Any]
    graph: dict[str, Any]
    target_tensors: dict[str, torch.Tensor]
    target_normalization: dict[str, Any]
    states: list[dict[str, torch.Tensor]]
    graphs: list[dict[str, torch.Tensor]]
    train_indices: list[int]
    validation_indices: list[int]
    identity: dict[str, Any]

    @classmethod
    def load(cls) -> "DevelopmentBundle":
        # Importing these helpers keeps the 5.1D source-of-truth action/state
        # adapter in one place; no historical training path is reused.
        from build_step5_1d_unified_model_chain_v1 import build_action, build_state
        from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch

        tensor_path = ROOT / "code/artifacts/protocols/pi_jwm_step5_1d_unified_model_chain_v1_20260922/unified_flow_tensor_package.npz"
        if not tensor_path.exists():
            tensor_path = BUNDLE / "unified_flow_tensor.npz"
        tensor = load_flow_tensor_batch(tensor_path)
        samples = json.loads((BUNDLE / "unified_flow_samples.json").read_text(encoding="utf-8"))
        targets = json.loads((TARGET / "extended_samples.json").read_text(encoding="utf-8"))
        if len(samples) != len(targets) or len(samples) != int(tensor["entity_presence"].shape[0]):
            raise ValueError("unified sample/target/tensor counts do not agree")
        if GRAPH_PACKAGE.exists():
            graph = load_typed_dual_graph_batch(GRAPH_PACKAGE)
        else:
            graph = build_typed_dual_graph_batch(tensor, PhysicalTopologyConfig(mode="radius_knn", radius_m=1000.0, k=2, self_loops=False))
        with np.load(TARGET_PACKAGE, allow_pickle=False) as target_np:
            target_tensors = {
                key: torch.from_numpy(value.copy()).float() if value.dtype != np.bool_ else torch.from_numpy(value.copy()).bool()
                for key, value in target_np.items()
                if key.startswith("target_")
            }
            target_contract = json.loads(str(target_np["__contract__"]))["contract"]
        states: list[dict[str, torch.Tensor]] = []
        graphs: list[dict[str, torch.Tensor]] = []
        # 5.1D stores source paths relative to the repository.  Historical
        # tests may temporarily change the process working directory, so
        # resolve those provenance paths in this local in-memory view rather
        # than relying on ambient CWD.  No artifact is rewritten.
        tensor_for_state = dict(tensor)
        tensor_for_state["sample_metadata"] = [
            {
                **metadata,
                "source_path": str((ROOT / metadata["source_path"]).resolve()) if not Path(metadata["source_path"]).is_absolute() else metadata["source_path"],
            }
            for metadata in tensor.get("sample_metadata", [])
        ]
        for index, sample in enumerate(samples):
            state, graph_t = build_state(tensor_for_state, graph, index)
            states.append(state)
            graphs.append(graph_t)
        train_indices = [index for index, sample in enumerate(samples) if sample["metadata"].get("split") == "dev_train"]
        validation_indices = [index for index, sample in enumerate(samples) if sample["metadata"].get("split") == "dev_validation"]
        if len(train_indices) != 8 or len(validation_indices) != 4:
            raise ValueError("STEP 5.2 requires the 8/4 unified development split")
        manifest = BUNDLE / "manifest.json"
        identity = {
            "bundle_manifest": str(manifest),
            "bundle_manifest_sha256": _sha(manifest),
            "tensor_path": str(tensor_path),
            "tensor_sha256": _sha(tensor_path),
            "target_path": str(TARGET_PACKAGE),
            "target_sha256": _sha(TARGET_PACKAGE),
            "sample_ids": [sample["metadata"]["sample_id"] for sample in samples],
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(validation_indices),
        }
        # Keep the imported helper reachable for type-checkers and make sure
        # the exact 5.1D adapter was imported, while action construction is
        # intentionally deferred until a recursive step has its current state.
        _ = build_action
        return cls(samples, targets, tensor, graph, target_tensors, target_contract["normalization_parameters"], states, graphs, train_indices, validation_indices, identity)

    @classmethod
    def from_formal_interface(cls, interface: Any) -> "DevelopmentBundle":
        from build_step5_1d_unified_model_chain_v1 import build_state
        from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
        from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
        packages = interface.runtime_package_paths
        samples = json.loads(packages["samples"].read_text(encoding="utf-8"))
        tensor = load_flow_tensor_batch(packages["tensor"])
        graph = load_typed_dual_graph_batch(packages["graph"])
        with np.load(packages["target"], allow_pickle=False) as target_np:
            target_tensors = {key: torch.from_numpy(value.copy()).bool() if value.dtype == np.bool_ else torch.from_numpy(value.copy()).float() for key, value in target_np.items() if key.startswith("target_")}
            target_contract = json.loads(str(target_np["__contract__"]))["contract"]
        targets = [{} for _ in samples]
        tensor_for_state = dict(tensor)
        tensor_for_state["sample_metadata"] = [{**metadata, "source_path": str((ROOT / metadata["source_path"]).resolve()) if not Path(metadata["source_path"]).is_absolute() else metadata["source_path"]} for metadata in tensor.get("sample_metadata", [])]
        states, graphs = [], []
        for index in range(len(samples)):
            state, graph_t = build_state(tensor_for_state, graph, index)
            states.append(state); graphs.append(graph_t)
        train_indices = [index for index, sample in enumerate(samples) if sample.get("metadata", {}).get("split") in {"train", "dev_train"}]
        validation_indices = [index for index, sample in enumerate(samples) if sample.get("metadata", {}).get("split") in {"validation", "dev_validation"}]
        if not train_indices or not validation_indices:
            raise ValueError("runtime packages must include train and validation samples")
        normalization = json.loads(packages["normalization"].read_text(encoding="utf-8"))
        identity = {"dataset_manifest_hash": interface.dataset_manifest_hash, "formal_package_hashes": dict(interface.manifest.get("hashes", {})), "runtime_package_hashes": {name: _sha(path) for name, path in packages.items()}, "sample_ids": [sample["metadata"]["sample_id"] for sample in samples], "train_sample_count": len(train_indices), "validation_sample_count": len(validation_indices)}
        return cls(samples, targets, tensor, graph, target_tensors, normalization.get("normalization_parameters", target_contract["normalization_parameters"]), states, graphs, train_indices, validation_indices, identity)


REQUIRED_STEP52_CHECKS = frozenset({
    "stage1_posterior_teacher", "stage2_prior_recursive", "curriculum_1_to_2",
    "kl_warmup", "free_bits_contract", "optimizer_parameter_groups", "known_rule_excluded",
    "parameter_update", "validation_prior_only", "validation_no_parameter_update",
    "checkpoint_reload", "resume_state", "prior_target_isolation", "target_posterior_sensitivity",
    "future_state_not_rollout_input", "independent_normalization", "finite_and_reproducible",
    "padding_and_noop_contract", "future_return_birth_unsupported", "locked_test_not_accessed",
    "current_posterior_initialization", "current_posterior_target_isolation",
    "validation_future_posterior_isolation", "validation_target_encoder_isolation",
    "global_validation_aggregation", "checkpoint_identity_guards",
})


class Step52Trainer(nn.Module):
    """A small CPU trainer around the accepted 5.1D paired bundle."""

    def __init__(self, config: Step52TrainingConfig, data: DevelopmentBundle, *, encoder_stats: Mapping[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.data = data
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self._seed_everything(config.seed)
        graph_contract = data.graph["contract"]
        tensor_contract = data.tensor["contract"]
        encoder_stats = encoder_stats if encoder_stats is not None else fit_encoder_normalization_stats(data.tensor, data.graph, UPSTREAM_STATS)
        self.encoder = PIJointGraphEncoder(config.encoder, tensor_contract, graph_contract, encoder_stats)
        self.model = StructuredRSSMWorldModel(config.rssm)
        self.initialization_contract = self._initialize_csi_decoder()
        target_config = Step5_1BConfig(target_dim=4, csi_dim=int(tensor_contract["n_comm_rb"]), hidden_dim=config.rssm.d_h, latent_dim=config.rssm.d_z, d_h=config.rssm.d_h, d_z=config.rssm.d_z, d_encoder=config.rssm.d_encoder)
        self.target_encoder = TargetEncoder(target_config)
        self.future_posterior = FuturePosterior(target_config)
        self.posterior_teacher_calls = 0
        self.future_target_encoder_calls = 0
        self.current_posterior_calls = 0
        self.posterior_rollout_calls = 0
        self.last_validation: dict[str, Any] | None = None
        self._move_data_to_device()
        self.to(self.device)
        self._optimizer_groups = self._build_optimizer_groups()
        self.optimizer = torch.optim.AdamW(self._optimizer_groups, lr=config.learning_rate, weight_decay=config.weight_decay)

    @staticmethod
    def _move_tree(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: Step52Trainer._move_tree(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [Step52Trainer._move_tree(item, device) for item in value]
        return value

    def _move_data_to_device(self) -> None:
        self.data.target_tensors = self._move_tree(self.data.target_tensors, self.device)
        self.data.states = self._move_tree(self.data.states, self.device)
        self.data.graphs = self._move_tree(self.data.graphs, self.device)

    def _initialize_csi_decoder(self) -> dict[str, Any]:
        """Apply the frozen train-only raw-CSI mean before optimizer creation."""
        stats = self.data.target_normalization
        resolved = float(stats["csi_mean"])
        with torch.no_grad():
            self.model.csi_decoder[-1].bias.fill_(resolved)
        return {
            "schema_version": "PI-JWM-STEP-5.3E-Initialization-v1",
            "csi_decoder_output_space": self.config.csi_decoder_output_space,
            "csi_decoder_bias_init_policy": self.config.csi_decoder_bias_init_policy,
            "resolved_bias_value": resolved,
            "bias_width": int(self.config.rssm.n_comm_rb),
            "normalization_provenance": {
                "source_split": "dev_train",
                "stats": _jsonable(stats),
                "validation_used": False,
                "future_target_used": False,
                "locked_test_used": False,
            },
            "motion_decoder_initialization_untouched": True,
        }

    @classmethod
    def from_unified_development_bundle(cls, config: Step52TrainingConfig | None = None) -> "Step52Trainer":
        return cls(config or Step52TrainingConfig(), DevelopmentBundle.load())

    @classmethod
    def from_formal_interface(cls, interface: Any, config: Step52TrainingConfig | None = None) -> "Step52Trainer":
        return cls(config or Step52TrainingConfig(), DevelopmentBundle.from_formal_interface(interface))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_optimizer_groups(self) -> list[dict[str, Any]]:
        excluded = ("phy_prior.", "comm_prior.", "vehicle_decoder.", "csi_decoder.", "phy_posterior.", "comm_posterior.")
        dynamics = [(name, parameter) for name, parameter in self.model.named_parameters() if not name.startswith(excluded) and parameter.requires_grad]
        groups = [
            ("encoder", list(self.encoder.named_parameters())),
            ("rssm_dynamics", dynamics),
            ("phy_prior", list(self.model.phy_prior.named_parameters())),
            ("comm_prior", list(self.model.comm_prior.named_parameters())),
            ("current_observation_posterior", [(n, p) for n, p in self.model.named_parameters() if n.startswith(("phy_posterior.", "comm_posterior."))]),
            ("phy_future_posterior", list(self.future_posterior.phy_future_posterior.named_parameters())),
            ("comm_future_posterior", list(self.future_posterior.comm_future_posterior.named_parameters())),
            ("motion_target_encoder", list(self.target_encoder.motion.named_parameters())),
            ("csi_target_encoder", list(self.target_encoder.csi.named_parameters())),
            ("vehicle_motion_decoder", list(self.model.vehicle_decoder.named_parameters())),
            ("csi_decoder", list(self.model.csi_decoder.named_parameters())),
        ]
        seen: set[int] = set()
        optimizer_groups: list[dict[str, Any]] = []
        for name, named in groups:
            parameters: list[nn.Parameter] = []
            for _, parameter in named:
                if id(parameter) in seen:
                    raise ValueError(f"duplicate optimizer parameter in {name}")
                seen.add(id(parameter)); parameters.append(parameter)
            if not parameters:
                raise ValueError(f"empty optimizer group: {name}")
            optimizer_groups.append({"name": name, "params": parameters})
        return optimizer_groups

    def optimizer_parameter_audit(self) -> dict[str, Any]:
        expected = {group["name"]: group for group in self._optimizer_groups}
        groups = {
            name: {"in_optimizer": name in expected, "parameter_count": len(group["params"]), "trainable_count": sum(int(parameter.requires_grad) for parameter in group["params"])}
            for name, group in expected.items()
        }
        model_names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        known_rule_names = [name for name in model_names if name.startswith("deterministic_transition.")]
        return {
            "groups": groups,
            "all_required_groups_present": set(groups) == {"encoder", "rssm_dynamics", "phy_prior", "comm_prior", "current_observation_posterior", "phy_future_posterior", "comm_future_posterior", "motion_target_encoder", "csi_target_encoder", "vehicle_motion_decoder", "csi_decoder"},
            "known_rule_parameter_count": len(known_rule_names),
            "known_rule_parameter_names": known_rule_names,
            "known_rule_parameters_excluded": not known_rule_names and all(parameter.requires_grad for group in self._optimizer_groups for parameter in group["params"]),
            "total_trainable_parameters": sum(parameter.numel() for group in self._optimizer_groups for parameter in group["params"]),
        }

    def _encoder_output(self) -> dict[str, Any]:
        tensor = _torch_tree(self.data.tensor, device=self.device)
        graph = _torch_tree(self.data.graph, device=self.device)
        return self.encoder(tensor, graph)

    @staticmethod
    def _slice_zpi(zpi: Mapping[str, Any], indices: Sequence[int], total: int) -> dict[str, Any]:
        def take(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == total:
                return value[list(indices)]
            if isinstance(value, Mapping):
                return {key: take(item) for key, item in value.items()}
            return value
        return take(zpi)

    def _batch_initial_state(self, indices: Sequence[int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return _cat_mappings([self.data.states[index] for index in indices]), _cat_mappings([self.data.graphs[index] for index in indices])

    def _recursive_rollout(self, indices: Sequence[int], horizon: int, *, stage: str) -> dict[str, Any]:
        if horizon <= 0 or horizon > self.config.max_horizon:
            raise ValueError("rollout horizon is outside configured development support")
        from build_step5_1d_unified_model_chain_v1 import build_action

        state, graph = self._batch_initial_state(indices)
        zpi = self._slice_zpi(self._encoder_output(), indices, len(self.data.samples))
        latent = self.model.initialize_latent(zpi, state, posterior_mode="mean")
        self.current_posterior_calls += 1
        states: list[dict[str, torch.Tensor]] = []
        graphs: list[dict[str, torch.Tensor]] = []
        latents: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        actions: list[dict[str, torch.Tensor]] = []
        state_signatures: list[list[float]] = []
        for step in range(horizon):
            single_actions = [build_action(self.data.samples[index], _slice_mapping(state, row, len(indices)), step)[0] for row, index in enumerate(indices)]
            action = _stack_actions(single_actions)
            latent, state, graph, trace = self.model.one_step(latent, state, graph, action, prior_mode="mean", service_mode="expectation", generator=None)
            # ``learned`` is the raw decoder output used by the deterministic
            # rule transition.  Keep it beside the latent for the prior-only
            # prediction head; it is not fed back as a future observation.
            latent_for_record = dict(latent)
            latent_for_record["learned"] = trace["learned"]
            states.append(state); graphs.append(graph); latents.append(latent_for_record); traces.append(trace); actions.append(action)
            state_signatures.append([float(value.detach().sum()) for value in (state["position"], state["flow_remaining"], state["task_progress"])])
        return {
            "states": states,
            "graphs": graphs,
            "latents": latents,
            "traces": traces,
            "actions": actions,
            "state_signatures": state_signatures,
            "prior_only": True,
            "posterior_used_as_rollout_state": False,
            "recursive_state_feedback": horizon < 2 or any(states[h]["position"].grad_fn is not None or not torch.equal(states[h]["position"], state["position"]) for h in range(min(2, horizon))),
            "future_target_consumed_by_rollout": False,
        }

    def _future_teacher(self, rollout: Mapping[str, Any], indices: Sequence[int], *, count_call: bool = True) -> dict[str, Any]:
        motion = self.data.target_tensors["target_vehicle_motion_normalized"][list(indices)]
        motion_mask = self.data.target_tensors["target_vehicle_motion_mask"][list(indices)]
        csi = self.data.target_tensors["target_comm_csi_normalized"][list(indices)]
        csi_mask = self.data.target_tensors["target_comm_csi_mask"][list(indices)]
        horizon = len(rollout["latents"])
        h_phy = torch.stack([item["h"]["physical"] for item in rollout["latents"]], dim=1)
        h_comm = torch.stack([item["h"]["communication"] for item in rollout["latents"]], dim=1)
        e_motion = self.target_encoder.motion(motion[:, :horizon], motion_mask[:, :horizon])
        e_csi = self.target_encoder.csi(csi[:, :horizon], csi_mask[:, :horizon])
        self.future_target_encoder_calls += 1
        if count_call:
            self.posterior_teacher_calls += 1
        q = self.future_posterior(h_phy, h_comm, e_motion, e_csi)
        return {"q": q, "motion": motion[:, :horizon], "motion_mask": motion_mask[:, :horizon], "csi": csi[:, :horizon], "csi_mask": csi_mask[:, :horizon]}

    def _prediction_and_loss(self, rollout: Mapping[str, Any], teacher: Mapping[str, Any] | None, beta_kl: float, *, stage: str) -> dict[str, Any]:
        target = teacher if teacher is not None else self._targets_for_rollout(rollout)
        motion_target, motion_mask = target["motion"], target["motion_mask"]
        csi_target, csi_mask = target["csi"], target["csi_mask"]
        motion_predictions: list[torch.Tensor] = []
        csi_predictions: list[torch.Tensor] = []
        phy_priors: list[Mapping[str, torch.Tensor]] = []
        comm_priors: list[Mapping[str, torch.Tensor]] = []
        for horizon_index, latent in enumerate(rollout["latents"]):
            if stage == "posterior_assisted_warmup":
                if teacher is None:
                    raise ValueError("stage 1 requires the future posterior teacher")
                phy_z = teacher["q"]["physical"].mean[:, horizon_index]
                comm_z = teacher["q"]["communication"].mean[:, horizon_index]
                h_phy = latent["h"]["physical"]
                h_comm = latent["h"]["communication"]
                motion_predictions.append(self.model.vehicle_decoder(torch.cat((h_phy, phy_z), dim=-1)) )
                csi_predictions.append(self.model.csi_decoder(torch.cat((h_comm, comm_z), dim=-1)))
            else:
                motion_predictions.append(latent["learned"]["vehicle_motion"])
                csi_predictions.append(latent["learned"]["csi"])
            phy_priors.append(latent["prior"]["physical"])
            comm_priors.append(latent["prior"]["communication"])
        motion_raw = torch.stack(motion_predictions, dim=1)
        csi_raw = torch.stack(csi_predictions, dim=1)
        params = self.data.target_normalization
        motion_pred = torch.zeros_like(motion_raw)
        motion_pred[..., :3] = motion_raw[..., :3] / motion_raw.new_tensor(params["position_std"])
        motion_pred[..., 3] = (motion_raw[..., 3] - float(params["speed_mean"])) / float(params["speed_std"])
        csi_pred = (csi_raw - float(params["csi_mean"])) / float(params["csi_std"])
        motion_loss, motion_count = masked_family_mse(motion_pred, motion_target, motion_mask)
        csi_loss, csi_count = masked_family_mse(csi_pred, csi_target, csi_mask)
        l_pred = 0.5 * motion_loss + 0.5 * csi_loss
        q = teacher["q"] if teacher is not None else None
        if q is None:
            zero = l_pred.new_zeros(())
            raw_phy = adjusted_phy = raw_comm = adjusted_comm = zero
            kl = zero
            phy_count = comm_count = l_pred.new_zeros(())
        else:
            p_phy = {key: torch.stack([item[key] for item in phy_priors], dim=1) for key in ("mean", "log_std")}
            p_comm = {key: torch.stack([item[key] for item in comm_priors], dim=1) for key in ("mean", "log_std")}
            phy_eligibility = motion_mask.any(dim=-1)
            comm_valid = torch.stack([item["comm_presence"] & item["comm_validity"] & item["comm_wireless_mask"] for item in rollout["states"]], dim=1)
            comm_eligibility = csi_mask.any(dim=-1) & comm_valid
            phy_kl = diagonal_gaussian_kl(q["physical"].mean, q["physical"].log_std, p_phy["mean"], p_phy["log_std"], phy_eligibility, free_bits=self.config.kl_schedule.free_bits)
            comm_kl = diagonal_gaussian_kl(q["communication"].mean, q["communication"].log_std, p_comm["mean"], p_comm["log_std"], comm_eligibility, free_bits=self.config.kl_schedule.free_bits)
            raw_phy, adjusted_phy, phy_count = phy_kl.raw, phy_kl.adjusted, phy_kl.count
            raw_comm, adjusted_comm, comm_count = comm_kl.raw, comm_kl.adjusted, comm_kl.count
            kl = adjusted_phy + adjusted_comm
        total = l_pred + float(beta_kl) * kl
        horizon_rows: list[dict[str, Any]] = []
        for h in range(motion_target.shape[1]):
            mot_h, mot_n = masked_family_mse(motion_pred[:, h], motion_target[:, h], motion_mask[:, h])
            csi_h, csi_n = masked_family_mse(csi_pred[:, h], csi_target[:, h], csi_mask[:, h])
            mot_num = ((motion_pred[:, h] - motion_target[:, h]).square() * motion_mask[:, h].to(motion_pred.dtype)).sum()
            csi_num = ((csi_pred[:, h] - csi_target[:, h]).square() * csi_mask[:, h].to(csi_pred.dtype)).sum()
            available_count = count_available_validation_windows(motion_mask[:, h], csi_mask[:, h])
            horizon_rows.append({"horizon": h + 1, "L_Mot": float(mot_h.detach()), "L_CSI": float(csi_h.detach()), "L_Pred": float((0.5 * mot_h + 0.5 * csi_h).detach()), "motion_numerator": float(mot_num.detach()), "csi_numerator": float(csi_num.detach()), "motion_count": int(mot_n), "csi_count": int(csi_n), "available": available_count > 0, "available_sample_count": available_count})
        return {
            "L_Total": total, "L_Pred": l_pred, "L_Mot": motion_loss, "L_CSI": csi_loss, "L_KL": kl,
            "L_KL_Phy_raw": raw_phy, "L_KL_Phy_adjusted": adjusted_phy, "L_KL_Comm_raw": raw_comm, "L_KL_Comm_adjusted": adjusted_comm,
            "motion_count": motion_count, "csi_count": csi_count, "kl_phy_count": phy_count, "kl_comm_count": comm_count,
            "motion_predictions": motion_raw, "csi_predictions": csi_raw, "horizon_rows": horizon_rows,
            "motion_mask": motion_mask, "csi_mask": csi_mask,
        }

    def _targets_for_rollout(self, rollout: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        # This helper is replaced by the explicit batch targets in train/valid
        # calls.  Keeping it defensive prevents accidental GT-state use.
        raise RuntimeError("rollout targets must be supplied from the paired sample batch")

    def _run_batch(self, indices: Sequence[int], horizon: int, *, stage: str, beta_kl: float, training: bool) -> dict[str, Any]:
        rollout = self._recursive_rollout(indices, horizon, stage=stage)
        teacher = self._future_teacher(rollout, indices) if training or stage == "posterior_assisted_warmup" else None
        if not training and stage == "prior_dominant_recursive":
            teacher = None
        # Validation and stage-2 prediction still use the same paired target
        # arrays for loss only; they are never passed into the model state.
        if teacher is None:
            motion = self.data.target_tensors["target_vehicle_motion_normalized"][list(indices), :horizon]
            motion_mask = self.data.target_tensors["target_vehicle_motion_mask"][list(indices), :horizon]
            csi = self.data.target_tensors["target_comm_csi_normalized"][list(indices), :horizon]
            csi_mask = self.data.target_tensors["target_comm_csi_mask"][list(indices), :horizon]
            teacher = {"q": None, "motion": motion, "motion_mask": motion_mask, "csi": csi, "csi_mask": csi_mask}
        loss = self._prediction_and_loss(rollout, teacher, beta_kl, stage=stage)
        loss.update({"rollout": rollout, "posterior_teacher_used": teacher["q"] is not None, "beta_kl": float(beta_kl), "stage": stage})
        return loss

    def _select_train_indices(self, global_step: int) -> list[int]:
        return [self.data.train_indices[(global_step * self.config.batch_size + offset) % len(self.data.train_indices)] for offset in range(self.config.batch_size)]

    @staticmethod
    def _finite_gradients(parameters: Iterable[nn.Parameter]) -> bool:
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                return False
        return True

    def _parameter_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    @staticmethod
    def _changed(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> tuple[bool, int]:
        changed = [name for name in before if not torch.equal(before[name], after[name])]
        return bool(changed), len(changed)

    def train_step(self, *, global_step: int, epoch: int) -> dict[str, Any]:
        self.train()
        stage = "posterior_assisted_warmup" if global_step < self.config.stage1_steps else "prior_dominant_recursive"
        horizon = self.config.curriculum.horizon_for_step(global_step, self.config.max_horizon)
        beta = self.config.kl_schedule.beta_at(global_step)
        indices = self._select_train_indices(global_step)
        self.optimizer.zero_grad(set_to_none=True)
        before = self._parameter_snapshot()
        result = self._run_batch(indices, horizon, stage=stage, beta_kl=beta, training=True)
        loss = result["L_Total"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("STEP 5.2 training loss is not finite")
        loss.backward()
        grad_finite = self._finite_gradients(parameter for group in self._optimizer_groups for parameter in group["params"])
        if not grad_finite:
            raise FloatingPointError("STEP 5.2 gradients contain NaN/Inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([parameter for group in self._optimizer_groups for parameter in group["params"]], self.config.gradient_clip_norm)) if self.config.gradient_clip_norm > 0 else 0.0
        self.optimizer.step()
        after = self._parameter_snapshot()
        changed, changed_count = self._changed(before, after)
        gradient_audit = {}
        for group in self._optimizer_groups:
            gradients = [parameter.grad for parameter in group["params"] if parameter.grad is not None]
            before_names = set()
            if group["name"] == "encoder":
                before_names = {f"encoder.{name}" for name, _ in self.encoder.named_parameters()}
            elif group["name"] == "rssm_dynamics":
                before_names = {f"model.{name}" for name, parameter in self.model.named_parameters() if not name.startswith(("phy_prior.", "comm_prior.", "vehicle_decoder.", "csi_decoder.", "phy_posterior.", "comm_posterior."))}
            else:
                prefixes = {"phy_prior": "model.phy_prior.", "comm_prior": "model.comm_prior.", "vehicle_motion_decoder": "model.vehicle_decoder.", "csi_decoder": "model.csi_decoder.", "current_observation_posterior": ("model.phy_posterior.", "model.comm_posterior."), "phy_future_posterior": "future_posterior.phy_future_posterior.", "comm_future_posterior": "future_posterior.comm_future_posterior.", "motion_target_encoder": "target_encoder.motion.", "csi_target_encoder": "target_encoder.csi."}
                prefix = prefixes[group["name"]]
                before_names = {prefix} if isinstance(prefix, str) else set(prefix)
                before_names = {name for name in before if any(name.startswith(item) for item in before_names)}
            gradient_audit[group["name"]] = {"gradient_present": bool(gradients), "gradient_finite": bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients), "gradient_norm": float(torch.sqrt(sum(grad.detach().square().sum() for grad in gradients))) if gradients else 0.0, "parameter_changed": any(not torch.equal(before[name], after[name]) for name in before_names if name in after)}
        result.update({
            "epoch": int(epoch), "global_step": int(global_step), "rollout_horizon": horizon,
            "loss_finite": bool(torch.isfinite(loss).item()), "gradients_finite": grad_finite,
            "gradient_norm_before_clip": grad_norm, "parameter_update": {"any_changed": changed, "changed_count": changed_count},
            "gradient_audit": gradient_audit,
            "prior_only_rollout": bool(result["rollout"]["prior_only"]), "posterior_used_as_rollout_state": bool(result["rollout"]["posterior_used_as_rollout_state"]),
            "recursive_state_feedback": bool(result["rollout"]["recursive_state_feedback"]),
            "future_state_not_rollout_input": not bool(result["rollout"]["future_target_consumed_by_rollout"]),
        })
        return _serializable_step(result)

    def validate(self) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        self.posterior_teacher_calls = 0
        self.posterior_rollout_calls = 0
        self.future_target_encoder_calls = 0
        self.current_posterior_calls = 0
        before = self._parameter_snapshot()
        horizon_rows: dict[int, list[dict[str, Any]]] = {}
        with torch.no_grad():
            for index in self.data.validation_indices:
                result = self._run_batch([index], self.config.max_horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
                for row in result["horizon_rows"]:
                    horizon_rows.setdefault(int(row["horizon"]), []).append(row)
        after = self._parameter_snapshot()
        _, changed_count = self._changed(before, after)
        per_horizon = aggregate_validation_horizon_rows(horizon_rows)
        l_val = float(np.mean([row["L_Pred"] for row in per_horizon])) if per_horizon else float("nan")
        result = {
            "model_eval": not was_training or not self.training,
            "no_grad": True,
            "prior_only_rollout": True,
            "posterior_rollout_calls": int(self.posterior_rollout_calls),
            "future_posterior_teacher_calls": int(self.posterior_teacher_calls),
            "future_target_encoder_calls": int(self.future_target_encoder_calls),
            "current_observation_posterior_calls": int(self.current_posterior_calls),
            "per_horizon": per_horizon,
            "L_Val": l_val,
            "l_val_finite": bool(np.isfinite(l_val)),
            "selector_uses_l_val_only": True,
            "parameter_changed_count": changed_count,
            "validation_no_parameter_update": changed_count == 0,
            "future_state_not_rollout_input": True,
        }
        self.last_validation = result
        if was_training:
            self.train()
        return result

    def current_latent_target_isolation_probe(self) -> dict[str, Any]:
        """Prove current-observation posterior initialization ignores Future Target."""
        self.eval()
        index = self.data.train_indices[0]
        zpi_all = self._encoder_output()
        zpi = self._slice_zpi(zpi_all, [index], len(self.data.samples))
        state, _ = self._batch_initial_state([index])
        with torch.no_grad():
            before = self.model.initialize_latent(zpi, state, posterior_mode="mean")["posterior"]
            originals = {key: value[index:index + 1].clone() for key, value in self.data.target_tensors.items() if key.startswith("target_")}
            for key, value in originals.items():
                if value.dtype.is_floating_point:
                    self.data.target_tensors[key][index:index + 1].add_(1.0)
            after = self.model.initialize_latent(zpi, state, posterior_mode="mean")["posterior"]
            for key, value in originals.items():
                self.data.target_tensors[key][index:index + 1].copy_(value)
        return {
            "current_posterior_uses_current_observation": True,
            "future_target_mutation_does_not_change_current_posterior": bool(
                torch.equal(before["physical"]["mean"], after["physical"]["mean"])
                and torch.equal(before["physical"]["log_std"], after["physical"]["log_std"])
                and torch.equal(before["communication"]["mean"], after["communication"]["mean"])
                and torch.equal(before["communication"]["log_std"], after["communication"]["log_std"])
            ),
        }

    def prior_target_isolation_probe(self) -> dict[str, Any]:
        self.eval()
        index = self.data.train_indices[0]
        with torch.no_grad():
            rollout = self._recursive_rollout([index], 1, stage="prior_dominant_recursive")
            p_phy = rollout["latents"][0]["prior"]["physical"]
            p_comm = rollout["latents"][0]["prior"]["communication"]
            original = self._future_teacher(rollout, [index], count_call=False)["q"]
            originals = {key: value[index:index + 1].clone() for key, value in self.data.target_tensors.items() if key.startswith("target_")}
            for key, value in originals.items():
                if value.dtype.is_floating_point:
                    self.data.target_tensors[key][index:index + 1].add_(1.0)
            mutated_rollout = self._recursive_rollout([index], 1, stage="prior_dominant_recursive")
            for key, value in originals.items():
                self.data.target_tensors[key][index:index + 1].copy_(value)
            p_phy_mut = mutated_rollout["latents"][0]["prior"]["physical"]
            p_comm_mut = mutated_rollout["latents"][0]["prior"]["communication"]
            motion = self.data.target_tensors["target_vehicle_motion_normalized"][index:index + 1, :1].clone()
            motion_mask = self.data.target_tensors["target_vehicle_motion_mask"][index:index + 1, :1]
            csi = self.data.target_tensors["target_comm_csi_normalized"][index:index + 1, :1].clone()
            csi_mask = self.data.target_tensors["target_comm_csi_mask"][index:index + 1, :1]
            motion = motion + motion_mask.to(motion.dtype)
            csi = csi + csi_mask.to(csi.dtype)
            h_phy = rollout["latents"][0]["h"]["physical"].unsqueeze(1)
            h_comm = rollout["latents"][0]["h"]["communication"].unsqueeze(1)
            e_motion = self.target_encoder.motion(motion, motion_mask)
            e_csi = self.target_encoder.csi(csi, csi_mask)
            q_mut = self.future_posterior(h_phy, h_comm, e_motion, e_csi)
        return {
            "prior_mean_unchanged": bool(torch.equal(p_phy["mean"], p_phy_mut["mean"]) and torch.equal(p_comm["mean"], p_comm_mut["mean"])),
            "prior_log_std_unchanged": bool(torch.equal(p_phy["log_std"], p_phy_mut["log_std"]) and torch.equal(p_comm["log_std"], p_comm_mut["log_std"])),
            "posterior_mean_changed": bool(not torch.equal(original["physical"].mean, q_mut["physical"].mean) or not torch.equal(original["communication"].mean, q_mut["communication"].mean)),
            "future_target_consumed_by_prior": False,
        }

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode("utf-8")); digest.update(parameter.detach().cpu().numpy().tobytes(order="C"))
        return digest.hexdigest()

    def deterministic_forward_digest(self) -> str:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            rollout = self._recursive_rollout([self.data.validation_indices[0]], self.config.max_horizon, stage="prior_dominant_recursive")
            digest = hashlib.sha256()
            for latent in rollout["latents"]:
                for family in ("physical", "communication"):
                    for key in ("mean", "log_std"):
                        digest.update(latent["prior"][family][key].detach().cpu().numpy().tobytes(order="C"))
                digest.update(latent["learned"]["vehicle_motion"].detach().cpu().numpy().tobytes(order="C"))
                digest.update(latent["learned"]["csi"].detach().cpu().numpy().tobytes(order="C"))
        if was_training:
            self.train()
        return digest.hexdigest()

    def state_snapshot(self, *, global_step: int = 0, epoch: int = 0, curriculum_horizon: int | None = None, best_l_val: float = float("inf"), early_stopping_counter: int = 0) -> dict[str, Any]:
        return {"stage": "posterior_assisted_warmup" if global_step < self.config.stage1_steps else "prior_dominant_recursive", "epoch": epoch, "global_step": global_step, "beta_kl": self.config.kl_schedule.beta_at(global_step), "curriculum_horizon": curriculum_horizon if curriculum_horizon is not None else self.config.curriculum.horizon_for_step(global_step, self.config.max_horizon), "best_l_val": best_l_val, "early_stopping_counter": early_stopping_counter}

    def update_validation_state(self, state: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
        """Update selector/early-stopping state using *only* ``L_Val``."""
        updated = dict(state)
        l_val = float(validation.get("L_Val", float("nan")))
        previous = float(updated.get("best_l_val", float("inf")))
        improved = bool(np.isfinite(l_val) and l_val < previous)
        updated["selector_metric"] = "L_Val"
        updated["selector_uses_l_val_only"] = True
        updated["best_l_val"] = l_val if improved else previous
        updated["early_stopping_counter"] = 0 if improved else int(updated.get("early_stopping_counter", 0)) + 1
        updated["early_stopping_should_stop"] = updated["early_stopping_counter"] >= self.config.checkpoint_patience
        updated["last_validation_l_val"] = l_val
        return updated

    def save_checkpoint(self, path: str | Path, *, state: Mapping[str, Any]) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        try:
            git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model_state": {"encoder": self.encoder.state_dict(), "rssm": self.model.state_dict(), "target_encoder": self.target_encoder.state_dict(), "future_posterior": self.future_posterior.state_dict()},
            "optimizer_state": self.optimizer.state_dict(),
            "state": dict(state), "config": _jsonable(asdict(self.config)), "data_identity": self.data.identity,
            "git_commit": git_commit, "normalization_provenance": {"target": self.data.target_normalization, "encoder_source_split": "dev_train"},
            "initialization_contract": _jsonable(self.initialization_contract),
            "architecture_identity": {"rssm": _jsonable(asdict(self.config.rssm)), "encoder": _jsonable(asdict(self.config.encoder))},
            "rng_state": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" and torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        try:
            payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=self.device)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported STEP 5.2 checkpoint schema")
        expected_norm = {"target": self.data.target_normalization, "encoder_source_split": "dev_train"}
        if payload.get("data_identity") != self.data.identity:
            raise ValueError("incompatible STEP 5.2 checkpoint data identity")
        if payload.get("normalization_provenance") != expected_norm:
            raise ValueError("incompatible STEP 5.2 checkpoint normalization provenance")
        expected_arch = {"rssm": _jsonable(asdict(self.config.rssm)), "encoder": _jsonable(asdict(self.config.encoder))}
        if payload.get("architecture_identity") != expected_arch:
            raise ValueError("incompatible STEP 5.2 checkpoint architecture config")
        if payload.get("initialization_contract") != _jsonable(self.initialization_contract):
            raise ValueError("incompatible STEP 5.3E CSI decoder initialization contract")
        states = payload["model_state"]
        self.encoder.load_state_dict(states["encoder"]); self.model.load_state_dict(states["rssm"]); self.target_encoder.load_state_dict(states["target_encoder"]); self.future_posterior.load_state_dict(states["future_posterior"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        rng = payload.get("rng_state", {})
        # torch.load(map_location=cuda) also moves RNG byte tensors to CUDA,
        # while both RNG restore APIs require CPU byte tensors.
        if "torch" in rng: torch.set_rng_state(rng["torch"].cpu())
        if self.device.type == "cuda" and rng.get("cuda") is not None: torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        if "numpy" in rng: np.random.set_state(rng["numpy"])
        if "python" in rng: random.setstate(rng["python"])
        return dict(payload["state"])


def _serializable_step(result: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return float(value.detach()) if value.ndim == 0 else _jsonable(value)
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items() if key not in {"rollout", "motion_predictions", "csi_predictions", "motion_mask", "csi_mask"}}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value
    return convert(result)


def validate_step52_receipt(receipt: Mapping[str, Any]) -> dict[str, bool]:
    required = receipt.get("required_checks", {})
    required_ok = REQUIRED_STEP52_CHECKS <= set(required) and all(required.get(name) is True for name in REQUIRED_STEP52_CHECKS)
    forbidden = receipt.get("forbidden_scope", {})
    forbidden_ok = bool(forbidden) and all(value is False for value in forbidden.values())
    passed = bool(receipt.get("passed") is True and required_ok and forbidden_ok)
    return {"required_checks": required_ok, "forbidden_scope": forbidden_ok, "passed": passed}


__all__ = [
    "CurriculumConfig", "KLSchedule", "Step52TrainingConfig", "DevelopmentBundle", "Step52Trainer", "aggregate_validation_horizon_rows", "count_available_validation_windows",
    "REQUIRED_STEP52_CHECKS", "validate_step52_receipt",
]
