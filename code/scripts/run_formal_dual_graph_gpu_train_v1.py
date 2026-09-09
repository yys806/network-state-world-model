"""Train and evaluate formal PI-JWM dual-graph models on a CUDA device."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import (
    FormalAirFogSimWindowDataset,
    FormalWindowConfig,
    select_stratified_window_ids,
)
from pi_jwm.formal_dual_graph_world_model_v1 import (
    FormalDualGraphWorldModel,
    FormalWorldModelConfig,
)
from pi_jwm.formal_dual_graph_world_model_v2 import (
    FormalDirectedDynamicWorldModelConfig,
    FormalDirectedDynamicWorldModelV2,
)
from pi_jwm.formal_complete_rssm_world_model_v1 import (
    FormalCompleteRSSMConfig,
    FormalCompleteRSSMWorldModel,
)
from pi_jwm.formal_entity_aligned_rssm_world_model_v1 import (
    FormalEntityAlignedRSSMConfig,
    FormalEntityAlignedRSSMWorldModel,
)
from pi_jwm.formal_system_window_v1 import FormalSystemWindowDataset
from pi_jwm.formal_world_model_baselines_v1 import build_rule_prediction, method_registry
from pi_jwm.formal_world_model_loss_v1 import (
    FormalLossWeights,
    compute_training_class_weights,
    formal_world_model_loss,
)
from pi_jwm.formal_binary_calibration_v1 import (
    InverseTemperatureCalibration,
    choose_legacy_threshold,
    fit_inverse_temperature,
)
from pi_jwm.formal_world_model_metrics_v1 import FormalMetricAccumulator
from pi_jwm.formal_p4_gate_v1 import evaluate_p4_checkpoint_gates
from run_formal_dual_graph_cpu_smoke_v1 import (
    _load_or_fit_stats,
    _manifest,
    _metric_value,
    _model_hash,
    _seed_everything,
    _sha256,
    _subset_for_ids,
    _write_json,
)


NONLOCKED_SPLITS = ("train", "validation", "calibration")
RULE_METHODS = ("zero_activity", "last_persistence")
V1_MODEL_SPECS = {
    "pooled_gru": {"model_version": "formal_v1", "mode": "pooled_gru", "residual": False},
    "independent_dual_gnn": {"model_version": "formal_v1", "mode": "independent_dual_gnn", "residual": False},
    "coupled_dual_gnn": {"model_version": "formal_v1", "mode": "coupled_dual_gnn", "residual": False},
    "independent_dual_gnn_residual": {"model_version": "formal_v1", "mode": "independent_dual_gnn", "residual": True},
    "coupled_dual_gnn_residual": {"model_version": "formal_v1", "mode": "coupled_dual_gnn", "residual": True},
}
V2_MODEL_SPECS = {
    "coupled_directed_dynamic_v2": {
        "model_version": "directed_dynamic_v2_1",
        "residual": False,
    },
    "coupled_directed_dynamic_residual_v2": {
        "model_version": "directed_dynamic_v2_1",
        "residual": True,
    },
}
MODEL_SPECS = {**V1_MODEL_SPECS, **V2_MODEL_SPECS}
MODEL_SPECS["complete_rssm_dual_graph_v1"] = {
    "model_version": "formal_complete_rssm_v1_1",
    "residual": True,
    "latent_dynamics": "complete_rssm_prior_posterior_v1",
    "rssm_residual_head_initialization": "zero_when_requested_v1",
    "node_x_residual_loss_contract": "none",
}
MODEL_SPECS["complete_rssm_node_x_safe_dual_graph_v1"] = {
    "model_version": "formal_complete_rssm_v1_1",
    "residual": True,
    "latent_dynamics": "complete_rssm_prior_posterior_v1",
    "rssm_residual_head_initialization": "zero_when_requested_v1",
    "node_x_residual_loss_contract": "node_x_residual_non_degradation_v1",
}
MODEL_SPECS["entity_aligned_dual_graph_rssm_v1"] = {
    "model_version": "formal_entity_aligned_rssm_v1",
    "residual": True,
    "latent_dynamics": "entity_aligned_complete_rssm_prior_posterior_v1",
    "rssm_residual_head_initialization": "zero_when_requested_v1",
    "entity_latent_layout": "node_physical_edge_flow_task_v1",
    "node_motion_contract": "causal_backward_difference_v1",
    "link_activity_method": "persistence_residual_v1",
}
MODEL_SPECS["link_activity_persistence_residual_v1"] = {
    "model_version": "formal_v1",
    "mode": "coupled_dual_gnn",
    "residual": True,
    "link_activity_method": "persistence_residual_v1",
}
LEARNED_METHODS = tuple(V1_MODEL_SPECS)
_PUBLISHED_OUTPUT_DIR: ContextVar[Path | None] = ContextVar("published_output_dir", default=None)


def _build_learned_model(
    method: str,
    *,
    hidden_dim: int,
    history_steps: int,
    horizon_steps: int,
    use_system_energy_head: bool,
    zero_init_residual_state_heads: bool = False,
    residual_state_scale: float = 1.0,
    deterministic_rule_layer: bool = False,
    rule_layer_stats: Mapping[str, Any] | None = None,
    n_rb: int = 1,
    link_pos_weight: float | None = None,
    link_activity_missing_prior: float | None = None,
    slot_seconds: float = 0.1,
    node_position_scale: Sequence[float] = (1.0, 1.0, 1.0),
    node_motion_mean: Sequence[float] = (0.0,) * 6,
    node_motion_scale: Sequence[float] = (1.0,) * 6,
) -> tuple[torch.nn.Module, Any]:
    spec = MODEL_SPECS[method]
    if spec["model_version"] == "formal_entity_aligned_rssm_v1":
        model_config = FormalEntityAlignedRSSMConfig(
            hidden_dim=hidden_dim,
            stochastic_dim=max(4, hidden_dim // 2),
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            overshooting_distance=min(5, horizon_steps),
            residual_state_prediction=True,
            zero_init_residual_state_heads=bool(zero_init_residual_state_heads),
            residual_state_scale=float(residual_state_scale),
            deterministic_rule_layer=bool(deterministic_rule_layer),
            rule_layer_stats=rule_layer_stats,
            n_rb=int(n_rb),
            use_system_energy_head=use_system_energy_head,
            link_activity_method=str(spec["link_activity_method"]),
            link_activity_pos_weight=float(link_pos_weight if link_pos_weight is not None else 1.0),
            link_activity_missing_history_prior=link_activity_missing_prior,
            slot_seconds=float(slot_seconds),
            node_position_scale=tuple(float(value) for value in node_position_scale),
            node_motion_mean=tuple(float(value) for value in node_motion_mean),
            node_motion_scale=tuple(float(value) for value in node_motion_scale),
        )
        return FormalEntityAlignedRSSMWorldModel(model_config), model_config
    if spec["model_version"] == "formal_complete_rssm_v1_1":
        model_config = FormalCompleteRSSMConfig(
            hidden_dim=hidden_dim,
            stochastic_dim=max(4, hidden_dim // 2),
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            overshooting_distance=min(5, horizon_steps),
            residual_state_prediction=True,
            zero_init_residual_state_heads=bool(zero_init_residual_state_heads),
            residual_state_scale=float(residual_state_scale),
            deterministic_rule_layer=bool(deterministic_rule_layer),
            rule_layer_stats=rule_layer_stats,
            n_rb=int(n_rb),
            use_system_energy_head=use_system_energy_head,
            link_activity_pos_weight=float(link_pos_weight if link_pos_weight is not None else 1.0),
            link_activity_missing_history_prior=link_activity_missing_prior,
            node_x_residual_loss_contract=str(
                spec.get("node_x_residual_loss_contract", "none")
            ),
        )
        return FormalCompleteRSSMWorldModel(model_config), model_config
    if spec["model_version"] == "formal_v1":
        model_config = FormalWorldModelConfig(
            mode=str(spec["mode"]),
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            residual_state_prediction=bool(spec["residual"]),
            zero_init_residual_state_heads=bool(zero_init_residual_state_heads),
            residual_state_scale=float(residual_state_scale),
            deterministic_rule_layer=bool(deterministic_rule_layer),
            rule_layer_stats=rule_layer_stats,
            n_rb=int(n_rb),
            use_system_energy_head=use_system_energy_head,
            link_activity_method=str(spec.get("link_activity_method", "absolute_v1")),
            link_activity_pos_weight=float(link_pos_weight if link_pos_weight is not None else 1.0),
            link_activity_missing_history_prior=link_activity_missing_prior,
        )
        return FormalDualGraphWorldModel(model_config), model_config
    model_config = FormalDirectedDynamicWorldModelConfig(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        residual_state_prediction=bool(spec["residual"]),
        use_system_energy_head=use_system_energy_head,
    )
    return FormalDirectedDynamicWorldModelV2(model_config), model_config




def _reload_learned_model(method: str, model_config: Mapping[str, Any]) -> torch.nn.Module:
    validate_checkpoint_method_semantics(method, model_config)
    spec = MODEL_SPECS[method]
    if spec["model_version"] == "formal_entity_aligned_rssm_v1":
        return FormalEntityAlignedRSSMWorldModel(
            FormalEntityAlignedRSSMConfig(**dict(model_config))
        )
    if spec["model_version"] == "formal_complete_rssm_v1_1":
        return FormalCompleteRSSMWorldModel(
            FormalCompleteRSSMConfig(**dict(model_config))
        )
    if spec["model_version"] == "formal_v1":
        return FormalDualGraphWorldModel(FormalWorldModelConfig(**dict(model_config)))
    return FormalDirectedDynamicWorldModelV2(
        FormalDirectedDynamicWorldModelConfig(**dict(model_config))
    )


def validate_checkpoint_method_semantics(
    method: str, model_config: Mapping[str, Any]
) -> None:
    spec = MODEL_SPECS[method]
    expected_latent = str(spec.get("latent_dynamics", "deterministic"))
    actual_latent = str(model_config.get("latent_dynamics", "deterministic"))
    if actual_latent != expected_latent:
        raise ValueError(
            f"checkpoint latent_dynamics={actual_latent!r} is incompatible with method {method!r}; "
            f"expected {expected_latent!r}"
        )
    if expected_latent == "complete_rssm_prior_posterior_v1":
        if model_config.get("training_posterior_teacher") is not True:
            raise ValueError("complete RSSM checkpoint lacks training posterior teacher semantics")
        if model_config.get("deployment_prior_only") is not True:
            raise ValueError("complete RSSM checkpoint lacks prior-only deployment semantics")
        expected_initialization = str(spec["rssm_residual_head_initialization"])
        if model_config.get("rssm_residual_head_initialization") != expected_initialization:
            raise ValueError("complete RSSM checkpoint has incompatible residual-head initialization semantics")
        expected_node_x_loss = str(
            spec.get("node_x_residual_loss_contract", "none")
        )
        actual_node_x_loss = str(
            model_config.get("node_x_residual_loss_contract", "none")
        )
        if actual_node_x_loss != expected_node_x_loss:
            raise ValueError(
                "complete RSSM checkpoint has incompatible node-x residual loss semantics"
            )
    if expected_latent == "entity_aligned_complete_rssm_prior_posterior_v1":
        if model_config.get("training_posterior_teacher") is not True:
            raise ValueError("entity RSSM checkpoint lacks training posterior teacher semantics")
        if model_config.get("deployment_prior_only") is not True:
            raise ValueError("entity RSSM checkpoint lacks prior-only deployment semantics")
        for field in (
            "rssm_residual_head_initialization",
            "entity_latent_layout",
            "node_motion_contract",
        ):
            if str(model_config.get(field, "")) != str(spec[field]):
                raise ValueError(f"entity RSSM checkpoint has incompatible {field}")
    expected = str(spec.get("link_activity_method", "absolute_v1"))
    actual = model_config.get("link_activity_method")
    if actual is None:
        if expected == "absolute_v1":
            return
        raise ValueError(
            f"checkpoint link_activity_method is missing for method {method!r}"
        )
    if str(actual) != expected:
        raise ValueError(
            f"checkpoint link_activity_method={actual!r} is incompatible with method {method!r}; "
            f"expected {expected!r}"
        )
    if expected == "persistence_residual_v1" and "mode" in spec:
        expected_mode = str(spec["mode"])
        actual_mode = model_config.get("mode")
        if actual_mode != expected_mode:
            raise ValueError(
                f"checkpoint mode={actual_mode!r} is incompatible with method {method!r}; "
                f"expected {expected_mode!r}"
            )
        expected_residual = bool(spec["residual"])
        actual_residual = model_config.get("residual_state_prediction")
        if actual_residual != expected_residual:
            raise ValueError(
                "checkpoint residual_state_prediction="
                f"{actual_residual!r} is incompatible with method {method!r}; "
                f"expected {expected_residual!r}"
            )


def _link_activity_train_prior(report: Mapping[str, Any]) -> float:
    if str(report.get("source_split")) != "train":
        raise ValueError("link activity prior may use only train samples")
    try:
        counts = report["counts"]["link_activity"]
        positive = int(counts["positive"])
        negative = int(counts["negative"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("train link activity counts are missing") from error
    if positive < 0 or negative < 0 or positive + negative <= 0:
        raise ValueError("train link activity counts must be non-empty")
    if positive == 0 or negative == 0:
        raise ValueError("train link activity prior requires both positive and negative counts")
    return float(positive / (positive + negative))


def _training_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _manifest(output_dir)
    manifest["schema_version"] = "PI-JWM-formal-training-manifest-v1"
    return manifest


def _remove_owned_staging_dir(staging_dir: Path, parent_dir: Path, prefix: str) -> None:
    if not staging_dir.exists():
        return
    resolved_staging = staging_dir.resolve()
    if (
        resolved_staging.parent != parent_dir
        or not staging_dir.name.startswith(prefix)
        or not resolved_staging.is_dir()
    ):
        return
    shutil.rmtree(resolved_staging)


def _publish_atomically(function: Any) -> Any:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "output_dir" not in kwargs:
            return function(*args, **kwargs)
        target_dir = Path(kwargs["output_dir"]).expanduser().resolve()
        if target_dir.exists():
            raise FileExistsError(f"output directory already exists: {target_dir}")
        parent_dir = target_dir.parent.resolve()
        staging_prefix = f".{target_dir.name}.staging-"
        parent_dir.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            raise FileExistsError(f"output directory already exists: {target_dir}")
        staging_dir = parent_dir / f"{staging_prefix}{uuid.uuid4().hex}"
        staging_dir.mkdir()
        staged_kwargs = dict(kwargs)
        staged_kwargs["output_dir"] = staging_dir
        publication_token = _PUBLISHED_OUTPUT_DIR.set(target_dir)
        try:
            result = function(*args, **staged_kwargs)
            if not (staging_dir / "manifest.json").is_file():
                raise RuntimeError("training output is missing manifest.json before publication")
            if target_dir.exists():
                raise FileExistsError(f"output directory already exists: {target_dir}")
            staging_dir.rename(target_dir)
            return result
        except BaseException:
            _remove_owned_staging_dir(staging_dir, parent_dir, staging_prefix)
            raise
        finally:
            _PUBLISHED_OUTPUT_DIR.reset(publication_token)

    return wrapper


def validate_gpu_protocol(splits: Iterable[str], device: str) -> None:
    requested = tuple(str(split) for split in splits)
    if "locked_test" in requested:
        raise ValueError("locked_test cannot be used by the GPU training protocol")
    unknown = set(requested) - set(NONLOCKED_SPLITS)
    if unknown:
        raise ValueError(f"unsupported GPU splits: {sorted(unknown)}")
    if not str(device).startswith("cuda"):
        raise ValueError("the formal GPU entry point requires a CUDA device")


def move_nested_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(item, device) for item in value)
    return value


def _edge_valid(static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)


def _prediction(
    method: str,
    model: torch.nn.Module | None,
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if method in RULE_METHODS:
        return build_rule_prediction(method, batch, stats)
    if model is None:
        raise ValueError(f"learned method {method} requires a model")
    return model(batch)


_BINARY_EVENT_KEYS = {
    "link_activity": ("link_activity_logits", "link_activity"),
    "flow_present": ("flow_presence_logits", "flow_present"),
    "task_present": ("task_presence_logits", "task_present"),
    "dag_release": ("dag_release_logits", "dag_release"),
    "dag_edge_present": ("dag_edge_presence_logits", "dag_edge_present"),
}


def _choose_event_thresholds(
    method: str,
    model: torch.nn.Module | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
    device: torch.device,
    link_pos_weight: float | None = None,
) -> tuple[
    dict[str, float],
    dict[str, Any],
    float,
    InverseTemperatureCalibration | None,
]:
    raw_logits = {name: [] for name in _BINARY_EVENT_KEYS}
    labels = {name: [] for name in _BINARY_EVENT_KEYS}
    inference_seconds = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            target = batch["target"]
            edge_valid = _edge_valid(batch["static"])
            activity = target.get("aggregate_link_activity", target["link_activity"])
            activity_mask = target.get("aggregate_link_activity_mask")
            activity_mask = torch.ones_like(activity, dtype=torch.bool) if activity_mask is None else activity_mask.bool()
            event_data = {
                "link_activity": (prediction["link_activity_logits"], activity, edge_valid[:, None, :].expand_as(activity) & activity_mask),
                "flow_present": (prediction["flow_presence_logits"], target["flow_present"], batch["static"]["flow_valid"][:, None, :].expand_as(target["flow_present"])),
                "task_present": (prediction["task_presence_logits"], target["task_present"], batch["static"]["task_valid"][:, None, :].expand_as(target["task_present"])),
                "dag_release": (prediction["dag_release_logits"], target["task_dag_state"][..., 2] > 0.5, target["task_dag_state_present"] & batch["static"]["task_valid"][:, None, :].expand_as(target["task_dag_state_present"])),
                "dag_edge_present": (prediction["dag_edge_presence_logits"], target["dag_edge_present"], batch["static"]["dag_edge_valid"][:, None, :].expand_as(target["dag_edge_present"])),
            }
            for name, (logits, truth, valid) in event_data.items():
                raw_logits[name].append(logits[valid].detach().cpu().double().reshape(-1))
                labels[name].append(truth.bool()[valid].detach().cpu().reshape(-1))
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {"selection_split": "calibration", "events": {}}
    link_calibration: InverseTemperatureCalibration | None = None
    for name in _BINARY_EVENT_KEYS:
        all_logits = (
            torch.cat(raw_logits[name]) if raw_logits[name] else torch.empty((0,), dtype=torch.float64)
        )
        all_labels = torch.cat(labels[name]) if labels[name] else torch.empty((0,), dtype=torch.bool)
        raw_scores = torch.sigmoid(all_logits)
        if name == "link_activity" and link_pos_weight is not None:
            if all_logits.numel() == 0 or all_labels.numel() == 0:
                raise ValueError("no valid calibration link samples")
            if not torch.any(all_labels).item() or torch.all(all_labels).item():
                raise ValueError(
                    "calibration link labels must contain both positive and negative samples"
                )
            link_calibration, fit_report = fit_inverse_temperature(
                all_logits,
                all_labels,
                pos_weight=float(link_pos_weight),
                fit_split="calibration",
            )
            selection = choose_legacy_threshold(raw_scores, all_labels, link_calibration)
            candidates = []
            for candidate in selection["candidates"]:
                row = dict(candidate)
                row["threshold"] = row["raw_threshold"]
                row["coordinate"] = "raw_weighted_score"
                candidates.append(row)
            selected = dict(selection["selected"])
            selected["threshold"] = selected["raw_threshold"]
            selected["coordinate"] = "raw_weighted_score"
            thresholds[name] = float(selected["raw_threshold"])
            report["events"][name] = {
                "candidates": candidates,
                "selected": selected,
                "decision_equivalence_to_legacy": selection["decision_equivalence_to_legacy"],
                "temperature_fit": fit_report,
            }
            continue
        all_scores = raw_scores.numpy()
        label_values = all_labels.numpy()
        rows = []
        for candidate in (0.1, 0.3, 0.5, 0.7, 0.9):
            predicted = all_scores >= candidate
            tp = int(np.count_nonzero(predicted & label_values))
            fp = int(np.count_nonzero(predicted & ~label_values))
            fn = int(np.count_nonzero(~predicted & label_values))
            denominator = 2 * tp + fp + fn
            rows.append(
                {
                    "threshold": candidate,
                    "raw_threshold": candidate,
                    "coordinate": "legacy_sigmoid_score",
                    "f1": float(2 * tp / denominator) if denominator else None,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
        comparable = [row for row in rows if row["f1"] is not None]
        selected = max(comparable, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5))) if comparable else rows[2]
        thresholds[name] = float(selected["raw_threshold"])
        report["events"][name] = {"candidates": rows, "selected": selected}
    return thresholds, report, inference_seconds, link_calibration


def _evaluate(
    method: str,
    model: torch.nn.Module | None,
    loader: DataLoader,
    stats: Mapping[str, Any],
    thresholds: Mapping[str, float],
    distribution_available: bool,
    device: torch.device,
    link_probability_calibration: InverseTemperatureCalibration | None = None,
) -> tuple[dict[str, Any], float]:
    accumulator = FormalMetricAccumulator(
        stats,
        thresholds=thresholds,
        distribution_available=distribution_available,
        link_probability_calibration=link_probability_calibration,
    )
    inference_seconds = 0.0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            started = time.perf_counter()
            prediction = _prediction(method, model, batch, stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize(), inference_seconds


def _mean_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    class_weights: Mapping[str, float],
    loss_weights: FormalLossWeights,
    stats: Mapping[str, Any],
    device: torch.device,
) -> float | None:
    report = _mean_loss_report(
        model, loader, class_weights, loss_weights, stats, device
    )
    return report["total_loss"]


def _mean_loss_report(
    model: torch.nn.Module,
    loader: DataLoader,
    class_weights: Mapping[str, float],
    loss_weights: FormalLossWeights,
    stats: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    losses: list[float] = []
    state_nlls: list[float] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_nested_to_device(cpu_batch, device)
            prediction = model(batch)
            loss, components = formal_world_model_loss(
                prediction,
                batch["target"],
                batch["static"],
                weights=loss_weights,
                class_weights=class_weights,
                system_target=batch.get("system_target"),
                normalization_stats=stats,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            losses.append(float(loss.detach()))
            state_nll = components["state_nll"]
            if not torch.isfinite(state_nll):
                raise RuntimeError("non-finite validation state NLL")
            state_nlls.append(float(state_nll.detach()))
    return {
        "total_loss": float(np.mean(losses)) if losses else None,
        "state_nll": float(np.mean(state_nlls)) if state_nlls else None,
    }


def _select_ids(
    dataset: Any,
    limit: int | None,
    seed: int,
) -> list[str]:
    actual_limit = len(dataset) if limit is None else min(int(limit), len(dataset))
    if actual_limit < 0:
        raise ValueError("sample limits cannot be negative")
    return select_stratified_window_ids(dataset.rows, actual_limit, seed)


def _validated_slot_seconds(
    datasets: Mapping[str, Any], fallback: float | None = None
) -> float:
    values = {
        float(row["simulation_interval"])
        for dataset in datasets.values()
        for row in dataset.rows
        if "simulation_interval" in row
    }
    if not values and fallback is not None:
        values = {float(fallback)}
    if len(values) != 1:
        raise ValueError(f"formal tensor must have one slot duration, found {sorted(values)}")
    value = values.pop()
    if not np.isfinite(value) or value <= 0:
        raise ValueError("formal tensor slot duration must be finite and positive")
    return value


def _set_entity_training_stage(
    model: FormalEntityAlignedRSSMWorldModel, stage: str
) -> tuple[torch.nn.Parameter, ...]:
    if stage not in {"base", "rssm"}:
        raise ValueError(f"unsupported entity training stage: {stage}")
    train_base = stage == "base"
    for parameter in model.base.parameters():
        parameter.requires_grad_(train_base)
    for name, parameter in model.named_parameters():
        if not name.startswith("base."):
            parameter.requires_grad_(not train_base)
    selected = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not selected:
        raise RuntimeError(f"entity training stage {stage} selected no parameters")
    return selected


def _training_prediction(
    method: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    stage: str,
) -> Mapping[str, torch.Tensor]:
    if method == "entity_aligned_dual_graph_rssm_v1" and stage == "base":
        if not isinstance(model, FormalEntityAlignedRSSMWorldModel):
            raise TypeError("entity method requires FormalEntityAlignedRSSMWorldModel")
        return model.base(batch)
    return model(batch)


@_publish_atomically
def run_formal_training(
    *,
    tensor_root: str | Path,
    system_root: str | Path | None = None,
    use_system_energy_head: bool = False,
    output_dir: str | Path,
    device: str,
    learned_methods: Sequence[str] = LEARNED_METHODS,
    seed: int = 20260802,
    data_seed: int | None = None,
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 0,
    state_mae_weight: float | None = None,
    zero_init_residual_state_heads: bool = False,
    residual_state_scale: float = 1.0,
    deterministic_rule_layer: bool = False,
    base_epochs: int = 0,
    checkpoint_selection: str = "validation_loss_v1",
    min_epochs: int = 0,
    patience: int = 0,
    save_every_epoch: bool = False,
    protocol_mode: str = "legacy",
) -> dict[str, Any]:
    unknown_methods = set(learned_methods) - set(MODEL_SPECS)
    if unknown_methods:
        raise ValueError(f"unsupported learned methods: {sorted(unknown_methods)}")
    if not learned_methods:
        raise ValueError("at least one learned method is required")
    if min(hidden_dim, epochs, batch_size) <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("training hyperparameters are invalid")
    if state_mae_weight is not None and state_mae_weight < 0:
        raise ValueError("state_mae_weight must be non-negative")
    if residual_state_scale < 0:
        raise ValueError("residual_state_scale must be non-negative")
    if base_epochs < 0:
        raise ValueError("base_epochs must be non-negative")
    if checkpoint_selection not in {"validation_loss_v1", "p4_gate_aware_v1"}:
        raise ValueError("unsupported checkpoint selection contract")
    if protocol_mode not in {
        "legacy",
        "cpu_consistency",
        "gpu_execution_sentinel",
        "formal_single_seed",
    }:
        raise ValueError("unsupported formal training protocol mode")
    if min_epochs < 0 or patience < 0:
        raise ValueError("min_epochs and patience must be non-negative")
    if min_epochs > epochs:
        raise ValueError("min_epochs cannot exceed epochs")
    if checkpoint_selection == "p4_gate_aware_v1" and set(learned_methods) != {
        "entity_aligned_dual_graph_rssm_v1"
    }:
        raise ValueError("P4 gate-aware selection requires the single entity RSSM method")
    if protocol_mode == "gpu_execution_sentinel" and (
        train_limit != 256
        or evaluation_limit != 128
        or base_epochs != 1
        or epochs != 1
        or checkpoint_selection != "validation_loss_v1"
    ):
        raise ValueError("GPU execution sentinel requires 256/128 samples and one epoch per stage")
    if protocol_mode == "formal_single_seed" and (
        train_limit is not None
        or evaluation_limit is not None
        or hidden_dim != 32
        or base_epochs != 20
        or epochs != 40
        or min_epochs != 20
        or patience != 10
        or checkpoint_selection != "p4_gate_aware_v1"
        or not save_every_epoch
        or not zero_init_residual_state_heads
        or not deterministic_rule_layer
        or learning_rate != 3e-4
        or weight_decay != 1e-5
    ):
        raise ValueError("formal single-seed P4 configuration is not frozen protocol v1")
    if "entity_aligned_dual_graph_rssm_v1" in learned_methods and base_epochs <= 0:
        raise ValueError("entity-aligned RSSM requires positive base_epochs for staged training")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    tensor_root = Path(tensor_root)
    if use_system_energy_head and system_root is None:
        raise ValueError("use_system_energy_head requires system_root")
    resolved_system_root = Path(system_root) if system_root is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    _seed_everything(seed)
    selected_data_seed = int(seed if data_seed is None else data_seed)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    window_config = FormalWindowConfig(
        history_steps=int(contract["history_steps"]),
        horizon_steps=int(contract["horizon_steps"]),
    )
    stats = _load_or_fit_stats(tensor_root)
    dataset_type = FormalSystemWindowDataset if use_system_energy_head else FormalAirFogSimWindowDataset
    datasets = {}
    for split in NONLOCKED_SPLITS:
        dataset_kwargs = {
            "split": split,
            "config": window_config,
            "stats": stats,
            "normalize": True,
        }
        if use_system_energy_head:
            dataset_kwargs["system_root"] = resolved_system_root
        datasets[split] = dataset_type(tensor_root, **dataset_kwargs)
    slot_seconds = _validated_slot_seconds(
        datasets, fallback=float(contract.get("slot_seconds_value", 0.1))
    )
    sample_ids = {
        "train": _select_ids(datasets["train"], train_limit, selected_data_seed),
        "validation": _select_ids(datasets["validation"], evaluation_limit, selected_data_seed + 1),
        "calibration": _select_ids(datasets["calibration"], evaluation_limit, selected_data_seed + 2),
    }
    subsets = {split: _subset_for_ids(datasets[split], sample_ids[split]) for split in NONLOCKED_SPLITS}
    loaders = {
        split: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=requested_device.type == "cuda",
        )
        for split, subset in subsets.items()
    }
    class_weight_report = compute_training_class_weights(subsets["train"])
    class_weights = class_weight_report["pos_weight"]
    loss_weights = FormalLossWeights()
    complete_rssm_methods = {
        "complete_rssm_dual_graph_v1",
        "complete_rssm_node_x_safe_dual_graph_v1",
        "entity_aligned_dual_graph_rssm_v1",
    }
    if complete_rssm_methods.intersection(learned_methods):
        loss_weights = replace(
            loss_weights,
            rssm_kl=0.1,
            rssm_teacher_reconstruction=0.5,
            rssm_overshooting=0.1,
            rssm_kl_balance=0.8,
        )
    if "complete_rssm_node_x_safe_dual_graph_v1" in learned_methods:
        loss_weights = replace(
            loss_weights,
            node_x_residual_non_degradation=1.0,
        )
    if state_mae_weight is not None:
        loss_weights = replace(loss_weights, state_mae=float(state_mae_weight))
    model_versions = {
        method: str(MODEL_SPECS[method]["model_version"])
        for method in learned_methods
    }
    latent_dynamics = {
        method: str(MODEL_SPECS[method].get("latent_dynamics", "deterministic"))
        for method in learned_methods
    }
    dataset_hash_source = tensor_root / "manifest.json"
    if not dataset_hash_source.is_file():
        dataset_hash_source = tensor_root / "tensor_contract.json"
    config = {
        "schema_version": "PI-JWM-formal-training-config-v1",
        "seed": seed,
        "data_seed": selected_data_seed,
        "device": str(requested_device),
        "history_steps": window_config.history_steps,
        "horizon_steps": window_config.horizon_steps,
        "slot_seconds": slot_seconds,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "base_epochs": base_epochs,
        "training_stage_contract": (
            "deterministic_base_then_frozen_entity_rssm_v1"
            if "entity_aligned_dual_graph_rssm_v1" in learned_methods
            else "joint_v1"
        ),
        "checkpoint_selection": checkpoint_selection,
        "min_epochs": min_epochs,
        "patience": patience,
        "save_every_epoch": bool(save_every_epoch),
        "protocol_mode": protocol_mode,
        "total_epoch_budget": int(base_epochs + epochs),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "train_limit": train_limit,
        "evaluation_limit": evaluation_limit,
        "splits": list(NONLOCKED_SPLITS),
        "threshold_selection_split": "calibration",
        "link_probability_calibration_method": "inverse_temperature_v1",
        "learned_methods": list(learned_methods),
        "model_versions": model_versions,
        "latent_dynamics": latent_dynamics,
        "loss_weights": asdict(loss_weights),
        "zero_init_residual_state_heads": bool(zero_init_residual_state_heads),
        "residual_state_scale": float(residual_state_scale),
        "deterministic_rule_layer": bool(deterministic_rule_layer),
        "use_system_energy_head": bool(use_system_energy_head),
        "system_root": str(resolved_system_root.resolve()) if resolved_system_root else None,
        "locked_test_accessed": False,
        "tensor_root": str(tensor_root.resolve()),
        "dataset_manifest_sha256": _sha256(dataset_hash_source),
    }
    registry = method_registry()
    _write_json(output_dir / "method_registry.json", registry)
    _write_json(output_dir / "sample_ids.json", sample_ids)
    _write_json(output_dir / "class_weights.json", class_weight_report)
    link_pos_weight_source = output_dir / "class_weights.json"
    published_output_dir = _PUBLISHED_OUTPUT_DIR.get()
    published_link_pos_weight_source = (
        link_pos_weight_source
        if published_output_dir is None
        else published_output_dir / "class_weights.json"
    )
    link_pos_weight_source_key = "link_activity"
    if link_pos_weight_source_key not in class_weight_report["pos_weight"]:
        raise ValueError("train-only class weights are missing link_activity pos_weight")
    link_pos_weight = float(class_weight_report["pos_weight"][link_pos_weight_source_key])
    persistence_methods = {
        method
        for method in learned_methods
        if MODEL_SPECS[method].get("link_activity_method") == "persistence_residual_v1"
    }
    link_activity_train_prior = (
        _link_activity_train_prior(class_weight_report) if persistence_methods else None
    )
    link_activity_method_by_name = {
        method: str(MODEL_SPECS[method].get("link_activity_method", "absolute_v1"))
        for method in learned_methods
    }
    config["link_activity_methods"] = link_activity_method_by_name
    config["link_activity_train_prior"] = link_activity_train_prior
    config["link_activity_train_prior_source"] = {
        "source_split": class_weight_report["source_split"],
        "counts": class_weight_report["counts"]["link_activity"],
        "source_key": "counts.link_activity",
    }
    _write_json(output_dir / "config.json", config)

    persistence_gate_reports: dict[str, Any] | None = None
    if checkpoint_selection == "p4_gate_aware_v1":
        persistence_thresholds, _, _, _ = _choose_event_thresholds(
            "last_persistence",
            None,
            loaders["calibration"],
            stats,
            requested_device,
            None,
        )
        persistence_validation, _ = _evaluate(
            "last_persistence",
            None,
            loaders["validation"],
            stats,
            persistence_thresholds,
            False,
            requested_device,
            None,
        )
        persistence_calibration, _ = _evaluate(
            "last_persistence",
            None,
            loaders["calibration"],
            stats,
            persistence_thresholds,
            False,
            requested_device,
            None,
        )
        persistence_gate_reports = {
            "validation": persistence_validation,
            "calibration": persistence_calibration,
        }

    histories: dict[str, list[dict[str, Any]]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}
    for method in learned_methods:
        _seed_everything(seed)
        model, model_config = _build_learned_model(
            method,
            hidden_dim=hidden_dim,
            history_steps=window_config.history_steps,
            horizon_steps=window_config.horizon_steps,
            use_system_energy_head=use_system_energy_head,
            zero_init_residual_state_heads=zero_init_residual_state_heads,
            residual_state_scale=residual_state_scale,
            deterministic_rule_layer=deterministic_rule_layer,
            rule_layer_stats=stats if deterministic_rule_layer else None,
            n_rb=int(contract.get("n_rb", 1)),
            link_pos_weight=(link_pos_weight if MODEL_SPECS[method].get("link_activity_method") else None),
            link_activity_missing_prior=(
                link_activity_train_prior if method in persistence_methods else None
            ),
            slot_seconds=slot_seconds,
            node_position_scale=(
                stats["features"]["node_state"]["scale"][:3]
                if method == "entity_aligned_dual_graph_rssm_v1"
                else (1.0, 1.0, 1.0)
            ),
            node_motion_mean=(
                stats["features"]["node_motion_state"]["mean"]
                if method == "entity_aligned_dual_graph_rssm_v1"
                else (0.0,) * 6
            ),
            node_motion_scale=(
                stats["features"]["node_motion_state"]["scale"]
                if method == "entity_aligned_dual_graph_rssm_v1"
                else (1.0,) * 6
            ),
        )
        initialization_hash = _model_hash(model)
        model.to(requested_device)
        base_history: list[dict[str, Any]] = []
        base_best_epoch = 0
        base_best_validation_loss: float | None = None
        if method == "entity_aligned_dual_graph_rssm_v1":
            if not isinstance(model, FormalEntityAlignedRSSMWorldModel):
                raise TypeError("entity method built an incompatible model")
            base_parameters = _set_entity_training_stage(model, "base")
            base_optimizer = torch.optim.AdamW(
                base_parameters, lr=learning_rate, weight_decay=weight_decay
            )
            base_weights = replace(
                loss_weights,
                rssm_kl=0.0,
                rssm_teacher_reconstruction=0.0,
                rssm_overshooting=0.0,
                node_x_residual_non_degradation=0.0,
            )
            base_best_state: dict[str, torch.Tensor] | None = None
            base_best_validation_loss = float("inf")
            for base_epoch in range(1, base_epochs + 1):
                model.train()
                base_losses: list[float] = []
                for cpu_batch in loaders["train"]:
                    batch = move_nested_to_device(cpu_batch, requested_device)
                    base_optimizer.zero_grad(set_to_none=True)
                    prediction = _training_prediction(method, model, batch, "base")
                    loss, _ = formal_world_model_loss(
                        prediction,
                        batch["target"],
                        batch["static"],
                        weights=base_weights,
                        class_weights=class_weights,
                        system_target=batch.get("system_target"),
                        normalization_stats=stats,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"non-finite base loss for {method} at epoch {base_epoch}"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(base_parameters, max_norm=5.0)
                    base_optimizer.step()
                    base_losses.append(float(loss.detach()))
                validation_loss = _mean_loss(
                    model.base,
                    loaders["validation"],
                    class_weights,
                    base_weights,
                    stats,
                    requested_device,
                )
                selection_loss = (
                    validation_loss
                    if validation_loss is not None
                    else float(np.mean(base_losses))
                )
                base_history.append(
                    {
                        "epoch": base_epoch,
                        "mean_train_loss": float(np.mean(base_losses)),
                        "mean_validation_loss": validation_loss,
                        "batch_count": len(base_losses),
                    }
                )
                if selection_loss < base_best_validation_loss:
                    base_best_epoch = base_epoch
                    base_best_validation_loss = selection_loss
                    base_best_state = copy.deepcopy(
                        {
                            name: value.detach().cpu()
                            for name, value in model.base.state_dict().items()
                        }
                    )
                if save_every_epoch:
                    torch.save(
                        {
                            "method": method,
                            "model_config": model_config.__dict__,
                            "model_version": MODEL_SPECS[method]["model_version"],
                            "latent_dynamics": getattr(model, "latent_dynamics", "deterministic"),
                            "stage": "base",
                            "epoch": base_epoch,
                            "model_state_dict": {
                                name: value.detach().cpu()
                                for name, value in model.state_dict().items()
                            },
                        },
                        output_dir / "checkpoints" / f"{method}__base_epoch_{base_epoch:03d}.pt",
                    )
            if base_best_state is None:
                raise RuntimeError("no deterministic base checkpoint was selected")
            model.base.load_state_dict(base_best_state, strict=True)
            optimizer_parameters = _set_entity_training_stage(model, "rssm")
        else:
            optimizer_parameters = tuple(model.parameters())
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=learning_rate, weight_decay=weight_decay
        )
        if requested_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(requested_device)
        started = time.perf_counter()
        best_epoch = 0
        best_validation_loss = float("inf")
        best_rank: tuple[float, ...] | None = None
        best_gate_report: dict[str, Any] | None = None
        epochs_without_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None
        method_history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for cpu_batch in loaders["train"]:
                batch = move_nested_to_device(cpu_batch, requested_device)
                optimizer.zero_grad(set_to_none=True)
                prediction = _training_prediction(method, model, batch, "rssm")
                loss, _ = formal_world_model_loss(
                    prediction,
                    batch["target"],
                    batch["static"],
                    weights=loss_weights,
                    class_weights=class_weights,
                    system_target=batch.get("system_target"),
                    normalization_stats=stats,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for {method} at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
            mean_train_loss = float(np.mean(train_losses)) if train_losses else None
            validation_loss_report = _mean_loss_report(
                model, loaders["validation"], class_weights, loss_weights, stats, requested_device
            )
            validation_loss = validation_loss_report["total_loss"]
            selection_loss = validation_loss if validation_loss is not None else mean_train_loss
            if selection_loss is None:
                raise RuntimeError("training and validation loaders are both empty")
            epoch_record: dict[str, Any] = {
                    "epoch": epoch,
                    "mean_train_loss": mean_train_loss,
                    "mean_validation_loss": validation_loss,
                    "batch_count": len(train_losses),
                }
            gate_report = None
            if checkpoint_selection == "p4_gate_aware_v1":
                if persistence_gate_reports is None:
                    raise RuntimeError("persistence gate reports were not prepared")
                thresholds, _, _, link_calibration = _choose_event_thresholds(
                    method,
                    model,
                    loaders["calibration"],
                    stats,
                    requested_device,
                    link_pos_weight,
                )
                validation_report, _ = _evaluate(
                    method,
                    model,
                    loaders["validation"],
                    stats,
                    thresholds,
                    True,
                    requested_device,
                    link_calibration,
                )
                calibration_report, _ = _evaluate(
                    method,
                    model,
                    loaders["calibration"],
                    stats,
                    thresholds,
                    True,
                    requested_device,
                    link_calibration,
                )
                if validation_loss_report["state_nll"] is None:
                    raise RuntimeError("gate-aware selection requires validation state NLL")
                gate_report = evaluate_p4_checkpoint_gates(
                    validation=validation_report,
                    calibration=calibration_report,
                    persistence_validation=persistence_gate_reports["validation"],
                    persistence_calibration=persistence_gate_reports["calibration"],
                    validation_state_nll=float(validation_loss_report["state_nll"]),
                )
                current_rank = tuple(float(value) for value in gate_report["rank_key"])
                epoch_record["p4_gate"] = gate_report
            else:
                current_rank = (float(selection_loss),)
            method_history.append(epoch_record)
            if save_every_epoch:
                torch.save(
                    {
                        "method": method,
                        "model_config": model_config.__dict__,
                        "model_version": MODEL_SPECS[method]["model_version"],
                        "latent_dynamics": getattr(model, "latent_dynamics", "deterministic"),
                        "epoch": epoch,
                        "checkpoint_selection": checkpoint_selection,
                        "p4_gate": gate_report,
                        "model_state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                    },
                    output_dir / "checkpoints" / f"{method}__epoch_{epoch:03d}.pt",
                )
            if best_rank is None or current_rank < best_rank:
                best_epoch = epoch
                best_validation_loss = selection_loss
                best_rank = current_rank
                best_gate_report = gate_report
                epochs_without_improvement = 0
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
            else:
                epochs_without_improvement += 1
            if (
                patience > 0
                and epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                break
        if best_state is None:
            raise RuntimeError(f"no checkpoint was selected for {method}")
        checkpoint_path = output_dir / "checkpoints" / f"{method}__best.pt"
        torch.save(
            {
                "method": method,
                "model_config": model_config.__dict__,
                "model_version": MODEL_SPECS[method]["model_version"],
                "latent_dynamics": getattr(model, "latent_dynamics", "deterministic"),
                "run_config": config,
                "model_state_dict": best_state,
                "initialization_hash": initialization_hash,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "checkpoint_selection": checkpoint_selection,
                "best_rank": best_rank,
                "p4_gate": best_gate_report,
                "base_best_epoch": base_best_epoch or None,
                "base_best_validation_loss": base_best_validation_loss,
                "base_frozen_during_rssm": method == "entity_aligned_dual_graph_rssm_v1",
                "link_activity_method": getattr(
                    model_config, "link_activity_method", "absolute_v1"
                ),
                "link_activity_train_prior": getattr(
                    model_config, "link_activity_missing_history_prior", None
                ),
                "link_activity_train_prior_source": config["link_activity_train_prior_source"],
            },
            checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = _reload_learned_model(method, checkpoint["model_config"])
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reloaded.to(requested_device)
        models[method] = reloaded
        histories[method] = (
            [{"stage": "base", **row} for row in base_history]
            + [{"stage": "rssm", **row} for row in method_history]
        )
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
            peak_device_bytes = int(torch.cuda.max_memory_allocated(requested_device))
        else:
            peak_device_bytes = 0
        runtime[method] = {
            "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
            "train_seconds": time.perf_counter() - started,
            "peak_device_memory_bytes": peak_device_bytes,
            "initialization_hash": initialization_hash,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "base_best_epoch": base_best_epoch or None,
            "base_best_validation_loss": base_best_validation_loss,
            "base_frozen_during_rssm": method == "entity_aligned_dual_graph_rssm_v1",
            "checkpoint_reload_verified": True,
            "checkpoint_selection": checkpoint_selection,
            "best_rank": list(best_rank) if best_rank is not None else None,
            "p4_gate": best_gate_report,
        }

    comparison_rows: list[dict[str, Any]] = []
    completed_methods = [*RULE_METHODS, *learned_methods]
    for method in completed_methods:
        model = models.get(method)
        thresholds, threshold_report, threshold_seconds, link_calibration = _choose_event_thresholds(
            method,
            model,
            loaders["calibration"],
            stats,
            requested_device,
            link_pos_weight if method not in RULE_METHODS else None,
        )
        distribution_available = bool(registry[method]["distribution_output"])
        validation_report, validation_seconds = _evaluate(
            method,
            model,
            loaders["validation"],
            stats,
            thresholds,
            distribution_available,
            requested_device,
            link_calibration,
        )
        calibration_report, calibration_seconds = _evaluate(
            method,
            model,
            loaders["calibration"],
            stats,
            thresholds,
            distribution_available,
            requested_device,
            link_calibration,
        )
        _write_json(output_dir / "metrics" / f"{method}__validation.json", validation_report)
        _write_json(output_dir / "metrics" / f"{method}__calibration.json", calibration_report)
        _write_json(output_dir / "metrics" / f"{method}__threshold_selection.json", threshold_report)
        if link_calibration is not None:
            link_selection = threshold_report["events"]["link_activity"]
            selected = link_selection["selected"]
            candidates = link_selection["candidates"]
            sidecar = {
                "schema_version": "PI-JWM-link-probability-calibration-v1",
                "calibration_method": "inverse_temperature_v1",
                "raw_logit_source": "link_activity_logits",
                "raw_weighted_score_definition": "sigmoid(raw_logit)",
                "pos_weight": link_calibration.pos_weight,
                "pos_weight_source_key": link_pos_weight_source_key,
                "pos_weight_source_path": str(published_link_pos_weight_source.resolve()),
                "pos_weight_source_sha256": _sha256(link_pos_weight_source),
                "inverse_probability_definition": "sigmoid((raw_logit - log(pos_weight)) / temperature)",
                "temperature": link_calibration.temperature,
                "log_temperature": link_calibration.log_temperature,
                "temperature_fit_split": link_selection["temperature_fit"]["fit_split"],
                "temperature_fit_objective": link_selection["temperature_fit"]["objective"],
                "temperature_fit_report": link_selection["temperature_fit"],
                "threshold_selection_split": "calibration",
                "validation_used_for_fit_or_selection": False,
                "legacy_raw_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
                "legacy_raw_candidates": candidates,
                "mapped_probability_thresholds": {
                    str(candidate["raw_threshold"]): candidate["probability_threshold"]
                    for candidate in candidates
                },
                "selected_legacy_raw_threshold": selected["raw_threshold"],
                "selected_probability_threshold": selected["probability_threshold"],
                "decision_equivalence_to_legacy": link_selection["decision_equivalence_to_legacy"],
                "locked_test_accessed": False,
                "gpu_execution": requested_device.type == "cuda",
                "formal_performance_claim_ready": False,
            }
            _write_json(output_dir / "metrics" / f"{method}__link_probability_calibration.json", sidecar)
        runtime.setdefault(method, {})
        runtime[method].update(
            {
                "threshold_selection_seconds": threshold_seconds,
                "validation_inference_seconds": validation_seconds,
                "calibration_inference_seconds": calibration_seconds,
            }
        )
        comparison_rows.append(
            {
                "method": method,
                "threshold": thresholds["link_activity"],
                "threshold_coordinate": "legacy_raw_threshold",
                "thresholds": json.dumps(thresholds, sort_keys=True),
                "link_legacy_raw_threshold": (
                    thresholds["link_activity"] if link_calibration is not None else None
                ),
                "link_probability_threshold": (
                    validation_report["thresholds"]["link_activity"]
                    if link_calibration is not None
                    else None
                ),
                "link_temperature": (
                    link_calibration.temperature if link_calibration is not None else None
                ),
                "link_calibration_method": (
                    "inverse_temperature_v1" if link_calibration is not None else None
                ),
                "validation_link_f1": _metric_value(validation_report, "event.link_activity.f1"),
                "calibration_link_f1": _metric_value(calibration_report, "event.link_activity.f1"),
                "validation_node_x_mae": _metric_value(validation_report, "state.node.x.mae"),
                "calibration_node_x_mae": _metric_value(calibration_report, "state.node.x.mae"),
                "validation_throughput_mae": _metric_value(validation_report, "system.communication_throughput.mae"),
                "calibration_throughput_mae": _metric_value(calibration_report, "system.communication_throughput.mae"),
                "validation_completion_rate_error": _metric_value(validation_report, "system.task_completion_rate.absolute_error"),
                "calibration_completion_rate_error": _metric_value(calibration_report, "system.task_completion_rate.absolute_error"),
                "validation_rb_occupancy_mae": _metric_value(validation_report, "resource.rb_occupancy.mae"),
                "calibration_rb_occupancy_mae": _metric_value(calibration_report, "resource.rb_occupancy.mae"),
                "validation_task_delay_mae": _metric_value(validation_report, "state.task.delay.mae"),
                "calibration_task_delay_mae": _metric_value(calibration_report, "state.task.delay.mae"),
                "validation_task_deadline_mae": _metric_value(validation_report, "state.task.deadline_remaining.mae"),
                "calibration_task_deadline_mae": _metric_value(calibration_report, "state.task.deadline_remaining.mae"),
                "validation_lifecycle_macro_f1": _metric_value(validation_report, "task.lifecycle.macro_f1"),
                "calibration_lifecycle_macro_f1": _metric_value(calibration_report, "task.lifecycle.macro_f1"),
                "validation_dag_unfinished_parent_mae": _metric_value(validation_report, "dag.unfinished_parent_count.mae"),
                "calibration_dag_unfinished_parent_mae": _metric_value(calibration_report, "dag.unfinished_parent_count.mae"),
                "best_epoch": runtime[method].get("best_epoch"),
                "best_validation_loss": runtime[method].get("best_validation_loss"),
                "parameter_count": runtime[method].get("parameter_count", 0),
                "train_seconds": runtime[method].get("train_seconds", 0.0),
                "peak_device_memory_bytes": runtime[method].get("peak_device_memory_bytes", 0),
            }
        )

    _write_json(output_dir / "training_history.json", histories)
    _write_json(output_dir / "runtime.json", runtime)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary = {
        "schema_version": "PI-JWM-formal-training-summary-v1",
        "training_run_complete": True,
        "gpu_execution": requested_device.type == "cuda",
        "formal_performance_claim_ready": False,
        "locked_test_accessed": False,
        "completed_methods": completed_methods,
        "model_versions": model_versions,
        "latent_dynamics": latent_dynamics,
        "link_activity_methods": link_activity_method_by_name,
        "link_activity_train_prior": link_activity_train_prior,
        "link_activity_train_prior_source": config["link_activity_train_prior_source"],
        "sample_counts": {split: len(values) for split, values in sample_ids.items()},
        "protocol_mode": protocol_mode,
        "result_boundary": (
            "GPU execution evidence only; short-budget performance is observational and cannot reject or accept P4."
            if protocol_mode == "gpu_execution_sentinel"
            else "Nonlocked training evidence only; locked-test remains sealed."
        ),
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "manifest.json", _training_manifest(output_dir))
    return summary


def run_gpu_training(**kwargs: Any) -> dict[str, Any]:
    device = str(kwargs.get("device", "cuda"))
    validate_gpu_protocol(NONLOCKED_SPLITS, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    kwargs["device"] = device
    return run_formal_training(**kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--system-root", type=Path)
    parser.add_argument("--use-system-energy-head", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--state-mae-weight", type=float)
    parser.add_argument("--zero-init-residual-state-heads", action="store_true")
    parser.add_argument("--residual-state-scale", type=float, default=1.0)
    parser.add_argument("--deterministic-rule-layer", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-epochs", type=int, default=0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_loss_v1", "p4_gate_aware_v1"),
        default="validation_loss_v1",
    )
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument(
        "--protocol-mode",
        choices=("legacy", "cpu_consistency", "gpu_execution_sentinel", "formal_single_seed"),
        default="legacy",
    )
    parser.add_argument(
        "--learned-methods",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(LEARNED_METHODS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_gpu_training(
        tensor_root=args.tensor_root,
        system_root=args.system_root,
        use_system_energy_head=args.use_system_energy_head,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        train_limit=args.train_limit,
        evaluation_limit=args.evaluation_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        state_mae_weight=args.state_mae_weight,
        zero_init_residual_state_heads=args.zero_init_residual_state_heads,
        residual_state_scale=args.residual_state_scale,
        deterministic_rule_layer=args.deterministic_rule_layer,
        base_epochs=args.base_epochs,
        checkpoint_selection=args.checkpoint_selection,
        min_epochs=args.min_epochs,
        patience=args.patience,
        save_every_epoch=args.save_every_epoch,
        protocol_mode=args.protocol_mode,
        learned_methods=args.learned_methods,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
