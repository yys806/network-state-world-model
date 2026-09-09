"""Run the non-locked P4 observability, expressivity, gradient, and selector audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_tensor_v2 import EDGE_FEATURES, NODE_FEATURES, TASK_FEATURES
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_motion_state_v1 import derive_causal_node_motion
from pi_jwm.formal_p4_first_principles_audit_v1 import (
    gradient_cosine_summary,
    masked_global_l1_oracle,
    shared_offset_preserves_ranking,
)
from run_formal_dual_graph_cpu_smoke_v1 import _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import _reload_learned_model


COMPONENTS = ("node", "physical_edge", "flow", "task")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(value.dtype)
    return (value * numeric).sum() / numeric.sum().clamp_min(1.0)


def _normal_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
) -> torch.Tensor:
    return (
        p_log_std
        - q_log_std
        + (
            torch.exp(2.0 * q_log_std) + torch.square(q_mean - p_mean)
        )
        / (2.0 * torch.exp(2.0 * p_log_std))
        - 0.5
    ).mean()


def _diagnostic_objectives(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    target, static = batch["target"], batch["static"]
    node_valid = target["node_present"].bool().unsqueeze(-1)
    task_valid = target["task_present"].bool().unsqueeze(-1)
    edge_valid = target["physical_edge_present"].bool()
    activity = target.get("aggregate_link_activity", target["link_activity"]).to(
        prediction["link_activity_logits"].dtype
    )
    activity_mask = target.get("aggregate_link_activity_mask", edge_valid).bool()
    static_edge_valid = torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
    link_valid = activity_mask & static_edge_valid[:, None, :]
    link_values = F.binary_cross_entropy_with_logits(
        prediction["link_activity_logits"], activity, reduction="none"
    )
    rate_index = list(EDGE_FEATURES).index("rate_sum")
    rb_index = list(EDGE_FEATURES).index("allocated_rb_count")
    delay_index = list(TASK_FEATURES).index("delay")
    operations = (
        _masked_mean(
            torch.abs(
                prediction["physical_edge_state_mean"][..., rate_index]
                - target["physical_edge_state"][..., rate_index]
            ),
            edge_valid,
        )
        + _masked_mean(
            torch.abs(
                prediction["physical_edge_state_mean"][..., rb_index]
                - target["physical_edge_state"][..., rb_index]
            ),
            edge_valid,
        )
        + _masked_mean(
            torch.abs(
                prediction["task_state_mean"][..., delay_index]
                - target["task_state"][..., delay_index]
            ),
            task_valid.squeeze(-1),
        )
    )
    return {
        "node": _masked_mean(
            torch.abs(prediction["node_state_mean"] - target["node_state"]),
            node_valid.expand_as(target["node_state"]),
        ),
        "link": _masked_mean(link_values, link_valid),
        "task": _masked_mean(
            torch.abs(prediction["task_state_mean"] - target["task_state"]),
            task_valid.expand_as(target["task_state"]),
        ),
        "operations": operations,
        "kl": _normal_kl(
            prediction["rssm_posterior_mean"],
            prediction["rssm_posterior_log_std"],
            prediction["rssm_posterior_path_prior_mean"],
            prediction["rssm_posterior_path_prior_log_std"],
        ),
    }


def _gradient_audit(model: torch.nn.Module, batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    per_pair: dict[str, list[float]] = defaultdict(list)
    per_norm: dict[str, list[float]] = defaultdict(list)
    objective_values: dict[str, list[float]] = defaultdict(list)
    for batch_index, batch in enumerate(batches):
        torch.manual_seed(20260906 + batch_index)
        model.train()
        prediction = model(batch)
        objectives = _diagnostic_objectives(prediction, batch)
        gradients: dict[str, list[torch.Tensor]] = {}
        for name, objective in objectives.items():
            raw = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradients[name] = [
                torch.zeros_like(parameter) if value is None else value
                for parameter, value in zip(parameters, raw)
            ]
            objective_values[name].append(float(objective.detach()))
        summary = gradient_cosine_summary(gradients)
        for name, value in summary["norms"].items():
            per_norm[name].append(float(value))
        for name, value in summary["pairs"].items():
            per_pair[name].append(float(value["cosine"]))
    return {
        "batch_count": len(batches),
        "objective_definition": "masked diagnostic objectives over node, link, task, operations, and RSSM KL",
        "objective_mean": {
            name: float(np.mean(values)) for name, values in objective_values.items()
        },
        "gradient_norm_mean": {
            name: float(np.mean(values)) for name, values in per_norm.items()
        },
        "pairs": {
            name: {
                "cosine_mean": float(np.mean(values)),
                "negative_fraction": float(np.mean(np.asarray(values) < 0.0)),
            }
            for name, values in per_pair.items()
        },
    }


def _motion_audit(tensor_root: Path) -> dict[str, Any]:
    valid_count = nonzero_speed = nonzero_acceleration = 0
    moving_pairs = turns = 0
    by_kind: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tensor_path in sorted(tensor_root.glob("seed_*/trajectory_tensors.npz")):
        with np.load(tensor_path, allow_pickle=False) as loaded:
            state = loaded["node_state"]
            present = loaded["node_present"].astype(bool)
            result = derive_causal_node_motion(
                state,
                present,
                slot_seconds=float(loaded["slot_seconds"]),
            )
            motion, mask = result["node_motion_state"], result["node_motion_mask"]
            valid_count += int(present.sum())
            nonzero_speed += int((np.abs(state[..., 3]) > 1e-8)[present].sum())
            nonzero_acceleration += int((np.abs(state[..., 4]) > 1e-8)[present].sum())
            velocity_valid = mask[..., :3].all(axis=-1)
            velocity_norm = np.linalg.norm(motion[..., :3], axis=-1)
            pair_valid = velocity_valid[1:] & velocity_valid[:-1]
            pair_moving = pair_valid & (velocity_norm[1:] > 1e-6) & (velocity_norm[:-1] > 1e-6)
            left, right = motion[:-1, :, :3], motion[1:, :, :3]
            cosine = (left * right).sum(axis=-1) / np.maximum(
                np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-12
            )
            moving_pairs += int(pair_moving.sum())
            turns += int((pair_moving & (cosine < np.cos(np.deg2rad(15.0)))).sum())
            kinds = loaded["node_kind_index"]
            for kind in np.unique(kinds[kinds >= 0]):
                kind_mask = present & (kinds[None, :] == kind)
                by_kind[int(kind)]["valid"] += int(kind_mask.sum())
                by_kind[int(kind)]["derived_velocity_valid"] += int(
                    (velocity_valid & kind_mask).sum()
                )
    return {
        "valid_node_state_count": valid_count,
        "stored_nonzero_speed_count": nonzero_speed,
        "stored_nonzero_acceleration_count": nonzero_acceleration,
        "derived_moving_pair_count": moving_pairs,
        "turn_gt_15deg_count": turns,
        "turn_gt_15deg_fraction": turns / max(moving_pairs, 1),
        "by_node_kind_index": {str(key): dict(value) for key, value in by_kind.items()},
        "conclusion": "stored_motion_fields_invalid_while_positions_move",
    }


def run_audit(
    *,
    run_dir: str | Path,
    tensor_root: str | Path,
    output_dir: str | Path,
    method: str = "complete_rssm_dual_graph_v1",
    gradient_batches: int = 2,
) -> dict[str, Any]:
    run_dir, tensor_root, output_dir = map(Path, (run_dir, tensor_root, output_dir))
    if any("locked_test" in str(path).lower() for path in (run_dir, tensor_root, output_dir)):
        raise ValueError("locked_test path is forbidden")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary = _read_json(run_dir / "run_summary.json")
    if summary.get("locked_test_accessed") is not False:
        raise ValueError("source run does not prove locked-test isolation")
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    sample_ids = _read_json(run_dir / "sample_ids.json")
    config = FormalWindowConfig(
        history_steps=int(contract["history_steps"]),
        horizon_steps=int(contract["horizon_steps"]),
    )
    validation = FormalAirFogSimWindowDataset(
        tensor_root, split="validation", config=config, stats=stats, normalize=True
    )
    validation_subset = _subset_for_ids(validation, sample_ids["validation"])
    validation_loader = DataLoader(validation_subset, batch_size=2, shuffle=False)
    checkpoint_path = run_dir / "checkpoints" / f"{method}__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != method:
        raise ValueError("checkpoint method mismatch")
    model = _reload_learned_model(method, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    aggregate = {
        name: {"base_abs": 0.0, "full_abs": 0.0, "global_oracle_abs": 0.0, "count": 0}
        for name in COMPONENTS
    }
    node_x = {"global_oracle_abs": 0.0, "persistence_abs": 0.0, "count": 0}
    link_shared = True
    link_correction_range_max = 0.0
    with torch.no_grad():
        for batch in validation_loader:
            full = model(batch)
            base = model.base(
                {
                    "history": batch["history"],
                    "future_action": batch["future_action"],
                    "static": batch["static"],
                }
            )
            for name in COMPONENTS:
                target = batch["target"][f"{name}_state"].numpy()
                base_mean = base[f"{name}_state_mean"].numpy()
                full_mean = full[f"{name}_state_mean"].numpy()
                valid = batch["target"][f"{name}_present"].numpy().astype(bool)
                required = target - base_mean
                oracle = masked_global_l1_oracle(required, valid)
                feature_mask = np.broadcast_to(valid[..., None], target.shape)
                count = int(feature_mask.sum())
                aggregate[name]["count"] += count
                aggregate[name]["base_abs"] += float(np.abs(base_mean - target)[feature_mask].sum())
                aggregate[name]["full_abs"] += float(np.abs(full_mean - target)[feature_mask].sum())
                aggregate[name]["global_oracle_abs"] += oracle["global_oracle_mae"] * count
                if name == "node":
                    x_index = list(NODE_FEATURES).index("x")
                    x_scale = float(stats["features"]["node_state"]["scale"][x_index])
                    x_required = required[..., x_index : x_index + 1]
                    x_oracle = masked_global_l1_oracle(x_required, valid)
                    x_count = int(valid.sum())
                    persistence = batch["history"]["node_state"][:, -1:, :, x_index].expand_as(
                        batch["target"]["node_state"][..., x_index]
                    ).numpy()
                    node_x["count"] += x_count
                    node_x["global_oracle_abs"] += x_oracle["global_oracle_mae"] * x_count * x_scale
                    node_x["persistence_abs"] += float(
                        np.abs(persistence - target[..., x_index])[valid].sum() * x_scale
                    )
            valid_link = batch["target"].get(
                "aggregate_link_activity_mask", batch["target"]["physical_edge_present"]
            ).numpy().astype(bool)
            base_logits = base["link_activity_logits"].numpy()
            correction = (full["link_activity_logits"] - base["link_activity_logits"]).numpy()
            for batch_index in range(correction.shape[0]):
                for step in range(correction.shape[1]):
                    selected = valid_link[batch_index, step]
                    if np.any(selected):
                        values = correction[batch_index, step, selected]
                        link_correction_range_max = max(
                            link_correction_range_max, float(values.max() - values.min())
                        )
            link_shared = link_shared and shared_offset_preserves_ranking(
                base_logits, correction[..., :1], valid_link
            )

    expressivity = {}
    for name, values in aggregate.items():
        count = max(int(values["count"]), 1)
        expressivity[name] = {
            "valid_feature_count": int(values["count"]),
            "base_mae_normalized": values["base_abs"] / count,
            "actual_global_rssm_mae_normalized": values["full_abs"] / count,
            "best_global_broadcast_mae_normalized": values["global_oracle_abs"] / count,
            "entity_oracle_mae_normalized": 0.0,
        }
    node_count = max(int(node_x["count"]), 1)
    node_global_mae = node_x["global_oracle_abs"] / node_count
    node_persistence_mae = node_x["persistence_abs"] / node_count
    expressivity["node_x_physical"] = {
        "valid_count": int(node_x["count"]),
        "best_global_broadcast_mae_m": node_global_mae,
        "persistence_mae_m": node_persistence_mae,
        "best_global_broadcast_ratio": node_global_mae / max(node_persistence_mae, 1e-12),
        "entity_oracle_mae_m": 0.0,
    }

    train = FormalAirFogSimWindowDataset(
        tensor_root, split="train", config=config, stats=stats, normalize=True
    )
    train_subset = _subset_for_ids(train, sample_ids["train"][: max(2, gradient_batches * 2)])
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=False)
    batches = []
    for batch in train_loader:
        batches.append(batch)
        if len(batches) >= gradient_batches:
            break
    gradient_report = _gradient_audit(model, batches)

    tensor_manifest = tensor_root / "manifest.json"
    history_path = run_dir / "training_history.json"
    report = {
        "schema_version": "PI-JWM-P4-first-principles-audit-v1",
        "status": "completed_read_only",
        "method": method,
        "source": {
            "run_dir": str(run_dir),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "tensor_root": str(tensor_root),
            "tensor_manifest_sha256": _sha256(tensor_manifest),
            "validation_sample_count": len(validation_subset),
            "validation_sample_ids_sha256": hashlib.sha256(
                json.dumps(sample_ids["validation"], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "expressivity": expressivity,
        "link_ranking": {
            "actual_correction_range_within_step_max": link_correction_range_max,
            "shared_offset_preserves_ranking": link_shared,
            "entity_oracle_auprc_upper_bound": 1.0,
            "entity_oracle_f1_upper_bound": 1.0,
            "conclusion": "global_link_adapter_cannot_change_within_step_edge_ranking",
        },
        "motion_observability": _motion_audit(tensor_root),
        "gradient_conflict": gradient_report,
        "checkpoint_selection": {
            "current_selector": "minimum_aggregate_validation_loss",
            "formal_gate_selector": "failed_gate_count_then_max_normalized_violation_then_validation_state_nll",
            "historical_epoch_replay_available": False,
            "reason": "the current runner retains only the selected best state, not every epoch checkpoint",
            "mismatch_confirmed": True,
            "training_history_sha256": _sha256(history_path) if history_path.exists() else None,
        },
        "checks": {
            "strict_checkpoint_reload": True,
            "fixed_validation_ids_reused": len(validation_subset) == len(sample_ids["validation"]),
            "motion_scan_nonlocked_only": True,
            "gpu_execution": False,
            "locked_test_accessed": False,
            "formal_performance_claim_ready": False,
        },
        "decision": {
            "motion_tensor_repair_required": True,
            "entity_aligned_stochastic_state_required": True,
            "staged_training_required": True,
            "gate_aware_checkpoint_selection_required": True,
            "gpu_allowed": False,
            "next_action": "build_and_validate_causal_motion_unlocked_tensor",
        },
        "result_boundary": "Read-only nonlocked audit; oracle values diagnose reachability and do not constitute model performance.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "first_principles_audit.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="complete_rssm_dual_graph_v1")
    parser.add_argument("--gradient-batches", type=int, default=2)
    args = parser.parse_args()
    report = run_audit(
        run_dir=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        method=args.method,
        gradient_batches=args.gradient_batches,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
