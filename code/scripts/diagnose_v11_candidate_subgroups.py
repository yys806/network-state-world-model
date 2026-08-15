"""Inventory PI-JWM v11 candidate policies across validation subgroups.

This script is CPU-only by default and is intended as a controlled strategy
diagnostic, not a training job.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch
from pi_jwm.v8_training import collect_v8_predictions
from pi_jwm.v11_rollout_value_calibrator import RolloutAlignedValueCalibrator, freeze_module

from diagnose_v11_bridge_operating_point import OperatingPoint, load_context, make_bridge_dataset
from evaluate_v11_adaptive_bridge import AdaptivePolicyBridgeDataset
from run_v11_rollout_value_calibrator import CalibratedBridgeDataset


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/v11_candidate_subgroup_refresh_20260621"
DEFAULT_CPU_BC64 = (
    PROJECT_ROOT
    / "artifacts/experiments/pi_jwm_v11_rollout_value_calibrator_adaptive_bc_confirm_20260621"
    / "bc64_e3_lr0005/checkpoints/v11_rollout_value_calibrator_cpu_smoke.pt"
)
DEFAULT_CPU_BC128 = (
    PROJECT_ROOT
    / "artifacts/experiments/pi_jwm_v11_rollout_value_calibrator_adaptive_bc_confirm_20260621"
    / "bc128_e2_lr0005/checkpoints/v11_rollout_value_calibrator_cpu_smoke.pt"
)
DEFAULT_GPU512 = (
    PROJECT_ROOT
    / "artifacts/experiments/pi_jwm_v11_rollout_value_calibrator_adaptive_gpu_samples_diagnostic_20260621"
    / "gpu512_e2_lr0005/checkpoints/v11_rollout_value_calibrator_cpu_smoke.pt"
)
CURRENT_CPU_BEST_VAL_ACTIVE_RMSE = 232.26805853434024
PRACTICAL_IMPROVEMENT_EPS = 1e-3


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    policy_threshold: float = 0.4
    value_scale: float = 1.0
    new_policy_threshold: float | None = None
    new_value_scale: float | None = None
    gate_feature: str = "none"
    gate_threshold: float = 450.0
    checkpoint: Path | None = None
    promotable: bool = True
    note: str = ""


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def _bin_label(left: float, right: float) -> str:
    left_text = f"{left:g}"
    right_text = "inf" if np.isinf(right) else f"{right:g}"
    return f"{left_text}-{right_text}"


def make_rate_bucket_labels(truth: np.ndarray, bins: list[float]) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    labels = np.empty(truth.shape, dtype=object)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (truth >= left) & (truth < right)
        labels[mask] = _bin_label(float(left), float(right))
    return labels


def make_step_labels(shape: tuple[int, int, int]) -> np.ndarray:
    sample_count, horizon, edge_count = shape
    steps = np.arange(horizon).reshape(1, horizon, 1)
    return np.broadcast_to(steps, (sample_count, horizon, edge_count)).astype(str)


def broadcast_step_labels(step_labels: np.ndarray, edge_count: int) -> np.ndarray:
    step_labels = np.asarray(step_labels)
    if step_labels.ndim != 2:
        raise ValueError("step_labels must have shape [sample, horizon]")
    return np.broadcast_to(step_labels[..., None], (*step_labels.shape, edge_count)).astype(object)


def make_step_bucket_labels(step_values: np.ndarray, bins: list[float]) -> np.ndarray:
    step_values = np.asarray(step_values, dtype=np.float64)
    labels = np.empty(step_values.shape, dtype=object)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (step_values >= left) & (step_values < right)
        labels[mask] = _bin_label(float(left), float(right))
    return labels


def summarize_candidate_groups(
    candidate: str,
    split: str,
    truth: np.ndarray,
    active: np.ndarray,
    pred: np.ndarray,
    group_name: str,
    group_labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    pred = np.asarray(pred, dtype=np.float64)
    group_labels = np.asarray(group_labels)
    if truth.shape != active.shape or truth.shape != pred.shape or truth.shape != group_labels.shape:
        raise ValueError("truth, active, pred, and group_labels must share shape")

    rows = []
    for label in sorted({str(value) for value in group_labels.reshape(-1)}):
        mask = active & (group_labels.astype(str) == label)
        error = pred[mask] - truth[mask]
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "group_name": group_name,
                "group": label,
                "active_count": int(mask.sum()),
                "active_rmse": _rmse(error),
                "active_mae": float(np.mean(np.abs(error))) if error.size else float("nan"),
            }
        )
    return rows


def summarize_overall(
    candidate: CandidateSpec,
    split: str,
    truth: np.ndarray,
    active: np.ndarray,
    pred: np.ndarray,
    link_true: np.ndarray,
    link_pred: np.ndarray,
) -> dict[str, float | int | str | bool]:
    active_error = pred[active] - truth[active]
    return {
        "candidate": candidate.name,
        "split": split,
        "promotable": bool(candidate.promotable),
        "kind": candidate.kind,
        "active_count": int(active.sum()),
        "active_rmse": _rmse(active_error),
        "active_mae": float(np.mean(np.abs(active_error))) if active_error.size else float("nan"),
        "link_rmse": _rmse(np.asarray(link_pred, dtype=np.float64) - np.asarray(link_true, dtype=np.float64)),
        "policy_threshold": float(candidate.policy_threshold),
        "value_scale": float(candidate.value_scale),
        "new_policy_threshold": "" if candidate.new_policy_threshold is None else float(candidate.new_policy_threshold),
        "new_value_scale": "" if candidate.new_value_scale is None else float(candidate.new_value_scale),
        "gate_feature": candidate.gate_feature,
        "gate_threshold": float(candidate.gate_threshold),
        "note": candidate.note,
    }


def robust_rank_candidates(
    overall_rows: list[dict],
    subgroup_rows: list[dict],
    min_group_count: int = 3,
) -> list[dict[str, float | int | str | bool]]:
    by_candidate: dict[str, dict[str, dict]] = {}
    for row in overall_rows:
        by_candidate.setdefault(str(row["candidate"]), {})[str(row["split"])] = row

    worst_by_candidate: dict[str, dict[str, float | str]] = {}
    for row in subgroup_rows:
        if str(row.get("split")) != "val":
            continue
        if int(row.get("active_count", 0)) < int(min_group_count):
            continue
        candidate = str(row["candidate"])
        rmse = float(row["active_rmse"])
        current = worst_by_candidate.get(candidate)
        if current is None or rmse > float(current["rmse"]):
            worst_by_candidate[candidate] = {
                "rmse": rmse,
                "group_name": str(row["group_name"]),
                "group": str(row["group"]),
                "active_count": int(row["active_count"]),
            }

    rows = []
    for candidate, splits in by_candidate.items():
        val = splits.get("val")
        if val is None:
            continue
        test = splits.get("test", {})
        worst = worst_by_candidate.get(candidate, {})
        rows.append(
            {
                "candidate": candidate,
                "promotable": bool(val.get("promotable", True)),
                "kind": val.get("kind", ""),
                "val_active_rmse": float(val["active_rmse"]),
                "test_active_rmse": float(test["active_rmse"]) if "active_rmse" in test else float("nan"),
                "val_link_rmse": float(val["link_rmse"]),
                "test_link_rmse": float(test["link_rmse"]) if "link_rmse" in test else float("nan"),
                "val_worst_group_active_rmse": float(worst["rmse"]) if worst else float("nan"),
                "val_worst_group_name": worst.get("group_name", ""),
                "val_worst_group": worst.get("group", ""),
                "val_worst_group_count": int(worst.get("active_count", 0)),
                "note": val.get("note", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["promotable"]),
            float(row["val_active_rmse"]),
            float(row["val_worst_group_active_rmse"]) if np.isfinite(row["val_worst_group_active_rmse"]) else float("inf"),
            str(row["candidate"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def load_calibrator_checkpoint(path: Path, device: torch.device) -> tuple[RolloutAlignedValueCalibrator, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    calibrator = RolloutAlignedValueCalibrator(**checkpoint["config"]).to(device)
    calibrator.load_state_dict(checkpoint["model_state"])
    freeze_module(calibrator)
    codebook = torch.as_tensor(checkpoint["codebook"], dtype=torch.float32, device=device)
    return calibrator, codebook


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_default_candidates(args: argparse.Namespace) -> list[CandidateSpec]:
    return [
        CandidateSpec("old_t040_s100", "point", 0.4, 1.0, note="old autonomous reference point"),
        CandidateSpec("new_t037_s106_global", "point", 0.37, 1.06, note="ungated new operating point"),
        CandidateSpec("adaptive_rbcpu_ge450", "adaptive", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 450.0, note="best hand adaptive point so far"),
        CandidateSpec("adaptive_rbcpu_ge500", "adaptive", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 500.0, note="stricter high-load gate diagnostic"),
        CandidateSpec("adaptive_rbcpu_ge550", "adaptive", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 550.0, note="stricter high-load gate diagnostic"),
        CandidateSpec("calib_cpu_bc64_e3", "calibrator", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 450.0, args.cpu_bc64_checkpoint, note="CPU value-calibrator candidate"),
        CandidateSpec("calib_cpu_bc128_e2", "calibrator", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 450.0, args.cpu_bc128_checkpoint, note="best CPU value-calibrator candidate"),
        CandidateSpec("calib_gpu512_e2_diag", "calibrator", 0.4, 1.0, 0.37, 1.06, "step_rb_cpu_total", 450.0, args.gpu512_checkpoint, promotable=False, note="GPU small-sample diagnostic only"),
    ]


def build_candidate_loader(
    candidate: CandidateSpec,
    arrays: dict,
    indices: np.ndarray,
    stats: dict,
    policy_model,
    action_scale_np: np.ndarray,
    value_vocab: dict | None,
    train_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> DataLoader | None:
    base = V6WorldModelDataset(arrays, indices, stats)
    if candidate.kind == "point":
        dataset = make_bridge_dataset(
            arrays,
            indices,
            stats,
            policy_model,
            action_scale_np,
            value_vocab,
            device,
            OperatingPoint(candidate.name, candidate.policy_threshold, candidate.value_scale),
            train_idx,
        )
    elif candidate.kind == "adaptive":
        old_dataset = make_bridge_dataset(
            arrays,
            indices,
            stats,
            policy_model,
            action_scale_np,
            value_vocab,
            device,
            OperatingPoint("old", candidate.policy_threshold, candidate.value_scale),
            train_idx,
        )
        new_dataset = make_bridge_dataset(
            arrays,
            indices,
            stats,
            policy_model,
            action_scale_np,
            value_vocab,
            device,
            OperatingPoint(
                "new",
                candidate.new_policy_threshold if candidate.new_policy_threshold is not None else candidate.policy_threshold,
                candidate.new_value_scale if candidate.new_value_scale is not None else candidate.value_scale,
            ),
            train_idx,
        )
        dataset = AdaptivePolicyBridgeDataset(
            base,
            old_dataset,
            new_dataset,
            stats,
            candidate.gate_feature,
            candidate.gate_threshold,
        )
    elif candidate.kind == "calibrator":
        if candidate.checkpoint is None or not candidate.checkpoint.exists():
            return None
        calibrator, codebook = load_calibrator_checkpoint(candidate.checkpoint, device)
        action_scale = torch.as_tensor(action_scale_np.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
        dataset = CalibratedBridgeDataset(
            base,
            policy_model,
            calibrator,
            action_scale,
            stats["edge_a_future"],
            codebook,
            candidate.policy_threshold,
            candidate.value_scale,
            candidate.new_policy_threshold,
            candidate.new_value_scale,
            candidate.gate_feature,
            candidate.gate_threshold,
            True,
            device,
            hard=True,
        )
    else:
        raise ValueError(f"unknown candidate kind: {candidate.kind}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--cpu-bc64-checkpoint", type=Path, default=DEFAULT_CPU_BC64)
    parser.add_argument("--cpu-bc128-checkpoint", type=Path, default=DEFAULT_CPU_BC128)
    parser.add_argument("--gpu512-checkpoint", type=Path, default=DEFAULT_GPU512)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    return parser.parse_args()


def run_inventory(args: argparse.Namespace) -> dict:
    device = torch.device("cpu")
    context = load_context(args, device)
    summary, arrays, stats, world_model, policy_model, action_scale_np, value_vocab, splits = context
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    config = summary["config"]
    split_names = ["val", "test"] if args.split == "both" else [args.split]
    candidates = make_default_candidates(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    skipped_rows: list[dict] = []
    for split_name in split_names:
        indices = splits[split_name]
        logged_actions = arrays["edge_a_future"][indices]
        logged_rb_total = logged_actions[..., 2].sum(axis=2)
        logged_rb_cpu_total = logged_actions[..., 2].sum(axis=2) + logged_actions[..., 4].sum(axis=2)
        for candidate in candidates:
            loader = build_candidate_loader(
                candidate,
                arrays,
                indices,
                stats,
                policy_model,
                action_scale_np,
                value_vocab,
                splits["train"],
                device,
                args.batch_size,
            )
            if loader is None:
                skipped_rows.append(
                    {
                        "candidate": candidate.name,
                        "split": split_name,
                        "reason": f"missing checkpoint: {candidate.checkpoint}",
                    }
                )
                continue
            predictions = collect_v8_predictions(
                world_model,
                loader,
                device,
                stats,
                rate_output_mode=config.get("rate_output_mode", "main"),
                inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
            )
            truth = predictions["link_rate_true"].squeeze(-1)
            pred = predictions["link_rate_pred"].squeeze(-1)
            active = predictions["link_activity_true"].squeeze(-1) > 0.5
            overall_rows.append(
                summarize_overall(
                    candidate,
                    split_name,
                    truth,
                    active,
                    pred,
                    predictions["link_rate_true"],
                    predictions["link_rate_pred"],
                )
            )
            group_specs = [
                ("step", make_step_labels(truth.shape)),
                ("truth_rate_bucket", make_rate_bucket_labels(truth, [0.0, 50.0, 100.0, 150.0, 250.0, float("inf")])),
                (
                    "logged_gate_rbcpu450",
                    broadcast_step_labels(np.where(logged_rb_cpu_total >= 450.0, "ge450", "lt450"), truth.shape[2]),
                ),
                (
                    "logged_rb_bucket",
                    broadcast_step_labels(make_step_bucket_labels(logged_rb_total, [0.0, 50.0, 100.0, 250.0, 500.0, float("inf")]), truth.shape[2]),
                ),
            ]
            for group_name, labels in group_specs:
                subgroup_rows.extend(
                    summarize_candidate_groups(candidate.name, split_name, truth, active, pred, group_name, labels)
                )

    ranking_rows = robust_rank_candidates(overall_rows, subgroup_rows, min_group_count=args.min_group_count)
    write_csv(args.output_dir / "candidate_overall.csv", overall_rows)
    write_csv(args.output_dir / "candidate_subgroups.csv", subgroup_rows)
    write_csv(args.output_dir / "candidate_by_step.csv", [r for r in subgroup_rows if r["group_name"] == "step"])
    write_csv(args.output_dir / "candidate_by_gate.csv", [r for r in subgroup_rows if r["group_name"] == "logged_gate_rbcpu450"])
    write_csv(args.output_dir / "candidate_by_rb_bucket.csv", [r for r in subgroup_rows if r["group_name"] == "logged_rb_bucket"])
    write_csv(args.output_dir / "candidate_by_rate_bucket.csv", [r for r in subgroup_rows if r["group_name"] == "truth_rate_bucket"])
    write_csv(args.output_dir / "candidate_robust_ranking.csv", ranking_rows)
    write_csv(args.output_dir / "skipped_candidates.csv", skipped_rows)

    best_promotable = next((row for row in ranking_rows if row["promotable"]), None)
    lines = [
        "# PI-JWM v11 Candidate Subgroup Inventory",
        "",
        "CPU-only diagnostic. Selection is based on validation active-rate RMSE first; test is read after validation ranking.",
        "",
        "| rank | candidate | promotable | val_active_rmse | test_active_rmse | val_worst_group | val_worst_group_rmse |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in ranking_rows:
        lines.append(
            f"| {row['rank']} | {row['candidate']} | {row['promotable']} | "
            f"{row['val_active_rmse']:.6f} | {row['test_active_rmse']:.6f} | "
            f"{row['val_worst_group_name']}={row['val_worst_group']} | {row['val_worst_group_active_rmse']:.6f} |"
        )
    lines.extend(["", "## Recommendation", ""])
    if best_promotable is None:
        lines.append("No promotable candidate was evaluated successfully; do not run GPU.")
    else:
        lines.append(
            f"Best promotable validation candidate: {best_promotable['candidate']} "
            f"(val active-rate RMSE {best_promotable['val_active_rmse']:.6f}, "
            f"test active-rate RMSE {best_promotable['test_active_rmse']:.6f})."
        )
        improvement = CURRENT_CPU_BEST_VAL_ACTIVE_RMSE - float(best_promotable["val_active_rmse"])
        if improvement <= PRACTICAL_IMPROVEMENT_EPS:
            lines.append(
                "This does not practically beat the current CPU value-calibrator validation point "
                f"{CURRENT_CPU_BEST_VAL_ACTIVE_RMSE:.12f} (tolerance {PRACTICAL_IMPROVEMENT_EPS:g}); keep work on CPU."
            )
        else:
            lines.append(
                "This beats the current CPU value-calibrator validation point; consider a short GPU confirmation only after reviewing subgroup regressions."
            )
    if skipped_rows:
        lines.extend(["", "Skipped candidates are listed in skipped_candidates.csv."])
    (args.output_dir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "candidate_subgroup_inventory",
        "output_dir": str(args.output_dir),
        "overall_rows": len(overall_rows),
        "subgroup_rows": len(subgroup_rows),
        "skipped_rows": skipped_rows,
        "ranking": ranking_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    result = run_inventory(args)
    print(json.dumps({"output_dir": result["output_dir"], "ranking": result["ranking"][:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
