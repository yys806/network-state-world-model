"""CPU conditional rb_total scale calibration for PI-JWM v11 candidate.

This diagnostic keeps the policy support fixed and only rescales positive
rb_total values after step 0. It tests whether the stable global upscale signal
can be improved by simple step/load-conditioned calibration before spending GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import load_context_limited, limit_indices


RB_DIM = 2
CPU_DIM = 4
EPS = 1e-9
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_conditional_scale_20260622"


@dataclass(frozen=True)
class ScaleCandidate:
    name: str
    mode: str
    scale: float = 1.0
    step1_scale: float = 1.0
    step2_scale: float = 1.0
    gate_feature: str = ""
    gate_threshold: float = 0.0
    low_scale: float = 1.0
    high_scale: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--global-scales", type=float, nargs="+", default=[1.04, 1.08, 1.10, 1.12, 1.15, 1.20])
    parser.add_argument("--step-scales", type=float, nargs="+", default=[1.00, 1.06, 1.10, 1.15, 1.20])
    parser.add_argument("--conditional-scales", type=float, nargs="+", default=[1.00, 1.08, 1.12, 1.16, 1.20])
    parser.add_argument("--conditional-features", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), nargs="+", default=["step_rb_total", "step_rb_cpu_total", "step_active_count"])
    parser.add_argument("--conditional-thresholds", type=float, nargs="+", default=[15.0, 17.0, 19.0, 300.0, 400.0, 450.0, 500.0, 550.0])
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional cap for quick smoke tests; 0 means no cap.")
    return parser.parse_args()


def compute_step_feature(actions: np.ndarray, feature: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    rb_total = np.sum(np.clip(actions[..., RB_DIM], 0.0, None), axis=2)
    cpu_total = np.sum(np.clip(actions[..., CPU_DIM], 0.0, None), axis=2)
    if feature == "step_rb_total":
        return rb_total
    if feature == "step_cpu_total":
        return cpu_total
    if feature == "step_rb_cpu_total":
        return rb_total + cpu_total
    if feature == "step_active_count":
        return np.sum(np.any(actions > EPS, axis=-1), axis=2).astype(np.float32)
    raise ValueError(f"unknown feature: {feature}")


def apply_conditional_scale(actions: np.ndarray, candidate: ScaleCandidate, preserve_step0: bool = True) -> np.ndarray:
    repaired = np.asarray(actions, dtype=np.float32).copy()
    base_rb = np.asarray(actions, dtype=np.float32)[..., RB_DIM]
    active = base_rb > EPS
    scales = np.ones(base_rb.shape[:2], dtype=np.float32)
    if candidate.mode == "identity":
        return repaired
    if candidate.mode == "global":
        scales[:] = float(candidate.scale)
    elif candidate.mode == "step":
        if scales.shape[1] >= 2:
            scales[:, 1] = float(candidate.step1_scale)
        if scales.shape[1] >= 3:
            scales[:, 2] = float(candidate.step2_scale)
    elif candidate.mode == "gate":
        score = compute_step_feature(actions, candidate.gate_feature)
        scales[:] = float(candidate.low_scale)
        scales[score >= float(candidate.gate_threshold)] = float(candidate.high_scale)
    else:
        raise ValueError(f"unknown mode: {candidate.mode}")
    if preserve_step0 and scales.shape[1] > 0:
        scales[:, 0] = 1.0
    repaired[..., RB_DIM] = np.where(active, np.clip(base_rb * scales[:, :, None], 0.0, None), 0.0)
    return repaired


def build_candidates(args: argparse.Namespace) -> list[ScaleCandidate]:
    candidates = [ScaleCandidate(name="identity", mode="identity")]
    for scale in args.global_scales:
        candidates.append(ScaleCandidate(name=f"global_scale_{scale:g}", mode="global", scale=float(scale)))
    for step1 in args.step_scales:
        for step2 in args.step_scales:
            if abs(float(step1) - 1.0) < 1e-12 and abs(float(step2) - 1.0) < 1e-12:
                continue
            candidates.append(
                ScaleCandidate(
                    name=f"step_scale_s1_{step1:g}_s2_{step2:g}",
                    mode="step",
                    step1_scale=float(step1),
                    step2_scale=float(step2),
                )
            )
    for feature in args.conditional_features:
        thresholds = []
        for threshold in args.conditional_thresholds:
            if feature == "step_active_count" and threshold <= 64:
                thresholds.append(float(threshold))
            elif feature != "step_active_count" and threshold > 64:
                thresholds.append(float(threshold))
        for threshold in thresholds:
            for low in args.conditional_scales:
                for high in args.conditional_scales:
                    if abs(float(low) - float(high)) < 1e-12:
                        continue
                    candidates.append(
                        ScaleCandidate(
                            name=f"gate_{feature}_thr_{threshold:g}_low_{low:g}_high_{high:g}",
                            mode="gate",
                            gate_feature=feature,
                            gate_threshold=float(threshold),
                            low_scale=float(low),
                            high_scale=float(high),
                        )
                    )
    if args.max_candidates and args.max_candidates > 0:
        return candidates[: int(args.max_candidates)]
    return candidates


def candidate_to_dict(candidate: ScaleCandidate) -> dict:
    return {
        "candidate": candidate.name,
        "scale_mode": candidate.mode,
        "scale": candidate.scale,
        "step1_scale": candidate.step1_scale,
        "step2_scale": candidate.step2_scale,
        "gate_feature_for_scale": candidate.gate_feature,
        "gate_threshold_for_scale": candidate.gate_threshold,
        "low_scale": candidate.low_scale,
        "high_scale": candidate.high_scale,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context_limited(args, device)
    splits = dict(splits)
    if args.limit_after_stats:
        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)

    payloads = {}
    rows = []
    baseline_rmse = {}
    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(
            args, arrays, splits[split_name], stats, policy_model, action_scale, value_vocab, device, splits["train"]
        )
        baseline_actions, _truth_actions = collect_raw_actions(adaptive_dataset, stats)
        baseline_predictions = evaluate_raw_actions(
            baseline_actions, base_dataset, stats, world_model, summary["config"], device, args.batch_size
        )
        base_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        base_row.update(candidate_to_dict(ScaleCandidate(name="identity", mode="identity")))
        base_row["improvement_vs_baseline"] = 0.0
        rows.append(base_row)
        baseline_rmse[split_name] = float(base_row["active_rate_rmse"])
        payloads[split_name] = {"base_dataset": base_dataset, "baseline_actions": baseline_actions}

    candidates = [candidate for candidate in build_candidates(args) if candidate.mode != "identity"]
    for idx, candidate in enumerate(candidates, start=1):
        for split_name in ("val", "test"):
            payload = payloads[split_name]
            actions = apply_conditional_scale(payload["baseline_actions"], candidate, preserve_step0=True)
            predictions = evaluate_raw_actions(actions, payload["base_dataset"], stats, world_model, summary["config"], device, args.batch_size)
            row = active_rate_row(candidate.name, split_name, predictions, baseline_rmse[split_name])
            row.update(candidate_to_dict(candidate))
            rows.append(row)
            print(
                f"[{idx}/{len(candidates)}] {split_name} {candidate.name} "
                f"active_rate_rmse={row['active_rate_rmse']:.6f} "
                f"link_rmse={row['link_rmse']:.6f} improvement={row['improvement_vs_baseline']:.6f}"
            )

    write_csv(args.output_dir / "conditional_scale_results.csv", rows)
    val_ranked = sorted([row for row in rows if row["split"] == "val"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    test_ranked = sorted([row for row in rows if row["split"] == "test"], key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    write_csv(args.output_dir / "conditional_scale_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "conditional_scale_test_ranked.csv", test_ranked)

    best_val_name = str(val_ranked[0]["candidate"]) if val_ranked else ""
    matched_test = [row for row in rows if row["split"] == "test" and row["candidate"] == best_val_name]
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_conditional_scale",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "candidate_count": len(candidates) + 1,
        "best_val": val_ranked[0] if val_ranked else {},
        "matched_test_for_best_val": matched_test[0] if matched_test else {},
        "best_test": test_ranked[0] if test_ranked else {},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
