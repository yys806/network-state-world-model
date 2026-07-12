"""Risk-controlled rb_total scale selection for PI-JWM v11 candidate.

This CPU diagnostic wraps the deployable conditional-scale candidates with a
validation risk rule: identity is the default, and a repair candidate is selected
only if validation active-rate improves while link RMSE increase stays bounded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_conditional_scale import (
    ScaleCandidate,
    apply_conditional_scale,
    build_candidates,
    candidate_to_dict,
)
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import load_context_limited, limit_indices


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_risk_controlled_scale_20260622"


def _as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def select_risk_controlled_candidate(
    val_rows: list[dict],
    max_link_delta: float,
    min_active_improvement: float,
) -> dict:
    """Select best validation active-rate candidate under a link-risk constraint."""

    identity_rows = [row for row in val_rows if row.get("candidate") == "identity"]
    if not identity_rows:
        raise ValueError("val_rows must contain identity")
    identity = identity_rows[0]
    base_active = _as_float(identity, "active_rate_rmse")
    base_link = _as_float(identity, "link_rmse")
    feasible = []
    for row in val_rows:
        if row.get("candidate") == "identity":
            continue
        active = _as_float(row, "active_rate_rmse")
        link = _as_float(row, "link_rmse")
        active_improvement = base_active - active
        link_delta = link - base_link
        if not np.isfinite(active_improvement) or not np.isfinite(link_delta):
            continue
        if active_improvement + 1e-12 >= float(min_active_improvement) and link_delta <= float(max_link_delta) + 1e-12:
            enriched = dict(row)
            enriched["risk_active_improvement"] = float(active_improvement)
            enriched["risk_link_delta"] = float(link_delta)
            feasible.append(enriched)
    if not feasible:
        fallback = dict(identity)
        fallback["risk_active_improvement"] = 0.0
        fallback["risk_link_delta"] = 0.0
        return fallback
    feasible.sort(key=lambda row: (_as_float(row, "active_rate_rmse"), _as_float(row, "link_rmse"), str(row.get("candidate", ""))))
    return feasible[0]


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
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--global-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--step-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--conditional-scales", type=float, nargs="+", default=[1.0, 1.04, 1.06, 1.08, 1.10])
    parser.add_argument("--conditional-features", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), nargs="+", default=["step_rb_cpu_total"])
    parser.add_argument("--conditional-thresholds", type=float, nargs="+", default=[475.0, 500.0, 525.0])
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-link-delta", type=float, nargs="+", default=[1.0, 2.0, 4.0, 6.0])
    parser.add_argument("--min-active-improvement", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
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
        identity = ScaleCandidate(name="identity", mode="identity")
        base_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        base_row.update(candidate_to_dict(identity))
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

    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    val_ranked = sorted(val_rows, key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    test_ranked = sorted(test_rows, key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])))
    selection_rows = []
    for max_link_delta in args.max_link_delta:
        for min_active_improvement in args.min_active_improvement:
            selected = select_risk_controlled_candidate(
                val_rows,
                max_link_delta=float(max_link_delta),
                min_active_improvement=float(min_active_improvement),
            )
            matched_test = next((row for row in test_rows if row["candidate"] == selected["candidate"]), None)
            selection_row = {
                "max_link_delta": float(max_link_delta),
                "min_active_improvement": float(min_active_improvement),
                "selected_candidate": selected["candidate"],
                "selected_val_active_rate_rmse": selected["active_rate_rmse"],
                "selected_val_link_rmse": selected["link_rmse"],
                "selected_val_active_improvement": selected.get("risk_active_improvement", float("nan")),
                "selected_val_link_delta": selected.get("risk_link_delta", float("nan")),
                "matched_test_active_rate_rmse": "" if matched_test is None else matched_test["active_rate_rmse"],
                "matched_test_link_rmse": "" if matched_test is None else matched_test["link_rmse"],
                "matched_test_improvement": "" if matched_test is None else matched_test["improvement_vs_baseline"],
            }
            selection_rows.append(selection_row)
    selection_ranked = sorted(
        selection_rows,
        key=lambda row: (
            float(row["selected_val_active_rate_rmse"]),
            float(row["selected_val_link_delta"]),
            float(row["max_link_delta"]),
            float(row["min_active_improvement"]),
        ),
    )

    write_csv(args.output_dir / "risk_controlled_scale_results.csv", rows)
    write_csv(args.output_dir / "risk_controlled_scale_val_ranked.csv", val_ranked)
    write_csv(args.output_dir / "risk_controlled_scale_test_ranked.csv", test_ranked)
    write_csv(args.output_dir / "risk_controlled_selection.csv", selection_rows)
    write_csv(args.output_dir / "risk_controlled_selection_ranked.csv", selection_ranked)

    best_selection = selection_ranked[0] if selection_ranked else {}
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "rb_total_risk_controlled_scale",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "candidate_count": len(candidates) + 1,
        "best_val": val_ranked[0] if val_ranked else {},
        "best_test": test_ranked[0] if test_ranked else {},
        "best_risk_controlled_selection": best_selection,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
