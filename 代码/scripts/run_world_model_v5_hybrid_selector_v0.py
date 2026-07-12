import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_utility_ranking_smoke import ROOT, display_path, grouped_ranking_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a heuristic-gated v5 hybrid decision selector.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-grid", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--baseline-mode", choices=["min_total_rb", "max_total_rb"], default="min_total_rb")
    parser.add_argument("--selection-rule", choices=["top1_first", "regret_first"], default="top1_first")
    return parser.parse_args()


def baseline_choice_indices(df, baseline_mode="min_total_rb", score_values=None):
    if "total_rb" not in df.columns:
        raise KeyError("total_rb is required for the RB-count baseline")
    if baseline_mode not in {"min_total_rb", "max_total_rb"}:
        raise ValueError(f"unsupported baseline mode: {baseline_mode}")
    ascending = baseline_mode == "min_total_rb"
    if score_values is None:
        score = df["v5_predicted_utility"].astype(float)
    else:
        score = pd.Series(score_values, index=df.index, dtype=float)
    choices = {}
    for group_id, part in df.groupby("decision_group_id", dropna=False):
        # Tie-break toward higher learned score so the baseline remains deterministic.
        ordered = part.assign(_baseline_tiebreak_score=score.loc[part.index]).sort_values(
            ["total_rb", "_baseline_tiebreak_score"],
            ascending=[ascending, False],
        )
        choices[group_id] = int(ordered.index[0])
    return choices


def apply_margin_hybrid_scores(df, threshold, baseline_mode="min_total_rb"):
    required = ["decision_group_id", "total_rb", "v5_predicted_utility"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing hybrid selector columns: {missing}")
    out = df.copy()
    learned_scores = out["v5_predicted_utility"].astype(float).copy()
    out["v5_predicted_utility"] = -1e9
    baseline_idx = baseline_choice_indices(out, baseline_mode=baseline_mode, score_values=learned_scores)
    for group_id, part in out.groupby("decision_group_id", dropna=False):
        base_idx = baseline_idx[group_id]
        learned_idx = int(learned_scores.loc[part.index].idxmax())
        base_score = float(learned_scores.loc[base_idx])
        learned_score = float(learned_scores.loc[learned_idx])
        chosen_idx = learned_idx if learned_score - base_score >= float(threshold) else base_idx
        out.loc[chosen_idx, "v5_predicted_utility"] = 1.0
    return out


def evaluate_threshold(df, threshold, baseline_mode="min_total_rb"):
    hybrid = apply_margin_hybrid_scores(df, threshold, baseline_mode=baseline_mode)
    return grouped_ranking_metrics(hybrid)


def select_threshold(train_df_or_sweep, threshold_grid=None, baseline_mode="min_total_rb", selection_rule="top1_first"):
    if threshold_grid is None and {"threshold", "top1_hit_mean", "normalized_top1_regret_mean"}.issubset(
        train_df_or_sweep.columns
    ):
        sweep = train_df_or_sweep.copy()
    else:
        rows = []
        for threshold in threshold_grid:
            metrics = evaluate_threshold(train_df_or_sweep, threshold, baseline_mode=baseline_mode)
            rows.append({"threshold": float(threshold), **metrics})
        sweep = pd.DataFrame(rows)
    if sweep.empty:
        return (0.0, sweep) if threshold_grid is not None else 0.0
    if selection_rule == "top1_first":
        best = sweep.sort_values(["top1_hit_mean", "normalized_top1_regret_mean"], ascending=[False, True]).iloc[0]
    elif selection_rule == "regret_first":
        best = sweep.sort_values(["normalized_top1_regret_mean", "top1_hit_mean"], ascending=[True, False]).iloc[0]
    else:
        raise ValueError(f"unsupported selection rule: {selection_rule}")
    threshold = float(best["threshold"])
    return (threshold, sweep) if threshold_grid is not None else threshold


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    if "target_utility" not in df.columns:
        raise KeyError("target_utility is required")
    if "split" not in df.columns:
        raise KeyError("split is required")
    train_df = df[df["split"].eq("train")].copy()
    test_df = df[df["split"].eq("test")].copy()
    threshold, sweep = select_threshold(
        train_df,
        args.threshold_grid,
        baseline_mode=args.baseline_mode,
        selection_rule=args.selection_rule,
    )
    rows = []
    for split_name, part in [("train", train_df), ("test", test_df), ("all", df)]:
        metrics = evaluate_threshold(part.copy(), threshold, baseline_mode=args.baseline_mode)
        rows.append({"split": split_name, "threshold": threshold, **metrics})
    metrics_df = pd.DataFrame(rows)
    sweep_path = args.output_dir / "world_model_v5_hybrid_selector_v0_threshold_sweep.csv"
    metrics_path = args.output_dir / "world_model_v5_hybrid_selector_v0_metrics.csv"
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "predictions_csv": display_path(args.predictions_csv),
        "baseline_mode": args.baseline_mode,
        "selection_rule": args.selection_rule,
        "selected_threshold": threshold,
        "outputs": {
            "threshold_sweep_csv": display_path(sweep_path),
            "metrics_csv": display_path(metrics_path),
        },
    }
    report_lines = [
        "# World model v5 hybrid selector v0",
        "",
        "## Goal",
        "",
        f"Use a training-selected confidence margin to switch from the `{args.baseline_mode}` heuristic to the learned v5 candidate only when the learned margin is large enough.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Threshold Sweep",
        "",
        sweep.to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        report_lines.append(f"- {key}: `{value}`")
    report_path = args.output_dir / "world_model_v5_hybrid_selector_v0_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_hybrid_selector_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
