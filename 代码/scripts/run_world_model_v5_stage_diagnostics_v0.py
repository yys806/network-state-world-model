import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_decision_baselines_v0 import add_decision_baseline_scores
from run_world_model_v5_utility_ranking_smoke import (
    ROOT,
    add_resource_aware_utility,
    display_path,
    grouped_ranking_metrics,
    split_decision_groups,
)


EXTENDED_LABELS = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_extended_v0"
    / "airfogsim_counterfactual_multifamily_v0_labels.csv"
)
EXTENDED_POINTS = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_extended_v0"
    / "airfogsim_counterfactual_multifamily_v0_points.csv"
)
ACTION_ONLY_PREDICTIONS = (
    ROOT
    / "reports"
    / "world_model_v5_extended_resource_aware_action_local"
    / "world_model_v5_utility_ranking_smoke_predictions.csv"
)
STATE_ACTION_PREDICTIONS = (
    ROOT
    / "reports"
    / "world_model_v5_extended_resource_aware_action_features_local"
    / "world_model_v5_utility_ranking_smoke_predictions.csv"
)
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_stage_diagnostics_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-aware diagnostics for v5 decision ranking.")
    parser.add_argument("--labels-csv", type=Path, default=EXTENDED_LABELS)
    parser.add_argument("--points-csv", type=Path, default=EXTENDED_POINTS)
    parser.add_argument("--action-predictions-csv", type=Path, default=ACTION_ONLY_PREDICTIONS)
    parser.add_argument("--state-action-predictions-csv", type=Path, default=STATE_ACTION_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rb-penalty", type=float, default=0.001)
    return parser.parse_args()


def stage_key_df(df):
    out = df.copy()
    out["seed"] = out["seed"].astype(int)
    out["decision_time_key"] = out["decision_time"].astype(float).round(6)
    return out


def attach_decision_stage(predictions, points):
    pred = stage_key_df(predictions)
    pts = stage_key_df(points)
    cols = [
        "seed",
        "decision_time_key",
        "decision_stage",
        "num_offload_options",
        "num_computing_tasks",
        "num_waiting_return_tasks",
    ]
    available = [col for col in cols if col in pts.columns]
    merged = pred.merge(
        pts[available].drop_duplicates(["seed", "decision_time_key"]),
        on=["seed", "decision_time_key"],
        how="left",
    )
    if "decision_stage" not in merged.columns:
        merged["decision_stage"] = "unknown"
    merged["decision_stage"] = merged["decision_stage"].fillna("unknown").astype(str)
    return merged.drop(columns=["decision_time_key"])


def summarize_stage_metrics(predictions, points, model_name):
    staged = attach_decision_stage(predictions, points)
    rows = []
    split_values = ["test", "train", "all"] if "split" in staged.columns else ["all"]
    for split_name in split_values:
        split_df = staged if split_name == "all" else staged[staged["split"].eq(split_name)]
        if split_df.empty:
            continue
        for stage, part in split_df.groupby("decision_stage", dropna=False):
            if part["decision_group_id"].nunique() == 0:
                continue
            metrics = grouped_ranking_metrics(part)
            metrics["model"] = model_name
            metrics["split"] = split_name
            metrics["decision_stage"] = str(stage)
            metrics["num_candidates"] = int(len(part))
            rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_stage_difficulty(labels, points, utility_col="resource_aware_utility", tie_epsilon=1e-6):
    staged = attach_decision_stage(labels, points)
    if utility_col not in staged.columns:
        raise KeyError(f"{utility_col} is required for stage difficulty diagnostics")
    group_rows = []
    for group_id, part in staged.groupby("decision_group_id", dropna=False):
        values = part[utility_col].to_numpy(dtype=np.float64)
        spread = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
        stage = str(part["decision_stage"].iloc[0]) if "decision_stage" in part.columns else "unknown"
        group_rows.append(
            {
                "decision_group_id": str(group_id),
                "decision_stage": stage,
                "num_candidates": int(len(part)),
                "utility_spread": spread,
                "is_nontrivial": bool(spread > float(tie_epsilon)),
            }
        )
    group_df = pd.DataFrame(group_rows)
    rows = []
    for stage, part in group_df.groupby("decision_stage", dropna=False):
        spreads = part["utility_spread"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "decision_stage": str(stage),
                "num_groups": int(len(part)),
                "num_nontrivial_groups": int(part["is_nontrivial"].sum()),
                "nontrivial_ratio": float(part["is_nontrivial"].mean()) if len(part) else 0.0,
                "mean_candidates_per_group": float(part["num_candidates"].mean()) if len(part) else 0.0,
                "mean_utility_spread": float(np.nanmean(spreads)) if spreads.size else 0.0,
                "median_utility_spread": float(np.nanmedian(spreads)) if spreads.size else 0.0,
                "max_utility_spread": float(np.nanmax(spreads)) if spreads.size else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("decision_stage").reset_index(drop=True)


def build_prediction_frame(df, score_col, utility_col, model_name):
    out = df.copy()
    out["target_utility"] = out[utility_col].to_numpy(dtype=np.float32)
    out["v5_predicted_utility"] = out[score_col].to_numpy(dtype=np.float32)
    out["model"] = model_name
    if "split" not in out.columns:
        out["split"] = "all"
    return out


def collect_baseline_stage_metrics(labels, points, rb_penalty, test_fraction=0.35, split_seed=42):
    labels = add_resource_aware_utility(labels, rb_penalty=rb_penalty)
    scored = add_decision_baseline_scores(labels)
    train_idx, test_idx = split_decision_groups(scored, test_fraction=test_fraction, seed=split_seed)
    baseline_cols = [col for col in scored.columns if col.startswith("predict_")]
    frames = []
    for col in baseline_cols:
        frame = build_prediction_frame(scored, col, "resource_aware_utility", col)
        frame.loc[:, "split"] = "all"
        frame.loc[train_idx, "split"] = "train"
        frame.loc[test_idx, "split"] = "test"
        frames.append(summarize_stage_metrics(frame, points, col))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_prediction_metrics(path, points, model_name):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return summarize_stage_metrics(df, points, model_name)


def best_stage_rows(metrics):
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (split, stage), part in metrics.groupby(["split", "decision_stage"], dropna=False):
        by_top1 = part.sort_values(
            ["top1_hit_mean", "normalized_top1_regret_mean"],
            ascending=[False, True],
        ).iloc[0]
        by_regret = part.sort_values(
            ["normalized_top1_regret_mean", "top1_hit_mean"],
            ascending=[True, False],
        ).iloc[0]
        rows.append(
            {
                "split": split,
                "decision_stage": stage,
                "criterion": "best_top1",
                "model": by_top1["model"],
                "num_groups": int(by_top1["num_groups"]),
                "top1_hit_mean": float(by_top1["top1_hit_mean"]),
                "normalized_top1_regret_mean": float(by_top1["normalized_top1_regret_mean"]),
                "spearman_mean": float(by_top1["spearman_mean"]),
            }
        )
        rows.append(
            {
                "split": split,
                "decision_stage": stage,
                "criterion": "best_regret",
                "model": by_regret["model"],
                "num_groups": int(by_regret["num_groups"]),
                "top1_hit_mean": float(by_regret["top1_hit_mean"]),
                "normalized_top1_regret_mean": float(by_regret["normalized_top1_regret_mean"]),
                "spearman_mean": float(by_regret["spearman_mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary, metrics, best_rows, difficulty, output_dir):
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    lines = [
        "# World model v5 stage diagnostics v0",
        "",
        "## Goal",
        "",
        "Diagnose whether v5 decision ranking fails mainly at offload/RB, compute, or return-route decision stages.",
        "",
        "## Key Readout",
        "",
    ]
    if test_metrics.empty:
        lines.append("- No test-split predictions were available; only all-split diagnostics were written.")
    else:
        for stage, part in test_metrics.groupby("decision_stage"):
            best_top1 = part.sort_values(["top1_hit_mean", "normalized_top1_regret_mean"], ascending=[False, True]).iloc[0]
            best_regret = part.sort_values(["normalized_top1_regret_mean", "top1_hit_mean"], ascending=[True, False]).iloc[0]
            lines.append(
                "- Stage `{}`: best top-1 `{:.6f}` from `{}`, best regret `{:.6f}` from `{}` over `{}` groups.".format(
                    stage,
                    float(best_top1["top1_hit_mean"]),
                    best_top1["model"],
                    float(best_regret["normalized_top1_regret_mean"]),
                    best_regret["model"],
                    int(best_top1["num_groups"]),
                )
            )
    lines.extend(
        [
            "",
            "## Best Rows",
            "",
            best_rows.to_markdown(index=False) if not best_rows.empty else "No best rows available.",
            "",
            "## Stage Difficulty",
            "",
            difficulty.to_markdown(index=False) if not difficulty.empty else "No difficulty diagnostics available.",
            "",
            "## Full Stage Metrics",
            "",
            metrics.to_markdown(index=False) if not metrics.empty else "No metrics available.",
            "",
            "## Outputs",
            "",
            f"- metrics_csv: `{summary['outputs']['metrics_csv']}`",
            f"- best_csv: `{summary['outputs']['best_csv']}`",
            f"- difficulty_csv: `{summary['outputs']['difficulty_csv']}`",
        ]
    )
    path = output_dir / "world_model_v5_stage_diagnostics_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels_csv)
    points = pd.read_csv(args.points_csv)
    frames = [
        collect_baseline_stage_metrics(labels, points, args.rb_penalty),
        read_prediction_metrics(args.action_predictions_csv, points, "v5_action_only_local"),
        read_prediction_metrics(args.state_action_predictions_csv, points, "v5_state_action_local"),
    ]
    metrics = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    metrics = metrics.sort_values(["split", "decision_stage", "model"]).reset_index(drop=True)
    best_rows = best_stage_rows(metrics)
    labels_for_difficulty = add_resource_aware_utility(labels, rb_penalty=args.rb_penalty)
    difficulty = summarize_stage_difficulty(labels_for_difficulty, points, utility_col="resource_aware_utility")
    metrics_path = args.output_dir / "world_model_v5_stage_diagnostics_v0_metrics.csv"
    best_path = args.output_dir / "world_model_v5_stage_diagnostics_v0_best.csv"
    difficulty_path = args.output_dir / "world_model_v5_stage_diagnostics_v0_difficulty.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    best_rows.to_csv(best_path, index=False, encoding="utf-8-sig")
    difficulty.to_csv(difficulty_path, index=False, encoding="utf-8-sig")
    summary = {
        "labels_csv": display_path(args.labels_csv),
        "points_csv": display_path(args.points_csv),
        "rb_penalty": args.rb_penalty,
        "num_metric_rows": int(len(metrics)),
        "models": sorted(metrics["model"].dropna().astype(str).unique().tolist()) if not metrics.empty else [],
        "decision_stages": sorted(metrics["decision_stage"].dropna().astype(str).unique().tolist())
        if not metrics.empty
        else [],
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "best_csv": display_path(best_path),
            "difficulty_csv": display_path(difficulty_path),
        },
    }
    report_path = write_report(summary, metrics, best_rows, difficulty, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_stage_diagnostics_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
