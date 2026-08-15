import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_utility_ranking_smoke import (
    ROOT,
    add_resource_aware_utility,
    display_path,
    grouped_ranking_metrics,
    split_by_test_seeds,
    split_decision_groups,
)


LABEL_PATH = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_label_dataset_v0"
    / "airfogsim_counterfactual_label_dataset_v0_labels.csv"
)
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_decision_baselines_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate simple decision-ranking baselines for v5 labels.")
    parser.add_argument("--candidate-csv", type=Path, default=LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rb-penalty", type=float, default=0.001)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-seeds", type=int, nargs="*", default=None)
    return parser.parse_args()


def add_decision_baseline_scores(df):
    out = df.copy()
    out["predict_total_rb"] = out["total_rb"].to_numpy(dtype=np.float32)
    out["predict_minus_total_rb"] = -out["total_rb"].to_numpy(dtype=np.float32)
    if "world_model_utility" in out.columns:
        out["predict_world_proxy"] = out["world_model_utility"].to_numpy(dtype=np.float32)
    if "action_family" in out.columns:
        family = out["action_family"].astype(str)
        out["predict_offload_family"] = family.eq("offload_target").astype(np.float32)
        out["predict_mixed_family"] = family.eq("mixed_offload_rb").astype(np.float32)
        out["predict_non_rb_family"] = (~family.eq("rb_count")).astype(np.float32)
        out["predict_low_rb_mixed_bonus"] = (
            -out["total_rb"].to_numpy(dtype=np.float32) + 20.0 * family.eq("mixed_offload_rb").astype(np.float32)
        )
    if "candidate_id" in out.columns:
        out["predict_default"] = out["candidate_id"].astype(str).eq("default").astype(np.float32)
    return out


def evaluate_baseline(df, utility_col, pred_col, split_idx, split_name):
    part = df.iloc[split_idx].copy()
    part["target_utility"] = part[utility_col].to_numpy(dtype=np.float32)
    part["v5_predicted_utility"] = part[pred_col].to_numpy(dtype=np.float32)
    out = grouped_ranking_metrics(part)
    out["utility"] = utility_col
    out["baseline"] = pred_col
    out["split"] = split_name
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.candidate_csv)
    df = add_resource_aware_utility(df, rb_penalty=args.rb_penalty)
    df = add_decision_baseline_scores(df)
    if args.test_seeds:
        train_idx, test_idx = split_by_test_seeds(df, args.test_seeds)
        split_strategy = f"test_seeds_{','.join(str(seed) for seed in args.test_seeds)}"
    else:
        train_idx, test_idx = split_decision_groups(df, test_fraction=args.test_fraction, seed=args.split_seed)
        split_strategy = f"group_fraction_{args.test_fraction}_seed_{args.split_seed}"
    baseline_cols = [col for col in df.columns if col.startswith("predict_")]
    rows = []
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(df)))]:
        for utility_col in ["airfogsim_utility", "resource_aware_utility"]:
            for pred_col in baseline_cols:
                rows.append(evaluate_baseline(df, utility_col, pred_col, split_idx, split_name))
    metrics = pd.DataFrame(rows)
    metrics_path = args.output_dir / "world_model_v5_decision_baselines_v0_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "candidate_csv": display_path(args.candidate_csv),
        "rb_penalty": args.rb_penalty,
        "num_candidates": int(len(df)),
        "num_decision_groups": int(df["decision_group_id"].nunique()),
        "num_train_groups": int(df.iloc[train_idx]["decision_group_id"].nunique()),
        "num_test_groups": int(df.iloc[test_idx]["decision_group_id"].nunique()),
        "split_strategy": split_strategy,
        "baseline_columns": baseline_cols,
        "outputs": {
            "metrics_csv": display_path(metrics_path),
        },
    }
    report_lines = [
        "# World model v5 decision baselines v0",
        "",
        "## Goal",
        "",
        "This report separates trivial RB-allocation rules from learned utility-ranking results.",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- metrics_csv: `{summary['outputs']['metrics_csv']}`",
    ]
    report_path = args.output_dir / "world_model_v5_decision_baselines_v0_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_decision_baselines_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
