import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_hybrid_selector_v0 import baseline_choice_indices
from run_world_model_v5_utility_ranking_smoke import display_path, grouped_ranking_metrics


DEFAULT_THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FAMILY_COLUMNS = ["rb_count", "offload_target", "mixed_offload_rb", "cpu_scale", "return_route"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a group-level v5 takeover calibrator.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-mode", choices=["min_total_rb", "max_total_rb"], default="max_total_rb")
    parser.add_argument("--threshold-grid", type=float, nargs="*", default=DEFAULT_THRESHOLD_GRID)
    return parser.parse_args()


def _normalized_regret(best_utility, chosen_utility, utility_min):
    spread = float(best_utility) - float(utility_min)
    if spread <= 1e-12:
        return 0.0
    return float((float(best_utility) - float(chosen_utility)) / spread)


def _family_value(row):
    if "action_family" not in row.index:
        return "other"
    value = str(row["action_family"])
    return value if value in FAMILY_COLUMNS else "other"


def build_takeover_rows(df, baseline_mode="max_total_rb"):
    required = ["decision_group_id", "total_rb", "v5_predicted_utility", "target_utility"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing takeover columns: {missing}")
    baseline_indices = baseline_choice_indices(df, baseline_mode=baseline_mode)
    rows = []
    for group_id, part in df.groupby("decision_group_id", dropna=False):
        baseline = part.loc[baseline_indices[group_id]]
        learned = part.loc[part["v5_predicted_utility"].idxmax()]
        best_utility = float(part["target_utility"].max())
        utility_min = float(part["target_utility"].min())
        learned_regret = _normalized_regret(best_utility, learned["target_utility"], utility_min)
        baseline_regret = _normalized_regret(best_utility, baseline["target_utility"], utility_min)
        learned_family = _family_value(learned)
        baseline_family = _family_value(baseline)
        row = {
            "decision_group_id": str(group_id),
            "split": str(part["split"].iloc[0]) if "split" in part.columns else "all",
            "seed": int(part["seed"].iloc[0]) if "seed" in part.columns else -1,
            "decision_time": float(part["decision_time"].iloc[0]) if "decision_time" in part.columns else 0.0,
            "num_candidates": int(len(part)),
            "baseline_index": int(baseline.name),
            "learned_index": int(learned.name),
            "baseline_candidate": str(baseline["candidate_id"]) if "candidate_id" in baseline.index else str(baseline.name),
            "learned_candidate": str(learned["candidate_id"]) if "candidate_id" in learned.index else str(learned.name),
            "baseline_family": baseline_family,
            "learned_family": learned_family,
            "baseline_total_rb": float(baseline["total_rb"]),
            "learned_total_rb": float(learned["total_rb"]),
            "learned_margin": float(learned["v5_predicted_utility"] - baseline["v5_predicted_utility"]),
            "learned_minus_baseline_rb": float(learned["total_rb"] - baseline["total_rb"]),
            "learned_regret": learned_regret,
            "baseline_regret": baseline_regret,
            "takeover_label": int(learned_regret < baseline_regret - 1e-12),
        }
        for family in FAMILY_COLUMNS:
            row[f"learned_family_{family}"] = float(learned_family == family)
            row[f"baseline_family_{family}"] = float(baseline_family == family)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["split", "seed", "decision_time", "decision_group_id"]).reset_index(drop=True)
    feature_columns = [
        "num_candidates",
        "learned_margin",
        "learned_minus_baseline_rb",
        "baseline_total_rb",
        "learned_total_rb",
        *[f"learned_family_{family}" for family in FAMILY_COLUMNS],
        *[f"baseline_family_{family}" for family in FAMILY_COLUMNS],
    ]
    out.attrs["feature_columns"] = feature_columns
    return out


def _fit_logistic_regression(x_train, y_train):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    if len(np.unique(y_train)) < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(x_train, y_train)
    return model


def fit_takeover_model(train_rows, feature_columns):
    x_train = train_rows[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_rows["takeover_label"].to_numpy(dtype=np.int64)
    model = _fit_logistic_regression(x_train, y_train)
    if model is not None:
        return {"kind": "logistic", "model": model}
    threshold = float(train_rows["learned_margin"].median()) if len(train_rows) else 0.0
    return {"kind": "margin_fallback", "threshold": threshold}


def predict_takeover_probability(fitted, rows, feature_columns):
    if fitted["kind"] == "logistic":
        x = rows[feature_columns].to_numpy(dtype=np.float32)
        return fitted["model"].predict_proba(x)[:, 1].astype(np.float32)
    margin = rows["learned_margin"].to_numpy(dtype=np.float32)
    return (margin >= float(fitted["threshold"])).astype(np.float32)


def apply_takeover_scores(df, takeover_rows, threshold, baseline_mode="max_total_rb"):
    required = ["decision_group_id", "total_rb", "v5_predicted_utility"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing takeover score columns: {missing}")
    if "takeover_probability" not in takeover_rows.columns:
        raise KeyError("takeover_probability is required")
    out = df.copy()
    learned_scores = out["v5_predicted_utility"].astype(float).copy()
    out["v5_predicted_utility"] = -1e9
    baseline_indices = baseline_choice_indices(out, baseline_mode=baseline_mode)
    takeover_map = {
        str(row["decision_group_id"]): float(row["takeover_probability"])
        for _, row in takeover_rows.iterrows()
    }
    for group_id, part in out.groupby("decision_group_id", dropna=False):
        base_idx = baseline_indices[group_id]
        learned_idx = int(learned_scores.loc[part.index].idxmax())
        probability = takeover_map.get(str(group_id), 0.0)
        chosen_idx = learned_idx if probability >= float(threshold) else base_idx
        out.loc[chosen_idx, "v5_predicted_utility"] = 1.0
    return out


def evaluate_takeover_threshold(df, takeover_rows, threshold, baseline_mode):
    scored = apply_takeover_scores(df, takeover_rows, threshold=threshold, baseline_mode=baseline_mode)
    return grouped_ranking_metrics(scored)


def select_threshold(train_df, train_rows, threshold_grid, baseline_mode):
    rows = []
    for threshold in threshold_grid:
        metrics = evaluate_takeover_threshold(train_df, train_rows, threshold, baseline_mode)
        rows.append({"threshold": float(threshold), **metrics})
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return 0.5, sweep
    best = sweep.sort_values(["normalized_top1_regret_mean", "top1_hit_mean"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), sweep


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    if "split" not in df.columns:
        raise KeyError("split is required")
    takeover_rows = build_takeover_rows(df, baseline_mode=args.baseline_mode)
    feature_columns = takeover_rows.attrs["feature_columns"]
    train_rows = takeover_rows[takeover_rows["split"].eq("train")].copy()
    fitted = fit_takeover_model(train_rows, feature_columns)
    takeover_rows["takeover_probability"] = predict_takeover_probability(fitted, takeover_rows, feature_columns)
    train_df = df[df["split"].eq("train")].copy()
    test_df = df[df["split"].eq("test")].copy()
    threshold, sweep = select_threshold(
        train_df,
        takeover_rows[takeover_rows["split"].eq("train")].copy(),
        args.threshold_grid,
        args.baseline_mode,
    )
    metric_rows = []
    for split_name, part in [("train", train_df), ("test", test_df), ("all", df)]:
        split_rows = takeover_rows if split_name == "all" else takeover_rows[takeover_rows["split"].eq(split_name)].copy()
        metrics = evaluate_takeover_threshold(part.copy(), split_rows, threshold, args.baseline_mode)
        metric_rows.append({"split": split_name, "threshold": threshold, **metrics})
    metrics_df = pd.DataFrame(metric_rows)
    takeover_path = args.output_dir / "world_model_v5_takeover_calibrator_v0_groups.csv"
    sweep_path = args.output_dir / "world_model_v5_takeover_calibrator_v0_threshold_sweep.csv"
    metrics_path = args.output_dir / "world_model_v5_takeover_calibrator_v0_metrics.csv"
    takeover_rows.to_csv(takeover_path, index=False, encoding="utf-8-sig")
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "predictions_csv": display_path(args.predictions_csv),
        "baseline_mode": args.baseline_mode,
        "model_kind": fitted["kind"],
        "feature_columns": feature_columns,
        "selected_threshold": threshold,
        "outputs": {
            "takeover_groups_csv": display_path(takeover_path),
            "threshold_sweep_csv": display_path(sweep_path),
            "metrics_csv": display_path(metrics_path),
        },
    }
    report_lines = [
        "# World model v5 takeover calibrator v0",
        "",
        "## Goal",
        "",
        f"Learn a train-only group-level takeover rule that decides when the learned v5 candidate should override the `{args.baseline_mode}` fallback.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Threshold Sweep",
        "",
        sweep.to_markdown(index=False),
        "",
        "## Takeover Features",
        "",
        pd.DataFrame({"feature": feature_columns}).to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        report_lines.append(f"- {key}: `{value}`")
    report_path = args.output_dir / "world_model_v5_takeover_calibrator_v0_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_takeover_calibrator_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
