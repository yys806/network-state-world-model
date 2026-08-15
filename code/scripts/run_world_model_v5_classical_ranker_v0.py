import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
except ModuleNotFoundError:
    GradientBoostingRegressor = None
    RandomForestRegressor = None

from run_world_model_v5_utility_ranking_smoke import (
    DEFAULT_STATE_DATASET_NPZ,
    DEFAULT_STATE_SAMPLE_INDEX_CSV,
    ROOT,
    add_resource_aware_utility,
    build_candidate_features,
    display_path,
    enrich_candidates_with_state_features,
    ensure_decision_group_id,
    filter_state_available_groups,
    grouped_ranking_metrics,
    load_state_arrays,
    split_by_test_seeds,
    split_decision_groups,
)


DEFAULT_CANDIDATE_CSV = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_offload_scaled_v0"
    / "airfogsim_counterfactual_multifamily_v0_labels.csv"
)
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_classical_ranker_v0"


class StandardizedRidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.mean_ = None
        self.std_ = None
        self.coef_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        xs = (x - self.mean_) / self.std_
        design = np.column_stack([np.ones(xs.shape[0]), xs])
        eye = np.eye(design.shape[1], dtype=np.float64)
        eye[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + self.alpha * eye) @ design.T @ y
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("model must be fitted before prediction")
        x = np.asarray(x, dtype=np.float64)
        xs = (x - self.mean_) / self.std_
        design = np.column_stack([np.ones(xs.shape[0]), xs])
        return design @ self.coef_


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classical v5 utility scorers on grouped candidate ranking.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-kind", choices=["ridge", "gbr", "random_forest"], default="ridge")
    parser.add_argument("--utility-column", type=str, default="resource_aware_utility")
    parser.add_argument("--rb-penalty", type=float, default=0.001)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--use-state", action="store_true")
    parser.add_argument("--state-sample-index-csv", type=Path, default=DEFAULT_STATE_SAMPLE_INDEX_CSV)
    parser.add_argument("--state-dataset-npz", type=Path, default=DEFAULT_STATE_DATASET_NPZ)
    parser.add_argument("--require-state-available", action="store_true")
    return parser.parse_args()


def make_model(model_kind):
    if model_kind == "ridge":
        return StandardizedRidgeRegressor(alpha=1.0)
    if model_kind == "gbr":
        if GradientBoostingRegressor is None:
            raise ModuleNotFoundError("scikit-learn is required for model_kind='gbr'")
        return GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42,
        )
    if model_kind == "random_forest":
        if RandomForestRegressor is None:
            raise ModuleNotFoundError("scikit-learn is required for model_kind='random_forest'")
        return RandomForestRegressor(
            n_estimators=160,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        )
    raise ValueError(f"unknown model_kind: {model_kind}")


def evaluate_classical_model(df, feature_cols, train_idx, test_idx, model_kind="ridge"):
    required = {"decision_group_id", "target_utility", *feature_cols}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"missing classical ranker columns: {missing}")
    x = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    y = df["target_utility"].to_numpy(dtype=np.float32)
    model = make_model(model_kind)
    model.fit(x[train_idx], y[train_idx])
    pred = np.asarray(model.predict(x), dtype=np.float32)
    out = df.copy()
    out["v5_predicted_utility"] = pred
    rows = {}
    for split_name, split_idx in [("train", train_idx), ("test", test_idx), ("all", np.arange(len(out)))]:
        metrics = grouped_ranking_metrics(out.iloc[split_idx].copy())
        for key, value in metrics.items():
            rows[f"{split_name}_{key}"] = value
    return rows, pred


def write_report(summary, metrics_df, output_dir):
    lines = [
        "# World model v5 classical ranker v0",
        "",
        "## Goal",
        "",
        "This experiment checks whether a regularized classical scorer gives more stable counterfactual candidate ranking than the neural MLP utility head.",
        "",
        "## Setup",
        "",
        f"- candidate_csv: `{summary['candidate_csv']}`",
        f"- model_kind: `{summary['model_kind']}`",
        f"- feature_mode: `{summary['feature_mode']}`",
        f"- utility_column: `{summary['utility_column']}`",
        f"- rb_penalty: `{summary['rb_penalty']}`",
        f"- split_strategy: `{summary['split_strategy']}`",
        f"- feature_dim: `{summary['feature_dim']}`",
        f"- state_available_ratio: `{summary['state_available_ratio']}`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "world_model_v5_classical_ranker_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_decision_group_id(pd.read_csv(args.candidate_csv))
    df = add_resource_aware_utility(df, args.rb_penalty)
    feature_mode = "action_only"
    if args.use_state:
        sample_index = pd.read_csv(args.state_sample_index_csv)
        state_arrays = load_state_arrays(args.state_dataset_npz)
        df = enrich_candidates_with_state_features(df, sample_index, state_arrays)
        feature_mode = "state_action"
        if args.require_state_available:
            df = filter_state_available_groups(df)
            if df.empty:
                raise ValueError("no decision groups remain after requiring state_available for every candidate")
    if args.utility_column not in df.columns:
        raise KeyError(f"utility column not found: {args.utility_column}")
    df["target_utility"] = df[args.utility_column].to_numpy(dtype=np.float32)
    features = build_candidate_features(df)
    feature_cols = [f"feature_{idx:03d}" for idx in range(features.shape[1])]
    feature_df = pd.DataFrame(features, columns=feature_cols)
    model_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    if args.test_seeds:
        train_idx, test_idx = split_by_test_seeds(model_df, args.test_seeds)
        split_strategy = f"test_seeds_{','.join(str(seed) for seed in args.test_seeds)}"
    else:
        train_idx, test_idx = split_decision_groups(model_df, test_fraction=args.test_fraction, seed=args.split_seed)
        split_strategy = f"group_fraction_{args.test_fraction}_seed_{args.split_seed}"
    metrics, pred = evaluate_classical_model(model_df, feature_cols, train_idx, test_idx, args.model_kind)
    out_df = df.copy()
    out_df["v5_predicted_utility"] = pred
    out_df["split"] = "unused"
    out_df.loc[train_idx, "split"] = "train"
    out_df.loc[test_idx, "split"] = "test"
    metrics_df = pd.DataFrame([{**metrics, "model_kind": args.model_kind, "feature_mode": feature_mode}])
    predictions_path = args.output_dir / "world_model_v5_classical_ranker_v0_predictions.csv"
    metrics_path = args.output_dir / "world_model_v5_classical_ranker_v0_metrics.csv"
    out_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = {
        "candidate_csv": display_path(args.candidate_csv),
        "model_kind": args.model_kind,
        "feature_mode": feature_mode,
        "utility_column": args.utility_column,
        "rb_penalty": float(args.rb_penalty),
        "num_candidates": int(len(df)),
        "num_decision_groups": int(df["decision_group_id"].nunique()),
        "num_train_groups": int(out_df.loc[train_idx, "decision_group_id"].nunique()),
        "num_test_groups": int(out_df.loc[test_idx, "decision_group_id"].nunique()),
        "split_strategy": split_strategy,
        "feature_dim": int(features.shape[1]),
        "state_available_ratio": float(df["state_available"].mean()) if "state_available" in df.columns else 0.0,
        "metrics": metrics,
        "outputs": {
            "predictions_csv": display_path(predictions_path),
            "metrics_csv": display_path(metrics_path),
        },
    }
    report_path = write_report(summary, metrics_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_classical_ranker_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
