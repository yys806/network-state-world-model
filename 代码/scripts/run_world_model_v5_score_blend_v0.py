import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_decision_baselines_v0 import add_decision_baseline_scores
from run_world_model_v5_utility_ranking_smoke import ROOT, display_path, grouped_ranking_metrics


DEFAULT_PREDICTIONS_CSV = (
    ROOT
    / "reports"
    / "world_model_v5_family_winner_offload_scaled_v3_seedheldout_gpu_w05_reg01"
    / "family_winner_w05_reg02_e120_seed89"
    / "world_model_v5_dual_graph_decision_head_v0_predictions.csv"
)
OUTPUT_DIR = ROOT / "reports" / "world_model_v5_score_blend_v0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Blend v5 learned score with fair action priors using train groups only.")
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prior-cols", type=str, nargs="*", default=["predict_total_rb", "predict_default"])
    parser.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
    return parser.parse_args(argv)


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(np.nanstd(values))
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(np.nanmean(values))) / std).astype(np.float32)


def ensure_scores(df):
    out = add_decision_baseline_scores(df)
    if "target_utility" not in out.columns:
        if "airfogsim_utility" not in out.columns:
            raise KeyError("target_utility or airfogsim_utility is required")
        out["target_utility"] = out["airfogsim_utility"].to_numpy(dtype=np.float32)
    if "v5_predicted_utility" not in out.columns:
        raise KeyError("v5_predicted_utility is required")
    return out


def evaluate_score(df, score_col, split_name):
    part = df[df["split"].eq(split_name)].copy()
    if part.empty:
        return {}
    part["v5_predicted_utility"] = part[score_col].to_numpy(dtype=np.float32)
    out = grouped_ranking_metrics(part)
    out["split"] = split_name
    return out


def evaluate_blend_candidates(df, lambdas=None, prior_cols=None):
    lambdas = list(lambdas or [0.0, 0.5, 1.0])
    prior_cols = list(prior_cols or ["predict_total_rb"])
    base = ensure_scores(df)
    learned = zscore(base["v5_predicted_utility"].to_numpy(dtype=np.float32))
    rows = []
    for prior_col in prior_cols:
        if prior_col not in base.columns:
            continue
        prior = zscore(base[prior_col].to_numpy(dtype=np.float32))
        for value in lambdas:
            score_col = f"blend_{prior_col}_{value:g}".replace(".", "p")
            work = base.copy()
            work[score_col] = (1.0 - float(value)) * learned + float(value) * prior
            train_metrics = evaluate_score(work, score_col, "train")
            test_metrics = evaluate_score(work, score_col, "test")
            for metrics in [train_metrics, test_metrics]:
                if not metrics:
                    continue
                metrics.update(
                    {
                        "prior_col": prior_col,
                        "lambda": float(value),
                        "score_col": score_col,
                        "is_selected": False,
                    }
                )
                rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise ValueError("no blend candidates were evaluated")
    train = metrics_df[metrics_df["split"].eq("train")].copy()
    train = train.sort_values(["top1_hit_mean", "normalized_top1_regret_mean"], ascending=[False, True])
    selected = train.iloc[0][["prior_col", "lambda", "score_col"]].to_dict()
    mask = (metrics_df["prior_col"].eq(selected["prior_col"])) & (metrics_df["lambda"].eq(float(selected["lambda"])))
    metrics_df.loc[mask, "is_selected"] = True
    return selected, metrics_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    selected, metrics = evaluate_blend_candidates(df, lambdas=args.lambdas, prior_cols=args.prior_cols)
    metrics_path = args.output_dir / "world_model_v5_score_blend_v0_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    selected_path = args.output_dir / "world_model_v5_score_blend_v0_selected.json"
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output_dir / "world_model_v5_score_blend_v0_report.md"
    selected_metrics = metrics[metrics["is_selected"]].copy()
    report_path.write_text(
        "\n".join(
            [
                "# v5 score blend v0",
                "",
                "## Selected",
                "",
                json.dumps(selected, ensure_ascii=False, indent=2),
                "",
                "## Selected Metrics",
                "",
                selected_metrics.to_markdown(index=False),
                "",
                "## Outputs",
                "",
                f"- metrics_csv: `{display_path(metrics_path)}`",
                f"- selected_json: `{display_path(selected_path)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "predictions_csv": display_path(args.predictions_csv),
        "selected": selected,
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "selected_json": display_path(selected_path),
            "report_md": display_path(report_path),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
