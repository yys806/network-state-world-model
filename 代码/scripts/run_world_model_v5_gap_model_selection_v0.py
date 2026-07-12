import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "world_model_v5_gap_model_selection_v0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select v5 gap-weighted models using train metrics only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("MODEL", "DIR"), default=[])
    parser.add_argument("--rule", choices=["regret_first", "top1_first"], default="regret_first")
    return parser.parse_args(argv)


def select_models_by_train_metrics(candidates, rule="regret_first"):
    required = {"seed_pair", "model", "train_top1", "train_regret", "train_spearman"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise KeyError(f"missing columns for train-only selection: {missing}")
    rows = []
    for _, group in candidates.groupby("seed_pair", sort=True):
        if rule == "regret_first":
            sort_cols = ["train_regret", "train_top1", "train_spearman", "model"]
            ascending = [True, False, False, True]
        elif rule == "top1_first":
            sort_cols = ["train_top1", "train_regret", "train_spearman", "model"]
            ascending = [False, True, False, True]
        else:
            raise ValueError(f"unsupported selection rule: {rule}")
        ranked = group.sort_values(
            sort_cols,
            ascending=ascending,
        )
        rows.append(ranked.iloc[0].to_dict())
    return pd.DataFrame(rows)


def collect_candidate_rows(candidate_specs):
    rows = []
    for model_name, folder in candidate_specs:
        folder = Path(folder)
        summary_path = folder / "family_winner_seedheldout_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path)
        for row in summary.itertuples(index=False):
            metrics_path = Path(getattr(row, "output_dir")) / "world_model_v5_dual_graph_decision_head_v0_metrics.csv"
            if not metrics_path.is_absolute():
                metrics_path = ROOT / metrics_path
            metrics = pd.read_csv(metrics_path).iloc[0]
            rows.append(
                {
                    "seed_pair": str(getattr(row, "seed_pair")),
                    "heldout_seeds": str(getattr(row, "heldout_seeds")),
                    "model": model_name,
                    "model_dir": str(folder),
                    "pair_dir": str(metrics_path.parent),
                    "train_top1": float(metrics["train_top1_hit_mean"]),
                    "train_regret": float(metrics["train_normalized_top1_regret_mean"]),
                    "train_spearman": float(metrics["train_spearman_mean"]),
                    "test_top1": float(metrics["test_top1_hit_mean"]),
                    "test_regret": float(metrics["test_normalized_top1_regret_mean"]),
                    "test_spearman": float(metrics["test_spearman_mean"]),
                    "test_groups": int(metrics["test_num_groups"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_selected(selected):
    return pd.DataFrame(
        [
            {
                "model": "train_selected_gap_variant",
                "mean_top1": float(selected["test_top1"].mean()),
                "mean_regret": float(selected["test_regret"].mean()),
                "mean_spearman": float(selected["test_spearman"].mean()),
                "groups": int(selected["test_groups"].sum()),
            }
        ]
    )


def main(argv=None):
    args = parse_args(argv)
    if not args.candidate:
        raise ValueError("at least one --candidate MODEL DIR is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidate_rows(args.candidate)
    selected = select_models_by_train_metrics(candidates, rule=args.rule)
    aggregate = aggregate_selected(selected)

    candidates_path = args.output_dir / "gap_model_selection_candidates.csv"
    selected_path = args.output_dir / "gap_model_selection_selected.csv"
    aggregate_path = args.output_dir / "gap_model_selection_aggregate.csv"
    report_path = args.output_dir / "gap_model_selection_report.md"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        "\n".join(
            [
                "# v5 gap model selection v0",
                "",
                "## Aggregate",
                "",
                aggregate.to_markdown(index=False),
                "",
                "## Selected Per Seed Pair",
                "",
                selected.to_markdown(index=False),
                "",
                "## Selection Rule",
                "",
                f"Rule: `{args.rule}`. Test metrics are used only after selection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
