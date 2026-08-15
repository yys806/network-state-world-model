import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_CSV = (
    ROOT
    / "reports"
    / "airfogsim_counterfactual_offload_scaled_v2"
    / "airfogsim_counterfactual_multifamily_v0_labels.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "world_model_v5_family_winner_offload_scaled_v2_seedheldout_gpu"
DEFAULT_STATE_SAMPLE_INDEX_CSV = ROOT / "datasets" / "dataset_multiseed_seed0_9_v0" / "sample_index.csv"
DEFAULT_STATE_DATASET_NPZ = ROOT / "datasets" / "world_model_dataset_seed0_9_v0" / "world_model_dataset_v0_samples.npz"


SEED_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run v5 family winner head over seed-heldout pairs.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-sample-index-csv", type=Path, default=DEFAULT_STATE_SAMPLE_INDEX_CSV)
    parser.add_argument("--state-dataset-npz", type=Path, default=DEFAULT_STATE_DATASET_NPZ)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--reg-weight", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--winner-weight", type=float, default=0.5)
    parser.add_argument("--winner-gap-weight-power", type=float, default=0.0)
    parser.add_argument("--anchor-mode", choices=["none", "minus_total_rb", "plus_total_rb"], default="none")
    parser.add_argument("--utility-column", type=str, default="airfogsim_utility")
    parser.add_argument("--rb-penalty", type=float, default=0.001)
    return parser.parse_args(argv)


def run_pair(args, seed_pair):
    seed_tag = f"seed{seed_pair[0]}{seed_pair[1]}"
    gap_tag = f"_gap{args.winner_gap_weight_power:g}".replace(".", "p") if args.winner_gap_weight_power else ""
    pair_dir = args.output_dir / f"family_winner_w05_reg02_e120{gap_tag}_{seed_tag}"
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "run_world_model_v5_dual_graph_decision_head_v0.py"),
        "--candidate-csv",
        str(args.candidate_csv),
        "--state-sample-index-csv",
        str(args.state_sample_index_csv),
        "--state-dataset-npz",
        str(args.state_dataset_npz),
        "--output-dir",
        str(pair_dir),
        "--utility-column",
        args.utility_column,
        "--rb-penalty",
        str(args.rb_penalty),
        "--test-seeds",
        str(seed_pair[0]),
        str(seed_pair[1]),
        "--model-kind",
        "family_mlp_rank",
        "--compact-interactions",
        "--epochs",
        str(args.epochs),
        "--hidden",
        str(args.hidden),
        "--lr",
        str(args.lr),
        "--reg-weight",
        str(args.reg_weight),
        "--rank-weight",
        str(args.rank_weight),
        "--winner-weight",
        str(args.winner_weight),
        "--winner-gap-weight-power",
        str(args.winner_gap_weight_power),
        "--anchor-mode",
        args.anchor_mode,
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)
    metrics_path = pair_dir / "world_model_v5_dual_graph_decision_head_v0_metrics.csv"
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    return {
        "seed_pair": seed_tag,
        "heldout_seeds": f"{seed_pair[0]},{seed_pair[1]}",
        "output_dir": str(pair_dir),
        "test_top1": float(metrics["test_top1_hit_mean"]),
        "test_regret": float(metrics["test_normalized_top1_regret_mean"]),
        "test_spearman": float(metrics["test_spearman_mean"]),
        "train_top1": float(metrics["train_top1_hit_mean"]),
        "test_groups": int(metrics["test_num_groups"]),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_pair(args, pair) for pair in SEED_PAIRS]
    summary_df = pd.DataFrame(rows)
    summary_path = args.output_dir / "family_winner_seedheldout_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate = {
        "model": f"family_winner_w{args.winner_weight:g}_reg{args.reg_weight:g}_e{args.epochs}",
        "winner_gap_weight_power": float(args.winner_gap_weight_power),
        "mean_top1": float(summary_df["test_top1"].mean()),
        "mean_regret": float(summary_df["test_regret"].mean()),
        "mean_spearman": float(summary_df["test_spearman"].mean()),
        "mean_train_top1": float(summary_df["train_top1"].mean()),
        "groups": int(summary_df["test_groups"].sum()),
    }
    aggregate_df = pd.DataFrame([aggregate])
    aggregate_path = args.output_dir / "family_winner_seedheldout_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    report_path = args.output_dir / "world_model_v5_family_winner_seedheldout_batch_v0_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# v5 family winner seed-heldout batch v0",
                "",
                "## Aggregate",
                "",
                aggregate_df.to_markdown(index=False),
                "",
                "## Seed Pairs",
                "",
                summary_df.to_markdown(index=False),
                "",
                "## Outputs",
                "",
                f"- summary_csv: `{summary_path}`",
                f"- aggregate_csv: `{aggregate_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    machine_summary = {
        "candidate_csv": str(args.candidate_csv),
        "output_dir": str(args.output_dir),
        "summary_csv": str(summary_path),
        "aggregate_csv": str(aggregate_path),
        "report_md": str(report_path),
        "aggregate": aggregate,
    }
    (args.output_dir / "family_winner_seedheldout_batch_summary.json").write_text(
        json.dumps(machine_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(machine_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
