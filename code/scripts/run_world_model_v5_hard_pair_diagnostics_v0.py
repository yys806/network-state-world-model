import argparse
import json
from pathlib import Path

import pandas as pd

from run_world_model_v5_utility_ranking_smoke import ROOT, display_path


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose hard seed-heldout v5 decision-ranking pairs.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pick_min_rb(part):
    return part.sort_values(["total_rb", "v5_predicted_utility"], ascending=[True, False]).iloc[0]


def summarize_decision_groups(df):
    required = [
        "decision_group_id",
        "candidate_id",
        "action_family",
        "total_rb",
        "target_utility",
        "v5_predicted_utility",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing diagnostic columns: {missing}")
    rows = []
    for group_id, part in df.groupby("decision_group_id", dropna=False):
        true = part.loc[part["target_utility"].idxmax()]
        learned = part.loc[part["v5_predicted_utility"].idxmax()]
        heuristic = pick_min_rb(part)
        utility_max = float(part["target_utility"].max())
        utility_min = float(part["target_utility"].min())
        spread = utility_max - utility_min
        rows.append(
            {
                "decision_group_id": str(group_id),
                "split": str(part["split"].iloc[0]) if "split" in part.columns else "all",
                "seed": int(part["seed"].iloc[0]) if "seed" in part.columns else -1,
                "decision_time": float(part["decision_time"].iloc[0]) if "decision_time" in part.columns else 0.0,
                "num_candidates": int(len(part)),
                "utility_spread": spread,
                "true_candidate": str(true["candidate_id"]),
                "true_family": str(true["action_family"]),
                "true_total_rb": float(true["total_rb"]),
                "learned_candidate": str(learned["candidate_id"]),
                "learned_family": str(learned["action_family"]),
                "learned_total_rb": float(learned["total_rb"]),
                "heuristic_candidate": str(heuristic["candidate_id"]),
                "heuristic_family": str(heuristic["action_family"]),
                "heuristic_total_rb": float(heuristic["total_rb"]),
                "learned_hit": bool(learned.name == true.name),
                "heuristic_hit": bool(heuristic.name == true.name),
                "learned_regret": float(utility_max - float(learned["target_utility"])),
                "heuristic_regret": float(utility_max - float(heuristic["target_utility"])),
                "learned_normalized_regret": float((utility_max - float(learned["target_utility"])) / spread) if spread > 1e-12 else 0.0,
                "heuristic_normalized_regret": float((utility_max - float(heuristic["target_utility"])) / spread) if spread > 1e-12 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "seed", "decision_time", "decision_group_id"]).reset_index(drop=True)


def summarize_by_family(group_df):
    if group_df.empty:
        return pd.DataFrame()
    rows = []
    for split_name, split_df in group_df.groupby("split", dropna=False):
        for true_family, part in split_df.groupby("true_family", dropna=False):
            rows.append(
                {
                    "split": str(split_name),
                    "true_family": str(true_family),
                    "num_groups": int(len(part)),
                    "learned_top1": float(part["learned_hit"].mean()),
                    "heuristic_top1": float(part["heuristic_hit"].mean()),
                    "learned_regret": float(part["learned_normalized_regret"].mean()),
                    "heuristic_regret": float(part["heuristic_normalized_regret"].mean()),
                    "mean_spread": float(part["utility_spread"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "true_family"]).reset_index(drop=True)


def summarize_confusions(group_df):
    if group_df.empty:
        return pd.DataFrame()
    return (
        group_df.groupby(["split", "true_family", "learned_family"], dropna=False)
        .size()
        .reset_index(name="num_groups")
        .sort_values(["split", "true_family", "num_groups"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def write_report(summary, group_df, family_df, confusion_df, output_dir):
    test_groups = group_df[group_df["split"].eq("test")].copy()
    worst = test_groups.sort_values("learned_normalized_regret", ascending=False).head(12)
    lines = [
        "# World model v5 hard-pair diagnostics v0",
        "",
        "## Goal",
        "",
        "Explain where the learned v5 decision model loses against AirFogSim counterfactual labels and resource-saving heuristics.",
        "",
        "## Summary",
        "",
        pd.DataFrame([summary["metrics"]]).to_markdown(index=False),
        "",
        "## Test Metrics By True Family",
        "",
        family_df[family_df["split"].eq("test")].to_markdown(index=False) if not family_df.empty else "No family diagnostics.",
        "",
        "## Test Learned-Family Confusions",
        "",
        confusion_df[confusion_df["split"].eq("test")].to_markdown(index=False) if not confusion_df.empty else "No confusion diagnostics.",
        "",
        "## Worst Test Groups",
        "",
        worst.to_markdown(index=False) if not worst.empty else "No test groups.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "world_model_v5_hard_pair_diagnostics_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    group_df = summarize_decision_groups(df)
    family_df = summarize_by_family(group_df)
    confusion_df = summarize_confusions(group_df)
    test = group_df[group_df["split"].eq("test")]
    metrics = {
        "num_test_groups": int(len(test)),
        "learned_top1": float(test["learned_hit"].mean()) if len(test) else 0.0,
        "heuristic_top1": float(test["heuristic_hit"].mean()) if len(test) else 0.0,
        "learned_regret": float(test["learned_normalized_regret"].mean()) if len(test) else 0.0,
        "heuristic_regret": float(test["heuristic_normalized_regret"].mean()) if len(test) else 0.0,
    }
    group_path = args.output_dir / "world_model_v5_hard_pair_group_diagnostics.csv"
    family_path = args.output_dir / "world_model_v5_hard_pair_family_diagnostics.csv"
    confusion_path = args.output_dir / "world_model_v5_hard_pair_confusions.csv"
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    family_df.to_csv(family_path, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    summary = {
        "predictions_csv": display_path(args.predictions_csv),
        "metrics": metrics,
        "outputs": {
            "group_diagnostics_csv": display_path(group_path),
            "family_diagnostics_csv": display_path(family_path),
            "confusions_csv": display_path(confusion_path),
        },
    }
    report_path = write_report(summary, group_df, family_df, confusion_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_hard_pair_diagnostics_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
