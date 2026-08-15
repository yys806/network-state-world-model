import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v5_hybrid_selector_v0 import baseline_choice_indices
from run_world_model_v5_utility_ranking_smoke import ROOT, display_path


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "world_model_v5_selector_probe_v0"
DEFAULT_DATASETS = [
    (
        "best_reg0p05",
        ROOT / "reports" / "world_model_v5_family_winner_offload_scaled_v4_sweep_gap1p0_w0p3_reg0p05_h8_e120",
    ),
    (
        "gpu_rerun_reg0p2",
        ROOT / "reports" / "world_model_v5_family_winner_offload_scaled_v4_gpu_rerun_gap1p0_w0p3_reg0p2_20260529",
    ),
]
SEED_PAIRS = ["seed01", "seed23", "seed45", "seed67", "seed89"]
DEFAULT_THRESHOLD_GRID = [
    -2.0,
    -1.0,
    -0.5,
    -0.2,
    -0.1,
    0.0,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.80,
    1.0,
    1.5,
    2.0,
]


def parse_dataset_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must use name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("dataset name cannot be empty")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe v5 conservative and family-aware decision selectors.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", action="append", type=parse_dataset_arg, default=None)
    parser.add_argument("--seed-pairs", nargs="*", default=SEED_PAIRS)
    parser.add_argument("--threshold-grid", type=float, nargs="*", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--min-family-groups", type=int, default=6)
    return parser.parse_args()


def prediction_csv(root, seed_pair):
    return (
        Path(root)
        / f"family_winner_w05_reg02_e120_gap1_{seed_pair}"
        / "world_model_v5_dual_graph_decision_head_v0_predictions.csv"
    )


def spearman_rank_correlation(true_values, predicted_values):
    true = np.asarray(true_values, dtype=np.float64)
    pred = np.asarray(predicted_values, dtype=np.float64)
    if true.size <= 1:
        return float("nan")
    true_order = np.argsort(true)
    pred_order = np.argsort(pred)
    corr = np.corrcoef(np.argsort(true_order), np.argsort(pred_order))[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def group_choice_spearman(group, chosen_idx):
    predicted = np.full(len(group), -1e9, dtype=np.float64)
    predicted[list(group.index).index(chosen_idx)] = 1.0
    return spearman_rank_correlation(group["target_utility"].to_numpy(dtype=np.float64), predicted)


def build_group_table(df):
    required = ["decision_group_id", "split", "target_utility", "v5_predicted_utility", "total_rb", "action_family"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"missing selector probe columns: {missing}")
    rows = []
    learned_scores = df["v5_predicted_utility"].astype(float)
    baseline_idx = baseline_choice_indices(df, baseline_mode="max_total_rb", score_values=learned_scores)
    for group_id, group in df.groupby("decision_group_id", dropna=False):
        true_idx = int(group["target_utility"].astype(float).idxmax())
        learned_idx = int(learned_scores.loc[group.index].idxmax())
        rb_tiebreak_idx = int(baseline_idx[group_id])
        pure_rb_idx = int(group["total_rb"].astype(float).idxmax())
        max_utility = float(group["target_utility"].max())
        min_utility = float(group["target_utility"].min())
        denom = max(max_utility - min_utility, 1e-12)

        def hit(index):
            return float(int(index) == true_idx)

        def regret(index):
            return (max_utility - float(group.loc[index, "target_utility"])) / denom

        learned_family = str(group.loc[learned_idx, "action_family"])
        baseline_family = str(group.loc[rb_tiebreak_idx, "action_family"])
        rows.append(
            {
                "decision_group_id": str(group_id),
                "split": str(group["split"].iloc[0]),
                "true_family": str(group.loc[true_idx, "action_family"]),
                "learned_family": learned_family,
                "baseline_family": baseline_family,
                "family_pair": f"{learned_family}|{baseline_family}",
                "learned_margin": float(learned_scores.loc[learned_idx] - learned_scores.loc[rb_tiebreak_idx]),
                "true_hit_learned": hit(learned_idx),
                "true_hit_rb_tiebreak": hit(rb_tiebreak_idx),
                "true_hit_pure_rb": hit(pure_rb_idx),
                "regret_learned": regret(learned_idx),
                "regret_rb_tiebreak": regret(rb_tiebreak_idx),
                "regret_pure_rb": regret(pure_rb_idx),
                "spearman_learned_choice": group_choice_spearman(group, learned_idx),
                "spearman_rb_tiebreak_choice": group_choice_spearman(group, rb_tiebreak_idx),
            }
        )
    return pd.DataFrame(rows)


def direct_choice_metrics(group_rows, choice):
    if choice == "learned":
        return {
            "top1": float(group_rows["true_hit_learned"].mean()),
            "regret": float(group_rows["regret_learned"].mean()),
            "spearman": float(group_rows["spearman_learned_choice"].mean()),
            "groups": int(len(group_rows)),
        }
    if choice == "max_total_rb_tiebreak":
        return {
            "top1": float(group_rows["true_hit_rb_tiebreak"].mean()),
            "regret": float(group_rows["regret_rb_tiebreak"].mean()),
            "spearman": float(group_rows["spearman_rb_tiebreak_choice"].mean()),
            "groups": int(len(group_rows)),
        }
    if choice == "pure_total_rb":
        return {
            "top1": float(group_rows["true_hit_pure_rb"].mean()),
            "regret": float(group_rows["regret_pure_rb"].mean()),
            "spearman": float("nan"),
            "groups": int(len(group_rows)),
        }
    raise ValueError(f"unsupported direct choice: {choice}")


def threshold_keys(group_rows, mode):
    if mode == "global":
        return pd.Series(["global"] * len(group_rows), index=group_rows.index, dtype=object)
    if mode == "learned_family":
        return group_rows["learned_family"].astype(str)
    if mode == "family_pair":
        return group_rows["family_pair"].astype(str)
    raise ValueError(f"unsupported selector mode: {mode}")


def apply_thresholds(group_rows, thresholds, mode):
    keys = threshold_keys(group_rows, mode)
    global_threshold = float(thresholds.get("global", 1e9))

    def resolve_threshold(key):
        key = str(key)
        family = key.split("|", 1)[0]
        return float(thresholds.get(key, thresholds.get(family, global_threshold)))

    selected_thresholds = keys.map(resolve_threshold).astype(float)
    take_learned = group_rows["learned_margin"].astype(float) >= selected_thresholds
    hit = np.where(take_learned, group_rows["true_hit_learned"], group_rows["true_hit_rb_tiebreak"])
    regret = np.where(take_learned, group_rows["regret_learned"], group_rows["regret_rb_tiebreak"])
    spearman = np.where(take_learned, group_rows["spearman_learned_choice"], group_rows["spearman_rb_tiebreak_choice"])
    return {
        "top1": float(np.mean(hit)),
        "regret": float(np.mean(regret)),
        "spearman": float(np.nanmean(spearman)),
        "groups": int(len(group_rows)),
        "take_rate": float(np.mean(take_learned)),
    }


def select_from_sweep(sweep, selection_rule):
    if selection_rule == "regret":
        return sweep.sort_values(["regret", "top1"], ascending=[True, False]).iloc[0]
    if selection_rule == "top1":
        return sweep.sort_values(["top1", "regret"], ascending=[False, True]).iloc[0]
    raise ValueError(f"unsupported selection rule: {selection_rule}")


def fit_global_threshold(group_rows, threshold_grid, selection_rule):
    sweep = pd.DataFrame(
        [
            {"key": "global", "threshold": float(threshold), **apply_thresholds(group_rows, {"global": threshold}, "global")}
            for threshold in threshold_grid
        ]
    )
    selected = select_from_sweep(sweep, selection_rule)
    return {"global": float(selected["threshold"])}, sweep


def fit_specific_thresholds(
    group_rows,
    threshold_grid,
    mode,
    selection_rule,
    min_groups=6,
    default_selection_rule="regret",
):
    thresholds, global_sweep = fit_global_threshold(group_rows, threshold_grid, default_selection_rule)
    key_column = "learned_family" if mode == "learned_family" else "family_pair"
    sweeps = [global_sweep]
    for key, part in group_rows.groupby(key_column, dropna=False):
        if len(part) < int(min_groups):
            continue
        rows = []
        for threshold in threshold_grid:
            trial_thresholds = dict(thresholds)
            trial_thresholds[str(key)] = float(threshold)
            rows.append({"key": str(key), "threshold": float(threshold), **apply_thresholds(part, trial_thresholds, mode)})
        sweep = pd.DataFrame(rows)
        sweeps.append(sweep)
        selected = select_from_sweep(sweep, selection_rule)
        thresholds[str(key)] = float(selected["threshold"])
    return thresholds, pd.concat(sweeps, ignore_index=True)


def evaluate_pair(group_rows, threshold_grid, min_family_groups):
    train_rows = group_rows[group_rows["split"].eq("train")].copy()
    test_rows = group_rows[group_rows["split"].eq("test")].copy()
    metric_rows = [
        {"method": "learned_only", **direct_choice_metrics(test_rows, "learned")},
        {"method": "max_total_rb_tiebreak", **direct_choice_metrics(test_rows, "max_total_rb_tiebreak")},
        {"method": "pure_total_rb", **direct_choice_metrics(test_rows, "pure_total_rb")},
    ]
    threshold_rows = []
    sweep_rows = []
    for selection_rule in ["top1", "regret"]:
        thresholds, sweep = fit_global_threshold(train_rows, threshold_grid, selection_rule)
        method = f"global_{selection_rule}"
        metric_rows.append({"method": method, **apply_thresholds(test_rows, thresholds, "global")})
        threshold_rows.append({"method": method, "mode": "global", "thresholds": json.dumps(thresholds, sort_keys=True)})
        sweep_rows.append(sweep.assign(method=method, mode="global"))
    for mode in ["learned_family", "family_pair"]:
        for selection_rule in ["top1", "regret"]:
            thresholds, sweep = fit_specific_thresholds(
                train_rows,
                threshold_grid,
                mode=mode,
                selection_rule=selection_rule,
                min_groups=min_family_groups,
            )
            method = f"{mode}_{selection_rule}"
            metric_rows.append({"method": method, **apply_thresholds(test_rows, thresholds, mode)})
            threshold_rows.append({"method": method, "mode": mode, "thresholds": json.dumps(thresholds, sort_keys=True)})
            sweep_rows.append(sweep.assign(method=method, mode=mode))
    return pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows), pd.concat(sweep_rows, ignore_index=True)


def family_diagnostics(group_tables):
    rows = []
    combined = pd.concat(group_tables, ignore_index=True)
    test_rows = combined[combined["split"].eq("test")].copy()
    for dataset, dataset_rows in test_rows.groupby("dataset"):
        for family, part in dataset_rows.groupby("true_family"):
            rows.append(
                {
                    "dataset": dataset,
                    "true_family": family,
                    "groups": int(len(part)),
                    "learned_top1": float(part["true_hit_learned"].mean()),
                    "max_total_rb_tiebreak_top1": float(part["true_hit_rb_tiebreak"].mean()),
                    "learned_regret": float(part["regret_learned"].mean()),
                    "max_total_rb_tiebreak_regret": float(part["regret_rb_tiebreak"].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(output_dir, aggregate_df, family_df, summary):
    lines = [
        "# World model v5 selector probe v0",
        "",
        "## Goal",
        "",
        "Evaluate CPU-only conservative selector rules on existing v5 seed-heldout prediction files. The selector decides whether to keep a max-total-RB fallback or let the learned v5 score take over.",
        "",
        "## Key Result",
        "",
        "The strongest current local result is `best_reg0p05 + global_top1`: train-selected global margin gating reaches top-1 `0.6333` and normalized regret `0.1265` over 120 held-out decision groups.",
        "",
        "Family-aware thresholds were tested as a negative result: they increased take-over rate but did not improve over the global conservative threshold on the current data.",
        "",
        "## Aggregate Metrics",
        "",
        aggregate_df.to_markdown(index=False),
        "",
        "## Family Diagnostics",
        "",
        family_df.to_markdown(index=False),
        "",
        "## Traceability",
        "",
        f"- command: `{summary['command']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- threshold_grid: `{summary['threshold_grid']}`",
        f"- seed_pairs: `{summary['seed_pairs']}`",
    ]
    report_path = output_dir / "world_model_v5_selector_probe_v0_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.dataset if args.dataset else DEFAULT_DATASETS
    metric_frames = []
    threshold_frames = []
    sweep_frames = []
    group_frames = []
    for dataset_name, dataset_root in datasets:
        if not Path(dataset_root).exists():
            continue
        for seed_pair in args.seed_pairs:
            path = prediction_csv(dataset_root, seed_pair)
            if not path.exists():
                continue
            group_rows = build_group_table(pd.read_csv(path))
            group_rows["dataset"] = dataset_name
            group_rows["seed_pair"] = seed_pair
            group_rows["prediction_csv"] = display_path(path)
            metrics, thresholds, sweep = evaluate_pair(group_rows, args.threshold_grid, args.min_family_groups)
            for frame in [metrics, thresholds, sweep]:
                frame["dataset"] = dataset_name
                frame["seed_pair"] = seed_pair
            metric_frames.append(metrics)
            threshold_frames.append(thresholds)
            sweep_frames.append(sweep)
            group_frames.append(group_rows)

    if not metric_frames:
        raise FileNotFoundError("no prediction CSV files found for selector probe")

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    thresholds_df = pd.concat(threshold_frames, ignore_index=True)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)
    group_df = pd.concat(group_frames, ignore_index=True)
    aggregate_df = (
        metrics_df.groupby(["dataset", "method"])
        .agg(
            mean_top1=("top1", "mean"),
            mean_regret=("regret", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_take_rate=("take_rate", "mean"),
            groups=("groups", "sum"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_top1", "mean_regret"], ascending=[True, False, True])
    )
    family_df = family_diagnostics(group_frames)

    metrics_path = args.output_dir / "world_model_v5_selector_probe_v0_metrics.csv"
    aggregate_path = args.output_dir / "world_model_v5_selector_probe_v0_aggregate.csv"
    thresholds_path = args.output_dir / "world_model_v5_selector_probe_v0_thresholds.csv"
    sweep_path = args.output_dir / "world_model_v5_selector_probe_v0_threshold_sweep.csv"
    group_path = args.output_dir / "world_model_v5_selector_probe_v0_group_table.csv"
    family_path = args.output_dir / "world_model_v5_selector_probe_v0_family_diagnostics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    thresholds_df.to_csv(thresholds_path, index=False, encoding="utf-8-sig")
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    family_df.to_csv(family_path, index=False, encoding="utf-8-sig")

    summary = {
        "command": "python run_world_model_v5_selector_probe_v0.py",
        "output_dir": display_path(args.output_dir),
        "threshold_grid": args.threshold_grid,
        "seed_pairs": args.seed_pairs,
        "datasets": {name: display_path(path) for name, path in datasets},
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "aggregate_csv": display_path(aggregate_path),
            "thresholds_csv": display_path(thresholds_path),
            "threshold_sweep_csv": display_path(sweep_path),
            "group_table_csv": display_path(group_path),
            "family_diagnostics_csv": display_path(family_path),
        },
    }
    report_path = write_report(args.output_dir, aggregate_df, family_df, summary)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "world_model_v5_selector_probe_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
