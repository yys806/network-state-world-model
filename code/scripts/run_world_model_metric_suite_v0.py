import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from run_world_model_v0 import FIGURE_DIR, ROOT
from run_world_model_v4_dual_graph_rollout import display_path


OUTPUT_DIR = ROOT / "reports" / "world_model_metric_suite_v0"


def read_csv_if_exists(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def probability_calibration_metrics(y_true, prob, n_bins=10):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    if y.size != p.size:
        raise ValueError("y_true and prob must have the same flattened size")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        mean_prob = float(p[mask].mean())
        empirical = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(mean_prob - empirical)
        rows.append(
            {
                "bin": idx,
                "prob_min": float(lo),
                "prob_max": float(hi),
                "mean_prob": mean_prob,
                "empirical_active_ratio": empirical,
                "count": int(mask.sum()),
            }
        )
    return {
        "brier_score": float(brier_score_loss(y.astype(int), p)),
        "ece": float(ece),
        "num_items": int(y.size),
        "num_bins_used": int(len(rows)),
    }


def summarize_metric_variation(df, group_cols, metric_cols):
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["runs"] = int(len(part))
        for col in metric_cols:
            if col not in part.columns:
                continue
            values = pd.to_numeric(part[col], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def add_metric_row(rows, category, metric, value, model="", split="", scope="", source="", target="", note=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    rows.append(
        {
            "category": category,
            "target": target,
            "metric": metric,
            "model": model,
            "split": split,
            "scope": scope,
            "value": float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value,
            "source": source,
            "note": note,
        }
    )


def collect_prediction_rows():
    rows = []
    metric_files = [
        ROOT / "reports" / "world_model_v4_dual_graph_rollout" / "world_model_v4_dual_graph_rollout_metrics.csv",
        ROOT / "reports" / "world_model_v4_dual_graph_ablation" / "world_model_v4_dual_graph_ablation_metrics.csv",
        ROOT / "reports" / "world_model_v4_seed_stability" / "world_model_v4_seed_stability_metrics.csv",
        ROOT / "reports" / "world_model_v3_graph_rollout" / "world_model_v3_graph_rollout_metrics.csv",
        ROOT / "reports" / "world_model_v2_latent_rollout" / "world_model_v2_latent_rollout_metrics.csv",
    ]
    for path in metric_files:
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        test = df[df.get("split", "").eq("test_seed_4")].copy()
        for _, row in test.iterrows():
            model = str(row.get("model", row.get("physical_variant", "")))
            source = display_path(path)
            for col, target, metric in [
                ("activity_precision", "link_activity", "precision"),
                ("activity_recall", "link_activity", "recall"),
                ("activity_f1", "link_activity", "f1"),
                ("activity_ap", "link_activity", "average_precision"),
                ("activity_auc", "link_activity", "roc_auc"),
                ("rate_all_rmse", "link_rate", "all_rmse"),
                ("rate_active_rmse", "active_link_rate", "active_rmse"),
                ("task_rmse", "task_state", "rmse"),
            ]:
                if col in row and pd.notna(row[col]):
                    add_metric_row(
                        rows,
                        "prediction",
                        metric,
                        row[col],
                        model=model,
                        split="test_seed_4",
                        target=target,
                        source=source,
                    )
    return pd.DataFrame(rows)


def collect_activity_calibration_rows():
    rows = []
    path = ROOT / "reports" / "world_model_v4_activity_calibration" / "world_model_v4_activity_calibration_summary.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    for _, row in df.iterrows():
        strategy = row["strategy"]
        for metric_col, metric in [
            ("precision_mean", "precision_mean"),
            ("precision_std", "precision_std"),
            ("recall_mean", "recall_mean"),
            ("recall_std", "recall_std"),
            ("f1_mean", "f1_mean"),
            ("f1_std", "f1_std"),
            ("predicted_active_ratio_mean", "predicted_active_ratio_mean"),
            ("predicted_active_ratio_std", "predicted_active_ratio_std"),
        ]:
            add_metric_row(
                rows,
                "activity_gating",
                metric,
                row[metric_col],
                model="world_model_v4_dual_full",
                split="test_seed_4",
                scope=strategy,
                target="link_activity",
                source=display_path(path),
            )
    return pd.DataFrame(rows)


def collect_probability_rows():
    rows = []
    path = ROOT / "reports" / "world_model_v4_activity_calibration" / "prediction_cache"
    dataset_path = ROOT / "datasets" / "world_model_dataset_v0" / "world_model_dataset_v0_samples.npz"
    if not path.exists() or not dataset_path.exists():
        return pd.DataFrame(rows)
    from run_world_model_v0 import load_dataset, split_by_seed

    arrays = load_dataset()
    _, _, test_idx = split_by_seed(arrays["sample_seed"])
    y_test = arrays["y_link_active"][test_idx]
    for cache in sorted(path.glob("v4_activity_dual_full_seed_*_predictions.npz")):
        data = np.load(cache, allow_pickle=True)
        prob = data["test_active_prob"]
        metrics = probability_calibration_metrics(y_test, prob, n_bins=10)
        seed = cache.stem.split("_seed_")[-1].replace("_predictions", "")
        for metric in ["brier_score", "ece", "num_items", "num_bins_used"]:
            add_metric_row(
                rows,
                "probability_calibration",
                metric,
                metrics[metric],
                model="world_model_v4_dual_full",
                split="test_seed_4",
                scope=f"torch_seed_{seed}",
                target="link_activity_probability",
                source=display_path(cache),
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        pivot = df[df["metric"].isin(["brier_score", "ece"])].copy()
        summary = summarize_metric_variation(
            pivot.rename(columns={"value": "metric_value"}),
            group_cols=["metric"],
            metric_cols=["metric_value"],
        )
        for _, row in summary.iterrows():
            add_metric_row(
                rows,
                "probability_calibration",
                f"{row['metric']}_mean",
                row.get("metric_value_mean"),
                model="world_model_v4_dual_full",
                split="test_seed_4",
                scope="torch_seed_11_42_73",
                target="link_activity_probability",
                source=display_path(path),
            )
            add_metric_row(
                rows,
                "probability_calibration",
                f"{row['metric']}_std",
                row.get("metric_value_std"),
                model="world_model_v4_dual_full",
                split="test_seed_4",
                scope="torch_seed_11_42_73",
                target="link_activity_probability",
                source=display_path(path),
            )
    return pd.DataFrame(rows)


def collect_active_rate_rows():
    rows = []
    metric_path = ROOT / "reports" / "world_model_v4_active_rate_calibration" / "world_model_v4_active_rate_metrics.csv"
    interval_path = ROOT / "reports" / "world_model_v4_active_rate_calibration" / "world_model_v4_active_rate_intervals.csv"
    metrics = read_csv_if_exists(metric_path)
    if not metrics.empty:
        test = metrics[metrics["split"].eq("test_seed_4")].copy()
        for _, row in test.iterrows():
            model = row["model"]
            scope = row["activity_policy"]
            for col, metric in [
                ("active_rmse", "active_rmse"),
                ("active_mae", "active_mae"),
                ("all_rmse", "all_rmse"),
                ("predicted_active_ratio", "predicted_active_ratio"),
            ]:
                if col in row and pd.notna(row[col]):
                    add_metric_row(
                        rows,
                        "active_rate",
                        metric,
                        row[col],
                        model=model,
                        split="test_seed_4",
                        scope=scope,
                        target="active_link_rate",
                        source=display_path(metric_path),
                    )
    intervals = read_csv_if_exists(interval_path)
    if not intervals.empty:
        for _, row in intervals.iterrows():
            coverage_col = "coverage" if "coverage" in intervals.columns else "active_coverage"
            width_col = "mean_width" if "mean_width" in intervals.columns else "active_mean_width"
            for col, metric in [(coverage_col, "interval_coverage"), (width_col, "interval_mean_width")]:
                add_metric_row(
                    rows,
                    "uncertainty_interval",
                    metric,
                    row[col],
                    model=row["model"],
                    split="test_seed_4",
                    scope=f"{row['activity_policy']}:{row['interval']}",
                    target="active_link_rate",
                    source=display_path(interval_path),
                )
    return pd.DataFrame(rows)


def collect_robustness_rows():
    rows = []
    path = ROOT / "reports" / "world_model_v3_diagnostics" / "world_model_v3_robustness_metrics.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    for _, row in df.iterrows():
        scope = f"noise_{row['noise_level']:.2f}"
        for col, target, metric in [
            ("activity_f1", "link_activity", "f1"),
            ("rate_all_rmse", "link_rate", "all_rmse"),
            ("task_rmse", "task_state", "rmse"),
        ]:
            add_metric_row(
                rows,
                "robustness",
                metric,
                row[col],
                model="world_model_v3_graph_rollout",
                split="test_seed_4",
                scope=scope,
                target=target,
                source=display_path(path),
            )
    return pd.DataFrame(rows)


def collect_runtime_rows():
    rows = []
    path = ROOT / "reports" / "world_model_runtime_comparison_v0" / "world_model_runtime_comparison_v0_summary.json"
    if not path.exists():
        return pd.DataFrame(rows)
    summary = json.loads(path.read_text(encoding="utf-8"))
    speed = summary["speedup"]
    for metric in ["sim_step_ms", "sim_k_step_ms", "model_sample_ms", "speedup"]:
        add_metric_row(
            rows,
            "runtime_comparison",
            metric,
            speed[metric],
            model="world_model_v4_dual_graph_forward",
            split="test_seed_4",
            scope=f"{speed['horizon']}_step_cpu",
            target="decision_runtime",
            source=display_path(path),
        )
    return pd.DataFrame(rows)


def collect_physical_rollout_rows():
    rows = []
    path = ROOT / "reports" / "world_model_physical_rollout_baseline_v0" / "world_model_physical_rollout_baseline_v0_metrics.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    test = df[df["split"].eq("test_seed_4")].copy()
    for _, row in test.iterrows():
        for col, target, metric in [
            ("position_rmse", "node_position", "rmse"),
            ("speed_rmse", "node_speed", "rmse"),
            ("edge_distance_rmse", "physical_edge_distance", "rmse"),
            ("U2I_distance_rmse", "physical_edge_distance_U2I", "rmse"),
            ("V2I_distance_rmse", "physical_edge_distance_V2I", "rmse"),
            ("V2U_distance_rmse", "physical_edge_distance_V2U", "rmse"),
        ]:
            if col in row and pd.notna(row[col]):
                add_metric_row(
                    rows,
                    "physical_graph_rollout",
                    metric,
                    row[col],
                    model=row["model"],
                    split="test_seed_4",
                    scope=col,
                    target=target,
                    source=display_path(path),
                )
    return pd.DataFrame(rows)


def collect_seed_stability_rows():
    rows = []
    path = ROOT / "reports" / "world_model_v4_seed_stability" / "world_model_v4_seed_stability_summary.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    for _, row in df.iterrows():
        variant = row["physical_variant"]
        for col, target, metric in [
            ("activity_f1_mean", "link_activity", "f1_mean"),
            ("activity_f1_std", "link_activity", "f1_std"),
            ("activity_precision_mean", "link_activity", "precision_mean"),
            ("activity_recall_mean", "link_activity", "recall_mean"),
            ("rate_all_rmse_mean", "link_rate", "all_rmse_mean"),
            ("task_rmse_mean", "task_state", "rmse_mean"),
            ("task_rmse_std", "task_state", "rmse_std"),
        ]:
            add_metric_row(
                rows,
                "seed_stability",
                metric,
                row[col],
                model=f"world_model_v4_{variant}",
                split="test_seed_4",
                scope="torch_seed_11_42_73",
                target=target,
                source=display_path(path),
            )
    return pd.DataFrame(rows)


def collect_logged_action_ranking_rows():
    rows = []
    path = (
        ROOT
        / "reports"
        / "world_model_logged_action_ranking_proxy_v0"
        / "world_model_logged_action_ranking_proxy_v0_metrics.csv"
    )
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    main = df[df["utility"].eq("backlog_plus_throughput")].copy()
    for _, row in main.iterrows():
        for col, metric in [
            ("spearman", "spearman"),
            ("top5_hit_rate", "top5_hit_rate"),
            ("top10_hit_rate", "top10_hit_rate"),
            ("top20_hit_rate", "top20_hit_rate"),
            ("normalized_top1_regret", "normalized_top1_regret"),
            ("top10_normalized_best_regret", "top10_normalized_best_regret"),
        ]:
            if col in row and pd.notna(row[col]):
                add_metric_row(
                    rows,
                    "decision_ranking_proxy",
                    metric,
                    row[col],
                    model=row["model"],
                    split="test_seed_4",
                    scope="logged_action_windows",
                    target="backlog_plus_throughput_utility",
                    source=display_path(path),
                    note="CPU-side proxy over held-out logged AirFogSim action windows; not full counterfactual action injection.",
                )
    return pd.DataFrame(rows)


def collect_counterfactual_smoke_rows():
    rows = []
    path = (
        ROOT
        / "reports"
        / "airfogsim_counterfactual_action_smoke_v0"
        / "airfogsim_counterfactual_action_smoke_v0_summary.json"
    )
    if not path.exists():
        return pd.DataFrame(rows)
    summary = json.loads(path.read_text(encoding="utf-8"))
    ranking = summary.get("ranking", {})
    for metric in [
        "num_candidates",
        "spearman",
        "top2_hit_rate",
        "top1_regret",
        "normalized_top1_regret",
        "top2_best_regret",
        "top2_normalized_best_regret",
    ]:
        if metric in ranking and ranking[metric] is not None:
            add_metric_row(
                rows,
                "counterfactual_action_smoke",
                metric,
                ranking[metric],
                model="airfogsim_rb_counterfactual_smoke",
                split=f"seed_{summary.get('seed', '')}",
                scope=f"decision_time_{summary.get('decision_time', '')}_horizon_{summary.get('horizon', '')}",
                target="action_utility_ranking",
                source=display_path(path),
                note="Small AirFogSim counterfactual action-injection smoke test; world_model_utility is a lightweight proxy.",
            )
    return pd.DataFrame(rows)


def collect_v5_gpu_smoke_rows():
    rows = []
    path = (
        ROOT
        / "reports"
        / "world_model_v5_utility_ranking_smoke"
        / "world_model_v5_utility_ranking_smoke_summary.json"
    )
    if not path.exists():
        return pd.DataFrame(rows)
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    for metric in [
        "spearman",
        "top1_hit",
        "top1_regret",
        "normalized_top1_regret",
        "utility_rmse",
    ]:
        if metric in metrics and metrics[metric] is not None:
            add_metric_row(
                rows,
                "v5_gpu_utility_ranking_smoke",
                metric,
                metrics[metric],
                model="world_model_v5_utility_head_smoke",
                split="seed_4_counterfactual_candidates",
                scope=f"{summary.get('num_candidates', '')}_candidates_{summary.get('epochs', '')}_epochs",
                target="action_utility_ranking",
                source=display_path(path),
                note=f"GPU smoke on {summary.get('gpu_name', '')}; tiny counterfactual candidate set only.",
            )
    add_metric_row(
        rows,
        "v5_gpu_utility_ranking_smoke",
        "cuda_available",
        1.0 if summary.get("cuda_available") else 0.0,
        model="world_model_v5_utility_head_smoke",
        split="server",
        scope=summary.get("gpu_name", ""),
        target="gpu_environment",
        source=display_path(path),
    )
    return pd.DataFrame(rows)


def collect_v5_gpu_batch_rows():
    rows = []
    path = (
        ROOT
        / "reports"
        / "world_model_v5_utility_ranking_batch_v0"
        / "world_model_v5_utility_ranking_smoke_summary.json"
    )
    if not path.exists():
        return pd.DataFrame(rows)
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    for metric in [
        "spearman",
        "top1_hit",
        "top1_regret",
        "normalized_top1_regret",
        "utility_rmse",
        "group_top1_hit_mean",
        "group_normalized_top1_regret_mean",
        "group_spearman_mean",
    ]:
        if metric in metrics and metrics[metric] is not None:
            add_metric_row(
                rows,
                "v5_gpu_utility_ranking_batch",
                metric,
                metrics[metric],
                model="world_model_v5_utility_head",
                split="held_out_decision_groups",
                scope=(
                    f"{summary.get('num_candidates', '')}_candidates_"
                    f"{summary.get('num_decision_groups', '')}_groups_"
                    f"{summary.get('epochs', '')}_epochs"
                ),
                target="action_utility_ranking",
                source=display_path(path),
                note="GPU run on expanded AirFogSim counterfactual labels; features exclude outcome leakage.",
            )
    for group_summary in summary.get("group_metrics", []):
        split = group_summary.get("split", "")
        for metric in [
            "top1_hit_mean",
            "normalized_top1_regret_mean",
            "spearman_mean",
            "utility_rmse_mean",
        ]:
            if metric in group_summary and group_summary[metric] is not None:
                add_metric_row(
                    rows,
                    "v5_gpu_utility_ranking_batch_grouped",
                    metric,
                    group_summary[metric],
                    model="world_model_v5_utility_head",
                    split=split,
                    scope=f"{group_summary.get('num_groups', '')}_decision_groups",
                    target="per_decision_action_choice",
                    source=display_path(path),
                    note="Per-decision-group ranking metric; this matches the candidate-action selection use case.",
                )
    add_metric_row(
        rows,
        "v5_gpu_utility_ranking_batch",
        "cuda_available",
        1.0 if summary.get("cuda_available") else 0.0,
        model="world_model_v5_utility_head",
        split="server",
        scope=summary.get("gpu_name", ""),
        target="gpu_environment",
        source=display_path(path),
    )
    return pd.DataFrame(rows)


def collect_v5_resource_aware_rows():
    rows = []
    experiments = [
        (
            "world_model_v5_resource_aware_rank_only_action_v1",
            "world_model_v5_resource_rank_only_action",
        ),
        (
            "world_model_v5_resource_aware_rank_only_state_action_v1",
            "world_model_v5_resource_rank_only_state_action",
        ),
    ]
    for report_dir, model_name in experiments:
        path = ROOT / "reports" / report_dir / "world_model_v5_utility_ranking_smoke_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        for metric in [
            "group_top1_hit_mean",
            "group_normalized_top1_regret_mean",
            "group_spearman_mean",
            "utility_rmse",
        ]:
            if metric in metrics and metrics[metric] is not None:
                add_metric_row(
                    rows,
                    "v5_resource_aware_ranking",
                    metric,
                    metrics[metric],
                    model=model_name,
                    split="held_out_decision_groups",
                    scope=(
                        f"{summary.get('feature_mode', '')}_"
                        f"{summary.get('pair_scope', '')}_"
                        f"rb_penalty_{summary.get('rb_penalty', '')}_"
                        f"reg_{summary.get('reg_weight', '')}_rank_{summary.get('rank_weight', '')}"
                    ),
                    target="resource_aware_action_choice",
                    source=display_path(path),
                    note="Resource-aware utility uses AirFogSim utility minus RB consumption penalty; outcome-leakage features are excluded.",
                )
    sweep_path = ROOT / "reports" / "world_model_v5_multifamily_resource_aware_sweep_gpu.csv"
    sweep = read_csv_if_exists(sweep_path)
    if not sweep.empty:
        ok = sweep[sweep["status"].eq("ok")].copy()
        for _, row in ok.iterrows():
            model_name = f"world_model_v5_multifamily_{row['mode']}"
            scope = f"epochs_{int(row['epochs'])}_hidden_{int(row['hidden'])}_lr_{row['lr']}"
            for metric_col, metric_name in [
                ("top1", "group_top1_hit_mean"),
                ("regret", "group_normalized_top1_regret_mean"),
                ("spearman", "group_spearman_mean"),
            ]:
                add_metric_row(
                    rows,
                    "v5_resource_aware_multifamily_sweep",
                    metric_name,
                    row[metric_col],
                    model=model_name,
                    split="held_out_decision_groups",
                    scope=scope,
                    target="resource_aware_multi_family_action_choice",
                    source=display_path(sweep_path),
                    note="GPU sweep on multi-family counterfactual candidates; reports action-only and state-action hyperparameter variants.",
                )
    extended_sweep_path = ROOT / "reports" / "world_model_v5_extended_resource_aware_sweep_gpu.csv"
    extended_sweep = read_csv_if_exists(extended_sweep_path)
    if not extended_sweep.empty:
        ok = extended_sweep[extended_sweep["status"].eq("ok")].copy()
        for _, row in ok.iterrows():
            scope = f"epochs_{int(row['epochs'])}_hidden_{int(row['hidden'])}_lr_{row['lr']}"
            for metric_col, metric_name in [
                ("top1", "group_top1_hit_mean"),
                ("regret", "group_normalized_top1_regret_mean"),
                ("spearman", "group_spearman_mean"),
            ]:
                add_metric_row(
                    rows,
                    "v5_resource_aware_extended_sweep",
                    metric_name,
                    row[metric_col],
                    model="world_model_v5_extended_action",
                    split="held_out_decision_groups",
                    scope=scope,
                    target="resource_aware_extended_action_choice",
                    source=display_path(extended_sweep_path),
                    note="GPU sweep on extended counterfactual candidates covering RB, offload, mixed, CPU-scale, and return-route action families.",
                )
    offload_sweep_path = (
        ROOT
        / "reports"
        / "world_model_v5_offload_scaled_action_sweep_v0"
        / "world_model_v5_offload_scaled_action_sweep_v0.csv"
    )
    offload_sweep = read_csv_if_exists(offload_sweep_path)
    if not offload_sweep.empty:
        ok = offload_sweep[offload_sweep["status"].eq("ok")].copy()
        for _, row in ok.iterrows():
            scope = f"epochs_{int(row['epochs'])}_hidden_{int(row['hidden'])}_lr_{row['lr']}"
            for metric_col, metric_name in [
                ("top1", "group_top1_hit_mean"),
                ("regret", "group_normalized_top1_regret_mean"),
                ("spearman", "group_spearman_mean"),
                ("rmse", "utility_rmse"),
            ]:
                add_metric_row(
                    rows,
                    "v5_resource_aware_offload_scaled_sweep",
                    metric_name,
                    row[metric_col],
                    model="world_model_v5_offload_scaled_action",
                    split="held_out_decision_groups",
                    scope=scope,
                    target="resource_aware_offload_rb_action_choice",
                    source=display_path(offload_sweep_path),
                    note="CPU sweep on scaled offload/RB-only counterfactual candidates; this stage was selected because all groups have non-trivial utility spread.",
                )
    seedheldout_path = (
        ROOT
        / "reports"
        / "world_model_v5_offload_scaled_seedheldout_gpu_current"
        / "seedheldout_gpu_current_summary.csv"
    )
    seedheldout = read_csv_if_exists(seedheldout_path)
    if not seedheldout.empty:
        for _, row in seedheldout.iterrows():
            model = row["kind"]
            scope = f"heldout_seeds_{row['heldout']}"
            for metric_col, metric_name in [
                ("top1", "group_top1_hit_mean"),
                ("regret", "group_normalized_top1_regret_mean"),
                ("spearman", "group_spearman_mean"),
            ]:
                if metric_col in row and pd.notna(row[metric_col]):
                    add_metric_row(
                        rows,
                        "v5_offload_scaled_seedheldout_stability",
                        metric_name,
                        row[metric_col],
                        model=model,
                        split="seed_heldout",
                        scope=scope,
                        target="resource_aware_offload_rb_action_choice",
                        source=display_path(seedheldout_path),
                        note="Five held-out seed-pair stability check on the scaled offload/RB counterfactual dataset.",
                    )
    dual_sweep_path = (
        ROOT
        / "reports"
        / "world_model_v5_dual_graph_compact_mlp_sweep_seed0_9_local"
        / "dual_graph_compact_mlp_sweep_seed0_9_summary.csv"
    )
    dual_sweep = read_csv_if_exists(dual_sweep_path)
    if not dual_sweep.empty:
        for _, row in dual_sweep.iterrows():
            scope = f"epochs_{int(row['epochs'])}_hidden_{int(row['hidden'])}_lr_{row['lr']}"
            for metric_col, metric_name in [
                ("top1_mean", "seedheldout_top1_mean"),
                ("regret_mean", "seedheldout_normalized_regret_mean"),
                ("spearman_mean", "seedheldout_spearman_mean"),
            ]:
                add_metric_row(
                    rows,
                    "v5_dual_graph_compact_mlp_seedheldout",
                    metric_name,
                    row[metric_col],
                    model="world_model_v5_dual_graph_compact_mlp",
                    split="five_seedheldout_pairs",
                    scope=scope,
                    target="resource_aware_offload_rb_action_choice",
                    source=display_path(dual_sweep_path),
                    note="Dual-graph compact state-action interaction MLP sweep using the new seed0-9 world-model state/action dataset.",
                )
    return pd.DataFrame(rows)


def collect_v5_decision_baseline_rows():
    rows = []
    paths = [
        ROOT / "reports" / "world_model_v5_decision_baselines_v0" / "world_model_v5_decision_baselines_v0_metrics.csv",
        ROOT
        / "reports"
        / "world_model_v5_decision_baselines_multifamily_v0"
        / "world_model_v5_decision_baselines_v0_metrics.csv",
        ROOT
        / "reports"
        / "world_model_v5_decision_baselines_extended_v0"
        / "world_model_v5_decision_baselines_v0_metrics.csv",
        ROOT
        / "reports"
        / "world_model_v5_decision_baselines_offload_scaled_v0"
        / "world_model_v5_decision_baselines_v0_metrics.csv",
    ]
    for path in paths:
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        test = df[df["split"].eq("test")].copy()
        if "offload_scaled" in str(path):
            category = "v5_decision_baseline_offload_scaled"
            target = "offload_rb_action_choice_baseline"
        elif "extended" in str(path):
            category = "v5_decision_baseline_extended"
            target = "extended_action_choice_baseline"
        elif "multifamily" in str(path):
            category = "v5_decision_baseline_multifamily"
            target = "multi_family_action_choice_baseline"
        else:
            category = "v5_decision_baseline"
            target = "action_choice_baseline"
        for _, row in test.iterrows():
            for metric in [
                "top1_hit_mean",
                "normalized_top1_regret_mean",
                "spearman_mean",
                "utility_rmse_mean",
            ]:
                if metric in row and pd.notna(row[metric]):
                    add_metric_row(
                        rows,
                        category,
                        metric,
                        row[metric],
                        model=row["baseline"],
                        split="held_out_decision_groups",
                        scope=row["utility"],
                        target=target,
                        source=display_path(path),
                        note="Heuristic baseline for v5 counterfactual decision ranking.",
                    )
    return pd.DataFrame(rows)


def collect_v5_conservative_selector_rows():
    rows = []
    sources = [
        (
            ROOT
            / "reports"
            / "world_model_v5_hybrid_selector_v4_max_total_rb"
            / "world_model_v5_hybrid_selector_v4_max_total_rb_aggregate.csv",
            "hybrid_top1_first",
            "Train-selected conservative selector using max-total-RB fallback and top-1-first threshold selection.",
        ),
        (
            ROOT
            / "reports"
            / "world_model_v5_hybrid_selector_v4_max_total_rb_regret_first"
            / "world_model_v5_hybrid_selector_v4_max_total_rb_regret_first_aggregate.csv",
            "hybrid_regret_first",
            "Train-selected conservative selector using max-total-RB fallback and regret-first threshold selection.",
        ),
        (
            ROOT
            / "reports"
            / "world_model_v5_takeover_calibrator_v4_max_total_rb"
            / "takeover_v4_max_total_rb_seedheldout_aggregate.csv",
            "learned_takeover",
            "Train-only group-level takeover calibrator using observable margin, RB, and action-family features.",
        ),
    ]
    for path, model_name, note in sources:
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        row = df.iloc[0]
        for metric_col, metric_name in [
            ("mean_top1", "seedheldout_top1_mean"),
            ("mean_regret", "seedheldout_normalized_regret_mean"),
            ("mean_spearman", "seedheldout_spearman_mean"),
        ]:
            if metric_col in row and pd.notna(row[metric_col]):
                add_metric_row(
                    rows,
                    "v5_conservative_selector_v4",
                    metric_name,
                    row[metric_col],
                    model=model_name,
                    split="five_seedheldout_pairs",
                    scope="offload_scaled_v4",
                    target="airfogsim_counterfactual_action_choice",
                    source=display_path(path),
                    note=note,
                )
    return pd.DataFrame(rows)


def collect_v5_selector_probe_rows():
    rows = []
    path = (
        ROOT
        / "reports"
        / "world_model_v5_selector_probe_v0"
        / "world_model_v5_selector_probe_v0_aggregate.csv"
    )
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame(rows)
    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        method = str(row["method"])
        note = (
            "CPU-only selector probe over existing seed-heldout v5 predictions. "
            "The strongest current row is best_reg0p05/global_top1; family-aware rows are retained as a negative result."
        )
        for metric_col, metric_name in [
            ("mean_top1", "seedheldout_top1_mean"),
            ("mean_regret", "seedheldout_normalized_regret_mean"),
            ("mean_spearman", "seedheldout_spearman_mean"),
            ("mean_take_rate", "selector_take_rate_mean"),
        ]:
            if metric_col in row and pd.notna(row[metric_col]):
                add_metric_row(
                    rows,
                    "v5_selector_probe_v0",
                    metric_name,
                    row[metric_col],
                    model=method,
                    split="five_seedheldout_pairs",
                    scope=dataset,
                    target="airfogsim_counterfactual_action_choice",
                    source=display_path(path),
                    note=note,
                )
    return pd.DataFrame(rows)


def collect_decision_gap_rows():
    rows = []
    gaps = [
        (
            "counterfactual_candidate_action_ranking",
            "Counterfactual ranking now exists and the scaled offload/RB action scorer beats simple resource-aware heuristics on held-out groups; remaining evidence should scale this beyond the offload/RB stage and make state-rich dual-graph representations consistently useful.",
        ),
    ]
    for scope, note in gaps:
        rows.append(
            {
                "category": "decision_value_gap",
                "target": "decision_interface",
                "metric": "missing_required_evidence",
                "model": "current_pipeline",
                "split": "",
                "scope": scope,
                "value": "",
                "source": "metric_suite_requirement",
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def plot_metric_suite(metric_df):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "world_model_metric_suite_v0_overview.png"
    focus = metric_df[
        (
            (metric_df["category"].eq("seed_stability") & metric_df["metric"].isin(["f1_mean", "rmse_mean"]))
            | (metric_df["category"].eq("activity_gating") & metric_df["metric"].eq("f1_mean"))
            | (metric_df["category"].eq("active_rate") & metric_df["metric"].eq("active_rmse"))
            | (metric_df["category"].eq("decision_ranking_proxy") & metric_df["metric"].isin(["top10_hit_rate", "spearman"]))
            | (metric_df["category"].eq("counterfactual_action_smoke") & metric_df["metric"].isin(["top2_hit_rate", "spearman"]))
            | (metric_df["category"].eq("v5_gpu_utility_ranking_smoke") & metric_df["metric"].isin(["top1_hit", "spearman", "utility_rmse"]))
            | (metric_df["category"].eq("v5_gpu_utility_ranking_batch") & metric_df["metric"].isin(["group_top1_hit_mean", "group_spearman_mean", "utility_rmse"]))
            | (metric_df["category"].eq("v5_resource_aware_ranking") & metric_df["metric"].isin(["group_top1_hit_mean", "group_spearman_mean"]))
            | (metric_df["category"].eq("v5_resource_aware_multifamily_sweep") & metric_df["metric"].isin(["group_top1_hit_mean", "group_spearman_mean"]))
            | (metric_df["category"].eq("v5_resource_aware_extended_sweep") & metric_df["metric"].isin(["group_top1_hit_mean", "group_spearman_mean"]))
            | (metric_df["category"].eq("v5_resource_aware_offload_scaled_sweep") & metric_df["metric"].isin(["group_top1_hit_mean", "group_spearman_mean"]))
            | (metric_df["category"].eq("v5_decision_baseline") & metric_df["metric"].isin(["top1_hit_mean", "spearman_mean"]))
            | (metric_df["category"].eq("v5_decision_baseline_multifamily") & metric_df["metric"].isin(["top1_hit_mean", "spearman_mean"]))
            | (metric_df["category"].eq("v5_decision_baseline_extended") & metric_df["metric"].isin(["top1_hit_mean", "spearman_mean"]))
            | (metric_df["category"].eq("v5_decision_baseline_offload_scaled") & metric_df["metric"].isin(["top1_hit_mean", "spearman_mean"]))
        )
    ].copy()
    focus = focus[pd.to_numeric(focus["value"], errors="coerce").notna()]
    focus["value"] = pd.to_numeric(focus["value"], errors="coerce")
    focus["label"] = focus["category"] + "\n" + focus["model"].str.replace("world_model_", "", regex=False) + "\n" + focus["scope"].astype(str)
    focus = focus.head(18)
    plt.figure(figsize=(12.5, 5.2))
    colors = focus["category"].map(
        {
            "seed_stability": "#2563eb",
            "activity_gating": "#d97706",
            "active_rate": "#dc2626",
            "decision_ranking_proxy": "#7c3aed",
            "counterfactual_action_smoke": "#059669",
            "v5_gpu_utility_ranking_smoke": "#0891b2",
            "v5_gpu_utility_ranking_batch": "#0f766e",
            "v5_resource_aware_ranking": "#16a34a",
            "v5_resource_aware_multifamily_sweep": "#22c55e",
            "v5_resource_aware_extended_sweep": "#84cc16",
            "v5_resource_aware_offload_scaled_sweep": "#15803d",
            "v5_decision_baseline": "#64748b",
            "v5_decision_baseline_multifamily": "#475569",
            "v5_decision_baseline_extended": "#334155",
            "v5_decision_baseline_offload_scaled": "#1f2937",
        }
    ).fillna("#6b7280")
    plt.bar(np.arange(len(focus)), focus["value"], color=colors)
    plt.xticks(np.arange(len(focus)), focus["label"], rotation=35, ha="right", fontsize=7)
    plt.ylabel("metric value")
    plt.title("World-model metric suite v0 overview")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(summary, metric_df):
    def best_row(category, metric, lower_is_better=False):
        part = metric_df[(metric_df["category"] == category) & (metric_df["metric"] == metric)].copy()
        part["num"] = pd.to_numeric(part["value"], errors="coerce")
        part = part.dropna(subset=["num"])
        if part.empty:
            return None
        return part.sort_values("num", ascending=lower_is_better).iloc[0]

    def best_row_scoped(category, metric, scope, lower_is_better=False):
        part = metric_df[
            (metric_df["category"] == category)
            & (metric_df["metric"] == metric)
            & (metric_df["scope"] == scope)
        ].copy()
        part["num"] = pd.to_numeric(part["value"], errors="coerce")
        part = part.dropna(subset=["num"])
        if part.empty:
            return None
        return part.sort_values("num", ascending=lower_is_better).iloc[0]

    best_activity = best_row("activity_gating", "f1_mean")
    best_rate = best_row("active_rate", "active_rmse", lower_is_better=True)
    best_physical = best_row("physical_graph_rollout", "rmse", lower_is_better=True)
    best_ranking = best_row("decision_ranking_proxy", "top10_hit_rate")
    counterfactual_smoke = best_row("counterfactual_action_smoke", "top2_hit_rate")
    v5_batch_group_top1 = best_row("v5_gpu_utility_ranking_batch", "group_top1_hit_mean")
    v5_batch_group_spearman = best_row("v5_gpu_utility_ranking_batch", "group_spearman_mean")
    v5_resource_top1 = best_row("v5_resource_aware_ranking", "group_top1_hit_mean")
    v5_resource_spearman = best_row("v5_resource_aware_ranking", "group_spearman_mean")
    v5_multifamily_top1 = best_row("v5_resource_aware_multifamily_sweep", "group_top1_hit_mean")
    v5_multifamily_regret = best_row(
        "v5_resource_aware_multifamily_sweep",
        "group_normalized_top1_regret_mean",
        lower_is_better=True,
    )
    v5_extended_top1 = best_row("v5_resource_aware_extended_sweep", "group_top1_hit_mean")
    v5_extended_regret = best_row(
        "v5_resource_aware_extended_sweep",
        "group_normalized_top1_regret_mean",
        lower_is_better=True,
    )
    v5_offload_scaled_top1 = best_row("v5_resource_aware_offload_scaled_sweep", "group_top1_hit_mean")
    v5_offload_scaled_regret = best_row(
        "v5_resource_aware_offload_scaled_sweep",
        "group_normalized_top1_regret_mean",
        lower_is_better=True,
    )
    seedheldout = metric_df[metric_df["category"] == "v5_offload_scaled_seedheldout_stability"].copy()
    dual_graph_seedheldout = metric_df[metric_df["category"] == "v5_dual_graph_compact_mlp_seedheldout"].copy()
    multifamily_baseline_top1 = best_row("v5_decision_baseline_multifamily", "top1_hit_mean")
    multifamily_baseline_regret = best_row(
        "v5_decision_baseline_multifamily",
        "normalized_top1_regret_mean",
        lower_is_better=True,
    )
    extended_baseline_top1 = best_row_scoped("v5_decision_baseline_extended", "top1_hit_mean", "resource_aware_utility")
    extended_baseline_regret = best_row_scoped(
        "v5_decision_baseline_extended",
        "normalized_top1_regret_mean",
        "resource_aware_utility",
        lower_is_better=True,
    )
    offload_scaled_baseline_top1 = best_row_scoped(
        "v5_decision_baseline_offload_scaled",
        "top1_hit_mean",
        "resource_aware_utility",
    )
    offload_scaled_baseline_regret = best_row_scoped(
        "v5_decision_baseline_offload_scaled",
        "normalized_top1_regret_mean",
        "resource_aware_utility",
        lower_is_better=True,
    )
    runtime = metric_df[
        (metric_df["category"] == "runtime_comparison")
        & (metric_df["metric"] == "speedup")
    ].copy()
    v4_task = metric_df[
        (metric_df["category"] == "seed_stability")
        & (metric_df["target"] == "task_state")
        & (metric_df["metric"] == "rmse_mean")
    ].copy()
    gaps = metric_df[metric_df["category"] == "decision_value_gap"].copy()
    lines = [
        "# World-model metric suite v0",
        "",
        "## Goal",
        "",
        "This report collects existing CPU-side evidence into one evaluation view. It avoids retraining and organizes metrics by prediction quality, activity gating, probability calibration, active-rate regression, interval reliability, robustness, runtime, physical rollout, seed stability, and decision-interface gaps.",
        "",
        "## Key Readout",
        "",
    ]
    if best_activity is not None:
        lines.append(
            f"- Best current v4 activity-gating mean F1 entry: `{best_activity['scope']}` with value `{float(best_activity['value']):.6f}`."
        )
    if best_rate is not None:
        lines.append(
            f"- Best active-rate RMSE entry: `{best_rate['model']}` under `{best_rate['scope']}` with value `{float(best_rate['value']):.3f}`."
        )
    if not runtime.empty:
        lines.append(
            f"- CPU runtime comparison now estimates `{float(runtime.iloc[0]['value']):.2f}x` speedup for v4 forward inference versus a 3-step AirFogSim rollout."
        )
    if best_physical is not None:
        lines.append(
            f"- Best physical-rollout RMSE entry: `{best_physical['model']}` on `{best_physical['target']}` with value `{float(best_physical['value']):.6f}`."
        )
    if best_ranking is not None:
        lines.append(
            f"- Logged-action ranking proxy top-10 hit rate is currently `{float(best_ranking['value']):.6f}` for `{best_ranking['model']}`, so decision-value ordering is still a major gap."
        )
    if counterfactual_smoke is not None:
        lines.append(
            f"- Small AirFogSim counterfactual smoke test has `{int(float(metric_df[(metric_df['category']=='counterfactual_action_smoke') & (metric_df['metric']=='num_candidates')].iloc[0]['value']))}` candidates and top-2 hit rate `{float(counterfactual_smoke['value']):.6f}`; this validates the action-injection pipeline only."
        )
    if v5_batch_group_top1 is not None:
        lines.append(
            f"- v5 batch utility-ranking on held-out decision groups reaches group top-1 hit `{float(v5_batch_group_top1['value']):.6f}`"
            f" and group Spearman `{float(v5_batch_group_spearman['value']):.6f}` without outcome-leakage features."
        )
    if v5_resource_top1 is not None:
        lines.append(
            f"- Resource-aware v5 ranking reaches group top-1 hit `{float(v5_resource_top1['value']):.6f}`"
            f" and group Spearman `{float(v5_resource_spearman['value']):.6f}`; heuristic baselines are also reported to avoid overstating this result."
        )
    if v5_multifamily_top1 is not None:
        baseline_text = ""
        if multifamily_baseline_top1 is not None:
            baseline_text = (
                f" The strongest multi-family baseline top-1 is `{float(multifamily_baseline_top1['value']):.6f}`"
                f" and its best regret is `{float(multifamily_baseline_regret['value']):.6f}`."
            )
        lines.append(
            f"- Multi-family resource-aware GPU sweep reaches group top-1 hit `{float(v5_multifamily_top1['value']):.6f}`"
            f" with best normalized regret `{float(v5_multifamily_regret['value']):.6f}`.{baseline_text}"
        )
    if v5_extended_top1 is not None:
        baseline_text = ""
        if extended_baseline_top1 is not None:
            baseline_text = (
                f" Extended baselines reach top-1 `{float(extended_baseline_top1['value']):.6f}`"
                f" and best regret `{float(extended_baseline_regret['value']):.6f}`."
            )
        lines.append(
            f"- Extended five-family sweep reaches group top-1 hit `{float(v5_extended_top1['value']):.6f}`"
            f" and best normalized regret `{float(v5_extended_regret['value']):.6f}`.{baseline_text}"
        )
    if v5_offload_scaled_top1 is not None:
        baseline_text = ""
        if offload_scaled_baseline_top1 is not None:
            baseline_text = (
                f" Strongest offload/RB baseline top-1 is `{float(offload_scaled_baseline_top1['value']):.6f}`"
                f" and best baseline regret is `{float(offload_scaled_baseline_regret['value']):.6f}`."
            )
        lines.append(
            f"- Scaled offload/RB v5 action sweep reaches group top-1 hit `{float(v5_offload_scaled_top1['value']):.6f}`"
            f" and best normalized regret `{float(v5_offload_scaled_regret['value']):.6f}`.{baseline_text}"
        )
    if not seedheldout.empty:
        top1 = seedheldout[seedheldout["metric"].eq("group_top1_hit_mean")].copy()
        regret = seedheldout[seedheldout["metric"].eq("group_normalized_top1_regret_mean")].copy()
        top1_mean = top1.groupby("model")["value"].mean().to_dict()
        regret_mean = regret.groupby("model")["value"].mean().to_dict()
        lines.append(
            "- Five seed-heldout pairs show the single-split offload/RB result is not yet stable: "
            f"action MLP mean top-1 `{float(top1_mean.get('action_mlp', float('nan'))):.6f}`, "
            f"state-action MLP `{float(top1_mean.get('state_action_mlp', float('nan'))):.6f}`, "
            f"ridge `{float(top1_mean.get('ridge', float('nan'))):.6f}`, "
            f"best heuristic top-1 `{float(top1_mean.get('baseline_best_top1', float('nan'))):.6f}`; "
            f"mean regret is action `{float(regret_mean.get('action_mlp', float('nan'))):.6f}`, "
            f"state-action `{float(regret_mean.get('state_action_mlp', float('nan'))):.6f}`, "
            f"and best-regret heuristic `{float(regret_mean.get('baseline_best_regret', float('nan'))):.6f}`."
        )
    if not dual_graph_seedheldout.empty:
        top1 = dual_graph_seedheldout[dual_graph_seedheldout["metric"].eq("seedheldout_top1_mean")].copy()
        regret = dual_graph_seedheldout[dual_graph_seedheldout["metric"].eq("seedheldout_normalized_regret_mean")].copy()
        best_top1 = top1.sort_values("value", ascending=False).iloc[0]
        best_regret = regret.sort_values("value", ascending=True).iloc[0]
        lines.append(
            "- Dual-graph compact state-action MLP improves the learned seed-heldout top-1 envelope to "
            f"`{float(best_top1['value']):.6f}` under `{best_top1['scope']}`, with best mean regret "
            f"`{float(best_regret['value']):.6f}`; this is better than the previous learned heads but still below the strongest heuristic baseline."
        )
    if not v4_task.empty:
        lines.append("- v4 task-side stability remains relevant because physical context improves task RMSE in several runs.")
    if gaps.empty:
        lines.append("- No decision-facing gap rows remain in the current metric suite.")
    else:
        gap_names = ", ".join(gaps["scope"].astype(str).tolist())
        lines.append(f"- Remaining decision-facing evidence gap: {gap_names}.")
    lines.extend(
        [
            "",
            "## Metric Categories",
            "",
            metric_df.groupby("category").size().reset_index(name="rows").to_markdown(index=False),
            "",
            "## Decision-facing Gaps",
            "",
            gaps[["scope", "note"]].to_markdown(index=False) if not gaps.empty else "No remaining gap rows.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_metric_suite_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collectors = [
        collect_prediction_rows,
        collect_activity_calibration_rows,
        collect_probability_rows,
        collect_active_rate_rows,
        collect_robustness_rows,
        collect_runtime_rows,
        collect_physical_rollout_rows,
        collect_seed_stability_rows,
        collect_logged_action_ranking_rows,
        collect_counterfactual_smoke_rows,
        collect_v5_gpu_smoke_rows,
        collect_v5_gpu_batch_rows,
        collect_v5_resource_aware_rows,
        collect_v5_decision_baseline_rows,
        collect_v5_conservative_selector_rows,
        collect_v5_selector_probe_rows,
        collect_decision_gap_rows,
    ]
    parts = [fn() for fn in collectors]
    metric_df = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    metric_df = metric_df[
        ["category", "target", "metric", "model", "split", "scope", "value", "source", "note"]
    ]
    metric_path = OUTPUT_DIR / "world_model_metric_suite_v0_metrics.csv"
    category_path = OUTPUT_DIR / "world_model_metric_suite_v0_category_counts.csv"
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    metric_df.groupby("category").size().reset_index(name="rows").to_csv(
        category_path, index=False, encoding="utf-8-sig"
    )
    plot_path = plot_metric_suite(metric_df)
    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "num_metric_rows": int(len(metric_df)),
        "categories": sorted(metric_df["category"].dropna().unique().tolist()),
        "outputs": {
            "metrics_csv": display_path(metric_path),
            "category_counts_csv": display_path(category_path),
            "overview_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metric_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_metric_suite_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
