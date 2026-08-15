import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_world_model_v0 import FIGURE_DIR, ROOT, load_dataset, split_by_seed
from run_world_model_v4_dual_graph_rollout import display_path


OUTPUT_DIR = ROOT / "reports" / "world_model_logged_action_ranking_proxy_v0"
CACHE_DIR = ROOT / "reports" / "world_model_v4_activity_calibration" / "prediction_cache"


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_rank_correlation(true_utility, predicted_utility):
    true_rank = average_ranks(true_utility)
    pred_rank = average_ranks(predicted_utility)
    if true_rank.size != pred_rank.size:
        raise ValueError("true_utility and predicted_utility must have the same size")
    if true_rank.size < 2 or np.std(true_rank) < 1e-12 or np.std(pred_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def topk_indices(values, k):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), values.size)
    return set(np.argsort(values)[-k:])


def topk_hit_rate(true_utility, predicted_utility, k):
    true_top = topk_indices(true_utility, k)
    pred_top = topk_indices(predicted_utility, k)
    if not true_top:
        return float("nan")
    return float(len(true_top & pred_top) / len(true_top))


def ranking_regret(true_utility, predicted_utility, top_k=1):
    true = np.asarray(true_utility, dtype=np.float64).reshape(-1)
    pred = np.asarray(predicted_utility, dtype=np.float64).reshape(-1)
    if true.size != pred.size:
        raise ValueError("true_utility and predicted_utility must have the same size")
    if true.size == 0:
        raise ValueError("utility arrays must be non-empty")
    pred_top1 = int(np.argmax(pred))
    true_best = float(np.max(true))
    top1_regret = true_best - float(true[pred_top1])
    pred_topk = list(topk_indices(pred, top_k))
    topk_best = float(np.max(true[pred_topk]))
    topk_best_regret = true_best - topk_best
    denom = float(np.max(true) - np.min(true))
    if denom < 1e-12:
        normalized_top1 = 0.0
        normalized_topk = 0.0
    else:
        normalized_top1 = top1_regret / denom
        normalized_topk = topk_best_regret / denom
    return {
        "top1_regret": float(top1_regret),
        "normalized_top1_regret": float(normalized_top1),
        "topk_best_regret": float(topk_best_regret),
        "normalized_topk_best_regret": float(normalized_topk),
    }


def task_feature_index(task_features, name):
    matches = np.where(np.asarray(task_features).astype(str) == name)[0]
    if len(matches) != 1:
        raise KeyError(f"task feature not found: {name}")
    return int(matches[0])


def make_true_utilities(arrays, test_idx):
    task_features = arrays["task_features"].astype(str)
    finished_idx = task_feature_index(task_features, "num_finished")
    offload_idx = task_feature_index(task_features, "num_to_offload")
    computing_idx = task_feature_index(task_features, "num_computing")

    x_last = arrays["x_task"][test_idx, -1, :]
    y_last = arrays["y_task"][test_idx, -1, :]
    delta_finished = y_last[:, finished_idx] - x_last[:, finished_idx]
    backlog_adjusted = delta_finished - 0.1 * y_last[:, offload_idx] - 0.05 * y_last[:, computing_idx]
    throughput = np.clip(arrays["y_link_rate"][test_idx], 0.0, None).sum(axis=(1, 2))
    throughput_scaled = np.log1p(throughput)
    return {
        "delta_finished": delta_finished.astype(np.float64),
        "backlog_adjusted": backlog_adjusted.astype(np.float64),
        "backlog_plus_throughput": (backlog_adjusted + 0.01 * throughput_scaled).astype(np.float64),
    }


def make_predicted_utilities(arrays, test_idx, pred):
    task_features = arrays["task_features"].astype(str)
    finished_idx = task_feature_index(task_features, "num_finished")
    offload_idx = task_feature_index(task_features, "num_to_offload")
    computing_idx = task_feature_index(task_features, "num_computing")

    x_last = arrays["x_task"][test_idx, -1, :]
    pred_last = pred["task_pred"][:, -1, :]
    delta_finished = pred_last[:, finished_idx] - x_last[:, finished_idx]
    backlog_adjusted = delta_finished - 0.1 * pred_last[:, offload_idx] - 0.05 * pred_last[:, computing_idx]
    expected_rate = np.clip(pred["rate_pred"], 0.0, None) * np.clip(pred["active_prob"], 0.0, 1.0)
    throughput_scaled = np.log1p(expected_rate.sum(axis=(1, 2)))
    return {
        "delta_finished": delta_finished.astype(np.float64),
        "backlog_adjusted": backlog_adjusted.astype(np.float64),
        "backlog_plus_throughput": (backlog_adjusted + 0.01 * throughput_scaled).astype(np.float64),
    }


def load_prediction_cache(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "active_prob": data["test_active_prob"],
            "rate_pred": data["test_rate_pred"],
            "task_pred": data["test_task_pred"],
        }


def ensemble_predictions(predictions):
    keys = ["active_prob", "rate_pred", "task_pred"]
    return {key: np.mean([pred[key] for pred in predictions], axis=0).astype(np.float32) for key in keys}


def evaluate_ranking(true_utilities, predicted_utilities, model, source):
    rows = []
    for utility_name, true_values in true_utilities.items():
        pred_values = predicted_utilities[utility_name]
        row = {
            "split": "test_seed_4",
            "model": model,
            "utility": utility_name,
            "num_logged_windows": int(len(true_values)),
            "spearman": spearman_rank_correlation(true_values, pred_values),
            "top1_true_utility": float(true_values[int(np.argmax(pred_values))]),
            "oracle_top1_true_utility": float(np.max(true_values)),
            "source": source,
        }
        for k in [5, 10, 20]:
            row[f"top{k}_hit_rate"] = topk_hit_rate(true_values, pred_values, k)
            regret = ranking_regret(true_values, pred_values, top_k=k)
            row[f"top{k}_best_regret"] = regret["topk_best_regret"]
            row[f"top{k}_normalized_best_regret"] = regret["normalized_topk_best_regret"]
        regret = ranking_regret(true_values, pred_values, top_k=1)
        row.update(regret)
        rows.append(row)
    return rows


def plot_ranking_summary(metrics_df):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "world_model_logged_action_ranking_proxy_v0.png"
    plot_df = metrics_df[metrics_df["utility"] == "backlog_plus_throughput"].copy()
    plot_df = plot_df.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, title in [
        (axes[0], "spearman", "Rank correlation"),
        (axes[1], "top10_hit_rate", "Top-10 hit rate"),
        (axes[2], "normalized_top1_regret", "Top-1 regret"),
    ]:
        plot_df[col].plot(kind="bar", ax=ax, color="#2563eb")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Logged-action ranking proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(summary, metrics_df):
    main_utility = "backlog_plus_throughput"
    main = metrics_df[metrics_df["utility"] == main_utility].copy()
    best = main.sort_values(["top10_hit_rate", "spearman"], ascending=False).iloc[0]
    lines = [
        "# World-model logged-action ranking proxy v0",
        "",
        "## Goal",
        "",
        "This CPU-only experiment adds a first decision-facing proxy. It treats held-out logged action windows from AirFogSim as candidate-like samples and compares the world model's predicted utility ranking with the utility ranking computed from AirFogSim outcomes.",
        "",
        "This is not a full counterfactual action-injection evaluation. It is a precursor that checks whether the current world model preserves useful ordering signal over real logged decisions.",
        "",
        "## Utility Definitions",
        "",
        "- `delta_finished`: future finished-task count minus the last observed finished-task count.",
        "- `backlog_adjusted`: `delta_finished - 0.1 * num_to_offload - 0.05 * num_computing` at the horizon end.",
        "- `backlog_plus_throughput`: `backlog_adjusted + 0.01 * log1p(sum expected link rate)`.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Key Readout",
        "",
        f"- Best `{main_utility}` top-10 hit rate model: `{best['model']}` with `{best['top10_hit_rate']:.6f}`.",
        f"- Its Spearman rank correlation is `{best['spearman']:.6f}`, and normalized top-1 regret is `{best['normalized_top1_regret']:.6f}`.",
        "- The next decision-facing step is full AirFogSim counterfactual candidate-action evaluation, where candidate actions are injected into the simulator and compared with world-model predictions.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "world_model_logged_action_ranking_proxy_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_dataset()
    _, _, test_idx = split_by_seed(arrays["sample_seed"])
    true_utilities = make_true_utilities(arrays, test_idx)

    rows = []
    loaded_predictions = []
    cache_paths = sorted(CACHE_DIR.glob("v4_activity_dual_full_seed_*_predictions.npz"))
    for cache_path in cache_paths:
        seed = cache_path.stem.split("_seed_")[-1].replace("_predictions", "")
        pred = load_prediction_cache(cache_path)
        loaded_predictions.append(pred)
        predicted_utilities = make_predicted_utilities(arrays, test_idx, pred)
        rows.extend(
            evaluate_ranking(
                true_utilities,
                predicted_utilities,
                model=f"world_model_v4_dual_full_seed_{seed}",
                source=display_path(cache_path),
            )
        )

    if loaded_predictions:
        pred = ensemble_predictions(loaded_predictions)
        predicted_utilities = make_predicted_utilities(arrays, test_idx, pred)
        rows.extend(
            evaluate_ranking(
                true_utilities,
                predicted_utilities,
                model="world_model_v4_dual_full_ensemble",
                source=display_path(CACHE_DIR),
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "world_model_logged_action_ranking_proxy_v0_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_path = plot_ranking_summary(metrics_df)
    summary = {
        "output_dir": display_path(OUTPUT_DIR),
        "num_test_windows": int(len(test_idx)),
        "num_prediction_sources": int(len(cache_paths)),
        "scope": "logged_action_ranking_proxy",
        "remaining_gap": "full_counterfactual_airfogsim_action_ranking",
        "outputs": {
            "metrics_csv": display_path(metrics_path),
            "summary_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, metrics_df)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = OUTPUT_DIR / "world_model_logged_action_ranking_proxy_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
