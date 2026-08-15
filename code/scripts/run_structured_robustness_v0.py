import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_structured_dual_branch_baseline_v0 import (
    ACTION_DIR,
    DATASET_DIR,
    FIGURE_DIR,
    LINK_TYPES,
    build_branch_features,
    build_targets_and_persistence,
    fit_standardizer,
    load_arrays,
    metrics,
    predict_model,
    split_by_seed,
    train_model,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "structured_robustness_v0"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 20260515


def noisy_arrays(arrays, level, seed):
    rng = np.random.default_rng(seed)
    out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arrays.items()}
    if level <= 0:
        return out

    node_scale = np.array([15.0, 15.0, 3.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32) * level
    noise = rng.normal(0.0, node_scale, size=out["x_node"].shape).astype(np.float32)
    out["x_node"] = out["x_node"].astype(np.float32) + noise

    link = out["x_link"].astype(np.float32)
    link[..., 0] = np.maximum(0.0, link[..., 0] + rng.normal(0.0, 20.0 * level, size=link[..., 0].shape))
    link[..., 1] = np.maximum(0.0, link[..., 1] * (1.0 + rng.normal(0.0, 0.50 * level, size=link[..., 1].shape)))
    link[..., 2] = np.maximum(0.0, link[..., 2] * (1.0 + rng.normal(0.0, 0.30 * level, size=link[..., 2].shape)))
    link[..., 3] = np.maximum(0.0, link[..., 3] + rng.normal(0.0, 1.0 * level, size=link[..., 3].shape))
    link[..., 4] = np.maximum(0.0, link[..., 4] + rng.normal(0.0, 1.0 * level, size=link[..., 4].shape))
    out["x_link"] = link.astype(np.float32)

    task = out["x_task"].astype(np.float32)
    scale = np.maximum(np.std(task, axis=(0, 1), keepdims=True), 1.0)
    task = task + rng.normal(0.0, level, size=task.shape).astype(np.float32) * scale
    out["x_task"] = np.maximum(0.0, task).astype(np.float32)
    return out


def plot_curves(df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, title in [
        (axes[0], "all_rmse", "All targets RMSE"),
        (axes[1], "link_rate_by_type_rmse", "Link rate RMSE"),
        (axes[2], "task_state_rmse", "Task state RMSE"),
    ]:
        for model in ["persistence", "structured_state", "structured_state_action"]:
            part = df[df["model"] == model]
            ax.plot(part["noise_level"], part[metric], marker="o", label=model)
        ax.set_xlabel("noise level")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Structured baseline robustness under synthetic input perturbations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metrics_df, summary):
    clean = metrics_df[metrics_df["noise_level"] == 0.0].set_index("model")
    noisy = metrics_df[metrics_df["noise_level"] == 0.3].set_index("model")
    lines = [
        "# Structured robustness report v0",
        "",
        "## Goal",
        "",
        "Evaluate whether the first structured state/action model remains stable when node, link, and task observations are perturbed at test time.",
        "",
        "## Setup",
        "",
        "- Training data: clean `dataset_multiseed_v0` train seeds 0, 1, 2.",
        "- Validation: seed 3.",
        "- Test: seed 4 with synthetic perturbations on input states only.",
        "- Actions: strict scheduler action tensors are kept unchanged, representing planned/recorded controls.",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Key observations",
        "",
    ]
    for model in ["persistence", "structured_state", "structured_state_action"]:
        lines.append(
            f"- `{model}` all RMSE: `{float(clean.loc[model, 'all_rmse']):.3f}` at noise 0.00, "
            f"`{float(noisy.loc[model, 'all_rmse']):.3f}` at noise 0.30."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The action-conditioned structured model remains better than the state-only structured model on task-state prediction, but persistence is still very strong for short-horizon link-rate prediction. This means the next model should focus on link-side structure and robustness training instead of only adding model capacity.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "structured_robustness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, actions, node_vocab, edge_vocab = load_arrays()
    clean_features = build_branch_features(arrays, actions, node_vocab, edge_vocab)
    y, persistence = build_targets_and_persistence(arrays, edge_vocab)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])

    y_res = y - persistence
    y_mean, y_std = fit_standardizer(y_res[train_idx])
    y_res_scaled = ((y_res - y_mean) / y_std).astype(np.float32)

    state_model, state_stats, _, state_info = train_model(
        clean_features, y_res_scaled, train_idx, val_idx, use_action=False, seed=52
    )
    action_model, action_stats, _, action_info = train_model(
        clean_features, y_res_scaled, train_idx, val_idx, use_action=True, seed=53
    )

    meta = {
        "link_target_dim": int(len(LINK_TYPES) * arrays["y_link"].shape[1]),
        "horizon": int(arrays["y_link"].shape[1]),
    }
    rows = []
    for level in NOISE_LEVELS:
        noisy = noisy_arrays(arrays, level, SEED + int(level * 1000))
        features = build_branch_features(noisy, actions, node_vocab, edge_vocab)
        y_true = y[test_idx]
        persistence_pred = persistence[test_idx]
        state_res = predict_model(state_model, features, test_idx, state_stats, use_action=False)
        action_res = predict_model(action_model, features, test_idx, action_stats, use_action=True)
        state_pred = np.maximum(persistence_pred + state_res * y_std + y_mean, 0.0)
        action_pred = np.maximum(persistence_pred + action_res * y_std + y_mean, 0.0)

        for model_name, pred in [
            ("persistence", persistence_pred),
            ("structured_state", state_pred),
            ("structured_state_action", action_pred),
        ]:
            rows.append({"noise_level": level, "model": model_name, **metrics(y_true, pred, meta["link_target_dim"])})

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "structured_robustness_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    curve_path = FIGURE_DIR / "structured_robustness_noise_vs_error.png"
    plot_curves(metrics_df, curve_path)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "action_dir": str(ACTION_DIR),
        "output_dir": str(OUTPUT_DIR),
        "noise_levels": NOISE_LEVELS,
        "training": {
            "structured_state": state_info,
            "structured_state_action": action_info,
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "curve_png": str(curve_path),
        },
    }
    report_path = write_report(metrics_df, summary)
    summary["outputs"]["report_md"] = str(report_path)
    summary_path = OUTPUT_DIR / "structured_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
