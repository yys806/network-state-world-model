# -*- coding: utf-8 -*-
"""Generate real-data interview figures for PI-JWM stage progress.

The output directory is code/artifacts/reports/interview_real_stage_charts.
Every figure in this script is derived from local JSON/CSV experiment
artifacts; no conceptual flowchart or AI-generated image is produced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
AIRFOG_ARTIFACTS = CODE_ROOT / "artifacts" / "experiments" / "airfogsim_v0"
REPORTS = AIRFOG_ARTIFACTS / "reports"
DATASETS = AIRFOG_ARTIFACTS / "datasets"
V6_FULL80 = CODE_ROOT / "artifacts" / "experiments" / "pi_jwm_v6_eval_full80"
OUT_DIR = CODE_ROOT / "artifacts" / "reports" / "interview_real_stage_charts"

PALETTE = [
    "#276FBF",
    "#2A9D8F",
    "#E76F51",
    "#F4A261",
    "#6D597A",
    "#43AA8B",
    "#8E7DBE",
    "#D1495B",
]


def setup_style() -> None:
    """Use a readable Chinese font when available."""

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            mpl.rcParams["font.family"] = font_name
            break

    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2F3542",
            "axes.labelcolor": "#2F3542",
            "xtick.color": "#2F3542",
            "ytick.color": "#2F3542",
            "grid.color": "#DCE1E8",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save(fig: plt.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3g}", pad: int = 3) -> None:
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2F3542",
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def test_row(summary: dict) -> dict:
    for row in summary["metrics"]:
        if row["split"] == "test_seed_4":
            return row
    raise ValueError("test_seed_4 row not found")


def plot_stage_data_chain() -> Path:
    dataset_v0 = read_json(REPORTS / "dataset_summary.json")
    dataset_5 = read_json(REPORTS / "dataset_multiseed_summary.json")
    dataset_10 = read_json(DATASETS / "dataset_multiseed_seed0_9_v0" / "dataset_multiseed_summary.json")
    world_ds = read_json(DATASETS / "world_model_dataset_seed0_9_v0" / "world_model_dataset_v0_summary.json")

    stages = ["单 seed\n数据集", "5 seed\n数据集", "10 seed\n数据集", "world-model\n样本"]
    samples = [
        dataset_v0["num_samples"],
        dataset_5["num_samples"],
        dataset_10["num_samples"],
        world_ds["shapes"]["x_node"][0],
    ]
    seed_rows = pd.DataFrame(dataset_10["seed_summaries"]).T.astype(float)
    seed_rows.index = seed_rows.index.astype(int)
    seed_rows = seed_rows.sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0]})
    fig.suptitle("阶段四：数据链路与训练样本规模（真实仿真日志统计）", fontsize=18, fontweight="bold", y=1.03)

    bars = axes[0].bar(stages, samples, color=PALETTE[:4])
    axes[0].set_ylabel("样本数")
    axes[0].set_title("样本规模扩展")
    add_bar_labels(axes[0], bars, "{:.0f}")
    style_axis(axes[0])

    axes[1].plot(seed_rows.index, seed_rows["node_rows"], marker="o", label="node rows", color=PALETTE[0])
    axes[1].plot(seed_rows.index, seed_rows["link_rows"], marker="o", label="link rows", color=PALETTE[1])
    axes[1].plot(seed_rows.index, seed_rows["task_rows"], marker="o", label="task rows", color=PALETTE[2])
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("日志行数")
    axes[1].set_title("10 seed 原始日志规模")
    axes[1].legend(loc="upper center", ncol=3, fontsize=8)
    style_axis(axes[1])

    active_pct = world_ds["active_link_item_ratio"] * 100
    inactive_pct = 100 - active_pct
    axes[2].barh(["候选链路格"], [inactive_pct], color="#C8D0D9", label="inactive")
    axes[2].barh(["候选链路格"], [active_pct], left=[inactive_pct], color=PALETTE[2], label="active")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("占比 (%)")
    axes[2].set_title("活跃链路稀疏性")
    axes[2].annotate(
        f"active={active_pct:.3f}%\nnode={world_ds['num_nodes']}, edge={world_ds['num_edges']}\nH=8, K=3",
        xy=(58, 0),
        xycoords=("data", "data"),
        fontsize=10,
        va="center",
        color="#2F3542",
    )
    axes[2].legend(loc="lower center", ncol=2, fontsize=8)
    style_axis(axes[2])

    return save(fig, "01_stage4_data_chain_real_stats.png")


def plot_stage_action_baselines() -> Path:
    action_summary = read_json(REPORTS / "action_conditioned_summary.json")
    action_df = read_csv(REPORTS / "action_conditioned_metrics.csv")
    structured_df = read_csv(REPORTS / "structured_dual_branch_baseline_v0" / "structured_dual_branch_metrics.csv")

    test_action = action_df[action_df["split"] == "test_seed_4"].copy()
    test_struct = structured_df[structured_df["split"] == "test_seed_4"].copy()

    action_order = ["persistence", "state_only_ridge", "state_action_ridge"]
    action_labels = ["persistence", "state-only", "state-action"]
    struct_order = ["persistence", "structured_state", "structured_state_action"]
    struct_labels = ["persistence", "structured state", "structured state-action"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    fig.suptitle("阶段五：baseline 与动作条件建模（test seed 4）", fontsize=18, fontweight="bold", y=1.02)

    x = np.arange(len(action_order))
    w = 0.28
    metrics = [
        ("all_rmse", "overall RMSE", PALETTE[0]),
        ("link_rate_by_type_rmse", "link-rate RMSE", PALETTE[1]),
        ("task_state_rmse", "task RMSE", PALETTE[2]),
    ]
    for i, (col, label, color) in enumerate(metrics):
        vals = [float(test_action[test_action["model"] == m][col].iloc[0]) for m in action_order]
        axes[0, 0].bar(x + (i - 1) * w, vals, width=w, label=label, color=color)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(action_labels, rotation=0)
    axes[0, 0].set_title("动作输入前后：RMSE 对比")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].legend(fontsize=8, ncol=3)
    style_axis(axes[0, 0])

    x2 = np.arange(len(struct_order))
    for i, (col, label, color) in enumerate(metrics):
        vals = [float(test_struct[test_struct["model"] == m][col].iloc[0]) for m in struct_order]
        axes[0, 1].bar(x2 + (i - 1) * w, vals, width=w, label=label, color=color)
    axes[0, 1].set_xticks(x2)
    axes[0, 1].set_xticklabels(struct_labels, rotation=12, ha="right")
    axes[0, 1].set_title("结构化 state-action baseline")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend(fontsize=8, ncol=3)
    style_axis(axes[0, 1])

    state_only = test_action[test_action["model"] == "state_only_ridge"].iloc[0]
    state_action = test_action[test_action["model"] == "state_action_ridge"].iloc[0]
    improve_cols = [
        ("all_rmse", "overall"),
        ("link_rate_by_type_rmse", "link-rate"),
        ("task_state_rmse", "task"),
    ]
    improve_vals = [
        (float(state_only[c]) - float(state_action[c])) / float(state_only[c]) * 100 for c, _ in improve_cols
    ]
    bars = axes[1, 0].bar([name for _, name in improve_cols], improve_vals, color=[PALETTE[0], PALETTE[1], PALETTE[2]])
    axes[1, 0].axhline(0, color="#2F3542", linewidth=0.8)
    axes[1, 0].set_ylabel("RMSE 降低比例 (%)")
    axes[1, 0].set_title("state-action 相对 state-only 的收益")
    add_bar_labels(axes[1, 0], bars, "{:.1f}%")
    style_axis(axes[1, 0])

    split = action_summary["split"]
    split_labels = ["train", "val", "test"]
    split_vals = [split["train_samples"], split["val_samples"], split["test_samples"]]
    bars = axes[1, 1].bar(split_labels, split_vals, color=[PALETTE[0], PALETTE[1], PALETTE[2]])
    axes[1, 1].set_title("baseline 使用的数据划分")
    axes[1, 1].set_ylabel("样本数")
    axes[1, 1].annotate(
        f"state features={action_summary['feature_dims']['state_features']}\n"
        f"action features={action_summary['feature_dims']['action_features']}\n"
        f"state-action={action_summary['feature_dims']['state_action_features']}",
        xy=(1.6, max(split_vals) * 0.55),
        fontsize=10,
        color="#2F3542",
    )
    add_bar_labels(axes[1, 1], bars, "{:.0f}")
    style_axis(axes[1, 1])

    fig.tight_layout()
    return save(fig, "02_stage5_action_baseline_real_metrics.png")


def plot_stage_edge_action() -> Path:
    activity_df = read_csv(REPORTS / "edge_action_link_model_v0" / "edge_action_activity_metrics.csv")
    link_df = read_csv(REPORTS / "edge_action_link_model_v0" / "edge_action_link_model_metrics.csv")

    test_act = activity_df[activity_df["split"] == "test_seed_4"].copy()
    model_map = {
        "global_action": "global action",
        "edge_action": "edge action",
    }
    link_models = [
        ("zero_rate", "zero-rate"),
        ("edge_state_action_ridge", "edge state-action"),
        ("edge_action_ridge", "edge-action ridge"),
        ("edge_action_two_stage_rmse", "two-stage"),
        ("edge_action_oracle_activity", "oracle activity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]})
    fig.suptitle("阶段六：边级动作建模（test seed 4）", fontsize=18, fontweight="bold", y=1.03)

    x = np.arange(len(model_map))
    w = 0.26
    for i, metric in enumerate(["precision", "recall", "f1"]):
        vals = [float(test_act[test_act["model"] == m][metric].iloc[0]) for m in model_map]
        axes[0].bar(x + (i - 1) * w, vals, width=w, label=metric, color=PALETTE[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([model_map[m] for m in model_map])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("active-edge 分类指标")
    axes[0].legend(fontsize=8, ncol=3)
    style_axis(axes[0])

    test_all = link_df[(link_df["split"] == "test_seed_4") & (link_df["link_type"] == "all")]
    labels = []
    values = []
    for model, label in link_models:
        row = test_all[test_all["model"] == model]
        if not row.empty:
            labels.append(label)
            values.append(float(row["rmse"].iloc[0]))
    bars = axes[1].bar(labels, values, color=PALETTE[: len(labels)])
    axes[1].set_title("全部候选边 link-rate RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=25)
    add_bar_labels(axes[1], bars, "{:.2f}")
    style_axis(axes[1])

    test_active = link_df[(link_df["split"] == "test_seed_4") & (link_df["link_type"] == "active_edges")]
    labels = []
    values = []
    for model, label in link_models:
        row = test_active[test_active["model"] == model]
        if not row.empty:
            labels.append(label)
            values.append(float(row["rmse"].iloc[0]))
    bars = axes[2].bar(labels, values, color=PALETTE[: len(labels)])
    axes[2].set_title("真实活跃边 active-rate RMSE")
    axes[2].set_ylabel("RMSE")
    axes[2].tick_params(axis="x", rotation=25)
    add_bar_labels(axes[2], bars, "{:.1f}")
    style_axis(axes[2])

    return save(fig, "03_stage6_edge_action_real_metrics.png")


def plot_stage_world_model_versions() -> Path:
    summaries = [
        ("v0", REPORTS / "world_model_v0" / "world_model_v0_summary.json"),
        ("v1", REPORTS / "world_model_v1_staged" / "world_model_v1_staged_summary.json"),
        ("v2", REPORTS / "world_model_v2_latent_rollout" / "world_model_v2_latent_rollout_summary.json"),
        ("v3", REPORTS / "world_model_v3_graph_rollout" / "world_model_v3_graph_rollout_summary.json"),
        ("v4", REPORTS / "world_model_v4_dual_graph_rollout" / "world_model_v4_dual_graph_rollout_summary.json"),
    ]
    rows = []
    for version, path in summaries:
        row = test_row(read_json(path))
        rows.append(
            {
                "version": version,
                "activity_f1": row["activity_f1"],
                "link_rmse": row["rate_all_rmse"],
                "active_rate_rmse": row["rate_active_rmse"],
                "task_rmse": row["task_rmse"],
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    fig.suptitle("阶段六到八：world model v0-v4 演进（test seed 4）", fontsize=18, fontweight="bold", y=1.02)

    axes[0, 0].plot(df["version"], df["activity_f1"], marker="o", linewidth=2.5, color=PALETTE[0])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("active-edge F1")
    axes[0, 0].set_ylabel("F1")
    for x, y in zip(df["version"], df["activity_f1"]):
        axes[0, 0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    style_axis(axes[0, 0])

    for ax, col, title, color, fmt in [
        (axes[0, 1], "task_rmse", "task RMSE", PALETTE[1], "{:.2f}"),
        (axes[1, 0], "link_rmse", "link-rate RMSE", PALETTE[2], "{:.2f}"),
        (axes[1, 1], "active_rate_rmse", "active-rate RMSE", PALETTE[3], "{:.0f}"),
    ]:
        bars = ax.bar(df["version"], df[col], color=color)
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        add_bar_labels(ax, bars, fmt)
        style_axis(ax)

    fig.tight_layout()
    return save(fig, "04_stage6_8_world_model_versions_real_metrics.png")


def plot_stage_v3_diagnostics() -> Path:
    transfer = read_csv(REPORTS / "world_model_v3_diagnostics" / "world_model_v3_threshold_transfer.csv")
    robust = read_csv(REPORTS / "world_model_v3_diagnostics" / "world_model_v3_robustness_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    fig.suptitle("阶段七：v3 graph rollout 诊断（阈值迁移与扰动鲁棒性）", fontsize=18, fontweight="bold", y=1.03)

    rules = transfer["threshold_rule"].unique()
    labels = ["val-F1", "val-prevalence", "fixed-0.5"]
    x = np.arange(len(rules))
    w = 0.35
    for i, split in enumerate(["val_seed_3", "test_seed_4"]):
        vals = [float(transfer[(transfer["threshold_rule"] == rule) & (transfer["split"] == split)]["f1"].iloc[0]) for rule in rules]
        axes[0].bar(x + (i - 0.5) * w, vals, width=w, label=split, color=PALETTE[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("threshold transfer：F1")
    axes[0].legend(fontsize=8)
    style_axis(axes[0])

    axes[1].plot(robust["noise_level"], robust["activity_f1"], marker="o", color=PALETTE[0], linewidth=2.5)
    axes[1].set_xlabel("输入扰动强度")
    axes[1].set_ylabel("activity F1")
    axes[1].set_ylim(0.82, 0.95)
    axes[1].set_title("扰动下 active-edge F1")
    style_axis(axes[1])

    axes[2].plot(robust["noise_level"], robust["task_rmse"], marker="o", color=PALETTE[2], linewidth=2.5, label="task RMSE")
    axes[2].plot(robust["noise_level"], robust["rate_all_rmse"], marker="s", color=PALETTE[1], linewidth=2.5, label="link RMSE")
    axes[2].set_xlabel("输入扰动强度")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("扰动下预测误差")
    axes[2].legend(fontsize=8)
    style_axis(axes[2])

    return save(fig, "05_stage7_v3_diagnostics_real_metrics.png")


def plot_stage_v4_dual_graph() -> Path:
    ablation = read_csv(REPORTS / "world_model_v4_dual_graph_ablation" / "world_model_v4_dual_graph_ablation_metrics.csv")
    stability = read_csv(REPORTS / "world_model_v4_seed_stability" / "world_model_v4_seed_stability_summary.csv")
    test = ablation[ablation["split"] == "test_seed_4"].copy()
    variant_order = ["dual_full", "no_physical", "distance_only", "distance_height_speed"]
    labels = ["dual-full", "no-physical", "distance-only", "dist+height+speed"]
    test = test.set_index("physical_variant").loc[variant_order].reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    fig.suptitle("阶段八：v4 双图原型、物理边消融与 seed 稳定性", fontsize=18, fontweight="bold", y=1.02)

    bars = axes[0, 0].bar(labels, test["activity_f1"], color=PALETTE[:4])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("v4 ablation：activity F1")
    axes[0, 0].tick_params(axis="x", rotation=12)
    add_bar_labels(axes[0, 0], bars, "{:.3f}")
    style_axis(axes[0, 0])

    bars = axes[0, 1].bar(labels, test["task_rmse"], color=PALETTE[:4])
    axes[0, 1].set_title("v4 ablation：task RMSE")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].tick_params(axis="x", rotation=12)
    add_bar_labels(axes[0, 1], bars, "{:.2f}")
    style_axis(axes[0, 1])

    bars = axes[1, 0].bar(labels, test["rate_all_rmse"], color=PALETTE[:4])
    axes[1, 0].set_title("v4 ablation：link-rate RMSE")
    axes[1, 0].set_ylabel("RMSE")
    axes[1, 0].tick_params(axis="x", rotation=12)
    add_bar_labels(axes[1, 0], bars, "{:.2f}")
    style_axis(axes[1, 0])

    st_labels = ["dual-full", "no-physical"]
    x = np.arange(len(st_labels))
    ax = axes[1, 1]
    f1_mean = stability["activity_f1_mean"].to_numpy()
    f1_std = stability["activity_f1_std"].to_numpy()
    task_mean = stability["task_rmse_mean"].to_numpy()
    task_std = stability["task_rmse_std"].to_numpy()
    ax.bar(x - 0.18, f1_mean, yerr=f1_std, width=0.36, label="F1 mean±std", color=PALETTE[0], capsize=4)
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, task_mean, yerr=task_std, width=0.36, label="task RMSE mean±std", color=PALETTE[2], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(st_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("activity F1")
    ax2.set_ylabel("task RMSE")
    ax.set_title("初始化 seed 稳定性（三次运行）")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, loc="upper center", fontsize=8)

    fig.tight_layout()
    return save(fig, "06_stage8_v4_dual_graph_real_metrics.png")


def plot_stage_active_rate_calibration() -> Path:
    v3 = read_csv(REPORTS / "world_model_v3_active_rate_calibration" / "world_model_v3_active_rate_metrics.csv")
    v3_int = read_csv(REPORTS / "world_model_v3_active_rate_calibration" / "world_model_v3_active_rate_intervals.csv")
    v4 = read_csv(REPORTS / "world_model_v4_active_rate_calibration" / "world_model_v4_active_rate_metrics.csv")
    v4_int = read_csv(REPORTS / "world_model_v4_active_rate_calibration" / "world_model_v4_active_rate_intervals.csv")

    cases = [
        ("v3 rate head\npred gate", v3, "v3_rate_head", "v3_predicted_activity"),
        ("v3 active ridge\npred gate", v3, "active_rate_ridge", "v3_predicted_activity"),
        ("v3 active ridge\noracle", v3, "active_rate_ridge", "oracle_activity"),
        ("v4 rate head\npred gate", v4, "v4_rate_head", "v4_predicted_activity"),
        ("v4 physical ridge\npred gate", v4, "v4_physical_active_rate_ridge", "v4_predicted_activity"),
        ("v4 physical ridge\noracle", v4, "v4_physical_active_rate_ridge", "oracle_activity"),
    ]

    labels = []
    active_rmse = []
    for label, df, model, policy in cases:
        row = df[(df["split"] == "test_seed_4") & (df["model"] == model) & (df["activity_policy"] == policy)]
        labels.append(label)
        active_rmse.append(float(row["active_rmse"].iloc[0]))

    interval_cases = [
        ("v3 rate head\npred gate", v3_int, "v3_rate_head", "v3_predicted_activity"),
        ("v3 active ridge\npred gate", v3_int, "active_rate_ridge", "v3_predicted_activity"),
        ("v3 active ridge\noracle", v3_int, "active_rate_ridge", "oracle_activity"),
        ("v4 rate head\npred gate", v4_int, "v4_rate_head", "v4_predicted_activity"),
        ("v4 physical ridge\npred gate", v4_int, "v4_physical_active_rate_ridge", "v4_predicted_activity"),
        ("v4 physical ridge\noracle", v4_int, "v4_physical_active_rate_ridge", "oracle_activity"),
    ]
    widths = []
    coverages = []
    for _, df, model, policy in interval_cases:
        row = df[(df["model"] == model) & (df["activity_policy"] == policy) & (df["interval"] == "90%")]
        widths.append(float(row["active_mean_width"].iloc[0]))
        coverages.append(float(row["active_coverage"].iloc[0]))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    fig.suptitle("阶段七到八：active-rate 回归与 90% 区间校准", fontsize=18, fontweight="bold", y=1.03)

    bars = axes[0].bar(labels, active_rmse, color=PALETTE[: len(labels)])
    axes[0].set_title("test active-rate RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    add_bar_labels(axes[0], bars, "{:.0f}")
    style_axis(axes[0])

    bars = axes[1].bar(labels, widths, color=PALETTE[: len(labels)])
    axes[1].set_title("90% active-edge 区间平均宽度")
    axes[1].set_ylabel("width")
    axes[1].tick_params(axis="x", rotation=25)
    add_bar_labels(axes[1], bars, "{:.0f}")
    style_axis(axes[1])

    bars = axes[2].bar(labels, coverages, color=PALETTE[: len(labels)])
    axes[2].axhline(0.90, color="#2F3542", linestyle="--", linewidth=1, label="目标 90%")
    axes[2].set_ylim(0.55, 0.93)
    axes[2].set_title("90% active-edge 区间覆盖率")
    axes[2].set_ylabel("coverage")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(fontsize=8)
    add_bar_labels(axes[2], bars, "{:.3f}")
    style_axis(axes[2])

    return save(fig, "07_stage7_8_active_rate_calibration_real_metrics.png")


def v6_rows() -> pd.DataFrame:
    summary = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    rd = summary["real_data_sanity"]
    rows = []
    for mode, run in rd["runs"].items():
        test = run["test_eval"]
        rows.append(
            {
                "mode": mode,
                "best_epoch": run["best_epoch"],
                "threshold": run["activity_threshold"],
                "activity_f1": test["activity"]["f1"],
                "active_rate_rmse": test["active_rate"]["active_rmse"],
                "link_rate_rmse": test["link_rate"]["rmse"],
                "node_rmse": test["node"]["rmse"],
                "task_rmse": test["task"]["rmse"],
                "tp": test["activity"]["tp"],
                "fp": test["activity"]["fp"],
                "fn": test["activity"]["fn"],
                "tn": test["activity"]["tn"],
                "history": run["history"],
            }
        )
    return pd.DataFrame(rows), summary


def plot_stage_v6_full80_metrics() -> Path:
    df, summary = v6_rows()
    labels = ["dual", "physical-only", "information-only"]
    mode_order = ["dual", "physical_only", "information_only"]
    df = df.set_index("mode").loc[mode_order].reset_index()
    split = summary["real_data_sanity"]["split_sizes"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    fig.suptitle(
        f"阶段八：PI-JWM v6 full80 GPU 三模式消融（train={split['train']}, val={split['val']}, test={split['test']}）",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    charts = [
        ("active_rate_rmse", "active-rate RMSE", PALETTE[0], "{:.1f}"),
        ("link_rate_rmse", "link-rate RMSE", PALETTE[1], "{:.2f}"),
        ("node_rmse", "node RMSE", PALETTE[2], "{:.1f}"),
        ("task_rmse", "task RMSE", PALETTE[3], "{:.2f}"),
    ]
    for ax, (col, title, color, fmt) in zip(axes.ravel(), charts):
        bars = ax.bar(labels, df[col], color=color)
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        add_bar_labels(ax, bars, fmt)
        style_axis(ax)
    fig.tight_layout()
    return save(fig, "08_stage8_v6_full80_real_metrics.png")


def plot_stage_v6_training_curves() -> Path:
    summary = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    runs = summary["real_data_sanity"]["runs"]
    mode_labels = {
        "dual": "dual",
        "physical_only": "physical-only",
        "information_only": "information-only",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    fig.suptitle("阶段八：PI-JWM v6 full80 训练曲线（三模式真实 GPU 运行）", fontsize=18, fontweight="bold", y=1.02)
    curve_specs = [
        ("train", "total", "train total loss", True),
        ("val", "total", "val total loss", True),
        ("val", "activity", "val activity loss", True),
        ("val", "task", "val task loss", False),
    ]
    for ax, (split, key, title, logy) in zip(axes.ravel(), curve_specs):
        for i, (mode, run) in enumerate(runs.items()):
            hist = pd.DataFrame(
                {
                    "epoch": [h["epoch"] for h in run["history"]],
                    "value": [h[split][key] for h in run["history"]],
                }
            )
            ax.plot(hist["epoch"], hist["value"], linewidth=2, color=PALETTE[i], label=mode_labels[mode])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        style_axis(ax)
    fig.tight_layout()
    return save(fig, "09_stage8_v6_full80_training_curves.png")


def plot_stage_v6_activity_confusion() -> Path:
    df, _ = v6_rows()
    mode_order = ["dual", "physical_only", "information_only"]
    mode_labels = ["dual", "physical-only", "information-only"]
    df = df.set_index("mode").loc[mode_order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    fig.suptitle("阶段八：PI-JWM v6 full80 activity 混淆矩阵（test seed）", fontsize=18, fontweight="bold")
    for ax, row, title in zip(axes, df.to_dict("records"), mode_labels):
        cm = np.array([[row["tp"], row["fn"]], [row["fp"], row["tn"]]], dtype=float)
        shown = np.log10(cm + 1)
        im = ax.imshow(shown, cmap="YlGnBu")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["预测活跃", "预测非活跃"])
        if ax is axes[0]:
            ax.set_yticklabels(["真实活跃", "真实非活跃"])
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{title}\nF1={row['activity_f1']:.3f}")
        for i in range(2):
            for j in range(2):
                text_color = "white" if shown[i, j] > 3.0 else "#1F2933"
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", color=text_color, fontsize=12)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("log10(count+1)")
    return save(fig, "10_stage8_v6_activity_confusion_real_counts.png")


def main() -> None:
    setup_style()
    paths = [
        plot_stage_data_chain(),
        plot_stage_action_baselines(),
        plot_stage_edge_action(),
        plot_stage_world_model_versions(),
        plot_stage_v3_diagnostics(),
        plot_stage_v4_dual_graph(),
        plot_stage_active_rate_calibration(),
        plot_stage_v6_full80_metrics(),
        plot_stage_v6_training_curves(),
        plot_stage_v6_activity_confusion(),
    ]
    print("Generated real-data charts:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
