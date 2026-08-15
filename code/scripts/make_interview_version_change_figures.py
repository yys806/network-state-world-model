# -*- coding: utf-8 -*-
"""Generate data-only version-transition charts for the final PI-JWM interview PPT.

The output focuses on version-to-version progress:
v0-v1, v1-v2, v2-v3, v3-v4, v4-v5, and v5-v6.

All exported figures are real data charts. Topic-introduction diagrams,
flowcharts, and quick-reference sheets are intentionally excluded.
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
NETWORK_ROOT = CODE_ROOT.parent
REPORTS = CODE_ROOT / "artifacts" / "experiments" / "airfogsim_v0" / "reports"
V6_FULL80 = CODE_ROOT / "artifacts" / "experiments" / "pi_jwm_v6_eval_full80"
OUT_DIR = NETWORK_ROOT / "文档" / "面试" / "src" / "version_changes"

BLUE = "#276FBF"
GREEN = "#2A9D8F"
RED = "#E76F51"
ORANGE = "#F4A261"
PURPLE = "#6D597A"
GRAY = "#B8C0CC"
TEXT = "#2F3542"
GRID = "#DCE1E8"


def setup_style() -> None:
    for path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            mpl.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break

    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_row(summary: dict) -> dict:
    for row in summary["metrics"]:
        if row["split"] == "test_seed_4":
            return row
    raise KeyError("test_seed_4 not found")


def load_rollout_metrics() -> dict[str, dict[str, float]]:
    paths = {
        "v0": REPORTS / "world_model_v0" / "world_model_v0_summary.json",
        "v1": REPORTS / "world_model_v1_staged" / "world_model_v1_staged_summary.json",
        "v2": REPORTS / "world_model_v2_latent_rollout" / "world_model_v2_latent_rollout_summary.json",
        "v3": REPORTS / "world_model_v3_graph_rollout" / "world_model_v3_graph_rollout_summary.json",
        "v4": REPORTS / "world_model_v4_dual_graph_rollout" / "world_model_v4_dual_graph_rollout_summary.json",
    }
    out: dict[str, dict[str, float]] = {}
    for version, path in paths.items():
        row = test_row(read_json(path))
        out[version] = {
            "activity_f1": float(row["activity_f1"]),
            "link_rmse": float(row["rate_all_rmse"]),
            "active_rate_rmse": float(row["rate_active_rmse"]),
            "task_rmse": float(row["task_rmse"]),
        }

    v6 = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    dual = v6["real_data_sanity"]["runs"]["dual"]["test_eval"]
    out["v6"] = {
        "activity_f1": float(dual["activity"]["f1"]),
        "link_rmse": float(dual["link_rate"]["rmse"]),
        "active_rate_rmse": float(dual["active_rate"]["active_rmse"]),
        "task_rmse": float(dual["task"]["rmse"]),
    }
    return out


METRIC_SPECS = [
    ("activity_f1", "activity F1", "higher", "{:.3f}"),
    ("link_rmse", "link-rate RMSE", "lower", "{:.2f}"),
    ("active_rate_rmse", "active-rate RMSE", "lower", "{:.1f}"),
    ("task_rmse", "task RMSE", "lower", "{:.2f}"),
]


def beneficial_change(before: float, after: float, direction: str) -> float:
    if before == 0:
        return 0.0
    if direction == "higher":
        return (after - before) / abs(before) * 100
    return (before - after) / abs(before) * 100


def add_value(ax: plt.Axes, bar, text: str) -> None:
    h = float(bar.get_height())
    ax.annotate(
        text,
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_common_transition(
    before_label: str,
    after_label: str,
    before: dict[str, float],
    after: dict[str, float],
    title: str,
    filename: str,
    note: str | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 8.8))
    fig.suptitle(title, fontsize=21, fontweight="bold", y=1.03)
    if note:
        fig.text(0.5, 0.965, note, ha="center", va="top", fontsize=11, color=TEXT)

    for ax, (key, label, direction, fmt) in zip(axes.ravel(), METRIC_SPECS):
        b = before[key]
        a = after[key]
        change = beneficial_change(b, a, direction)
        color_after = GREEN if change >= 0 else RED
        bars = ax.bar([before_label, after_label], [b, a], color=[GRAY, color_after], width=0.56)
        add_value(ax, bars[0], fmt.format(b))
        add_value(ax, bars[1], fmt.format(a))
        status = "变好" if change >= 0 else "变差"
        ax.set_title(f"{label} | {status} {change:+.1f}%", fontsize=13, pad=10)
        ax.set_ylabel(label)
        if key == "activity_f1":
            ax.set_ylim(0, max(1.05, b * 1.18, a * 1.18))
        else:
            ax.set_ylim(0, max(b, a) * 1.24)
        style_axis(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94 if note else 0.98))
    return save(fig, filename)


def plot_v4_v5_decision_change() -> Path:
    metrics_path = (
        REPORTS
        / "world_model_v5_decision_baselines_offload_scaled_v3_seedheldout"
        / "seed89"
        / "world_model_v5_decision_baselines_v0_metrics.csv"
    )
    df = pd.read_csv(metrics_path)
    test = df[(df["split"] == "test") & (df["utility"] == "airfogsim_utility")].copy()
    keep = [
        "predict_total_rb",
        "predict_world_proxy",
        "predict_default",
        "predict_offload_family",
        "predict_minus_total_rb",
    ]
    label_map = {
        "predict_total_rb": "total-RB",
        "predict_world_proxy": "world-proxy",
        "predict_default": "default",
        "predict_offload_family": "offload-family",
        "predict_minus_total_rb": "minus-RB",
    }
    test = test[test["baseline"].isin(keep)].set_index("baseline").loc[keep].reset_index()
    labels = [label_map[x] for x in test["baseline"]]
    colors = [GRAY, GREEN, ORANGE, PURPLE, RED]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.0))
    fig.suptitle("v4-v5 决策诊断指标变化", fontsize=21, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.95,
        "v5 新增候选动作排序评价，用 Top-1 命中、归一化后悔值和 Spearman 相关性衡量动作选择质量",
        ha="center",
        va="top",
        fontsize=11,
        color=TEXT,
    )

    bars = axes[0].bar(labels, test["top1_hit_mean"], color=colors)
    axes[0].set_title("Top-1 hit（越高越好）")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=18)
    for bar in bars:
        add_value(axes[0], bar, f"{bar.get_height():.2f}")
    style_axis(axes[0])

    bars = axes[1].bar(labels, test["normalized_top1_regret_mean"], color=colors)
    axes[1].set_title("normalized regret（越低越好）")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylim(0, max(test["normalized_top1_regret_mean"]) * 1.25)
    for bar in bars:
        add_value(axes[1], bar, f"{bar.get_height():.3f}")
    style_axis(axes[1])

    bars = axes[2].bar(labels, test["spearman_mean"], color=colors)
    axes[2].axhline(0, color=TEXT, linewidth=0.8)
    axes[2].set_title("Spearman rank corr.（越高越好）")
    axes[2].tick_params(axis="x", rotation=18)
    y_min = min(-0.2, float(test["spearman_mean"].min()) * 1.2)
    y_max = max(0.8, float(test["spearman_mean"].max()) * 1.2)
    axes[2].set_ylim(y_min, y_max)
    for bar in bars:
        y = bar.get_height()
        offset = 5 if y >= 0 else -13
        axes[2].annotate(
            f"{y:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
            color=TEXT,
        )
    style_axis(axes[2])

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return save(fig, "v4_v5_decision_interface_metric_change.png")


def plot_aux_cumulative_trajectory(metrics: dict[str, dict[str, float]]) -> Path:
    versions = ["v0", "v1", "v2", "v3", "v4", "v6"]
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.6))
    fig.suptitle("辅助图：v0-v6 主线 rollout 指标总体轨迹", fontsize=21, fontweight="bold", y=1.03)

    for ax, (key, label, _, fmt) in zip(axes.ravel(), METRIC_SPECS):
        vals = [metrics[v][key] for v in versions]
        ax.plot(versions, vals, marker="o", linewidth=2.8, color=BLUE)
        ax.set_title(label)
        ax.set_ylabel(label)
        for x, y in zip(versions, vals):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        if key == "activity_f1":
            ax.set_ylim(0, 1.05)
        style_axis(ax)

    fig.tight_layout()
    return save(fig, "aux_v0_v6_rollout_metric_trajectory.png")


def plot_aux_transition_heatmap(metrics: dict[str, dict[str, float]]) -> Path:
    transitions = [
        ("v0-v1", "v0", "v1"),
        ("v1-v2", "v1", "v2"),
        ("v2-v3", "v2", "v3"),
        ("v3-v4", "v3", "v4"),
        ("v5-v6\n(v4参照)", "v4", "v6"),
    ]
    metric_labels = [m[1] for m in METRIC_SPECS]
    data = []
    for _, before, after in transitions:
        row = []
        for key, _, direction, _ in METRIC_SPECS:
            row.append(beneficial_change(metrics[before][key], metrics[after][key], direction))
        data.append(row)
    arr = np.array(data)

    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    fig.suptitle("辅助图：各代指标改善率热力图", fontsize=21, fontweight="bold", y=1.03)
    vmax = max(80, float(np.nanmax(np.abs(arr))) * 1.05)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(transitions)))
    ax.set_yticklabels([t[0] for t in transitions])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if abs(arr[i, j]) > vmax * 0.55 else TEXT
            ax.text(j, i, f"{arr[i, j]:+.1f}%", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("指标改善率（正值表示变好）")
    return save(fig, "aux_transition_improvement_heatmap.png")


def plot_compact_v0_v3_progress(metrics: dict[str, dict[str, float]]) -> Path:
    """One compact PPT-ready figure for the early v0-v3 evolution."""
    transitions = [
        ("v0→v1", "v0", "v1"),
        ("v1→v2", "v1", "v2"),
        ("v2→v3", "v2", "v3"),
    ]
    metric_labels = [m[1] for m in METRIC_SPECS]
    data = []
    for _, before, after in transitions:
        row = []
        for key, _, direction, _ in METRIC_SPECS:
            row.append(beneficial_change(metrics[before][key], metrics[after][key], direction))
        data.append(row)
    arr = np.array(data)

    versions = [
        ("v0", "5.12-5.18", "统一接口\n状态+动作预测"),
        ("v1", "5.18-5.20", "+ 分阶段训练\nactivity/rate/task/joint"),
        ("v2", "5.20-5.23", "+ latent rollout\n暴露阈值迁移问题"),
        ("v3", "5.23-5.26", "+ 通信图消息传递\nactivity F1 提升"),
    ]

    fig = plt.figure(figsize=(15.6, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.25], hspace=0.28)
    fig.suptitle("v0-v3 动作条件世界模型探索：新增模块与指标变化", fontsize=20, fontweight="bold", y=0.985)

    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")
    xs = np.linspace(0.08, 0.92, len(versions))
    for i, (version, date, note) in enumerate(versions):
        x = xs[i]
        box_color = BLUE if i == 0 else GREEN
        ax_top.text(
            x,
            0.62,
            f"{version}\n{date}\n{note}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.08", facecolor=box_color, edgecolor="none"),
            transform=ax_top.transAxes,
        )
        if i < len(versions) - 1:
            ax_top.annotate(
                "",
                xy=(xs[i + 1] - 0.09, 0.62),
                xytext=(x + 0.09, 0.62),
                arrowprops=dict(arrowstyle="->", lw=2.0, color=TEXT),
                xycoords=ax_top.transAxes,
                textcoords=ax_top.transAxes,
            )
    ax_top.text(
        0.5,
        0.14,
        "绿色表示相较上一版指标改善，红色表示回退；改善率按“F1 越高越好、RMSE 越低越好”统一计算。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT,
        transform=ax_top.transAxes,
    )

    ax = fig.add_subplot(gs[1])
    vmax = max(80, float(np.nanmax(np.abs(arr))) * 1.05)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(np.arange(len(transitions)))
    ax.set_yticklabels([t[0] for t in transitions], fontsize=11)
    ax.set_title("相较上一版的核心指标变化", fontsize=13, pad=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if abs(arr[i, j]) > vmax * 0.55 else TEXT
            status = "↑" if arr[i, j] >= 0 else "↓"
            ax.text(j, i, f"{status} {arr[i, j]:+.1f}%", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("指标改善率")
    return save(fig, "compact_v0_v3_progress.png")


def plot_compact_v3_v6_progress(metrics: dict[str, dict[str, float]]) -> Path:
    """One compact PPT-ready figure for the later v3-v6 evolution."""
    transitions = [
        ("v3→v4", "v3", "v4"),
        ("v4→v6", "v4", "v6"),
    ]
    metric_labels = [m[1] for m in METRIC_SPECS]
    data = []
    for _, before, after in transitions:
        row = []
        for key, _, direction, _ in METRIC_SPECS:
            row.append(beneficial_change(metrics[before][key], metrics[after][key], direction))
        data.append(row)
    arr = np.array(data)

    metrics_path = (
        REPORTS
        / "world_model_v5_decision_baselines_offload_scaled_v3_seedheldout"
        / "seed89"
        / "world_model_v5_decision_baselines_v0_metrics.csv"
    )
    v5 = pd.read_csv(metrics_path)
    world_proxy = v5[
        (v5["split"] == "test")
        & (v5["utility"] == "airfogsim_utility")
        & (v5["baseline"] == "predict_world_proxy")
    ].iloc[0]

    versions = [
        ("v3", "5.23-5.26", "通信图消息传递\nactivity F1≈0.926"),
        ("v4", "5.28", "+ 物理边特征\n距离/高度差/速度差"),
        ("v5", "5.28-5.29", "+ 动作排序诊断\nTop-1 hit=0.75"),
        ("v6", "5.29-5.31", "+ PI-JWM 双图 rollout\ndual full80 GPU"),
    ]

    fig = plt.figure(figsize=(15.8, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.3], width_ratios=[1.45, 0.8], hspace=0.30, wspace=0.25)
    fig.suptitle("v3-v6 PI-JWM 双图建模推进：新增模块与指标变化", fontsize=20, fontweight="bold", y=0.985)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    xs = np.linspace(0.08, 0.92, len(versions))
    for i, (version, date, note) in enumerate(versions):
        x = xs[i]
        box_color = GREEN if version in {"v4", "v6"} else BLUE if version == "v3" else ORANGE
        ax_top.text(
            x,
            0.62,
            f"{version}\n{date}\n{note}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.08", facecolor=box_color, edgecolor="none"),
            transform=ax_top.transAxes,
        )
        if i < len(versions) - 1:
            ax_top.annotate(
                "",
                xy=(xs[i + 1] - 0.09, 0.62),
                xytext=(x + 0.09, 0.62),
                arrowprops=dict(arrowstyle="->", lw=2.0, color=TEXT),
                xycoords=ax_top.transAxes,
                textcoords=ax_top.transAxes,
            )
    ax_top.text(
        0.5,
        0.14,
        "v5 是候选动作排序诊断接口；v6 回到同口径 rollout 指标，因此主线指标用 v4→v6 对比。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT,
        transform=ax_top.transAxes,
    )

    ax_heat = fig.add_subplot(gs[1, 0])
    vmax = max(80, float(np.nanmax(np.abs(arr))) * 1.05)
    im = ax_heat.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(np.arange(len(metric_labels)))
    ax_heat.set_xticklabels(metric_labels, fontsize=10)
    ax_heat.set_yticks(np.arange(len(transitions)))
    ax_heat.set_yticklabels([t[0] for t in transitions], fontsize=11)
    ax_heat.set_title("主线 rollout 指标变化", fontsize=13, pad=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if abs(arr[i, j]) > vmax * 0.55 else TEXT
            status = "↑" if arr[i, j] >= 0 else "↓"
            ax_heat.text(j, i, f"{status} {arr[i, j]:+.1f}%", ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.78, pad=0.02)
    cbar.set_label("改善率")

    ax_v5 = fig.add_subplot(gs[1, 1])
    labels = ["Top-1 hit", "1-regret", "Spearman"]
    vals = [
        float(world_proxy["top1_hit_mean"]),
        1 - float(world_proxy["normalized_top1_regret_mean"]),
        float(world_proxy["spearman_mean"]),
    ]
    colors = [GREEN, GREEN, BLUE]
    bars = ax_v5.bar(labels, vals, color=colors, width=0.58)
    ax_v5.set_title("v5 world-proxy 动作诊断", fontsize=13, pad=10)
    ax_v5.set_ylim(0, 1.02)
    ax_v5.tick_params(axis="x", rotation=12)
    for bar in bars:
        add_value(ax_v5, bar, f"{bar.get_height():.3f}")
    style_axis(ax_v5)

    return save(fig, "compact_v3_v6_progress.png")


def load_v6_modes() -> pd.DataFrame:
    summary = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    rows = []
    for mode, run in summary["real_data_sanity"]["runs"].items():
        test = run["test_eval"]
        rows.append(
            {
                "mode": mode,
                "activity_f1": float(test["activity"]["f1"]),
                "active_rate_rmse": float(test["active_rate"]["active_rmse"]),
                "link_rmse": float(test["link_rate"]["rmse"]),
                "node_rmse": float(test["node"]["rmse"]),
                "task_rmse": float(test["task"]["rmse"]),
            }
        )
    order = ["dual", "physical_only", "information_only"]
    return pd.DataFrame(rows).set_index("mode").loc[order].reset_index()


def plot_aux_v6_mode_ablation() -> Path:
    df = load_v6_modes()
    labels = ["dual", "physical", "info"]
    charts = [
        ("active_rate_rmse", "active-rate RMSE", GREEN, "{:.1f}"),
        ("link_rmse", "link-rate RMSE", BLUE, "{:.2f}"),
        ("node_rmse", "node RMSE", ORANGE, "{:.1f}"),
        ("task_rmse", "task RMSE", PURPLE, "{:.2f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.6))
    fig.suptitle("辅助图：v6 full80 三种图输入模式消融", fontsize=21, fontweight="bold", y=1.03)
    for ax, (key, title, color, fmt) in zip(axes.ravel(), charts):
        bars = ax.bar(labels, df[key], color=color)
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        for bar in bars:
            add_value(ax, bar, fmt.format(bar.get_height()))
        style_axis(ax)
    fig.tight_layout()
    return save(fig, "aux_v6_full80_mode_ablation.png")


def plot_v6_result_dashboard() -> Path:
    """Dense one-page result figure for the v6 experiment slide."""
    summary = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    split = summary["real_data_sanity"]["split_sizes"]
    rows = []
    for mode, run in summary["real_data_sanity"]["runs"].items():
        test = run["test_eval"]
        rows.append(
            {
                "mode": mode,
                "activity_f1": float(test["activity"]["f1"]),
                "active_rate_rmse": float(test["active_rate"]["active_rmse"]),
                "link_rmse": float(test["link_rate"]["rmse"]),
                "node_rmse": float(test["node"]["rmse"]),
                "task_rmse": float(test["task"]["rmse"]),
                "threshold": float(test["activity"]["threshold"]),
                "tp": int(test["activity"]["tp"]),
                "fp": int(test["activity"]["fp"]),
                "fn": int(test["activity"]["fn"]),
                "tn": int(test["activity"]["tn"]),
            }
        )
    order = ["dual", "physical_only", "information_only"]
    df = pd.DataFrame(rows).set_index("mode").loc[order].reset_index()
    labels = ["dual", "physical", "info"]

    fig = plt.figure(figsize=(16.2, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1.0, 1.0], hspace=0.36, wspace=0.28)
    fig.suptitle("v6 full80 实验结果：三模式消融与链路活动识别", fontsize=21, fontweight="bold", y=0.985)

    ax_rate = fig.add_subplot(gs[0, 0])
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax_rate.bar(x - width / 2, df["link_rmse"], width, color=BLUE, label="link-rate RMSE")
    bars2 = ax_rate.bar(x + width / 2, df["active_rate_rmse"], width, color=GREEN, label="active-rate RMSE")
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(labels)
    ax_rate.set_title("链路速率预测：dual 最优")
    ax_rate.set_ylabel("RMSE")
    ax_rate.legend(loc="upper right", fontsize=9)
    for bar in list(bars1) + list(bars2):
        add_value(ax_rate, bar, f"{bar.get_height():.1f}")
    style_axis(ax_rate)

    ax_state = fig.add_subplot(gs[0, 1])
    bars = ax_state.bar(labels, df["node_rmse"], color=[BLUE, GREEN, ORANGE])
    ax_state.set_title("节点状态预测：physical 略优")
    ax_state.set_ylabel("node RMSE")
    for bar in bars:
        add_value(ax_state, bar, f"{bar.get_height():.1f}")
    style_axis(ax_state)

    ax_task = fig.add_subplot(gs[0, 2])
    bars = ax_task.bar(labels, df["task_rmse"], color=[BLUE, GREEN, ORANGE])
    ax_task.set_title("任务状态预测：info 最优")
    ax_task.set_ylabel("task RMSE")
    for bar in bars:
        add_value(ax_task, bar, f"{bar.get_height():.2f}")
    style_axis(ax_task)

    ax_conf = fig.add_subplot(gs[1, 0])
    dual = df[df["mode"] == "dual"].iloc[0]
    cm = np.array([[dual["tp"], dual["fn"]], [dual["fp"], dual["tn"]]], dtype=float)
    im = ax_conf.imshow(np.log10(cm + 1), cmap="YlGnBu")
    ax_conf.set_title("dual activity 混淆矩阵（log10(count+1)）")
    ax_conf.set_xticks([0, 1])
    ax_conf.set_xticklabels(["pred active", "pred inactive"])
    ax_conf.set_yticks([0, 1])
    ax_conf.set_yticklabels(["true active", "true inactive"])
    for i in range(2):
        for j in range(2):
            ax_conf.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=12, color=TEXT)
    cbar = fig.colorbar(im, ax=ax_conf, shrink=0.78, pad=0.02)
    cbar.set_label("log count")

    ax_thresh = fig.add_subplot(gs[1, 1])
    bars = ax_thresh.bar(labels, df["threshold"], color=[BLUE, GREEN, ORANGE], width=0.58)
    ax_thresh.plot(labels, df["activity_f1"], marker="o", color=RED, linewidth=2.2, label="activity F1")
    ax_thresh.set_ylim(0, 1.05)
    ax_thresh.set_title("activity 阈值与 F1")
    ax_thresh.legend(loc="lower right", fontsize=9)
    for bar in bars:
        add_value(ax_thresh, bar, f"{bar.get_height():.2f}")
    style_axis(ax_thresh)

    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")
    total_items = int(dual["tp"] + dual["fp"] + dual["fn"] + dual["tn"])
    active_ratio = (dual["tp"] + dual["fn"]) / total_items * 100
    info_lines = [
        ("训练设置", f"train={split['train']}  val={split['val']}  test={split['test']}"),
        ("设备与轮数", "GPU / full80"),
        ("activity 稀疏性", f"active={int(dual['tp'] + dual['fn'])}/{total_items} ({active_ratio:.4f}%)"),
        ("dual 速率结果", f"link RMSE={dual['link_rmse']:.3f}"),
        ("dual active-rate", f"RMSE={dual['active_rate_rmse']:.3f}"),
    ]
    y = 0.86
    for title, value in info_lines:
        ax_info.text(0.02, y, title, fontsize=12, fontweight="bold", color=TEXT, transform=ax_info.transAxes)
        ax_info.text(0.02, y - 0.09, value, fontsize=12, color=TEXT, transform=ax_info.transAxes)
        y -= 0.18
    ax_info.text(
        0.02,
        0.02,
        "结论：dual 更利于链路速率；physical 对节点状态更直接；info 对任务状态更直接。",
        fontsize=11,
        color=TEXT,
        transform=ax_info.transAxes,
        wrap=True,
    )

    return save(fig, "v6_full80_result_dashboard.png")


def plot_aux_active_rate_bottleneck(metrics: dict[str, dict[str, float]]) -> Path:
    v3_rate = pd.read_csv(REPORTS / "world_model_v3_active_rate_calibration" / "world_model_v3_active_rate_metrics.csv")
    v4_rate = pd.read_csv(REPORTS / "world_model_v4_active_rate_calibration" / "world_model_v4_active_rate_metrics.csv")
    v3_head = float(
        v3_rate[
            (v3_rate["split"] == "test_seed_4")
            & (v3_rate["model"] == "v3_rate_head")
            & (v3_rate["activity_policy"] == "v3_predicted_activity")
        ]["active_rmse"].iloc[0]
    )
    v3_ridge = float(
        v3_rate[
            (v3_rate["split"] == "test_seed_4")
            & (v3_rate["model"] == "active_rate_ridge")
            & (v3_rate["activity_policy"] == "v3_predicted_activity")
        ]["active_rmse"].iloc[0]
    )
    v4_physical = float(
        v4_rate[
            (v4_rate["split"] == "test_seed_4")
            & (v4_rate["model"] == "v4_physical_active_rate_ridge")
            & (v4_rate["activity_policy"] == "v4_predicted_activity")
        ]["active_rmse"].iloc[0]
    )
    labels = ["v3 rate head", "v3 active ridge", "v4 physical ridge", "v6 dual"]
    vals = [v3_head, v3_ridge, v4_physical, metrics["v6"]["active_rate_rmse"]]
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    fig.suptitle("辅助图：active-rate RMSE 瓶颈推进", fontsize=21, fontweight="bold", y=1.03)
    bars = ax.bar(labels, vals, color=[RED, ORANGE, GREEN, BLUE])
    ax.set_ylabel("active-rate RMSE")
    ax.tick_params(axis="x", rotation=12)
    for bar in bars:
        add_value(ax, bar, f"{bar.get_height():.1f}")
    style_axis(ax)
    return save(fig, "aux_active_rate_bottleneck_progress.png")


def plot_spotlight_v0_dataset_interface() -> Path:
    summary = read_json(
        CODE_ROOT
        / "artifacts"
        / "experiments"
        / "airfogsim_v0"
        / "datasets"
        / "world_model_dataset_seed0_9_v0"
        / "world_model_dataset_v0_summary.json"
    )
    shapes = summary["shapes"]
    counts = {
        "samples": shapes["x_node"][0],
        "nodes": summary["num_nodes"],
        "comm edges": summary["num_edges"],
    }
    dims = {
        "history H": shapes["x_node"][1],
        "future K": shapes["y_link_rate"][1],
        "node feat": shapes["x_node"][3],
        "link feat": shapes["x_link"][3],
        "edge action feat": shapes["edge_a_hist"][3],
        "task feat": shapes["x_task"][2],
    }
    active_pct = float(summary["active_link_item_ratio"]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.9))
    fig.suptitle("v0 亮点：统一 node-link-task-action 数据接口", fontsize=21, fontweight="bold", y=1.04)

    bars = axes[0].bar(list(counts), list(counts.values()), color=[BLUE, ORANGE, GREEN])
    axes[0].set_title("样本与图规模")
    axes[0].set_ylabel("count")
    for bar in bars:
        add_value(axes[0], bar, f"{bar.get_height():.0f}")
    style_axis(axes[0])

    bars = axes[1].bar(list(dims), list(dims.values()), color=PURPLE)
    axes[1].set_title("时序窗口与特征维度")
    axes[1].tick_params(axis="x", rotation=18)
    for bar in bars:
        add_value(axes[1], bar, f"{bar.get_height():.0f}")
    style_axis(axes[1])

    vals = [active_pct, 100 - active_pct]
    bars = axes[2].bar(["active", "inactive"], vals, color=[GREEN, GRAY])
    axes[2].set_title("链路 activity 稀疏性")
    axes[2].set_ylabel("percentage (%)")
    axes[2].set_ylim(0, 105)
    for bar, fmt in zip(bars, ["{:.3f}%", "{:.3f}%"]):
        add_value(axes[2], bar, fmt.format(bar.get_height()))
    style_axis(axes[2])

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save(fig, "spotlight_v0_dataset_interface.png")


def plot_spotlight_v2_threshold_transfer() -> Path:
    summary = read_json(REPORTS / "world_model_v2_latent_rollout" / "world_model_v2_latent_rollout_summary.json")
    val = next(row for row in summary["metrics"] if row["split"] == "val_seed_3")
    test = next(row for row in summary["metrics"] if row["split"] == "test_seed_4")

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    fig.suptitle("v2 亮点：latent rollout 暴露阈值迁移问题", fontsize=21, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.95,
        f"验证 seed 选择阈值 {summary['selected_threshold']['threshold']:.2f}；测试 seed 上 AP/AUC 仍高，但 F1 明显下降。",
        ha="center",
        va="top",
        fontsize=11,
        color=TEXT,
    )

    metrics = ["activity_f1", "activity_ap", "activity_auc"]
    labels = ["F1", "AP", "AUC"]
    x = np.arange(len(labels))
    width = 0.36
    bars1 = axes[0].bar(x - width / 2, [val[m] for m in metrics], width, label="val seed 3", color=GRAY)
    bars2 = axes[0].bar(x + width / 2, [test[m] for m in metrics], width, label="test seed 4", color=BLUE)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("排序指标与 F1 的分化")
    axes[0].legend()
    for bar in list(bars1) + list(bars2):
        add_value(axes[0], bar, f"{bar.get_height():.3f}")
    style_axis(axes[0])

    pr_metrics = ["activity_precision", "activity_recall"]
    pr_labels = ["precision", "recall"]
    x = np.arange(len(pr_labels))
    bars1 = axes[1].bar(x - width / 2, [val[m] for m in pr_metrics], width, label="val seed 3", color=GRAY)
    bars2 = axes[1].bar(x + width / 2, [test[m] for m in pr_metrics], width, label="test seed 4", color=ORANGE)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pr_labels)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("测试 seed 上 precision 被拉低")
    axes[1].legend()
    for bar in list(bars1) + list(bars2):
        add_value(axes[1], bar, f"{bar.get_height():.3f}")
    style_axis(axes[1])

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save(fig, "spotlight_v2_threshold_transfer.png")


def plot_spotlight_v3_noise_robustness() -> Path:
    df = pd.read_csv(REPORTS / "world_model_v3_diagnostics" / "world_model_v3_robustness_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8))
    fig.suptitle("v3 亮点：通信图消息传递后的扰动稳定性", fontsize=21, fontweight="bold", y=1.04)

    axes[0].plot(df["noise_level"], df["activity_f1"], marker="o", linewidth=2.8, color=GREEN)
    axes[0].set_title("activity F1 随输入噪声变化")
    axes[0].set_xlabel("noise level")
    axes[0].set_ylabel("activity F1")
    axes[0].set_ylim(0.78, 0.96)
    for x, y in zip(df["noise_level"], df["activity_f1"]):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    style_axis(axes[0])

    axes[1].plot(df["noise_level"], df["rate_all_rmse"], marker="o", linewidth=2.5, label="link-rate RMSE", color=BLUE)
    axes[1].plot(df["noise_level"], df["task_rmse"], marker="o", linewidth=2.5, label="task RMSE", color=ORANGE)
    axes[1].set_title("rate/task RMSE 随输入噪声变化")
    axes[1].set_xlabel("noise level")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    style_axis(axes[1])

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, "spotlight_v3_noise_robustness.png")


def plot_spotlight_v3_active_rate_decomposition() -> Path:
    df = pd.read_csv(REPORTS / "world_model_v3_active_rate_calibration" / "world_model_v3_active_rate_metrics.csv")
    test = df[df["split"] == "test_seed_4"]
    rows = [
        ("zero-rate", "zero_rate", "none"),
        ("v3 rate head", "v3_rate_head", "v3_predicted_activity"),
        ("ridge + oracle mask", "active_rate_ridge", "oracle_activity"),
        ("ridge + v3 mask", "active_rate_ridge", "v3_predicted_activity"),
        ("log-ridge + v3 mask", "active_rate_log_ridge", "v3_predicted_activity"),
    ]
    vals = []
    for _, model, policy in rows:
        vals.append(float(test[(test["model"] == model) & (test["activity_policy"] == policy)]["active_rmse"].iloc[0]))

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    fig.suptitle("v3 发现：active-rate 误差可拆成 mask 与 rate 两部分", fontsize=21, fontweight="bold", y=1.03)
    bars = ax.bar([r[0] for r in rows], vals, color=[GRAY, RED, GREEN, BLUE, ORANGE])
    ax.set_ylabel("active-rate RMSE")
    ax.tick_params(axis="x", rotation=12)
    for bar in bars:
        add_value(ax, bar, f"{bar.get_height():.1f}")
    style_axis(ax)
    fig.tight_layout()
    return save(fig, "spotlight_v3_active_rate_decomposition.png")


def plot_spotlight_v4_activity_calibration() -> Path:
    df = pd.read_csv(REPORTS / "world_model_v4_activity_calibration" / "world_model_v4_activity_calibration_metrics.csv")
    keep = ["fixed_0.50_raw", "val_f1_raw", "val_ratio_raw", "precision_0.80_raw"]
    test = df[(df["split"] == "test_seed_4") & (df["strategy"].isin(keep))].copy()
    numeric_cols = ["precision", "recall", "f1", "predicted_active_count", "true_active_count"]
    for col in numeric_cols:
        test[col] = pd.to_numeric(test[col], errors="coerce")
    mean = test.groupby("strategy")[numeric_cols].mean().loc[keep]
    std = test.groupby("strategy")[numeric_cols].std().fillna(0.0).loc[keep]
    labels = ["fixed 0.50", "val F1", "val ratio", "P>=0.80"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    fig.suptitle("v4 亮点：activity 阈值校准与跨种子稳定性", fontsize=21, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.95,
        "每个策略汇总 3 个训练种子的 test seed 4 结果，柱高为均值，误差线为标准差。",
        ha="center",
        va="top",
        fontsize=11,
        color=TEXT,
    )

    x = np.arange(len(labels))
    width = 0.26
    for offset, metric, color, name in [
        (-width, "precision", BLUE, "precision"),
        (0.0, "recall", ORANGE, "recall"),
        (width, "f1", GREEN, "F1"),
    ]:
        bars = axes[0].bar(
            x + offset,
            mean[metric],
            width,
            yerr=std[metric],
            capsize=3,
            color=color,
            label=name,
        )
        for bar in bars:
            add_value(axes[0], bar, f"{bar.get_height():.3f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("不同阈值策略的 activity 指标")
    axes[0].legend()
    style_axis(axes[0])

    true_count = float(mean["true_active_count"].iloc[0])
    bars = axes[1].bar(
        labels,
        mean["predicted_active_count"],
        yerr=std["predicted_active_count"],
        capsize=3,
        color=[GRAY, BLUE, GREEN, ORANGE],
        label="predicted",
    )
    axes[1].axhline(true_count, color=RED, linewidth=2.0, linestyle="--", label=f"true = {true_count:.0f}")
    axes[1].set_title("预测活跃边数量与真实数量")
    axes[1].set_ylabel("active edge count")
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].legend()
    for bar in bars:
        add_value(axes[1], bar, f"{bar.get_height():.0f}")
    style_axis(axes[1])

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, "spotlight_v4_activity_calibration.png")


def plot_spotlight_v6_activity_confusion() -> Path:
    summary = read_json(V6_FULL80 / "v6_dual_graph_smoke_summary.json")
    activity = summary["real_data_sanity"]["runs"]["dual"]["test_eval"]["activity"]
    counts = {
        "TP": float(activity["tp"]),
        "FP": float(activity["fp"]),
        "FN": float(activity["fn"]),
        "TN": float(activity["tn"]),
    }
    active = counts["TP"] + counts["FN"]
    inactive = counts["TN"] + counts["FP"]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    fig.suptitle("v6 亮点：极稀疏 activity 下的稳定识别", fontsize=21, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.95,
        f"threshold={activity['threshold']:.2f}, precision={activity['precision']:.3f}, recall={activity['recall']:.3f}, F1={activity['f1']:.3f}",
        ha="center",
        va="top",
        fontsize=11,
        color=TEXT,
    )

    vals = [counts[k] for k in ["TP", "FP", "FN", "TN"]]
    bars = axes[0].bar(["TP", "FP", "FN", "TN"], vals, color=[GREEN, RED, ORANGE, GRAY])
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("混淆矩阵计数（symlog）")
    axes[0].set_ylabel("count")
    for bar in bars:
        add_value(axes[0], bar, f"{bar.get_height():.0f}")
    style_axis(axes[0])

    ratio_vals = [active / (active + inactive) * 100, inactive / (active + inactive) * 100]
    bars = axes[1].bar(["active", "inactive"], ratio_vals, color=[GREEN, GRAY])
    axes[1].set_title("测试集 activity 稀疏比例")
    axes[1].set_ylabel("percentage (%)")
    axes[1].set_ylim(0, 105)
    for bar in bars:
        add_value(axes[1], bar, f"{bar.get_height():.4f}%")
    style_axis(axes[1])

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save(fig, "spotlight_v6_activity_confusion.png")


def write_readme() -> None:
    content = """# v0-v6 版本变化纯数据图

本目录按版本迁移组织，只保留数据图，不包含课题介绍表格、流程图或速查表。

## 主图顺序

1. `v0_v1_staged_training_metric_change.png`：v0 到 v1，分阶段训练后的四类 rollout 指标变化。
2. `v1_v2_latent_rollout_metric_change.png`：v1 到 v2，引入 latent rollout 后的四类 rollout 指标变化。
3. `v2_v3_comm_graph_metric_change.png`：v2 到 v3，引入通信图消息传递后的四类 rollout 指标变化。
4. `v3_v4_physical_edge_metric_change.png`：v3 到 v4，引入物理边特征后的四类 rollout 指标变化。
5. `v4_v5_decision_interface_metric_change.png`：v4 到 v5，新增候选动作排序诊断指标。
6. `v5_v6_dual_graph_full80_metric_change.png`：v5 到 v6，v6 full80 dual 回到 PI-JWM 双图 rollout 后的主线指标变化；图中采用 v4 rollout 指标作为同口径参照。

绿色表示该指标方向上变好，红色表示该指标方向上变差；RMSE 类指标越低越好，activity F1 越高越好。

## 辅助图

- `aux_v0_v6_rollout_metric_trajectory.png`：v0-v6 主线 rollout 指标总体轨迹。
- `aux_transition_improvement_heatmap.png`：各代指标改善率热力图。
- `aux_v6_full80_mode_ablation.png`：v6 dual / physical-only / information-only 三种模式消融。
- `aux_active_rate_bottleneck_progress.png`：active-rate RMSE 从 v3 到 v6 的瓶颈推进。

## 单版本亮点图

- `spotlight_v0_dataset_interface.png`：v0 统一 node-link-task-action 数据接口和 activity 稀疏性。
- `spotlight_v2_threshold_transfer.png`：v2 latent rollout 中 AP/AUC 高但阈值迁移导致 F1 不稳。
- `spotlight_v3_noise_robustness.png`：v3 通信图消息传递后的输入扰动稳定性。
- `spotlight_v3_active_rate_decomposition.png`：v3 active-rate 误差拆解，区分 activity mask 与 rate regression。
- `spotlight_v4_activity_calibration.png`：v4 activity 阈值校准与活跃边数量对齐。
- `spotlight_v6_activity_confusion.png`：v6 full80 dual 在极稀疏 activity 下的混淆矩阵。
"""
    (OUT_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    metrics = load_rollout_metrics()
    paths = [
        plot_common_transition(
            "v0",
            "v1",
            metrics["v0"],
            metrics["v1"],
            "v0-v1 分阶段训练后指标变化",
            "v0_v1_staged_training_metric_change.png",
        ),
        plot_common_transition(
            "v1",
            "v2",
            metrics["v1"],
            metrics["v2"],
            "v1-v2 latent rollout 指标变化",
            "v1_v2_latent_rollout_metric_change.png",
        ),
        plot_common_transition(
            "v2",
            "v3",
            metrics["v2"],
            metrics["v3"],
            "v2-v3 通信图消息传递指标变化",
            "v2_v3_comm_graph_metric_change.png",
        ),
        plot_common_transition(
            "v3",
            "v4",
            metrics["v3"],
            metrics["v4"],
            "v3-v4 物理边特征加入后指标变化",
            "v3_v4_physical_edge_metric_change.png",
        ),
        plot_v4_v5_decision_change(),
        plot_common_transition(
            "v4",
            "v6 dual",
            metrics["v4"],
            metrics["v6"],
            "v5-v6 双图 full80 rollout 指标变化",
            "v5_v6_dual_graph_full80_metric_change.png",
            note="v5 是决策诊断接口；v6 的 rollout 指标采用 v4 作为同口径参照",
        ),
        plot_aux_cumulative_trajectory(metrics),
        plot_aux_transition_heatmap(metrics),
        plot_compact_v0_v3_progress(metrics),
        plot_compact_v3_v6_progress(metrics),
        plot_aux_v6_mode_ablation(),
        plot_v6_result_dashboard(),
        plot_aux_active_rate_bottleneck(metrics),
        plot_spotlight_v0_dataset_interface(),
        plot_spotlight_v2_threshold_transfer(),
        plot_spotlight_v3_noise_robustness(),
        plot_spotlight_v3_active_rate_decomposition(),
        plot_spotlight_v4_activity_calibration(),
        plot_spotlight_v6_activity_confusion(),
    ]
    write_readme()
    print("Generated version-change figures:")
    for path in paths:
        print(path)
    print(OUT_DIR / "README.md")


if __name__ == "__main__":
    main()
