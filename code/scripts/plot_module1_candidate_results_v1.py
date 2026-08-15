"""Plot Module-1 complete-combination results for the PI-JWM PPT.

The figure is intentionally a chart rather than a table.  Every colour is one
complete combination of field-history encoding, intra-graph encoding and
cross-graph coupling.  Values are read from the verified R5.1 aggregate CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


ROOT = Path(r"D:\shen\网络组")
SOURCE = (
    ROOT
    / "代码"
    / "artifacts"
    / "formal_training"
    / "pi_jwm_r5_module_confirmation_analysis_v1"
    / "aggregate_summary.csv"
)
OUTPUT = ROOT / "代码" / "artifacts" / "figures" / "ppt_module1_r5_complete_combinations_v1.png"


COMBINATIONS = {
    "B": "Masked MLP | 有向关系均值聚合 | 门控显式耦合",
    "F": "Masked MLP | 有向关系均值聚合 | 无跨图耦合",
    "G": "Masked MLP | 有向关系均值聚合 | 关系约束 Cross-Attention",
    "H": "Masked MLP | Edge-conditioned MPNN | 门控显式耦合",
}

COLOURS = {
    "B": "#16855B",  # primary working candidate
    "F": "#7B8794",  # no-coupling ablation
    "G": "#D97706",  # task-event specialist
    "H": "#C0392B",  # alternate intra-graph encoder
}

METRICS = [
    ("protocol_score", "综合协议分数（越低越好）", "protocol_score", "↓"),
    (
        "selection.required_continuous.normalized_error",
        "连续状态归一化误差（越低越好）",
        "selection.required_continuous.normalized_error",
        "↓",
    ),
    ("task.lifecycle.macro_f1", "任务生命周期 Macro-F1（越高越好）", "task.lifecycle.macro_f1", "↑"),
    ("link.active_only_rate.mae", "活动链路速率 MAE（越低越好）", "link.active_only_rate.mae", "↓"),
]


def _font() -> FontProperties:
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties(family="DejaVu Sans")


def main() -> None:
    df = pd.read_csv(SOURCE)
    df = df[(df["split"] == "validation") & df["combination_id"].isin(COMBINATIONS)]

    # Keep the candidate order stable in every panel.
    order = ["B", "G", "F", "H"]
    missing = set(order) - set(df["combination_id"])
    if missing:
        raise ValueError(f"Missing verified R5.1 combinations: {sorted(missing)}")

    font = _font()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "模块一：双图编码与耦合的真实验证结果",
        fontproperties=font,
        fontsize=25,
        color="#0B3C8C",
        y=0.982,
    )
    fig.text(
        0.5,
        0.908,
        "R5.1 验证集｜3 个随机种子｜每种颜色代表一个完整组合",
        ha="center",
        va="center",
        fontproperties=font,
        fontsize=15,
        color="#555555",
    )

    for ax, (metric_id, title, csv_metric_id, direction) in zip(axes.flat, METRICS):
        rows = []
        for combo in order:
            match = df[(df["combination_id"] == combo) & (df["metric_id"] == csv_metric_id)]
            if match.empty:
                raise ValueError(f"Missing metric {csv_metric_id} for combination {combo}")
            row = match.iloc[0]
            rows.append((combo, float(row["mean"]), float(row["sample_std"])))

        labels = [x[0] for x in rows]
        means = [x[1] for x in rows]
        stds = [x[2] for x in rows]
        y = list(range(len(rows)))
        bars = ax.barh(
            y,
            means,
            xerr=stds,
            color=[COLOURS[x] for x in labels],
            alpha=0.9,
            height=0.56,
            error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#444444"},
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontproperties=font, pad=12, loc="left")
        ax.grid(axis="x", color="#D9DEE8", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        low = min(means) - max(stds) * 0.35
        high = max(means) + max(stds) * 0.65
        if low > 0 and metric_id != "task.lifecycle.macro_f1":
            ax.set_xlim(left=low)
        ax.set_xlim(right=high)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_width() + max(stds) * 0.18,
                bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}" if mean < 10 else f"{mean:.1f}",
                va="center",
                fontsize=12,
                fontproperties=font,
                color="#222222",
            )

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOURS[c], markersize=11, label=f"{c}：{COMBINATIONS[c]}")
        for c in order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        prop=FontProperties(fname=font.get_file(), size=13),
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.text(
        0.5,
        0.005,
        "说明：本图展示已完成的 R5.1 多种子完整组合，不等同于 27 种理论组合的穷举；综合协议分数仅用于候选筛选。",
        ha="center",
        va="bottom",
        fontproperties=font,
        fontsize=12,
        color="#666666",
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.22, wspace=0.28, hspace=0.34)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
