"""Generate PI-JWM v7 fusion full80 comparison assets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR


EXPERIMENTS = {
    "concat": "pi_jwm_v7_concat_fusion_full80",
    "gated": "pi_jwm_v7_gated_fusion_full80",
    "cross_attention": "pi_jwm_v7_cross_attention_fusion_full80",
}

METRICS = [
    ("activity_f1", "Activity F1", "higher"),
    ("active_rate_rmse", "Active-rate RMSE", "lower"),
    ("link_rate_rmse", "Link-rate RMSE", "lower"),
    ("node_rmse", "Node RMSE", "lower"),
    ("task_rmse", "Task RMSE", "lower"),
]


def main() -> None:
    rows = load_rows()
    artifact_dir = ARTIFACTS_DIR / "experiments" / "pi_jwm_v7_fusion_full80"
    meeting_dir = WORKSPACE_ROOT / "文档" / "组会" / "6.9"
    fig_dir = meeting_dir / "figs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / "pi_jwm_v7_fusion_full80_metrics.csv"
    write_csv(csv_path, rows)
    meeting_csv_path = meeting_dir / "pi_jwm_v7_fusion_full80_metrics.csv"
    write_csv(meeting_csv_path, rows)

    fig_path = fig_dir / "pi_jwm_v7_fusion_full80_comparison.png"
    plot_comparison(fig_path, rows)

    report_path = meeting_dir / "PI-JWM_v7双图融合增强补充说明.md"
    report_path.write_text(render_report(rows, fig_path, meeting_csv_path), encoding="utf-8")

    print(f"csv_path={csv_path}")
    print(f"meeting_csv_path={meeting_csv_path}")
    print(f"fig_path={fig_path}")
    print(f"report_path={report_path}")


def load_rows() -> list[dict[str, float | int | str]]:
    rows = []
    for fusion, exp_name in EXPERIMENTS.items():
        summary_path = ARTIFACTS_DIR / "experiments" / exp_name / "v6_dual_graph_smoke_summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        run = data["real_data_sanity"]["runs"]["dual"]
        test_eval = run["test_eval"]
        rows.append(
            {
                "fusion": fusion,
                "best_epoch": int(run["best_epoch"]),
                "activity_threshold": float(run["activity_threshold"]),
                "activity_f1": float(test_eval["activity"]["f1"]),
                "active_rate_rmse": float(test_eval["active_rate"]["active_rmse"]),
                "link_rate_rmse": float(test_eval["link_rate"]["rmse"]),
                "node_rmse": float(test_eval["node"]["rmse"]),
                "task_rmse": float(test_eval["task"]["rmse"]),
                "tp": int(test_eval["activity"]["tp"]),
                "fp": int(test_eval["activity"]["fp"]),
                "fn": int(test_eval["activity"]["fn"]),
                "tn": int(test_eval["activity"]["tn"]),
            }
        )

    baseline = rows[0]
    for row in rows:
        for metric, _, direction in METRICS:
            base_value = float(baseline[metric])
            value = float(row[metric])
            if metric == "activity_f1":
                delta = value - base_value
                row[f"{metric}_delta"] = delta
                row[f"{metric}_relative_pct"] = 0.0 if base_value == 0 else delta / base_value * 100.0
            elif direction == "lower":
                delta = base_value - value
                row[f"{metric}_delta"] = delta
                row[f"{metric}_relative_pct"] = 0.0 if base_value == 0 else delta / base_value * 100.0
            else:
                delta = value - base_value
                row[f"{metric}_delta"] = delta
                row[f"{metric}_relative_pct"] = 0.0 if base_value == 0 else delta / base_value * 100.0
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    labels = [str(row["fusion"]) for row in rows]
    colors = ["#5B677A", "#2E8B57", "#C65D3A"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=180)
    axes = axes.ravel()

    for ax, (metric, title, direction), idx in zip(axes, METRICS, range(len(METRICS))):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=12)
        base_value = values[0]
        ax.axhline(base_value, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        value_range = max(values) - min(values)
        headroom = value_range * 0.35 if value_range > 1e-9 else max(values) * 0.12
        ax.set_ylim(0, max(values) + headroom)
        for bar, row, value in zip(bars, rows, values):
            delta = float(row[f"{metric}_relative_pct"])
            if row["fusion"] == "concat":
                label = f"{value:.3f}\nbase"
                text_color = "#2F2F2F"
            else:
                improved = delta > 0
                sign = "+" if delta >= 0 else ""
                trend = "better" if improved else "worse"
                label = f"{value:.3f}\n{sign}{delta:.1f}% {trend}"
                text_color = "#136F3A" if improved else "#A33A2B"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=text_color,
            )
        guide = "lower is better" if direction == "lower" else "higher is better"
        ax.set_xlabel(guide, fontsize=8, color="#555555", labelpad=8)

    axes[-1].axis("off")
    summary = [
        "same split full80",
        "train=1520, val=190, test=190",
        "H=3 in current dataset export",
        "cross_attention improves active-rate slightly",
        "gated improves task RMSE, but hurts link metrics",
    ]
    axes[-1].text(
        0.02,
        0.92,
        "\n".join(summary),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.55,
        color="#2F2F2F",
    )
    fig.suptitle("PI-JWM v7 Fusion Full80 Comparison", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_report(rows: list[dict[str, float | int | str]], fig_path: Path, csv_path: Path) -> str:
    table_lines = [
        "| fusion | best epoch | threshold | activity F1 | active-rate RMSE | link-rate RMSE | node RMSE | task RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {fusion} | {best_epoch} | {activity_threshold:.2f} | {activity_f1:.3f} | "
            "{active_rate_rmse:.3f} | {link_rate_rmse:.3f} | {node_rmse:.3f} | {task_rmse:.3f} |".format(
                **row
            )
        )

    cross = next(row for row in rows if row["fusion"] == "cross_attention")
    gated = next(row for row in rows if row["fusion"] == "gated")
    return "\n".join(
        [
            "# PI-JWM v7 双图融合增强补充说明",
            "",
            "本次实验在 v6 相同 train/val/test split 上，对比三种双图融合方式：",
            "",
            "- `concat`：v6 默认融合，三路表示拼接后进入 MLP。",
            "- `gated`：每条边学习 physical、information、action 三路权重。",
            "- `cross_attention`：把每条边的三路表示看成 3 个 token，做边内三模态注意力。",
            "",
            "## Full80 结果",
            "",
            *table_lines,
            "",
            f"配套图：`{fig_path}`",
            f"配套 CSV：`{csv_path}`",
            "",
            "## 结果解读",
            "",
            f"- `cross_attention` 的 active-rate RMSE 为 {cross['active_rate_rmse']:.3f}，相对 concat 的改善约 {cross['active_rate_rmse_relative_pct']:.2f}%。这说明边内三模态注意力对活动链路的速率幅值有一点帮助。",
            f"- `cross_attention` 的 task RMSE 为 {cross['task_rmse']:.3f}，相对 concat 改善约 {cross['task_rmse_relative_pct']:.2f}%，但 activity F1 从 1.000 降到 {cross['activity_f1']:.3f}，node RMSE 也变差。",
            f"- `gated` 的 task RMSE 为 {gated['task_rmse']:.3f}，相对 concat 改善约 {gated['task_rmse_relative_pct']:.2f}%，但 active-rate RMSE 和 link-rate RMSE 均略差。",
            "- 当前结论要谨慎：attention 还没有全面胜出，但增强融合已经对关键指标产生影响，其中 cross-attention 对 active-rate 有小幅正向信号。下一步需要配合 activity gating 和专门 rate head 继续推进。",
            "",
            "## 明天汇报说法",
            "",
            "可以这样讲：",
            "",
            "> 我们把 v6 的固定拼接融合扩展成了两类 v7 融合机制：gated fusion 和 per-edge cross-attention。三组 full80 已经跑完。结果上，cross-attention 在 active-rate RMSE 上从 228.318 降到 226.394，有小幅改善；task RMSE 也从 3.664 降到 3.226。但 activity F1 和 node RMSE 有所下降，说明简单换成 attention 还不是最终解法。下一步会把这个方向和 activity gating、专门 active-rate head 结合起来，让活动检测和速率回归分工更清楚。",
            "",
            "## 下一步",
            "",
            "- 保留 `cross_attention` 作为 v7 主候选，因为它对 active-rate 有正向信号。",
            "- 对 `gated` 不直接放弃，用它做融合权重可解释性分析。",
            "- 继续做 activity gating + specialized active-rate head，目标是把 active-rate RMSE 往 active-only Ridge 的 95.931 靠近。",
            "- 后续再补 seed-heldout 和扰动鲁棒性，看 attention 改进是否稳定。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
