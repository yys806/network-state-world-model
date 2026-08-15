"""Generate large-label PI-JWM v7 figures for direct PPT insertion."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = WORKSPACE_ROOT / "\u6587\u6863" / "\u5f00\u4f1a" / "6.9" / "figs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_rate_focus()
    make_specialist_focus()
    make_action_focus()
    print(OUT_DIR / "pi_jwm_v7_ppt_rate_focus.png")
    print(OUT_DIR / "pi_jwm_v7_ppt_specialist_focus.png")
    print(OUT_DIR / "pi_jwm_v7_ppt_action_focus.png")


def make_rate_focus() -> None:
    labels = ["v6\nbaseline", "attention\nfull80", "active-mixed\n200", "hybrid\n200"]
    values = [228.318, 226.394, 209.778, 216.147]
    colors = ["#4a5568", "#718096", "#2f855a", "#805ad5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_title("End-to-end active-rate RMSE", fontsize=15, weight="bold")
    ax.set_ylabel("RMSE, lower is better", fontsize=11)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, 260)
    ax.axhline(values[0], color="#a0aec0", linestyle="--", linewidth=1)
    for idx, bar in enumerate(bars):
        value = values[idx]
        ax.text(bar.get_x() + bar.get_width() / 2, value + 6, f"{value:.1f}", ha="center", fontsize=12, weight="bold")
    ax.annotate(
        "best neural\n-8.1%",
        xy=(2, values[2]),
        xytext=(2.45, 150),
        arrowprops={"arrowstyle": "->", "color": "#2f855a", "lw": 1.5},
        fontsize=11,
        color="#276749",
        ha="center",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pi_jwm_v7_ppt_rate_focus.png", dpi=240)
    plt.close(fig)


def make_specialist_focus() -> None:
    labels = ["v6 dual", "best neural\nv7", "active-rate\nspecialist"]
    values = [228.318, 209.778, 92.862]
    colors = ["#4a5568", "#2f855a", "#c05621"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.56)
    ax.set_title("Active-rate specialist headroom", fontsize=15, weight="bold")
    ax.set_xlabel("RMSE, lower is better", fontsize=11)
    ax.set_yticks(y, labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 260)
    for idx, bar in enumerate(bars):
        value = values[idx]
        suffix = ""
        if idx == 1:
            suffix = "  -8.1%"
        elif idx == 2:
            suffix = "  -59.3%"
        ax.text(value + 5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}{suffix}", va="center", fontsize=12, weight="bold")
    ax.text(94, 2.35, "two-stage specialist / not yet end-to-end", fontsize=10, color="#92400e")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pi_jwm_v7_ppt_specialist_focus.png", dpi=240)
    plt.close(fig)


def make_action_focus() -> None:
    labels = ["zero", "neural", "specialist", "budget\n top-k", "oracle\nbudget"]
    f1 = [0.0, 0.074, 0.154, 0.216, 0.598]
    colors = ["#a0aec0", "#718096", "#2f855a", "#805ad5", "#c05621"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, f1, color=colors, width=0.62)
    ax.set_title("State-to-action activity F1", fontsize=15, weight="bold")
    ax.set_ylabel("F1, higher is better", fontsize=11)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, 0.72)
    for idx, bar in enumerate(bars):
        value = f1[idx]
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=12, weight="bold")
    ax.axvspan(3.5, 4.5, color="#fef3c7", alpha=0.45, zorder=-1)
    ax.text(4, 0.67, "upper bound", ha="center", fontsize=10, color="#92400e")
    ax.annotate(
        "budget matters",
        xy=(3, f1[3]),
        xytext=(2.25, 0.42),
        arrowprops={"arrowstyle": "->", "color": "#805ad5", "lw": 1.5},
        fontsize=11,
        color="#553c9a",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pi_jwm_v7_ppt_action_focus.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
