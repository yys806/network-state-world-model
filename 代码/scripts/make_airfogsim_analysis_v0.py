from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DATASET_DIR = Path(__file__).resolve().parent / "outputs" / "dataset_v0_from_demo_run_20260507_190930"
OUT_DIR = DATASET_DIR / "airfogsim_analysis_v0"


def add_box(ax, xy, text, color):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        2.7,
        0.72,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 1.35, y + 0.36, text, ha="center", va="center", fontsize=9.5, color="#111827")


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color="#374151",
        )
    )


def draw_state_transition():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        ((0.4, 5.4), "Config + SUMO map\nnodes / routes / profiles", "#dbeafe"),
        ((3.4, 5.4), "Scheduler decisions\noffload / RB / CPU / UAV action", "#fef3c7"),
        ((6.4, 5.4), "env.step(): traffic update\nvehicle + UAV mobility", "#dcfce7"),
        ((9.4, 5.4), "Task update\ngenerate / check / lifecycle", "#fce7f3"),
        ((0.4, 3.5), "Wireless channel\npathloss + shadowing + fading", "#ede9fe"),
        ((3.4, 3.5), "Rate + transmission\nSINR, RB, data progress", "#e0f2fe"),
        ((6.4, 3.5), "Computation + storage\nCPU allocation, cache", "#fee2e2"),
        ((9.4, 3.5), "Energy / blockchain\noptional system modules", "#f3f4f6"),
        ((3.0, 1.25), "Logged states\nnode_states / link_states / task_states", "#ecfccb"),
        ((6.9, 1.25), "dataset_v0\nhistory window -> future labels", "#cffafe"),
    ]
    for xy, text, color in boxes:
        add_box(ax, xy, text, color)

    add_arrow(ax, (3.1, 5.76), (3.38, 5.76))
    add_arrow(ax, (6.1, 5.76), (6.38, 5.76))
    add_arrow(ax, (9.1, 5.76), (9.38, 5.76))
    add_arrow(ax, (10.75, 5.38), (1.75, 4.25))
    add_arrow(ax, (3.1, 3.86), (3.38, 3.86))
    add_arrow(ax, (6.1, 3.86), (6.38, 3.86))
    add_arrow(ax, (9.1, 3.86), (9.38, 3.86))
    add_arrow(ax, (5.05, 3.48), (4.6, 2.05))
    add_arrow(ax, (7.75, 3.48), (4.9, 2.05))
    add_arrow(ax, (10.75, 3.48), (5.1, 2.05))
    add_arrow(ax, (5.72, 1.61), (6.86, 1.61))

    ax.text(
        0.4,
        6.62,
        "AirFogSim state transition pipeline",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.4,
        6.25,
        "The simulator advances states by explicit mobility, channel, task, communication, and computing rules; our dataset turns these logs into supervised world-model samples.",
        fontsize=10.5,
        color="#4b5563",
    )
    ax.text(
        0.4,
        0.45,
        "Key separation: AirFogSim produces controllable trajectories and network logs; the learning model later learns a fast state predictor / surrogate from these logs.",
        fontsize=10.5,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "airfogsim_state_transition_flow.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    draw_state_transition()
    print(OUT_DIR / "airfogsim_state_transition_flow.png")
