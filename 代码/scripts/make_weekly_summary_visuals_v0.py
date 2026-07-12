from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DATASET_DIR = Path(__file__).resolve().parent / "outputs" / "dataset_v0_from_demo_run_20260507_190930"
OUT_DIR = DATASET_DIR / "airfogsim_analysis_v0"


def box(ax, xy, wh, text, fc, fontsize=10):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#1f2937",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111827")


def arrow(ax, start, end, color="#374151"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color=color)
    )


def draw_complexity_comparison():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.4, 6.55, "Simulator rollout vs. learned world-model inference", fontsize=17, fontweight="bold")
    ax.text(
        0.4,
        6.18,
        "AirFogSim explicitly updates every module at each future step; a learned model can amortize this into encoded history + latent rollout.",
        fontsize=10.5,
        color="#4b5563",
    )

    box(ax, (0.55, 4.8), (2.3, 0.75), "Current state\n$s_t$", "#dbeafe")
    box(ax, (3.35, 5.2), (2.35, 0.75), "Traffic + mobility\n$C_{traffic}$", "#dcfce7")
    box(ax, (6.05, 5.2), (2.35, 0.75), "Channel + rate\n$C_{channel}$", "#ede9fe")
    box(ax, (8.75, 5.2), (2.35, 0.75), "Task + compute\n$C_{task}$", "#fee2e2")
    box(ax, (10.65, 4.05), (1.8, 0.75), "Repeat $K$ steps", "#fef3c7")

    arrow(ax, (2.85, 5.18), (3.32, 5.55))
    arrow(ax, (5.7, 5.55), (6.02, 5.55))
    arrow(ax, (8.4, 5.55), (8.72, 5.55))
    arrow(ax, (10.95, 5.18), (11.45, 4.82))
    arrow(ax, (10.65, 4.42), (2.2, 4.8), "#9ca3af")

    ax.text(
        0.7,
        3.95,
        r"$C_{sim}(K) \approx K(C_{traffic}+C_{channel}+C_{task}+C_{scheduling})$",
        fontsize=12,
        color="#111827",
    )
    ax.text(
        0.7,
        3.65,
        "Used for accurate controllable simulation, but repeated rollout can be expensive.",
        fontsize=10,
        color="#4b5563",
    )

    box(ax, (0.55, 2.05), (2.3, 0.75), "History window\n$X_{t-H+1:t}$", "#cffafe")
    box(ax, (3.35, 2.05), (2.35, 0.75), "Encoder\n$C_{encode}$", "#e0f2fe")
    box(ax, (6.05, 2.05), (2.35, 0.75), "Latent rollout\n$K C_{rollout}$", "#fce7f3")
    box(ax, (8.75, 2.05), (2.35, 0.75), "Decoder\n$C_{decode}$", "#ecfccb")
    box(ax, (11.35, 2.05), (1.25, 0.75), "Future states\n$\\hat{s}_{t+1:t+K}$", "#f3f4f6", fontsize=8.5)

    arrow(ax, (2.85, 2.42), (3.32, 2.42))
    arrow(ax, (5.7, 2.42), (6.02, 2.42))
    arrow(ax, (8.4, 2.42), (8.72, 2.42))
    arrow(ax, (11.1, 2.42), (11.32, 2.42))

    ax.text(
        0.7,
        1.1,
        r"$C_{wm}(K) \approx C_{encode}(H)+K C_{latent\ rollout}+C_{decode}(K)$",
        fontsize=12,
        color="#111827",
    )
    ax.text(
        0.7,
        0.8,
        "This is a potential online surrogate; the speed advantage must be validated by timing experiments.",
        fontsize=10,
        color="#4b5563",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "complexity_simulator_vs_world_model.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    draw_complexity_comparison()
    print(OUT_DIR / "complexity_simulator_vs_world_model.png")
