import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "figures"
OUTPUT_DIR = REPORT_DIR / "model_comparison_v0"


MODEL_INFO = {
    "persistence": {
        "input": "last observed state",
        "uses_action": "no",
        "family": "naive baseline",
        "conclusion": "short-horizon strong baseline",
    },
    "state_only_ridge": {
        "input": "compact state features",
        "uses_action": "no",
        "family": "linear residual baseline",
        "conclusion": "limited cross-seed generalization",
    },
    "state_action_ridge": {
        "input": "compact state + strict actions",
        "uses_action": "yes",
        "family": "linear action-conditioned baseline",
        "conclusion": "actions improve Ridge, especially task state",
    },
    "structured_state": {
        "input": "node/link/task branches",
        "uses_action": "no",
        "family": "structured branch baseline",
        "conclusion": "task improves, link side remains weak",
    },
    "structured_state_action": {
        "input": "node/link/task/action branches",
        "uses_action": "yes",
        "family": "structured action-conditioned baseline",
        "conclusion": "best task RMSE; link side is next bottleneck",
    },
}


def load_metrics():
    action = pd.read_csv(REPORT_DIR / "action_conditioned_metrics.csv")
    structured = pd.read_csv(REPORT_DIR / "structured_dual_branch_baseline_v0" / "structured_dual_branch_metrics.csv")
    action = action[action["split"] == "test_seed_4"].copy()
    structured = structured[structured["split"] == "test_seed_4"].copy()
    keep_action = action[action["model"].isin(["persistence", "state_only_ridge", "state_action_ridge"])]
    keep_structured = structured[structured["model"].isin(["structured_state", "structured_state_action"])]
    df = pd.concat([keep_action, keep_structured], ignore_index=True)
    order = list(MODEL_INFO)
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")
    for col in ["input", "uses_action", "family", "conclusion"]:
        df[col] = df["model"].astype(str).map(lambda m: MODEL_INFO[m][col])
    cols = [
        "model",
        "family",
        "input",
        "uses_action",
        "all_rmse",
        "link_rate_by_type_rmse",
        "task_state_rmse",
        "conclusion",
    ]
    return df[cols]


def plot_comparison(df, output_path):
    labels = df["model"].astype(str).tolist()
    x = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    metrics = [
        ("all_rmse", "All targets RMSE"),
        ("link_rate_by_type_rmse", "Link-rate RMSE"),
        ("task_state_rmse", "Task-state RMSE"),
    ]
    colors = ["#6b7280", "#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
    for ax, (metric, title) in zip(axes, metrics):
        vals = df[metric].to_numpy()
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(vals):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Held-out seed 4: baseline and structured model comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(df, output_path, csv_path, figure_path):
    best_task = df.loc[df["task_state_rmse"].idxmin()]
    best_all = df.loc[df["all_rmse"].idxmin()]
    lines = [
        "# Model comparison v0",
        "",
        "## Purpose",
        "",
        "This report gathers all current held-out seed 4 results into one table, so the weekly story is no longer scattered across separate Ridge, action-conditioned, and structured-model reports.",
        "",
        "## Test setting",
        "",
        "- Train seeds: 0, 1, 2",
        "- Validation seed: 3",
        "- Test seed: 4",
        "- Input history length: H=8",
        "- Prediction horizon: K=3",
        "",
        "## Summary table",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Key takeaways",
        "",
        f"- Best overall RMSE is `{best_all['model']}` with all RMSE `{best_all['all_rmse']:.3f}`. Persistence is still a very strong short-horizon baseline.",
        f"- Best task-state RMSE is `{best_task['model']}` with task RMSE `{best_task['task_state_rmse']:.3f}`. This supports the value of strict action variables for task transition prediction.",
        "- The structured state-action model improves task prediction but not link-rate prediction. The next technical step should focus on edge-level link modeling rather than only increasing model capacity.",
        "",
        "## Outputs",
        "",
        f"- comparison_csv: `{csv_path}`",
        f"- comparison_figure: `{figure_path}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    csv_path = OUTPUT_DIR / "model_comparison_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figure_path = FIGURE_DIR / "model_comparison_rmse_bar.png"
    plot_comparison(df, figure_path)
    report_path = OUTPUT_DIR / "model_comparison_report.md"
    write_report(df, report_path, csv_path, figure_path)
    summary = {
        "comparison_csv": str(csv_path),
        "comparison_figure": str(figure_path),
        "report_md": str(report_path),
        "models": df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "model_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
