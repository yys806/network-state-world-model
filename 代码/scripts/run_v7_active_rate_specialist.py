"""Train active-link rate specialists for PI-JWM v7.

This script evaluates a two-stage direction: keep an activity gate, then use a
specialized regressor only on active link-step samples to predict rate magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import build_physical_edge_history, load_world_model_arrays, split_by_seed


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_seed0_9_v0"
)
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v7_active_rate_specialist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-JWM v7 active-rate specialist regressors.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx = split_by_seed(arrays["sample_seed"])
    features, feature_names, target, active = build_active_rate_table(arrays)

    split_tables = {}
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        flat_idx = sample_indices_to_flat_indices(idx, arrays["y_link_active"].shape[1], arrays["y_link_active"].shape[2])
        active_idx = flat_idx[active[flat_idx]]
        split_tables[split_name] = {
            "x": features[active_idx],
            "y": target[active_idx],
            "count": int(len(active_idx)),
        }

    models = make_models(args.seed)
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(split_tables["train"]["x"], split_tables["train"]["y"])
        row = {"model": name}
        for split_name in ["train", "val", "test"]:
            pred = model.predict(split_tables[split_name]["x"])
            pred = np.clip(pred, 0.0, None)
            true = split_tables[split_name]["y"]
            row[f"{split_name}_rmse"] = rmse(true, pred)
            row[f"{split_name}_mae"] = float(mean_absolute_error(true, pred))
            row[f"{split_name}_count"] = split_tables[split_name]["count"]
            predictions[(name, split_name)] = pred
        rows.append(row)

    baseline_row = active_mean_baseline(split_tables)
    rows.append(baseline_row)
    zero_row = zero_baseline(split_tables)
    rows.append(zero_row)
    rows = sorted(rows, key=lambda item: item["test_rmse"])
    best = rows[0]

    csv_path = args.output_dir / "v7_active_rate_specialist_metrics.csv"
    write_csv(csv_path, rows)
    fig_path = args.output_dir / "v7_active_rate_specialist_comparison.png"
    plot_metrics(fig_path, rows)
    report_path = args.output_dir / "v7_active_rate_specialist_report.md"
    summary = {
        "framework": "PI-JWM",
        "module": "v7_active_rate_specialist",
        "note": "Two-stage active-rate specialist evaluated on true active link-step samples. Use with an activity gate for end-to-end rollout.",
        "dataset_dir": str(args.dataset_dir),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        "split_active_counts": {name: table["count"] for name, table in split_tables.items()},
        "rows": rows,
        "best": best,
        "outputs": {
            "csv": str(csv_path),
            "figure": str(fig_path),
            "report": str(report_path),
        },
    }
    summary_path = args.output_dir / "v7_active_rate_specialist_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")

    meeting_dir = WORKSPACE_ROOT / "文档" / "组会" / "6.9"
    fig_meeting_dir = meeting_dir / "figs"
    fig_meeting_dir.mkdir(parents=True, exist_ok=True)
    meeting_csv = meeting_dir / "pi_jwm_v7_active_rate_specialist_metrics.csv"
    write_csv(meeting_csv, rows)
    meeting_fig = fig_meeting_dir / "pi_jwm_v7_active_rate_specialist_comparison.png"
    plot_metrics(meeting_fig, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")
    print(f"meeting_csv={meeting_csv}")
    print(f"meeting_fig={meeting_fig}")


def build_active_rate_table(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    num_samples, horizon, num_edges = arrays["y_link_active"].shape
    history = arrays["x_link"].shape[1]
    physical = build_physical_edge_history(
        arrays["x_node"],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()

    last_link = arrays["x_link"][:, -1]
    mean_link = arrays["x_link"].mean(axis=1)
    last_physical = physical[:, -1]
    mean_physical = physical.mean(axis=1)
    last_action = arrays["edge_a_hist"][:, -1]
    sum_action_hist = arrays["edge_a_hist"].sum(axis=1)
    future_action = arrays["edge_a_future"]
    task_last = arrays["x_task"][:, -1]
    task_mean = arrays["x_task"].mean(axis=1)

    parts = []
    names = []
    add_edge_features(parts, names, last_link, [f"last_link_{name}" for name in arrays["link_features"]])
    add_edge_features(parts, names, mean_link, [f"mean_link_{name}" for name in arrays["link_features"]])
    add_edge_features(parts, names, last_physical, [f"last_phys_{i}" for i in range(last_physical.shape[-1])])
    add_edge_features(parts, names, mean_physical, [f"mean_phys_{i}" for i in range(mean_physical.shape[-1])])
    add_edge_features(parts, names, last_action, [f"last_action_{name}" for name in arrays["edge_action_features"]])
    add_edge_features(parts, names, sum_action_hist, [f"sum_action_{name}" for name in arrays["edge_action_features"]])

    repeated_parts = []
    for step in range(horizon):
        step_parts = [part for part in parts]
        step_parts.append(future_action[:, step])
        if step == 0:
            names_with_future = names + [f"future_action_{name}" for name in arrays["edge_action_features"]]
        task_features = np.concatenate([task_last, task_mean], axis=-1)
        task_repeated = np.repeat(task_features[:, None, :], num_edges, axis=1)
        step_parts.append(task_repeated)
        if step == 0:
            names_with_future += [f"task_last_{name}" for name in arrays["task_features"]]
            names_with_future += [f"task_mean_{name}" for name in arrays["task_features"]]
        step_col = np.full((num_samples, num_edges, 1), float(step), dtype=np.float32)
        step_parts.append(step_col)
        if step == 0:
            names_with_future.append("horizon_step")
        repeated_parts.append(np.concatenate(step_parts, axis=-1))

    features = np.stack(repeated_parts, axis=1).reshape(num_samples * horizon * num_edges, -1).astype(np.float32)
    target = arrays["y_link_rate"].reshape(-1).astype(np.float32)
    active = arrays["y_link_active"].reshape(-1) > 0.5
    return features, names_with_future, target, active


def add_edge_features(parts: list[np.ndarray], names: list[str], values: np.ndarray, value_names: list[str]) -> None:
    parts.append(values.astype(np.float32))
    names.extend([str(name) for name in value_names])


def sample_indices_to_flat_indices(sample_idx: np.ndarray, horizon: int, num_edges: int) -> np.ndarray:
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    offsets = sample_idx[:, None, None] * horizon * num_edges
    h = np.arange(horizon, dtype=np.int64)[None, :, None] * num_edges
    e = np.arange(num_edges, dtype=np.int64)[None, None, :]
    return (offsets + h + e).reshape(-1)


def make_models(seed: int) -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gbr": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def active_mean_baseline(split_tables: dict) -> dict:
    mean_value = float(np.mean(split_tables["train"]["y"]))
    row = {"model": "train_active_mean"}
    for split_name, table in split_tables.items():
        pred = np.full_like(table["y"], mean_value)
        row[f"{split_name}_rmse"] = rmse(table["y"], pred)
        row[f"{split_name}_mae"] = float(mean_absolute_error(table["y"], pred))
        row[f"{split_name}_count"] = table["count"]
    return row


def zero_baseline(split_tables: dict) -> dict:
    row = {"model": "zero_rate"}
    for split_name, table in split_tables.items():
        pred = np.zeros_like(table["y"])
        row[f"{split_name}_rmse"] = rmse(table["y"], pred)
        row[f"{split_name}_mae"] = float(mean_absolute_error(table["y"], pred))
        row[f"{split_name}_count"] = table["count"]
    return row


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(true, pred)))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path: Path, rows: list[dict]) -> None:
    top_rows = [row for row in rows if row["model"] != "zero_rate"]
    labels = [row["model"] for row in top_rows]
    test = [row["test_rmse"] for row in top_rows]
    val = [row["val_rmse"] for row in top_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, val, width=width, label="val", color="#6B7A90")
    ax.bar(x + width / 2, test, width=width, label="test", color="#C65D3A")
    ax.axhline(228.318, color="#333333", linestyle="--", linewidth=1, label="v6 concat active-rate")
    ax.axhline(95.931, color="#2E8B57", linestyle=":", linewidth=1.2, label="previous active-only Ridge")
    for i, value in enumerate(test):
        ax.text(i + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Active-rate RMSE")
    ax.set_title("PI-JWM v7 Active-rate Specialist", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_report(summary: dict) -> str:
    lines = [
        "# PI-JWM v7 Active-rate Specialist",
        "",
        "This is a two-stage active-rate diagnostic: first identify active link-step samples, then predict the rate magnitude with a specialist regressor.",
        "",
        f"- Feature dim: `{summary['feature_dim']}`",
        f"- Split active counts: `{summary['split_active_counts']}`",
        "",
        "| model | val RMSE | test RMSE | test MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(f"| {row['model']} | {row['val_rmse']:.3f} | {row['test_rmse']:.3f} | {row['test_mae']:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The specialist is evaluated on true active link-step samples. It should be paired with the activity gate for end-to-end rollout.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
