import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = EXAMPLE_DIR / "outputs" / "multiseed_raw_v0"
DEFAULT_DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_multiseed_v0"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "outputs" / "action_proxy_v0"

ACTION_FEATURES = [
    "offload_decision_count",
    "offload_to_vehicle_count",
    "offload_to_uav_count",
    "offload_to_rsu_count",
    "offload_to_cloud_count",
    "offload_to_unknown_count",
    "active_link_count",
    "allocated_rb_total",
    "allocated_rb_mean_per_active_link",
    "cpu_progress_total",
    "computing_task_count",
    "uav_mean_speed",
    "uav_mean_displacement",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build action proxy tensors from AirFogSim multi-seed CSV logs.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def target_type(node_id):
    if not isinstance(node_id, str) or not node_id:
        return "unknown"
    lower = node_id.lower()
    if lower.startswith("vehicle"):
        return "vehicle"
    if lower.startswith("uav"):
        return "uav"
    if lower.startswith("rsu"):
        return "rsu"
    if lower.startswith("cloud"):
        return "cloud"
    return "unknown"


def read_seed_logs(raw_root, seed):
    seed_dir = raw_root / f"seed_{int(seed):03d}"
    if not seed_dir.exists():
        raise FileNotFoundError(seed_dir)
    node = pd.read_csv(seed_dir / "node_states.csv")
    link = pd.read_csv(seed_dir / "link_states.csv")
    task = pd.read_csv(seed_dir / "task_states.csv")
    for df in [node, link, task]:
        df["time"] = df["time"].round(3)
    return node, link, task


def build_time_action_table(seed, times, node, link, task):
    times = [round(float(t), 3) for t in times]
    base = pd.DataFrame({"seed": int(seed), "time": times})

    task_sorted = task.sort_values(["task_id", "time"]).copy()
    task_sorted["prev_assigned_to"] = task_sorted.groupby("task_id")["assigned_to"].shift(1)
    is_assigned = task_sorted["assigned_to"].notna()
    was_unassigned = task_sorted["prev_assigned_to"].isna()
    new_offload = task_sorted[is_assigned & was_unassigned].copy()
    new_offload["target_type"] = new_offload["assigned_to"].map(target_type)

    offload_counts = new_offload.groupby("time").size().rename("offload_decision_count")
    offload_by_type = (
        new_offload.pivot_table(index="time", columns="target_type", values="task_id", aggfunc="count", fill_value=0)
        if len(new_offload)
        else pd.DataFrame(index=pd.Index([], name="time"))
    )
    for name in ["vehicle", "uav", "rsu", "cloud", "unknown"]:
        if name not in offload_by_type.columns:
            offload_by_type[name] = 0
    offload_by_type = offload_by_type[["vehicle", "uav", "rsu", "cloud", "unknown"]].rename(
        columns={
            "vehicle": "offload_to_vehicle_count",
            "uav": "offload_to_uav_count",
            "rsu": "offload_to_rsu_count",
            "cloud": "offload_to_cloud_count",
            "unknown": "offload_to_unknown_count",
        }
    )

    active = link[link["active_task_count"] > 0].copy()
    link_actions = active.groupby("time").agg(
        active_link_count=("active_task_count", "sum"),
        allocated_rb_total=("allocated_rb_count", "sum"),
    )
    link_actions["allocated_rb_mean_per_active_link"] = (
        link_actions["allocated_rb_total"] / link_actions["active_link_count"].clip(lower=1)
    )

    task_progress = task_sorted.copy()
    task_progress["prev_computed_size"] = task_progress.groupby("task_id")["computed_size"].shift(1).fillna(0.0)
    task_progress["cpu_delta"] = (task_progress["computed_size"] - task_progress["prev_computed_size"]).clip(lower=0.0)
    cpu_progress = task_progress.groupby("time")["cpu_delta"].sum().rename("cpu_progress_total")
    computing_task_count = (
        task_progress[task_progress["lifecycle_state"] == "computing"]
        .groupby("time")
        .size()
        .rename("computing_task_count")
    )

    uav = node[node["node_type"] == "uav"].copy().sort_values(["node_id", "time"])
    uav[["prev_x", "prev_y", "prev_z"]] = uav.groupby("node_id")[["x", "y", "z"]].shift(1)
    uav["uav_displacement"] = np.sqrt(
        (uav["x"] - uav["prev_x"]).fillna(0.0) ** 2
        + (uav["y"] - uav["prev_y"]).fillna(0.0) ** 2
        + (uav["z"] - uav["prev_z"]).fillna(0.0) ** 2
    )
    uav_actions = uav.groupby("time").agg(
        uav_mean_speed=("speed", "mean"),
        uav_mean_displacement=("uav_displacement", "mean"),
    )

    out = base.set_index("time")
    for part in [offload_counts, offload_by_type, link_actions, cpu_progress, computing_task_count, uav_actions]:
        out = out.join(part, how="left")
    out[ACTION_FEATURES] = out[ACTION_FEATURES].fillna(0.0)
    out = out.reset_index()
    return out[["seed", "time", *ACTION_FEATURES]]


def slice_actions(action_array, sample_index, start_col, end_col):
    samples = []
    for row in sample_index.itertuples(index=False):
        start = int(getattr(row, start_col))
        end = int(getattr(row, end_col))
        samples.append(action_array[start : end + 1])
    return np.stack(samples, axis=0).astype(np.float32)


def plot_action_summary(output_dir, all_actions):
    generated = []
    if plt is None:
        return generated

    by_seed = all_actions.groupby("seed")[["offload_decision_count", "allocated_rb_total", "cpu_progress_total"]].sum()
    ax = by_seed.plot(kind="bar", figsize=(8, 4.5))
    ax.set_xlabel("seed")
    ax.set_ylabel("total count / amount")
    ax.set_title("Action proxy totals by seed")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "action_proxy_totals_by_seed.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    seed0 = all_actions[all_actions["seed"] == all_actions["seed"].min()]
    ax.plot(seed0["time"], seed0["offload_decision_count"], label="offload decisions")
    ax.plot(seed0["time"], seed0["active_link_count"], label="active links")
    ax.plot(seed0["time"], seed0["uav_mean_speed"], label="UAV mean speed")
    ax.set_xlabel("time")
    ax.set_title("Action proxy time series for the first seed")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = output_dir / "action_proxy_timeseries_seed0.png"
    plt.savefig(path, dpi=200)
    plt.close()
    generated.append(str(path))
    return generated


def write_report(output_dir, summary):
    lines = [
        "# Action proxy report v0",
        "",
        "## Purpose",
        "",
        "This file adds the first action-side interface for later action-conditioned world-model training.",
        "The values are action proxies extracted from observable AirFogSim logs, not a complete reinforcement-learning action record.",
        "",
        "## Action features",
        "",
    ]
    for feature in ACTION_FEATURES:
        lines.append(f"- `{feature}`")
    lines.extend(
        [
            "",
            "## Tensor shapes",
            "",
            f"- `a_hist`: `{tuple(summary['shapes']['a_hist'])}`",
            f"- `a_future`: `{tuple(summary['shapes']['a_future'])}`",
            "",
            "## Interpretation",
            "",
            "`a_hist` aligns with the historical input window. `a_future` aligns with the future label window.",
            "For strict action-conditioned rollout, the next version should log exact scheduler decisions before `env.step()`: offload route, RB indices, CPU allocation, and UAV mobility command.",
        ]
    )
    path = output_dir / "action_proxy_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = pd.read_csv(args.dataset_dir / "sample_index.csv")
    seeds = sorted(sample_index["seed"].unique().tolist())

    a_hist_parts = []
    a_future_parts = []
    all_action_tables = []
    per_seed_summary = []
    for seed in seeds:
        node, link, task = read_seed_logs(args.raw_root, seed)
        seed_samples = sample_index[sample_index["seed"] == seed].reset_index(drop=True)
        times = sorted({round(float(t), 3) for t in set(node["time"]).union(set(link["time"])).union(set(task["time"]))})
        action_table = build_time_action_table(seed, times, node, link, task)
        action_array = action_table[ACTION_FEATURES].to_numpy(dtype=np.float32)
        a_hist_parts.append(slice_actions(action_array, seed_samples, "input_start_idx", "input_end_idx"))
        a_future_parts.append(slice_actions(action_array, seed_samples, "label_start_idx", "label_end_idx"))
        all_action_tables.append(action_table)
        per_seed_summary.append(
            {
                "seed": int(seed),
                "time_steps": int(len(action_table)),
                "offload_decisions": float(action_table["offload_decision_count"].sum()),
                "allocated_rb_total": float(action_table["allocated_rb_total"].sum()),
                "cpu_progress_total": float(action_table["cpu_progress_total"].sum()),
                "uav_mean_speed_mean": float(action_table["uav_mean_speed"].mean()),
            }
        )

    a_hist = np.concatenate(a_hist_parts, axis=0)
    a_future = np.concatenate(a_future_parts, axis=0)
    all_actions = pd.concat(all_action_tables, ignore_index=True)

    action_csv = args.output_dir / "action_proxy_timeseries.csv"
    all_actions.to_csv(action_csv, index=False, encoding="utf-8-sig")
    npz_path = args.output_dir / "action_proxy_v0_samples.npz"
    np.savez_compressed(
        npz_path,
        action_features=np.array(ACTION_FEATURES),
        a_hist=a_hist,
        a_future=a_future,
        sample_seed=sample_index["seed"].to_numpy(dtype=np.int32),
    )
    figures = plot_action_summary(args.output_dir, all_actions)

    summary = {
        "raw_root": str(args.raw_root),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "action_features": ACTION_FEATURES,
        "shapes": {
            "a_hist": list(a_hist.shape),
            "a_future": list(a_future.shape),
        },
        "per_seed": per_seed_summary,
        "outputs": {
            "npz": str(npz_path),
            "action_csv": str(action_csv),
            "figures": figures,
        },
    }
    report_path = write_report(args.output_dir, summary)
    summary["outputs"]["report_md"] = str(report_path)
    (args.output_dir / "action_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
