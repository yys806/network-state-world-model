import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler
from export_multiseed_dataset_v0 import make_config
from strict_action_helpers import allocate_cpu_by_assigned_node


EXAMPLE_DIR = Path(os.environ.get("PI_JWM_AIRFOGSIM_EXAMPLE_DIR", Path(__file__).resolve().parent))
DEFAULT_DATASET_DIR = EXAMPLE_DIR / "outputs" / "dataset_multiseed_v0"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "outputs" / "strict_action_logs_v0"

ACTION_FEATURES = [
    "offload_decision_count",
    "return_route_count",
    "rb_task_count",
    "rb_total",
    "rb_mean_per_task",
    "cpu_task_count",
    "cpu_total_alloc",
    "cpu_mean_alloc",
    "uav_command_count",
    "uav_speed_mean",
    "uav_angle_sin_mean",
    "uav_angle_cos_mean",
    "uav_phi_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export strict scheduler action logs and aligned action tensors.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def as_float(value, default=0.0):
    try:
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def node_type_from_id(node_id):
    if not isinstance(node_id, str):
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


def action_effect_time(env):
    interval = getattr(env, "simulation_interval", 0.1)
    return round(as_float(env.simulation_time) + as_float(interval, 0.1), 3)


class LoggingAlgorithmModule(BaseAlgorithmModule):
    """Default AirFogSim scheduler with explicit action logging.

    The scheduling logic mirrors BaseAlgorithmModule. The difference is that every
    decision written into the environment is also written to a CSV-friendly list.
    """

    def __init__(self, seed):
        super().__init__()
        self.seed = int(seed)
        self.offload_rows = []
        self.return_rows = []
        self.rb_rows = []
        self.cpu_rows = []
        self.uav_rows = []

    def scheduleReturning(self, env):
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        rsu_infos = self.trafficScheduler.getRSUTrafficInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                distance_dict = {}
                current_node_id = task.getCurrentNodeId()
                for rsu_id in rsu_infos:
                    distance_dict[rsu_id] = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, rsu_id)
                if not distance_dict:
                    continue
                distance_list = sorted(distance_dict.items(), key=lambda item: item[1], reverse=False)
                return_route = [distance_list[0][0]]
                self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)
                self.return_rows.append(
                    {
                        "seed": self.seed,
                        "time": action_effect_time(env),
                        "task_id": task.getTaskId(),
                        "task_node_id": task_node_id,
                        "current_node_id": current_node_id,
                        "return_target_id": return_route[0],
                        "return_distance": as_float(distance_list[0][1]),
                    }
                )

    def scheduleOffloading(self, env):
        all_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        for task_dict in all_task_infos:
            task_node_id = task_dict["task_node_id"]
            task_id = task_dict["task_id"]
            neighbor_infos = self.entityScheduler.getNeighborNodeInfosById(env, task_node_id, sorted_by="distance", max_num=5)
            if len(neighbor_infos) > 0:
                target_node_id = neighbor_infos[0]["id"]
                flag = self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, target_node_id)
                assert flag
                self.offload_rows.append(
                    {
                        "seed": self.seed,
                        "time": action_effect_time(env),
                        "task_id": task_id,
                        "task_node_id": task_node_id,
                        "source_node_id": task_node_id,
                        "target_node_id": target_node_id,
                        "target_node_type": node_type_from_id(target_node_id),
                        "candidate_count": len(neighbor_infos),
                        "nearest_distance": as_float(neighbor_infos[0].get("distance", 0.0)),
                    }
                )

    def scheduleCommunication(self, env):
        n_rb = self.commScheduler.getNumberOfRB(env)
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        all_offloading_task_infos = all_offloading_task_infos[:n_rb]
        avg_rb_nos = max(1, n_rb // max(1, len(all_offloading_task_infos)))
        rb_ctr = 0
        for task_dict in all_offloading_task_infos:
            allocated_rb_nos = [(rb_ctr + i) % n_rb for i in range(avg_rb_nos)]
            rb_ctr = (rb_ctr + avg_rb_nos) % n_rb
            self.commScheduler.setCommunicationWithRB(env, task_dict["task_id"], allocated_rb_nos)
            self.rb_rows.append(
                {
                    "seed": self.seed,
                    "time": action_effect_time(env),
                    "task_id": task_dict["task_id"],
                    "task_node_id": task_dict.get("task_node_id", ""),
                    "current_node_id": task_dict.get("current_node_id", ""),
                    "assigned_to": task_dict.get("assigned_to", ""),
                    "rb_count": len(allocated_rb_nos),
                    "rb_indices": " ".join(str(item) for item in allocated_rb_nos),
                }
            )

    def scheduleComputing(self, env):
        def alloc_cpu_callback(computing_tasks, **kwargs):
            task_list = []
            for tasks in computing_tasks.values():
                for task in tasks:
                    task_list.append(task.to_dict())

            alloc_cpu_dict, accepted_tasks = allocate_cpu_by_assigned_node(
                task_list,
                lambda node_id: self.entityScheduler.getNodeInfoById(env, node_id),
            )
            task_counts = {}
            for task_dict in accepted_tasks:
                assigned_node_id = task_dict["assigned_to"]
                task_counts[assigned_node_id] = task_counts.get(assigned_node_id, 0) + 1
            for task_dict in accepted_tasks:
                task_id = task_dict["task_id"]
                assigned_node_id = task_dict["assigned_to"]
                alloc_cpu = alloc_cpu_dict[task_id]
                self.cpu_rows.append(
                    {
                        "seed": self.seed,
                        "time": action_effect_time(env),
                        "task_id": task_id,
                        "task_node_id": task_dict.get("task_node_id", ""),
                        "assigned_to": assigned_node_id,
                        "assigned_node_type": node_type_from_id(assigned_node_id),
                        "allocated_cpu": as_float(alloc_cpu),
                        "num_tasks_on_node": task_counts[assigned_node_id],
                    }
                )
            return alloc_cpu_dict

        self.compScheduler.setComputingCallBack(env, alloc_cpu_callback)

    def scheduleTraffic(self, env):
        uavs_info = self.trafficScheduler.getUAVTrafficInfos(env)
        uavs_mobile_pattern = {}
        for uav_id, uav_info in uavs_info.items():
            current_position = uav_info["position"]
            target_position = self.missionScheduler.getNearestMissionPosition(env, uav_id, uav_info["position"])
            target_source = "nearest_mission"
            if target_position is None:
                target_source = "random"
                random_angle = np.random.uniform(0, 2 * np.pi)
                mobility_pattern = {"angle": random_angle, "phi": 0}
                uav_speed_range = self.trafficScheduler.getConfig(env, "UAV_speed_range")
                mobility_pattern["speed"] = random.uniform(uav_speed_range[0], uav_speed_range[1])
            else:
                delta_x = target_position[0] - current_position[0]
                delta_y = target_position[1] - current_position[1]
                delta_z = target_position[2] - current_position[2]
                angle = np.arctan2(delta_y, delta_x)
                distance_xy = np.sqrt(delta_x**2 + delta_y**2)
                phi = np.arctan2(delta_z, distance_xy)
                mobility_pattern = {"angle": angle, "phi": phi}
                uav_speed_range = self.trafficScheduler.getConfig(env, "UAV_speed_range")
                mobility_pattern["speed"] = random.uniform(uav_speed_range[0], uav_speed_range[1])
            uavs_mobile_pattern[uav_id] = mobility_pattern
            target = target_position if target_position is not None else ("", "", "")
            self.uav_rows.append(
                {
                    "seed": self.seed,
                    "time": action_effect_time(env),
                    "uav_id": uav_id,
                    "current_x": as_float(current_position[0]),
                    "current_y": as_float(current_position[1]),
                    "current_z": as_float(current_position[2]),
                    "target_source": target_source,
                    "target_x": as_float(target[0], 0.0),
                    "target_y": as_float(target[1], 0.0),
                    "target_z": as_float(target[2], 0.0),
                    "angle": as_float(mobility_pattern["angle"]),
                    "phi": as_float(mobility_pattern["phi"]),
                    "speed": as_float(mobility_pattern["speed"]),
                }
            )
        self.trafficScheduler.setUAVMobilityPatterns(env, uavs_mobile_pattern)


def write_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(seed, max_time, output_dir):
    np.random.seed(seed)
    random.seed(seed)
    config = make_config(load_config(EXAMPLE_DIR / "config.yaml"), max_time=max_time)
    run_dir = output_dir / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm = LoggingAlgorithmModule(seed)
    algorithm.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")

    try:
        while not env.isDone():
            algorithm.scheduleStep(env)
            env.step()
            done_task_num = TaskScheduler.getDoneTaskNum(env)
            failed_task_num = TaskScheduler.getOutOfDDLTasks(env)
            success_ratio = done_task_num / max(1, done_task_num + failed_task_num)
            print(
                "strict-actions seed={} time={:.1f} success={:.2f} done={} failed={}".format(
                    seed, env.simulation_time, success_ratio, done_task_num, failed_task_num
                ),
                end="\r",
            )
    finally:
        env.close()
        print()

    write_rows(
        run_dir / "offload_actions.csv",
        [
            "seed",
            "time",
            "task_id",
            "task_node_id",
            "source_node_id",
            "target_node_id",
            "target_node_type",
            "candidate_count",
            "nearest_distance",
        ],
        algorithm.offload_rows,
    )
    write_rows(
        run_dir / "return_actions.csv",
        ["seed", "time", "task_id", "task_node_id", "current_node_id", "return_target_id", "return_distance"],
        algorithm.return_rows,
    )
    write_rows(
        run_dir / "rb_actions.csv",
        ["seed", "time", "task_id", "task_node_id", "current_node_id", "assigned_to", "rb_count", "rb_indices"],
        algorithm.rb_rows,
    )
    write_rows(
        run_dir / "cpu_actions.csv",
        ["seed", "time", "task_id", "task_node_id", "assigned_to", "assigned_node_type", "allocated_cpu", "num_tasks_on_node"],
        algorithm.cpu_rows,
    )
    write_rows(
        run_dir / "uav_mobility_actions.csv",
        [
            "seed",
            "time",
            "uav_id",
            "current_x",
            "current_y",
            "current_z",
            "target_source",
            "target_x",
            "target_y",
            "target_z",
            "angle",
            "phi",
            "speed",
        ],
        algorithm.uav_rows,
    )
    return {
        "seed": int(seed),
        "run_dir": str(run_dir),
        "offload_actions": len(algorithm.offload_rows),
        "return_actions": len(algorithm.return_rows),
        "rb_actions": len(algorithm.rb_rows),
        "cpu_actions": len(algorithm.cpu_rows),
        "uav_mobility_actions": len(algorithm.uav_rows),
    }


def read_action_file(run_dir, filename):
    path = run_dir / filename
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required action log is missing or empty: {path}")
    return pd.read_csv(path)


def aggregate_actions_for_seed(seed, run_dir, times):
    base = pd.DataFrame({"seed": int(seed), "time": [round(float(t), 3) for t in times]}).set_index("time")
    offload = read_action_file(run_dir, "offload_actions.csv")
    returns = read_action_file(run_dir, "return_actions.csv")
    rb = read_action_file(run_dir, "rb_actions.csv")
    cpu = read_action_file(run_dir, "cpu_actions.csv")
    uav = read_action_file(run_dir, "uav_mobility_actions.csv")

    if len(offload):
        offload_count = offload.groupby("time").size().rename("offload_decision_count")
        base = base.join(offload_count, how="left")
    if len(returns):
        return_count = returns.groupby("time").size().rename("return_route_count")
        base = base.join(return_count, how="left")
    if len(rb):
        rb_agg = rb.groupby("time").agg(rb_task_count=("task_id", "count"), rb_total=("rb_count", "sum"))
        rb_agg["rb_mean_per_task"] = rb_agg["rb_total"] / rb_agg["rb_task_count"].clip(lower=1)
        base = base.join(rb_agg, how="left")
    if len(cpu):
        cpu_agg = cpu.groupby("time").agg(cpu_task_count=("task_id", "count"), cpu_total_alloc=("allocated_cpu", "sum"))
        cpu_agg["cpu_mean_alloc"] = cpu_agg["cpu_total_alloc"] / cpu_agg["cpu_task_count"].clip(lower=1)
        base = base.join(cpu_agg, how="left")
    if len(uav):
        uav = uav.copy()
        uav["angle_sin"] = np.sin(uav["angle"])
        uav["angle_cos"] = np.cos(uav["angle"])
        uav_agg = uav.groupby("time").agg(
            uav_command_count=("uav_id", "count"),
            uav_speed_mean=("speed", "mean"),
            uav_angle_sin_mean=("angle_sin", "mean"),
            uav_angle_cos_mean=("angle_cos", "mean"),
            uav_phi_mean=("phi", "mean"),
        )
        base = base.join(uav_agg, how="left")

    for feature in ACTION_FEATURES:
        if feature not in base.columns:
            base[feature] = 0.0
    base[ACTION_FEATURES] = base[ACTION_FEATURES].fillna(0.0)
    return base.reset_index()[["seed", "time", *ACTION_FEATURES]]


def slice_actions(action_array, sample_index, start_col, end_col):
    samples = []
    for row in sample_index.itertuples(index=False):
        start = int(getattr(row, start_col))
        end = int(getattr(row, end_col))
        samples.append(action_array[start : end + 1])
    return np.stack(samples, axis=0).astype(np.float32)


def build_aligned_tensors(output_dir, dataset_dir, summaries):
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv")
    a_hist_parts = []
    a_future_parts = []
    action_tables = []
    for summary in summaries:
        seed = summary["seed"]
        seed_samples = sample_index[sample_index["seed"] == seed].reset_index(drop=True)
        times = sorted(set(seed_samples["input_start_time"]).union(seed_samples["label_end_time"]))
        # sample_index only stores window boundaries, so reconstruct the full 0.1s grid from min/max.
        min_time = min(seed_samples["input_start_time"].min(), seed_samples["label_start_time"].min())
        max_time = max(seed_samples["input_end_time"].max(), seed_samples["label_end_time"].max())
        times = np.round(np.arange(min_time, max_time + 1e-6, 0.1), 3).tolist()
        table = aggregate_actions_for_seed(seed, Path(summary["run_dir"]), times)
        action_array = table[ACTION_FEATURES].to_numpy(dtype=np.float32)
        a_hist_parts.append(slice_actions(action_array, seed_samples, "input_start_idx", "input_end_idx"))
        a_future_parts.append(slice_actions(action_array, seed_samples, "label_start_idx", "label_end_idx"))
        action_tables.append(table)

    a_hist = np.concatenate(a_hist_parts, axis=0)
    a_future = np.concatenate(a_future_parts, axis=0)
    all_actions = pd.concat(action_tables, ignore_index=True)
    all_actions.to_csv(output_dir / "strict_action_timeseries.csv", index=False, encoding="utf-8-sig")
    npz_path = output_dir / "strict_action_v0_samples.npz"
    np.savez_compressed(
        npz_path,
        action_features=np.array(ACTION_FEATURES),
        a_hist=a_hist,
        a_future=a_future,
        sample_seed=sample_index["seed"].to_numpy(dtype=np.int32),
    )
    return all_actions, npz_path, a_hist, a_future


def plot_outputs(output_dir, all_actions):
    paths = []
    totals = all_actions.groupby("seed")[["offload_decision_count", "rb_total", "cpu_total_alloc", "uav_command_count"]].sum()
    ax = totals.plot(kind="bar", figsize=(8.5, 4.8))
    ax.set_title("Strict action totals by seed")
    ax.set_xlabel("seed")
    ax.set_ylabel("count / amount")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "strict_action_totals_by_seed.png"
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(str(path))

    first_seed = int(all_actions["seed"].min())
    seed0 = all_actions[all_actions["seed"] == first_seed]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(seed0["time"], seed0["offload_decision_count"], label="offload decisions")
    ax.plot(seed0["time"], seed0["rb_task_count"], label="RB tasks")
    ax.plot(seed0["time"], seed0["cpu_task_count"], label="CPU tasks")
    ax.plot(seed0["time"], seed0["uav_speed_mean"], label="UAV speed mean")
    ax.set_title(f"Strict action time series for seed {first_seed}")
    ax.set_xlabel("time")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = output_dir / "strict_action_timeseries_seed0.png"
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(str(path))
    return paths


def write_report(output_dir, summary):
    lines = [
        "# Strict action log report v0",
        "",
        "## Purpose",
        "",
        "This experiment records scheduler decisions directly while AirFogSim is running.",
        "It is stricter than `action_proxy_v0`, which inferred action-side variables from observable state logs.",
        "",
        "## Recorded actions",
        "",
        "- Offloading: task id, source task node, selected target node, target node type.",
        "- Returning: task id, current node, selected return RSU.",
        "- Communication: task id and allocated RB indices.",
        "- Computation: task id and allocated CPU from the scheduler callback.",
        "- UAV mobility: UAV id, speed, angle, phi, and target source.",
        "",
        "## Tensor outputs",
        "",
        f"- `a_hist`: `{tuple(summary['shapes']['a_hist'])}`",
        f"- `a_future`: `{tuple(summary['shapes']['a_future'])}`",
        "",
        "## Key point",
        "",
        "These tensors are now aligned with `dataset_multiseed_v0`, so future models can use historical states plus action variables and predict future node/link/task labels.",
        "",
        "## Per-seed action counts",
        "",
        "| seed | offload | return | rb | cpu | uav |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["runs"]:
        lines.append(
            "| {seed} | {offload_actions} | {return_actions} | {rb_actions} | {cpu_actions} | {uav_mobility_actions} |".format(
                **item
            )
        )
    path = output_dir / "strict_action_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_seed(seed, args.max_time, args.output_dir) for seed in args.seeds]
    all_actions, npz_path, a_hist, a_future = build_aligned_tensors(args.output_dir, args.dataset_dir, summaries)
    figures = plot_outputs(args.output_dir, all_actions)
    summary = {
        "seeds": args.seeds,
        "max_time": args.max_time,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "action_features": ACTION_FEATURES,
        "shapes": {
            "a_hist": list(a_hist.shape),
            "a_future": list(a_future.shape),
        },
        "runs": summaries,
        "outputs": {
            "npz": str(npz_path),
            "timeseries_csv": str(args.output_dir / "strict_action_timeseries.csv"),
            "figures": figures,
        },
    }
    report_path = write_report(args.output_dir, summary)
    summary["outputs"]["report_md"] = str(report_path)
    (args.output_dir / "strict_action_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
