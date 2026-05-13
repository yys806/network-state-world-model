import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler


EXAMPLE_DIR = Path(__file__).resolve().parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_demo_config(config):
    """Use a short, local, report-friendly scenario without modifying config.yaml."""
    config["simulation"]["max_simulation_time"] = 20
    config["traffic"]["max_n_vehicles"] = 50
    config["traffic"]["max_n_UAVs"] = 2
    config["traffic"]["RSU_positions"] = [
        [100, 100, 0],
        [700, 100, 0],
        [100, 700, 0],
        [700, 700, 0],
    ]
    config["task_profile"]["task_node_gen_poss"] = 0.8
    return config


def as_float(value, default=0.0):
    try:
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def position_of(node):
    x, y, z = node.getPosition()
    return as_float(x), as_float(y), as_float(z)


def node_rows(env):
    rows = []
    groups = [
        ("vehicle", env.vehicles),
        ("uav", env.UAVs),
        ("rsu", env.RSUs),
        ("cloud", env.cloudServers),
    ]
    for node_type, nodes in groups:
        for node_id, node in nodes.items():
            x, y, z = position_of(node)
            fog_profile = getattr(node, "getFogProfile", lambda: {})() or {}
            task_profile = getattr(node, "getTaskProfile", lambda: {})() or {}
            rows.append(
                {
                    "time": round(env.simulation_time, 3),
                    "node_id": node_id,
                    "node_type": node_type,
                    "x": x,
                    "y": y,
                    "z": z,
                    "speed": as_float(node.getSpeed()),
                    "acceleration": as_float(node.getAcceleration()),
                    "cpu": fog_profile.get("cpu", ""),
                    "storage": fog_profile.get("storage", ""),
                    "task_profile": json.dumps(task_profile, ensure_ascii=False),
                }
            )
    return rows


def channel_rate_sum(env, tx_id, rx_id, channel_type):
    try:
        tx_idx = env._getNodeIdxById(tx_id)
        rx_idx = env._getNodeIdxById(rx_id)
        rate = env.channel_manager.getRateByChannelType(tx_idx, rx_idx, channel_type)
        if hasattr(rate, "get"):
            rate = rate.get()
        return as_float(np.sum(rate))
    except Exception:
        return 0.0


def channel_csi_mean(env, tx_id, rx_id):
    try:
        tx_idx = env._getNodeIdxById(tx_id)
        rx_idx = env._getNodeIdxById(rx_id)
        tx_type = env._getNodeTypeById(tx_id)
        rx_type = env._getNodeTypeById(rx_id)
        csi = env.channel_manager.getCSI(tx_idx, rx_idx, tx_type, rx_type)
        if hasattr(csi, "get"):
            csi = csi.get()
        return as_float(np.mean(csi))
    except Exception:
        return 0.0


def active_link_counts(env):
    active = {}
    for task_id, rb_nos in env.activated_offloading_tasks_with_RB_Nos.items():
        task = env.task_manager.getTaskByTaskId(task_id)
        if task is None:
            continue
        route = task.getToOffloadRoute()
        if not route:
            continue
        tx_id = task.getCurrentNodeId()
        rx_id = route[0]
        tx_type = env._getNodeTypeById(tx_id)
        rx_type = env._getNodeTypeById(rx_id)
        if tx_type is None or rx_type is None:
            continue
        key = (tx_id, rx_id, f"{tx_type}2{rx_type}")
        if key not in active:
            active[key] = {"task_count": 0, "rb_count": 0}
        active[key]["task_count"] += 1
        active[key]["rb_count"] += len(rb_nos)
    return active


def link_rows(env, active):
    rows = []
    link_groups = [
        ("V2U", env.vehicles, env.UAVs),
        ("V2I", env.vehicles, env.RSUs),
        ("U2I", env.UAVs, env.RSUs),
    ]
    for link_type, tx_nodes, rx_nodes in link_groups:
        for tx_id in tx_nodes:
            for rx_id in rx_nodes:
                active_info = active.get((tx_id, rx_id, link_type), {})
                rows.append(
                    {
                        "time": round(env.simulation_time, 3),
                        "tx_id": tx_id,
                        "rx_id": rx_id,
                        "link_type": link_type,
                        "distance": as_float(env.getDistanceBetweenNodesById(tx_id, rx_id)),
                        "rate_sum": channel_rate_sum(env, tx_id, rx_id, link_type),
                        "csi_mean": channel_csi_mean(env, tx_id, rx_id),
                        "active_task_count": active_info.get("task_count", 0),
                        "allocated_rb_count": active_info.get("rb_count", 0),
                    }
                )
    return rows


def task_rows(env):
    rows = []
    for task in env.task_manager.getAllTasks():
        rows.append(
            {
                "time": round(env.simulation_time, 3),
                "task_id": task.getTaskId(),
                "task_node_id": task.getTaskNodeId(),
                "current_node_id": task.getCurrentNodeId(),
                "assigned_to": task.getAssignedTo(),
                "task_size": as_float(task.getTaskSize()),
                "task_cpu": as_float(task.getTaskCPU()),
                "deadline": as_float(task.getTaskDeadline()),
                "priority": as_float(task.getTaskPriority()),
                "arrival_time": as_float(task.getTaskArrivalTime()),
                "transmitted_size": as_float(task.getTransmittedSize()),
                "computed_size": as_float(task.getComputedSize()),
                "lifecycle_state": task.task_lifecycle_state,
                "failure_reason": task.getTaskFailureReason(),
            }
        )
    return rows


def write_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    np.random.seed(0)
    random.seed(0)

    config = make_demo_config(load_config(EXAMPLE_DIR / "config.yaml"))
    run_dir = EXAMPLE_DIR / "outputs" / f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm_module = BaseAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")

    accumulated_reward = 0.0
    node_data = []
    link_data = []
    task_data = []
    v2u_rate = []
    v2i_rate = []
    u2i_rate = []

    while not env.isDone():
        algorithm_module.scheduleStep(env)
        active_links = active_link_counts(env)
        env.step()

        accumulated_reward += algorithm_module.getRewardByTask(env)
        done_task_num = TaskScheduler.getDoneTaskNum(env)
        failed_task_num = TaskScheduler.getOutOfDDLTasks(env)
        success_ratio = done_task_num / max(1, done_task_num + failed_task_num)

        node_data.extend(node_rows(env))
        link_data.extend(link_rows(env, active_links))
        task_data.extend(task_rows(env))

        v2u_rate.append(env.getChannelAvgRate("V2U"))
        v2i_rate.append(env.getChannelAvgRate("V2I"))
        u2i_rate.append(env.getChannelAvgRate("U2I"))

        print(
            "time={:.1f} success={:.2f} done={} failed={} V2U={:.2f} V2I={:.2f} U2I={:.2f}".format(
                env.simulation_time,
                success_ratio,
                done_task_num,
                failed_task_num,
                v2u_rate[-1],
                v2i_rate[-1],
                u2i_rate[-1],
            ),
            end="\r",
        )

    env.close()
    print()

    write_rows(
        run_dir / "node_states.csv",
        ["time", "node_id", "node_type", "x", "y", "z", "speed", "acceleration", "cpu", "storage", "task_profile"],
        node_data,
    )
    write_rows(
        run_dir / "link_states.csv",
        [
            "time",
            "tx_id",
            "rx_id",
            "link_type",
            "distance",
            "rate_sum",
            "csi_mean",
            "active_task_count",
            "allocated_rb_count",
        ],
        link_data,
    )
    write_rows(
        run_dir / "task_states.csv",
        [
            "time",
            "task_id",
            "task_node_id",
            "current_node_id",
            "assigned_to",
            "task_size",
            "task_cpu",
            "deadline",
            "priority",
            "arrival_time",
            "transmitted_size",
            "computed_size",
            "lifecycle_state",
            "failure_reason",
        ],
        task_data,
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(v2u_rate, label="V2U")
    plt.plot(v2i_rate, label="V2I")
    plt.plot(u2i_rate, label="U2I")
    plt.xlabel("step")
    plt.ylabel("avg transmission rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "rate.png", dpi=300)
    plt.close()

    summary = {
        "max_simulation_time": config["simulation"]["max_simulation_time"],
        "max_n_vehicles": config["traffic"]["max_n_vehicles"],
        "max_n_UAVs": config["traffic"]["max_n_UAVs"],
        "RSU_positions": config["traffic"]["RSU_positions"],
        "node_rows": len(node_data),
        "link_rows": len(link_data),
        "task_rows": len(task_data),
        "final_v2u_avg_rate": as_float(v2u_rate[-1]) if v2u_rate else 0,
        "final_v2i_avg_rate": as_float(v2i_rate[-1]) if v2i_rate else 0,
        "final_u2i_avg_rate": as_float(u2i_rate[-1]) if u2i_rate else 0,
        "accumulated_reward": as_float(accumulated_reward),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Simulation done. Output dir: {run_dir}")
    print("Generated: node_states.csv, link_states.csv, task_states.csv, rate.png, summary.json")


if __name__ == "__main__":
    main()
