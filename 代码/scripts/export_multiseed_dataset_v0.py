import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler
from strict_action_helpers import allocate_cpu_by_assigned_node
from export_dataset_demo import (
    active_link_counts,
    as_float,
    link_rows,
    node_rows,
    task_rows,
)


EXAMPLE_DIR = Path(os.environ.get("PI_JWM_AIRFOGSIM_EXAMPLE_DIR", Path(__file__).resolve().parent))
DEFAULT_OUTPUT_ROOT = EXAMPLE_DIR / "outputs" / "multiseed_raw_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Export AirFogSim node/link/task logs for multiple seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_config(config, max_time):
    config["simulation"]["max_simulation_time"] = max_time
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


def write_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CorrectedCPUAlgorithmModule(BaseAlgorithmModule):
    """Default scheduler with per-node CPU capacity allocation fixed."""

    def scheduleComputing(self, env):
        def alloc_cpu_callback(computing_tasks, **kwargs):
            task_dicts = [task.to_dict() for tasks in computing_tasks.values() for task in tasks]
            allocations, _ = allocate_cpu_by_assigned_node(
                task_dicts,
                lambda node_id: self.entityScheduler.getNodeInfoById(env, node_id),
            )
            return allocations

        self.compScheduler.setComputingCallBack(env, alloc_cpu_callback)


def run_seed(seed, max_time, output_root):
    np.random.seed(seed)
    random.seed(seed)
    config = make_config(load_config(EXAMPLE_DIR / "config.yaml"), max_time=max_time)
    run_dir = output_root / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm_module = CorrectedCPUAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")

    accumulated_reward = 0.0
    node_data = []
    link_data = []
    task_data = []
    v2u_rate = []
    v2i_rate = []
    u2i_rate = []

    try:
        while not env.isDone():
            algorithm_module.scheduleStep(env)
            active_links = active_link_counts(env)
            env.step()

            accumulated_reward += algorithm_module.getRewardByTask(env)
            done_task_num = TaskScheduler.getDoneTaskNum(env)
            failed_task_num = TaskScheduler.getOutOfDDLTasks(env)
            success_ratio = done_task_num / max(1, done_task_num + failed_task_num)

            for row in node_rows(env):
                row["seed"] = seed
                node_data.append(row)
            for row in link_rows(env, active_links):
                row["seed"] = seed
                link_data.append(row)
            for row in task_rows(env):
                row["seed"] = seed
                task_data.append(row)

            v2u_rate.append(as_float(env.getChannelAvgRate("V2U")))
            v2i_rate.append(as_float(env.getChannelAvgRate("V2I")))
            u2i_rate.append(as_float(env.getChannelAvgRate("U2I")))

            print(
                "seed={} time={:.1f} success={:.2f} done={} failed={} V2U={:.2f} V2I={:.2f} U2I={:.2f}".format(
                    seed,
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
    finally:
        env.close()
        print()

    write_rows(
        run_dir / "node_states.csv",
        ["seed", "time", "node_id", "node_type", "x", "y", "z", "speed", "acceleration", "cpu", "storage", "task_profile"],
        node_data,
    )
    write_rows(
        run_dir / "link_states.csv",
        [
            "seed",
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
            "seed",
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
    plt.savefig(run_dir / "rate.png", dpi=220)
    plt.close()

    summary = {
        "seed": seed,
        "max_simulation_time": max_time,
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
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        summaries.append(run_seed(seed, args.max_time, args.output_root))
    summary = {
        "seeds": args.seeds,
        "max_time": args.max_time,
        "runs": summaries,
    }
    (args.output_root / "multiseed_raw_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
