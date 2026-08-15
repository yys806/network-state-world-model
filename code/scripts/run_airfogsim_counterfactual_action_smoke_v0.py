import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_runtime import make_diagnostic_config, resolve_airfogsim_paths

AIRFOGSIM_ROOT, AIRFOGSIM_EXAMPLES = resolve_airfogsim_paths(ROOT)


OUTPUT_DIR = ROOT / "reports" / "airfogsim_counterfactual_action_smoke_v0"


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_rank_correlation(true_utility, predicted_utility):
    true_rank = average_ranks(true_utility)
    pred_rank = average_ranks(predicted_utility)
    if true_rank.size != pred_rank.size:
        raise ValueError("true_utility and predicted_utility must have the same size")
    if true_rank.size < 2 or np.std(true_rank) < 1e-12 or np.std(pred_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def topk_indices(values, k):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), values.size)
    return set(np.argsort(values)[-k:])


def topk_hit_rate(true_utility, predicted_utility, k):
    true_top = topk_indices(true_utility, k)
    pred_top = topk_indices(predicted_utility, k)
    if not true_top:
        return float("nan")
    return float(len(true_top & pred_top) / len(true_top))


def ranking_regret(true_utility, predicted_utility, top_k=1):
    true = np.asarray(true_utility, dtype=np.float64).reshape(-1)
    pred = np.asarray(predicted_utility, dtype=np.float64).reshape(-1)
    if true.size != pred.size:
        raise ValueError("true_utility and predicted_utility must have the same size")
    pred_top1 = int(np.argmax(pred))
    true_best = float(np.max(true))
    top1_regret = true_best - float(true[pred_top1])
    pred_topk = list(topk_indices(pred, top_k))
    topk_best_regret = true_best - float(np.max(true[pred_topk]))
    denom = float(np.max(true) - np.min(true))
    return {
        "top1_regret": float(top1_regret),
        "normalized_top1_regret": float(top1_regret / denom) if denom >= 1e-12 else 0.0,
        "topk_best_regret": float(topk_best_regret),
        "normalized_topk_best_regret": float(topk_best_regret / denom) if denom >= 1e-12 else 0.0,
    }


def import_airfogsim_runtime():
    if str(AIRFOGSIM_ROOT) not in sys.path:
        sys.path.insert(0, str(AIRFOGSIM_ROOT))
    if str(AIRFOGSIM_EXAMPLES) not in sys.path:
        sys.path.insert(0, str(AIRFOGSIM_EXAMPLES))
    if "airfogsim.airfogsim_visual" not in sys.modules:
        import types

        visual_stub = types.ModuleType("airfogsim.airfogsim_visual")

        class AirFogSimEnvVisualizer:
            def __init__(self, *args, **kwargs):
                pass

            def render(self, *args, **kwargs):
                pass

        visual_stub.AirFogSimEnvVisualizer = AirFogSimEnvVisualizer
        sys.modules["airfogsim.airfogsim_visual"] = visual_stub
    if "airfogsim.utils.tk_utils" not in sys.modules:
        import xml.etree.ElementTree as ET
        import types

        tk_stub = types.ModuleType("airfogsim.utils.tk_utils")

        def parse_location_info(file_path):
            root = ET.parse(file_path).getroot()
            location = root.find("location")
            net_offset = tuple(map(float, location.get("netOffset", "0,0").split(",")))
            return (
                location.get("convBoundary", "0,0,1000,1000"),
                location.get("origBoundary", "0,0,1000,1000"),
                location.get("projParameter", ""),
                net_offset,
            )

        tk_stub.parse_location_info = parse_location_info
        sys.modules["airfogsim.utils.tk_utils"] = tk_stub
    from airfogsim import AirFogSimEnv, BaseAlgorithmModule
    from airfogsim.scheduler import RewardScheduler, TaskScheduler
    return AirFogSimEnv, BaseAlgorithmModule, RewardScheduler, TaskScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal AirFogSim counterfactual action injection smoke test.")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--decision-time", type=float, default=2.0)
    parser.add_argument("--auto-find-rb-time", action="store_true", default=True)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=6.0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def compute_counterfactual_utility(start_done, end_done, start_failed, end_failed, throughput):
    delta_done = float(end_done - start_done)
    delta_failed = float(end_failed - start_failed)
    utility = delta_done - delta_failed + 0.01 * float(np.log1p(max(0.0, throughput)))
    return {
        "delta_done": delta_done,
        "delta_failed": delta_failed,
        "throughput": float(throughput),
        "utility": float(utility),
    }


def rb_count_variants(default_count, n_rb):
    default_count = max(1, int(default_count))
    n_rb = max(1, int(n_rb))
    candidates = [1, default_count, default_count * 2, max(1, n_rb // 2)]
    return sorted({min(n_rb, max(1, int(value))) for value in candidates})


def summarize_candidate_ranking(candidate_df, top_k=2):
    if candidate_df.empty:
        return {}
    true = candidate_df["airfogsim_utility"].to_numpy(dtype=float)
    pred = candidate_df["world_model_utility"].to_numpy(dtype=float)
    regret = ranking_regret(true, pred, top_k=top_k)
    best_pred_idx = int(np.argmax(pred))
    return {
        "num_candidates": int(len(candidate_df)),
        "spearman": spearman_rank_correlation(true, pred),
        f"top{top_k}_hit_rate": topk_hit_rate(true, pred, top_k),
        "top1_regret": regret["top1_regret"],
        "normalized_top1_regret": regret["normalized_top1_regret"],
        f"top{top_k}_best_regret": regret["topk_best_regret"],
        f"top{top_k}_normalized_best_regret": regret["normalized_topk_best_regret"],
        "best_world_model_candidate": str(candidate_df.iloc[best_pred_idx]["candidate_id"]),
        "best_airfogsim_candidate": str(candidate_df.iloc[int(np.argmax(true))]["candidate_id"]),
    }


def make_env(seed, max_time):
    AirFogSimEnv, BaseAlgorithmModule, RewardScheduler, _ = import_airfogsim_runtime()
    np.random.seed(seed)
    random.seed(seed)
    config = make_diagnostic_config(load_config(AIRFOGSIM_EXAMPLES / "config.yaml"), max_time=max_time)
    os.chdir(AIRFOGSIM_EXAMPLES)
    env = AirFogSimEnv(config, interactive_mode=None)
    algorithm = BaseAlgorithmModule()
    algorithm.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")
    return env, algorithm


def channel_throughput(env):
    data = getattr(env, "channel", None)
    if not isinstance(data, dict):
        raise TypeError("AirFogSim environment does not expose the global channel statistics")
    return float(data.get("data_size", 0.0))


def step_default_until(env, algorithm, target_time):
    while (not env.isDone()) and env.simulation_time < target_time - 1e-9:
        algorithm.scheduleStep(env)
        env.step()


def current_counts(env):
    _, _, _, TaskScheduler = import_airfogsim_runtime()
    return {
        "done": int(TaskScheduler.getDoneTaskNum(env)),
        "failed": int(TaskScheduler.getOutOfDDLTasks(env)),
        "throughput": channel_throughput(env),
    }


def default_rb_plan(env, algorithm):
    algorithm.scheduleReturning(env)
    algorithm.scheduleOffloading(env)
    algorithm.scheduleCommunication(env)
    algorithm.scheduleComputing(env)
    algorithm.scheduleMission(env)
    algorithm.scheduleTraffic(env)
    return {task_id: list(rbs) for task_id, rbs in env.activated_offloading_tasks_with_RB_Nos.items()}


def build_candidate_plans(default_plan, n_rb, max_candidates):
    if not default_plan:
        return [{"candidate_id": "default_no_rb", "action_family": "default", "rb_scale": 1.0, "rb_plan": {}}]
    task_ids = sorted(default_plan.keys())
    variants = []
    variants.append(("default", 1.0, {task_id: list(rbs) for task_id, rbs in default_plan.items()}))
    first_task = task_ids[0]
    default_count = len(default_plan[first_task])
    for count in rb_count_variants(default_count, n_rb):
        rb_plan = {task_id: list(rbs) for task_id, rbs in default_plan.items()}
        rb_plan[first_task] = list(range(count))
        scale = count / max(1, default_count)
        variants.append((f"rb_first_task_{count}", scale, rb_plan))
    unique = []
    seen = set()
    for candidate_id, scale, rb_plan in variants:
        key = tuple((task_id, tuple(rbs)) for task_id, rbs in sorted(rb_plan.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "candidate_id": candidate_id,
                "action_family": "rb_count",
                "rb_scale": float(scale),
                "rb_plan": rb_plan,
            }
        )
    return unique[: max(1, int(max_candidates))]


def find_first_rb_decision_time(seed, max_time, scan_until):
    env, algorithm = make_env(seed, max_time=max_time)
    try:
        while (not env.isDone()) and env.simulation_time <= scan_until + 1e-9:
            decision_time = float(env.simulation_time)
            plan = default_rb_plan(env, algorithm)
            if plan:
                n_rb = algorithm.commScheduler.getNumberOfRB(env)
                return decision_time, plan, n_rb
            env.step()
        return None, {}, 0
    finally:
        env.close()


def run_candidate(seed, decision_time, horizon, max_time, candidate):
    start = time.perf_counter()
    env, algorithm = make_env(seed, max_time=max(max_time, decision_time + horizon + 1.0))
    try:
        step_default_until(env, algorithm, decision_time)
        algorithm.scheduleReturning(env)
        algorithm.scheduleOffloading(env)
        env.activated_offloading_tasks_with_RB_Nos = {
            task_id: list(rbs) for task_id, rbs in candidate["rb_plan"].items()
        }
        algorithm.scheduleComputing(env)
        algorithm.scheduleMission(env)
        algorithm.scheduleTraffic(env)
        start_counts = current_counts(env)
        start_throughput = channel_throughput(env)
        steps = 0
        while (not env.isDone()) and steps < horizon:
            env.step()
            steps += 1
            if steps < horizon and not env.isDone():
                algorithm.scheduleStep(env)
        end_counts = current_counts(env)
        throughput = max(0.0, end_counts["throughput"] - start_throughput)
        utility = compute_counterfactual_utility(
            start_counts["done"],
            end_counts["done"],
            start_counts["failed"],
            end_counts["failed"],
            throughput,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "seed": int(seed),
            "decision_time": float(decision_time),
            "horizon": int(horizon),
            "candidate_id": candidate["candidate_id"],
            "action_family": candidate["action_family"],
            "rb_scale": float(candidate["rb_scale"]),
            "num_rb_tasks": int(len(candidate["rb_plan"])),
            "total_rb": int(sum(len(rbs) for rbs in candidate["rb_plan"].values())),
            "airfogsim_utility": utility["utility"],
            "delta_done": utility["delta_done"],
            "delta_failed": utility["delta_failed"],
            "throughput": utility["throughput"],
            "runtime_ms": float(elapsed_ms),
        }
    finally:
        env.close()


def world_model_proxy_utility(row):
    return float(row["delta_done"] - row["delta_failed"] + 0.001 * row["total_rb"] + 0.00001 * row["throughput"])


def plot_candidates(candidate_df):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "airfogsim_counterfactual_action_smoke_v0.png"
    plot_df = candidate_df.set_index("candidate_id")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    plot_df[["airfogsim_utility", "world_model_utility"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("utility")
    ax.set_title("AirFogSim counterfactual action smoke")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(summary, candidate_df, ranking):
    lines = [
        "# AirFogSim counterfactual action smoke v0",
        "",
        "## Goal",
        "",
        "This CPU smoke test checks whether a small set of scheduler-action candidates can be injected into AirFogSim and evaluated with decision-facing ranking metrics.",
        "",
        "## Scope",
        "",
        "- Current smoke test changes RB allocation counts for the first active communication task at one decision time.",
        "- It re-runs AirFogSim from the same seed for each candidate instead of relying on a simulator snapshot interface.",
        "- The `world_model_utility` column is a lightweight proxy used only to exercise the ranking pipeline. Training a learned utility/ranking head is the next GPU-suitable stage.",
        "",
        "## Candidate Results",
        "",
        candidate_df.to_markdown(index=False),
        "",
        "## Ranking Summary",
        "",
        pd.DataFrame([ranking]).to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = OUTPUT_DIR / "airfogsim_counterfactual_action_smoke_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_time = float(args.decision_time)
    default_plan = {}
    n_rb = 0
    if args.auto_find_rb_time:
        found_time, found_plan, found_n_rb = find_first_rb_decision_time(args.seed, args.max_time, args.max_time)
        if found_plan:
            decision_time = float(found_time)
            default_plan = found_plan
            n_rb = found_n_rb
    if not default_plan:
        probe_env, probe_algorithm = make_env(args.seed, max_time=max(args.max_time, decision_time + 1.0))
        try:
            step_default_until(probe_env, probe_algorithm, decision_time)
            default_plan = default_rb_plan(probe_env, probe_algorithm)
            n_rb = probe_algorithm.commScheduler.getNumberOfRB(probe_env)
        finally:
            probe_env.close()

    candidates = build_candidate_plans(default_plan, n_rb, args.max_candidates)
    rows = []
    for candidate in candidates:
        rows.append(run_candidate(args.seed, decision_time, args.horizon, args.max_time, candidate))
    candidate_df = pd.DataFrame(rows)
    candidate_df["world_model_utility"] = candidate_df.apply(world_model_proxy_utility, axis=1)
    ranking = summarize_candidate_ranking(candidate_df, top_k=min(2, len(candidate_df)))
    metrics_path = args.output_dir / "airfogsim_counterfactual_action_smoke_v0_candidates.csv"
    candidate_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_path = plot_candidates(candidate_df)
    summary = {
        "output_dir": display_path(args.output_dir),
        "seed": int(args.seed),
        "decision_time": float(decision_time),
        "horizon": int(args.horizon),
        "num_candidates": int(len(candidate_df)),
        "ranking": ranking,
        "outputs": {
            "candidates_csv": display_path(metrics_path),
            "summary_plot": display_path(plot_path),
        },
    }
    report_path = write_report(summary, candidate_df, ranking)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "airfogsim_counterfactual_action_smoke_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
