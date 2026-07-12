import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_airfogsim_counterfactual_action_smoke_v0 import (
    ROOT,
    build_candidate_plans,
    channel_throughput,
    compute_counterfactual_utility,
    current_counts,
    display_path,
    make_env,
    rb_count_variants,
    step_default_until,
    world_model_proxy_utility,
)


OUTPUT_DIR = ROOT / "reports" / "airfogsim_counterfactual_multifamily_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AirFogSim offload/RB multi-family counterfactual labels.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--scan-step-limit", type=int, default=80)
    parser.add_argument("--decision-times-per-seed", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-utility-spread", type=float, default=1e-6)
    parser.add_argument("--include-compute-return", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_offload_options(task_infos, neighbor_map):
    options = []
    for task in task_infos:
        task_id = task["task_id"]
        task_node_id = task["task_node_id"]
        neighbors = neighbor_map.get(task_id, [])
        if not neighbors:
            continue
        default_target = neighbors[0]["id"]
        alternative = next((item for item in neighbors[1:] if item["id"] != default_target), None)
        if alternative is None:
            continue
        options.append(
            {
                "task_id": task_id,
                "task_node_id": task_node_id,
                "default_target_id": default_target,
                "alternative_target_id": alternative["id"],
                "default_target_type": neighbors[0].get("node_type", ""),
                "alternative_target_type": alternative.get("node_type", ""),
                "default_distance": float(neighbors[0].get("distance", 0.0)),
                "alternative_distance": float(alternative.get("distance", 0.0)),
            }
        )
    return options


def fill_neighbor_distances(env, source_id, neighbors):
    out = []
    for neighbor in neighbors:
        item = dict(neighbor)
        distance = float(item.get("distance", 0.0) or 0.0)
        if distance <= 0.0 and item.get("id") is not None:
            distance = float(env.getDistanceBetweenNodesById(source_id, item["id"]))
        item["distance"] = distance
        if item.get("id") is not None and "node_type" not in item and hasattr(env, "_getNodeTypeById"):
            item["node_type"] = env._getNodeTypeById(item["id"])
        out.append(item)
    return out


def candidate_from_plan(
    candidate_id,
    action_family,
    rb_scale=1.0,
    rb_plan=None,
    offload_overrides=None,
    cpu_overrides=None,
    return_route_overrides=None,
    cpu_scale=1.0,
    offload_features=None,
):
    rb_plan = rb_plan or {}
    offload_overrides = offload_overrides or {}
    cpu_overrides = cpu_overrides or {}
    return_route_overrides = return_route_overrides or {}
    offload_features = offload_features or {}
    return {
        "candidate_id": candidate_id,
        "action_family": action_family,
        "rb_scale": float(rb_scale),
        "cpu_scale": float(cpu_scale),
        "rb_plan": {task_id: list(rbs) for task_id, rbs in rb_plan.items()},
        "offload_overrides": dict(offload_overrides),
        "cpu_overrides": {task_id: float(cpu) for task_id, cpu in cpu_overrides.items()},
        "return_route_overrides": {task_id: list(route) for task_id, route in return_route_overrides.items()},
        "num_offload_overrides": int(len(offload_overrides)),
        "num_cpu_overrides": int(len(cpu_overrides)),
        "num_return_route_overrides": int(len(return_route_overrides)),
        "total_rb": int(sum(len(rbs) for rbs in rb_plan.values())),
        "total_cpu": float(sum(cpu_overrides.values())),
        "offload_default_distance": float(offload_features.get("default_distance", 0.0)),
        "offload_alternative_distance": float(offload_features.get("alternative_distance", 0.0)),
        "offload_distance_delta": float(offload_features.get("distance_delta", 0.0)),
        "offload_distance_ratio": float(offload_features.get("distance_ratio", 0.0)),
        "offload_default_is_vehicle": float(offload_features.get("default_is_vehicle", 0.0)),
        "offload_default_is_uav": float(offload_features.get("default_is_uav", 0.0)),
        "offload_default_is_rsu": float(offload_features.get("default_is_rsu", 0.0)),
        "offload_default_is_cloud": float(offload_features.get("default_is_cloud", 0.0)),
        "offload_alternative_is_vehicle": float(offload_features.get("alternative_is_vehicle", 0.0)),
        "offload_alternative_is_uav": float(offload_features.get("alternative_is_uav", 0.0)),
        "offload_alternative_is_rsu": float(offload_features.get("alternative_is_rsu", 0.0)),
        "offload_alternative_is_cloud": float(offload_features.get("alternative_is_cloud", 0.0)),
        "offload_target_type_changed": float(offload_features.get("target_type_changed", 0.0)),
    }


def type_flags(prefix, node_type):
    node_type = str(node_type or "").upper()
    return {
        f"{prefix}_is_vehicle": 1.0 if node_type == "V" else 0.0,
        f"{prefix}_is_uav": 1.0 if node_type == "U" else 0.0,
        f"{prefix}_is_rsu": 1.0 if node_type == "I" else 0.0,
        f"{prefix}_is_cloud": 1.0 if node_type == "C" else 0.0,
    }


def offload_features_from_option(option):
    default_distance = float(option.get("default_distance", 0.0))
    alternative_distance = float(option.get("alternative_distance", 0.0))
    default_type = option.get("default_target_type", "")
    alternative_type = option.get("alternative_target_type", "")
    out = {
        "default_distance": default_distance,
        "alternative_distance": alternative_distance,
        "distance_delta": alternative_distance - default_distance,
        "distance_ratio": alternative_distance / max(default_distance, 1e-6),
        "target_type_changed": 1.0 if str(default_type).upper() != str(alternative_type).upper() else 0.0,
    }
    out.update(type_flags("default", default_type))
    out.update(type_flags("alternative", alternative_type))
    return out


def build_cpu_candidates(computing_tasks, max_candidates):
    out = []
    if not computing_tasks:
        return out
    scales = [0.5, 1.5]
    for scale in scales:
        overrides = {}
        for task in computing_tasks:
            task_id = task["task_id"]
            base_cpu = float(task.get("base_cpu", 0.0))
            if base_cpu <= 0:
                continue
            overrides[task_id] = base_cpu * scale
        if not overrides:
            continue
        out.append(
            candidate_from_plan(
                f"cpu_scale_{scale:g}",
                "cpu_scale",
                cpu_overrides=overrides,
                cpu_scale=scale,
            )
        )
    return out[: max(1, int(max_candidates))]


def build_return_route_candidates(waiting_tasks, max_candidates):
    out = []
    for task in waiting_tasks:
        task_id = task["task_id"]
        for route in task.get("direct_routes", []):
            if not route:
                continue
            route_tag = "_".join(str(item) for item in route)
            out.append(
                candidate_from_plan(
                    f"return_{task_id}_direct_{route_tag}",
                    "return_route",
                    return_route_overrides={task_id: route},
                )
            )
        for route in task.get("relay_routes", []):
            if not route:
                continue
            route_tag = "_".join(str(item) for item in route)
            out.append(
                candidate_from_plan(
                    f"return_{task_id}_relay_{route_tag}",
                    "return_route",
                    return_route_overrides={task_id: route},
                )
            )
    return out[: max(1, int(max_candidates))]


def build_multifamily_candidates(offload_options, default_rb_plan, n_rb, max_candidates):
    rb_candidates = build_candidate_plans(default_rb_plan, n_rb, max_candidates=max_candidates)
    out = []
    seen = set()

    def add(candidate):
        key = (
            candidate["action_family"],
            tuple(sorted(candidate.get("offload_overrides", {}).items())),
            tuple(sorted(candidate.get("cpu_overrides", {}).items())),
            tuple((task_id, tuple(route)) for task_id, route in sorted(candidate.get("return_route_overrides", {}).items())),
            tuple((task_id, tuple(rbs)) for task_id, rbs in sorted(candidate["rb_plan"].items())),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(candidate)

    for rb_candidate in rb_candidates:
        add(
            candidate_from_plan(
                rb_candidate["candidate_id"],
                "rb_count",
                rb_candidate["rb_scale"],
                rb_candidate["rb_plan"],
            )
        )
    if offload_options and default_rb_plan:
        base_plan = {task_id: list(rbs) for task_id, rbs in default_rb_plan.items()}
        first = offload_options[0]
        override = {first["task_id"]: first["alternative_target_id"]}
        offload_features = offload_features_from_option(first)
        add(candidate_from_plan(f"offload_{first['task_id']}_alt", "offload_target", 1.0, base_plan, override, offload_features=offload_features))
        task_id = first["task_id"]
        default_count = len(base_plan.get(task_id, [])) or max(1, n_rb // max(1, len(base_plan)))
        for count in rb_count_variants(default_count, n_rb):
            if count == default_count:
                continue
            rb_plan = {key: list(value) for key, value in base_plan.items()}
            rb_plan[task_id] = list(range(count))
            add(
                candidate_from_plan(
                    f"mixed_{task_id}_alt_rb_{count}",
                    "mixed_offload_rb",
                    count / max(1, default_count),
                    rb_plan,
                    override,
                    offload_features=offload_features,
                )
            )
    return out[: max(1, int(max_candidates))]


def build_extended_candidates(point, max_candidates):
    out = []
    seen = set()

    def add_many(candidates):
        for candidate in candidates:
            key = (
                candidate["action_family"],
                candidate["candidate_id"],
                tuple(sorted(candidate.get("offload_overrides", {}).items())),
                tuple(sorted(candidate.get("cpu_overrides", {}).items())),
                tuple((task_id, tuple(route)) for task_id, route in sorted(candidate.get("return_route_overrides", {}).items())),
                tuple((task_id, tuple(rbs)) for task_id, rbs in sorted(candidate.get("rb_plan", {}).items())),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)

    add_many(
        build_multifamily_candidates(
            point.get("offload_options", []),
            point.get("default_rb_plan", {}),
            point.get("n_rb", 1),
            max_candidates=max_candidates,
        )
    )
    add_many(build_cpu_candidates(point.get("computing_tasks", []), max_candidates=max_candidates))
    add_many(build_return_route_candidates(point.get("return_route_tasks", []), max_candidates=max_candidates))
    return out[: max(1, int(max_candidates))]


def collect_decision_context(env, algorithm):
    algorithm.scheduleReturning(env)
    task_infos = algorithm.taskScheduler.getAllToOffloadTaskInfos(env)
    neighbor_map = {}
    for task in task_infos:
        neighbors = algorithm.entityScheduler.getNeighborNodeInfosById(
            env, task["task_node_id"], sorted_by="distance", max_num=5
        )
        neighbors = fill_neighbor_distances(env, task["task_node_id"], neighbors)
        neighbor_map[task["task_id"]] = neighbors
    offload_options = build_offload_options(task_infos, neighbor_map)
    for task in task_infos:
        neighbors = neighbor_map.get(task["task_id"], [])
        if neighbors:
            algorithm.taskScheduler.setTaskOffloading(env, task["task_node_id"], task["task_id"], neighbors[0]["id"])
    algorithm.scheduleCommunication(env)
    algorithm.scheduleComputing(env)
    algorithm.scheduleMission(env)
    algorithm.scheduleTraffic(env)
    default_rb_plan = {task_id: list(rbs) for task_id, rbs in env.activated_offloading_tasks_with_RB_Nos.items()}
    n_rb = algorithm.commScheduler.getNumberOfRB(env)
    return {
        "decision_stage": "offload_rb",
        "offload_options": offload_options,
        "default_rb_plan": default_rb_plan,
        "n_rb": int(n_rb),
        "num_to_offload_tasks": int(len(task_infos)),
    }


def collect_computing_tasks(env):
    algorithm_cpu = {}
    for node_id, tasks in env.task_manager.getComputingTasks().items():
        if not tasks:
            continue
        node_info = env._getNodeById(node_id).to_dict()
        node_cpu = float(node_info.get("fog_profile", {}).get("cpu", 0.0))
        base_cpu = node_cpu / max(1, len(tasks))
        for task in tasks:
            algorithm_cpu[task.getTaskId()] = {
                "task_id": task.getTaskId(),
                "assigned_to": node_id,
                "base_cpu": base_cpu,
            }
    return list(algorithm_cpu.values())


def sorted_node_ids_by_distance(env, source_id, node_ids):
    rows = []
    for node_id in node_ids:
        distance = env.getDistanceBetweenNodesById(source_id, node_id)
        rows.append((node_id, float(distance)))
    return [node_id for node_id, _ in sorted(rows, key=lambda item: item[1])]


def collect_return_route_tasks(env):
    out = []
    rsu_ids = list(env.RSUs.keys())
    uav_ids = list(env.UAVs.keys())
    for _, tasks in env.task_manager.getWaitingToReturnTaskInfos().items():
        for task in tasks:
            current_node_id = task.getCurrentNodeId()
            current_node_type = env._getNodeTypeById(current_node_id)
            direct_routes = []
            relay_routes = []
            if rsu_ids:
                sorted_rsus = sorted_node_ids_by_distance(env, current_node_id, rsu_ids)
                direct_routes.append([sorted_rsus[0]])
                if len(sorted_rsus) > 1:
                    direct_routes.append([sorted_rsus[-1]])
            if current_node_type == "V" and uav_ids and rsu_ids:
                sorted_uavs = sorted_node_ids_by_distance(env, current_node_id, uav_ids)
                sorted_rsus = sorted_node_ids_by_distance(env, sorted_uavs[0], rsu_ids)
                relay_routes.append([sorted_uavs[0], sorted_rsus[0]])
            if direct_routes or relay_routes:
                out.append(
                    {
                        "task_id": task.getTaskId(),
                        "current_node_id": current_node_id,
                        "direct_routes": direct_routes,
                        "relay_routes": relay_routes,
                    }
                )
    return out


def collect_extended_context(env, algorithm):
    offload_context = collect_decision_context(env, algorithm)
    computing_tasks = collect_computing_tasks(env)
    return_tasks = collect_return_route_tasks(env)
    offload_context.update(
        {
            "computing_tasks": computing_tasks,
            "return_route_tasks": return_tasks,
            "num_computing_tasks": int(len(computing_tasks)),
            "num_waiting_return_tasks": int(len(return_tasks)),
        }
    )
    if return_tasks:
        offload_context["decision_stage"] = "return_route"
    elif computing_tasks:
        offload_context["decision_stage"] = "compute"
    return offload_context


def discover_decision_points(seed, max_time, max_points, scan_step_limit, include_compute_return=False):
    env, algorithm = make_env(seed, max_time=max_time)
    points = []
    stage_counts = {"offload_rb": 0, "compute": 0, "return_route": 0}
    try:
        steps = 0
        while (not env.isDone()) and steps < scan_step_limit:
            decision_time = float(env.simulation_time)
            context = collect_extended_context(env, algorithm) if include_compute_return else collect_decision_context(env, algorithm)
            has_offload_rb = bool(context["default_rb_plan"] and context["offload_options"])
            has_compute = bool(include_compute_return and context.get("computing_tasks"))
            has_return = bool(include_compute_return and context.get("return_route_tasks"))
            if include_compute_return:
                stage = context.get("decision_stage", "offload_rb")
                if (has_offload_rb or has_compute or has_return) and stage_counts.get(stage, 0) < max_points:
                    points.append({"seed": seed, "decision_time": decision_time, **context})
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                if all(stage_counts.get(stage, 0) >= max_points for stage in stage_counts):
                    break
            elif has_offload_rb:
                points.append({"seed": seed, "decision_time": decision_time, **context})
                if len(points) >= max_points:
                    break
            env.step()
            steps += 1
    finally:
        env.close()
    return points


def apply_offload_overrides(env, algorithm, candidate):
    applied = 0
    for task_id, target_id in candidate.get("offload_overrides", {}).items():
        task = env.task_manager.getTaskByTaskId(task_id)
        if task is None:
            continue
        task.changeOffloadTo(target_id, [target_id], env.simulation_time)
        applied += 1
    return applied


def apply_cpu_overrides(env, candidate):
    overrides = {task_id: float(cpu) for task_id, cpu in candidate.get("cpu_overrides", {}).items()}
    if not overrides:
        return 0

    def callback(computing_tasks, **kwargs):
        return dict(overrides)

    env.alloc_cpu_callback = callback
    return len(overrides)


def apply_return_route_overrides(env, algorithm, candidate):
    applied = 0
    for task_id, route in candidate.get("return_route_overrides", {}).items():
        task = env.task_manager.getTaskByTaskId(task_id)
        if task is None:
            for _, tasks in env.task_manager.getWaitingToReturnTaskInfos().items():
                task = next((item for item in tasks if item.getTaskId() == task_id), None)
                if task is not None:
                    break
        if task is None:
            continue
        algorithm.taskScheduler.setTaskReturnRoute(env, task_id, list(route))
        applied += 1
    return applied


def run_multifamily_candidate(seed, decision_time, horizon, max_time, candidate):
    start = time.perf_counter()
    env, algorithm = make_env(seed, max_time=max(max_time, decision_time + horizon + 1.0))
    try:
        step_default_until(env, algorithm, decision_time)
        context = collect_extended_context(env, algorithm)
        applied_offload_overrides = apply_offload_overrides(env, algorithm, candidate)
        applied_cpu_overrides = apply_cpu_overrides(env, candidate)
        applied_return_route_overrides = apply_return_route_overrides(env, algorithm, candidate)
        env.activated_offloading_tasks_with_RB_Nos = {
            task_id: list(rbs) for task_id, rbs in candidate["rb_plan"].items()
        }
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
        return {
            "seed": int(seed),
            "decision_time": float(decision_time),
            "horizon": int(horizon),
            "candidate_id": candidate["candidate_id"],
            "action_family": candidate["action_family"],
            "rb_scale": float(candidate["rb_scale"]),
            "cpu_scale": float(candidate.get("cpu_scale", 1.0)),
            "num_rb_tasks": int(len(candidate["rb_plan"])),
            "total_rb": int(candidate["total_rb"]),
            "total_cpu": float(candidate.get("total_cpu", 0.0)),
            "num_offload_overrides": int(candidate.get("num_offload_overrides", 0)),
            "applied_offload_overrides": int(applied_offload_overrides),
            "num_cpu_overrides": int(candidate.get("num_cpu_overrides", 0)),
            "applied_cpu_overrides": int(applied_cpu_overrides),
            "num_return_route_overrides": int(candidate.get("num_return_route_overrides", 0)),
            "applied_return_route_overrides": int(applied_return_route_overrides),
            "offload_default_distance": float(candidate.get("offload_default_distance", 0.0)),
            "offload_alternative_distance": float(candidate.get("offload_alternative_distance", 0.0)),
            "offload_distance_delta": float(candidate.get("offload_distance_delta", 0.0)),
            "offload_distance_ratio": float(candidate.get("offload_distance_ratio", 0.0)),
            "offload_default_is_vehicle": float(candidate.get("offload_default_is_vehicle", 0.0)),
            "offload_default_is_uav": float(candidate.get("offload_default_is_uav", 0.0)),
            "offload_default_is_rsu": float(candidate.get("offload_default_is_rsu", 0.0)),
            "offload_default_is_cloud": float(candidate.get("offload_default_is_cloud", 0.0)),
            "offload_alternative_is_vehicle": float(candidate.get("offload_alternative_is_vehicle", 0.0)),
            "offload_alternative_is_uav": float(candidate.get("offload_alternative_is_uav", 0.0)),
            "offload_alternative_is_rsu": float(candidate.get("offload_alternative_is_rsu", 0.0)),
            "offload_alternative_is_cloud": float(candidate.get("offload_alternative_is_cloud", 0.0)),
            "offload_target_type_changed": float(candidate.get("offload_target_type_changed", 0.0)),
            "airfogsim_utility": utility["utility"],
            "delta_done": utility["delta_done"],
            "delta_failed": utility["delta_failed"],
            "throughput": utility["throughput"],
            "runtime_ms": float((time.perf_counter() - start) * 1000.0),
            "context_num_to_offload_tasks": int(context["num_to_offload_tasks"]),
            "context_num_computing_tasks": int(context.get("num_computing_tasks", 0)),
            "context_num_waiting_return_tasks": int(context.get("num_waiting_return_tasks", 0)),
        }
    finally:
        env.close()


def summarize_label_dataset(df):
    if df.empty:
        return {"num_rows": 0, "num_decision_groups": 0}
    groups = df.groupby(["seed", "decision_time"], dropna=False)
    return {
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed"].nunique()),
        "num_decision_groups": int(groups.ngroups),
        "mean_candidates_per_group": float(groups.size().mean()),
        "action_families": sorted(df["action_family"].dropna().unique().tolist()),
        "utility_min": float(df["airfogsim_utility"].min()),
        "utility_max": float(df["airfogsim_utility"].max()),
    }


def summarize_group_utility_spread(df, utility_col="airfogsim_utility", min_utility_spread=1e-6):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "decision_group_id",
                "seed",
                "decision_time",
                "num_candidates",
                "utility_min",
                "utility_max",
                "utility_spread",
                "is_nontrivial",
            ]
        )
    if utility_col not in df.columns:
        raise KeyError(f"{utility_col} is required for utility-spread diagnostics")

    group_col = "decision_group_id" if "decision_group_id" in df.columns else ["seed", "decision_time"]
    rows = []
    for group_key, part in df.groupby(group_col, dropna=False):
        values = part[utility_col].to_numpy(dtype=np.float64)
        utility_min = float(np.nanmin(values)) if values.size else 0.0
        utility_max = float(np.nanmax(values)) if values.size else 0.0
        spread = utility_max - utility_min
        first = part.iloc[0]
        decision_group_id = str(group_key)
        if isinstance(group_key, tuple):
            decision_group_id = f"seed{int(first['seed'])}_t{float(first['decision_time']):.3f}"
        rows.append(
            {
                "decision_group_id": decision_group_id,
                "seed": int(first["seed"]) if "seed" in part.columns else -1,
                "decision_time": float(first["decision_time"]) if "decision_time" in part.columns else np.nan,
                "num_candidates": int(len(part)),
                "utility_min": utility_min,
                "utility_max": utility_max,
                "utility_spread": float(spread),
                "is_nontrivial": bool(spread > float(min_utility_spread)),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "decision_time", "decision_group_id"]).reset_index(drop=True)


def generate_label_rows(seeds, max_time, decision_times_per_seed, horizon, max_candidates, scan_step_limit, include_compute_return=False):
    rows = []
    point_rows = []
    for seed in seeds:
        points = discover_decision_points(seed, max_time, decision_times_per_seed, scan_step_limit, include_compute_return)
        for point_idx, point in enumerate(points):
            candidates = (
                build_extended_candidates(point, max_candidates=max_candidates)
                if include_compute_return
                else build_multifamily_candidates(
                    point["offload_options"],
                    point["default_rb_plan"],
                    point["n_rb"],
                    max_candidates=max_candidates,
                )
            )
            point_rows.append(
                {
                    "seed": int(seed),
                    "decision_time": float(point["decision_time"]),
                    "num_candidates": int(len(candidates)),
                    "point_index": int(point_idx),
                    "decision_stage": str(point.get("decision_stage", "offload_rb")),
                    "num_offload_options": int(len(point["offload_options"])),
                    "num_computing_tasks": int(point.get("num_computing_tasks", 0)),
                    "num_waiting_return_tasks": int(point.get("num_waiting_return_tasks", 0)),
                }
            )
            for candidate in candidates:
                row = run_multifamily_candidate(seed, point["decision_time"], horizon, max_time, candidate)
                row["decision_group_id"] = f"seed{seed}_t{point['decision_time']:.3f}"
                row["point_index"] = int(point_idx)
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def write_report(summary, labels_df, points_df, group_spread_df, output_dir):
    lines = [
        "# AirFogSim counterfactual multi-family v0",
        "",
        "## Goal",
        "",
        "Generate offload/RB mixed counterfactual labels so v5 ranking is not limited to RB-count heuristics.",
        "",
        "## Summary",
        "",
        pd.DataFrame([summary["dataset"]]).to_markdown(index=False),
        "",
        "## Decision Points",
        "",
        points_df.to_markdown(index=False) if not points_df.empty else "No decision points found.",
        "",
        "## Utility Spread By Decision Group",
        "",
        group_spread_df.to_markdown(index=False) if not group_spread_df.empty else "No decision groups found.",
        "",
        "## Label Preview",
        "",
        labels_df.head(30).to_markdown(index=False) if not labels_df.empty else "No labels generated.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "airfogsim_counterfactual_multifamily_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_df, points_df = generate_label_rows(
        args.seeds,
        args.max_time,
        args.decision_times_per_seed,
        args.horizon,
        args.max_candidates,
        args.scan_step_limit,
        args.include_compute_return,
    )
    if not labels_df.empty:
        labels_df["world_model_utility"] = labels_df.apply(world_model_proxy_utility, axis=1)
    labels_path = args.output_dir / "airfogsim_counterfactual_multifamily_v0_labels.csv"
    points_path = args.output_dir / "airfogsim_counterfactual_multifamily_v0_points.csv"
    labels_df.to_csv(labels_path, index=False, encoding="utf-8-sig")
    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    summary = {
        "seeds": [int(seed) for seed in args.seeds],
        "max_time": float(args.max_time),
        "horizon": int(args.horizon),
        "max_candidates": int(args.max_candidates),
        "decision_times_per_seed": int(args.decision_times_per_seed),
        "min_utility_spread": float(args.min_utility_spread),
        "include_compute_return": bool(args.include_compute_return),
        "dataset": summarize_label_dataset(labels_df),
        "outputs": {
            "labels_csv": display_path(labels_path),
            "points_csv": display_path(points_path),
        },
    }
    group_spread_df = summarize_group_utility_spread(
        labels_df,
        utility_col="airfogsim_utility",
        min_utility_spread=args.min_utility_spread,
    )
    group_spread_path = args.output_dir / "airfogsim_counterfactual_multifamily_v0_group_spread.csv"
    group_spread_df.to_csv(group_spread_path, index=False, encoding="utf-8-sig")
    if not group_spread_df.empty:
        summary["dataset"]["num_nontrivial_groups"] = int(group_spread_df["is_nontrivial"].sum())
        summary["dataset"]["nontrivial_group_ratio"] = float(group_spread_df["is_nontrivial"].mean())
        summary["dataset"]["mean_utility_spread"] = float(group_spread_df["utility_spread"].mean())
        summary["dataset"]["median_utility_spread"] = float(group_spread_df["utility_spread"].median())
    summary["outputs"]["group_spread_csv"] = display_path(group_spread_path)
    report_path = write_report(summary, labels_df, points_df, group_spread_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "airfogsim_counterfactual_multifamily_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
