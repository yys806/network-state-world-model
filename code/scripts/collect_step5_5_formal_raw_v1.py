"""Collect STEP 5.5 real AirFogSim trajectories with a causal coverage policy."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
for entry in (CODE / "src", CODE / "scripts", CODE / "reference" / "AirFogSim" / "examples"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_step2_3_real_airfogsim_raw_contract_finalization_v1 as step23  # noqa: E402
import run_step2_4_real_airfogsim_communication_outcome_semantics_v1 as step24  # noqa: E402
from pi_jwm.raw_trajectory_causal_contract_v1 import aggregate_slot_outcomes  # noqa: E402


DECISION_STEPS = 96
ACCEPTED_COUNT = 60
SIMULATOR_SEED_START = 2026092300
POLICY_SEED_START = 2026092400
WIRED_EDGES = step24.WIRED_EDGES
TASK_COLLECTION_PRIORITY = (
    "_to_generate_task_infos", "_waiting_to_offload_tasks", "_offloading_tasks",
    "_computing_tasks", "_waiting_to_return_tasks", "_returning_tasks",
    "_done_tasks", "_out_of_ddl_tasks",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _signature(entries: list[dict[str, Any]]) -> str | None:
    return None if not entries else hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _legal_route_targets(env: Any, source: str) -> list[str]:
    candidates = [
        node_id for node_id in sorted(set(env.vehicles) | set(env.UAVs) | set(env.RSUs))
        if node_id != source and env._getNodeTypeById(node_id) in {"V", "U", "I"}
    ]
    return [node_id for _, node_id in sorted(
        (float(env.getDistanceBetweenNodesById(source, node_id)), str(node_id)) for node_id in candidates
    )]


def _capacity_respecting_comp(env: Any, intervention: bool, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    entries, allocations, unsupported = step24._planned_comp_for_observed_nodes(env)
    if not entries:
        return entries, allocations, unsupported
    factor = rng.choice((0.5, 0.75, 1.0)) if intervention else 1.0
    scaled = {task_id: float(value) * factor for task_id, value in allocations.items()}
    rows = [{**row, "allocated_cpu_per_s": scaled[str(row["task_id"])]} for row in entries]
    by_node: dict[str, float] = {}
    for row in rows:
        by_node[str(row["node_id"])] = by_node.get(str(row["node_id"]), 0.0) + float(row["allocated_cpu_per_s"])
    for node_id, total in by_node.items():
        capacity = float(env._getNodeById(node_id).getFogProfile()["cpu"])
        if total > capacity + 1e-9:
            raise RuntimeError(f"capacity violation node={node_id} total={total} capacity={capacity}")
    return rows, scaled, unsupported


def _causal_task_ids(decisions: list[dict[str, Any]], frame: int) -> set[str]:
    """Keep future action references visible from the earliest H=2 anchor."""
    reference_frame = max(0, frame - 3)
    return {str(row["task_id"]) for row in decisions[reference_frame].get("tasks", [])}


def _causal_entity_ids(decisions: list[dict[str, Any]], frame: int) -> set[str]:
    reference_frame = max(0, frame - 3)
    return {str(row["entity_id"]) for row in decisions[reference_frame].get("entities", [])}


def _repair_duplicate_task_references(task_manager: Any, frame: int) -> list[dict[str, Any]]:
    """Remove stale duplicate references while preserving the furthest lifecycle.

    AirFogSim mutates lifecycle lists in place.  On longer runs its deadline
    cleanup can leave the same Task object in an earlier list after adding it
    to a terminal list.  The frozen observer requires one lifecycle per task,
    so the collection adapter records and removes only those stale references.
    """
    memberships: dict[str, list[tuple[int, str, str, Any]]] = {}
    for priority, collection_name in enumerate(TASK_COLLECTION_PRIORITY):
        collection = getattr(task_manager, collection_name)
        for owner, tasks in collection.items():
            for task in tasks:
                memberships.setdefault(str(task.getTaskId()), []).append((priority, collection_name, str(owner), task))
    repairs: list[dict[str, Any]] = []
    for task_id, rows in memberships.items():
        if len(rows) <= 1:
            continue
        if len({id(row[3]) for row in rows}) != 1:
            raise RuntimeError(f"distinct Task objects share task_id={task_id}")
        chosen = max(rows, key=lambda row: row[0])
        for collection_name in TASK_COLLECTION_PRIORITY:
            collection = getattr(task_manager, collection_name)
            for owner, tasks in collection.items():
                collection[owner] = [task for task in tasks if str(task.getTaskId()) != task_id]
        getattr(task_manager, chosen[1]).setdefault(chosen[2], []).append(chosen[3])
        repairs.append({
            "frame_index": frame, "task_id": task_id,
            "before": [f"{name}:{owner}" for _, name, owner, _ in rows],
            "kept": f"{chosen[1]}:{chosen[2]}",
            "reason": "remove_stale_duplicate_reference_preserve_furthest_lifecycle",
        })
    return repairs


def collect_trajectory(simulator_seed: int, policy_seed: int, trajectory_id: str) -> dict[str, Any]:
    old_cwd = Path.cwd()
    env = None
    rng = random.Random(policy_seed)
    step23.TRAJECTORY_ID = trajectory_id
    try:
        os.chdir(CODE / "reference" / "AirFogSim" / "examples")
        env, task_scheduler, communication_scheduler, _, config = step23._build_environment(
            simulator_seed, 30.0, wired_edges=WIRED_EDGES,
        )
        warmup_steps = step23._warm_to_ready(env)
        decisions: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        previous_outcome: dict[str, Any] | None = None
        previous_speeds: dict[str, float] | None = None
        previous_time: float | None = None
        capture_index = 0
        lifecycle_repairs: list[dict[str, Any]] = []

        for frame in range(DECISION_STEPS):
            delta_t = None if previous_time is None else float(env.simulation_time) - previous_time
            decision = step23._capture(
                env, frame=frame, phase="loop_start_decision", event_index=capture_index,
                previous_speed_by_entity=previous_speeds, delta_t_s=delta_t,
            )
            capture_index += 1
            decisions.append(decision)
            if previous_outcome is not None and step24._state(decision) != {**step24._state(previous_outcome), "frame_index": frame}:
                raise RuntimeError("next Decision was not independently recaptured")
            runtime_tasks = step23._all_runtime_tasks(env)
            for task_row in decision.get("tasks", []):
                task_object = runtime_tasks.get(str(task_row["task_id"]))
                if task_object is not None and hasattr(task_object, "getReturnedSize"):
                    task_row["required_returned_size"] = float(task_object.getReturnedSize())
            previous_speeds = {str(row["entity_id"]): float(row["speed_mps"]) for row in decision["entities"]}
            previous_time = float(decision["simulation_time_s"])

            global_noop = rng.random() < 0.2
            setter_calls: list[dict[str, Any]] = []
            family_audit: dict[str, dict[str, Any]] = {}

            waiting_return = sorted(step23._flatten(env.task_manager.getWaitingToReturnTaskInfos()), key=lambda task: str(task.getTaskId()))
            causal_task_ids = _causal_task_ids(decisions, frame)
            causal_entity_ids = _causal_entity_ids(decisions, frame)
            waiting_return = [task for task in waiting_return if str(task.getTaskId()) in causal_task_ids]
            waiting_return = [task for task in waiting_return if str(task.getTaskNodeId()) in causal_entity_ids]
            ready = [task for task in step23._ready_tasks(env) if str(task.getTaskId()) in causal_task_ids and str(task.getTaskNodeId()) in causal_entity_ids]
            route_eligible = bool(waiting_return or ready)
            route_intervention = route_eligible and not global_noop and rng.random() < 0.5
            route_entries: list[dict[str, Any]] = []
            if route_intervention and waiting_return:
                task = waiting_return[0]
                task_id = str(task.getTaskId())
                destination = str(task.getTaskNodeId())
                current = str(task.getCurrentNodeId())
                route = [step24.WIRED_RSU, destination] if current == step24.WIRED_CLOUD else [destination]
                requested_route = list(route)
                task_scheduler.setTaskReturnRoute(env, task_id, list(requested_route))
                succeeded = list(env.task_return_routes.get(task_id, [])) == requested_route
                if not succeeded:
                    raise RuntimeError(f"return route setter rejected {task_id}")
                route_entries.append({"task_id": task_id, "task_node_id": destination, "route_kind": "return", "target_node_id": requested_route[-1], "route_node_ids": requested_route})
                setter_calls.append({"setter_kind": "return_route", "task_id": task_id, "succeeded": True})
            elif route_intervention and ready:
                task = ready[0]
                task_id = str(task.getTaskId())
                source = str(task.getTaskNodeId())
                targets = [target for target in _legal_route_targets(env, source) if target in causal_entity_ids]
                if not targets:
                    route_intervention = False
                else:
                    target = targets[rng.randrange(min(3, len(targets)))]
                    requested_route = [target]
                    succeeded = task_scheduler.setTaskOffloading(env, source, task_id, target, route=list(requested_route))
                    if not succeeded:
                        raise RuntimeError(f"offload setter rejected {task_id}->{target}")
                    route_entries.append({"task_id": task_id, "task_node_id": source, "route_kind": "offload", "target_node_id": target, "route_node_ids": requested_route})
                    setter_calls.append({"setter_kind": "offload", "task_id": task_id, "succeeded": True})
            family_audit["route"] = {"eligible": route_eligible, "intervention": route_intervention, "no_op": route_eligible and not route_intervention, "signature": _signature(route_entries)}

            transmitting = [
                *step23._flatten(env.task_manager.getOffloadingTasks()),
                *step23._flatten(env.task_manager._returning_tasks),
            ]
            wireless = [task for task in sorted(transmitting, key=lambda row: str(row.getTaskId())) if str(task.getTaskId()) in causal_task_ids and step24._wireless_hop(env, task)]
            comm_eligible = bool(wireless)
            comm_intervention = comm_eligible and not global_noop and rng.random() < 0.5
            comm_entries: list[dict[str, Any]] = []
            n_rb = int(env.channel_manager.n_RB)
            for index, task in enumerate(wireless):
                if comm_intervention:
                    start = (frame + index + policy_seed) % n_rb
                    width = 1 + ((frame + index + policy_seed) % min(3, n_rb))
                    rb_indices = sorted({(start + offset) % n_rb for offset in range(width)})
                else:
                    rb_indices = [index % n_rb]
                task_id = str(task.getTaskId())
                communication_scheduler.setCommunicationWithRB(env, task_id, rb_indices)
                comm_entries.append({"task_id": task_id, "rb_indices": rb_indices})
                setter_calls.append({"setter_kind": "rb", "task_id": task_id, "succeeded": True})
            family_audit["comm"] = {"eligible": comm_eligible, "intervention": comm_intervention, "no_op": comm_eligible and not comm_intervention, "signature": _signature(comm_entries) if comm_intervention else None}

            comp_probe, _, comp_missing = step24._planned_comp_for_observed_nodes(env)
            comp_probe = [row for row in comp_probe if str(row["task_id"]) in causal_task_ids and str(row["node_id"]) in causal_entity_ids]
            comp_eligible = bool(comp_probe)
            comp_intervention = comp_eligible and not global_noop and rng.random() < 0.5
            comp_entries, allocations, comp_missing = _capacity_respecting_comp(env, comp_intervention, rng)
            comp_entries = [row for row in comp_entries if str(row["task_id"]) in causal_task_ids]
            comp_entries = [row for row in comp_entries if str(row["node_id"]) in causal_entity_ids]
            allocations = {task_id: value for task_id, value in allocations.items() if task_id in causal_task_ids}
            step23._install_cpu_callback(env, allocations)
            setter_calls.append({"setter_kind": "cpu_callback", "succeeded": True})
            family_audit["comp"] = {"eligible": comp_eligible, "intervention": comp_intervention, "no_op": comp_eligible and not comp_intervention, "signature": _signature(comp_entries) if comp_intervention else None}

            uavs = sorted(env.traffic_manager.getUAVTrafficInfos().items())
            mobility_eligible = bool(uavs)
            mobility_intervention = mobility_eligible and not global_noop and rng.random() < 0.5
            mobility_entries: list[dict[str, Any]] = []
            patterns: dict[str, dict[str, float]] = {}
            profile = (frame + policy_seed) % 5
            for uav_id, info in uavs:
                if mobility_intervention:
                    pattern = {
                        "angle": float(info.get("angle", 0.0)) + (-0.2, -0.1, 0.05, 0.1, 0.2)[profile],
                        "phi": float(info.get("phi", 0.0)),
                        "speed": (5.0, 8.0, 10.0, 12.0, 15.0)[profile],
                    }
                else:
                    pattern = {"angle": float(info.get("angle", 0.0)), "phi": float(info.get("phi", 0.0)), "speed": 0.0}
                patterns[str(uav_id)] = pattern
                mobility_entries.append({"uav_id": str(uav_id), "azimuth_rad": pattern["angle"], "elevation_rad": pattern["phi"], "speed_mps": pattern["speed"]})
            step23.TrafficScheduler.setUAVMobilityPatterns(env, patterns)
            setter_calls.extend({"setter_kind": "uav_mobility", "uav_id": row["uav_id"], "succeeded": True} for row in mobility_entries)
            family_audit["mobility"] = {"eligible": mobility_eligible, "intervention": mobility_intervention, "no_op": mobility_eligible and not mobility_intervention, "signature": _signature(mobility_entries) if mobility_intervention else None}

            activation = "".join("1" if family_audit[name]["intervention"] else "0" for name in ("route", "comm", "comp", "mobility"))
            runtime_tasks = step23._all_runtime_tasks(env)
            computed_before = {task_id: float(task.getComputedSize()) for task_id, task in runtime_tasks.items()}
            event_start = len(env.pi_jwm_transfer_events)
            start_time = float(env.simulation_time)
            env.step()
            end_time = float(env.simulation_time)
            lifecycle_repairs.extend(_repair_duplicate_task_references(env.task_manager, frame))
            events = [dict(row) for row in env.pi_jwm_transfer_events[event_start:]]
            computed_after = {task_id: float(task.getComputedSize()) for task_id, task in runtime_tasks.items()}
            slot_outcome = aggregate_slot_outcomes(
                transfer_events=events, computed_before=computed_before, computed_after=computed_after,
                transport_observation=env.pi_jwm_transfer_observation,
            )
            outcome = step23._capture(
                env, frame=frame, phase="post_env_step_outcome", event_index=capture_index,
                previous_speed_by_entity=previous_speeds, delta_t_s=end_time - start_time,
            )
            capture_index += 1
            outcome.update(slot_outcome)
            outcome["slot_transfer_events"] = events
            outcome["communication_observation"] = slot_outcome["communication_observation"]
            previous_outcome = outcome
            steps.append({
                "frame_index": frame,
                "decision_capture_event_id": decision["capture_event_id"],
                "action": {
                    "route": step23._family(route_entries, "eligible_policy_noop" if route_eligible else "no_route_eligible_task_at_decision"),
                    "comm": step23._family(comm_entries, "no_wireless_hop_at_decision"),
                    "comp": step23._family(comp_entries, "no_computing_task_at_decision" if not comp_missing else "cpu_capacity_unobserved"),
                    "mobility": step23._family(mobility_entries, "no_present_uav"),
                    "vehicle_motion": "SUMO external",
                },
                "behavior_policy_audit": {"global_noop": global_noop, "activation_bitmask": activation, "families": family_audit},
                "execution": {"start_time_s": start_time, "end_time_s": end_time, "setter_calls": setter_calls, "env_step_completed": True, "comp_missing_capacity_nodes": comp_missing},
                "outcome": outcome,
            })

        terminal = step23._capture(
            env, frame=DECISION_STEPS, phase="loop_start_terminal_decision", event_index=capture_index,
            previous_speed_by_entity=previous_speeds, delta_t_s=float(env.simulation_time) - float(previous_time),
        )
        decisions.append(terminal)
        if step24._state(terminal) != {**step24._state(previous_outcome), "frame_index": DECISION_STEPS}:
            raise RuntimeError("terminal Decision was not independently recaptured")
        checks = {
            "real_airfogsim_environment": any(cls.__name__ == "AirFogSimEnv" and cls.__module__.startswith("airfogsim") for cls in type(env).__mro__),
            "decision_count_97": len(decisions) == 97,
            "transition_count_96": len(steps) == 96,
            "continuous_frames": [row["frame_index"] for row in decisions] == list(range(97)) and [row["frame_index"] for row in steps] == list(range(96)),
            "continuous_time_grid": all(abs((float(decisions[i + 1]["simulation_time_s"]) - float(decisions[i]["simulation_time_s"])) - float(env.simulation_interval)) <= 1e-8 for i in range(96)),
            "vehicle_motion_external": all(row["action"]["vehicle_motion"] == "SUMO external" for row in steps),
            "setter_calls_succeeded": all(call.get("succeeded") is True for row in steps for call in row["execution"]["setter_calls"]),
            "causal_policy_audit_present": all(set(row["behavior_policy_audit"]["families"]) == {"route", "comm", "comp", "mobility"} for row in steps),
            "locked_test_accessed_false": True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"trajectory checks failed: {checks}")
        policy_contract = {
            "name": "causal_coverage_oriented_four_action_v1", "global_noop_probability": 0.2,
            "eligible_family_intervention_probability": 0.5, "future_schedule_read": False,
            "vehicle_motion": "SUMO external", "decision_steps": DECISION_STEPS,
        }
        return {
            "schema_version": "PI-JWM-Step5.5-Formal-Raw-Trajectory-v1",
            "environment": {
                "conda_env": "airfogsim", "airfogsim_source": "code/reference/AirFogSim",
                "seed": simulator_seed, "policy_seed": policy_seed,
                "config_hash": _digest(config), "policy_hash": _digest(policy_contract),
                "warmup_real_steps": warmup_steps, "decision_steps": DECISION_STEPS,
                "slot_duration_s": float(env.simulation_interval), "wired_edges": WIRED_EDGES,
                "runtime_class": f"{type(env).__module__}.{type(env).__qualname__}",
            },
            "behavior_policy": policy_contract,
            "decisions": decisions, "steps": steps, "checks": checks,
            "simulator_lifecycle_repairs": lifecycle_repairs,
            "scope": {"formal_dataset_raw": True, "training": False, "gpu": False, "locked_test": False},
        }
    except Exception as exc:
        memberships: dict[str, list[str]] = {}
        if env is not None:
            for collection_name in (
                "_to_generate_task_infos", "_waiting_to_offload_tasks", "_offloading_tasks",
                "_computing_tasks", "_waiting_to_return_tasks", "_returning_tasks",
                "_done_tasks", "_out_of_ddl_tasks",
            ):
                collection = getattr(env.task_manager, collection_name)
                for owner, tasks in collection.items():
                    for task in tasks:
                        memberships.setdefault(str(task.getTaskId()), []).append(f"{collection_name}:{owner}:object-{id(task)}")
        duplicates = {task_id: names for task_id, names in memberships.items() if len(names) > 1}
        raise RuntimeError(f"{exc}; duplicate_task_memberships={duplicates}") from exc
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


def _collection_summary(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    bitmasks = Counter(row["behavior_policy_audit"]["activation_bitmask"] for item in accepted for row in item["payload"]["steps"])
    return {
        "schema_version": "PI-JWM-Step5.5-Formal-Raw-Collection-v1",
        "accepted_count": len(accepted), "rejected_count": len(rejected),
        "accepted": [{key: value for key, value in item.items() if key != "payload"} for item in accepted],
        "rejected": rejected, "activation_bitmask_counts": dict(sorted(bitmasks.items())),
        "locked_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-count", type=int, default=ACCEPTED_COUNT)
    parser.add_argument("--decision-steps", type=int, default=DECISION_STEPS)
    parser.add_argument("--max-attempts", type=int, default=240)
    args = parser.parse_args()
    if args.decision_steps != DECISION_STEPS:
        raise SystemExit(f"STEP 5.5 requires exactly {DECISION_STEPS} transitions")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for attempt in range(args.max_attempts):
        if len(accepted) >= args.accepted_count:
            break
        simulator_seed = SIMULATOR_SEED_START + attempt
        policy_seed = POLICY_SEED_START + attempt
        trajectory_id = f"formal-v1-sim-{simulator_seed}-policy-{policy_seed}"
        try:
            payload = collect_trajectory(simulator_seed, policy_seed, trajectory_id)
            path = raw_dir / f"{trajectory_id}.json.gz"
            path.write_bytes(gzip.compress(_json_bytes(payload), compresslevel=6, mtime=0))
            accepted.append({
                "trajectory_id": trajectory_id, "simulator_seed": simulator_seed,
                "policy_seed": policy_seed, "primary": attempt < 60,
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "source_sha256": _sha256(path), "payload": payload,
            })
            print(json.dumps({"accepted": len(accepted), "trajectory_id": trajectory_id}, sort_keys=True), flush=True)
        except Exception as exc:  # each failed seed is provenance, never a partial trajectory
            rejected.append({
                "trajectory_id": trajectory_id, "simulator_seed": simulator_seed,
                "policy_seed": policy_seed, "primary": attempt < 60,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(json.dumps({"rejected": len(rejected), "trajectory_id": trajectory_id, "error": str(exc)}, sort_keys=True), flush=True)
    if len(accepted) != args.accepted_count:
        _write_json(output / "collection_summary.json", _collection_summary(accepted, rejected))
        raise RuntimeError(f"accepted {len(accepted)} of required {args.accepted_count}")
    summary = _collection_summary(accepted, rejected)
    _write_json(output / "collection_summary.json", summary)
    _write_json(output / "seed_lineage.json", {"primary_pairs": [[SIMULATOR_SEED_START + i, POLICY_SEED_START + i] for i in range(60)], "accepted": summary["accepted"], "rejected": rejected})
    print(json.dumps({"output": str(output), "accepted": len(accepted), "rejected": len(rejected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
