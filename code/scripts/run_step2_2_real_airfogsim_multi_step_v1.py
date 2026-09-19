"""Run a short real AirFogSim trajectory for PI-JWM Step 2.2 acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
for path in (
    CODE / "src",
    CODE / "scripts",
    CODE / "reference" / "AirFogSim" / "examples",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_p2_single_step_collector_preflight_v1 import _build_environment  # noqa: E402

from airfogsim.scheduler import (  # noqa: E402
    CommunicationScheduler,
    ComputationScheduler,
    TaskScheduler,
    TrafficScheduler,
)
from pi_jwm.airfogsim_cpu_inner_rule_v1 import allocate_airfogsim_cpu  # noqa: E402
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import (  # noqa: E402
    observe_airfogsim_snapshot,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import SnapshotPhase  # noqa: E402


OUTPUT = (
    CODE
    / "artifacts"
    / "protocols"
    / "pi_jwm_raw_multi_decision_step_real_airfogsim_v2_20260919"
)
TRAJECTORY_ID = "step2.2-real-seed0"
DECISION_STEPS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _traffic_rows(env) -> list[dict[str, object]]:
    manager = env.traffic_manager
    rows: list[dict[str, object]] = []
    moving = (
        ("vehicle", manager.getVehicleTrafficInfos()),
        ("uav", manager.getUAVTrafficInfos()),
    )
    for entity_type, infos in moving:
        for entity_id, info in sorted(infos.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": entity_type,
                    "position_m": [float(value) for value in info["position"]],
                    "speed_mps": float(info.get("speed", 0.0)),
                    "acceleration_mps2": float(info.get("acceleration", 0.0)),
                    "heading": float(info.get("angle", 0.0)),
                    "heading_unit": "degree" if entity_type == "vehicle" else "rad",
                    "elevation_rad": (
                        None if entity_type != "uav" else float(info.get("phi", 0.0))
                    ),
                    "vehicle_route_id": (
                        None if entity_type != "vehicle" else str(info.get("routeId"))
                    ),
                }
            )
    fixed = (
        ("rsu", manager.getRSUInfos()),
        ("cloud", manager.getCloudServerInfos()),
    )
    for entity_type, infos in fixed:
        for entity_id, info in sorted(infos.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": entity_type,
                    "position_m": [float(value) for value in info["position"]],
                    "speed_mps": 0.0,
                    "acceleration_mps2": 0.0,
                    "heading": 0.0,
                    "heading_unit": None,
                    "elevation_rad": None,
                    "vehicle_route_id": None,
                }
            )
    return rows


def _task_rows(env) -> list[dict[str, object]]:
    snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
    return [
        {
            "task_id": row.task_id,
            "task_node_id": row.task_node_id,
            "current_node_id": row.current_node_id,
            "lifecycle": row.lifecycle.value,
            "route_node_ids": list(row.route_nodes),
            "return_destination_id": row.return_destination_id,
            "arrival_time_s": row.arrival_time,
            "task_size": row.task_size,
            "task_cpu_work": row.task_cpu,
            "computed_cpu_work": row.computed_size,
            "transmitted_size": row.in_stage_transmitted_size,
        }
        for row in snapshot.tasks
    ]


def _capture(env, *, frame: int, phase: str, event_index: int) -> dict[str, object]:
    return {
        "capture_event_id": f"capture-{event_index:03d}",
        "capture_method": "fresh_direct_real_environment_read",
        "capture_phase": phase,
        "trajectory_id": TRAJECTORY_ID,
        "frame_index": frame,
        "simulation_time_s": float(env.simulation_time),
        "entities": _traffic_rows(env),
        "tasks": _task_rows(env),
        "n_rb": int(env.channel_manager.n_RB),
    }


def _state(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        key: snapshot[key]
        for key in ("trajectory_id", "frame_index", "simulation_time_s", "entities", "tasks", "n_rb")
    }


def _flatten(collection: dict[str, list[object]]) -> list[object]:
    return [task for owner in sorted(collection) for task in collection[owner]]


def _ready_tasks(env) -> list[object]:
    return sorted(
        (
            task
            for task in _flatten(env.task_manager.getWaitingToOffloadTasks())
            if env.task_manager.checkTaskDependency(task.getTaskNodeId(), task.getTaskId())
            is True
        ),
        key=lambda task: (str(task.getTaskNodeId()), str(task.getTaskId())),
    )


def _warm_to_ready(env, max_steps: int = 50) -> int:
    for step in range(max_steps + 1):
        if (
            _ready_tasks(env)
            and env.traffic_manager.getVehicleTrafficInfos()
            and env.traffic_manager.getUAVTrafficInfos()
        ):
            return step
        env.alloc_cpu_callback = lambda _tasks: {}
        env.step()
    raise RuntimeError("no ready task with vehicle and UAV after real warm-up")


def _nearest_target(env, source: str) -> str:
    candidates = [
        node_id
        for node_id in sorted(set(env.vehicles) | set(env.UAVs) | set(env.RSUs))
        if node_id != source and env._getNodeTypeById(node_id) in "VUI"
    ]
    if not candidates:
        raise RuntimeError(f"no real offload target for {source}")
    return min(candidates, key=lambda node_id: (env.getDistanceBetweenNodesById(source, node_id), node_id))


def _family(entries: list[dict[str, object]], no_op_reason: str | None) -> dict[str, object]:
    return {
        "field_present": True,
        "entries": entries,
        "empty": len(entries) == 0,
        "no_op_reason": no_op_reason if not entries else None,
    }


def _planned_comp(env) -> tuple[list[dict[str, object]], dict[str, float]]:
    computing = env.task_manager.getComputingTasks()
    active = {node_id: list(tasks) for node_id, tasks in computing.items() if tasks}
    if not active:
        return [], {}
    decision = allocate_airfogsim_cpu(env, active)
    node_by_task = {
        str(task.getTaskId()): str(node_id)
        for node_id, tasks in active.items()
        for task in tasks
    }
    entries = [
        {
            "task_id": task_id,
            "node_id": node_by_task[task_id],
            "allocated_cpu_per_s": float(allocation),
        }
        for task_id, allocation in sorted(decision.allocations.items())
    ]
    return entries, {key: float(value) for key, value in decision.allocations.items()}


def _install_fixed_cpu_callback(env, allocations: dict[str, float], runtime_log: list[dict[str, object]]) -> None:
    def callback(computing_tasks):
        runtime_task_ids = sorted(
            str(task.getTaskId())
            for tasks in computing_tasks.values()
            for task in tasks
        )
        runtime_log.append(
            {
                "runtime_task_ids": runtime_task_ids,
                "returned_allocations": dict(allocations),
            }
        )
        return dict(allocations)

    callback.__name__ = "pi_jwm_step2_2_fixed_decision_cpu_callback"
    ComputationScheduler.setComputingCallBack(env, callback)


def _acceleration_rows(decisions: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for previous, current in zip(decisions, decisions[1:]):
        delta_t = float(current["simulation_time_s"]) - float(previous["simulation_time_s"])
        before = {row["entity_id"]: row for row in previous["entities"]}
        after = {row["entity_id"]: row for row in current["entities"]}
        for entity_id in sorted(set(before) & set(after)):
            if after[entity_id]["entity_type"] not in {"vehicle", "uav"}:
                continue
            finite_difference = (
                float(after[entity_id]["speed_mps"]) - float(before[entity_id]["speed_mps"])
            ) / delta_t
            reported = float(after[entity_id]["acceleration_mps2"])
            if math.isclose(reported, finite_difference, rel_tol=0.0, abs_tol=1e-8):
                relation = "matches_forward_difference"
            elif math.isclose(reported, -finite_difference, rel_tol=0.0, abs_tol=1e-8):
                relation = "matches_negative_difference"
            else:
                relation = "differs_from_both"
            rows.append(
                {
                    "from_frame": previous["frame_index"],
                    "to_frame": current["frame_index"],
                    "entity_id": entity_id,
                    "entity_type": after[entity_id]["entity_type"],
                    "delta_t_s": delta_t,
                    "speed_previous_mps": before[entity_id]["speed_mps"],
                    "speed_current_mps": after[entity_id]["speed_mps"],
                    "finite_difference_mps2": finite_difference,
                    "airfogsim_reported_mps2": reported,
                    "relation": relation,
                }
            )
    return rows


def main() -> None:
    old_cwd = os.getcwd()
    env = None
    try:
        os.chdir(CODE / "reference" / "AirFogSim" / "examples")
        env, _, _, _, config = _build_environment(0, 4.0)
        warmup_steps = _warm_to_ready(env)
        decisions: list[dict[str, object]] = []
        steps: list[dict[str, object]] = []
        capture_index = 0
        previous_outcome = None
        independent_capture_checks: list[bool] = []

        for frame in range(DECISION_STEPS):
            decision = _capture(
                env,
                frame=frame,
                phase="loop_start_decision",
                event_index=capture_index,
            )
            capture_index += 1
            decisions.append(decision)
            if previous_outcome is not None:
                independent_capture_checks.append(
                    decision is not previous_outcome
                    and decision["capture_event_id"] != previous_outcome["capture_event_id"]
                    and _state(decision) == {
                        **_state(previous_outcome),
                        "frame_index": frame,
                    }
                )

            setter_calls: list[dict[str, object]] = []
            route_entries: list[dict[str, object]] = []
            ready = _ready_tasks(env)
            if ready:
                task = ready[0]
                task_id = str(task.getTaskId())
                source = str(task.getTaskNodeId())
                target = _nearest_target(env, source)
                succeeded = TaskScheduler.setTaskOffloading(
                    env, source, task_id, target, route=[target]
                )
                if not succeeded:
                    raise RuntimeError(f"real AirFogSim rejected route action for {task_id}")
                route_entries.append(
                    {
                        "task_id": task_id,
                        "task_node_id": source,
                        "route_kind": "offload",
                        "target_node_id": target,
                        "route_node_ids": [target],
                    }
                )
                setter_calls.append(
                    {"setter_kind": "offload", "task_id": task_id, "succeeded": True}
                )

            comm_entries: list[dict[str, object]] = []
            offloading = _flatten(env.task_manager.getOffloadingTasks())
            if frame != 2:
                for index, task in enumerate(sorted(offloading, key=lambda item: str(item.getTaskId()))):
                    task_id = str(task.getTaskId())
                    rb_indices = [index % int(env.channel_manager.n_RB)]
                    CommunicationScheduler.setCommunicationWithRB(env, task_id, rb_indices)
                    comm_entries.append({"task_id": task_id, "rb_indices": rb_indices})
                    setter_calls.append(
                        {"setter_kind": "rb", "task_id": task_id, "succeeded": True}
                    )

            comp_entries, fixed_allocations = _planned_comp(env)
            comp_runtime: list[dict[str, object]] = []
            _install_fixed_cpu_callback(env, fixed_allocations, comp_runtime)
            setter_calls.append({"setter_kind": "cpu_callback", "succeeded": True})

            mobility_entries: list[dict[str, object]] = []
            patterns: dict[str, dict[str, float]] = {}
            for uav_id, info in sorted(env.traffic_manager.getUAVTrafficInfos().items()):
                pattern = {
                    "angle": float(info.get("angle", 0.0)) + 0.05,
                    "phi": float(info.get("phi", 0.0)),
                    "speed": max(float(info.get("speed", 0.0)), 10.0),
                }
                patterns[str(uav_id)] = pattern
                mobility_entries.append(
                    {
                        "uav_id": str(uav_id),
                        "azimuth_rad": pattern["angle"],
                        "elevation_rad": pattern["phi"],
                        "speed_mps": pattern["speed"],
                    }
                )
            TrafficScheduler.setUAVMobilityPatterns(env, patterns)
            setter_calls.extend(
                {"setter_kind": "uav_mobility", "uav_id": row["uav_id"], "succeeded": True}
                for row in mobility_entries
            )

            action = {
                "trajectory_id": TRAJECTORY_ID,
                "frame_index": frame,
                "decision_time_s": decision["simulation_time_s"],
                "route": _family(route_entries, "no_ready_task_at_decision"),
                "comm": _family(
                    comm_entries,
                    "intentional_real_no_op_frame_for_empty_semantics"
                    if frame == 2 and offloading
                    else "no_offloading_task_at_decision",
                ),
                "comp": _family(comp_entries, "no_computing_task_at_decision"),
                "mobility": _family(mobility_entries, None),
                "vehicle_motion": "SUMO external",
            }
            start_time = float(env.simulation_time)
            env.step()
            end_time = float(env.simulation_time)
            outcome = _capture(
                env,
                frame=frame,
                phase="post_env_step_outcome",
                event_index=capture_index,
            )
            capture_index += 1
            previous_outcome = outcome
            steps.append(
                {
                    "frame_index": frame,
                    "decision_capture_event_id": decision["capture_event_id"],
                    "action": action,
                    "execution": {
                        "start_time_s": start_time,
                        "end_time_s": end_time,
                        "setter_calls": setter_calls,
                        "comp_runtime": comp_runtime,
                        "env_step_completed": True,
                    },
                    "outcome": outcome,
                }
            )

        terminal_decision = _capture(
            env,
            frame=DECISION_STEPS,
            phase="loop_start_terminal_decision",
            event_index=capture_index,
        )
        decisions.append(terminal_decision)
        independent_capture_checks.append(
            terminal_decision is not previous_outcome
            and terminal_decision["capture_event_id"] != previous_outcome["capture_event_id"]
            and _state(terminal_decision)
            == {**_state(previous_outcome), "frame_index": DECISION_STEPS}
        )

        acceleration = _acceleration_rows(decisions)
        family_names = ("route", "comm", "comp", "mobility")
        checks = {
            "real_airfogsim_environment": any(
                cls.__name__ == "AirFogSimEnv" and cls.__module__.startswith("airfogsim")
                for cls in type(env).__mro__
            ),
            "decision_count_is_steps_plus_one": len(decisions) == DECISION_STEPS + 1,
            "outcome_count_is_step_count": len(steps) == DECISION_STEPS,
            "next_decisions_independently_recaptured": all(independent_capture_checks),
            "trajectory_id_continuous": all(
                row["trajectory_id"] == TRAJECTORY_ID for row in decisions
            ),
            "frame_continuous": [row["frame_index"] for row in decisions]
            == list(range(DECISION_STEPS + 1)),
            "simulation_time_continuous": all(
                math.isclose(
                    float(decisions[index + 1]["simulation_time_s"])
                    - float(decisions[index]["simulation_time_s"]),
                    float(env.simulation_interval),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for index in range(DECISION_STEPS)
            ),
            "outcome_next_decision_entity_ids_align": all(
                {row["entity_id"] for row in steps[index]["outcome"]["entities"]}
                == {row["entity_id"] for row in decisions[index + 1]["entities"]}
                for index in range(DECISION_STEPS)
            ),
            "outcome_next_decision_task_ids_align": all(
                {row["task_id"] for row in steps[index]["outcome"]["tasks"]}
                == {row["task_id"] for row in decisions[index + 1]["tasks"]}
                for index in range(DECISION_STEPS)
            ),
            "outcome_next_decision_lifecycle_align": all(
                {row["task_id"]: row["lifecycle"] for row in steps[index]["outcome"]["tasks"]}
                == {row["task_id"]: row["lifecycle"] for row in decisions[index + 1]["tasks"]}
                for index in range(DECISION_STEPS)
            ),
            "four_action_fields_present_each_step": all(
                all(name in step["action"] for name in family_names) for step in steps
            ),
            "empty_action_is_explicit_no_op_not_missing": all(
                family["field_present"] is True
                and isinstance(family["entries"], list)
                and (
                    (family["empty"] is True and family["no_op_reason"] is not None)
                    or (family["empty"] is False and family["no_op_reason"] is None)
                )
                for step in steps
                for family in (step["action"][name] for name in family_names)
            ),
            "route_comm_comp_each_have_real_and_empty_frames": all(
                any(step["action"][name]["entries"] for step in steps)
                and any(not step["action"][name]["entries"] for step in steps)
                for name in ("route", "comm", "comp")
            ),
            "mobility_uav_only_and_complete": all(
                {row["uav_id"] for row in step["action"]["mobility"]["entries"]}
                == {
                    row["entity_id"]
                    for row in decisions[index]["entities"]
                    if row["entity_type"] == "uav"
                }
                for index, step in enumerate(steps)
            ),
            "vehicle_motion_from_sumo_observed": any(
                before["position_m"] != after["position_m"]
                for previous, current in zip(decisions, decisions[1:])
                for entity_id, before in {
                    row["entity_id"]: row
                    for row in previous["entities"]
                    if row["entity_type"] == "vehicle"
                }.items()
                for after in [
                    next(
                        (
                            row
                            for row in current["entities"]
                            if row["entity_id"] == entity_id
                            and row["entity_type"] == "vehicle"
                        ),
                        before,
                    )
                ]
            ),
            "acceleration_comparison_recorded": bool(acceleration),
            "uav_negative_difference_semantics_observed": any(
                row["entity_type"] == "uav"
                and row["relation"] == "matches_negative_difference"
                and not math.isclose(float(row["finite_difference_mps2"]), 0.0)
                for row in acceleration
            ),
            "scope_non_locked_no_training_no_gpu": True,
        }
        failed = [name for name, passed in checks.items() if passed is not True]
        if failed:
            raise RuntimeError(json.dumps({"failed": failed, "checks": checks}, ensure_ascii=False))

        payload = {
            "schema_version": "PIJWM-Step-2.2-Real-AirFogSim-Multi-Step-v1",
            "environment": {
                "conda_env": "airfogsim",
                "airfogsim_source": "code/reference/AirFogSim",
                "seed": 0,
                "config_hash": hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "warmup_real_steps": warmup_steps,
                "decision_steps": DECISION_STEPS,
                "slot_duration_s": float(env.simulation_interval),
                "runtime_class": f"{type(env).__module__}.{type(env).__qualname__}",
                "runtime_mro": [
                    f"{cls.__module__}.{cls.__qualname__}" for cls in type(env).__mro__
                ],
            },
            "decisions": decisions,
            "steps": steps,
            "acceleration_comparison": {
                "formula": "(speed_t-speed_t_minus_1)/delta_t",
                "interpretation_boundary": (
                    "observational only; simulator unchanged; Dataset field choice not decided"
                ),
                "rows": acceleration,
            },
            "checks": checks,
            "scope": {
                "non_locked": True,
                "training": False,
                "gpu": False,
                "locked_test": False,
            },
        }
        OUTPUT.mkdir(parents=True, exist_ok=False)
        artifact = OUTPUT / "real_multi_step.json"
        write_json(artifact, payload)
        sources = (
            Path(__file__),
            CODE / "src" / "pi_jwm" / "airfogsim_full_dual_graph_observer_v1.py",
            CODE / "src" / "pi_jwm" / "airfogsim_cpu_inner_rule_v1.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "traffic_manager.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "task_manager.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "scheduler" / "task_sched.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "scheduler" / "communication_sched.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "scheduler" / "computation_sched.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "scheduler" / "traffic_sched.py",
        )
        manifest = {
            "artifact": "real_multi_step.json",
            "artifact_sha256": sha256(artifact),
            "passed": True,
            "all_checks": checks,
            "source_files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in sources
            },
            "git_tracking_requirement": (
                "artifact and manifest must be force-added because code/artifacts is ignored"
            ),
        }
        write_json(OUTPUT / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output": str(OUTPUT),
                    "decision_steps": DECISION_STEPS,
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
