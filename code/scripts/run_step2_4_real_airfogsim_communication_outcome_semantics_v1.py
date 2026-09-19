"""Real AirFogSim acceptance for Step 2.4 communication Outcome semantics."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import argparse
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

import run_step2_3_real_airfogsim_raw_contract_finalization_v1 as step23  # noqa: E402
from pi_jwm.raw_trajectory_causal_contract_v1 import (  # noqa: E402
    aggregate_slot_outcomes,
)


OUTPUT = (
    CODE
    / "artifacts"
    / "protocols"
    / "pi_jwm_communication_outcome_semantics_v2_20260919"
)
TRAJECTORY_ID = "step2.4-real-communication-seed0"
DECISION_STEPS = 6
WIRED_RSU = "RSU_0"
WIRED_CLOUD = "cloudServer_4"
WIRED_EDGES = [
    {
        "u": WIRED_RSU,
        "v": WIRED_CLOUD,
        "capacity_mbps": 100,
        "prop_ms": 1.0,
        "bidirectional": True,
    }
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _state(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        key: snapshot[key]
        for key in (
            "trajectory_id",
            "frame_index",
            "simulation_time_s",
            "entities",
            "tasks",
            "input_side_entity_index",
            "internal_metadata",
            "channel_rows",
            "node_cpu_capacity_per_s",
            "node_cpu_capacity_observation_rows",
            "n_rb",
        )
    }


def _wireless_hop(env, task) -> bool:
    route = list(task.getToOffloadRoute())
    if not route:
        return False
    source_type = env._getNodeTypeById(task.getCurrentNodeId())
    target_type = env._getNodeTypeById(route[0])
    return source_type in {"V", "U", "I"} and target_type in {"V", "U", "I"}


def _return_route(env, task) -> list[str]:
    task_node = str(task.getTaskNodeId())
    current = str(task.getCurrentNodeId())
    if current == WIRED_CLOUD:
        return [WIRED_RSU, task_node]
    return [task_node]


def _task_rows_by_id(outcome: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["task_id"]): row for row in outcome["tasks"]}


def _planned_comp_for_observed_nodes(env) -> tuple[list[dict[str, object]], dict[str, float], list[str]]:
    """Use the existing CPU rule only where the real profile exposes cpu."""

    computing = env.task_manager.getComputingTasks()
    active = {node_id: list(tasks) for node_id, tasks in computing.items() if tasks}
    unsupported = [
        str(node_id)
        for node_id in sorted(active)
        if "cpu" not in env._getNodeById(node_id).getFogProfile()
    ]
    supported = {
        node_id: tasks for node_id, tasks in active.items() if str(node_id) not in unsupported
    }
    if not supported:
        return [], {}, unsupported
    decision = step23.allocate_airfogsim_cpu(env, supported)
    node_by_task = {
        str(task.getTaskId()): str(node_id)
        for node_id, tasks in supported.items()
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
    return entries, {key: float(value) for key, value in decision.allocations.items()}, unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one non-locked Step 2.4 trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-id", default=TRAJECTORY_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")

    old_cwd = os.getcwd()
    env = None
    step23.TRAJECTORY_ID = str(args.trajectory_id)
    try:
        os.chdir(CODE / "reference" / "AirFogSim" / "examples")
        env, task_scheduler, communication_scheduler, _, config = step23._build_environment(
            args.seed,
            5.0,
            wired_edges=WIRED_EDGES,
        )
        if not env.wired_manager.hasLink(WIRED_RSU, WIRED_CLOUD):
            raise RuntimeError("configured real wired link is not present")
        warmup_steps = step23._warm_to_ready(env)
        decisions: list[dict[str, object]] = []
        steps: list[dict[str, object]] = []
        previous_outcome = None
        previous_speeds: dict[str, float] | None = None
        previous_time: float | None = None
        capture_index = 0
        wired_route_task_id: str | None = None

        for frame in range(DECISION_STEPS):
            delta_t = None if previous_time is None else float(env.simulation_time) - previous_time
            decision = step23._capture(
                env,
                frame=frame,
                phase="loop_start_decision",
                event_index=capture_index,
                previous_speed_by_entity=previous_speeds,
                delta_t_s=delta_t,
            )
            capture_index += 1
            decisions.append(decision)
            if previous_outcome is not None and (
                decision is previous_outcome
                or decision["capture_event_id"] == previous_outcome["capture_event_id"]
                or _state(decision) != {**_state(previous_outcome), "frame_index": frame}
            ):
                raise RuntimeError("next Decision was not independently recaptured")
            previous_speeds = {
                row["entity_id"]: float(row["speed_mps"])
                for row in decision["entities"]
            }
            previous_time = float(decision["simulation_time_s"])

            setter_calls: list[dict[str, object]] = []
            route_entries: list[dict[str, object]] = []
            waiting_return = sorted(
                step23._flatten(env.task_manager.getWaitingToReturnTaskInfos()),
                key=lambda task: str(task.getTaskId()),
            )
            if waiting_return:
                task = waiting_return[0]
                task_id = str(task.getTaskId())
                route = _return_route(env, task)
                task_scheduler.setTaskReturnRoute(env, task_id, route)
                queued = list(env.task_return_routes.get(task_id, [])) == route
                route_entries.append(
                    {
                        "task_id": task_id,
                        "task_node_id": str(task.getTaskNodeId()),
                        "route_kind": "return",
                        "target_node_id": route[-1],
                        "route_node_ids": list(route),
                    }
                )
                setter_calls.append(
                    {"setter_kind": "return_route", "task_id": task_id, "succeeded": queued}
                )
            else:
                ready = step23._ready_tasks(env)
                if ready and wired_route_task_id is None:
                    task = ready[0]
                    task_id = str(task.getTaskId())
                    source = str(task.getTaskNodeId())
                    route = [WIRED_RSU, WIRED_CLOUD]
                    succeeded = task_scheduler.setTaskOffloading(
                        env,
                        source,
                        task_id,
                        WIRED_CLOUD,
                        route=route,
                    )
                    if not succeeded:
                        raise RuntimeError(f"AirFogSim rejected wired-chain offload: {task_id}")
                    wired_route_task_id = task_id
                    route_entries.append(
                        {
                            "task_id": task_id,
                            "task_node_id": source,
                            "route_kind": "offload",
                            "target_node_id": WIRED_CLOUD,
                            "route_node_ids": list(route),
                        }
                    )
                    setter_calls.append(
                        {"setter_kind": "offload", "task_id": task_id, "succeeded": True}
                    )

            transmitting = [
                *step23._flatten(env.task_manager.getOffloadingTasks()),
                *step23._flatten(env.task_manager._returning_tasks),
            ]
            comm_entries: list[dict[str, object]] = []
            for index, task in enumerate(
                sorted(transmitting, key=lambda row: str(row.getTaskId()))
            ):
                if not _wireless_hop(env, task):
                    continue
                task_id = str(task.getTaskId())
                rb_indices = [index % int(env.channel_manager.n_RB)]
                communication_scheduler.setCommunicationWithRB(env, task_id, rb_indices)
                comm_entries.append({"task_id": task_id, "rb_indices": rb_indices})
                setter_calls.append(
                    {"setter_kind": "rb", "task_id": task_id, "succeeded": True}
                )

            comp_entries, allocations, comp_missing_nodes = _planned_comp_for_observed_nodes(env)
            step23._install_cpu_callback(env, allocations)
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
            step23.TrafficScheduler.setUAVMobilityPatterns(env, patterns)
            setter_calls.extend(
                {"setter_kind": "uav_mobility", "uav_id": row["uav_id"], "succeeded": True}
                for row in mobility_entries
            )

            runtime_tasks = step23._all_runtime_tasks(env)
            computed_before = {
                task_id: float(task.getComputedSize())
                for task_id, task in runtime_tasks.items()
            }
            event_start = len(env.pi_jwm_transfer_events)
            start_time = float(env.simulation_time)
            env.step()
            end_time = float(env.simulation_time)
            events = [dict(row) for row in env.pi_jwm_transfer_events[event_start:]]
            computed_after = {
                task_id: float(task.getComputedSize())
                for task_id, task in runtime_tasks.items()
            }
            slot_outcome = aggregate_slot_outcomes(
                transfer_events=events,
                computed_before=computed_before,
                computed_after=computed_after,
                transport_observation=env.pi_jwm_transfer_observation,
            )
            outcome = step23._capture(
                env,
                frame=frame,
                phase="post_env_step_outcome",
                event_index=capture_index,
                previous_speed_by_entity=previous_speeds,
                delta_t_s=end_time - start_time,
            )
            capture_index += 1
            outcome.update(slot_outcome)
            outcome["slot_transfer_events"] = events
            outcome["communication_observation"] = slot_outcome["communication_observation"]
            previous_outcome = outcome
            steps.append(
                {
                    "frame_index": frame,
                    "decision_capture_event_id": decision["capture_event_id"],
                    "action": {
                        "route": step23._family(route_entries, "no_route_action_at_decision"),
                        "comm": step23._family(comm_entries, "no_wireless_hop_at_decision"),
                        "comp": step23._family(
                            comp_entries,
                            "no_cpu_capacity_observed_for_computing_task"
                            if comp_missing_nodes
                            else "no_computing_task_at_decision",
                        ),
                        "mobility": step23._family(mobility_entries, "no_present_uav"),
                        "vehicle_motion": "SUMO external",
                    },
                    "execution": {
                        "start_time_s": start_time,
                        "end_time_s": end_time,
                        "setter_calls": setter_calls,
                        "env_step_completed": True,
                        "comp_missing_capacity_nodes": comp_missing_nodes,
                    },
                    "outcome": outcome,
                }
            )

        terminal = step23._capture(
            env,
            frame=DECISION_STEPS,
            phase="loop_start_terminal_decision",
            event_index=capture_index,
            previous_speed_by_entity=previous_speeds,
            delta_t_s=float(env.simulation_time) - float(previous_time),
        )
        decisions.append(terminal)
        if _state(terminal) != {**_state(previous_outcome), "frame_index": DECISION_STEPS}:
            raise RuntimeError("terminal Decision was not independently recaptured")

        events = [
            event
            for step in steps
            for event in step["outcome"]["slot_transfer_events"]
        ]
        wireless_events = [event for event in events if event.get("transport") == "wireless"]
        wired_events = [event for event in events if event.get("transport") == "wired"]
        all_outcome_task_rows = {
            str(row["task_id"]): row
            for step in steps
            for row in step["outcome"]["tasks"]
        }
        progress_checks = []
        lifecycle_checks = []
        for event in events:
            delivered = float(event["delivered_data"])
            before = float(event["remaining_before"])
            progress_checks.append(0.0 <= delivered <= before + 1e-9)
            row = all_outcome_task_rows.get(str(event["task_id"]))
            lifecycle_checks.append(
                row is not None
                and (
                    not bool(event.get("flow_completed"))
                    or row["current_node_id"] == event["target"]
                    or row["lifecycle"] in {"computing", "returning", "done"}
                )
            )
        empty_observed_checks = [
            step["outcome"]["communication_observation"][transport]["observed_mask"]
            and isinstance(step["outcome"][f"{transport}_delivered_data_by_task"], dict)
            for step in steps
            for transport in ("wireless", "wired")
            if not step["outcome"][f"{transport}_delivered_data_by_task"]
        ]
        total_checks = []
        for step in steps:
            outcome = step["outcome"]
            wireless = outcome["wireless_delivered_data_by_task"]
            wired = outcome["wired_delivered_data_by_task"]
            total = outcome["delivered_data_by_task"]
            expected = {
                task_id: wireless.get(task_id, 0.0) + wired.get(task_id, 0.0)
                for task_id in sorted(set(wireless) | set(wired))
            }
            total_checks.append(total == expected)

        checks = {
            "real_airfogsim_environment": any(
                cls.__name__ == "AirFogSimEnv" and cls.__module__.startswith("airfogsim")
                for cls in type(env).__mro__
            ),
            "real_wired_link_configured": env.wired_manager.hasLink(WIRED_RSU, WIRED_CLOUD),
            "wireless_event_capture_observed": bool(wireless_events),
            "wired_event_capture_observed": bool(wired_events),
            "wired_event_source_is_real_manager_result": all(
                event["source_method"] == "wired_manager.step" for event in wired_events
            ),
            "transport_explicit_on_all_events": bool(events)
            and all(event.get("transport") in {"wireless", "wired"} for event in events),
            "slot_maps_have_explicit_observation_semantics": all(
                step["outcome"]["communication_observation"]["wireless"]["observed_mask"]
                and step["outcome"]["communication_observation"]["wired"]["observed_mask"]
                and step["outcome"]["communication_observation"]["total"]["observed_mask"]
                for step in steps
            ),
            "empty_map_is_not_missing": bool(empty_observed_checks) and all(empty_observed_checks),
            "total_equals_wireless_plus_wired": bool(total_checks) and all(total_checks),
            "transmitted_progress_matches_remaining_before": bool(progress_checks)
            and all(progress_checks),
            "completed_transfer_lifecycle_is_aligned": bool(lifecycle_checks)
            and all(lifecycle_checks),
            "wired_route_task_reached_wired_service": wired_route_task_id
            is not None
            and any(event["task_id"] == wired_route_task_id for event in wired_events),
            "trajectory_decision_outcome_alignment": len(decisions) == DECISION_STEPS + 1
            and len(steps) == DECISION_STEPS,
            "scope_non_locked_no_training_no_gpu": True,
        }
        failed = [name for name, passed in checks.items() if passed is not True]
        if failed:
            raise RuntimeError(json.dumps({"failed": failed, "checks": checks}, ensure_ascii=False))

        payload = {
            "schema_version": "PIJWM-Step-2.4-Communication-Outcome-Semantics-v2-dag-amendment",
            "environment": {
                "conda_env": "airfogsim",
                "airfogsim_source": "code/reference/AirFogSim",
                "seed": args.seed,
                "config_hash": hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "wired_edges": WIRED_EDGES,
                "warmup_real_steps": warmup_steps,
                "decision_steps": DECISION_STEPS,
                "slot_duration_s": float(env.simulation_interval),
                "runtime_class": f"{type(env).__module__}.{type(env).__qualname__}",
            },
            "field_semantics": {
                "wireless_delivered_data_by_task": {
                    "source": "ObservedAirFogSimEnv._updateWirelessCommunication -> AirFogSim _execute_communication profile",
                    "unit": "AirFogSim native data unit/slot",
                    "empty_map": "observed wireless hook with no wireless task service in this slot",
                    "missing": "null map plus observed_mask=false only when the wireless hook is unavailable",
                },
                "wired_delivered_data_by_task": {
                    "source": "ObservedAirFogSimEnv._updateWiredCommunication -> WiredNetworkManager.step result",
                    "unit": "AirFogSim wired manager native data unit/slot",
                    "empty_map": "observed wired hook with no wired task service in this slot",
                    "missing": "null map plus observed_mask=false only when the wired result hook is unavailable",
                },
                "delivered_data_by_task": {
                    "definition": "per-task sum of the observed wireless and wired maps",
                    "missing": "null plus total observed_mask=false when either component is unavailable",
                },
                "transmitted_progress": {
                    "source": "Task.transmit_to_Node state transition and event remaining_before/delivered_data",
                    "validation": "0 <= delivered_data <= remaining_before; completed events align with post-step current node/lifecycle",
                },
            },
            "decisions": decisions,
            "steps": steps,
            "checks": checks,
            "scope": {"non_locked": True, "training": False, "gpu": False, "locked_test": False},
        }
        output.mkdir(parents=True, exist_ok=False)
        artifact = output / "real_communication_outcome_semantics.json"
        write_json(artifact, payload)
        sources = (
            Path(__file__),
            CODE / "scripts" / "run_step2_3_real_airfogsim_raw_contract_finalization_v1.py",
            CODE / "scripts" / "run_p2_single_step_collector_preflight_v1.py",
            CODE / "src" / "pi_jwm" / "raw_trajectory_causal_contract_v1.py",
            CODE / "src" / "pi_jwm" / "raw_single_decision_step_contract_v1.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "airfogsim_env.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "wired_manager.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "entities" / "task.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "task_manager.py",
        )
        manifest = {
            "schema_version": "PIJWM-Step-2.4-Communication-Outcome-Manifest-v2-dag-amendment",
            "artifact": artifact.name,
            "artifact_sha256": sha256(artifact),
            "passed": True,
            "all_checks": checks,
            "source_files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in sources
            },
            "git_tracking_requirement": "artifact and manifest must be force-added",
        }
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"output": str(output), "checks": checks}, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
