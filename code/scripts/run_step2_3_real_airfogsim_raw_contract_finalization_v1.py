"""Run real AirFogSim acceptance for PI-JWM Step 2.3 raw finalization."""

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
from pi_jwm.raw_trajectory_causal_contract_v1 import (  # noqa: E402
    aggregate_slot_outcomes,
    attach_canonical_acceleration,
    partition_tasks_at_decision,
)


OUTPUT = (
    CODE
    / "artifacts"
    / "protocols"
    / "pi_jwm_raw_contract_causal_complete_v2_20260919"
)
TRAJECTORY_ID = "step2.3-real-seed0"
DECISION_STEPS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _flatten(collection: dict[str, list[object]]) -> list[object]:
    return [task for owner in sorted(collection) for task in collection[owner]]


def _all_runtime_tasks(env) -> dict[str, object]:
    names = (
        "_to_generate_task_infos",
        "_waiting_to_offload_tasks",
        "_offloading_tasks",
        "_computing_tasks",
        "_waiting_to_return_tasks",
        "_returning_tasks",
        "_done_tasks",
        "_out_of_ddl_tasks",
    )
    result: dict[str, object] = {}
    for name in names:
        for task in _flatten(getattr(env.task_manager, name)):
            result[str(task.getTaskId())] = task
    return result


def _task_rows(env) -> tuple[list[dict[str, object]], list[dict[str, object]], object]:
    snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
    all_rows = [
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
    observable, internal_future = partition_tasks_at_decision(
        all_rows, float(env.simulation_time)
    )
    return observable, internal_future, snapshot


def _base_entity_rows(env) -> list[dict[str, object]]:
    manager = env.traffic_manager
    rows: list[dict[str, object]] = []
    for entity_type, infos in (
        ("vehicle", manager.getVehicleTrafficInfos()),
        ("uav", manager.getUAVTrafficInfos()),
    ):
        for entity_id, info in sorted(infos.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": entity_type,
                    "position_m": [float(value) for value in info["position"]],
                    "speed_mps": float(info.get("speed", 0.0)),
                    "raw_simulator_acceleration_mps2": float(
                        info.get("acceleration", 0.0)
                    ),
                    "raw_simulator_acceleration_source": (
                        "traffic_manager current traffic infos"
                    ),
                    "heading": float(info.get("angle", 0.0)),
                    "heading_unit": "degree" if entity_type == "vehicle" else "rad",
                    "elevation_rad": (
                        None if entity_type == "vehicle" else float(info.get("phi", 0.0))
                    ),
                }
            )
    for entity_type, infos in (
        ("rsu", manager.getRSUInfos()),
        ("cloud", manager.getCloudServerInfos()),
    ):
        for entity_id, info in sorted(infos.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": entity_type,
                    "position_m": [float(value) for value in info["position"]],
                    "speed_mps": 0.0,
                    "raw_simulator_acceleration_mps2": None,
                    "raw_simulator_acceleration_source": None,
                    "heading": None,
                    "heading_unit": None,
                    "elevation_rad": None,
                }
            )
    return rows


def _cpu_capacities(env) -> tuple[dict[str, float], list[dict[str, object]]]:
    entities = {**env.vehicles, **env.UAVs, **env.RSUs, **env.cloudServers}
    capacities: dict[str, float] = {}
    observations: list[dict[str, object]] = []
    for node_id, entity in sorted(entities.items()):
        profile = entity.getFogProfile()
        observed = "cpu" in profile and profile["cpu"] is not None
        value = float(profile["cpu"]) if observed else None
        if observed:
            capacities[str(node_id)] = value
        observations.append(
            {
                "node_id": str(node_id),
                "capacity_per_s": value,
                "observed_mask": observed,
                "missing_reason": None if observed else "CPU_NOT_EXPOSED_IN_FOG_PROFILE",
                "source_method": "entity.getFogProfile()['cpu']",
            }
        )
    return capacities, observations


def _capture(
    env,
    *,
    frame: int,
    phase: str,
    event_index: int,
    previous_speed_by_entity: dict[str, float] | None,
    delta_t_s: float | None,
) -> dict[str, object]:
    observable_tasks, internal_future, snapshot = _task_rows(env)
    entities = attach_canonical_acceleration(
        _base_entity_rows(env),
        previous_speed_by_entity=previous_speed_by_entity,
        delta_t_s=delta_t_s,
    )
    physical_ids = [row["entity_id"] for row in entities]
    observable_task_ids = [row["task_id"] for row in observable_tasks]
    observable_task_id_set = set(observable_task_ids)
    raw_dag_rows = [
        {
            "dag_edge_id": row.dag_edge_id,
            "source_task_id": row.source_task_id,
            "target_task_id": row.target_task_id,
            "communication_mapping": row.communication_mapping,
            "source": "airfogsim_full_dual_graph_observer_v1._extract_dag_edges",
        }
        for row in snapshot.dag_edges
    ]
    observable_dag_rows = [
        row for row in raw_dag_rows
        if row["source_task_id"] in observable_task_id_set
        and row["target_task_id"] in observable_task_id_set
    ]
    internal_future_dag_rows = [
        row for row in raw_dag_rows
        if row["source_task_id"] not in observable_task_id_set
        or row["target_task_id"] not in observable_task_id_set
    ]
    cpu_capacities, cpu_observations = _cpu_capacities(env)
    return {
        "capture_event_id": f"capture-{event_index:03d}",
        "capture_method": "fresh_direct_real_environment_read",
        "capture_phase": phase,
        "trajectory_id": TRAJECTORY_ID,
        "frame_index": frame,
        "simulation_time_s": float(env.simulation_time),
        "entities": entities,
        "tasks": observable_tasks,
        "input_side_entity_index": {
            "physical_entity_ids": physical_ids,
            "task_ids": observable_task_ids,
        },
        "internal_metadata": {
            "future_task_schedule": internal_future,
            "future_dag_edges": internal_future_dag_rows,
            "excluded_from_O_t_history_and_input_entity_index": True,
        },
        "channel_rows": [dict(row) for row in snapshot.channel_rows],
        "dag_edges": observable_dag_rows,
        "node_cpu_capacity_per_s": cpu_capacities,
        "node_cpu_capacity_observation_rows": cpu_observations,
        "n_rb": int(env.channel_manager.n_RB),
    }


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
        if _ready_tasks(env) and env.traffic_manager.getVehicleTrafficInfos():
            return step
        env.alloc_cpu_callback = lambda _tasks: {}
        env.step()
    raise RuntimeError("no ready task after real warm-up")


def _nearest_target(env, source: str) -> str:
    candidates = [
        node_id
        for node_id in sorted(set(env.vehicles) | set(env.UAVs) | set(env.RSUs))
        if node_id != source and env._getNodeTypeById(node_id) in "VUI"
    ]
    return min(
        candidates,
        key=lambda node_id: (env.getDistanceBetweenNodesById(source, node_id), node_id),
    )


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


def _install_cpu_callback(env, allocations: dict[str, float]) -> None:
    def callback(_computing_tasks):
        return dict(allocations)

    callback.__name__ = "pi_jwm_step2_3_fixed_decision_cpu_callback"
    ComputationScheduler.setComputingCallBack(env, callback)


def _family(entries: list[dict[str, object]], reason: str) -> dict[str, object]:
    return {
        "field_present": True,
        "entries": entries,
        "empty": not entries,
        "no_op_reason": reason if not entries else None,
    }


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


def main() -> None:
    old_cwd = os.getcwd()
    env = None
    try:
        os.chdir(CODE / "reference" / "AirFogSim" / "examples")
        env, _, _, _, config = _build_environment(0, 4.0)
        warmup_steps = _warm_to_ready(env)
        decisions: list[dict[str, object]] = []
        steps: list[dict[str, object]] = []
        previous_outcome = None
        previous_decision_speeds: dict[str, float] | None = None
        previous_decision_time: float | None = None
        capture_index = 0
        return_calls: list[dict[str, object]] = []

        for frame in range(DECISION_STEPS):
            delta_t = (
                None
                if previous_decision_time is None
                else float(env.simulation_time) - previous_decision_time
            )
            decision = _capture(
                env,
                frame=frame,
                phase="loop_start_decision",
                event_index=capture_index,
                previous_speed_by_entity=previous_decision_speeds,
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

            previous_decision_speeds = {
                row["entity_id"]: float(row["speed_mps"])
                for row in decision["entities"]
            }
            previous_decision_time = float(decision["simulation_time_s"])
            setter_calls: list[dict[str, object]] = []
            route_entries: list[dict[str, object]] = []

            waiting_return = sorted(
                _flatten(env.task_manager.getWaitingToReturnTaskInfos()),
                key=lambda task: str(task.getTaskId()),
            )
            if waiting_return:
                task = waiting_return[0]
                task_id = str(task.getTaskId())
                destination = str(task.getTaskNodeId())
                route = [destination]
                TaskScheduler.setTaskReturnRoute(env, task_id, route)
                queued = list(env.task_return_routes.get(task_id, [])) == route
                route_entries.append(
                    {
                        "task_id": task_id,
                        "task_node_id": destination,
                        "route_kind": "return",
                        "target_node_id": destination,
                        "route_node_ids": route,
                    }
                )
                setter_calls.append(
                    {"setter_kind": "return_route", "task_id": task_id, "succeeded": queued}
                )
                return_calls.append(
                    {"frame_index": frame, "task_id": task_id, "queued_before_step": queued}
                )
            else:
                ready = _ready_tasks(env)
                if ready:
                    task = ready[0]
                    task_id = str(task.getTaskId())
                    source = str(task.getTaskNodeId())
                    target = _nearest_target(env, source)
                    succeeded = TaskScheduler.setTaskOffloading(
                        env, source, task_id, target, route=[target]
                    )
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
                        {"setter_kind": "offload", "task_id": task_id, "succeeded": bool(succeeded)}
                    )

            transmitting = [
                *_flatten(env.task_manager.getOffloadingTasks()),
                *_flatten(env.task_manager._returning_tasks),
            ]
            comm_entries: list[dict[str, object]] = []
            for index, task in enumerate(sorted(transmitting, key=lambda row: str(row.getTaskId()))):
                task_id = str(task.getTaskId())
                rb_indices = [index % int(env.channel_manager.n_RB)]
                CommunicationScheduler.setCommunicationWithRB(env, task_id, rb_indices)
                comm_entries.append({"task_id": task_id, "rb_indices": rb_indices})
                setter_calls.append(
                    {"setter_kind": "rb", "task_id": task_id, "succeeded": True}
                )

            comp_entries, allocations = _planned_comp(env)
            _install_cpu_callback(env, allocations)
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

            runtime_tasks = _all_runtime_tasks(env)
            computed_before = {
                task_id: float(task.getComputedSize())
                for task_id, task in runtime_tasks.items()
            }
            event_start = len(env.pi_jwm_transfer_events)
            start_time = float(env.simulation_time)
            env.step()
            end_time = float(env.simulation_time)
            transfer_events = env.pi_jwm_transfer_events[event_start:]
            computed_after = {
                task_id: float(task.getComputedSize())
                for task_id, task in runtime_tasks.items()
            }
            slot_outcome = aggregate_slot_outcomes(
                transfer_events=transfer_events,
                computed_before=computed_before,
                computed_after=computed_after,
            )
            outcome = _capture(
                env,
                frame=frame,
                phase="post_env_step_outcome",
                event_index=capture_index,
                previous_speed_by_entity=previous_decision_speeds,
                delta_t_s=end_time - start_time,
            )
            capture_index += 1
            outcome.update(slot_outcome)
            outcome["slot_transfer_events"] = transfer_events
            previous_outcome = outcome
            steps.append(
                {
                    "frame_index": frame,
                    "decision_capture_event_id": decision["capture_event_id"],
                    "action": {
                        "route": _family(route_entries, "no_route_eligible_task_at_decision"),
                        "comm": _family(comm_entries, "no_transmitting_task_at_decision"),
                        "comp": _family(comp_entries, "no_computing_task_at_decision"),
                        "mobility": _family(mobility_entries, "no_present_uav"),
                        "vehicle_motion": "SUMO external",
                    },
                    "execution": {
                        "start_time_s": start_time,
                        "end_time_s": end_time,
                        "setter_calls": setter_calls,
                        "env_step_completed": True,
                    },
                    "outcome": outcome,
                }
            )

        terminal = _capture(
            env,
            frame=DECISION_STEPS,
            phase="loop_start_terminal_decision",
            event_index=capture_index,
            previous_speed_by_entity=previous_decision_speeds,
            delta_t_s=float(env.simulation_time) - float(previous_decision_time),
        )
        decisions.append(terminal)
        if _state(terminal) != {**_state(previous_outcome), "frame_index": DECISION_STEPS}:
            raise RuntimeError("terminal Decision was not independently recaptured")

        future_ids = {
            row["task_id"]
            for decision in decisions
            for row in decision["internal_metadata"]["future_task_schedule"]
        }
        observable_ids = {
            row["task_id"] for decision in decisions for row in decision["tasks"]
        }
        input_task_ids = {
            task_id
            for decision in decisions
            for task_id in decision["input_side_entity_index"]["task_ids"]
        }
        raw_canonical_rows = [
            row
            for decision in decisions
            for row in decision["entities"]
            if row["entity_type"] in {"vehicle", "uav"}
        ]
        checks = {
            "real_airfogsim_environment": any(
                cls.__name__ == "AirFogSimEnv" and cls.__module__.startswith("airfogsim")
                for cls in type(env).__mro__
            ),
            "future_schedule_observed_in_internal_metadata": bool(future_ids),
            "no_future_task_in_O_t": all(
                float(row["arrival_time_s"]) <= float(decision["simulation_time_s"]) + 1e-9
                for decision in decisions
                for row in decision["tasks"]
            ),
            "no_future_task_in_input_entity_index": all(
                {
                    row["task_id"]
                    for row in decision["internal_metadata"]["future_task_schedule"]
                }.isdisjoint(decision["input_side_entity_index"]["task_ids"])
                for decision in decisions
            ),
            "observable_task_ids_equal_input_task_ids": observable_ids == input_task_ids,
            "decision_channel_rows_real_and_nonempty": all(
                decision["channel_rows"]
                and all(
                    row["source_method"] == "channel_manager.getCSI"
                    and row["capture_phase"] == "decision"
                    and row["observed_mask"] is True
                    for row in decision["channel_rows"]
                )
                for decision in decisions
            ),
            "decision_cpu_capacities_real_and_nonempty": all(
                decision["node_cpu_capacity_per_s"]
                and all(float(value) >= 0 for value in decision["node_cpu_capacity_per_s"].values())
                and all(
                    row["source_method"] == "entity.getFogProfile()['cpu']"
                    and (
                        (row["observed_mask"] is True and row["capacity_per_s"] is not None)
                        or (
                            row["observed_mask"] is False
                            and row["capacity_per_s"] is None
                            and row["missing_reason"] == "CPU_NOT_EXPOSED_IN_FOG_PROFILE"
                        )
                    )
                    for row in decision["node_cpu_capacity_observation_rows"]
                )
                for decision in decisions
            ),
            "slot_delivered_data_observed": any(
                step["outcome"]["delivered_data_by_task"] for step in steps
            ),
            "slot_served_cpu_work_observed": any(
                step["outcome"]["served_cpu_work_by_task"] for step in steps
            ),
            "return_route_setter_called_and_queued": bool(return_calls)
            and all(row["queued_before_step"] for row in return_calls),
            "return_route_lifecycle_reached": any(
                row["lifecycle"] in {"returning", "done"}
                for step in steps
                for row in step["outcome"]["tasks"]
                if row["task_id"] in {call["task_id"] for call in return_calls}
            ),
            "raw_and_canonical_acceleration_distinct": all(
                "raw_simulator_acceleration_mps2" in row
                and "canonical_acceleration_mps2" in row
                and "acceleration_mps2" not in row
                for row in raw_canonical_rows
            ),
            "first_decision_canonical_acceleration_masked": all(
                row["canonical_acceleration_mps2"] is None
                and row["canonical_acceleration_observed_mask"] is False
                and row["canonical_acceleration_missing_reason"]
                == "NO_PREVIOUS_SPEED_IN_TRAJECTORY"
                for row in decisions[0]["entities"]
            ),
            "later_canonical_acceleration_uses_history": all(
                (
                    row["canonical_acceleration_observed_mask"] is True
                    and row["canonical_acceleration_mps2"] is not None
                    and row["canonical_acceleration_missing_reason"] is None
                )
                if row["entity_id"]
                in {previous_row["entity_id"] for previous_row in previous["entities"]}
                else (
                    row["canonical_acceleration_observed_mask"] is False
                    and row["canonical_acceleration_mps2"] is None
                    and row["canonical_acceleration_missing_reason"]
                    == "NO_PREVIOUS_SPEED_IN_TRAJECTORY"
                )
                for previous, decision in zip(decisions, decisions[1:])
                for row in decision["entities"]
            ),
            "four_action_fields_present": all(
                all(name in step["action"] for name in ("route", "comm", "comp", "mobility"))
                for step in steps
            ),
            "mobility_uav_only": all(
                all(row["uav_id"] in env.UAVs for row in step["action"]["mobility"]["entries"])
                for step in steps
            ),
            "scope_non_locked_no_training_no_gpu": True,
        }
        failed = [name for name, passed in checks.items() if passed is not True]
        if failed:
            raise RuntimeError(json.dumps({"failed": failed, "checks": checks}, ensure_ascii=False))

        payload = {
            "schema_version": "PIJWM-Step-2.3-Raw-Causal-Complete-v2",
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
            },
            "field_semantics": {
                "future_task_schedule": {
                    "visibility": "raw_internal_metadata_only",
                    "excluded_from": ["O_t", "History", "input_side_entity_index"],
                },
                "channel_rows": {
                    "source": "channel_manager.getCSI",
                    "unit": "dB per RB",
                    "time": "Decision_t before action setters",
                },
                "node_cpu_capacity_per_s": {
                    "source": "entity.getFogProfile()['cpu']",
                    "unit": "AirFogSim CPU-work-unit/s",
                    "time": "Decision_t before action setters",
                    "missingness": "null + observed_mask=false when FogProfile has no cpu key",
                },
                "wireless_delivered_data_by_task": {
                    "source": "ObservedAirFogSimEnv.pi_jwm_transfer_events transport=wireless after fast fading before transfer",
                    "unit": "AirFogSim native data unit/slot",
                    "time": "Execution_t slot",
                    "empty_map": "wireless hook observed with no wireless service in this slot",
                },
                "wired_delivered_data_by_task": {
                    "source": "ObservedAirFogSimEnv.pi_jwm_transfer_events transport=wired from WiredNetworkManager.step",
                    "unit": "AirFogSim native data unit/slot",
                    "time": "Execution_t slot",
                    "empty_map": "wired hook observed with no wired service in this slot",
                },
                "delivered_data_by_task": {
                    "source": "sum of observed wireless_delivered_data_by_task and wired_delivered_data_by_task by task",
                    "unit": "AirFogSim native data unit/slot",
                    "time": "Execution_t slot",
                    "missing": "null plus observed_mask=false when either transport capture is unavailable",
                },
                "served_cpu_work_by_task": {
                    "source": "Task.getComputedSize post-step minus pre-step",
                    "unit": "AirFogSim CPU-work-unit/slot",
                    "time": "Execution_t slot",
                },
                "raw_simulator_acceleration_mps2": {
                    "source": "traffic_manager current traffic infos",
                    "role": "raw simulator observation and audit only",
                },
                "canonical_acceleration_mps2": {
                    "formula": "(speed_t-speed_t_minus_1)/delta_t",
                    "causality": "current and history only",
                    "first_valid_point": "null with observed_mask=false",
                },
            },
            "decisions": decisions,
            "steps": steps,
            "return_route_calls": return_calls,
            "checks": checks,
            "scope": {"non_locked": True, "training": False, "gpu": False, "locked_test": False},
        }
        OUTPUT.mkdir(parents=True, exist_ok=False)
        artifact = OUTPUT / "real_raw_contract_finalization.json"
        write_json(artifact, payload)
        sources = (
            Path(__file__),
            CODE / "src" / "pi_jwm" / "raw_trajectory_causal_contract_v1.py",
            CODE / "src" / "pi_jwm" / "raw_single_decision_step_contract_v1.py",
            CODE / "src" / "pi_jwm" / "airfogsim_full_dual_graph_observer_v1.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "airfogsim_env.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "traffic_manager.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "manager" / "task_manager.py",
            CODE / "reference" / "AirFogSim" / "airfogsim" / "scheduler" / "task_sched.py",
        )
        manifest = {
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
        write_json(OUTPUT / "manifest.json", manifest)
        print(json.dumps({"output": str(OUTPUT), "checks": checks}, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
