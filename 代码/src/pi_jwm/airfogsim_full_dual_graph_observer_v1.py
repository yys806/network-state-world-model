"""Direct AirFogSim snapshots for the PI-JWM full dual-graph collector."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isclose
from typing import Callable, Iterator

import numpy as np

from .full_dual_graph_collector_contract_v1 import (
    CollectorContractError,
    DagEdge,
    PhysicalEdge,
    PhysicalNode,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
)


OBSERVER_VERSION = "PIJWM-AirFogSim-Full-Observer-v1"


@dataclass(frozen=True)
class AirFogSimSnapshot:
    phase: SnapshotPhase
    simulation_time: float
    nodes: tuple[PhysicalNode, ...]
    physical_edges: tuple[PhysicalEdge, ...]
    tasks: tuple[TaskSnapshot, ...]
    dag_edges: tuple[DagEdge, ...]
    channel_rows: tuple[dict[str, object], ...]


_NODE_COLLECTIONS = (
    ("vehicles", "V"),
    ("UAVs", "U"),
    ("RSUs", "I"),
    ("cloudServers", "C"),
)
_TASK_COLLECTIONS = (
    ("_to_generate_task_infos", TaskLifecycle.TO_GENERATE),
    ("_waiting_to_offload_tasks", TaskLifecycle.WAITING_TO_OFFLOAD),
    ("_offloading_tasks", TaskLifecycle.OFFLOADING),
    ("_computing_tasks", TaskLifecycle.COMPUTING),
    ("_waiting_to_return_tasks", TaskLifecycle.WAITING_TO_RETURN),
    ("_returning_tasks", TaskLifecycle.RETURNING),
    ("_done_tasks", TaskLifecycle.DONE),
    ("_out_of_ddl_tasks", TaskLifecycle.FAILED),
)


def _plain_vector(value: object) -> tuple[float, ...]:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise CollectorContractError(
            "invalid_channel_observation", "getCSI returned empty or nonfinite values"
        )
    return tuple(float(item) for item in array)


def _extract_nodes(env) -> tuple[PhysicalNode, ...]:
    rows: list[PhysicalNode] = []
    for collection_name, node_type in _NODE_COLLECTIONS:
        collection = getattr(env, collection_name)
        for node_id in sorted(collection):
            position = tuple(float(value) for value in collection[node_id].getPosition())
            rows.append(
                PhysicalNode(
                    node_id=str(node_id),
                    node_type=node_type,
                    present=True,
                    position=position,
                )
            )
    return tuple(sorted(rows, key=lambda row: row.node_id))


def _physical_structure(
    env, nodes: tuple[PhysicalNode, ...], phase: SnapshotPhase
) -> tuple[tuple[PhysicalEdge, ...], tuple[dict[str, object], ...]]:
    node_types = {row.node_id: row.node_type for row in nodes}
    wireless_ids = sorted(
        node_id for node_id, node_type in node_types.items() if node_type in "VUI"
    )
    edges: list[PhysicalEdge] = []
    channel_rows: list[dict[str, object]] = []
    for source_id in wireless_ids:
        for target_id in wireless_ids:
            if source_id == target_id:
                continue
            source_type = node_types[source_id]
            target_type = node_types[target_id]
            edge_type = f"{source_type}2{target_type}"
            edge_id = f"physical::{source_id}::{target_id}::{edge_type}"
            edges.append(
                PhysicalEdge(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    present=True,
                )
            )
            # Immediately after traffic movement, node indices are new while
            # AirFogSim fast-fading arrays still belong to the preceding slot.
            # The execution snapshot therefore records structure only; outcome
            # channel evidence is captured after the communication refresh.
            if phase != SnapshotPhase.EXECUTION:
                csi = env.channel_manager.getCSI(
                    env._getNodeIdxById(source_id),
                    env._getNodeIdxById(target_id),
                    source_type,
                    target_type,
                )
                channel_rows.append(
                    {
                        "physical_edge_id": edge_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "channel_type": edge_type,
                        "rb_indices": tuple(range(int(env.channel_manager.n_RB))),
                        "channel_attenuation_db": _plain_vector(csi),
                        "capture_phase": phase.value,
                        "simulation_time": float(env.simulation_time),
                        "source_method": "channel_manager.getCSI",
                    }
                )

    wired_ids = sorted(
        node_id for node_id, node_type in node_types.items() if node_type in "IC"
    )
    for source_id in wired_ids:
        for target_id in wired_ids:
            if source_id == target_id or not env.wired_manager.hasLink(source_id, target_id):
                continue
            edges.append(
                PhysicalEdge(
                    edge_id=f"physical::{source_id}::{target_id}::wired",
                    source_id=source_id,
                    target_id=target_id,
                    edge_type="wired",
                    present=True,
                )
            )
    return (
        tuple(sorted(edges, key=lambda row: row.edge_id)),
        tuple(sorted(channel_rows, key=lambda row: str(row["physical_edge_id"]))),
    )


def _flatten_task_collection(collection: object) -> list[object]:
    if not isinstance(collection, dict):
        raise CollectorContractError(
            "invalid_task_collection", "AirFogSim task collection must be a dictionary"
        )
    tasks: list[object] = []
    for owner_id in sorted(collection, key=str):
        tasks.extend(collection[owner_id])
    return tasks


def _extract_tasks(env) -> tuple[TaskSnapshot, ...]:
    task_manager = env.task_manager
    by_id: dict[str, TaskSnapshot] = {}
    for collection_name, lifecycle in _TASK_COLLECTIONS:
        collection = getattr(task_manager, collection_name)
        for task in _flatten_task_collection(collection):
            task_id = str(task.getTaskId())
            if task_id in by_id:
                raise CollectorContractError(
                    "duplicate_task_lifecycle",
                    f"task {task_id} occurs in more than one lifecycle collection",
                )
            by_id[task_id] = TaskSnapshot(
                task_id=task_id,
                task_node_id=str(task.getTaskNodeId()),
                lifecycle=lifecycle,
                current_node_id=str(task.getCurrentNodeId()),
                route_nodes=tuple(str(node_id) for node_id in task.getToOffloadRoute()),
                return_destination_id=(
                    None
                    if task.getToReturnNodeId() is None
                    else str(task.getToReturnNodeId())
                ),
                arrival_time=float(task.getTaskArrivalTime()),
            )
    return tuple(by_id[task_id] for task_id in sorted(by_id))


def _extract_dag_edges(env) -> tuple[DagEdge, ...]:
    rows: list[DagEdge] = []
    dependencies = env.task_manager._task_dependencies
    for task_node_id in sorted(dependencies, key=str):
        graph = dependencies[task_node_id]
        for source_task_id, target_task_id in sorted(graph.edges(), key=lambda edge: tuple(map(str, edge))):
            source = str(source_task_id)
            target = str(target_task_id)
            rows.append(
                DagEdge(
                    dag_edge_id=f"dag::{task_node_id}::{source}::{target}",
                    source_task_id=source,
                    target_task_id=target,
                    communication_mapping="not_modeled",
                )
            )
    return tuple(rows)


def observe_airfogsim_snapshot(
    env, *, phase: SnapshotPhase
) -> AirFogSimSnapshot:
    """Read one direct, non-mutating snapshot from AirFogSim."""

    if not isinstance(phase, SnapshotPhase):
        raise CollectorContractError("invalid_snapshot_phase", "phase is not SnapshotPhase")
    nodes = _extract_nodes(env)
    physical_edges, channel_rows = _physical_structure(env, nodes, phase)
    return AirFogSimSnapshot(
        phase=phase,
        simulation_time=float(env.simulation_time),
        nodes=nodes,
        physical_edges=physical_edges,
        tasks=_extract_tasks(env),
        dag_edges=_extract_dag_edges(env),
        channel_rows=channel_rows,
    )


@contextmanager
def capture_execution_snapshot(
    env, observer: Callable[[], AirFogSimSnapshot]
) -> Iterator[list[AirFogSimSnapshot]]:
    """Capture once after traffic movement and before AirFogSim task updates."""

    traffic_interval = float(env.traffic_interval)
    simulation_interval = float(env.simulation_interval)
    if simulation_interval <= 0 or not isclose(
        traffic_interval / simulation_interval, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CollectorContractError(
            "multi_substep_not_supported",
            "traffic_interval / simulation_interval must equal 1",
        )
    if not callable(observer):
        raise TypeError("observer must be callable")

    captures: list[AirFogSimSnapshot] = []
    original_update_traffics = env._updateTraffics

    def observed_update_traffics():
        result = original_update_traffics()
        if captures:
            raise CollectorContractError(
                "duplicate_execution_snapshot", "traffic hook ran more than once"
            )
        captures.append(observer())
        return result

    env._updateTraffics = observed_update_traffics
    try:
        yield captures
    finally:
        env._updateTraffics = original_update_traffics
