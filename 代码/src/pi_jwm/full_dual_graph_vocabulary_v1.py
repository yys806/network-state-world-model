"""Append-only identities for the PI-JWM v4 full dual-graph collector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .full_dual_graph_collector_contract_v1 import (
    CarryingHop,
    DagEdge,
    LogicalFlow,
    PhysicalEdge,
    PhysicalNode,
    TaskSnapshot,
)


VOCABULARY_VERSION = "PIJWM-P2-Full-Dual-Graph-Vocabulary-v1"


def _ids(rows: Iterable[object], *, attr: str, kind: str) -> tuple[str, ...]:
    values = tuple(getattr(row, attr) for row in rows)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{kind} identifiers must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {kind} identifier")
    return values


def _sorted_indices(existing: dict[str, int], observed: Iterable[str]) -> dict[str, int]:
    result = dict(existing)
    for identifier in sorted(set(observed) - set(result)):
        result[identifier] = len(result)
    return result


@dataclass(frozen=True)
class FullVocabularySnapshot:
    node_indices: dict[str, int]
    physical_edge_indices: dict[str, int]
    task_indices: dict[str, int]
    dag_edge_indices: dict[str, int]
    flow_indices: dict[str, int]
    hop_indices: dict[str, int]
    node_presence: tuple[bool, ...]
    physical_edge_presence: tuple[bool, ...]
    task_presence: tuple[bool, ...]
    dag_edge_presence: tuple[bool, ...]
    flow_presence: tuple[bool, ...]
    hop_presence: tuple[bool, ...]


class FullTrajectoryVocabulary:
    """Maintain six isolated append-only identity spaces transactionally."""

    def __init__(self) -> None:
        self._node_indices: dict[str, int] = {}
        self._physical_edge_indices: dict[str, int] = {}
        self._task_indices: dict[str, int] = {}
        self._dag_edge_indices: dict[str, int] = {}
        self._flow_indices: dict[str, int] = {}
        self._hop_indices: dict[str, int] = {}
        self._physical_edge_bindings: dict[str, PhysicalEdge] = {}
        self._dag_edge_bindings: dict[str, DagEdge] = {}
        self._flow_bindings: dict[str, LogicalFlow] = {}
        self._hop_bindings: dict[str, CarryingHop] = {}
        self._last_snapshot = FullVocabularySnapshot({}, {}, {}, {}, {}, {}, (), (), (), (), (), ())

    def observe(
        self,
        *,
        nodes: Sequence[PhysicalNode],
        physical_edges: Sequence[PhysicalEdge],
        tasks: Sequence[TaskSnapshot],
        dag_edges: Sequence[DagEdge],
        flows: Sequence[LogicalFlow],
        hops: Sequence[CarryingHop],
    ) -> FullVocabularySnapshot:
        node_ids = _ids(nodes, attr="node_id", kind="node")
        edge_ids = _ids(physical_edges, attr="edge_id", kind="physical edge")
        task_ids = _ids(tasks, attr="task_id", kind="task")
        dag_ids = _ids(dag_edges, attr="dag_edge_id", kind="DAG edge")
        flow_ids = _ids(flows, attr="flow_id", kind="flow")
        hop_ids = _ids(hops, attr="hop_id", kind="hop")

        spaces = {
            "node": set(node_ids),
            "physical edge": set(edge_ids),
            "task": set(task_ids),
            "DAG edge": set(dag_ids),
            "flow": set(flow_ids),
            "hop": set(hop_ids),
        }
        names = tuple(spaces)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                shared = spaces[left_name] & spaces[right_name]
                if shared:
                    raise ValueError(f"{right_name} identifier collides with {left_name}: {sorted(shared)}")

        current_nodes = set(node_ids)
        for edge in physical_edges:
            if edge.source_id not in current_nodes or edge.target_id not in current_nodes:
                raise ValueError(f"physical edge endpoint is not present: {edge.edge_id}")
            previous = self._physical_edge_bindings.get(edge.edge_id)
            if previous is not None and previous != edge:
                raise ValueError(f"physical edge binding changed: {edge.edge_id}")
        for dag_edge in dag_edges:
            previous = self._dag_edge_bindings.get(dag_edge.dag_edge_id)
            if previous is not None and previous != dag_edge:
                raise ValueError(f"DAG edge binding changed: {dag_edge.dag_edge_id}")
        for flow in flows:
            previous = self._flow_bindings.get(flow.flow_id)
            if previous is not None and previous != flow:
                raise ValueError(f"flow binding changed: {flow.flow_id}")
        for hop in hops:
            if hop.flow_id not in set(flow_ids) and hop.hop_id not in self._hop_bindings:
                raise ValueError(f"hop references unobserved flow: {hop.hop_id}")
            previous = self._hop_bindings.get(hop.hop_id)
            if previous is not None and previous != hop:
                raise ValueError(f"hop binding changed: {hop.hop_id}")

        next_nodes = _sorted_indices(self._node_indices, node_ids)
        next_edges = _sorted_indices(self._physical_edge_indices, edge_ids)
        next_tasks = _sorted_indices(self._task_indices, task_ids)
        next_dags = _sorted_indices(self._dag_edge_indices, dag_ids)
        next_flows = _sorted_indices(self._flow_indices, flow_ids)
        next_hops = _sorted_indices(self._hop_indices, hop_ids)
        next_edge_bindings = dict(self._physical_edge_bindings)
        next_edge_bindings.update({edge.edge_id: edge for edge in physical_edges})
        next_dag_bindings = dict(self._dag_edge_bindings)
        next_dag_bindings.update({edge.dag_edge_id: edge for edge in dag_edges})
        next_flow_bindings = dict(self._flow_bindings)
        next_flow_bindings.update({flow.flow_id: flow for flow in flows})
        next_hop_bindings = dict(self._hop_bindings)
        next_hop_bindings.update({hop.hop_id: hop for hop in hops})

        snapshot = FullVocabularySnapshot(
            node_indices=dict(next_nodes),
            physical_edge_indices=dict(next_edges),
            task_indices=dict(next_tasks),
            dag_edge_indices=dict(next_dags),
            flow_indices=dict(next_flows),
            hop_indices=dict(next_hops),
            node_presence=tuple(identifier in current_nodes for identifier, _ in sorted(next_nodes.items(), key=lambda item: item[1])),
            physical_edge_presence=tuple(identifier in set(edge_ids) for identifier, _ in sorted(next_edges.items(), key=lambda item: item[1])),
            task_presence=tuple(identifier in set(task_ids) for identifier, _ in sorted(next_tasks.items(), key=lambda item: item[1])),
            dag_edge_presence=tuple(identifier in set(dag_ids) for identifier, _ in sorted(next_dags.items(), key=lambda item: item[1])),
            flow_presence=tuple(identifier in set(flow_ids) for identifier, _ in sorted(next_flows.items(), key=lambda item: item[1])),
            hop_presence=tuple(identifier in set(hop_ids) for identifier, _ in sorted(next_hops.items(), key=lambda item: item[1])),
        )

        self._node_indices = next_nodes
        self._physical_edge_indices = next_edges
        self._task_indices = next_tasks
        self._dag_edge_indices = next_dags
        self._flow_indices = next_flows
        self._hop_indices = next_hops
        self._physical_edge_bindings = next_edge_bindings
        self._dag_edge_bindings = next_dag_bindings
        self._flow_bindings = next_flow_bindings
        self._hop_bindings = next_hop_bindings
        self._last_snapshot = snapshot
        return snapshot

    def snapshot(self) -> FullVocabularySnapshot:
        current = self._last_snapshot
        return FullVocabularySnapshot(
            node_indices=dict(current.node_indices),
            physical_edge_indices=dict(current.physical_edge_indices),
            task_indices=dict(current.task_indices),
            dag_edge_indices=dict(current.dag_edge_indices),
            flow_indices=dict(current.flow_indices),
            hop_indices=dict(current.hop_indices),
            node_presence=tuple(current.node_presence),
            physical_edge_presence=tuple(current.physical_edge_presence),
            task_presence=tuple(current.task_presence),
            dag_edge_presence=tuple(current.dag_edge_presence),
            flow_presence=tuple(current.flow_presence),
            hop_presence=tuple(current.hop_presence),
        )


class RouteRevisionLedger:
    """Track route revisions independently for offload and return phases."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str, str], tuple[int, tuple[str, ...]]] = {}

    @staticmethod
    def _key(trajectory_id: str, task_id: str, phase: str) -> tuple[str, str, str]:
        if not all(isinstance(value, str) and value.strip() for value in (trajectory_id, task_id, phase)):
            raise ValueError("trajectory, task and phase identifiers must be non-empty")
        if phase not in {"offload", "return"}:
            raise ValueError("phase must be offload or return")
        return trajectory_id, task_id, phase

    def assign(self, trajectory_id: str, task_id: str, phase: str, route: tuple[str, ...]) -> int:
        key = self._key(trajectory_id, task_id, phase)
        if not isinstance(route, tuple) or not route or any(not isinstance(node, str) or not node.strip() for node in route):
            raise ValueError("route must be a non-empty tuple")
        previous = self._routes.get(key)
        if previous is None:
            revision = 0
        elif previous[1] == route:
            return previous[0]
        else:
            revision = previous[0] + 1
        self._routes[key] = (revision, tuple(route))
        return revision

    def import_revision(
        self,
        trajectory_id: str,
        task_id: str,
        phase: str,
        revision: int,
        route: tuple[str, ...],
    ) -> None:
        key = self._key(trajectory_id, task_id, phase)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if not isinstance(route, tuple) or not route:
            raise ValueError("route must be a non-empty tuple")
        previous = self._routes.get(key)
        expected = 0 if previous is None else previous[0] + 1
        if revision != expected:
            raise ValueError("revision must be contiguous")
        self._routes[key] = (revision, tuple(route))

    def snapshot(self) -> dict[tuple[str, str, str], tuple[int, tuple[str, ...]]]:
        return {key: (revision, tuple(route)) for key, (revision, route) in self._routes.items()}
