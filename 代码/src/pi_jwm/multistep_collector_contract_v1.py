"""Pure identity and history contracts for the P2 multistep smoke."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from .information_edge_contract_v4 import MissingReason


MULTISTEP_CONTRACT_VERSION = "PIJWM-P2-Multistep-Collector-v1"


def _identifiers(values: Iterable[str], *, kind: str) -> tuple[str, ...]:
    rows = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in rows):
        raise ValueError(f"{kind} identifiers must be non-empty strings")
    if len(set(rows)) != len(rows):
        raise ValueError(f"duplicate {kind} identifier")
    return rows


@dataclass(frozen=True)
class EdgeIdentity:
    edge_id: str
    source_id: str
    target_id: str
    edge_class: str


@dataclass(frozen=True)
class VocabularySnapshot:
    node_indices: dict[str, int]
    edge_indices: dict[str, int]
    flow_indices: dict[str, int]
    node_presence: tuple[bool, ...]
    edge_presence: tuple[bool, ...]
    flow_presence: tuple[bool, ...]


class TrajectoryVocabulary:
    """Maintain stable append-only identities across trajectory frames."""

    def __init__(self) -> None:
        self._node_indices: dict[str, int] = {}
        self._edge_indices: dict[str, int] = {}
        self._flow_indices: dict[str, int] = {}
        self._edge_bindings: dict[str, EdgeIdentity] = {}
        self._last_snapshot = VocabularySnapshot({}, {}, {}, (), (), ())

    def observe(
        self,
        *,
        node_ids: Iterable[str],
        edges: Iterable[EdgeIdentity],
        flow_ids: Iterable[str],
    ) -> VocabularySnapshot:
        nodes = _identifiers(node_ids, kind="node")
        flows = _identifiers(flow_ids, kind="flow")
        edge_rows = tuple(edges)
        if any(not isinstance(edge, EdgeIdentity) for edge in edge_rows):
            raise TypeError("edges must contain EdgeIdentity values")
        edge_ids = _identifiers((edge.edge_id for edge in edge_rows), kind="edge")

        present_nodes = set(nodes)
        for edge in edge_rows:
            _identifiers(
                (edge.source_id, edge.target_id, edge.edge_class), kind="edge binding"
            )
            if edge.source_id not in present_nodes or edge.target_id not in present_nodes:
                raise ValueError(f"edge endpoint is not present: {edge.edge_id}")
            previous = self._edge_bindings.get(edge.edge_id)
            if previous is not None and previous != edge:
                raise ValueError(f"edge binding changed: {edge.edge_id}")

        next_nodes = dict(self._node_indices)
        next_edges = dict(self._edge_indices)
        next_flows = dict(self._flow_indices)
        next_bindings = dict(self._edge_bindings)
        for node_id in sorted(set(nodes) - set(next_nodes)):
            next_nodes[node_id] = len(next_nodes)
        for edge in sorted(edge_rows, key=lambda row: row.edge_id):
            if edge.edge_id not in next_edges:
                next_edges[edge.edge_id] = len(next_edges)
                next_bindings[edge.edge_id] = edge
        for flow_id in sorted(set(flows) - set(next_flows)):
            next_flows[flow_id] = len(next_flows)

        present_edges = set(edge_ids)
        present_flows = set(flows)
        snapshot = VocabularySnapshot(
            node_indices=next_nodes,
            edge_indices=next_edges,
            flow_indices=next_flows,
            node_presence=tuple(
                node_id in present_nodes
                for node_id, _ in sorted(next_nodes.items(), key=lambda row: row[1])
            ),
            edge_presence=tuple(
                edge_id in present_edges
                for edge_id, _ in sorted(next_edges.items(), key=lambda row: row[1])
            ),
            flow_presence=tuple(
                flow_id in present_flows
                for flow_id, _ in sorted(next_flows.items(), key=lambda row: row[1])
            ),
        )
        self._node_indices = next_nodes
        self._edge_indices = next_edges
        self._flow_indices = next_flows
        self._edge_bindings = next_bindings
        self._last_snapshot = snapshot
        return snapshot

    def snapshot(self) -> VocabularySnapshot:
        return VocabularySnapshot(
            node_indices=dict(self._last_snapshot.node_indices),
            edge_indices=dict(self._last_snapshot.edge_indices),
            flow_indices=dict(self._last_snapshot.flow_indices),
            node_presence=tuple(self._last_snapshot.node_presence),
            edge_presence=tuple(self._last_snapshot.edge_presence),
            flow_presence=tuple(self._last_snapshot.flow_presence),
        )


@dataclass(frozen=True)
class LinkOutcome:
    active_flow_count: float
    effective_rate_per_s: float
    served_data: float

    def values(self) -> tuple[float, float, float]:
        return (
            self.active_flow_count,
            self.effective_rate_per_s,
            self.served_data,
        )


@dataclass(frozen=True)
class ProjectedLinkHistory:
    edge_id: str
    values: tuple[float, float, float]
    valid: bool
    missing_reason: MissingReason


@dataclass(frozen=True)
class LinkHistorySnapshot:
    last_frame_index: int
    outcomes: tuple[tuple[str, LinkOutcome], ...]


class LinkHistoryLedger:
    """Commit only complete validated frames for next-frame projection."""

    def __init__(self, *, edge_ids: Iterable[str]) -> None:
        self._edge_ids = _identifiers(edge_ids, kind="edge")
        self._last_frame_index = -1
        self._outcomes: dict[str, LinkOutcome] | None = None

    def project(self, *, edge_ids: Iterable[str]) -> tuple[ProjectedLinkHistory, ...]:
        requested = _identifiers(edge_ids, kind="edge")
        unknown = sorted(set(requested) - set(self._edge_ids))
        if unknown:
            raise ValueError(f"unknown edge identifiers: {unknown}")
        rows = []
        for edge_id in requested:
            if self._outcomes is None:
                rows.append(
                    ProjectedLinkHistory(
                        edge_id=edge_id,
                        values=(0.0, 0.0, 0.0),
                        valid=False,
                        missing_reason=MissingReason.NO_HISTORY,
                    )
                )
            else:
                outcome = self._outcomes[edge_id]
                rows.append(
                    ProjectedLinkHistory(
                        edge_id=edge_id,
                        values=outcome.values(),
                        valid=True,
                        missing_reason=MissingReason.NONE,
                    )
                )
        return tuple(rows)

    def commit(
        self,
        *,
        frame_index: int,
        outcomes: Mapping[str, LinkOutcome],
        frame_validated: bool,
    ) -> None:
        if frame_validated is not True:
            raise ValueError("frame must be validated before history commit")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("frame index must be an integer")
        if frame_index != self._last_frame_index + 1:
            raise ValueError("frame indices must be contiguous")
        unknown = sorted(set(outcomes) - set(self._edge_ids))
        if unknown:
            raise ValueError(f"unknown edge identifiers: {unknown}")
        missing = sorted(set(self._edge_ids) - set(outcomes))
        if missing:
            raise ValueError(f"missing edge outcomes: {missing}")

        normalized: dict[str, LinkOutcome] = {}
        for edge_id in self._edge_ids:
            outcome = outcomes[edge_id]
            if not isinstance(outcome, LinkOutcome):
                raise TypeError("outcomes must contain LinkOutcome values")
            values = outcome.values()
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
                for value in values
            ):
                raise ValueError("link outcomes must be nonnegative finite values")
            normalized[edge_id] = LinkOutcome(*(float(value) for value in values))

        self._outcomes = normalized
        self._last_frame_index = frame_index

    def snapshot(self) -> LinkHistorySnapshot:
        rows = () if self._outcomes is None else tuple(
            (edge_id, self._outcomes[edge_id]) for edge_id in self._edge_ids
        )
        return LinkHistorySnapshot(self._last_frame_index, rows)
