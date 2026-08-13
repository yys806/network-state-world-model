"""Deterministic non-training coverage policy for the v4 collector."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence


COVERAGE_POLICY_VERSION = "PIJWM-Balanced-Coverage-v1"
BALANCED_TWO_ARM_VERSION = "balanced_two_arm_v1"
TARGET_FAMILIES = ("local", "nearest_remote", "capacity_remote")


@dataclass(frozen=True)
class TargetCandidate:
    node_id: str
    is_local: bool
    distance: float
    available_cpu: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TargetCandidate":
        return cls(
            node_id=str(value["node_id"]),
            is_local=bool(value["is_local"]),
            distance=float(value["distance"]),
            available_cpu=float(value["available_cpu"]),
        )


@dataclass(frozen=True)
class TargetChoice:
    requested_family: str
    executed_family: str | None
    target_node_id: str | None
    fallback: bool
    reason: str


@dataclass(frozen=True)
class WirelessFlowRequest:
    flow_id: str
    hop_id: str
    transmitter_id: str
    receiver_id: str


@dataclass(frozen=True)
class FlowResourceDecision:
    flow_id: str
    hop_id: str
    transmitter_id: str
    receiver_id: str
    selected: bool
    reason: str
    rb_indices: tuple[int, ...]


def _candidate_rows(candidates: Sequence[TargetCandidate | Mapping[str, object]]) -> tuple[TargetCandidate, ...]:
    rows = tuple(
        candidate if isinstance(candidate, TargetCandidate) else TargetCandidate.from_mapping(candidate)
        for candidate in candidates
    )
    if len({row.node_id for row in rows}) != len(rows):
        raise ValueError("duplicate target candidate")
    for row in rows:
        if not row.node_id.strip():
            raise ValueError("target node_id must be non-empty")
        if row.distance < 0 or row.available_cpu < 0:
            raise ValueError("target candidate values must be nonnegative")
    return rows


def _requested_family(trajectory_id: str, task_id: str, route_revision: int) -> str:
    if not all(isinstance(value, str) and value.strip() for value in (trajectory_id, task_id)):
        raise ValueError("trajectory_id and task_id must be non-empty")
    if isinstance(route_revision, bool) or not isinstance(route_revision, int) or route_revision < 0:
        raise ValueError("route_revision must be a nonnegative integer")
    digest = hashlib.sha256(
        f"{trajectory_id}\0{task_id}\0{route_revision}".encode("utf-8")
    ).digest()
    return TARGET_FAMILIES[int.from_bytes(digest[:8], "big") % 3]


def target_family_for_ordinal(ordinal: int) -> str:
    """Return the explicit 1:1:1 coverage family for a stable task ordinal."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a nonnegative integer")
    return TARGET_FAMILIES[ordinal % len(TARGET_FAMILIES)]


def choose_target_family(
    *,
    trajectory_id: str,
    task_id: str,
    route_revision: int,
    candidates: Sequence[TargetCandidate | Mapping[str, object]],
    requested_family: str | None = None,
) -> TargetChoice:
    rows = _candidate_rows(candidates)
    requested = requested_family or _requested_family(trajectory_id, task_id, route_revision)
    if requested not in set(TARGET_FAMILIES):
        raise ValueError("requested_family must be local, nearest_remote or capacity_remote")

    family_rows = {
        "local": tuple(row for row in rows if row.is_local),
        "nearest_remote": tuple(row for row in rows if not row.is_local),
        "capacity_remote": tuple(row for row in rows if not row.is_local),
    }
    order = ("local", "nearest_remote", "capacity_remote")
    selected_family = requested
    fallback = False
    if not family_rows[selected_family]:
        fallback = True
        selected_family = next((family for family in order if family_rows[family]), None)
    if selected_family is None:
        return TargetChoice(requested, None, None, fallback, "no_legal_target")

    if selected_family == "local":
        ranked = sorted(family_rows[selected_family], key=lambda row: row.node_id)
    elif selected_family == "nearest_remote":
        ranked = sorted(
            family_rows[selected_family],
            key=lambda row: (row.distance, -row.available_cpu, row.node_id),
        )
    else:
        ranked = sorted(
            family_rows[selected_family],
            key=lambda row: (-row.available_cpu, row.distance, row.node_id),
        )
    return TargetChoice(
        requested_family=requested,
        executed_family=selected_family,
        target_node_id=ranked[0].node_id,
        fallback=fallback,
        reason="target_family_fallback" if fallback else "selected",
    )


def choose_resource_arm(trajectory_id: str, seed: int) -> str:
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        raise ValueError("trajectory_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    digest = hashlib.sha256(f"{trajectory_id}\0{seed}".encode("utf-8")).digest()
    return "orthogonal" if digest[0] % 2 == 0 else "interference_reuse"


def _requests(rows: Sequence[WirelessFlowRequest]) -> tuple[WirelessFlowRequest, ...]:
    values = tuple(rows)
    if len({row.flow_id for row in values}) != len(values):
        raise ValueError("duplicate flow request")
    if any(
        not isinstance(row, WirelessFlowRequest)
        or not all(isinstance(value, str) and value.strip() for value in (row.flow_id, row.hop_id, row.transmitter_id, row.receiver_id))
        for row in values
    ):
        raise ValueError("invalid wireless flow request")
    return tuple(sorted(values, key=lambda row: (row.flow_id, row.hop_id, row.transmitter_id, row.receiver_id)))


def allocate_rb_coverage(
    requests: Sequence[WirelessFlowRequest],
    *,
    n_rb: int,
    arm: str,
) -> tuple[FlowResourceDecision, ...]:
    if isinstance(n_rb, bool) or not isinstance(n_rb, int) or n_rb <= 0:
        raise ValueError("n_rb must be a positive integer")
    if arm not in {"orthogonal", "interference_reuse"}:
        raise ValueError("arm must be orthogonal or interference_reuse")
    rows = _requests(requests)
    used_global: set[int] = set()
    used_by_transmitter: dict[str, set[int]] = {}
    decisions: list[FlowResourceDecision] = []
    for row in rows:
        transmitter_used = used_by_transmitter.setdefault(row.transmitter_id, set())
        if arm == "orthogonal":
            available = [rb for rb in range(n_rb) if rb not in used_global and rb not in transmitter_used]
        else:
            available = [rb for rb in range(n_rb) if rb not in transmitter_used]
        if not available:
            decisions.append(
                FlowResourceDecision(
                    row.flow_id,
                    row.hop_id,
                    row.transmitter_id,
                    row.receiver_id,
                    False,
                    "rb_budget_exhausted",
                    (),
                )
            )
            continue
        chosen = (available[0],)
        transmitter_used.update(chosen)
        used_global.update(chosen)
        decisions.append(
            FlowResourceDecision(
                row.flow_id,
                row.hop_id,
                row.transmitter_id,
                row.receiver_id,
                True,
                "selected",
                chosen,
            )
        )
    return tuple(decisions)


def overlapping_rbs(decisions: Sequence[FlowResourceDecision]) -> tuple[int, ...]:
    counts: dict[int, int] = {}
    for decision in decisions:
        if not decision.selected:
            continue
        for rb in decision.rb_indices:
            counts[rb] = counts.get(rb, 0) + 1
    return tuple(sorted(rb for rb, count in counts.items() if count > 1))


def has_same_transmitter_rb_conflict(decisions: Sequence[FlowResourceDecision]) -> bool:
    seen: set[tuple[str, int]] = set()
    for decision in decisions:
        if not decision.selected:
            continue
        for rb in decision.rb_indices:
            key = (decision.transmitter_id, rb)
            if key in seen:
                return True
            seen.add(key)
    return False
