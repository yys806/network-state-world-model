"""CPU-only candidate structures and adapter for the current four-action contract.

This module intentionally contains no model rollout, scoring, optimizer or training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

import torch


class ConstraintStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


class Backend(str, Enum):
    SEARCH = "SEARCH"
    LEARNED = "LEARNED"
    HYBRID = "HYBRID"
    RULE_FALLBACK = "RULE_FALLBACK"


@dataclass(frozen=True)
class Constraint:
    constraint_name: str
    status: ConstraintStatus
    reason_code: str
    source: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CandidateActionStep:
    # Rows retain the causal Sample action names, IDs, units and route semantics.
    # Empty tuples are explicit no-op; absent/missing action families are forbidden.
    route: tuple[Mapping[str, Any], ...] = ()
    comm: tuple[Mapping[str, Any], ...] = ()
    comp: tuple[Mapping[str, Any], ...] = ()
    mob: tuple[Mapping[str, Any], ...] = ()

    def frame(self) -> dict[str, Any]:
        empty = lambda rows: {"entries": copy.deepcopy(list(rows)), "empty": not bool(rows), "missing": False}
        return {"route": empty(self.route), "comm": empty(self.comm),
                "comp": empty(self.comp), "mobility": empty(self.mob)}


@dataclass(frozen=True)
class CandidateActionSequence:
    steps: tuple[CandidateActionStep, ...]
    candidate_id: str
    generator_backend: Backend
    seed: int | None
    provenance: str
    source_tags: frozenset[str] = frozenset()
    constraints: tuple[Constraint, ...] = ()
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= len(self.steps) <= 4:
            raise ValueError("candidate horizon must be 1..4 (H1-H4 evidence boundary)")
        if any(c.status is ConstraintStatus.VIOLATED for c in self.constraints):
            raise ValueError("known violated constraint rejects candidate")
        _canonical([s.frame() for s in self.steps])

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical([s.frame() for s in self.steps]).encode()).hexdigest()

    @property
    def constraint_status(self) -> ConstraintStatus:
        return ConstraintStatus.UNKNOWN if self.unresolved_constraints else ConstraintStatus.SATISFIED

    @property
    def unresolved_constraints(self) -> tuple[Constraint, ...]:
        unresolved = [c for c in self.constraints if c.status is ConstraintStatus.UNKNOWN]
        if any(step.comp for step in self.steps):
            unresolved.append(Constraint("dynamic_available_cpu", ConstraintStatus.UNKNOWN,
                "NO_RELIABLE_CAUSAL_SOURCE", "STEP_4_2A_SOURCE_AUDIT"))
        if any(step.mob for step in self.steps):
            unresolved.append(Constraint("mobility_numeric_bounds", ConstraintStatus.UNKNOWN,
                "BOUNDS_NOT_FROZEN", "STEP_6_0A_ACTION_AUDIT"))
        return tuple(dict.fromkeys(unresolved))


@dataclass(frozen=True)
class PlannerCandidateContext:
    # Only current/causal fields. No future action, target, observation or label.
    static: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    current_state: Mapping[str, torch.Tensor]
    causal_provenance: str
    constraints: tuple[Constraint, ...] = ()
    latent_reference: Any = None  # Opaque future interface; never consumed by v1.

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any], state: Mapping[str, torch.Tensor], provenance: str) -> "PlannerCandidateContext":
        return cls(copy.deepcopy(sample["static"]), tuple(copy.deepcopy(sample["history"])),
                   state, provenance)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def validate_step(step: CandidateActionStep, context: PlannerCandidateContext) -> None:
    ids = context.static["input_entity_index"]
    physical, tasks, flows = (ids[name] for name in ("physical", "task", "logical_flow"))
    state = context.current_state
    seen: set[tuple[Any, ...]] = set()
    for row in step.route:
        _require(str(row["task_id"]) in tasks, "route task outside current support")
        _require(int(row["task_index"]) == int(tasks[str(row["task_id"])]), "route task identity mismatch")
        _require(row["route_kind"] in ("offload", "return"), "unknown route kind")
        for key in ("task_node_index", "target_node_index"):
            if row.get(key) is not None:
                _require(int(row[key]) in physical.values(), f"route {key} outside current support")
        for node in row.get("route_node_indices", []):
            _require(int(node) in physical.values(), "route path outside current support")
        if row.get("flow_id") is not None:
            _require(str(row["flow_id"]) in flows, "route Flow outside current fixed support")
        if row["route_kind"] == "return":
            _require(any(k.startswith(f"flow::{row['task_id']}::Return::") for k in flows),
                     "future-only Return birth cannot create current Flow slot")
        key = ("route", row["task_id"])
        _require(key not in seen, "duplicate route task")
        seen.add(key)
    for row in step.comm:
        task_id = str(row["task_id"])
        _require(task_id in tasks and int(row["task_index"]) == int(tasks[task_id]), "comm task outside current support")
        relation = int(row["relation_index"])
        _require(0 <= relation < state["comm_presence"].shape[1] and
                 bool(state["comm_presence"][0, relation]) and bool(state["comm_validity"][0, relation]),
                 "Comm relation is not current and valid")
        rb_count = int(state["rb_active_mask"].shape[-1])
        rb = [int(x) for x in row["rb_indices"]]
        _require(bool(rb) and len(rb) == len(set(rb)) and all(0 <= x < rb_count for x in rb), "invalid/duplicate RB assignment")
        key = ("comm", task_id)
        _require(key not in seen, "duplicate comm task assignment")
        seen.add(key)
    for row in step.comp:
        _require(str(row["node_id"]) in physical and str(row["task_id"]) in tasks,
                 "comp node/task outside current support")
        _require(_finite(row["allocated_cpu_per_s"], "allocated CPU") >= 0, "negative Comp allocation")
        key = ("comp", row["node_id"], row["task_id"])
        _require(key not in seen, "duplicate comp assignment")
        seen.add(key)
    for row in step.mob:
        idx = int(row["uav_index"])
        _require(0 <= idx < state["uav_mask"].shape[1] and bool(state["uav_mask"][0, idx])
                 and bool(state["entity_presence"][0, idx]), "mobility entity is not current UAV")
        for key in ("azimuth_rad", "elevation_rad", "speed_mps"):
            _finite(row[key], key)
        key = ("mob", idx)
        _require(key not in seen, "duplicate UAV action")
        seen.add(key)


def compile_candidate(sequence: CandidateActionSequence, context: PlannerCandidateContext,
                      states: tuple[Mapping[str, torch.Tensor], ...], *,
                      synthetic_contract_test_only: bool = False) -> tuple[tuple[dict[str, torch.Tensor], ...], tuple[dict[str, Any], ...]]:
    """Wrap the actual training adapter; later states must be supplied causally.

    UNKNOWN is retained in the pool but cannot be compiled until the researcher
    freezes the policy. No predicted state is manufactured here.
    """
    _require(not any(c.status is ConstraintStatus.VIOLATED for c in context.constraints),
             "known violated context constraint rejects compilation")
    _require(synthetic_contract_test_only or (not sequence.unresolved_constraints and
             not any(c.status is ConstraintStatus.UNKNOWN for c in context.constraints)),
             "UNKNOWN constraint requires researcher policy before compilation")
    _require(len(states) == sequence.horizon, "one causal state required per action step")
    for step in sequence.steps:
        validate_step(step, context)
    sample = {"static": copy.deepcopy(context.static), "history": copy.deepcopy(list(context.history)),
              "future_action": [step.frame() for step in sequence.steps]}
    from build_step5_1d_unified_model_chain_v1 import build_action
    result = [build_action(sample, dict(state), index) for index, state in enumerate(states)]
    for step, (tensor, mapping), state in zip(sequence.steps, result, states):
        for declared, row in zip(step.comm, mapping["comm"]):
            ri = int(row["relation_index"])
            _require(ri >= 0, "Comm relation unresolved or fixed-support blocked")
            _require(ri == int(declared["relation_index"]), "Comm relation identity differs from current adapter")
            _require(bool(state["comm_presence"][0, ri]) and bool(state["comm_validity"][0, ri]),
                     "Comm relation is not current and valid")
        _require(bool(torch.isfinite(tensor["mobility_values"]).all()) and
                 bool(torch.isfinite(tensor["comp_values"]).all()), "nonfinite action tensor")
    return tuple(x[0] for x in result), tuple(x[1] for x in result)


class CandidatePool:
    def __init__(self) -> None:
        self._candidates: dict[str, CandidateActionSequence] = {}

    def add(self, candidate: CandidateActionSequence) -> None:
        key = candidate.fingerprint
        old = self._candidates.get(key)
        if old is None:
            self._candidates[key] = candidate
        else:
            tags = old.source_tags | candidate.source_tags | {old.generator_backend.value, candidate.generator_backend.value}
            constraints = tuple(dict.fromkeys((*old.constraints, *candidate.constraints)))
            sources = list(old.generation_metadata.get("duplicate_sources", [{"candidate_id": old.candidate_id,
                              "provenance": old.provenance, "backend": old.generator_backend.value}]))
            sources.append({"candidate_id": candidate.candidate_id, "provenance": candidate.provenance,
                            "backend": candidate.generator_backend.value})
            sources = sorted({ _canonical(item): item for item in sources }.values(), key=_canonical)
            self._candidates[key] = replace(old, source_tags=frozenset(tags), constraints=constraints,
                generation_metadata={**old.generation_metadata, "duplicate_sources": sources})

    @property
    def candidates(self) -> tuple[CandidateActionSequence, ...]:
        return tuple(self._candidates[key] for key in sorted(self._candidates))

    def __len__(self) -> int:
        return len(self._candidates)


class CandidateGenerator(Protocol):
    def generate(self, context: PlannerCandidateContext, horizon: int, budget: int, seed: int) -> CandidatePool: ...


def rule_fallback(context: PlannerCandidateContext, horizon: int, seed: int = 0) -> CandidateActionSequence:
    return CandidateActionSequence(tuple(CandidateActionStep() for _ in range(horizon)),
        "rule_noop", Backend.RULE_FALLBACK, seed, context.causal_provenance,
        frozenset({"RULE_FALLBACK"}), context.constraints,
        {"fallback_constraint_scope": "structural current support only", "unverified_safety_conditions": ["dynamic CPU", "future feasibility", "safety"]})


class SearchStubBackend:
    development_stub = True

    def generate(self, context: PlannerCandidateContext, horizon: int, budget: int, seed: int) -> CandidatePool:
        _require(budget >= 1, "budget must be positive")
        pool = CandidatePool()
        pool.add(replace(rule_fallback(context, horizon, seed), generator_backend=Backend.SEARCH,
                         candidate_id=f"search_stub_{seed}", source_tags=frozenset({"SEARCH"}),
                         generation_metadata={"development_stub": True, "contract_mechanism_only": True}))
        return pool


class LearnedProposalBackend:
    def __init__(self, proposal: Callable[[PlannerCandidateContext, int, int, int], CandidatePool]):
        self.proposal = proposal

    def generate(self, context: PlannerCandidateContext, horizon: int, budget: int, seed: int) -> CandidatePool:
        return self.proposal(context, horizon, budget, seed)


class HybridCompositionBackend:
    def __init__(self, learned: CandidateGenerator, search: CandidateGenerator):
        self.learned, self.search = learned, search

    def generate(self, context: PlannerCandidateContext, horizon: int, budget: int, seed: int,
                 warm_start: CandidateActionSequence | None = None) -> CandidatePool:
        pool = CandidatePool()
        for backend in (self.learned, self.search):
            for candidate in backend.generate(context, horizon, budget, seed).candidates:
                pool.add(candidate)
        if warm_start is not None:
            pool.add(warm_start)
        pool.add(rule_fallback(context, horizon, seed))
        return pool


def shift_warm_start(previous: CandidateActionSequence, context: PlannerCandidateContext,
                     tail_filler: Callable[[PlannerCandidateContext], CandidateActionStep], seed: int) -> CandidateActionSequence:
    _require(previous.horizon >= 2, "warm start needs a remaining action")
    # The caller supplies a fresh current context. The old first action is discarded.
    tail = tail_filler(context)  # Explicit TAIL_REQUIRED slot filled by deterministic stub/caller.
    for step in (*previous.steps[1:], tail):
        validate_step(step, context)
    return CandidateActionSequence((*previous.steps[1:], tail), f"warm_{previous.candidate_id}",
        previous.generator_backend, seed, context.causal_provenance,
        previous.source_tags | {"WARM_START"}, context.constraints,
        {"warm_start": True, "parent_candidate_id": previous.candidate_id,
         "shift_count": 1, "tail_status": "TAIL_REQUIRED_FILLED_BY_STUB"})
