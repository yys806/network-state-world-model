"""Planner v1 operational action domain; CPU-only, causal and rollout-free.

Static compute budget and six UAV profiles are researcher decisions. They are
not AirFogSim native feasibility or safety guarantees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, CandidatePool, Constraint,
    ConstraintStatus, PlannerCandidateContext, validate_step,
)

CPU_UNIT = "AirFogSim CPU-work-unit/s"
CPU_POLICY = "STATIC_PER_SLOT_BUDGET_V1"
MOB_POLICY = "FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1"
CPU_SOURCE = "Raw decision.node_cpu_capacity_observation_rows; entity.getFogProfile()['cpu']"
MOB_SOURCE = "Raw decision.entities UAV heading/heading_unit/elevation_rad"
PROFILES = (
    ("PROFILE_HOLD", 0.0, 0.0),
    ("PROFILE_1", -0.2, 5.0),
    ("PROFILE_2", -0.1, 8.0),
    ("PROFILE_3", 0.05, 10.0),
    ("PROFILE_4", 0.1, 12.0),
    ("PROFILE_5", 0.2, 15.0),
)


@dataclass(frozen=True)
class PlannerComputeBudgetEvidence:
    node_id: str
    static_capacity_per_s: float | None
    observed_mask: bool
    source: str = CPU_SOURCE
    unit: str = CPU_UNIT
    missing_reason: str | None = None
    observation_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.unit != CPU_UNIT:
            raise ValueError("static CPU budget requires raw CPU-work-unit/s")
        if self.observed_mask:
            if self.static_capacity_per_s is None or not math.isfinite(self.static_capacity_per_s) or self.static_capacity_per_s < 0:
                raise ValueError("observed raw static CPU capacity must be finite and nonnegative")
        elif self.static_capacity_per_s is not None:
            raise ValueError("unobserved static CPU capacity cannot carry a value")


@dataclass(frozen=True)
class PlannerMobilityControlState:
    entity_id: str
    uav_index: int
    heading_rad: float | None
    elevation_rad: float | None
    heading_observed: bool
    elevation_observed: bool
    source: str = MOB_SOURCE
    heading_unit: str | None = "rad"
    observation_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.heading_observed and (self.heading_unit != "rad" or self.heading_rad is None or not math.isfinite(self.heading_rad)):
            raise ValueError("observed UAV heading requires finite radians")
        if self.elevation_observed and (self.elevation_rad is None or not math.isfinite(self.elevation_rad)):
            raise ValueError("observed UAV elevation requires finite radians")


@dataclass(frozen=True)
class PlannerActionDomainContext:
    compute_budgets: Mapping[str, PlannerComputeBudgetEvidence]
    mobility_states: Mapping[int, PlannerMobilityControlState]
    causal_provenance: str
    dynamic_available_cpu_available: bool = False
    planner_requires_dynamic_available_cpu: bool = False
    compute_feasibility_policy: str = CPU_POLICY
    raw_decision_identity: str | None = None

    def __post_init__(self) -> None:
        if self.dynamic_available_cpu_available or self.planner_requires_dynamic_available_cpu or self.compute_feasibility_policy != CPU_POLICY:
            raise ValueError("Planner v1 cannot fabricate or require dynamic available CPU")


def context_from_current_raw(base: PlannerCandidateContext, decision: Mapping[str, Any]) -> PlannerActionDomainContext:
    """Consume only the current Raw decision; never infer missing observations."""
    if base.history:
        anchor = base.history[-1]
        if "frame_index" in anchor and int(decision.get("frame_index", -1)) != int(anchor["frame_index"]):
            raise ValueError("Raw decision frame differs from causal Sample anchor")
        if "simulation_time_s" in anchor and not math.isclose(
                float(decision.get("simulation_time_s", float("nan"))), float(anchor["simulation_time_s"]),
                rel_tol=0, abs_tol=1e-9):
            raise ValueError("Raw decision time differs from causal Sample anchor")
    physical = {str(k): int(v) for k, v in base.static["input_entity_index"]["physical"].items()}
    raw_identity = ":".join(str(decision.get(key, "UNSPECIFIED")) for key in
                            ("trajectory_id", "capture_event_id", "frame_index", "simulation_time_s"))
    budget_rows = {str(row["node_id"]): row for row in decision.get("node_cpu_capacity_observation_rows", ())}
    if len(budget_rows) != len(decision.get("node_cpu_capacity_observation_rows", ())):
        raise ValueError("duplicate current static CPU observation")
    budgets = {}
    for node_id in physical:
        row = budget_rows.get(node_id)
        observed = bool(row and row.get("observed_mask") and row.get("capacity_per_s") is not None)
        budgets[node_id] = PlannerComputeBudgetEvidence(
            node_id, float(row["capacity_per_s"]) if observed else None, observed,
            missing_reason=None if observed else (row.get("missing_reason") if row else "CPU_OBSERVATION_ROW_MISSING"),
            observation_provenance=raw_identity,
        )
    # Sample static is a causal, raw-unit cross-check; its normalized_value is ignored.
    for row in base.static.get("agent_static_capability", ()):
        node_id = str(row["entity_id"])
        if node_id not in physical or int(row["agent_index"]) != physical[node_id]:
            raise ValueError("Sample CPU identity differs from current support")
        wrapped = row["cpu_capacity_per_s"]
        if wrapped.get("unit") not in (None, CPU_UNIT):
            raise ValueError("Sample static CPU capacity must use raw CPU-work-unit/s")
        if budgets[node_id].observed_mask and wrapped.get("observed_mask") and wrapped.get("value") is not None:
            if not math.isclose(float(wrapped["value"]), budgets[node_id].static_capacity_per_s, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Raw and Sample static CPU capacity disagree")
    states = {}
    for row in decision.get("entities", ()):
        if row.get("entity_type") != "uav":
            continue
        entity_id = str(row["entity_id"])
        if entity_id not in physical:
            raise ValueError("Raw UAV outside current fixed support")
        slot = physical[entity_id]
        if not (slot < base.current_state["uav_mask"].shape[1] and
                bool(base.current_state["uav_mask"][0, slot]) and
                bool(base.current_state["entity_presence"][0, slot])):
            raise ValueError("Raw UAV differs from current present UAV support")
        if slot in states:
            raise ValueError("duplicate current UAV control state")
        unit = row.get("heading_unit")
        if unit not in (None, "rad"):
            raise ValueError("UAV heading unit must be rad; no silent conversion")
        heading = row.get("heading") if unit == "rad" else None
        elevation = row.get("elevation_rad")
        states[slot] = PlannerMobilityControlState(
            entity_id, slot, float(heading) if heading is not None else None,
            float(elevation) if elevation is not None else None,
            heading is not None, elevation is not None, heading_unit=unit,
            observation_provenance=raw_identity,
        )
    return PlannerActionDomainContext(budgets, states, base.causal_provenance, raw_decision_identity=raw_identity)


@dataclass(frozen=True)
class DomainResult:
    constraints: tuple[Constraint, ...]
    joint_behavior_support_class: tuple[str, ...]
    final_mobility_states: Mapping[int, PlannerMobilityControlState] = field(default_factory=dict)

    @property
    def status(self) -> ConstraintStatus:
        if any(c.status is ConstraintStatus.VIOLATED for c in self.constraints):
            return ConstraintStatus.VIOLATED
        if any(c.status is ConstraintStatus.UNKNOWN for c in self.constraints):
            return ConstraintStatus.UNKNOWN
        return ConstraintStatus.SATISFIED


def _constraint(name: str, status: ConstraintStatus, reason: str, source: str) -> Constraint:
    return Constraint(name, status, reason, source)


def _with_provenance(source: str, provenance: str | None) -> str:
    return f"{source} @ {provenance}" if provenance else source


def _profile(row: Mapping[str, Any], state: PlannerMobilityControlState) -> str | None:
    if not (state.heading_observed and state.elevation_observed):
        return None
    angle, elevation, speed = (float(row[key]) for key in ("azimuth_rad", "elevation_rad", "speed_mps"))
    # These controls originate from binary floats in the collection policy.
    # Tolerance only absorbs arithmetic roundoff, never interpolated controls.
    if not math.isclose(elevation, state.elevation_rad, rel_tol=0, abs_tol=1e-9):
        return None
    for name, delta, velocity in PROFILES:
        if math.isclose(angle, state.heading_rad + delta, rel_tol=0, abs_tol=1e-9) and math.isclose(speed, velocity, rel_tol=0, abs_tol=1e-9):
            return name
    return None


def validate_domain_sequence(candidate: CandidateActionSequence, base: PlannerCandidateContext,
                             domain: PlannerActionDomainContext) -> DomainResult:
    if candidate.provenance != base.causal_provenance or domain.causal_provenance != base.causal_provenance:
        raise ValueError("causal provenance mismatch")
    current = dict(domain.mobility_states)
    constraints = []
    joint_classes = []
    physical = base.static["input_entity_index"]["physical"]
    for time_index, step in enumerate(candidate.steps):
        # The generic 6.0A contract raises on negative Comp. The operational
        # domain retains a named VIOLATED result so the source/reason is auditable.
        negative_rows = tuple(row for row in step.comp if float(row["allocated_cpu_per_s"]) < 0)
        if negative_rows:
            validate_step(replace(step, comp=tuple(row for row in step.comp if row not in negative_rows)), base)
        else:
            validate_step(step, base)
        by_node: dict[str, float] = {}
        for row in step.comp:
            node_id = str(row["node_id"])
            amount = float(row["allocated_cpu_per_s"])
            if amount < 0:
                constraints.append(_constraint(CPU_POLICY, ConstraintStatus.VIOLATED, "NEGATIVE_COMP_ALLOCATION", CPU_SOURCE))
                continue
            by_node[node_id] = by_node.get(node_id, 0.0) + amount
        for node_id, requested in by_node.items():
            evidence = domain.compute_budgets.get(node_id)
            if node_id not in physical or evidence is None or evidence.node_id != node_id or not evidence.observed_mask:
                status, reason = (ConstraintStatus.VIOLATED, "STATIC_CPU_CAPACITY_UNOBSERVED") if requested > 0 else (ConstraintStatus.SATISFIED, "ZERO_COMP_REQUEST")
            elif evidence.unit != CPU_UNIT:
                status, reason = ConstraintStatus.VIOLATED, "STATIC_CPU_RAW_UNIT_REQUIRED"
            elif requested <= evidence.static_capacity_per_s:
                status, reason = ConstraintStatus.SATISFIED, "WITHIN_STATIC_PER_SLOT_BUDGET"
            else:
                status, reason = ConstraintStatus.VIOLATED, "STATIC_PER_SLOT_BUDGET_EXCEEDED"
            constraints.append(_constraint(f"{CPU_POLICY}[{time_index},{node_id}]", status, reason,
                                           _with_provenance(evidence.source if evidence else CPU_SOURCE,
                                                            evidence.observation_provenance if evidence else domain.raw_decision_identity)))
        present_uavs = {int(i) for i in range(base.current_state["uav_mask"].shape[1])
                        if bool(base.current_state["uav_mask"][0, i]) and bool(base.current_state["entity_presence"][0, i])}
        commanded = {int(row["uav_index"]) for row in step.mob}
        if commanded != present_uavs:
            constraints.append(_constraint(f"{MOB_POLICY}[{time_index}]", ConstraintStatus.VIOLATED,
                                           "NO_MOBILITY_COMMAND_IS_NOT_HOLD", MOB_SOURCE))
        used_profiles = []
        for row in step.mob:
            slot = int(row["uav_index"])
            state = current.get(slot)
            if state is None or not state.heading_observed or not state.elevation_observed:
                constraints.append(_constraint(f"{MOB_POLICY}[{time_index},{slot}]", ConstraintStatus.UNKNOWN,
                                               "CURRENT_UAV_CONTROL_STATE_UNOBSERVED",
                                               _with_provenance(MOB_SOURCE, state.observation_provenance if state else domain.raw_decision_identity)))
                continue
            name = _profile(row, state)
            if name is None:
                constraints.append(_constraint(f"{MOB_POLICY}[{time_index},{slot}]", ConstraintStatus.VIOLATED,
                                               "OUTSIDE_PLANNER_MOBILITY_CORE_DOMAIN_V1",
                                               _with_provenance("STEP_5_5_COLLECTION_POLICY_AND_RESEARCHER_DECISION", state.observation_provenance)))
                continue
            used_profiles.append(name)
            constraints.append(_constraint(f"{MOB_POLICY}[{time_index},{slot}]", ConstraintStatus.SATISFIED,
                                           name, _with_provenance("STEP_5_5_COLLECTION_POLICY_AND_RESEARCHER_DECISION",
                                                                  state.observation_provenance)))
            current[slot] = replace(state, heading_rad=float(row["azimuth_rad"]),
                                    elevation_rad=float(row["elevation_rad"]),
                                    source="previous Planner mobility command; control side-state only")
        joint_classes.append("EXACT_COLLECTION_SHARED_PROFILE" if len(set(used_profiles)) <= 1 and len(used_profiles) == len(present_uavs)
                             else "PER_UAV_MARGINAL_SUPPORT_COMPOSITION" if len(used_profiles) == len(present_uavs)
                             else "UNRESOLVED_OR_INCOMPLETE")
    return DomainResult(tuple(constraints), tuple(joint_classes), current)


def require_domain_admissible(candidate: CandidateActionSequence, base: PlannerCandidateContext,
                              domain: PlannerActionDomainContext) -> DomainResult:
    result = validate_domain_sequence(candidate, base, domain)
    if result.status is not ConstraintStatus.SATISFIED:
        reasons = ",".join(c.reason_code for c in result.constraints if c.status is not ConstraintStatus.SATISFIED)
        raise ValueError(f"Planner v1 domain {result.status.value}: {reasons}")
    return result


def annotate_domain(candidate: CandidateActionSequence, base: PlannerCandidateContext,
                    domain: PlannerActionDomainContext) -> CandidateActionSequence:
    result = require_domain_admissible(candidate, base, domain)
    return replace(candidate, constraints=tuple(dict.fromkeys((*candidate.constraints, *result.constraints))),
        generation_metadata={**candidate.generation_metadata,
        "planner_action_domain": "V1", "compute_feasibility_policy": CPU_POLICY,
        "dynamic_available_cpu_available": False, "planner_requires_dynamic_available_cpu": False,
        "domain_observation_provenance": domain.raw_decision_identity,
        "joint_behavior_support_class": list(result.joint_behavior_support_class)})


def rule_fallback_v1(base: PlannerCandidateContext, domain: PlannerActionDomainContext,
                     horizon: int, seed: int = 0) -> CandidateActionSequence:
    if not 1 <= horizon <= 4:
        raise ValueError("candidate horizon must be 1..4")
    uav_slots = [i for i in range(base.current_state["uav_mask"].shape[1])
                 if bool(base.current_state["uav_mask"][0, i]) and bool(base.current_state["entity_presence"][0, i])]
    rows = []
    for slot in uav_slots:
        state = domain.mobility_states.get(slot)
        if state is None or not state.heading_observed or not state.elevation_observed:
            raise ValueError("RULE_FALLBACK requires observed current UAV heading and elevation")
        rows.append({"uav_index": slot, "azimuth_rad": state.heading_rad,
                     "elevation_rad": state.elevation_rad, "speed_mps": 0.0})
    candidate = CandidateActionSequence(tuple(CandidateActionStep(mob=tuple(rows)) for _ in range(horizon)),
        "rule_hold_v1", Backend.RULE_FALLBACK, seed, base.causal_provenance,
        frozenset({"RULE_FALLBACK"}), generation_metadata={
            "fallback_constraint_scope": "Planner v1 operational domain and current fixed support",
            "safe": False, "unverified_safety_conditions": ["spatial/geofence", "future feasibility", "safety"]})
    return annotate_domain(candidate, base, domain)


@dataclass(frozen=True)
class DomainPoolAdmission:
    pool: CandidatePool
    rejected: tuple[Mapping[str, str], ...]


def admit_candidate_pool_v1(source_pool: CandidatePool, base: PlannerCandidateContext,
                            domain: PlannerActionDomainContext, horizon: int, seed: int = 0) -> DomainPoolAdmission:
    """Apply one operational gate to all backends; report each rejected seed.

    This performs no model scoring, ranking, optimization or action execution.
    """
    admitted = CandidatePool()
    rejected = []
    for candidate in source_pool.candidates:
        try:
            if candidate.horizon != horizon:
                raise ValueError("candidate horizon differs from requested horizon")
            admitted.add(annotate_domain(candidate, base, domain))
        except ValueError as error:
            rejected.append({"candidate_id": candidate.candidate_id, "fingerprint": candidate.fingerprint,
                             "reason": str(error)})
    admitted.add(rule_fallback_v1(base, domain, horizon, seed))
    return DomainPoolAdmission(admitted, tuple(sorted(rejected, key=lambda row: (row["fingerprint"], row["candidate_id"]))))
