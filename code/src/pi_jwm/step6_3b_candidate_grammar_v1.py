"""CPU-only, search-independent single-step Planner candidate grammar.

One caller supplies a causal or predicted state at each horizon. This module
does not choose a candidate, call the World Model, score, or rank alternatives.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from pi_jwm.cpu_inner_rule_v1 import CpuTaskDemand, allocate_work_conserving_cpu
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB
from pi_jwm.step4_2a_graph_input_extension_v1 import TASK_AGENT_RELATION_TYPE_VOCAB
from pi_jwm.step6_0a_candidate_generation_v1 import (
    CandidateActionSequence, CandidateActionStep, PlannerCandidateContext,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import (
    PROFILES, PlannerActionDomainContext, PlannerMobilityControlState, _profile,
)
from pi_jwm.step6_3b_candidate_support_v1 import (
    CandidateSupportLabel, Signature, TrainStructuralSupportCatalog,
)


@dataclass(frozen=True)
class CommBlockChoice:
    task_id: str
    start_rb: int
    width: int


@dataclass(frozen=True)
class StructuredStepChoice:
    comm: tuple[CommBlockChoice, ...]
    comp_mode: str  # NOOP, SCALE_0.5, SCALE_0.75, SCALE_1.0
    mobility_profile: str  # PROFILE_HOLD or PROFILE_1..PROFILE_5


@dataclass(frozen=True)
class BoundStructuredStep:
    action: CandidateActionStep
    structural_signature: Signature
    support: CandidateSupportLabel
    next_mobility_control: Mapping[int, PlannerMobilityControlState]
    binding: Mapping[str, object]


@dataclass(frozen=True)
class StructuredCandidateAdmission:
    candidate_id: str
    candidate_fingerprint: str
    horizon: int
    admitted: bool
    reason_codes: tuple[str, ...]
    per_step: tuple[BoundStructuredStep, ...]
    h_sup: int | None  # remains unknown until an actual candidate rollout


class CandidateGrammarViolation(ValueError):
    """A proposed structural choice cannot bind to this causal state."""


def _int(state: Mapping[str, torch.Tensor], name: str, slot: int) -> int:
    return int(state[name][0, slot])


def _bool(state: Mapping[str, torch.Tensor], name: str, slot: int) -> bool:
    return bool(state[name][0, slot])


def _wireless_bindings(state: Mapping[str, torch.Tensor],
                       task_ids: Mapping[str, int]) -> dict[str, int]:
    """Return only Tasks with one valid existing wireless Flow binding."""
    inverse = {int(slot): str(task_id) for task_id, slot in task_ids.items()}
    candidates: dict[str, list[int]] = {}
    unresolved: set[str] = set()
    for flow in range(state["flow_presence"].shape[1]):
        if not (_bool(state, "flow_known", flow) and _bool(state, "flow_presence", flow)
                and _bool(state, "carrying_active", flow)):
            continue
        task_slot = _int(state, "flow_task_index", flow)
        if task_slot not in inverse or not _bool(state, "task_presence", task_slot):
            continue
        task_id = inverse[task_slot]
        relation = _int(state, "flow_comm_relation_index", flow)
        if relation < 0:
            unresolved.add(task_id)
            continue
        if relation >= state["comm_presence"].shape[1] or not (
                _bool(state, "comm_presence", relation)
                and _bool(state, "comm_validity", relation)):
            unresolved.add(task_id)
            continue
        if (_int(state, "comm_source_index", relation) != _int(state, "carrying_hop_source_index", flow)
                or _int(state, "comm_target_index", relation) != _int(state, "carrying_hop_destination_index", flow)):
            unresolved.add(task_id)
            continue
        if not _bool(state, "comm_wireless_mask", relation):
            continue  # wired service has no RB action row
        candidates.setdefault(task_id, []).append(relation)
    return {task_id: relations[0] for task_id, relations in candidates.items()
            if task_id not in unresolved and len(relations) == 1}


def _compute_base(state: Mapping[str, torch.Tensor], task_ids: Mapping[str, int],
                  physical_ids: Mapping[str, int], domain: PlannerActionDomainContext,
                  slot_seconds: float) -> dict[str, tuple[str, float]]:
    computing = LIFECYCLE_VOCAB.index("computing")
    exec_type = TASK_AGENT_RELATION_TYPE_VOCAB.index("exec")
    inverse_nodes = {int(slot): str(node_id) for node_id, slot in physical_ids.items()}
    demands = []
    capacities = {}
    for task_id, task_slot in task_ids.items():
        task_slot = int(task_slot)
        if not (_bool(state, "task_presence", task_slot)
                and _int(state, "task_lifecycle_index", task_slot) == computing
                and not _bool(state, "task_completed", task_slot)):
            continue
        remaining = float(state["task_work_remaining"][0, task_slot])
        if remaining < 0:
            raise CandidateGrammarViolation("NEGATIVE_PREDICTED_COMPUTE_WORK")
        matches = [r for r in range(state["task_agent_validity"].shape[1])
                   if _bool(state, "task_agent_validity", r)
                   and _int(state, "task_agent_task_index", r) == task_slot
                   and _int(state, "task_agent_relation_type_index", r) == exec_type]
        if len(matches) != 1:
            raise CandidateGrammarViolation("COMPUTING_TASK_EXEC_BINDING_NOT_UNIQUE")
        node_slot = _int(state, "task_agent_agent_index", matches[0])
        if node_slot not in inverse_nodes or not _bool(state, "entity_presence", node_slot):
            raise CandidateGrammarViolation("COMPUTING_NODE_OUTSIDE_CAUSAL_SUPPORT")
        node_id = inverse_nodes[node_slot]
        evidence = domain.compute_budgets.get(node_id)
        if evidence is None or not evidence.observed_mask or evidence.static_capacity_per_s is None:
            raise CandidateGrammarViolation("STATIC_CPU_CAPACITY_UNOBSERVED")
        capacities[node_id] = evidence.static_capacity_per_s
        demands.append(CpuTaskDemand(str(task_id), node_id, remaining))
    if not demands:
        return {}
    decision = allocate_work_conserving_cpu(demands, capacities, slot_seconds)
    # The Formal collector records every current computing Task row, even if
    # this deterministic rule assigns it zero under a saturated node budget.
    return {row.task_id: (row.node_id, row.allocated_cpu)
            for row in decision.allocations}


def bind_structured_step(choice: StructuredStepChoice,
                         context: PlannerCandidateContext,
                         domain: PlannerActionDomainContext,
                         state: Mapping[str, torch.Tensor],
                         mobility_control: Mapping[int, PlannerMobilityControlState],
                         catalog: TrainStructuralSupportCatalog,
                         prior_signatures: Sequence[Signature] = (), *,
                         slot_seconds: float = 0.1) -> BoundStructuredStep:
    """Construct one action from this step's state and label its TRAIN support."""
    if not 0 <= len(prior_signatures) < 4 or slot_seconds <= 0:
        raise ValueError("H1-H4 and positive slot duration required")
    indices = context.static["input_entity_index"]
    tasks = indices["task"]
    physical = indices["physical"]
    wireless = _wireless_bindings(state, tasks)
    n_rb = int(state["rb_active_mask"].shape[-1])
    if len(choice.comm) > 4:
        raise CandidateGrammarViolation("COMM_ROW_COUNT_OUTSIDE_TRAIN_SUPPORT")
    if any(block.width not in (1, 2, 3) or not 0 <= block.start_rb < n_rb
           for block in choice.comm):
        raise CandidateGrammarViolation("COMM_BLOCK_OUTSIDE_TRAIN_CORE")
    comm_signature_hint = ("NOOP" if not choice.comm else
                           "rows:" + ",".join(map(str, sorted(row.width for row in choice.comm))))
    selected_tasks = {row.task_id for row in choice.comm}
    if comm_signature_hint not in catalog.comm_selected_task_counts:
        raise CandidateGrammarViolation("COMM_STRUCTURAL_SIGNATURE_UNSEEN_IN_TRAIN")
    allowed_counts = catalog.comm_selected_task_counts.get(comm_signature_hint, frozenset())
    if len(selected_tasks) not in allowed_counts:
        raise CandidateGrammarViolation("COMM_SELECTED_TASK_COUNT_OUTSIDE_TRAIN_SUPPORT")
    if any(task_id not in wireless for task_id in selected_tasks):
        raise CandidateGrammarViolation("COMM_TASK_OUTSIDE_CAUSAL_ELIGIBILITY")
    comm_rows = []
    for block in choice.comm:
        if (block.start_rb, block.width) not in catalog.comm_start_width_pairs:
            raise CandidateGrammarViolation("COMM_START_WIDTH_PAIR_UNSEEN_IN_TRAIN")
        comm_rows.append({"task_id": block.task_id, "task_index": int(tasks[block.task_id]),
                          "relation_index": wireless[block.task_id],
                          "rb_indices": sorted({(block.start_rb + offset) % n_rb
                                                for offset in range(block.width)})})
    # A canonical order makes candidate fingerprints independent of method output order.
    comm_rows.sort(key=lambda row: (row["task_id"], tuple(row["rb_indices"])))
    base = _compute_base(state, tasks, physical, domain, slot_seconds)
    if not base:
        if choice.comp_mode != "NOOP":
            raise CandidateGrammarViolation("COMP_SCALE_WITHOUT_CAUSAL_BASE")
        comp_rows = []
        comp_signature = "NOOP"
    else:
        alpha = {"SCALE_0.5": 0.5, "SCALE_0.75": 0.75, "SCALE_1.0": 1.0}.get(choice.comp_mode)
        if alpha is None:
            raise CandidateGrammarViolation("ELIGIBLE_COMP_REQUIRES_TRAIN_ALPHA")
        comp_rows = [{"task_id": task_id, "node_id": node_id,
                      "allocated_cpu_per_s": allocation * alpha}
                     for task_id, (node_id, allocation) in sorted(base.items())]
        comp_signature = f"tasks:{len(comp_rows)};alpha:{alpha}"

    profiles = {name: (delta, speed) for name, delta, speed in PROFILES}
    if choice.mobility_profile not in profiles:
        raise CandidateGrammarViolation("MOBILITY_PROFILE_OUTSIDE_CORE")
    present_uavs = [slot for slot in range(state["uav_mask"].shape[1])
                    if _bool(state, "uav_mask", slot) and _bool(state, "entity_presence", slot)]
    if not present_uavs:
        if choice.mobility_profile != "PROFILE_HOLD":
            raise CandidateGrammarViolation("MOBILITY_PROFILE_WITHOUT_PRESENT_UAV")
        mob_rows = []
        mob_signature = "NOOP"
        next_control = dict(mobility_control)
    else:
        delta, speed = profiles[choice.mobility_profile]
        mob_rows = []
        next_control = dict(mobility_control)
        for slot in present_uavs:
            control = mobility_control.get(slot)
            if control is None or not control.heading_observed or not control.elevation_observed:
                raise CandidateGrammarViolation("CURRENT_UAV_CONTROL_STATE_UNOBSERVED")
            heading = float(control.heading_rad) + delta
            elevation = float(control.elevation_rad)
            mob_rows.append({"uav_index": slot, "azimuth_rad": heading,
                             "elevation_rad": elevation, "speed_mps": speed})
            next_control[slot] = replace(control, heading_rad=heading, elevation_rad=elevation,
                                         source="previous Planner mobility command; control side-state only")
        name = "HOLD" if choice.mobility_profile == "PROFILE_HOLD" else choice.mobility_profile
        mob_signature = ",".join(name for _ in present_uavs)
    comm_signature = ("NOOP" if not comm_rows else
                      "rows:" + ",".join(map(str, sorted(len(row["rb_indices"]) for row in comm_rows))))
    signature: Signature = (comm_signature, comp_signature, mob_signature)
    support = catalog.label(signature, prior_signatures)
    action = CandidateActionStep(route=(), comm=tuple(comm_rows), comp=tuple(comp_rows), mob=tuple(mob_rows))
    binding = {"wireless_task_to_relation": dict(sorted(wireless.items())),
               "compute_base": {task: {"node_id": node, "allocation_per_s": amount}
                                for task, (node, amount) in sorted(base.items())},
               "present_uav_slots": present_uavs,
               "current_state_only": True,
               "rb_reuse_between_tasks_allowed": True}
    return BoundStructuredStep(action, signature, support, next_control, binding)


def _decode_choice(step: CandidateActionStep, context: PlannerCandidateContext,
                   domain: PlannerActionDomainContext,
                   state: Mapping[str, torch.Tensor],
                   mobility_control: Mapping[int, PlannerMobilityControlState],
                   slot_seconds: float) -> StructuredStepChoice:
    if step.route:
        raise CandidateGrammarViolation("ROUTE_MUST_BE_EXPLICIT_NOOP")
    n_rb = int(state["rb_active_mask"].shape[-1])
    blocks = []
    for row in step.comm:
        rb = tuple(int(value) for value in row["rb_indices"])
        if not rb or len(set(rb)) != len(rb):
            raise CandidateGrammarViolation("COMM_RB_ROW_EMPTY_OR_DUPLICATED")
        starts = [start for start in range(n_rb)
                  if tuple(sorted((start + offset) % n_rb for offset in range(len(rb)))) == tuple(sorted(rb))]
        if len(starts) != 1:
            raise CandidateGrammarViolation("COMM_ROW_NOT_UNIQUE_CYCLIC_BLOCK")
        blocks.append(CommBlockChoice(str(row["task_id"]), starts[0], len(rb)))
    if not step.comp:
        mode = "NOOP"
    else:
        indices = context.static["input_entity_index"]
        base = _compute_base(state, indices["task"], indices["physical"], domain, slot_seconds)
        if not base or len(step.comp) != len(base) or {str(row["task_id"]) for row in step.comp} != set(base):
            raise CandidateGrammarViolation("COMP_ROWS_DIFFER_FROM_CAUSAL_BASE_TASK_SET")
        ratios = []
        for row in step.comp:
            task_id = str(row["task_id"])
            node, amount = base[task_id]
            requested = float(row["allocated_cpu_per_s"])
            if str(row["node_id"]) != node or not math.isfinite(requested) or requested < 0:
                raise CandidateGrammarViolation("COMP_NODE_OR_AMOUNT_INVALID")
            if amount > 0:
                ratios.append(requested / amount)
            elif requested != 0:
                raise CandidateGrammarViolation("COMP_POSITIVE_REQUEST_WITH_ZERO_BASE")
        if not ratios:
            raise CandidateGrammarViolation("COMP_ALPHA_UNIDENTIFIABLE_FROM_ZERO_BASE")
        alpha = next((value for value in (0.5, 0.75, 1.0)
                      if all(math.isclose(ratio, value, rel_tol=0, abs_tol=1e-7) for ratio in ratios)), None)
        if alpha is None:
            raise CandidateGrammarViolation("COMP_MIXED_OR_OUTSIDE_TRAIN_ALPHA")
        mode = f"SCALE_{alpha}"

    present = {slot for slot in range(state["uav_mask"].shape[1])
               if _bool(state, "uav_mask", slot) and _bool(state, "entity_presence", slot)}
    if {int(row["uav_index"]) for row in step.mob} != present or len(step.mob) != len(present):
        raise CandidateGrammarViolation("MOBILITY_ROWS_DIFFER_FROM_PRESENT_UAV_SET")
    observed_profiles = []
    for row in step.mob:
        control = mobility_control.get(int(row["uav_index"]))
        if control is None:
            raise CandidateGrammarViolation("MOBILITY_CONTROL_STATE_UNOBSERVED")
        name = _profile(row, control)
        if name is None:
            raise CandidateGrammarViolation("MOBILITY_ROW_OUTSIDE_SIX_PROFILES")
        observed_profiles.append(name)
    if len(set(observed_profiles)) > 1:
        raise CandidateGrammarViolation("ASYMMETRIC_MOBILITY_OUTSIDE_PLANNER_V1")
    profile = observed_profiles[0] if observed_profiles else "PROFILE_HOLD"
    return StructuredStepChoice(tuple(blocks), mode, profile)


def admit_structured_candidate(candidate: CandidateActionSequence,
                               context: PlannerCandidateContext,
                               domain: PlannerActionDomainContext,
                               predicted_input_states: Sequence[Mapping[str, torch.Tensor]],
                               catalog: TrainStructuralSupportCatalog, *,
                               slot_seconds: float = 0.1) -> StructuredCandidateAdmission:
    """Validate any backend's supplied candidate with the same grammar/policy.

    The caller provides S_t,...,S_(t+H-1) from one causal recursive rollout.
    The grammar never queries a Future Target or executes the World Model.
    """
    if candidate.provenance != context.causal_provenance or domain.causal_provenance != context.causal_provenance:
        raise ValueError("causal provenance mismatch")
    if len(predicted_input_states) != candidate.horizon:
        raise ValueError("one predicted input state per candidate step required")
    control = dict(domain.mobility_states)
    accepted: list[BoundStructuredStep] = []
    for step, state in zip(candidate.steps, predicted_input_states):
        try:
            choice = _decode_choice(step, context, domain, state, control, slot_seconds)
            bound = bind_structured_step(choice, context, domain, state, control, catalog,
                                         tuple(item.structural_signature for item in accepted),
                                         slot_seconds=slot_seconds)
            if not bound.support.formal_pool_admitted:
                raise CandidateGrammarViolation(",".join(bound.support.reason_codes))
            # Bound rows must match the supplied action's causal identities and
            # numeric values; a structural label alone cannot license it.
            if len(step.comm) != len(bound.action.comm) or len(step.comp) != len(bound.action.comp):
                raise CandidateGrammarViolation("ACTION_ROW_COUNT_DIFFERS_FROM_GRAMMAR")
            actual_comm = sorted((str(row["task_id"]), int(row["task_index"]),
                                  int(row["relation_index"]), tuple(sorted(map(int, row["rb_indices"]))))
                                 for row in step.comm)
            expected_comm = sorted((str(row["task_id"]), int(row["task_index"]),
                                    int(row["relation_index"]), tuple(row["rb_indices"]))
                                   for row in bound.action.comm)
            if actual_comm != expected_comm:
                raise CandidateGrammarViolation("COMM_CAUSAL_RELATION_BINDING_DIFFERS")
            actual_comp = sorted((str(row["task_id"]), str(row["node_id"]), float(row["allocated_cpu_per_s"]))
                                 for row in step.comp)
            expected_comp = sorted((str(row["task_id"]), str(row["node_id"]), float(row["allocated_cpu_per_s"]))
                                   for row in bound.action.comp)
            if len(actual_comp) != len(expected_comp) or any(
                    a[:2] != b[:2] or not math.isclose(a[2], b[2], rel_tol=0, abs_tol=1e-7)
                    for a, b in zip(actual_comp, expected_comp)):
                raise CandidateGrammarViolation("COMP_ALLOCATION_DIFFERS_FROM_CAUSAL_RULE")
            # Candidate identity includes row order and exact JSON numeric
            # representation. Require the canonical grammar output so two
            # backends cannot assign different fingerprints to one choice.
            canonical = lambda item: json.dumps(item.frame(), sort_keys=True,
                                                separators=(",", ":"), allow_nan=False)
            if canonical(step) != canonical(bound.action):
                raise CandidateGrammarViolation("CANDIDATE_ACTION_NOT_CANONICAL_GRAMMAR_OUTPUT")
        except (CandidateGrammarViolation, KeyError, IndexError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, CandidateGrammarViolation) else "MALFORMED_OR_MISSING_CAUSAL_FIELD"
            return StructuredCandidateAdmission(candidate.candidate_id, candidate.fingerprint,
                                                candidate.horizon, False, (reason,), tuple(accepted), None)
        accepted.append(bound)
        control = dict(bound.next_mobility_control)
    return StructuredCandidateAdmission(candidate.candidate_id, candidate.fingerprint,
                                        candidate.horizon, True, (), tuple(accepted), None)
