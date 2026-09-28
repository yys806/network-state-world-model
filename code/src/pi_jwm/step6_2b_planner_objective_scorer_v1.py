"""CPU-only scoring of already supplied, frozen-support candidate rollouts.

No model call, future truth, optimizer, or candidate generation occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from pi_jwm.step4_2a_graph_input_extension_v1 import TASK_AGENT_RELATION_TYPE_VOCAB
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import FLOW_STATUS_VOCAB, FLOW_TYPE_VOCAB
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionSequence
from pi_jwm.step6_1_trained_candidate_rollout_v1 import CandidateRolloutTrace
from pi_jwm.step6_2a_planner_objective_side_state_v1 import (
    PlannerObjectiveCausalSideState, PlannerTaskCausalSideState,
    planner_derived_return_birth_required, support_horizon,
)


class ScorerStateInconsistency(ValueError):
    """Predicted fixed-support state contradicts the frozen Flow/Task contract."""


class BurdenSemanticsBlocked(ValueError):
    """An unfinished task has neither anchor service component."""


@dataclass(frozen=True)
class CandidateObjectiveHorizonRow:
    horizon: int
    deadline_violation: int
    unfinished: int
    burden: float
    effort: float
    tasks: tuple[Mapping[str, Any], ...]
    effort_components: Mapping[str, float]


@dataclass(frozen=True)
class CandidateObjectiveScore:
    candidate_id: str
    candidate_fingerprint: str
    anchor_fingerprint: str
    H_sup: int
    H_eff: int
    N_DDL: int
    A_DDL: float
    J_Delay: float
    J_Burden: float
    J_Effort: float
    objective_tuple: tuple[int, float, float, float, float]
    per_horizon_rows: tuple[CandidateObjectiveHorizonRow, ...]
    per_task_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    support_boundary_reason: str | None
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSetObjectiveResult:
    status: str
    H_eff: int
    scores: tuple[CandidateObjectiveScore, ...]
    support_horizons: Mapping[str, int]
    diagnostics: Mapping[str, Any]


def _scalar(state: Mapping[str, Any], key: str, index: int) -> float:
    try:
        value = float(state[key][0, index])
    except (KeyError, IndexError, TypeError) as exc:
        raise ScorerStateInconsistency(f"missing {key}[{index}]") from exc
    if not math.isfinite(value):
        raise ScorerStateInconsistency(f"nonfinite {key}[{index}]")
    return value


def _flag(state: Mapping[str, Any], key: str, index: int) -> bool:
    return bool(_scalar(state, key, index))


def _compute_host(state: Mapping[str, Any], task_index: int) -> int | None:
    """Exec is preferred; Host is the source-backed fallback after compute ends."""
    for relation_name in ("exec", "host"):
        relation_type = TASK_AGENT_RELATION_TYPE_VOCAB.index(relation_name)
        matches = [r for r in range(state["task_agent_validity"].shape[1])
                   if _flag(state, "task_agent_validity", r)
                   and int(_scalar(state, "task_agent_task_index", r)) == task_index
                   and int(_scalar(state, "task_agent_relation_type_index", r)) == relation_type]
        if len(matches) > 1:
            raise ScorerStateInconsistency(f"ambiguous {relation_name} relation for task {task_index}")
        if matches:
            return int(_scalar(state, "task_agent_agent_index", matches[0]))
    return None


def _return_slot(state: Mapping[str, Any], task_index: int) -> bool:
    slot = int(_scalar(state, "return_flow_index", task_index))
    if slot < 0:
        return False
    if slot >= state["flow_known"].shape[1] or not _flag(state, "flow_known", slot):
        raise ScorerStateInconsistency("Return index outside frozen known support")
    if (int(_scalar(state, "flow_task_index", slot)) != task_index or
            int(_scalar(state, "flow_type_index", slot)) != FLOW_TYPE_VOCAB.index("Return")):
        raise ScorerStateInconsistency("Return index does not identify this task's Return Flow")
    return True


def _anchor_flows(state: Mapping[str, Any], task_index: int) -> tuple[int, ...]:
    return tuple(f for f in range(state["flow_known"].shape[1])
                 if _flag(state, "flow_known", f) and _flag(state, "flow_presence", f)
                 and int(_scalar(state, "flow_task_index", f)) == task_index)


def _single_hop_flow_burden(state: Mapping[str, Any], flow_index: int) -> float:
    remaining = _scalar(state, "flow_remaining", flow_index)
    hop = _scalar(state, "hop_remaining", flow_index)
    present = _flag(state, "flow_presence", flow_index)
    if remaining < 0 or hop < 0:
        raise ScorerStateInconsistency("negative Flow remaining")
    if not present:
        if remaining > 0:
            raise ScorerStateInconsistency("absent Flow has positive remaining")
        if int(_scalar(state, "flow_status_index", flow_index)) != FLOW_STATUS_VOCAB.index("COMPLETED"):
            raise ScorerStateInconsistency("absent anchor Flow is not completed")
        return 0.0
    if not _flag(state, "carrying_active", flow_index):
        raise ScorerStateInconsistency("present Flow has no active carrying hop")
    nodes = state["route_node_mask"][0, flow_index]
    if (int(nodes.sum()) != 1 or int(_scalar(state, "current_hop_index", flow_index)) != 0 or
            int(state["route_node_indices"][0, flow_index, 0]) !=
            int(_scalar(state, "flow_destination_index", flow_index))):
        raise ScorerStateInconsistency("Flow outside Planner v1 single-hop support")
    if hop > remaining:
        raise ScorerStateInconsistency("single-hop remaining exceeds E2E remaining")
    return hop


def _tx_burden(state: Mapping[str, Any], flows: tuple[int, ...]) -> float:
    return sum(_single_hop_flow_burden(state, f) for f in flows)


def _effort(candidate: CandidateActionSequence, side: PlannerObjectiveCausalSideState,
            horizon: int, anchor_state: Mapping[str, Any]) -> dict[str, float]:
    step = candidate.steps[horizon - 1]
    names = ("Comm", "Comp", "Mob")
    numerators = (sum(len(row["rb_indices"]) for row in step.comm),
                  sum(float(row["allocated_cpu_per_s"]) for row in step.comp),
                  sum(float(row["speed_mps"]) for row in step.mob))
    result = {}
    for name, numerator, applicable, denominator in zip(
            names, numerators, side.effort_component_mask, side.effort_denominators):
        if numerator < 0 or not math.isfinite(numerator):
            raise ValueError(f"invalid {name} effort request")
        if not applicable:
            if numerator:
                raise ScorerStateInconsistency(f"{name} requested outside anchor applicability")
            continue
        if denominator is None or denominator <= 0:
            raise ScorerStateInconsistency(f"{name} denominator not anchor-frozen")
        result[name] = float(numerator / denominator)
    # RB identity is global; per-task assignments may reuse the same RB and
    # therefore normalized Comm effort can exceed one. The allocation count
    # is nevertheless in the same RB-assignment units as AirFogSim's input.
    support = (anchor_state["comm_presence"][0] & anchor_state["comm_validity"][0]
               & anchor_state["comm_wireless_mask"][0])
    for row in step.comm:
        relation = int(row["relation_index"])
        if relation >= support.shape[0] or not bool(support[relation]):
            raise ScorerStateInconsistency("Comm request is not on current wireless support")
        rb_count = int(anchor_state["rb_active_mask"].shape[-1])
        if any(int(rb) < 0 or int(rb) >= rb_count for rb in row["rb_indices"]):
            raise ScorerStateInconsistency("Comm RB outside current allocatable support")
    return result


def _first_unsupported(side: PlannerObjectiveCausalSideState,
                       states: Sequence[Mapping[str, Any]]) -> tuple[int | None, str | None]:
    for h, state in enumerate(states, 1):
        for task in side.tasks:
            if task.terminal_at_anchor:
                continue
            ti = task.task_index
            finished = _scalar(state, "task_work_remaining", ti) <= 0
            existing = _return_slot(state, ti)
            host = _compute_host(state, ti) if finished and not existing else None
            if planner_derived_return_birth_required(task, finished, existing, host):
                return h, f"UNSUPPORTED_FUTURE_RETURN_BIRTH:{task.task_id}:H{h}"
    return None, None


def objective_sort_key(objective: tuple[int, float, float, float, float],
                       candidate_fingerprint: str) -> tuple[Any, ...]:
    if len(objective) != 5 or any(not math.isfinite(float(value)) for value in objective):
        raise ValueError("finite frozen five-part objective required")
    return (*objective, candidate_fingerprint)


def compare_objective_scores(left: CandidateObjectiveScore,
                             right: CandidateObjectiveScore) -> int:
    a = objective_sort_key(left.objective_tuple, left.candidate_fingerprint)
    b = objective_sort_key(right.objective_tuple, right.candidate_fingerprint)
    return (a > b) - (a < b)


def score_candidate_set(anchor_state: Mapping[str, Any],
                        side: PlannerObjectiveCausalSideState,
                        candidates_and_traces: Sequence[tuple[CandidateActionSequence, CandidateRolloutTrace]],
                        *, slot_duration_s: float) -> CandidateSetObjectiveResult:
    """Score one supplied set at its common support horizon, without rollout."""
    pairs = tuple(candidates_and_traces)
    if not pairs or not (0 < slot_duration_s < math.inf):
        raise ValueError("candidate set and positive finite slot duration required")
    horizon = pairs[0][0].horizon
    if not 1 <= horizon <= 4 or any(c.horizon != horizon or len(t.states) != horizon for c, t in pairs):
        raise ValueError("all candidates need supplied H1-H4 traces of common length")
    anchor_fingerprint = str(pairs[0][1].initial_fingerprints["anchor"])
    if any(c.candidate_id != t.candidate_id or t.initial_fingerprints["anchor"] != anchor_fingerprint
           for c, t in pairs):
        raise ValueError("candidate/trace identity or same-start anchor mismatch")
    if len({c.fingerprint for c, _ in pairs}) != len(pairs):
        raise ValueError("duplicate candidate fingerprint has no unique tie-break")
    if any(c.generation_metadata.get("planner_action_domain") != "V1" for c, _ in pairs):
        raise ValueError("Planner v1 domain admission required")
    if not any(side.effort_component_mask):
        raise ScorerStateInconsistency("NO_ANCHOR_APPLICABLE_EFFORT_COMPONENT")
    for c, trace in pairs:
        for h, step in enumerate(c.steps):
            for route in step.route:
                if len(route.get("route_node_indices", ())) != 1:
                    raise ValueError("multi-hop Route outside Planner v1")
            if step.route:
                if h >= len(trace.mappings) or len(trace.mappings[h].get("route", ())) != len(step.route):
                    raise ScorerStateInconsistency("Route action mapping evidence required")
                if any(row.get("mode") == "pending_flow" or int(row.get("flow_index", -1)) < 0
                       for row in trace.mappings[h]["route"]):
                    raise ScorerStateInconsistency(
                        "UNSUPPORTED_PENDING_FLOW_ROUTE_OBJECTIVE: fixed-support model creates no Flow")
    cohort = tuple(task for task in side.tasks if not task.terminal_at_anchor and
                   _flag(anchor_state, "task_presence", task.task_index) and
                   not _flag(anchor_state, "task_completed", task.task_index))
    if not cohort:
        return CandidateSetObjectiveResult("OBJECTIVE_UNSCOREABLE_EMPTY_COHORT", 0, (), {},
                                           {"cohort_size": 0})
    support = {}
    reasons = {}
    for candidate, trace in pairs:
        first, reason = _first_unsupported(side, trace.states)
        support[candidate.candidate_id] = support_horizon(first, horizon)
        reasons[candidate.candidate_id] = reason
    H_eff = min(support.values())
    if H_eff == 0:
        return CandidateSetObjectiveResult("OBJECTIVE_UNSCOREABLE", 0, (), support,
                                           {"support_boundary_reasons": reasons})
    anchor_components = {}
    for task in cohort:
        flows = _anchor_flows(anchor_state, task.task_index)
        tx0 = _tx_burden(anchor_state, flows)
        work0 = _scalar(anchor_state, "task_work_remaining", task.task_index)
        if work0 < 0:
            raise ScorerStateInconsistency("negative anchor compute remaining")
        anchor_components[task.task_id] = (flows, tx0, work0, tx0 > 0, work0 > 0)
    scored = []
    for candidate, trace in pairs:
        statuses = {task.task_id: "ACTIVE" for task in cohort}
        rows = []
        failed_tasks = set()
        for h, state in enumerate(trace.states[:H_eff], 1):
            now = side.current_time_s + h * slot_duration_s
            task_rows = []
            for task in cohort:
                ti = task.task_index
                status = statuses[task.task_id]
                completed = _flag(state, "task_completed", ti)
                if status == "ACTIVE":
                    existing_return = _return_slot(state, ti)
                    host = _compute_host(state, ti) if task.required_returned_size > 0 and not existing_return else None
                    no_return = not existing_return and (task.required_returned_size == 0 or host == task.return_destination_index)
                    if completed:
                        timely = now <= task.absolute_deadline_s + (1e-5 if no_return else 0.0)
                        status = "SUCCESSFULLY_COMPLETED" if timely else "DEADLINE_FAILED"
                    elif now > task.absolute_deadline_s:
                        status = "DEADLINE_FAILED"
                    statuses[task.task_id] = status
                failed = status == "DEADLINE_FAILED"
                unfinished = status != "SUCCESSFULLY_COMPLETED"
                if failed:
                    failed_tasks.add(task.task_id)
                flows, tx0, work0, m_tx, m_comp = anchor_components[task.task_id]
                tx = _tx_burden(state, flows)
                work = _scalar(state, "task_work_remaining", ti)
                if work < 0:
                    raise ScorerStateInconsistency("negative predicted compute remaining")
                if status == "SUCCESSFULLY_COMPLETED":
                    burden = 0.0
                elif not (m_tx or m_comp):
                    raise BurdenSemanticsBlocked(
                        f"BURDEN_SEMANTICS_BLOCKED:WAITING_RETURN_WITH_NO_CURRENT_SERVICE_BURDEN:{task.task_id}")
                else:
                    burden = ((tx / tx0 if m_tx else 0.0) +
                              (work / work0 if m_comp else 0.0)) / (int(m_tx) + int(m_comp))
                task_rows.append({"task_id": task.task_id, "task_index": ti, "status": status,
                                  "deadline_violation": int(failed), "unfinished": int(unfinished),
                                  "anchor_flow_slots": flows, "anchor_tx_burden": tx0,
                                  "tx_burden": tx, "anchor_work_remaining": work0,
                                  "work_remaining": work, "m_Tx": m_tx, "m_Comp": m_comp,
                                  "burden": burden})
            efforts = _effort(candidate, side, h, anchor_state)
            rows.append(CandidateObjectiveHorizonRow(
                h, sum(row["deadline_violation"] for row in task_rows),
                sum(row["unfinished"] for row in task_rows),
                sum(row["burden"] for row in task_rows) / len(cohort),
                sum(efforts.values()) / len(efforts),
                tuple(task_rows), efforts))
        denominator = len(cohort) * H_eff
        n_ddl = len(failed_tasks)
        a_ddl = sum(row.deadline_violation for row in rows) / denominator
        delay = sum(row.unfinished for row in rows) / denominator
        burden = sum(row.burden for row in rows) / H_eff
        effort = sum(row.effort for row in rows) / H_eff
        objective = (n_ddl, a_ddl, delay, burden, effort)
        scored.append(CandidateObjectiveScore(candidate.candidate_id, candidate.fingerprint,
            anchor_fingerprint, support[candidate.candidate_id], H_eff,
            *objective, objective, tuple(rows),
            {task.task_id: tuple(next(item for item in row.tasks if item["task_id"] == task.task_id)
                                 for row in rows) for task in cohort},
            reasons[candidate.candidate_id],
            {"cohort_task_ids": tuple(task.task_id for task in cohort),
             "effort_component_mask": side.effort_component_mask,
             "effort_denominators": side.effort_denominators,
             "future_truth_used": False, "throughput_diagnostic_only": True}))
    scored.sort(key=lambda row: objective_sort_key(row.objective_tuple, row.candidate_fingerprint))
    return CandidateSetObjectiveResult("SCOREABLE", H_eff, tuple(scored), support,
                                       {"cohort_size": len(cohort), "support_boundary_reasons": reasons})
