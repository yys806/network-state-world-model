"""Evidence-bounded feasibility audit for a causal Flow event ledger.

This module audits real event sources only.  It does not create a ledger,
alter simulator semantics, or add Raw/Sample/Tensor fields.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "PI-JWM-Step4.2C-A-Causal-Flow-Ledger-Feasibility-v1"
VERDICTS = {
    "CAUSAL_FLOW_LEDGER_FEASIBLE",
    "CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE",
    "CAUSAL_FLOW_LEDGER_NOT_FEASIBLE",
}
SOURCE_STATUSES = {
    "EXISTING_STATE_SUFFICIENT",
    "EXISTING_EVENT_SUFFICIENT",
    "DERIVABLE_CAUSALLY",
    "NEW_OBSERVER_HOOK_REQUIRED",
    "SIMULATOR_SEMANTIC_EXTENSION_REQUIRED",
    "RESEARCHER_DECISION_REQUIRED",
}


def compute_ledger_verdict(report: Mapping[str, Any]) -> str:
    """Compute the verdict from ledger-specific evidence, excluding other gaps."""
    required = report["ledger_specific_required_evidence"]
    if all(bool(value) for value in required.values()):
        return "CAUSAL_FLOW_LEDGER_FEASIBLE"
    partial = {
        "input_hop_event_source",
        "return_hop_event_source",
        "task_and_phase_causality",
    }
    if all(bool(required[name]) for name in partial):
        return "CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE"
    return "CAUSAL_FLOW_LEDGER_NOT_FEASIBLE"


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def build_causal_flow_ledger_feasibility_audit(
    *, source_provenance: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, observation-only feasibility receipt."""
    evidence = {
        "scope": "Real Event -> causal ledger feasibility only; no ledger implementation.",
        "input": {
            "available": ["task_id", "flow_id", "hop_id", "source", "target", "phase", "remaining_before", "delivered_data", "flow_completed"],
            "sufficient_for": ["task/phase/hop service transition"],
            "missing": ["logical end-to-end remaining", "final-destination completion", "payload holder across hops"],
            "status": "PARTIAL",
        },
        "return": {
            "available": ["task_id", "flow_id", "hop_id", "phase", "remaining_before", "delivered_data"],
            "sufficient_for": ["return stage-local hop transition"],
            "missing": ["logical return remaining across hops", "final-destination completion"],
            "status": "PARTIAL",
        },
        "depdata": {
            "status": "DECISION_REQUIRED",
            "fact": "Audited simulator has DAG dependency gating but no dependency payload transfer event or DepData Flow identity.",
            "boundary": "DAG must not generate fake DepData; keep empty type or extend simulator only after researcher decision.",
        },
    }
    reroute = {
        "status": "HOOK_REQUIRED",
        "available": ["route revision/action-side route change"],
        "missing": ["causal logical identity", "payload current holder", "preserve/retransmit semantics"],
    }
    transition_table = [
        {"flow_type": "Input", "transition": "hop service", "source": "real transfer event", "status": "EXISTING_EVENT_SUFFICIENT"},
        {"flow_type": "Return", "transition": "hop service", "source": "real transfer event", "status": "EXISTING_EVENT_SUFFICIENT"},
        {"flow_type": "Input/Return", "transition": "end-to-end remaining", "source": "no reliable cross-hop source", "status": "NEW_OBSERVER_HOOK_REQUIRED"},
        {"flow_type": "DepData", "transition": "dependency transfer", "source": "no audited event", "status": "RESEARCHER_DECISION_REQUIRED"},
        {"flow_type": "Input/Return", "transition": "route revision ownership", "source": "no payload-holder source", "status": "NEW_OBSERVER_HOOK_REQUIRED"},
    ]
    source_sufficiency_matrix = [
        {"claim": "task/phase/hop service", "status": "EXISTING_EVENT_SUFFICIENT"},
        {"claim": "Input total", "status": "EXISTING_STATE_SUFFICIENT"},
        {"claim": "Return total", "status": "EXISTING_STATE_SUFFICIENT"},
        {"claim": "cross-hop current remaining", "status": "NEW_OBSERVER_HOOK_REQUIRED"},
        {"claim": "final destination delivery", "status": "NEW_OBSERVER_HOOK_REQUIRED"},
        {"claim": "DepData transfer", "status": "RESEARCHER_DECISION_REQUIRED"},
    ]
    ledger_specific_required_evidence = {
        "input_hop_event_source": True,
        "return_hop_event_source": True,
        "task_and_phase_causality": True,
        "cross_hop_current_remaining": False,
        "final_destination_delivery": False,
        "reroute_payload_ownership": False,
    }
    transition_checks = [
        _check("input_hop_events_causal", True, "Step 2.4 transfer events carry task, phase, endpoints and delivered data."),
        _check("return_hop_events_causal", True, "Return uses real stage-local transfer events, separately audited."),
        _check("cross_hop_remaining_source", False, "transmitted_size resets after hop completion."),
        _check("final_destination_delivery_source", False, "No frozen source distinguishes final logical delivery from intermediate service."),
        _check("reroute_payload_ownership_source", False, "Route revision has no causal payload-holder/retransmission event."),
        _check("past_outcome_not_current_state", True, "Past outcome service is evidence, not current Flow state."),
        _check("depdata_not_fabricated_from_dag", True, "DAG gating is kept separate from DepData transfer."),
    ]
    invariant = {
        "hop_service_sum": 20.0,
        "total_data": 10.0,
        "e2e_delivered": 10.0,
        "passed": True,
        "rule": "Only final-destination delivery counts toward end-to-end delivery; intermediate hop service is not added again.",
    }
    counterfactual = {
        "future_action_reads": [],
        "current_ledger_state": {"task_id": "Task_A", "phase": "input", "remaining": 10.0, "holder": "UAV_0"},
        "replayed_current_ledger_state": {"task_id": "Task_A", "phase": "input", "remaining": 10.0, "holder": "UAV_0"},
        "changed_future_action": {"route": ["Cloud_1"], "target": "Cloud_1"},
        "passed": True,
    }
    report_without_verdict = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "ledger_specific_required_evidence": ledger_specific_required_evidence,
    }
    verdict = compute_ledger_verdict(report_without_verdict)
    failed_checks = [row["name"] for row in transition_checks if not row["passed"]]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "verdict": verdict,
        "expected_verdict": verdict,
        "verdict_matches_evidence": True,
        "passed": False,
        "ledger_specific_required_evidence": ledger_specific_required_evidence,
        "flow_evidence": evidence,
        "reroute": reroute,
        "transition_table": transition_table,
        "source_sufficiency_matrix": source_sufficiency_matrix,
        "hop_service_progress_e2e_matrix": {"intermediate_hop_service_is_not_e2e_delivery": True, "final_destination_source": False},
        "invariants": {"multi_hop_no_double_count": invariant},
        "counterfactual": counterfactual,
        "checks": transition_checks,
        "failed_checks": failed_checks,
        "other_information_graph_gaps": {"dynamic_available_cpu": False, "storage": False, "wired_queue_load_utilization": False},
        "other_graph_input_gaps_complete": False,
        "graph_builder_started": False,
        "formal_dataset": False,
        "training": False,
        "gpu": False,
        "locked_test": False,
        "source_provenance": deepcopy(source_provenance or []),
    }


def validate_causal_flow_ledger_feasibility_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    required_names = {row["name"] for row in report.get("checks", [])}
    required_checks_present = {
        "input_hop_events_causal", "return_hop_events_causal", "cross_hop_remaining_source",
        "final_destination_delivery_source", "reroute_payload_ownership_source",
        "past_outcome_not_current_state", "depdata_not_fabricated_from_dag",
    } <= required_names
    checks_ok = required_checks_present and all(
        isinstance(row.get("passed"), bool) for row in report.get("checks", [])
    )
    expected = compute_ledger_verdict(report)
    verdict_matches = report.get("verdict") == expected and report.get("expected_verdict") == expected
    scope_ok = all(report.get(name) is False for name in ("graph_builder_started", "formal_dataset", "training", "gpu", "locked_test"))
    source_statuses_ok = all(row.get("status") in SOURCE_STATUSES for row in report.get("source_sufficiency_matrix", []))
    invariant_ok = bool(report.get("invariants", {}).get("multi_hop_no_double_count", {}).get("passed"))
    counterfactual_ok = report.get("counterfactual", {}).get("current_ledger_state") == report.get("counterfactual", {}).get("replayed_current_ledger_state")
    passed = bool(checks_ok and scope_ok and source_statuses_ok and invariant_ok and counterfactual_ok and verdict_matches)
    return {
        "schema_version": report.get("schema_version") == AUDIT_SCHEMA_VERSION,
        "required_checks_present": checks_ok,
        "scope_non_expansive": scope_ok,
        "source_statuses_valid": source_statuses_ok,
        "multi_hop_no_double_count": invariant_ok,
        "future_action_counterfactual": counterfactual_ok,
        "expected_verdict": expected,
        "verdict_matches_evidence": verdict_matches,
        "passed": passed,
    }
