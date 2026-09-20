"""Evidence-bounded audit for the remaining stateful Flow contract sources.

This module does not build graph inputs.  It records what the current
AirFogSim source can prove at decision time and keeps task progress, hop
progress, and end-to-end Flow state separate.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "PI-JWM-Step4.2B-Stateful-Flow-Source-Audit-v1"
VERDICTS = {
    "FLOW_CONTRACT_CONSTRUCTIBLE",
    "FLOW_CONTRACT_PARTIALLY_CONSTRUCTIBLE",
    "FLOW_CONTRACT_NOT_YET_SUPPORTED",
}


def expected_flow_verdict(report: Mapping[str, Any]) -> str:
    """Compute the Flow-only verdict; unrelated resource gaps are excluded."""

    required = report["flow_specific_required_evidence"]
    values = [bool(required[name]) for name in required]
    if all(values):
        return "FLOW_CONTRACT_CONSTRUCTIBLE"
    critical_missing = {
        "input_current_remaining_causal",
        "return_current_remaining_causal",
    }
    if any(not bool(required[name]) for name in critical_missing):
        return "FLOW_CONTRACT_NOT_YET_SUPPORTED"
    return "FLOW_CONTRACT_PARTIALLY_CONSTRUCTIBLE"


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def build_stateful_flow_source_audit(*, source_provenance: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return the deterministic machine receipt for the current source audit."""

    evidence = {
        "minimum_definition_03": {
            "required": [
                "stable_flow_identity",
                "flow_type",
                "source_agent",
                "destination_agent",
                "presence",
                "total_data",
                "current_remaining_data",
                "associated_task",
            ],
            "flow_is_stateful_directed_multiedge": True,
            "task_dag_is_not_dependency_flow": True,
        },
        "input": {
            "candidate_a_logical_end_to_end": {
                "total_data": "task.task_size is observable",
                "current_remaining_data": "not causally reconstructable across completed hops",
                "stable_identity": {
                    "implementation_fact": "simulator-issued Flow ID = unavailable",
                    "research_boundary": "RESEARCHER_DECISION_REQUIRED / DERIVABLE_IF_LOGICAL_FLOW_SEMANTICS_SELECTED",
                    "candidate_not_selected": "task_id + input",
                },
                "endpoints": "current task node and current node observable; end-to-end endpoint history not frozen",
                "verdict": "PARTIALLY_CONSTRUCTIBLE",
            },
            "candidate_b_hop_local": {
                "total_data": "task.task_size is observable for offload stage",
                "current_remaining_data": "in-stage transmitted progress is observable before hop completion",
                "stable_identity": {
                    "implementation_fact": "no runtime stable hop ID",
                    "research_boundary": "hop-local naming is not selected as Definition 03 logical Flow identity",
                },
                "endpoints": "current route/current node and transfer event endpoints are observable",
                "verdict": "PARTIALLY_CONSTRUCTIBLE",
            },
            "selection": "neither candidate satisfies all Definition 03 minimum fields",
        },
        "return": {
            "total_data": "task.required_returned_size/getReturnedSize is observable",
            "current_remaining_data": "returning reuses stage-local transmitted_size and resets after each hop",
            "stable_identity": {
                "implementation_fact": "simulator-issued Flow ID = unavailable",
                "research_boundary": "RESEARCHER_DECISION_REQUIRED / DERIVABLE_IF_LOGICAL_FLOW_SEMANTICS_SELECTED",
                "candidate_not_selected": "task_id + return",
            },
            "endpoints": "return destination and current node are observable when configured",
            "verdict": "PARTIALLY_CONSTRUCTIBLE",
        },
        "dependency_data": {
            "dag_gate": "TaskManager _task_dependencies gates parent completion",
            "transfer_source": "no dependency payload, Flow ID, or dependency transfer event in audited source",
            "fact": "current AirFogSim audited mechanism has no real DepData transfer process",
            "research_boundary": "retain empty DepData type under current simulator semantics, or extend simulator with dependency-result transfer",
            "verdict": "RESEARCHER_DECISION_REQUIRED",
        },
        "task_progress_vs_flow_progress": {
            "task_size": "task-level offload total; not sufficient for end-to-end remaining",
            "return_size": "task-level return requirement; not already-returned amount",
            "in_stage_transmitted_size": "current transmission stage/hop accumulator; resets on hop completion",
            "computed_size": "CPU work progress; not communication Flow progress",
            "past_outcome_flow_service": "outcome-only evidence; forbidden as current Flow state",
        },
        "route_revision": {
            "route_change": "changeOffloadTo replaces unfinished route suffix",
            "identity_source": "no simulator-issued Flow identity or revision; builder revision is an action-side convention",
            "research_boundary": "logical business Flow identity may be derived only after researcher selects that semantic",
            "causal_status": "route revision cannot be promoted to stable Flow identity without additive source",
        },
        "remaining_raw_sources": {
            "dynamic_available_cpu": "RAW_INSUFFICIENT",
            "storage": "RAW_INSUFFICIENT",
            "wired_queue_load_utilization": "RAW_INSUFFICIENT",
            "task_return_size_deadline_priority": "SIMULATOR_OBSERVER_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED",
            "dag_dependency_payload": "RAW_INSUFFICIENT",
        },
    }
    flow_specific_required_evidence = {
        "identity_semantics_selected": False,
        "flow_type_semantics_available": True,
        "associated_task_causal": True,
        "logical_source_causal": False,
        "logical_destination_causal": False,
        "presence_causal": True,
        "input_total_source": True,
        "input_current_remaining_causal": False,
        "return_total_source": True,
        "return_current_remaining_causal": False,
        "route_multi_hop_semantics": False,
        "route_revision_identity_causal": False,
    }
    checks = [
        _check("definition_03_minimum_fields_audited", True, "All required Flow fields are listed separately."),
        _check("input_end_to_end_remaining_proven", flow_specific_required_evidence["input_current_remaining_causal"], "transmitted_size resets at hop completion; no causal end-to-end remaining source."),
        _check("return_end_to_end_remaining_proven", flow_specific_required_evidence["return_current_remaining_causal"], "return stage reuses the same hop-local accumulator."),
        _check("dependency_data_transfer_proven", False, "DAG provides gating only; no dependency-data transfer source; this is not an Input/Return blocker."),
        _check("past_outcome_not_used_as_current_state", True, "Outcome service is explicitly excluded from current Flow state."),
    ]
    other_graph_input_gaps = {
        "dynamic_available_cpu": False,
        "storage": False,
        "wired_queue_load_utilization": False,
    }
    report_without_verdict = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "flow_specific_required_evidence": flow_specific_required_evidence,
        "other_graph_input_gaps": other_graph_input_gaps,
        "evidence": evidence,
    }
    verdict = expected_flow_verdict(report_without_verdict)
    failed = [row["name"] for row in checks if not row["passed"]]
    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "verdict": verdict,
        "expected_verdict": verdict,
        "flow_specific_required_evidence": flow_specific_required_evidence,
        "other_graph_input_gaps": other_graph_input_gaps,
        "other_graph_input_gaps_complete": all(other_graph_input_gaps.values()),
        "graph_builder_started": False,
        "formal_dataset": False,
        "training": False,
        "gpu": False,
        "locked_test": False,
        "evidence": evidence,
        "checks": checks,
        "failed_checks": failed,
        "source_provenance": deepcopy(source_provenance or []),
    }
    if verdict not in VERDICTS:
        raise AssertionError("invalid audit verdict")
    return result


def validate_stateful_flow_source_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = list(report.get("checks", []))
    required = {
        "definition_03_minimum_fields_audited",
        "input_end_to_end_remaining_proven",
        "return_end_to_end_remaining_proven",
        "dependency_data_transfer_proven",
        "past_outcome_not_used_as_current_state",
    }
    names = {str(row.get("name")) for row in checks}
    checks_ok = required <= names and all(isinstance(row.get("passed"), bool) for row in checks if row.get("name") in required)
    scope_ok = all(report.get(name) is False for name in ("graph_builder_started", "formal_dataset", "training", "gpu", "locked_test"))
    verdict_ok = report.get("verdict") in VERDICTS
    expected = expected_flow_verdict(report)
    verdict_matches_evidence = report.get("verdict") == expected
    other_gaps_present = isinstance(report.get("other_graph_input_gaps"), Mapping)
    return {
        "schema_version": report.get("schema_version") == AUDIT_SCHEMA_VERSION,
        "required_checks_present": checks_ok,
        "scope_non_expansive": scope_ok,
        "verdict_valid": verdict_ok,
        "expected_verdict": expected,
        "verdict_matches_flow_evidence": verdict_matches_evidence,
        "other_graph_gaps_separate": other_gaps_present,
        "passed": bool(checks_ok and scope_ok and verdict_ok and verdict_matches_evidence and other_gaps_present),
    }
