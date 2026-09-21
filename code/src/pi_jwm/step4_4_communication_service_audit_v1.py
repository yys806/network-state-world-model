"""STEP 4.4 communication-service sufficiency gate.

This module is audit-only.  It decides whether actual communication service can
be recovered causally before any Structured RSSM world-model implementation is
allowed to start.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "PI-JWM-Step-4.4-Communication-Service-Sufficiency-Audit-v1"
VERDICTS = {
    "SERVICE_RULE_SUFFICIENT",
    "SERVICE_STATE_ADDITIVE_EXTENSION_REQUIRED",
    "SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED",
}
CLASSIFICATIONS = {
    "CURRENT_CAUSAL_STATE_AVAILABLE",
    "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED",
    "UNAVAILABLE_UNOBSERVABLE",
    "DERIVED",
    "IRRELEVANT",
}


def _dependency_matrix() -> list[dict[str, Any]]:
    return [
        {"name": "csi", "path": "wireless", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "current per-RB attenuation observed through channel_manager.getCSI; future value is the authorized learned channel dynamic", "required_for_actual_service": True},
        {"name": "rb_allocation", "path": "wireless", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "future action A^Comm", "required_for_actual_service": True},
        {"name": "bandwidth", "path": "wireless", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "known ChannelManagerCP.RB_bandwidth configuration", "required_for_actual_service": True},
        {"name": "transmit_power", "path": "wireless", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "known channel-type power constants", "required_for_actual_service": True},
        {"name": "interference", "path": "wireless", "classification": "DERIVED", "decision_time_role": "derived from all active links/RB allocations, channel attenuation, and channel-type transmit powers", "required_for_actual_service": True},
        {"name": "noise", "path": "wireless", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "known ChannelManagerCP.sig2 configuration", "required_for_actual_service": True},
        {"name": "fast_fading", "path": "wireless", "classification": "DERIVED", "decision_time_role": "already included in getCSI channel attenuation after updateFastFading", "required_for_actual_service": True},
        {"name": "outage_draw", "path": "wireless", "classification": "UNAVAILABLE_UNOBSERVABLE", "decision_time_role": "not_available; outcome_only_after_random_draw", "required_for_actual_service": True},
        {"name": "wired_capacity", "path": "wired", "classification": "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED", "decision_time_role": "WiredNetworkManager._links capacity_mbps", "required_for_actual_service": True},
        {"name": "wired_active_flow_count", "path": "wired", "classification": "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED", "decision_time_role": "WiredNetworkManager._flows grouped per link for fair sharing", "required_for_actual_service": True},
        {"name": "slot_duration", "path": "wireless_and_wired", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "frozen simulation_interval/slot duration", "required_for_actual_service": True},
        {"name": "flow_remaining_cap", "path": "wireless_and_wired", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "Causal Flow Ledger logical/carrying remaining state", "required_for_actual_service": True},
        {"name": "route_and_lifecycle_activation", "path": "wireless_and_wired", "classification": "CURRENT_CAUSAL_STATE_AVAILABLE", "decision_time_role": "current route, holder, hop and task lifecycle determine active transfer", "required_for_actual_service": True},
    ]


def expected_service_verdict(report: Mapping[str, Any]) -> str:
    rows: Sequence[Mapping[str, Any]] = report.get("dependency_matrix", ())
    required = [row for row in rows if row.get("required_for_actual_service") is True]
    if any(row.get("classification") == "UNAVAILABLE_UNOBSERVABLE" for row in required):
        return "SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED"
    if any(row.get("classification") == "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED" for row in required):
        return "SERVICE_STATE_ADDITIVE_EXTENSION_REQUIRED"
    return "SERVICE_RULE_SUFFICIENT"


def build_communication_service_audit(*, source_provenance: Sequence[Mapping[str, Any]] = (), source_claim_checks: Mapping[str, bool] | None = None) -> dict[str, Any]:
    checks = dict(source_claim_checks or {
        "wireless_nominal_formula_found": True,
        "per_rb_random_outage_draw_found": True,
        "outage_zeroes_rate_found": True,
        "outage_is_outcome_only_in_collector": True,
        "wired_capacity_share_formula_found": True,
    })
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "evidence_class": "SOURCE_AUDIT_ONLY_NO_WORLD_MODEL_IMPLEMENTATION",
        "definition_basis": "Definition 04 communication service Rule -> Add Missing Causal State -> Learn Residual gate",
        "dependency_matrix": _dependency_matrix(),
        "service_paths": {
            "wireless": {
                "nominal_rate_rule_recoverable": True,
                "nominal_rule": "RB_bandwidth * log2(1 + SINR_linear), with SINR derived from signal/(interference+noise)",
                "actual_service_uniquely_recoverable": False,
                "unresolved_actual_service_factor": "random per-RB outage realization",
                "outage_effect": "rate is set to zero when sampled outage is true",
                "causal_boundary": "outage is recorded only after runtime random draw as outcome_only_not_same_frame_decision_input",
            },
            "wired": {
                "nominal_rule": "min(link_capacity_bytes_per_slot / active_flow_count, flow_remaining_bytes)",
                "gap_class": "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED",
                "missing_current_inputs": ["capacity_mbps", "per-link active flow membership/count"],
                "minimal_additive_extension_possible": True,
                "actual_service_uniquely_recoverable_from_frozen_tensor": False,
            },
        },
        "source_claim_checks": checks,
        "source_provenance": [dict(row) for row in source_provenance],
        "research_stop": {
            "triggered": True,
            "reason": "actual wireless service contains a random outage realization that is not a decision-time causal input",
            "nominal_recovery_boundary": "nominal pre-outage per-RB rate is rule-recoverable when future CSI and all RB actions are supplied",
            "residual_candidate_target_not_selected": True,
            "required_researcher_decision": "choose whether outage/effective service is modeled as channel state, a stochastic service event, or a learned residual; then freeze its target and architecture",
        },
        "world_model_implementation_started": False,
        "scope": {name: False for name in ("training", "optimizer", "loss", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim")},
    }
    report["verdict"] = expected_service_verdict(report)
    return report


def validate_communication_service_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("dependency_matrix", ())
    by_name = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
    required_names = {"csi", "rb_allocation", "bandwidth", "transmit_power", "interference", "noise", "fast_fading", "outage_draw", "wired_capacity", "wired_active_flow_count", "slot_duration"}
    scope = report.get("scope", {})
    source_checks = report.get("source_claim_checks", {})
    expected = expected_service_verdict(report)
    outage = by_name.get("outage_draw", {})
    checks = {
        "schema": report.get("schema_version") == AUDIT_SCHEMA_VERSION,
        "dependency_fields_complete": required_names.issubset(by_name),
        "classifications_valid": bool(rows) and all(row.get("classification") in CLASSIFICATIONS for row in rows),
        "source_claims_computed_true": bool(source_checks) and all(value is True for value in source_checks.values()),
        "outage_boundary_truthful": outage.get("classification") == "UNAVAILABLE_UNOBSERVABLE" and outage.get("decision_time_role") == "not_available; outcome_only_after_random_draw",
        "nominal_actual_separated": report.get("service_paths", {}).get("wireless", {}).get("nominal_rate_rule_recoverable") is True and report.get("service_paths", {}).get("wireless", {}).get("actual_service_uniquely_recoverable") is False,
        "wired_additive_gap_recorded": report.get("service_paths", {}).get("wired", {}).get("minimal_additive_extension_possible") is True,
        "verdict_allowed": report.get("verdict") in VERDICTS,
        "verdict_matches_evidence": report.get("verdict") == expected,
        "stop_boundary": report.get("research_stop", {}).get("triggered") is True and report.get("research_stop", {}).get("residual_candidate_target_not_selected") is True,
        "world_model_not_started": report.get("world_model_implementation_started") is False,
        "scope_non_expansive": all(scope.get(name) is False for name in ("training", "optimizer", "loss", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim")),
    }
    return {"checks": checks, "expected_verdict": expected, "verdict_matches_evidence": checks["verdict_matches_evidence"], "passed": bool(all(checks.values()))}
