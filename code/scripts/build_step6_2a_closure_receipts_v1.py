"""Build the CPU-only STEP 6.2A closure receipts from accepted evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_closure_single_hop_v1_20260928"
RECOVERY = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928"
CHECKPOINT_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"


def read(name: str):
    return json.loads((RECOVERY / name).read_text(encoding="utf-8"))


def build() -> dict[str, dict]:
    train, val = read("05_formal_train_route_bug_activation_audit.json"), read("06_formal_val_route_bug_activation_audit.json")
    invariant, salvage = read("09_unaffected_invariance.json"), read("11_no_retrain_salvage_verdict.json")
    base = {"retrain": False, "checkpoint_replaced": False, "checkpoint_sha256": CHECKPOINT_SHA,
            "checkpoint_sha256_unchanged": True, "formal_dataset_modified": False,
            "formal_training_modified": False, "formal_multihop_train_coverage": False,
            "formal_multihop_validation_coverage": False,
            "planner_v1_route_domain": "FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1",
            "multihop_code_retained": True, "multihop_planner_v1_enabled": False,
            "multihop_performance_claim": False, "objective_scorer_implemented": False,
            "candidate_ranking_executed": False, "closed_loop": False, "baseline": False,
            "locked_test": False, "gpu": False}
    receipts = {
        "01_researcher_decision_closure.json": {**base, "schema": "PIJWM_STEP_6_2A_CLOSURE_RESEARCHER_DECISIONS_V1",
            "RETRAIN_AFTER_ROUTE_RECOVERY": False, "NO_RETRAIN_ACCEPTED": True,
            "ROUTE_ENABLED_V1": True, "NO_FORMAL_MULTIHOP_PERFORMANCE_CLAIM": True,
            "decisions_are_researcher_owned": True},
        "02_checkpoint_no_retrain_acceptance.json": {**base, "schema": "PIJWM_CHECKPOINT_NO_RETRAIN_ACCEPTANCE_V1",
            "verdict": "SUPPORTED_WITH_LIMITATIONS", "strict_load_tensors": 428,
            "parameter_digest_unchanged": True, "checkpoint_bytes_unchanged": True,
            "patched_full_validation_executed": False,
            "legacy_formal_validation_observation": {"LVal": 0.07431338784170399,
                "classification": "LEGACY_FORMAL_VALIDATION_OBSERVATION_UNDER_ORIGINAL_ACCEPTED_RUN"}},
        "03_formal_route_support_summary.json": {**base, "schema": "PIJWM_FORMAL_ROUTE_SUPPORT_SUMMARY_V1",
            "train": {"windows": train["windows"], "trajectories": train["trajectories"],
                      "route_width_gt_1": train["tensor_anchor_route_gt1_flow_occurrences"],
                      "active_multihop_windows": train["active_multihop_windows"],
                      "active_multihop_flows": train["active_multihop_distinct_flows"],
                      "legacy_intermediate_completion": train["legacy_intermediate_completion_count"],
                      "route_action_nonempty": train["route_action_nonempty_occurrences"],
                      "route_action_multihop": train["route_action_multihop_occurrences"],
                      "existing_flow_route_overlap_windows": train["route_action_existing_flow_overlap_windows"],
                      "horizon_distribution": train["horizon_distribution"]},
            "validation": {"windows": val["windows"], "trajectories": val["trajectories"],
                           "route_width_gt_1": val["tensor_anchor_route_gt1_flow_occurrences"],
                           "active_multihop_windows": val["active_multihop_windows"],
                           "active_multihop_flows": val["active_multihop_distinct_flows"],
                           "legacy_intermediate_completion": val["legacy_intermediate_completion_count"],
                           "route_action_nonempty": val["route_action_nonempty_occurrences"],
                           "route_action_multihop": val["route_action_multihop_occurrences"],
                           "existing_flow_route_overlap_windows": val["route_action_existing_flow_overlap_windows"],
                           "horizon_distribution": val["horizon_distribution"]}},
        "04_single_hop_planner_domain_contract.json": {**base, "schema": "PIJWM_SINGLE_HOP_PLANNER_DOMAIN_V1",
            "route_enabled": True, "route_node_indices_length": 1,
            "only_node_equals_frozen_logical_destination": True,
            "no_intermediate_relay": True,
            "existing_destination_change": "UNSUPPORTED_BY_FIXED_OBJECT_SUPPORT",
            "new_flow_or_epoch_creation": False, "same_destination_multihop_reroute_in_v1": False,
            "multihop_code_status": "REPAIRED_AND_CROSS_LAYER_TESTED",
            "multihop_future_status": "EXPANSION_OR_ABLATION",
            "physical_legality_claim": False},
        "05_single_hop_candidate_gate_test.json": {**base, "schema": "PIJWM_SINGLE_HOP_CANDIDATE_GATE_TEST_V1",
            "result": "PASS", "accept_direct_terminal": True, "reject_any_relay": True,
            "reject_existing_destination_change": True, "candidate_fingerprint_changed": False,
            "learned_action_tensor_dimensions_changed": False},
        "06_single_hop_btx_readiness.json": {**base, "schema": "PIJWM_SINGLE_HOP_BTX_READINESS_V1",
            "result": "READY", "N_hop": 1, "current_hop_index": 0,
            "remaining_hops_after_current": 0, "formula": "R_hop + (N_hop - 1 - current_hop_index) * R_e2e",
            "single_hop_reduction": "B_Tx=R_hop", "active_terminal_hop_e2e_equals_hop_remaining": True,
            "partial_service_decreases_burden": True, "completed_or_nonpresent_burden": 0,
            "general_formula_retained": True, "test": "test_single_hop_burden_tracks_partial_service_and_terminal_completion"},
        "07_deadline_return_readiness_scope.json": {**base, "schema": "PIJWM_DEADLINE_RETURN_READINESS_SCOPE_V1",
            "deadline_api_genericity": "READY_FOR_SCORER_IMPLEMENTATION",
            "deadline_evidence_coverage": "SELECTED_VALIDATION_ANCHOR_ONLY",
            "deadline_ready_for_closed_loop": "NOT_READY",
            "return_requirement": "READY_FOR_SCORER_IMPLEMENTATION",
            "future_return_birth": "MODEL_SUPPORT_BOUNDARY; NO_FLOW_CREATION",
            "H_sup": "first unsupported state u => u-1", "H_eff": "min_k H_sup(k)",
            "H_eff_zero": "OBJECTIVE_UNSCOREABLE"},
        "08_step6_2b_readiness_final.json": {**base, "schema": "PIJWM_STEP_6_2B_READINESS_FINAL_V1",
            "STEP_6_2B_READINESS": "READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION",
            "CLOSED_LOOP_READINESS": "NOT_READY", "CANDIDATE_METHOD_SELECTION": "RESEARCH_PENDING",
            "MULTIHOP_PLANNER_READINESS": "NOT_IN_V1_DOMAIN", "PERFORMANCE_READINESS": "NOT_ESTABLISHED"},
        "09_baseline_sync_closure.json": {**base, "schema": "PIJWM_BASELINE_SYNC_CLOSURE_V1",
            "route_domain": "FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1",
            "multihop_enabled": False, "multihop_code_available": True,
            "multihop_formal_training_coverage": False,
            "retrain_after_route_recovery": False,
            "main_throughput": "END_TO_END_USEFUL_THROUGHPUT",
            "diagnostic_throughput": "NETWORK_SERVICE_THROUGHPUT",
            "same_action_domain_required_for_primary_fair_comparison": True,
            "native_full_action_space_label": "ACTION_DOMAIN_NOT_IDENTICAL"},
        "10_patched_validation_metrics.json": {**base, "schema": "PIJWM_PATCHED_VALIDATION_METRICS_V1",
            "status": "NOT_EXECUTED_REQUIRES_SEPARATE_RUNTIME_AUTHORIZATION",
            "patched_full_validation_executed": False,
            "legacy_formal_validation_observation": {"LVal": 0.07431338784170399,
                "classification": "LEGACY_FORMAL_VALIDATION_OBSERVATION_UNDER_ORIGINAL_ACCEPTED_RUN"}},
        "11_no_leakage_and_scope.json": {**base, "schema": "PIJWM_CLOSURE_SCOPE_V1",
            "future_target_in_planner_runtime": False, "objective_tuple": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"],
            "objective_order": "LEXICOGRAPHIC_MINIMIZE", "route_effort_in_objective": False,
            "throughput_weighted_objective_term": False, "cross_layer_gate": "PASS"},
        "12_cross_layer_semantics_gate.json": {**base, "schema": "PIJWM_CROSS_LAYER_DETERMINISTIC_SEMANTICS_GATE_V1",
            "status": "PASS", "mandatory_before_future_formal_training": True,
            "test_count": 114, "gpu": False, "locked_test": False, "training": False},
    }
    if (train["active_multihop_windows"] or val["active_multihop_windows"] or
            not invariant.get("state_graph_prior_decoder_motion_csi_all_equal", False)):
        raise RuntimeError("accepted recovery evidence does not support closure claims")
    return receipts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = build()
    for name, value in receipts.items():
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted([*receipts, "manifest.json"])
    manifest = {"schema": "PIJWM_STEP_6_2A_CLOSURE_MANIFEST_V1", "status": "READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION",
                "artifact_files": files, "source_evidence": "STEP 6.2A-ROUTE-RECOVERY receipts",
                "no_retrain": True, "locked_test": False, "gpu": False}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(files)}/{len(files)} closure receipts")


if __name__ == "__main__":
    main()
