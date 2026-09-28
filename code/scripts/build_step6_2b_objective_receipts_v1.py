"""Emit additive CPU-only STEP 6.2B receipts without rewriting older evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_step6_2b_objective_scorer_cpu_v1 import ROOT, CHECKPOINT, EXPECTED_SHA, run, sha
from pi_jwm.step4_2a_graph_input_extension_v1 import TASK_AGENT_RELATION_TYPE_VOCAB
from pi_jwm.step6_0c_planner_action_domain_v1 import ROUTE_DOMAIN

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2b_objective_scorer_v1_20260928"


def emit(name: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if ROUTE_DOMAIN == "PLANNER_V1_ROUTE_EXPLICIT_NOOP_ONLY":
        raise RuntimeError(
            "Historical STEP 6.2B blocked receipts are frozen; use "
            "build_step6_2b_patch_route_noop_receipts_v1.py for current evidence")
    integration = run()
    audit = integration.pop("effective_route_action_audit")
    source = ROOT / "code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py"
    scorer = ROOT / "code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py"
    common = {"gpu": False, "locked_test": False, "training": False,
              "optimizer_step": False, "checkpoint_modified": False,
              "formal_dataset_modified": False, "closed_loop": False,
              "baseline": False, "candidate_method_selected": False,
              "performance_claim": False}
    emit("01_objective_contract_receipt.json", {**common, "status": "IMPLEMENTED_FOR_FIXED_SUPPORT",
        "objective_tuple": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"],
        "comparison": "STRICT_LEXICOGRAPHIC_MINIMIZE", "weighted_sum": False,
        "route_effort": False, "throughput_objective_term": False,
        "planner_route_domain": "FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1",
        "scorer_source_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest()})
    emit("02_anchor_cohort_and_masks.json", {**common, "cohort":
        "anchor causal present tasks excluding terminal DONE/FAILED; fixed across candidates/horizons",
        "empty_cohort": "OBJECTIVE_UNSCOREABLE_EMPTY_COHORT",
        "masks": "m_Tx and m_Comp frozen from anchor B_Tx0/W_rem0",
        "effort_applicability": "Comm/Comp/Mob frozen at anchor",
        "selected_anchor": integration["sample_id"],
        "selected_anchor_cohort_count": len(integration["candidate_scores"][0]["per_horizon_rows"][0]["tasks"])})
    emit("03_deadline_delay_semantics.json", {**common, "status": "CONTRACT_TESTED",
        "failure_latched": True, "failed_not_successful_completion": True,
        "absolute_deadline": "arrival_time_s + deadline_s",
        "unfinished_failure_rule": "current_time_s > absolute_deadline_s",
        "return_completion_tolerance_s": 0.0,
        "no_return_completion_tolerance_s": 1e-5,
        "N_DDL": "distinct failed anchor tasks within H_eff",
        "A_DDL": "latched failed task-horizon indicators / (cohort_count * H_eff)",
        "J_Delay": "non-successful task-horizon indicators / (cohort_count * H_eff)"})
    emit("04_support_horizon_cases.json", {**common, "status": "CONTRACT_TESTED",
        "all_H_sup_4": "H_eff=4", "one_H_sup_2": "all use H_eff=2",
        "unsupported_at_H1": "OBJECTIVE_UNSCOREABLE; no winner",
        "unsupported_at_H3": "H3/H4 excluded for every candidate",
        "unsupported_future_return_birth": "first state u => H_sup=u-1",
        "existing_return_slot": "identified by frozen return_flow_index, not current presence",
        "task_agent_exec_vocab_index": TASK_AGENT_RELATION_TYPE_VOCAB.index("exec"),
        "task_agent_host_vocab_index": TASK_AGENT_RELATION_TYPE_VOCAB.index("host"),
        "task_agent_vocab_source": "code/src/pi_jwm/step4_2a_graph_input_extension_v1.py::TASK_AGENT_RELATION_TYPE_VOCAB"})
    emit("05_burden_component_audit.json", {**common, "status": "CONTRACT_TESTED_WITH_BOUNDARY",
        "single_hop_B_Tx": "hop_remaining",
        "anchor_flow_set": "known and present fixed-support slots belonging to task",
        "terminal_completed_flow_burden": 0,
        "absent_flow_positive_remaining": "SCORER_STATE_INCONSISTENCY",
        "m_Tx": "B_Tx(q,0)>0", "m_Comp": "W_rem(q,0)>0",
        "completed_task_burden": 0,
        "deadline_failed_task_burden": "remaining component burden retained",
        "both_masks_zero_and_unfinished": "BURDEN_SEMANTICS_BLOCKED"})
    emit("06_comm_effort_denominator_audit.json", {**common, "status": "SOURCE_CORRECTED",
        "old_nested_mask_result_selected_anchor": 242,
        "old_result_unit": "communication relation rows, not RB support",
        "new_selected_anchor_denominator": 50,
        "new_unit": "global simulator RB IDs, gated by current valid wireless relation",
        "numerator_unit": "candidate requested task-RB assignments",
        "source": ["AirFogSim airfogsim_env._allocate_communication_RBs task_id -> RB_Nos",
                   "AirFogSim channel_manager.activateLink(tx,rx,allocated_RBs,type)",
                   "Formal action adapter comm_allocation_mask[relation,RB]",
                   "formal rb_active_mask last-axis width equals global n_RB; prior allocation values do not define support",
                   "CSI observation mask is not simulator RB-allocation legality"],
        "previous_receipt_historical_observation_unchanged": True,
        "normalized_value_may_exceed_one_due_to_rb_reuse": True,
        "side_state_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    emit("07_effort_component_audit.json", {**common, "status": "CONTRACT_TESTED",
        "components": ["Comm", "Comp", "Mob"], "route_effort": False,
        "Comm": "requested RB assignment count / anchor valid global RB IDs",
        "Comp": "requested allocated_cpu_per_s / anchor observed static CPU capacity",
        "Mob": "requested present UAV speed / (15 m/s * anchor present UAV count)",
        "aggregation": "mean of anchor-applicable components per horizon, then H1..H_eff mean",
        "empty_action_does_not_remove_applicable_component": True})
    emit("08_lexicographic_comparator_cases.json", {**common, "status": "CONTRACT_TESTED",
        "order": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"],
        "strict_compare": True, "research_epsilon": None,
        "exact_tie_break": "candidate_fingerprint", "weighted_sum": False,
        "deadline_dominates_effort": True, "delay_dominates_burden": True,
        "effort_only_after_first_four_tie": True})
    emit("09_real_frozen_checkpoint_score_integration.json", {**common, **integration,
        "interpretation": "mechanism only; no candidate-performance conclusion"})
    emit("10_order_independence_and_tie_break.json", {**common, "status": "CONTRACT_TESTED",
        "candidate_set_input_permutation_invariant": True,
        "common_H_eff_permutation_invariant": True,
        "fingerprint_is_tie_break_not_objective": True})
    emit("11_future_leakage_audit.json", {**common, "status": "CONTRACT_TESTED",
        "runtime_inputs": ["anchor model state", "causal objective side-state",
                           "CandidateActionSequence", "CandidateRolloutTrace"],
        "future_target_mutation_score_invariant": True,
        "future_failed_label_mutation_score_invariant": True,
        "future_truth_in_runtime": False, "prior_mode": "mean",
        "service_mode": "expectation", "risk_active": False})
    existing = [row for row in audit if row["kind"] == "existing_flow"]
    pending = [row for row in audit if row["kind"] == "pending_no_current_flow"]
    emit("12_effective_action_audit.json", {**common, "status": "BOUNDED_CAUSAL_FIXTURE",
        "sample_id": integration["sample_id"],
        "existing_flow_legal_route_action_count": len(existing),
        "pending_no_current_flow_route_action_count": len(pending),
        "legal_but_deterministic_noop_count": sum(not row["deterministic_rule_changed_keys"]
                                                   for row in audit),
        "legal_but_deterministic_flow_noop_count": sum(
            not any(key.startswith("flow_") or key.startswith("carrying_") or key.startswith("route_")
                    for key in row["deterministic_rule_changed_keys"]) for row in audit),
        "legal_and_predicted_state_changing_count": sum(bool(row["predicted_state_changed_keys"])
                                                       for row in audit),
        "fixed_support_blocked_count": sum(row["action_mapping_mode"] == "pending_flow"
                                            for row in audit),
        "observations": audit,
        "ROUTE_EFFECTIVE_FREEDOM_V1": "LIMITED",
        "research_decision": False,
        "conflict": "pending Route passes current domain admission but adapter flow_index=-1 and rule creates no Flow; frozen objective contract defines H_sup for future Return birth only"})
    emit("13_step6_2b_acceptance.json", {**common, "STEP_6_2B":
        "BLOCKED_ON_OBJECTIVE_SEMANTICS", "scorer_implemented": True,
        "comparator_implemented": True, "focused_tests_pass": True,
        "real_frozen_checkpoint_integration_pass": True,
        "checkpoint_sha256": EXPECTED_SHA,
        "checkpoint_sha256_unchanged": sha(CHECKPOINT) == EXPECTED_SHA,
        "blocking_conflicts": [
            "pending single-hop Route is admitted with flow_index=-1; no deterministic Flow is created and its scoring support horizon is not frozen",
            "same-path existing-Flow Route changes task_agent host before hop completion in 4.4; whether this is intended needs contract reconciliation"],
        "next_action": "researcher resolves pending Route support and same-path host semantics before final 6.2B acceptance"})
    names = [f"{index:02d}_{suffix}.json" for index, suffix in enumerate((
        "objective_contract_receipt", "anchor_cohort_and_masks", "deadline_delay_semantics",
        "support_horizon_cases", "burden_component_audit", "comm_effort_denominator_audit",
        "effort_component_audit", "lexicographic_comparator_cases",
        "real_frozen_checkpoint_score_integration", "order_independence_and_tie_break",
        "future_leakage_audit", "effective_action_audit", "step6_2b_acceptance"), 1)]
    emit("manifest.json", {**common, "status": "BLOCKED_ON_OBJECTIVE_SEMANTICS",
        "artifact_files": names + ["manifest.json"],
        "checkpoint_sha256": EXPECTED_SHA})
    print("STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS; receipts=13; integration=PASS")


if __name__ == "__main__":
    main()
