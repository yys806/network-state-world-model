import copy
import unittest

from pi_jwm.step4_2c_a_causal_flow_ledger_feasibility_v1 import (
    AUDIT_SCHEMA_VERSION,
    build_causal_flow_ledger_feasibility_audit,
    compute_ledger_verdict,
    apply_real_transfer_event_to_audit_ledger,
    replay_audit_ledger_events,
    validate_causal_flow_ledger_feasibility_audit,
)


class Step42CAFeasibilityTests(unittest.TestCase):
    def test_machine_verdict_is_computed_and_tamper_is_rejected(self):
        report = build_causal_flow_ledger_feasibility_audit()
        self.assertEqual(report["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(report["verdict"], "CAUSAL_FLOW_LEDGER_FEASIBLE")
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE"
        receipt = validate_causal_flow_ledger_feasibility_audit(tampered)
        self.assertFalse(receipt["verdict_matches_evidence"])
        self.assertFalse(receipt["passed"])

    def test_future_action_counterfactual_does_not_change_current_ledger_state(self):
        report = build_causal_flow_ledger_feasibility_audit()
        changed = copy.deepcopy(report["counterfactual"]["current_ledger_state"])
        changed["future_action_target"] = "D"
        self.assertEqual(report["counterfactual"]["current_ledger_state"], report["counterfactual"]["replayed_current_ledger_state"])
        self.assertNotEqual(changed["future_action_target"], report["counterfactual"]["current_ledger_state"].get("future_action_target"))

    def test_no_double_count_invariant_uses_final_destination_delivery_only(self):
        report = build_causal_flow_ledger_feasibility_audit()
        inv = report["invariants"]["multi_hop_no_double_count"]
        self.assertTrue(inv["passed"])
        self.assertEqual(inv["hop_service_sum"], 20.0)
        self.assertEqual(inv["e2e_delivered"], 10.0)
        self.assertLessEqual(inv["e2e_delivered"], inv["total_data"])

    def test_input_return_and_reroute_evidence_are_separate(self):
        report = build_causal_flow_ledger_feasibility_audit()
        self.assertEqual(report["flow_evidence"]["input"]["status"], "FEASIBLE")
        self.assertEqual(report["flow_evidence"]["return"]["status"], "FEASIBLE")
        self.assertEqual(report["flow_evidence"]["depdata"]["status"], "DECISION_REQUIRED")
        self.assertEqual(report["reroute"]["status"], "SAME_DESTINATION_FEASIBLE")

    def test_source_status_uses_explicit_enum(self):
        report = build_causal_flow_ledger_feasibility_audit()
        allowed = {"EXISTING_STATE_SUFFICIENT", "EXISTING_EVENT_SUFFICIENT", "DERIVABLE_CAUSALLY", "NEW_OBSERVER_HOOK_REQUIRED", "SIMULATOR_SEMANTIC_EXTENSION_REQUIRED", "RESEARCHER_DECISION_REQUIRED"}
        statuses = [row["status"] for row in report["source_sufficiency_matrix"]]
        self.assertTrue(set(statuses) <= allowed)

    def test_input_replay_does_not_double_count_intermediate_hop(self):
        state = {"task_id": "T", "phase": "offload", "logical_destination": "C", "total": 10.0, "e2e_delivered": 0.0, "e2e_remaining": 10.0, "current_holder": "A"}
        result = replay_audit_ledger_events(state, [
            {"task_id": "T", "phase": "offload", "source_id": "A", "target_id": "B", "delivered_data": 10.0, "stage_or_hop_completed": True},
            {"task_id": "T", "phase": "offload", "source_id": "B", "target_id": "C", "delivered_data": 3.0, "stage_or_hop_completed": False},
        ])
        self.assertEqual(result["e2e_delivered"], 3.0)
        self.assertEqual(result["e2e_remaining"], 7.0)
        self.assertEqual(result["current_holder"], "B")

    def test_return_replay_and_holder_transition(self):
        state = {"task_id": "T", "phase": "return", "logical_destination": "A", "total": 10.0, "e2e_delivered": 0.0, "e2e_remaining": 10.0, "current_holder": "C"}
        result = replay_audit_ledger_events(state, [
            {"task_id": "T", "phase": "return", "source_id": "C", "target_id": "B", "delivered_data": 10.0, "stage_or_hop_completed": True},
            {"task_id": "T", "phase": "return", "source_id": "B", "target_id": "A", "delivered_data": 4.0, "stage_or_hop_completed": False},
        ])
        self.assertEqual(result["e2e_delivered"], 4.0)
        self.assertEqual(result["current_holder"], "B")

    def test_flow_completed_is_forbidden_as_logical_completion(self):
        state = {"task_id": "T", "phase": "offload", "logical_destination": "B", "total": 1.0, "e2e_delivered": 0.0, "e2e_remaining": 1.0, "current_holder": "A"}
        with self.assertRaises(ValueError):
            apply_real_transfer_event_to_audit_ledger(state, {"task_id": "T", "phase": "offload", "source_id": "A", "target_id": "B", "delivered_data": 1.0, "flow_completed": True})

    def test_destination_change_is_a_research_boundary(self):
        report = build_causal_flow_ledger_feasibility_audit()
        self.assertIn("RESEARCHER_DECISION_REQUIRED", report["reroute"]["destination_change"])

    def test_real_schema_endpoint_alias_and_stage_completion_overlay(self):
        state = {"task_id": "T", "phase": "offload", "logical_destination": "B", "total": 2.0, "e2e_delivered": 0.0, "e2e_remaining": 2.0, "current_holder": "A"}
        result = apply_real_transfer_event_to_audit_ledger(state, {
            "task_id": "T", "phase": "offload", "source": "A", "target": "B",
            "delivered_data": 2.0, "flow_completed": True, "stage_or_hop_completed": True,
        })
        self.assertEqual(result["e2e_delivered"], 2.0)
        self.assertEqual(result["current_holder"], "B")
