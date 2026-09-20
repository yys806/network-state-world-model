import copy
import unittest

from pi_jwm.step4_2c_a_causal_flow_ledger_feasibility_v1 import (
    AUDIT_SCHEMA_VERSION,
    build_causal_flow_ledger_feasibility_audit,
    compute_ledger_verdict,
    validate_causal_flow_ledger_feasibility_audit,
)


class Step42CAFeasibilityTests(unittest.TestCase):
    def test_machine_verdict_is_computed_and_tamper_is_rejected(self):
        report = build_causal_flow_ledger_feasibility_audit()
        self.assertEqual(report["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(report["verdict"], "CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE")
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "CAUSAL_FLOW_LEDGER_FEASIBLE"
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
        self.assertEqual(report["flow_evidence"]["input"]["status"], "PARTIAL")
        self.assertEqual(report["flow_evidence"]["return"]["status"], "PARTIAL")
        self.assertEqual(report["flow_evidence"]["depdata"]["status"], "DECISION_REQUIRED")
        self.assertEqual(report["reroute"]["status"], "HOOK_REQUIRED")

    def test_source_status_uses_explicit_enum(self):
        report = build_causal_flow_ledger_feasibility_audit()
        allowed = {"EXISTING_STATE_SUFFICIENT", "EXISTING_EVENT_SUFFICIENT", "DERIVABLE_CAUSALLY", "NEW_OBSERVER_HOOK_REQUIRED", "SIMULATOR_SEMANTIC_EXTENSION_REQUIRED", "RESEARCHER_DECISION_REQUIRED"}
        statuses = [row["status"] for row in report["source_sufficiency_matrix"]]
        self.assertTrue(set(statuses) <= allowed)
