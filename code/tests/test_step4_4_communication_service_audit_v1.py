import copy
import unittest

from pi_jwm.step4_4_communication_service_audit_v1 import (
    AUDIT_SCHEMA_VERSION,
    build_communication_service_audit,
    expected_service_verdict,
    validate_communication_service_audit,
)


class Step44CommunicationServiceAuditTests(unittest.TestCase):
    def test_actual_service_dependency_matrix_is_complete(self):
        report = build_communication_service_audit()
        self.assertEqual(report["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertTrue(
            {
                "csi", "rb_allocation", "bandwidth", "transmit_power",
                "interference", "noise", "fast_fading", "outage_draw",
                "wired_capacity", "wired_active_flow_count", "slot_duration",
            }.issubset({row["name"] for row in report["dependency_matrix"]}),
        )

    def test_verdict_is_derived_from_unobservable_outage(self):
        report = build_communication_service_audit()
        outage = next(row for row in report["dependency_matrix"] if row["name"] == "outage_draw")
        self.assertEqual(outage["classification"], "UNAVAILABLE_UNOBSERVABLE")
        self.assertEqual(outage["decision_time_role"], "not_available; outcome_only_after_random_draw")
        self.assertEqual(expected_service_verdict(report), "SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED")
        self.assertEqual(report["verdict"], expected_service_verdict(report))
        self.assertTrue(validate_communication_service_audit(report)["passed"])

    def test_nominal_wireless_rule_and_actual_service_are_distinct(self):
        report = build_communication_service_audit()
        wireless = report["service_paths"]["wireless"]
        self.assertTrue(wireless["nominal_rate_rule_recoverable"])
        self.assertFalse(wireless["actual_service_uniquely_recoverable"])
        self.assertEqual(wireless["unresolved_actual_service_factor"], "random per-RB outage realization")

    def test_wired_missing_state_is_additive_not_residual(self):
        report = build_communication_service_audit()
        wired = report["service_paths"]["wired"]
        self.assertEqual(wired["gap_class"], "SIMULATOR_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED")
        self.assertTrue(wired["minimal_additive_extension_possible"])

    def test_verdict_tamper_is_rejected(self):
        report = build_communication_service_audit()
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "SERVICE_RULE_SUFFICIENT"
        receipt = validate_communication_service_audit(tampered)
        self.assertFalse(receipt["verdict_matches_evidence"])
        self.assertFalse(receipt["passed"])

    def test_outcome_only_outage_cannot_be_promoted_to_causal_input(self):
        report = build_communication_service_audit()
        tampered = copy.deepcopy(report)
        outage = next(row for row in tampered["dependency_matrix"] if row["name"] == "outage_draw")
        outage["classification"] = "CURRENT_CAUSAL_STATE_AVAILABLE"
        outage["decision_time_role"] = "decision_input"
        self.assertFalse(validate_communication_service_audit(tampered)["passed"])

    def test_scope_is_non_expansive(self):
        report = build_communication_service_audit()
        self.assertFalse(report["world_model_implementation_started"])
        for name in ("training", "optimizer", "loss", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim"):
            self.assertFalse(report["scope"][name])


if __name__ == "__main__":
    unittest.main()
