import copy
import unittest

from pi_jwm.step4_2b_stateful_flow_source_audit_v1 import (
    AUDIT_SCHEMA_VERSION,
    build_stateful_flow_source_audit,
    validate_stateful_flow_source_audit,
)


class Step42BStatefulFlowSourceAuditTests(unittest.TestCase):
    def test_receipt_is_explicitly_not_yet_supported_and_scoped(self):
        report = build_stateful_flow_source_audit()
        self.assertEqual(report["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(report["verdict"], "FLOW_CONTRACT_NOT_YET_SUPPORTED")
        self.assertFalse(report["graph_builder_started"])
        self.assertFalse(report["training"])
        self.assertFalse(report["gpu"])
        self.assertFalse(report["locked_test"])
        self.assertTrue(validate_stateful_flow_source_audit(report)["passed"])

    def test_input_return_dependency_are_separate(self):
        evidence = build_stateful_flow_source_audit()["evidence"]
        self.assertEqual(evidence["input"]["candidate_a_logical_end_to_end"]["verdict"], "PARTIALLY_CONSTRUCTIBLE")
        self.assertEqual(evidence["return"]["verdict"], "PARTIALLY_CONSTRUCTIBLE")
        self.assertEqual(evidence["dependency_data"]["verdict"], "NOT_YET_SUPPORTED")
        self.assertTrue(evidence["minimum_definition_03"]["task_dag_is_not_dependency_flow"])

    def test_hop_reset_blocks_end_to_end_remaining_derivation(self):
        evidence = build_stateful_flow_source_audit()["evidence"]
        progress = evidence["task_progress_vs_flow_progress"]
        self.assertIn("resets", progress["in_stage_transmitted_size"])
        self.assertFalse(next(row for row in build_stateful_flow_source_audit()["checks"] if row["name"] == "input_end_to_end_remaining_proven")["passed"])

    def test_negative_tamper_fails_machine_validator(self):
        report = build_stateful_flow_source_audit()
        tampered = copy.deepcopy(report)
        tampered["gpu"] = True
        self.assertFalse(validate_stateful_flow_source_audit(tampered)["scope_non_expansive"])

    def test_candidate_fields_are_not_silently_promoted(self):
        evidence = build_stateful_flow_source_audit()["evidence"]
        self.assertIn("no simulator Flow ID", evidence["input"]["candidate_a_logical_end_to_end"]["stable_identity"])
        self.assertEqual(evidence["remaining_raw_sources"]["dynamic_available_cpu"], "RAW_INSUFFICIENT")
        self.assertEqual(evidence["remaining_raw_sources"]["dag_dependency_payload"], "RAW_INSUFFICIENT")
