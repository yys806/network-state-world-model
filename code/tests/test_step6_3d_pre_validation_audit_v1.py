"""Audit evidence regression; confirms findings, not production acceptance."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_step6_3d_pre_validation_v1 as audit


class PreValidationAuditTests(unittest.TestCase):
    def test_exact_rational_update_and_method_layers(self):
        self.assertEqual(audit.math_and_layers()["verdict"], "PASS")

    def test_cached_old_elites_reproduce_unresolved_gate(self):
        receipt = audit.repeated_elite_counterexample()
        self.assertEqual(receipt["verdict"], "BLOCKED_PENDING_NEWLY_SCOREABLE_DEFINITION")
        for row in receipt["evidence"].values():
            self.assertEqual(row["new_scoreable_after_first_iteration"], 0)
            self.assertTrue(all(call["proposal_tables_changed"] for call in row["update_calls"]))

    def test_identical_requests_accounted_equally(self):
        self.assertEqual(audit.fairness()["verdict"], "PASS")

    def test_selection_cluster_oracle(self):
        self.assertEqual(audit.selection_oracle()["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
