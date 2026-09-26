"""Read-only receipt validation for source facts; no AirFogSim runtime."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from build_step6_0b_planner_action_feasibility_audit_v1 import OUT, payloads


class Step60BSourceAuditTests(unittest.TestCase):
    def test_rebuild_exact_receipts(self):
        for name, record in payloads().items():
            self.assertEqual(json.loads((OUT / name).read_text(encoding="utf-8")), record, name)

    def test_no_git_sha_substitution(self):
        receipt = payloads()["constraint_closure_receipt.json"]
        self.assertIsNone(receipt["airfogsim_source_sha"])
        self.assertFalse(receipt["source_git_provenance_complete"])
        self.assertIn("NO_INDEPENDENT_GIT_METADATA", receipt["airfogsim_git_identity_reason"])
        self.assertEqual(receipt["cpu_verdict"], "STATIC_CAPACITY_ONLY")
        self.assertFalse(receipt["candidate_constraint_code_modified"])
        self.assertEqual(set(receipt["unknown_constraints_remaining"]),
                         {"dynamic_available_cpu", "mobility_numeric_bounds"})

    def test_source_anchors_have_current_hashes(self):
        records = payloads()
        for name in ("cpu_feasibility_source_receipt.json", "uav_mobility_bound_source_receipt.json",
                     "world_model_vs_simulator_mobility_semantics_receipt.json"):
            for anchor in records[name]["source_symbols"].values():
                path = ROOT / anchor["file"]
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertIn(anchor["symbol_or_expression"], lines[anchor["line"] - 1])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), anchor["sha256"])

    def test_no_hard_bound_or_dynamic_cpu_promotion(self):
        records = payloads()
        cpu = records["cpu_feasibility_source_receipt.json"]
        uav = records["uav_mobility_bound_source_receipt.json"]
        self.assertIsNone(cpu["dynamic_available_cpu"]["direct_field"])
        self.assertIsNone(cpu["dynamic_available_cpu"]["causal_derivation"])
        self.assertTrue(all(value is None for value in uav["simulator_hard_bounds"].values()))
        self.assertEqual(uav["formal_dataset_behavior_support"]["kind"], "DATASET_BEHAVIOR_SUPPORT_ONLY")
        self.assertFalse(records["constraint_closure_receipt.json"]["runtime_probe_performed"])


if __name__ == "__main__":
    unittest.main()
