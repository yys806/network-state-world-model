import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts")]


class Step62AAuditTests(unittest.TestCase):
    def test_receipts_contract_and_scope(self):
        out = ROOT / "code" / "artifacts" / "protocols" / "pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928"
        receipt = json.loads((out / "01_source_semantics_receipt.json").read_text(encoding="utf-8"))
        contract = json.loads((out / "11_planner_objective_contract_v1.json").read_text(encoding="utf-8"))
        deadline = json.loads((out / "02_deadline_lifecycle_audit.json").read_text(encoding="utf-8"))
        provenance = json.loads((out / "10_objective_field_provenance_matrix.json").read_text(encoding="utf-8"))
        readiness = json.loads((out / "13_step6_2b_readiness.json").read_text(encoding="utf-8"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["scope"]["cpu_only"])
        self.assertFalse(any(receipt["scope"][key] for key in ("checkpoint_loaded", "training", "optimizer_step", "formal_dataset_modified", "locked_test_accessed", "ranking", "closed_loop")))
        self.assertEqual(contract["tuple"], ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"])
        self.assertEqual(contract["order"], "lexicographic_minimize")
        self.assertEqual(readiness["STEP_6_2B_READINESS"], "BLOCKED")
        self.assertFalse(manifest["future_task_schedule_used_by_objective"])
        self.assertFalse(manifest["future_only_task_in_objective_cohort"])
        self.assertFalse(manifest["locked_test_accessed"])
        self.assertEqual(len(manifest["artifact_files"]), 14)
        self.assertNotIn("CAUSALLY_DERIVABLE", provenance["allowed_layer_statuses"])
        rows = {row["field"]: row for row in provenance["rows"]}
        self.assertEqual(rows["deadline"]["layer_status"]["formal_raw"], "SOURCE_AVAILABLE_BUT_NOT_EXPOSED")
        self.assertEqual(rows["deadline"]["layer_status"]["planner"], "PLANNER_SIDE_STATE_REQUIRED")
        self.assertEqual(rows["arrival_time"]["layer_status"]["formal_raw"], "CAUSALLY_EXPOSED")
        self.assertIn("delay <= deadline + 1e-5", deadline["same_step_tie"])
        self.assertIn("delay <= deadline", deadline["same_step_tie"])


if __name__ == "__main__":
    unittest.main()
