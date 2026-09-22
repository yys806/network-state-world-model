import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "code" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_step5_1d_unified_model_chain_v1 import OUT  # noqa: E402
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (  # noqa: E402
    load_flow_tensor_batch,
    save_flow_tensor_batch,
)


class Step51DUnifiedModelChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = OUT
        cls.receipt = json.loads((cls.output / "acceptance_receipt.json").read_text(encoding="utf-8"))
        cls.samples = json.loads(
            (ROOT / "code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922/unified_flow_samples.json").read_text(encoding="utf-8")
        )
        cls.pairing = json.loads((cls.output / "sample_pairing_audit.json").read_text(encoding="utf-8"))
        cls.identity = json.loads((cls.output / "identity_audit.json").read_text(encoding="utf-8"))
        cls.actions = json.loads((cls.output / "action_mapping_audit.json").read_text(encoding="utf-8"))
        cls.normalization_lineage = json.loads((cls.output / "normalization_lineage_audit.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((cls.output / "manifest.json").read_text(encoding="utf-8"))

    def test_full_tensor_package_semantic_roundtrip(self):
        package = self.output / "unified_flow_tensor_package.npz"
        tensor = load_flow_tensor_batch(package)
        required = {
            "contract",
            "sample_ids",
            "sample_static",
            "sample_metadata",
            "base_step3_3_validation_checks",
            "flow_normalization_stats",
        }
        self.assertTrue(required <= set(tensor))
        self.assertEqual(tensor["sample_ids"], [row["metadata"]["sample_id"] for row in self.samples])
        self.assertTrue(tensor["base_step3_3_validation_checks"]["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "roundtrip.npz"
            save_flow_tensor_batch(tensor, copied)
            loaded = load_flow_tensor_batch(copied)
        for key in required:
            self.assertEqual(tensor[key], loaded[key], key)
        for key, value in tensor.items():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.array_equal(value, loaded[key]), key)

    def test_receipt_and_pairing_identity(self):
        self.assertTrue(self.receipt["passed"])
        self.assertEqual(self.receipt["sample_count"], 12)
        self.assertEqual(len(self.receipt["checks"]), 47)
        self.assertTrue(all(self.receipt["checks"].values()))
        self.assertTrue(all(self.receipt["executed_scope"].values()))
        self.assertTrue(all(value is False for value in self.receipt["forbidden_scope"].values()))
        self.assertEqual(self.receipt["action_coverage"]["route_nonempty_coverage"], 0)
        self.assertEqual(self.receipt["action_coverage"]["comp_nonempty_coverage"], 0)
        self.assertEqual(self.receipt["action_coverage"]["comm_nonempty_coverage"], 1)
        self.assertEqual(self.receipt["action_coverage"]["mobility_nonempty_coverage"], 48)
        self.assertTrue(self.receipt["checks"]["upstream_normalization_exact_train_lineage"])
        self.assertTrue(self.normalization_lineage["exact_stats_source_lineage"])
        self.assertTrue(self.normalization_lineage["unified_stats_source_lineage_exact"])
        self.assertTrue(self.normalization_lineage["upstream_stats_source_lineage_exact"])
        self.assertEqual(self.normalization_lineage["upstream_stats_source_ids_mode"], "recovered_from_frozen_4_2a_batch")
        self.assertTrue(self.receipt["checks"]["prior_target_runtime_isolation"])
        self.assertTrue(self.receipt["checks"]["posterior_target_runtime_sensitivity"])
        self.assertTrue(self.receipt["provenance"]["source_script_sha256"])
        self.assertTrue(self.manifest["passed"])
        tracked = {"acceptance_receipt.json", "sample_pairing_audit.json", "identity_audit.json", "action_mapping_audit.json", "gradient_summary.json", "recursive_horizon_summary.json", "metric_summary.json"}
        self.assertTrue(tracked <= set(self.manifest["github_tracked_evidence"]))
        self.assertEqual(len(self.pairing), 12)
        self.assertTrue(all(row["sample_id"] == row["target_sample_id"] for row in self.pairing))
        self.assertTrue(all(row["trajectory_id"] == row["target_trajectory_id"] for row in self.pairing))
        self.assertTrue(all(row["anchor_decision_frame"] == row["target_anchor_decision_frame"] for row in self.pairing))
        self.assertEqual(len(self.identity), 12)
        self.assertTrue(all(all(row[name] for name in ("motion_slot_identity", "comm_relation_identity", "flow_slot_identity", "task_slot_identity")) for row in self.identity))

    def test_real_action_mapping_and_explicit_noops(self):
        self.assertEqual(len(self.actions), 24)
        self.assertEqual(self.receipt["action_counts"], {"route": 0, "comm": 1, "comp": 0, "mobility": 48})
        self.assertTrue(all(row["mapping"]["comp_explicit_noop"] for row in self.actions))
        self.assertTrue(all(row["mapping"]["route_explicit_noop"] for row in self.actions))
        self.assertTrue(all(not family["missing"] for sample in self.samples for action in sample["future_action"] for family in action.values()))

    def test_no_hardcoded_sample_index_or_optimizer_step(self):
        source = (SCRIPT_DIR / "build_step5_1d_unified_model_chain_v1.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        hardcoded = any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"samples", "targets"}
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 2
            for node in ast.walk(tree)
        )
        optimizer_step = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            for node in ast.walk(tree)
        )
        self.assertFalse(hardcoded)
        self.assertFalse(optimizer_step)

    def test_receipt_tamper_negative(self):
        tampered = copy.deepcopy(self.receipt["checks"])
        tampered["paired_12"] = False
        self.assertFalse(all(tampered.values()))
        self.assertFalse(bool(all(tampered.values()) and all(not value for value in self.receipt["forbidden_scope"].values())))
        self.assertTrue(self.receipt["checks"]["receipt_tamper_negative"])
        forbidden = copy.deepcopy(self.receipt["forbidden_scope"])
        forbidden["training"] = True
        self.assertFalse(all(not value for value in forbidden.values()))
        self.assertFalse(bool(all(self.receipt["checks"].values()) and all(not value for value in forbidden.values())))
        self.assertTrue(self.receipt["checks"]["forbidden_scope_tamper_negative"])


if __name__ == "__main__":
    unittest.main()
