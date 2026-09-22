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
        self.assertEqual(len(self.receipt["checks"]), 42)
        self.assertTrue(all(self.receipt["checks"].values()))
        self.assertTrue(all(value is False for value in self.receipt["scope"].values()))
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
        self.assertTrue(self.receipt["checks"]["receipt_tamper_negative"])


if __name__ == "__main__":
    unittest.main()
