import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "scripts"))

import audit_v11_gpu_handoff as handoff


class V11GpuHandoffTests(unittest.TestCase):
    def test_split_protocol_is_disjoint_and_locked_seeds_are_not_formal(self):
        result = handoff._check_split_protocol()
        self.assertTrue(result["passed"])
        self.assertEqual(result["overlaps"], [])
        formal = set().union(*handoff.EXPECTED_SEEDS.values())
        self.assertNotIn(18, formal)
        self.assertNotIn(19, formal)

    def test_sample_index_has_causal_full_formal_coverage(self):
        result = handoff._check_sample_index(handoff.SAMPLE_INDEX_DEFAULT)
        self.assertTrue(result["formal_seed_coverage"])
        self.assertEqual(result["rows"], 23400)

    def test_gpu_scripts_use_schema6_then_candidate_set_selector(self):
        result = handoff._check_scripts()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["checks"]["label_script_validation_before_train"])
        self.assertTrue(result["checks"]["selector_script_requires_physical_manifest"])

    def test_schema6_smoke_is_protocol_valid_but_not_a_selector_unlock(self):
        result = handoff._check_schema6_smoke(handoff.SCHEMA6_SMOKE_DEFAULT)
        self.assertTrue(result["protocol_passed"])
        self.assertEqual(result["cache_schema_version"], 6)
        self.assertFalse(result["candidate_gate_passed"])
        self.assertTrue(result["label_generation_allowed"])

    def test_run_writes_handoff_gate_without_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = type(
                "Args",
                (),
                {
                    "output_dir": Path(directory),
                    "sample_index": handoff.SAMPLE_INDEX_DEFAULT,
                    "world_checkpoint": handoff.WORLD_CHECKPOINT_DEFAULT,
                    "policy_checkpoint": handoff.POLICY_CHECKPOINT_DEFAULT,
                    "smoke_gate": handoff.SMOKE_GATE_DEFAULT,
                    "schema6_smoke": handoff.SCHEMA6_SMOKE_DEFAULT,
                    "full_schema6_summary": handoff.FULL_SCHEMA6_SUMMARY_DEFAULT,
                    "bridge_manifest": handoff.BRIDGE_MANIFEST_DEFAULT,
                },
            )()
            with mock.patch.object(handoff.subprocess, "check_output", return_value=""):
                result = handoff.run(args)
            self.assertEqual(result["status"], "ready_for_gpu_selector_training")
            self.assertTrue(result["selector_training_allowed"])
            self.assertFalse(result["formal_train_calibration_labels_required"])
            self.assertEqual(result["physical_bridge"]["bridge_mode"], "task_only")
            self.assertEqual(result["physical_bridge"]["physical_energy_result_kind"], "audit_only")
            self.assertFalse(result["locked_split_access"]["matched_test_accessed"])
            gate = json.loads((Path(directory) / "gpu_handoff_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["commands"]["selector_training"], "bash 代码/scripts/run_v11_candidate_set_selector_gpu.sh")
            manifest = json.loads((Path(directory) / "sha256_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({row["path"] for row in manifest}, {
                "formal_label_command.txt",
                "gpu_handoff_gate.json",
                "gpu_handoff_report.md",
                "protocol_manifest.json",
                "selector_gpu_command.txt",
                "split_manifest.json",
            })
            protocol = json.loads((Path(directory) / "protocol_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["candidate_protocol"], "support_constrained_edge_step_repair_v2")
            self.assertEqual(len(protocol["protocol_digest"]), 64)

    def test_dirty_tracked_tree_blocks_selector_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = type(
                "Args",
                (),
                {
                    "output_dir": Path(directory),
                    "sample_index": handoff.SAMPLE_INDEX_DEFAULT,
                    "world_checkpoint": handoff.WORLD_CHECKPOINT_DEFAULT,
                    "policy_checkpoint": handoff.POLICY_CHECKPOINT_DEFAULT,
                    "smoke_gate": handoff.SMOKE_GATE_DEFAULT,
                    "schema6_smoke": handoff.SCHEMA6_SMOKE_DEFAULT,
                    "full_schema6_summary": handoff.FULL_SCHEMA6_SUMMARY_DEFAULT,
                    "bridge_manifest": handoff.BRIDGE_MANIFEST_DEFAULT,
                },
            )()
            with mock.patch.object(
                handoff.subprocess,
                "check_output",
                return_value=" M tracked_source.py\n",
            ):
                result = handoff.run(args)
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["selector_training_allowed"])
            self.assertFalse(result["source"]["tracked_tree_clean"])
            self.assertIn("tracked source tree is dirty", result["blockers"])


if __name__ == "__main__":
    unittest.main()
