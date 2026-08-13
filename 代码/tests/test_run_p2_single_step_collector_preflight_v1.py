from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_p2_single_step_collector_preflight_v1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("run_p2_single_step_collector_preflight_v1", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class P2SingleStepCollectorRunnerTests(unittest.TestCase):
    def test_writer_outputs_required_files_and_blocks_training_claims(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            payloads = runner.fake_passing_payloads_for_test()
            summary = runner.write_preflight_bundle(output_dir, payloads)
            self.assertEqual(set(summary["required_files"]), set(runner.REQUIRED_FILES))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["status_flags"]["gpu_started"])
            self.assertFalse(manifest["status_flags"]["training_eligible"])
            self.assertTrue(runner.verify_preflight_bundle(output_dir)["passed"])

    def test_writer_refuses_success_when_validation_failed(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        payloads["validation_report.json"]["passed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "validation"):
                runner.write_preflight_bundle(Path(tmp), payloads)
            self.assertFalse((Path(tmp) / "manifest.json").exists())

    def test_verifier_detects_manifest_source_hash_tampering(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner.write_preflight_bundle(output_dir, runner.fake_passing_payloads_for_test())
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runner_key = next(
                key for key in manifest["source_hashes"]
                if key.endswith("run_p2_single_step_collector_preflight_v1.py")
            )
            manifest["source_hashes"][runner_key] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = runner.verify_preflight_bundle(output_dir)
            self.assertFalse(report["passed"])
            self.assertTrue(any("source hash mismatch" in row for row in report["errors"]))


if __name__ == "__main__":
    unittest.main()
