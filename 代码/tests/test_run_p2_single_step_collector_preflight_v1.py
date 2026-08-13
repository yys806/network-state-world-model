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
    @staticmethod
    def _field(payloads, candidate_index, name):
        fields = payloads["field_mask_audit.json"]["candidate_audits"][candidate_index][
            "fields"
        ]
        return next(row for row in fields if row["name"] == name)

    def test_fixture_separates_decision_time_and_outcome_channel_evidence(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        audit = payloads["field_mask_audit.json"]["candidate_audits"][0]
        event = payloads["transfer_events.json"]["candidates"][0]["events"][0]
        pre_rb = self._field(
            payloads, 0, "pre_rb_optional.channel_attenuation_db"
        )

        self.assertEqual("before_action_setters", audit["decision_time_channel"]["capture_phase"])
        self.assertEqual("after_fast_fading_before_transfer", event["capture_phase"])
        self.assertEqual(
            "direct_decision_time_csi_before_setters", pre_rb["provenance"]
        )
        self.assertEqual(
            "outcome_only_not_same_frame_decision_input", event["temporal_role"]
        )
        self.assertLess(
            audit["temporal_trace"].index("decision_time_observation_captured"),
            audit["temporal_trace"].index("action_setters_called"),
        )

    def test_writer_rejects_outcome_channel_provenance_as_action_pre(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        self._field(
            payloads, 0, "pre_rb_optional.channel_attenuation_db"
        )["provenance"] = "direct_runtime_channel_event"
        payloads["validation_report.json"]["passed"] = True

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "temporal evidence"):
                runner.write_preflight_bundle(Path(tmp), payloads)
            self.assertFalse((Path(tmp) / "manifest.json").exists())

    def test_temporal_validator_accepts_float32_serialization_of_direct_csi(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        decision = payloads["field_mask_audit.json"]["candidate_audits"][0][
            "decision_time_channel"
        ]
        decision["channel_attenuation_db"] = [99.67203853753406]
        pre_rb = self._field(
            payloads, 0, "pre_rb_optional.channel_attenuation_db"
        )
        pre_rb["value"] = [99.67203521728516]
        self._field(
            payloads, 0, "pre_link.channel_attenuation_mean_db"
        )["value"] = 99.67203521728516

        self.assertEqual([], runner.validate_temporal_payloads(payloads))

    def test_verifier_recomputes_temporal_evidence_after_tampering(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner.write_preflight_bundle(output_dir, runner.fake_passing_payloads_for_test())
            audit_path = output_dir / "field_mask_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["candidate_audits"][0]["temporal_trace"] = [
                "action_setters_called",
                "decision_time_observation_captured",
            ]
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            report = runner.verify_preflight_bundle(output_dir)
            self.assertFalse(report["passed"])
            self.assertTrue(any("temporal evidence" in row for row in report["errors"]))

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
