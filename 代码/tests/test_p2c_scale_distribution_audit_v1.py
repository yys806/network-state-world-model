from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from pi_jwm.p2c_scale_distribution_audit_v1 import (
    AuditContractError,
    audit_bundle,
    audit_e1_rows,
    build_candidate_formal_data_config,
    validate_candidate_formal_data_config,
    verify_input_manifest,
)


P2B_BUNDLE = Path(
    r"D:\shen\网络组\代码\artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v1"
)


class P2CScaleDistributionAuditTests(unittest.TestCase):
    @unittest.skipUnless(P2B_BUNDLE.is_dir(), "canonical P2-B bundle is not available")
    def test_real_p2b_bundle_reports_observed_facts_without_approval(self):
        report = audit_bundle(P2B_BUNDLE)

        self.assertEqual(report["schema_version"], "PIJWM-P2C-Scale-Distribution-Audit-v1")
        self.assertEqual(report["observed_facts"]["natural"]["episode_count"], 6)
        self.assertEqual(report["observed_facts"]["natural"]["frame_count"], 120)
        self.assertEqual(report["observed_facts"]["fixture"]["frame_count"], 0)
        self.assertEqual(report["observed_facts"]["natural"]["episode_steps"], [20])
        self.assertEqual(
            {tuple(pair) for pair in report["observed_facts"]["natural"]["seed_arm_pairs"]},
            {(0, "orthogonal"), (0, "interference_reuse"),
             (1, "orthogonal"), (1, "interference_reuse"),
             (2, "orthogonal"), (2, "interference_reuse")},
        )
        self.assertEqual(
            report["e1_field_validity"]["field_names"],
            [
                "channel_attenuation_mean_db",
                "channel_attenuation_std_db",
                "prev_active_flow_count",
                "prev_effective_rate_per_s",
                "prev_served_data",
            ],
        )
        self.assertEqual(report["e1_field_validity"]["legacy_placeholder_field_count"], 0)
        self.assertFalse(report["candidate_formal_data_config"]["formal_data_approved"])
        self.assertIn("formal_split_not_frozen", report["blocking_reasons"])
        self.assertEqual(report["observed_facts"]["replay"]["passed_episode_count"], 6)
        self.assertEqual(
            report["observed_facts"]["runtime_guards"]["cpu_rule_versions"],
            {"PIJWM-CPU-Inner-Rule-v1": 120},
        )
        self.assertEqual(
            report["observed_facts"]["runtime_guards"]["lifecycle_alias_guard_frame_count"],
            120,
        )

    def test_e1_audit_rejects_legacy_or_wrong_width(self):
        with self.assertRaises(AuditContractError):
            audit_e1_rows(
                [
                    {
                        "physical_edge_id": "edge-0",
                        "fields": {
                            **{
                                name: {"value": 0.0, "valid_mask": True, "missing_reason": None}
                                for name in (
                                    "channel_attenuation_mean_db",
                                    "channel_attenuation_std_db",
                                    "prev_active_flow_count",
                                    "prev_effective_rate_per_s",
                                    "prev_served_data",
                                )
                            },
                            "legacy_slot_00": {
                                "value": 0.0,
                                "valid_mask": True,
                                "missing_reason": None,
                            },
                        },
                    }
                ]
            )

    def test_e1_audit_requires_mask_reason_consistency(self):
        with self.assertRaises(AuditContractError):
            audit_e1_rows(
                [
                    {
                        "physical_edge_id": "edge-0",
                        "fields": {
                            name: {
                                "value": 0.0,
                                "valid_mask": name != "prev_served_data",
                                "missing_reason": "NO_HISTORY" if name != "prev_served_data" else None,
                            }
                            for name in (
                                "channel_attenuation_mean_db",
                                "channel_attenuation_std_db",
                                "prev_active_flow_count",
                                "prev_effective_rate_per_s",
                                "prev_served_data",
                            )
                        },
                    }
                ]
            )

    def test_candidate_config_rejects_overlapping_splits(self):
        config = build_candidate_formal_data_config(
            {"observed_facts": {"natural": {"seed_values": [0, 1, 2]}}}
        )
        config["seed_split"] = {"train": [0, 1], "validation": [1], "locked_test": [2]}
        errors = validate_candidate_formal_data_config(config)
        self.assertIn("seed_split_overlap", errors)

    def test_input_manifest_verification_detects_source_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            project = root / "project"
            bundle.mkdir()
            (project / "代码").mkdir(parents=True)
            artifact = bundle / "artifact.json"
            source = project / "代码" / "source.py"
            artifact.write_text("{}\n", encoding="utf-8")
            source.write_text("VALUE = 1\n", encoding="utf-8")
            manifest = {
                "required_files": ["artifact.json", "manifest.json"],
                "artifact_hashes": {
                    "artifact.json": hashlib.sha256(artifact.read_bytes()).hexdigest()
                },
                "source_hashes": {
                    "代码/source.py": hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertTrue(verify_input_manifest(bundle, project_root=project)["passed"])
            source.write_text("VALUE = 2\n", encoding="utf-8")
            verification = verify_input_manifest(bundle, project_root=project)
            self.assertFalse(verification["passed"])
            self.assertEqual(verification["source_hash_mismatches"], ["代码/source.py"])


if __name__ == "__main__":
    unittest.main()
