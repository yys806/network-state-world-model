from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2c_scale_distribution_audit_v1 as runner  # noqa: E402


P2B_BUNDLE = CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_p2_full_dual_graph_collector_v1"
P2C_ADVISOR_DOCUMENT = (
    runner.PROJECT_ROOT
    / "记录"
    / "研究进展"
    / "2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md"
)
P2C_ADVISOR_DOCUMENT_KEY = P2C_ADVISOR_DOCUMENT.relative_to(
    runner.PROJECT_ROOT
).as_posix()


class P2CAuditRunnerTests(unittest.TestCase):
    def test_cli_requires_explicit_bundle_and_output_and_has_no_expansive_options(self):
        options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            {"-h", "--help", "--bundle", "--output-dir", "--verify-only"},
            options,
        )
        bundle_action = next(
            action for action in runner.build_parser()._actions if "--bundle" in action.option_strings
        )
        output_action = next(
            action for action in runner.build_parser()._actions if "--output-dir" in action.option_strings
        )
        self.assertTrue(bundle_action.required)
        self.assertTrue(output_action.required)

    @unittest.skipUnless(P2B_BUNDLE.is_dir(), "canonical P2-B bundle is not available")
    def test_publish_and_verify_bind_inputs_outputs_and_portable_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            result = runner.publish_audit_bundle(P2B_BUNDLE, output)
            self.assertEqual(result["audit_status"], "blocked")
            self.assertFalse(result["formal_data_approved"])
            self.assertEqual(
                {
                    "p2c_scale_distribution_audit_v1.json",
                    "p2c_formal_data_config_candidate_v1.json",
                    "manifest.json",
                },
                {path.name for path in output.iterdir()},
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(8, len(manifest["input_hashes"]))
            self.assertEqual(2, len(manifest["artifact_hashes"]))
            self.assertIn(P2C_ADVISOR_DOCUMENT_KEY, manifest["source_hashes"])
            self.assertEqual(
                runner._sha256(P2C_ADVISOR_DOCUMENT),
                manifest["source_hashes"][P2C_ADVISOR_DOCUMENT_KEY],
            )
            self.assertTrue(all(not Path(key).is_absolute() for key in manifest["source_hashes"]))
            self.assertTrue(all(not key.startswith(".worktrees/") for key in manifest["source_hashes"]))
            self.assertTrue(runner.verify_audit_bundle(P2B_BUNDLE, output)["passed"])

            report_path = output / "p2c_scale_distribution_audit_v1.json"
            report_path.write_text(report_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            verification = runner.verify_audit_bundle(P2B_BUNDLE, output)
            self.assertFalse(verification["passed"])
            self.assertTrue(any("artifact hash mismatch" in row for row in verification["errors"]))

    @unittest.skipUnless(P2B_BUNDLE.is_dir(), "canonical P2-B bundle is not available")
    def test_p2c_advisor_document_tampering_breaks_source_verification(self):
        temporary_parent = CODE_ROOT / "artifacts" / "tmp"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary:
            root = Path(temporary)
            copied_document = root / "p2c_advisor_document.md"
            copied_document.write_bytes(P2C_ADVISOR_DOCUMENT.read_bytes())
            source_paths = tuple(
                copied_document if path == P2C_ADVISOR_DOCUMENT else path
                for path in runner.CANONICAL_SOURCE_PATHS
            )
            output = root / "audit"
            runner.publish_audit_bundle(P2B_BUNDLE, output, source_paths=source_paths)

            copied_document.write_text(
                copied_document.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            verification = runner.verify_audit_bundle(
                P2B_BUNDLE,
                output,
                source_paths=source_paths,
            )

            self.assertFalse(verification["passed"])
            self.assertIn("source hash mismatch", verification["errors"])

    def test_publish_refuses_to_write_into_input_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "different"):
                runner.publish_audit_bundle(root, root)


if __name__ == "__main__":
    unittest.main()
