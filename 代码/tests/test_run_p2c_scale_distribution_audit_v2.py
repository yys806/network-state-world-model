from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

spec = importlib.util.spec_from_file_location(
    "p2c_runner_v2", SCRIPTS_ROOT / "run_p2c_scale_distribution_audit_v2.py"
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

from pi_jwm.full_dual_graph_artifact_v2 import publish_success_bundle  # noqa: E402
from test_p2c_scale_distribution_audit_v2 import (  # noqa: E402
    build_real_shape_v2_payloads,
)


class P2CAuditRunnerV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "p2b_source.py"
        self.source.write_text("value = 1\n", encoding="utf-8")
        self.audit_source = self.root / "audit_source.py"
        self.audit_source.write_text("value = 1\n", encoding="utf-8")
        self.bundle = self.root / "p2b-v2"
        publish_success_bundle(
            self.bundle, build_real_shape_v2_payloads(), [self.source]
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_cli_requires_explicit_bundle_and_output_without_expansive_options(self):
        parser = runner.build_parser()
        actions = {option for action in parser._actions for option in action.option_strings}
        self.assertEqual(
            actions,
            {"-h", "--help", "--bundle", "--output-dir", "--verify-only"},
        )
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_publish_and_verify_bind_nine_inputs_and_sources(self):
        output = self.root / "p2c-v2"
        result = runner.publish_audit_bundle(
            self.bundle,
            output,
            source_paths=[self.audit_source],
            project_root=self.root,
        )

        self.assertEqual(result["audit_status"], "blocked")
        self.assertFalse(result["formal_data_approved"])
        self.assertEqual(len(list(output.iterdir())), 3)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["input_hashes"]), 9)
        self.assertEqual(
            runner.verify_audit_bundle(
                self.bundle,
                output,
                source_paths=[self.audit_source],
                project_root=self.root,
            ),
            {"passed": True, "errors": []},
        )

    def test_verify_detects_input_artifact_and_source_tampering(self):
        output = self.root / "p2c-v2"
        runner.publish_audit_bundle(
            self.bundle,
            output,
            source_paths=[self.audit_source],
            project_root=self.root,
        )
        self.audit_source.write_text("value = 2\n", encoding="utf-8")
        report = runner.verify_audit_bundle(
            self.bundle,
            output,
            source_paths=[self.audit_source],
            project_root=self.root,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("source hash mismatch" in error for error in report["errors"]))

    def test_publish_refuses_input_output_alias_and_overwrite(self):
        with self.assertRaises(ValueError):
            runner.publish_audit_bundle(
                self.bundle,
                self.bundle,
                source_paths=[self.audit_source],
                project_root=self.root,
            )
        output = self.root / "p2c-v2"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            runner.publish_audit_bundle(
                self.bundle,
                output,
                source_paths=[self.audit_source],
                project_root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
