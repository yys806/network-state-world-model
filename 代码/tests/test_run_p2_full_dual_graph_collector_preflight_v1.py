from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2_full_dual_graph_collector_preflight_v1 as runner  # noqa: E402


class RunnerContractTests(unittest.TestCase):
    def test_canonical_matrix_has_three_seeds_and_both_arms(self):
        specs = runner.canonical_episode_specs((0, 1, 2))
        self.assertEqual(6, len(specs))
        self.assertEqual({0, 1, 2}, {row["seed"] for row in specs})
        for seed in (0, 1, 2):
            self.assertEqual(
                {"orthogonal", "interference_reuse"},
                {row["arm"] for row in specs if row["seed"] == seed},
            )

    def test_request_gate_rejects_noncanonical_seed_step_and_substep(self):
        runner.validate_run_request((0, 1, 2), 20, 0.1, 0.1)
        with self.assertRaisesRegex(ValueError, "canonical seeds"):
            runner.validate_run_request((0, 1), 20, 0.1, 0.1)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            runner.validate_run_request((0, 1, 2), 19, 0.1, 0.1)
        with self.assertRaisesRegex(ValueError, "ratio"):
            runner.validate_run_request((0, 1, 2), 20, 0.2, 0.1)

    def test_cli_exposes_no_gpu_training_locked_or_dataset_option(self):
        options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            {"-h", "--help", "--output-dir", "--verify-only", "--seeds", "--steps"},
            options,
        )

    def test_required_fixture_matrix_is_exact_and_nontraining(self):
        payloads = runner.fake_passing_payloads_for_test()
        fixtures = payloads["coverage_report.json"]["fixtures"]
        self.assertEqual(set(runner.REQUIRED_FIXTURES), set(fixtures))
        for name, row in fixtures.items():
            self.assertEqual(name, row["fixture_name"])
            self.assertIs(row["fixture"], True)
            self.assertIs(row["training_eligible"], False)
            self.assertGreaterEqual(row["real_airfogsim_step_count"], 1)
        self.assertEqual([], runner.validate_preflight_payloads(payloads, allow_test_payload=True))

    def test_missing_or_failed_fixture_blocks_publication(self):
        payloads = runner.fake_passing_payloads_for_test()
        del payloads["coverage_report.json"]["fixtures"][runner.REQUIRED_FIXTURES[0]]
        self.assertTrue(any("fixture matrix" in row for row in runner.validate_preflight_payloads(payloads, allow_test_payload=True)))
        payloads = runner.fake_passing_payloads_for_test()
        payloads["coverage_report.json"]["fixtures"][runner.REQUIRED_FIXTURES[0]]["passed"] = False
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "preflight payload invalid"):
                runner.publish_preflight_bundle(
                    Path(temporary) / "blocked",
                    payloads,
                    source_paths=[Path(runner.__file__)],
                    allow_test_payload=True,
                )


class RunnerManifestTests(unittest.TestCase):
    def test_canonical_manifest_binds_runtime_test_design_and_config_closure(self):
        project_root = runner.CODE_ROOT.parent
        required_paths = {
            runner.CODE_ROOT / "scripts" / "run_p2_single_step_collector_preflight_v1.py",
            runner.CODE_ROOT / "scripts" / "small_experiments" / "airfogsim_strict_dual_graph_preflight.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "airfogsim_contract_adapter.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "airfogsim_single_step_collector_v1.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "airfogsim_cpu_inner_rule_v1.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "cpu_inner_rule_v1.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "information_edge_contract_v4.py",
            runner.CODE_ROOT / "src" / "pi_jwm" / "single_step_collector_contract_v1.py",
            runner.CODE_ROOT / "tests" / "test_run_p2_full_dual_graph_collector_preflight_v1.py",
            project_root / "docs" / "superpowers" / "plans" / "2026-08-13-v4-full-dual-graph-collector.md",
            project_root / "文档" / "研究进展" / "2026-08-13-PI-JWM-v4全双图采集器设计.md",
            runner.CODE_ROOT / "reference" / "AirFogSim" / "examples" / "config.yaml",
        }
        airfogsim_sources = set(
            (runner.CODE_ROOT / "reference" / "AirFogSim" / "airfogsim").rglob("*.py")
        )
        canonical_paths = set(runner.CANONICAL_SOURCE_PATHS)
        self.assertTrue(required_paths.issubset(canonical_paths))
        self.assertTrue(airfogsim_sources.issubset(canonical_paths))

        payloads = runner.fake_passing_payloads_for_test()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            runner.publish_preflight_bundle(
                output,
                payloads,
                source_paths=runner.CANONICAL_SOURCE_PATHS,
                allow_test_payload=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            expected_keys = set(runner._source_keys(runner.CANONICAL_SOURCE_PATHS))
            self.assertEqual(expected_keys, set(manifest["source_hashes"]))
            self.assertTrue(all(not key.startswith(".worktrees/") for key in expected_keys))
            self.assertIn(
                "代码/reference/AirFogSim/airfogsim/airfogsim_env.py",
                expected_keys,
            )

    def test_verify_only_detects_frame_and_source_tampering(self):
        payloads = runner.fake_passing_payloads_for_test()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            output = root / "bundle"
            runner.publish_preflight_bundle(
                output,
                payloads,
                source_paths=[source],
                allow_test_payload=True,
            )
            self.assertTrue(runner.verify_preflight_bundle(output, source_paths=[source])["passed"])

            frames_path = output / "frames.jsonl"
            frames_path.write_text(frames_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            frame_report = runner.verify_preflight_bundle(output, source_paths=[source])
            self.assertFalse(frame_report["passed"])
            self.assertTrue(any("artifact hash mismatch" in error for error in frame_report["errors"]))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            manifest["artifact_hashes"]["frames.jsonl"] = hashlib.sha256(frames_path.read_bytes()).hexdigest()
            (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            source.write_text("VALUE = 2\n", encoding="utf-8")
            source_report = runner.verify_preflight_bundle(output, source_paths=[source])
            self.assertFalse(source_report["passed"])
            self.assertTrue(any("source hash mismatch" in error for error in source_report["errors"]))


if __name__ == "__main__":
    unittest.main()
