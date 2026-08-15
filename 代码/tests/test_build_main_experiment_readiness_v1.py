from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_main_experiment_readiness_v1.py"


def load_subject():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("main_readiness_subject", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def passing_reports():
    return (
        {
            "experiment_completed": True,
            "strict_dual_graph_ready": True,
            "reproducibility_passed": True,
            "corruption_detection_passed": True,
        },
        {
            "experiment_completed": True,
            "conservation_ready": True,
            "reproducibility_passed": True,
            "corruption_detection_passed": True,
            "gates": {
                "task_flow_conservation": True,
                "dependency_accounting_valid": True,
                "rb_valid": True,
                "cpu_valid": True,
                "energy_equation_valid": True,
                "channel_energy_input_valid": True,
                "same_seed_reproducible": True,
            },
        },
        {
            "experiment_completed": True,
            "action_sensitivity_ready": True,
            "total_pairs": 6,
            "accepted_pairs": 6,
            "failed_pair_ids": [],
            "corruption_detection_passed": True,
        },
    )


class MainReadinessArtifactTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_script_exists_and_cli_exposes_evidence_paths(self):
        self.assertTrue(SCRIPT_PATH.exists(), msg="main readiness builder is missing")
        if not SCRIPT_PATH.exists():
            return
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("--exp03-validation", result.stdout)
        self.assertIn("--exp04-validation", result.stdout)
        self.assertIn("--exp05-validation", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_builder_freezes_contract_readiness_report_and_hash_manifest(self):
        builder = getattr(self.subject, "build_main_experiment_readiness", None)
        self.assertTrue(callable(builder), msg="readiness artifact builder is missing")
        if not callable(builder):
            return
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exp03, exp04, exp05 = passing_reports()
            paths = [root / f"exp{index}.json" for index in (3, 4, 5)]
            for path, report in zip(paths, (exp03, exp04, exp05)):
                write_json(path, report)
            output_dir = root / "output"

            result = builder(output_dir, paths[0], paths[1], paths[2])

            self.assertEqual(
                {"contract.json", "readiness_report.json", "REPORT.md", "manifest.json"},
                {path.name for path in output_dir.iterdir()},
            )
            self.assertTrue(result["simulation_training_ready"])
            self.assertFalse(result["formal_dataset_ready"])
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(3, len(manifest["input_files"]))
            self.assertEqual(3, len(manifest["output_files"]))

    def test_failed_conservation_evidence_is_not_overridden_by_writer(self):
        builder = getattr(self.subject, "build_main_experiment_readiness", None)
        self.assertTrue(callable(builder), msg="readiness artifact builder is missing")
        if not callable(builder):
            return
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exp03, exp04, exp05 = passing_reports()
            exp04["conservation_ready"] = False
            exp04["gates"]["cpu_valid"] = False
            paths = [root / f"exp{index}.json" for index in (3, 4, 5)]
            for path, report in zip(paths, (exp03, exp04, exp05)):
                write_json(path, report)

            result = builder(root / "output", paths[0], paths[1], paths[2])

            self.assertFalse(result["simulation_training_ready"])
            self.assertIn("exp04_conservation_ready", result["blocking_checks"])


if __name__ == "__main__":
    unittest.main()
