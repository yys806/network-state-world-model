from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
SCRIPT_PATH = CODE_ROOT / "scripts" / "run_cpu_inner_rule_preflight_v1.py"
DESIGN_PATH = WORKSPACE_ROOT / "记录" / "研究进展" / "2026-08-13-PI-JWM-P1-A-CPU动作边界冻结设计.md"
AIRFOGSIM_ROOT = CODE_ROOT / "reference" / "AirFogSim"


def load_subject():
    spec = importlib.util.spec_from_file_location("run_cpu_inner_rule_preflight_v1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load P2-A preflight script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CpuInnerRulePreflightV1Test(unittest.TestCase):
    def test_bundle_is_truthful_hashed_and_non_training(self) -> None:
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "bundle"
            summary = subject.run_preflight(
                output_dir=output,
                design_path=DESIGN_PATH,
                airfogsim_root=AIRFOGSIM_ROOT,
            )
            expected_files = {
                "rule_contract.json",
                "sample_cases.csv",
                "rejected_records.csv",
                "summary.json",
                "manifest.json",
            }
            self.assertEqual(expected_files, {path.name for path in output.iterdir()})
            self.assertTrue(summary["p2_a_cpu_inner_rule_preflight_verified"])
            self.assertTrue(summary["airfogsim_task_callback_interface_parity"])
            self.assertFalse(summary["v4_collector_implemented"])
            self.assertFalse(summary["v4_dataset_complete"])
            self.assertFalse(summary["gpu_started"])
            self.assertFalse(summary["locked_test_accessed"])
            self.assertFalse(summary["final_method_frozen"])

            with (output / "sample_cases.csv").open(encoding="utf-8", newline="") as handle:
                sample_rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(sample_rows), 10)
            self.assertTrue(all(row["sample_origin"] == "contract_fixture" for row in sample_rows))
            self.assertTrue(all(row["training_eligible"] == "False" for row in sample_rows))
            self.assertTrue(all(row["passed"] == "True" for row in sample_rows))
            self.assertIn("airfogsim_callback_parity", {row["case_id"] for row in sample_rows})

            with (output / "rejected_records.csv").open(encoding="utf-8", newline="") as handle:
                rejected = list(csv.DictReader(handle))
            self.assertEqual(6, len(rejected))
            self.assertIn("nonfinite_demand_rate", {row["case_id"] for row in rejected})
            self.assertTrue(all(row["expected_rejection"] == "True" for row in rejected))
            self.assertTrue(all(row["training_eligible"] == "False" for row in rejected))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subject.PREFLIGHT_VERSION, manifest["preflight_version"])
            self.assertEqual(DESIGN_PATH.resolve().as_posix(), manifest["design_input"]["path"])
            self.assertEqual(sha256(DESIGN_PATH), manifest["design_input"]["sha256"])
            self.assertIn("airfogsim/entities/task.py", manifest["airfogsim_source_files"])
            for name in expected_files - {"manifest.json"}:
                metadata = manifest["output_files"][name]
                self.assertEqual((output / name).resolve().as_posix(), metadata["path"])
                self.assertEqual(sha256(output / name), metadata["sha256"])
                self.assertEqual((output / name).stat().st_size, metadata["size_bytes"])

    def test_existing_output_is_never_overwritten(self) -> None:
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "bundle"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                subject.run_preflight(
                    output_dir=output,
                    design_path=DESIGN_PATH,
                    airfogsim_root=AIRFOGSIM_ROOT,
                )
            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))

    def test_failure_before_publish_leaves_no_partial_bundle(self) -> None:
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "bundle"
            with patch.object(subject, "build_evidence", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    subject.run_preflight(
                        output_dir=output,
                        design_path=DESIGN_PATH,
                        airfogsim_root=AIRFOGSIM_ROOT,
                    )
            self.assertFalse(output.exists())
            self.assertEqual([], list(root.glob(".bundle.tmp-*")))


if __name__ == "__main__":
    unittest.main()
