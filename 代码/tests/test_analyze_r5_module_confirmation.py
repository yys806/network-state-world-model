from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_r5_module_confirmation.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_r5_module_confirmation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load analysis script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyzeR5ModuleConfirmationScriptTest(unittest.TestCase):
    def test_output_directory_must_not_exist(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "output directory"):
                module.require_new_output_directory(output)

    def test_checkpoint_resolution_uses_existing_root_only_for_B(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            confirmation = root / "confirmation"
            existing = root / "existing"
            report_b = {"combination_id": "B", "checkpoint": "combinations/B/seed_1/best.pt"}
            report_j = {"combination_id": "J", "checkpoint": "combinations/J/seed_1/best.pt"}

            self.assertEqual(
                module.resolve_checkpoint_path(report_b, confirmation, existing),
                existing / "combinations/B/seed_1/best.pt",
            )
            self.assertEqual(
                module.resolve_checkpoint_path(report_j, confirmation, existing),
                confirmation / "combinations/J/seed_1/best.pt",
            )

    def test_input_loader_rejects_locked_test_or_incomplete_report_bundle(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trained_run_reports.json").write_text("[]", encoding="utf-8")
            (root / "reused_run_reports.json").write_text("[]", encoding="utf-8")
            (root / "validation_windows.json").write_text("{}", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "input_provenance.json").write_text(
                json.dumps({"locked_test_accessed": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "locked-test"):
                module.load_confirmation_bundle_inputs(root)


if __name__ == "__main__":
    unittest.main()
