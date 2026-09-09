from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class ReevaluateFormalCpuRunsV1Tests(unittest.TestCase):
    def test_comparison_rows_have_no_duplicate_persistence_columns(self):
        from reevaluate_formal_cpu_runs_v1 import _comparison_row

        report = {
            "horizons": {"overall": {"metrics": {
                "event.link_activity.f1": {"status": "computed", "value": 0.5},
                "state.node.x.mae": {"status": "computed", "value": 1.0},
                "system.communication_throughput.mae": {"status": "computed", "value": 2.0},
                "resource.rb_occupancy.mae": {"status": "computed", "value": 3.0},
                "state.task.delay.mae": {"status": "computed", "value": 4.0},
            }}}
        }
        row = _comparison_row(
            "last_persistence", {"link_activity": 0.5},
            {"validation": report, "calibration": report}, None,
        )
        self.assertNotIn("validation_persistence_persistence_link_f1", row)
        self.assertIn("validation_persistence_link_f1", row)

    def test_reevaluation_requires_existing_nonlocked_run_and_writes_new_summary(self):
        from reevaluate_formal_cpu_runs_v1 import reevaluate_run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "out"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"locked_test_accessed": False, "device": "cpu", "tensor_root": str(root / "tensor")}), encoding="utf-8"
            )
            (run_dir / "sample_ids.json").write_text(
                json.dumps({"train": [], "validation": [], "calibration": []}), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError):
                reevaluate_run(run_dir=run_dir, output_dir=output_dir)

    def test_reevaluation_rejects_locked_test_run_metadata(self):
        from reevaluate_formal_cpu_runs_v1 import reevaluate_run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"locked_test_accessed": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "locked_test"):
                reevaluate_run(run_dir=run_dir, output_dir=root / "out")


if __name__ == "__main__":
    unittest.main()
