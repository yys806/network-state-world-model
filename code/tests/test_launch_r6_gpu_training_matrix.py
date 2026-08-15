from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch_r6_gpu_training_matrix.py"


def _load_subject():
    spec = importlib.util.spec_from_file_location("launch_r6_gpu_training_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class R6GPUTrainingMatrixLauncherTest(unittest.TestCase):
    def test_builds_eighteen_isolated_resumable_run_commands(self) -> None:
        subject = _load_subject()
        with tempfile.TemporaryDirectory() as directory:
            commands = subject.build_formal_commands(
                python_executable="python",
                output_dir=Path(directory),
                device="cuda",
                target_environment_steps=10000,
                sumo_port_base=18813,
            )
        self.assertEqual(18, len(commands))
        run_ids = {row.run_id for row in commands}
        self.assertEqual(18, len(run_ids))
        self.assertTrue(all("--run-id" in row.argv for row in commands))
        self.assertTrue(all("10000" in row.argv for row in commands))
        self.assertEqual(list(range(18813, 18831)), [row.sumo_port for row in commands])
        self.assertTrue(all("locked_test" not in " ".join(row.argv) for row in commands))

    def test_completed_summary_is_skipped_only_at_full_budget(self) -> None:
        subject = _load_subject()
        self.assertTrue(
            subject.is_complete_summary(
                {
                    "status": "complete",
                    "formal": True,
                    "environment_steps": 100000,
                    "state_source": "online_airfogsim_strict_dual_graph",
                    "locked_test_accessed": False,
                    "checkpoint_reload_verified": True,
                },
                target_steps=100000,
            )
        )
        self.assertFalse(
            subject.is_complete_summary(
                {
                    "status": "complete",
                    "formal": True,
                    "environment_steps": 2,
                    "state_source": "online_airfogsim_strict_dual_graph",
                    "locked_test_accessed": False,
                    "checkpoint_reload_verified": True,
                },
                target_steps=100000,
            )
        )


if __name__ == "__main__":
    unittest.main()
