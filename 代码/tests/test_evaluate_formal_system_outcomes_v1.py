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

from test_formal_airfogsim_window_v1 import _write_formal_fixture
from test_formal_system_window_v1 import _write_system_fixture


class EvaluateFormalSystemOutcomesV1Tests(unittest.TestCase):
    def test_evaluates_same_nonlocked_samples_and_reports_missing_baseline_energy(self):
        from evaluate_formal_system_outcomes_v1 import evaluate_training_run
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            system_root = root / "system"
            training_root = root / "training"
            output_root = root / "evaluation"
            tensor_root.mkdir()
            system_root.mkdir()
            _write_formal_fixture(tensor_root)
            _write_system_fixture(system_root)
            run_formal_training(
                tensor_root=tensor_root,
                system_root=system_root,
                use_system_energy_head=True,
                output_dir=training_root,
                device="cpu",
                learned_methods=("pooled_gru",),
                seed=47,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            summary = evaluate_training_run(
                tensor_root=tensor_root,
                system_root=system_root,
                training_root=training_root,
                output_dir=output_root,
                device="cpu",
                batch_size=1,
                step_seconds=0.1,
            )

            self.assertTrue(summary["system_evaluation_complete"])
            self.assertFalse(summary["locked_test_accessed"])
            learned = json.loads(
                (output_root / "pooled_gru__validation.json").read_text(encoding="utf-8")
            )
            baseline = json.loads(
                (output_root / "last_persistence__validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "computed", learned["macro_metrics"]["system.uav_energy.mae"]["status"]
            )
            self.assertEqual(
                "not_computable",
                baseline["macro_metrics"]["system.uav_energy.mae"]["status"],
            )
            self.assertTrue((output_root / "comparison.csv").is_file())
            self.assertTrue((output_root / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
