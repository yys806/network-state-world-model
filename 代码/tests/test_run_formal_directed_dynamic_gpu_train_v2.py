from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_formal_airfogsim_window_v1 import _write_formal_fixture


class RunFormalDirectedDynamicGpuTrainV2Tests(unittest.TestCase):
    def test_protocol_rejects_cpu_and_locked_test(self):
        from run_formal_directed_dynamic_gpu_train_v2 import (
            validate_directed_dynamic_gpu_protocol,
        )

        validate_directed_dynamic_gpu_protocol(
            ("train", "validation", "calibration"), "cuda"
        )
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_directed_dynamic_gpu_protocol(("train", "locked_test"), "cuda")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_directed_dynamic_gpu_protocol(("train", "validation"), "cpu")

    def test_device_agnostic_core_saves_explicit_v2_checkpoint_boundary(self):
        from run_formal_directed_dynamic_gpu_train_v2 import (
            run_directed_dynamic_training,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_formal_fixture(tensor_root)

            result = run_directed_dynamic_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("coupled_directed_dynamic_residual_v2",),
                seed=47,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertTrue(result["training_run_complete"])
            self.assertFalse(result["locked_test_accessed"])
            checkpoint = torch.load(
                output_dir
                / "checkpoints"
                / "coupled_directed_dynamic_residual_v2__best.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual("directed_dynamic_v2", checkpoint["model_version"])
            self.assertEqual("deterministic", checkpoint["latent_dynamics"])
            self.assertTrue(checkpoint["model_config"]["residual_state_prediction"])
            self.assertNotIn("mode", checkpoint["model_config"])
            registry = json.loads(
                (output_dir / "method_registry.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "directed_dynamic_v2",
                registry["coupled_directed_dynamic_residual_v2"]["model_version"],
            )
            runtime = json.loads(
                (output_dir / "runtime.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                runtime["coupled_directed_dynamic_residual_v2"][
                    "checkpoint_reload_verified"
                ]
            )


if __name__ == "__main__":
    unittest.main()
