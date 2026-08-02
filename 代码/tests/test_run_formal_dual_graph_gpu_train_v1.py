from __future__ import annotations

import hashlib
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


class RunFormalDualGraphGpuTrainV1Tests(unittest.TestCase):
    def test_gpu_protocol_accepts_only_nonlocked_splits_and_cuda(self):
        from run_formal_dual_graph_gpu_train_v1 import validate_gpu_protocol

        validate_gpu_protocol(("train", "validation", "calibration"), "cuda")
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_gpu_protocol(("train", "locked_test"), "cuda")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_gpu_protocol(("train", "validation"), "cpu")

    def test_move_nested_to_device_preserves_structure_and_metadata(self):
        from run_formal_dual_graph_gpu_train_v1 import move_nested_to_device

        value = {
            "tensor": torch.tensor([1.0]),
            "nested": [torch.tensor([2]), (torch.tensor([3]), "sample")],
            "split": "train",
        }
        moved = move_nested_to_device(value, torch.device("cpu"))

        self.assertEqual("cpu", moved["tensor"].device.type)
        self.assertEqual("cpu", moved["nested"][0].device.type)
        self.assertEqual("cpu", moved["nested"][1][0].device.type)
        self.assertEqual("sample", moved["nested"][1][1])
        self.assertEqual("train", moved["split"])

    def test_device_agnostic_core_writes_reloadable_auditable_artifacts(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_formal_fixture(tensor_root)

            result = run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("pooled_gru",),
                seed=37,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertTrue(result["training_run_complete"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual(["zero_activity", "last_persistence", "pooled_gru"], result["completed_methods"])
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(["train", "validation", "calibration"], config["splits"])
            self.assertFalse(config["locked_test_accessed"])
            self.assertEqual("cpu", config["device"])

            checkpoint_path = output_dir / "checkpoints" / "pooled_gru__best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual("pooled_gru", checkpoint["method"])
            self.assertEqual(1, checkpoint["best_epoch"])
            self.assertIn("model_state_dict", checkpoint)

            for relative in (
                "sample_ids.json",
                "class_weights.json",
                "training_history.json",
                "comparison.csv",
                "runtime.json",
                "run_summary.json",
                "manifest.json",
            ):
                self.assertTrue((output_dir / relative).is_file(), relative)
            for split in ("validation", "calibration"):
                self.assertTrue((output_dir / "metrics" / f"pooled_gru__{split}.json").is_file())

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("PI-JWM-formal-training-manifest-v1", manifest["schema_version"])
            for relative, metadata in manifest["files"].items():
                payload = (output_dir / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])
                self.assertEqual(len(payload), metadata["bytes"])


if __name__ == "__main__":
    unittest.main()
