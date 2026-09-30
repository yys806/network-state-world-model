"""Execution identity must prevent CPU/GPU or batch mixing on resume."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

import run_step6_3d_formal_cpu_matrix_v1 as matrix

execution_identity = matrix.execution_identity
validate_resume_result = matrix.validate_resume_result


class GpuRunnerIdentityTests(unittest.TestCase):
    def test_cuda_identity_includes_frozen_execution_fields(self):
        source = {"runner.py": "abc"}
        gpu = execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
                                 batch_size=16, checkpoint_sha256="checkpoint",
                                 source_sha256=source, bucket_strategy="none")
        self.assertEqual(gpu["execution_device"], "cuda")
        self.assertEqual(gpu["precision"], "FP32")
        self.assertFalse(gpu["locked_test"])
        self.assertEqual(gpu["batch_size"], 16)
        self.assertEqual(gpu["gpu_model"], "NVIDIA GeForce RTX 3080 Ti")
        for change in ({"batch_size": 32}, {"device": "cpu", "gpu_model": None},
                       {"source_sha256": {"runner.py": "changed"}}):
            kwargs = dict(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
                          batch_size=16, checkpoint_sha256="checkpoint",
                          source_sha256=source, bucket_strategy="none")
            kwargs.update(change)
            self.assertNotEqual(execution_identity(**kwargs)["execution_config_id"],
                                gpu["execution_config_id"])
        with self.assertRaises(ValueError):
            execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
                               batch_size=16, checkpoint_sha256="checkpoint",
                               source_sha256=source, bucket_strategy="by_shape")

    def test_resume_rejects_legacy_cpu_and_different_gpu_batch(self):
        gpu = execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
                                 batch_size=16, checkpoint_sha256="checkpoint",
                                 source_sha256={"runner.py": "abc"}, bucket_strategy="none")
        base = dict(sample_id="anchor", method="HRS", seed=6301, budget=64,
                    iterations=1, elite_ratio=None)
        valid = {**base, **gpu}
        validate_resume_result(valid, base, gpu)
        with self.assertRaises(ValueError):
            validate_resume_result({**base, "checkpoint_sha256": "checkpoint",
                                    "source_sha256": {"runner.py": "abc"}}, base, gpu)
        with self.assertRaises(ValueError):
            validate_resume_result({**valid, "batch_size": 32}, base, gpu)

    def test_formal_result_file_rejects_legacy_cpu_partial_before_running(self):
        gpu = execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
                                 batch_size=16, checkpoint_sha256="checkpoint",
                                 source_sha256={"runner.py": "abc"}, bucket_strategy="none")
        with tempfile.TemporaryDirectory() as directory, patch.object(matrix, "RESULTS", Path(directory)):
            path = matrix.result_path("smoke", "anchor", "HRS", 6301, 64, 1, None)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"sample_id": "anchor", "method": "HRS",
                "seed": 6301, "budget": 64, "iterations": 1, "elite_ratio": None,
                "checkpoint_sha256": "checkpoint", "source_sha256": {"runner.py": "abc"}}))
            with self.assertRaisesRegex(ValueError, "resume identity mismatch"):
                matrix.solve_or_resume("smoke", "anchor", "HRS", 6301, 64, 1, None,
                                       {"runner.py": "abc"}, gpu)


if __name__ == "__main__":
    unittest.main()
