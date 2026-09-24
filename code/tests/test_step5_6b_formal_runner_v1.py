"""CPU/static checks for launch bookkeeping; no formal optimizer work."""
from pathlib import Path
import json
import math
import tempfile
import unittest

from pi_jwm.step5_6b_formal_runner_v1 import atomic_json, estimate_seconds, sha256


class FormalRunnerBookkeepingTests(unittest.TestCase):
    def test_estimate_uses_measured_h4_and_validation_with_labelled_scaling(self) -> None:
        estimate = estimate_seconds(0, 0)
        self.assertAlmostEqual(estimate["training_estimate_seconds"],
                               1104 * 25.026449158787727 / 4 + 1104 * 25.026449158787727 / 2 + 3312 * 25.026449158787727)
        self.assertAlmostEqual(estimate["validation_estimate_seconds"], 5 * 7103.09)
        self.assertAlmostEqual(estimate["total_estimate_seconds"],
                               estimate["training_estimate_seconds"] + estimate["validation_estimate_seconds"])
        self.assertGreater(estimate["total_estimate_seconds"] / 3600, 35)
        self.assertLess(estimate["total_estimate_seconds"] / 3600, 40)
        self.assertEqual(0, estimate_seconds(5520, 5)["remaining_estimate_seconds"])

    def test_atomic_progress_is_parseable_and_replaces_old_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            atomic_json(path, {"status": "RUNNING", "completed_steps": 0})
            atomic_json(path, {"status": "RUNNING", "completed_steps": 1})
            self.assertEqual(1, json.loads(path.read_text(encoding="utf-8"))["completed_steps"])
            self.assertFalse(path.with_name("progress.json.tmp").exists())
            self.assertEqual(sha256(path), sha256(path))
            with self.assertRaises(ValueError):
                atomic_json(path, {"bad": math.nan})


if __name__ == "__main__":
    unittest.main()
