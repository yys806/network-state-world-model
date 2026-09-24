"""CPU-only checks for exact full-validation aggregation across GPU workers."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from run_step5_6a_gpu_smoke_v1 import merge_validation


MANIFEST = Path(__file__).parents[1] / "artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"


class ValidationMergeTests(unittest.TestCase):
    def test_exact_1104_aggregation_and_duplicate_rejection(self) -> None:
        interface = FormalTrainingInterface.from_manifest(MANIFEST)
        trajectory_order = list(dict.fromkeys(interface.samples[i]["metadata"]["trajectory_id"] for i in interface.validation_indices))
        # Keep this long-running merge fixture out of Windows' global Temp,
        # which may be cleaned by another process during the three merges.
        with tempfile.TemporaryDirectory(dir=MANIFEST.parents[2] / "audit") as directory:
            root = Path(directory)
            paths = []
            for shard_id in range(4):
                trajectories = set(trajectory_order[shard_id::4])
                sample_ids = sorted(interface.samples[i]["metadata"]["sample_id"] for i in interface.validation_indices
                                    if interface.samples[i]["metadata"]["trajectory_id"] in trajectories)
                row = {
                    "passed": True, "validation_shard_id": shard_id, "validation_shard_count": 4,
                    "dataset_manifest_sha256": interface.dataset_manifest_hash, "checkpoint_sha256": "a" * 64,
                    "sample_ids": sample_ids, "selected_trajectory_ids": sorted(trajectories),
                    "per_horizon_loss": [
                        {"horizon": h, "motion_numerator": 2.0, "motion_count": 2,
                         "csi_numerator": 6.0, "csi_count": 2} for h in range(1, 5)
                    ],
                    "raw_aggregates": {family: {str(h): {"sum_abs": 4.0, "sum_squared": 8.0, "count": 2}
                                                 for h in range(1, 5)} for family in ("motion", "csi")},
                    "batch_size": 8, "wall_seconds": 5.0 + shard_id, "data_loading_seconds": 0.5,
                    "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                    "prior_only": True, "parameter_unchanged": True,
                    "future_posterior_teacher_calls": 0, "future_target_encoder_calls": 0,
                    "posterior_rollout_calls": 0,
                }
                path = root / f"part-{shard_id}.json"
                path.write_text(json.dumps(row), encoding="utf-8")
                paths.append(path)
            result = merge_validation(MANIFEST, paths, root / "merged.json")
            self.assertTrue(result["passed"])
            self.assertEqual((1104, 12), (result["validation_windows"], result["validation_trajectories"]))
            self.assertEqual(2.0, result["L_Val"])
            self.assertEqual(2.0, result["raw_metrics"]["motion"][0]["raw_mae"])
            self.assertEqual(26.0, result["validation_worker_wall_seconds_sum"])
            self.assertAlmostEqual(1104 / 26.0, result["throughput_windows_per_second_serial"])
            broken = json.loads(paths[1].read_text(encoding="utf-8"))
            broken["sample_ids"][0] = json.loads(paths[0].read_text(encoding="utf-8"))["sample_ids"][0]
            paths[1].write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeated"):
                merge_validation(MANIFEST, paths, root / "rejected.json")
            missing = json.loads(paths[1].read_text(encoding="utf-8"))
            missing["sample_ids"] = [sample_id for sample_id in missing["sample_ids"] if sample_id != broken["sample_ids"][0]]
            paths[1].write_text(json.dumps(missing), encoding="utf-8")
            self.assertFalse(merge_validation(MANIFEST, paths, root / "missing.json")["passed"])


if __name__ == "__main__":
    unittest.main()
