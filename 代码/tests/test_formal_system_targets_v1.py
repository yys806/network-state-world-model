from __future__ import annotations

import sys
import tempfile
import unittest
import importlib.util
import json
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load_builder():
    path = SCRIPTS_ROOT / "build_formal_system_targets_v1.py"
    spec = importlib.util.spec_from_file_location("build_formal_system_targets_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal system-target builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FormalSystemTargetsV1Tests(unittest.TestCase):
    def test_accepts_uniform_float32_time_grid_from_formal_tensor_npz(self):
        from pi_jwm.formal_system_targets_v1 import build_system_target_arrays

        time_values = (np.arange(1, 301, dtype=np.float32) / np.float32(10.0)).astype(
            np.float64
        )
        arrays = build_system_target_arrays(
            time_values=time_values,
            node_vocab=["UAV_0"],
            task_vocab=[],
            task_snapshots=[],
            energy_rows=[],
            transfer_events=[],
        )

        self.assertEqual(300, len(arrays["time"]))

    def test_builds_unique_completion_energy_service_and_delivered_targets(self):
        from pi_jwm.formal_system_targets_v1 import build_system_target_arrays

        arrays = build_system_target_arrays(
            time_values=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            node_vocab=["UAV_0", "vehicle_0"],
            task_vocab=["Task_1", "Task_2"],
            task_snapshots=[
                {
                    "id": "Task_1",
                    "observed_time": 0.2,
                    "completion_time": 0.2,
                    "lifecycle_state": "finished",
                    "task_delay": 0.15,
                    "source": "vehicle_0",
                },
                {
                    "id": "Task_1",
                    "observed_time": 0.3,
                    "completion_time": 0.2,
                    "lifecycle_state": "finished",
                    "task_delay": 0.15,
                    "source": "vehicle_0",
                },
                {
                    "id": "Task_2",
                    "observed_time": 0.3,
                    "completion_time": None,
                    "lifecycle_state": "computing",
                    "task_delay": None,
                    "source": "UAV_0",
                },
            ],
            energy_rows=[
                {"time": 0.0, "uav_id": "UAV_0", "energy_before": 10.0, "energy_after": 8.5},
                {"time": 0.1, "uav_id": "UAV_0", "energy_before": 8.5, "energy_after": 8.0},
            ],
            transfer_events=[
                {"time": 0.1, "delivered_data": 2.0},
                {"time": 0.1, "delivered_data": 3.0},
                {"time": 0.3, "delivered_data": 1.5},
            ],
        )

        self.assertEqual((3, 2), arrays["task_completion_event"].shape)
        self.assertEqual(1, int(arrays["task_completion_event"].sum()))
        self.assertTrue(arrays["task_completion_event"][1, 0])
        self.assertTrue(arrays["completed_task_delay_valid"][1, 0])
        self.assertAlmostEqual(0.15, float(arrays["completed_task_delay"][1, 0]))
        np.testing.assert_allclose(arrays["uav_energy_delta"][:, 0], [1.5, 0.5, 0.0])
        np.testing.assert_array_equal(arrays["uav_energy_valid"][:, 0], [True, True, False])
        self.assertEqual(1.0, float(arrays["source_service_delta"][1, 1]))
        np.testing.assert_array_equal(arrays["source_population_valid"], [True, True])
        np.testing.assert_allclose(arrays["delivered_data_total"], [5.0, 0.0, 1.5])

    def test_rejects_unknown_ids_negative_energy_and_off_grid_events(self):
        from pi_jwm.formal_system_targets_v1 import build_system_target_arrays

        common = {
            "time_values": np.asarray([0.1, 0.2], dtype=np.float64),
            "node_vocab": ["UAV_0"],
            "task_vocab": ["Task_1"],
            "task_snapshots": [],
            "energy_rows": [],
            "transfer_events": [],
        }
        with self.assertRaisesRegex(ValueError, "unknown task"):
            build_system_target_arrays(
                **{
                    **common,
                    "task_snapshots": [
                        {
                            "id": "Task_9",
                            "completion_time": 0.1,
                            "lifecycle_state": "finished",
                            "task_delay": 0.1,
                            "source": "UAV_0",
                        }
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "negative energy"):
            build_system_target_arrays(
                **{
                    **common,
                    "energy_rows": [
                        {"time": 0.0, "uav_id": "UAV_0", "energy_before": 1.0, "energy_after": 2.0}
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "time grid"):
            build_system_target_arrays(
                **{
                    **common,
                    "transfer_events": [{"time": 0.15, "delivered_data": 1.0}],
                }
            )

    def test_dataset_builder_writes_only_nonlocked_seed_sidecars_and_manifest(self):
        subject = load_builder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            tensor = root / "tensor"
            output = root / "output"
            trajectory = source / "trajectories" / "train_trajectory"
            trajectory.mkdir(parents=True)
            seed_dir = tensor / "seed_000"
            seed_dir.mkdir(parents=True)
            (seed_dir / "tensor_report.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "split": "train",
                        "trajectory_id": "train_trajectory",
                        "node_vocab": ["UAV_0", "vehicle_0"],
                        "task_vocab": ["Task_1"],
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                seed_dir / "trajectory_tensors.npz",
                time=np.asarray([0.1, 0.2], dtype=np.float32),
            )
            (trajectory / "dual_graph_v2_bundle.json").write_text(
                json.dumps(
                    {
                        "source_task_snapshots": [
                            {
                                "id": "Task_1",
                                "completion_time": 0.2,
                                "lifecycle_state": "finished",
                                "task_delay": 0.1,
                                "source": "vehicle_0",
                            }
                        ],
                        "source_transfer_events": [{"time": 0.2, "delivered_data": 1.0}],
                    }
                ),
                encoding="utf-8",
            )
            (trajectory / "resource_bundle.json").write_text(
                json.dumps(
                    {
                        "uav_energy_ledger": [
                            {
                                "time": 0.0,
                                "uav_id": "UAV_0",
                                "energy_before": 5.0,
                                "energy_after": 4.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subject.build_formal_system_target_dataset(
                source_dir=source,
                tensor_dir=tensor,
                output_dir=output,
            )

            self.assertTrue(result["system_targets_ready"])
            self.assertEqual(1, result["nonlocked_seed_count"])
            self.assertFalse((output / "seed_600").exists())
            with np.load(output / "seed_000" / "system_targets.npz", allow_pickle=False) as arrays:
                self.assertEqual(1, int(arrays["task_completion_event"].sum()))
                self.assertAlmostEqual(1.0, float(arrays["uav_energy_delta"].sum()))
                np.testing.assert_array_equal(
                    arrays["source_population_valid"], [False, True]
                )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("seed_000/system_targets.npz", manifest["files"])
            self.assertNotIn("seed_600/system_targets.npz", manifest["files"])


if __name__ == "__main__":
    unittest.main()
