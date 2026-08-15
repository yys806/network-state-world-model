from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_formal_airfogsim_window_v1 import _write_formal_fixture


def _write_system_fixture(root: Path, *, mismatched_time: bool = False) -> None:
    for seed, split in ((0, "train"), (1, "validation")):
        seed_dir = root / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True)
        steps, tasks, nodes = 4, 3, 3
        time = np.arange(steps, dtype=np.float32) / 10
        if mismatched_time and seed == 0:
            time[2] = 0.25
        arrays = {
            "time": time,
            "task_completion_event": np.eye(steps, tasks, dtype=bool),
            "task_on_time_completion_event": np.eye(steps, tasks, dtype=bool),
            "completed_task_delay": np.arange(steps * tasks, dtype=np.float32).reshape(steps, tasks),
            "completed_task_delay_valid": np.eye(steps, tasks, dtype=bool),
            "uav_energy_delta": np.arange(steps * nodes, dtype=np.float32).reshape(steps, nodes),
            "uav_energy_valid": np.ones((steps, nodes), dtype=bool),
            "source_service_delta": np.eye(steps, nodes, dtype=np.float32),
            "source_on_time_service_delta": np.eye(steps, nodes, dtype=np.float32),
            "source_population_valid": np.asarray([True, True, False]),
            "source_evaluable_task_count": np.asarray([2, 1, 0], dtype=np.int32),
            "delivered_data_total": np.asarray([0.0, 1.0, 2.0, 3.0]),
        }
        np.savez_compressed(seed_dir / "system_targets.npz", **arrays)
        (seed_dir / "system_target_report.json").write_text(
            json.dumps({"seed": seed, "split": split, "time_count": steps}),
            encoding="utf-8",
        )


class FormalSystemWindowV1Tests(unittest.TestCase):
    def test_aligns_real_system_history_targets_and_static_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            system_root = root / "system"
            tensor_root.mkdir()
            system_root.mkdir()
            _write_formal_fixture(tensor_root)
            _write_system_fixture(system_root)

            from pi_jwm.formal_airfogsim_window_v1 import FormalWindowConfig
            from pi_jwm.formal_system_window_v1 import FormalSystemWindowDataset

            dataset = FormalSystemWindowDataset(
                tensor_root,
                system_root=system_root,
                split="train",
                config=FormalWindowConfig(history_steps=2, horizon_steps=2),
            )
            sample = dataset[0]

            np.testing.assert_allclose(
                sample["system_history"]["delivered_data_total"].numpy(), [0.0, 1.0]
            )
            np.testing.assert_allclose(
                sample["system_target"]["delivered_data_total"].numpy(), [2.0, 3.0]
            )
            self.assertEqual((2, 3), tuple(sample["system_target"]["uav_energy_delta"].shape))
            np.testing.assert_array_equal(
                sample["system_static"]["source_population_valid"].numpy(),
                [True, True, False],
            )
            np.testing.assert_array_equal(
                sample["system_static"]["source_evaluable_task_count"].numpy(), [2, 1, 0]
            )
            self.assertEqual(1, dataset.loaded_system_seed_count)

    def test_rejects_sidecar_time_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            system_root = root / "system"
            tensor_root.mkdir()
            system_root.mkdir()
            _write_formal_fixture(tensor_root)
            _write_system_fixture(system_root, mismatched_time=True)

            from pi_jwm.formal_airfogsim_window_v1 import FormalWindowConfig
            from pi_jwm.formal_system_window_v1 import FormalSystemWindowDataset

            dataset = FormalSystemWindowDataset(
                tensor_root,
                system_root=system_root,
                split="train",
                config=FormalWindowConfig(history_steps=2, horizon_steps=2),
            )
            with self.assertRaisesRegex(ValueError, "time grid"):
                _ = dataset[0]

    def test_pads_seed_local_task_and_node_axes_to_formal_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            system_root = root / "system"
            tensor_root.mkdir()
            system_root.mkdir()
            _write_formal_fixture(tensor_root)
            _write_system_fixture(system_root)
            path = system_root / "seed_000" / "system_targets.npz"
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            for key in (
                "task_completion_event",
                "task_on_time_completion_event",
                "completed_task_delay",
                "completed_task_delay_valid",
            ):
                arrays[key] = arrays[key][:, :2]
            for key in (
                "uav_energy_delta",
                "uav_energy_valid",
                "source_service_delta",
                "source_on_time_service_delta",
            ):
                arrays[key] = arrays[key][:, :2]
            arrays["source_population_valid"] = arrays["source_population_valid"][:2]
            arrays["source_evaluable_task_count"] = arrays[
                "source_evaluable_task_count"
            ][:2]
            np.savez_compressed(path, **arrays)

            from pi_jwm.formal_airfogsim_window_v1 import FormalWindowConfig
            from pi_jwm.formal_system_window_v1 import FormalSystemWindowDataset

            sample = FormalSystemWindowDataset(
                tensor_root,
                system_root=system_root,
                split="train",
                config=FormalWindowConfig(history_steps=2, horizon_steps=2),
            )[0]

            self.assertEqual((2, 3), tuple(sample["system_target"]["task_completion_event"].shape))
            self.assertEqual((2, 3), tuple(sample["system_target"]["uav_energy_delta"].shape))
            self.assertFalse(sample["system_target"]["uav_energy_valid"][:, 2].any())
            self.assertFalse(sample["system_static"]["source_population_valid"][2])
            self.assertEqual(
                0, int(sample["system_static"]["source_evaluable_task_count"][2])
            )


if __name__ == "__main__":
    unittest.main()
