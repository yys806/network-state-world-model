from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_fixture(root: Path) -> None:
    contract = {
        "schema_version": "PI-JWM-AirFogSim-tensor-v2",
        "max_nodes": 2,
        "max_physical_edges": 1,
        "max_flows": 1,
        "max_tasks": 1,
        "max_dag_edges": 0,
        "history_steps": 2,
        "horizon_steps": 1,
    }
    (root / "tensor_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    rows = [
        {"sample_id": "seed000::window000000", "seed": 0, "split": "dev_train", "input_start_index": 0, "input_end_index": 2, "label_start_index": 2, "label_end_index": 3},
        {"sample_id": "seed001::window000000", "seed": 1, "split": "dev_validation", "input_start_index": 0, "input_end_index": 2, "label_start_index": 2, "label_end_index": 3},
    ]
    with (root / "window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for seed, value in ((0, 1.0), (1, 100.0)):
        seed_dir = root / f"seed_{seed:03d}"
        seed_dir.mkdir()
        arrays = {
            "time": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            "node_state": np.asarray([[[value] * 7, [0.0] * 7]] * 3, dtype=np.float32),
            "node_present": np.asarray([[True, False]] * 3),
            "physical_edge_state": np.zeros((3, 1, 5), dtype=np.float32),
            "physical_edge_present": np.zeros((3, 1), dtype=bool),
            "flow_state": np.zeros((3, 1, 5), dtype=np.float32),
            "flow_present": np.zeros((3, 1), dtype=bool),
            "flow_completed": np.zeros((3, 1), dtype=bool),
            "task_state": np.zeros((3, 1, 8), dtype=np.float32),
            "task_present": np.zeros((3, 1), dtype=bool),
            "task_action": np.zeros((3, 1, 5), dtype=np.float32),
            "task_action_present": np.zeros((3, 1), dtype=bool),
            "physical_edge_endpoint_index": np.full((1, 2), -1, dtype=np.int32),
            "flow_endpoint_index": np.full((1, 2), -1, dtype=np.int32),
            "flow_bearer_mask": np.zeros((3, 1, 1), dtype=bool),
            "flow_bearer_edge_index": np.full((3, 1), -1, dtype=np.int32),
            "task_node_index": np.full((3, 1, 4), -1, dtype=np.int32),
            "task_lifecycle_index": np.full((3, 1), -1, dtype=np.int16),
            "task_action_node_index": np.full((3, 1, 3), -1, dtype=np.int32),
        }
        np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)


class AirFogSimWindowDatasetV2Tests(unittest.TestCase):
    def test_lazy_seed_loading_and_window_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset

            dataset = AirFogSimTensorWindowDataset(root, split="dev_train")
            self.assertEqual(1, len(dataset))
            self.assertEqual(0, dataset.loaded_seed_count)
            sample = dataset[0]
            self.assertEqual(1, dataset.loaded_seed_count)
            self.assertEqual((2, 2, 7), tuple(sample["history"]["node_state"].shape))
            self.assertEqual((1, 2, 7), tuple(sample["target"]["node_state"].shape))
            self.assertEqual("seed000::window000000", sample["sample_id"])

    def test_padding_and_integer_indices_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset

            sample = AirFogSimTensorWindowDataset(root, split="dev_train")[0]
            self.assertTrue(np.all(sample["history"]["node_state"].numpy()[:, 1] == 0.0))
            self.assertTrue(np.all(sample["history"]["node_present"].numpy()[:, 1] == 0))
            self.assertTrue(np.all(sample["history"]["task_node_index"].numpy() == -1))

    def test_normalization_keeps_padding_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset, fit_training_stats

            stats = fit_training_stats(root, split="dev_train")
            sample = AirFogSimTensorWindowDataset(root, split="dev_train", stats=stats, normalize=True)[0]
            self.assertTrue(np.all(sample["history"]["node_state"].numpy()[:, 1] == 0.0))
            self.assertTrue(np.all(sample["target"]["node_state"].numpy()[:, 1] == 0.0))

    def test_training_stats_use_only_requested_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import fit_training_stats

            stats = fit_training_stats(root, split="dev_train")
            self.assertAlmostEqual(1.0, stats["features"]["node_state"]["mean"][0])
            self.assertNotAlmostEqual(100.0, stats["features"]["node_state"]["mean"][0])

    def test_validation_split_is_empty_when_not_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset

            self.assertEqual(1, len(AirFogSimTensorWindowDataset(root, split="dev_validation")))
            self.assertEqual(0, len(AirFogSimTensorWindowDataset(root, split="missing")))


if __name__ == "__main__":
    unittest.main()
