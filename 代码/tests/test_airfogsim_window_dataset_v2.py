from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_fixture(root: Path, *, zero_positive: bool = False, train_lifecycle_index: int = 2) -> None:
    contract = {
        "schema_version": "PI-JWM-AirFogSim-tensor-v2",
        "max_nodes": 2,
        "max_physical_edges": 53,
        "max_flows": 53,
        "max_tasks": 5,
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
        physical_edge_state = np.zeros((3, 53, 5), dtype=np.float32)
        physical_edge_present = np.ones((3, 53), dtype=bool)
        physical_edge_present[:, -1] = False
        physical_edge_state[:2, :, 3] = 1.0
        flow_present = np.zeros((3, 53), dtype=bool)
        task_present = np.zeros((3, 5), dtype=bool)
        task_lifecycle_index = np.full((3, 5), -1, dtype=np.int16)
        if seed == 0 and not zero_positive:
            physical_edge_state[2, 0, 3] = 1.0
            physical_edge_state[2, -1, 3] = 1.0
            flow_present[2, 0] = True
            flow_present[2, -1] = True
            task_present[2, :2] = True
            task_lifecycle_index[2, :2] = train_lifecycle_index
        elif seed != 0:
            physical_edge_present[:] = True
            physical_edge_state[:, :, 3] = 1.0
            flow_present[:] = True
            task_present[:] = True
            task_lifecycle_index[:] = 4
        arrays = {
            "time": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            "node_state": np.asarray([[[value] * 7, [0.0] * 7]] * 3, dtype=np.float32),
            "node_present": np.asarray([[True, False]] * 3),
            "physical_edge_state": physical_edge_state,
            "physical_edge_present": physical_edge_present,
            "flow_state": np.zeros((3, 53, 5), dtype=np.float32),
            "flow_present": flow_present,
            "flow_completed": np.zeros((3, 53), dtype=bool),
            "flow_valid": np.asarray([True] * 52 + [False]),
            "task_state": np.zeros((3, 5, 8), dtype=np.float32),
            "task_present": task_present,
            "task_valid": np.asarray([True] * 4 + [False]),
            "task_action": np.zeros((3, 5, 5), dtype=np.float32),
            "task_action_present": np.zeros((3, 5), dtype=bool),
            "physical_edge_endpoint_index": np.full((53, 2), -1, dtype=np.int32),
            "flow_endpoint_index": np.full((53, 2), -1, dtype=np.int32),
            "flow_bearer_mask": np.zeros((3, 53, 53), dtype=bool),
            "flow_bearer_edge_index": np.full((3, 53), -1, dtype=np.int32),
            "task_node_index": np.full((3, 5, 4), -1, dtype=np.int32),
            "task_lifecycle_index": task_lifecycle_index,
            "task_action_node_index": np.full((3, 5, 3), -1, dtype=np.int32),
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

    def test_exposes_raw_masked_link_activity_before_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset, fit_training_stats
            from pi_jwm.airfogsim_tensor_v2 import EDGE_FEATURES

            stats = fit_training_stats(root, split="dev_train")
            sample = AirFogSimTensorWindowDataset(root, split="dev_train", stats=stats, normalize=True)[0]
            activity_index = EDGE_FEATURES.index("active_task_count")

            self.assertEqual(torch.bool, sample["history"]["link_activity"].dtype)
            self.assertEqual(torch.bool, sample["target"]["link_activity"].dtype)
            self.assertTrue(sample["target"]["link_activity"][0, 0])
            self.assertFalse(sample["target"]["link_activity"][0, -1])
            self.assertAlmostEqual(0.0, sample["target"]["physical_edge_state"][0, 0, activity_index].item())

    def test_sparse_label_stats_use_only_label_slices_and_requested_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import fit_sparse_label_stats

            stats = fit_sparse_label_stats(root, split="dev_train", max_pos_weight=50.0)

            self.assertEqual("dev_train", stats["source_split"])
            self.assertEqual(1, stats["sample_count"])
            self.assertEqual(
                {
                    "positive_count": 1,
                    "negative_count": 51,
                    "positive_rate": 1.0 / 52.0,
                    "pos_weight": 50.0,
                },
                stats["labels"]["link_activity"],
            )
            self.assertEqual(1, stats["labels"]["flow_present"]["positive_count"])
            self.assertEqual(51, stats["labels"]["flow_present"]["negative_count"])
            self.assertEqual(50.0, stats["labels"]["flow_present"]["pos_weight"])
            self.assertEqual(2, stats["labels"]["task_present"]["positive_count"])
            self.assertEqual(2, stats["labels"]["task_present"]["negative_count"])
            self.assertEqual([0, 0, 2, 0, 0], stats["task_lifecycle"]["counts"])
            self.assertEqual(2, stats["task_lifecycle"]["majority_index"])
            self.assertEqual(2, stats["task_lifecycle"]["valid_count"])

    def test_sparse_label_stats_use_unit_weight_when_no_positive_labels_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, zero_positive=True)
            from pi_jwm.airfogsim_window_dataset_v2 import fit_sparse_label_stats

            stats = fit_sparse_label_stats(root, split="dev_train")

            self.assertEqual(1, stats["sample_count"])
            for label in ("link_activity", "flow_present", "task_present"):
                self.assertEqual(0, stats["labels"][label]["positive_count"])
                self.assertEqual(0.0, stats["labels"][label]["positive_rate"])
                self.assertEqual(1.0, stats["labels"][label]["pos_weight"])

    def test_lifecycle_statistics_follow_the_tensor_lifecycle_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, train_lifecycle_index=5)
            import pi_jwm.airfogsim_window_dataset_v2 as dataset_module
            from pi_jwm.airfogsim_tensor_v2 import LIFECYCLE_TYPES

            extended_types = (*LIFECYCLE_TYPES, "future")
            with patch.object(dataset_module, "LIFECYCLE_TYPES", extended_types, create=True):
                stats = dataset_module.fit_sparse_label_stats(root, split="dev_train")

            self.assertEqual([0, 0, 0, 0, 0, 2], stats["task_lifecycle"]["counts"])
            self.assertEqual(5, stats["task_lifecycle"]["majority_index"])
            self.assertEqual(2, stats["task_lifecycle"]["valid_count"])

    def test_validation_split_is_empty_when_not_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            from pi_jwm.airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset

            self.assertEqual(1, len(AirFogSimTensorWindowDataset(root, split="dev_validation")))
            self.assertEqual(0, len(AirFogSimTensorWindowDataset(root, split="missing")))


if __name__ == "__main__":
    unittest.main()
