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


def _write_formal_fixture(root: Path) -> None:
    contract = {
        "schema_version": "PI-JWM-AirFogSim-formal-tensor-v1",
        "history_steps": 2,
        "horizon_steps": 2,
        "max_nodes": 3,
        "max_physical_edges": 2,
        "max_flows": 2,
        "max_tasks": 3,
        "max_dag_edges": 2,
        "action_features": [
            "offload",
            "rb",
            "return",
            "rb_count",
            "rb_fraction",
            "cpu",
            "cpu_allocated",
            "cpu_fraction",
        ],
    }
    (root / "tensor_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    rows = [
        {
            "sample_id": "seed000::window000000",
            "seed": 0,
            "split": "train",
            "input_start_index": 0,
            "input_end_index": 2,
            "label_start_index": 2,
            "label_end_index": 4,
            "decision_time": 0.2,
            "label_start_time": 0.3,
        },
        {
            "sample_id": "seed001::window000000",
            "seed": 1,
            "split": "validation",
            "input_start_index": 0,
            "input_end_index": 2,
            "label_start_index": 2,
            "label_end_index": 4,
            "decision_time": 0.2,
            "label_start_time": 0.3,
        },
    ]
    with (root / "window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for seed in (0, 1):
        seed_dir = root / f"seed_{seed:03d}"
        seed_dir.mkdir()
        steps = 4
        task_action = np.zeros((steps, 3, 8), dtype=np.float32)
        for step in range(steps):
            task_action[step, 0] = float(step + seed * 10)
        arrays = {
            "time": np.arange(steps, dtype=np.float32) / 10,
            "node_state": np.ones((steps, 3, 7), dtype=np.float32) * seed,
            "node_present": np.asarray([[True, True, False]] * steps),
            "node_kind_index": np.asarray([0, 2, -1], dtype=np.int16),
            "physical_edge_state": np.ones((steps, 2, 5), dtype=np.float32),
            "physical_edge_present": np.asarray([[True, False]] * steps),
            "physical_edge_endpoint_index": np.asarray([[0, 1], [-1, -1]], dtype=np.int32),
            "physical_edge_kind_index": np.asarray([0, -1], dtype=np.int16),
            "flow_state": np.ones((steps, 2, 5), dtype=np.float32),
            "flow_present": np.asarray([[True, False]] * steps),
            "flow_completed": np.zeros((steps, 2), dtype=bool),
            "flow_valid": np.asarray([True, False]),
            "flow_endpoint_index": np.asarray([[0, 1], [-1, -1]], dtype=np.int32),
            "flow_type_index": np.asarray([0, -1], dtype=np.int16),
            "flow_bearer_mask": np.asarray([[[True, False], [False, False]]] * steps),
            "flow_bearer_edge_index": np.asarray([[0, -1]] * steps, dtype=np.int32),
            "task_state": np.ones((steps, 3, 8), dtype=np.float32),
            "task_present": np.asarray([[True, True, False]] * steps),
            "task_valid": np.asarray([True, True, False]),
            "task_lifecycle_index": np.asarray([[0, 1, -1]] * steps, dtype=np.int16),
            "task_node_index": np.asarray([[[0, 1, -1, -1], [1, 0, -1, -1], [-1] * 4]] * steps, dtype=np.int32),
            "task_action": task_action,
            "task_action_present": np.asarray([[True, False, False]] * steps),
            "task_action_node_index": np.asarray([[[0, 1, -1, -1], [-1] * 4, [-1] * 4]] * steps, dtype=np.int32),
            "dag_edge_index": np.asarray([[0, -1], [1, -1]], dtype=np.int32),
            "dag_edge_valid": np.asarray([True, False]),
            "agent_node_index": np.asarray([0, 1, -1], dtype=np.int32),
            "dag_edge_present": np.asarray([[True, False]] * steps),
            "task_dag_state": np.asarray([[[0, 0, 1], [1, 1, 0], [0, 0, 0]]] * steps, dtype=np.float32),
            "task_dag_state_present": np.asarray([[True, True, False]] * steps),
        }
        np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)


class FormalAirFogSimWindowV1Tests(unittest.TestCase):
    def test_exposes_history_future_actions_targets_and_graph_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_formal_fixture(root)
            from pi_jwm.formal_airfogsim_window_v1 import (
                FormalAirFogSimWindowDataset,
                FormalWindowConfig,
            )

            sample = FormalAirFogSimWindowDataset(
                root,
                split="train",
                config=FormalWindowConfig(history_steps=2, horizon_steps=2),
            )[0]

            self.assertEqual(2, sample["history"]["node_state"].shape[0])
            self.assertEqual(2, sample["target"]["node_state"].shape[0])
            self.assertEqual(
                {"task_action", "task_action_present", "task_action_node_index"},
                set(sample["future_action"]),
            )
            np.testing.assert_allclose(
                sample["future_action"]["task_action"][:, 0, 0].numpy(),
                np.asarray([2.0, 3.0], dtype=np.float32),
            )
            self.assertIn("task_dag_state", sample["history"])
            self.assertIn("task_dag_state_present", sample["history"])
            self.assertIn("dag_edge_present", sample["history"])
            self.assertIn("task_dag_state", sample["target"])
            self.assertIn("dag_edge_present", sample["target"])
            self.assertEqual((2, 2), tuple(sample["static"]["dag_edge_index"].shape))
            self.assertEqual((3,), tuple(sample["static"]["agent_node_index"].shape))

    def test_rejects_locked_test_before_explicit_unlock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_formal_fixture(root)
            from pi_jwm.formal_airfogsim_window_v1 import (
                FormalAirFogSimWindowDataset,
                FormalWindowConfig,
            )

            with self.assertRaises(PermissionError):
                FormalAirFogSimWindowDataset(root, split="locked_test")
            unlocked = FormalAirFogSimWindowDataset(
                root,
                split="locked_test",
                config=FormalWindowConfig(history_steps=2, horizon_steps=2, allow_locked_test=True),
            )
            self.assertEqual(0, len(unlocked))

    def test_contract_window_lengths_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_formal_fixture(root)
            from pi_jwm.formal_airfogsim_window_v1 import (
                FormalAirFogSimWindowDataset,
                FormalWindowConfig,
            )

            with self.assertRaisesRegex(ValueError, "history_steps"):
                FormalAirFogSimWindowDataset(
                    root,
                    split="train",
                    config=FormalWindowConfig(history_steps=8, horizon_steps=2),
                )

    def test_stratified_window_ids_are_deterministic_and_seed_balanced(self):
        from pi_jwm.formal_airfogsim_window_v1 import select_stratified_window_ids

        rows = [
            {"sample_id": f"seed{seed:03d}::window{window:06d}", "seed": seed, "split": "train"}
            for seed in range(3)
            for window in range(5)
        ]
        first = select_stratified_window_ids(rows, limit=6, seed=17)
        second = select_stratified_window_ids(rows, limit=6, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(6, len(first))
        self.assertEqual({"seed000", "seed001", "seed002"}, {item.split("::")[0] for item in first})
        self.assertTrue(all(item.startswith("seed") for item in first))


if __name__ == "__main__":
    unittest.main()
