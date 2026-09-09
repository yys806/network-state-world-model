from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_formal_airfogsim_tensor_v1.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract
from pi_jwm.formal_airfogsim_graph_v1 import tensorize_formal_graph


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "build_formal_airfogsim_tensor_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load formal tensor builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_graph(seed: int) -> dict:
    value = 1.0 if seed == 0 else 100.0
    times = [0.1, 0.2, 0.3, 0.4, 0.5]
    nodes = [
        {"id": "vehicle_0", "kind": "vehicle"},
        {"id": "RSU_0", "kind": "rsu"},
    ]
    edge = {
        "id": "pe::vehicle_0::RSU_0",
        "src": "vehicle_0",
        "dst": "RSU_0",
        "kind": "V2I",
    }
    return {
        "schema_version": "PI-JWM-AirFogSim-dual-graph-v2",
        "physical_nodes": nodes,
        "physical_edges": [edge],
        "information_nodes": [],
        "information_edges": [],
        "agent_attachments": [],
        "flow_bearers": [],
        "task_nodes": [],
        "task_dag_edges": [],
        "source_physical_node_snapshots": [
            {
                "id": node["id"],
                "kind": node["kind"],
                "position": [value, 0.0, 0.0],
                "observed_time": time,
            }
            for time in times
            for node in nodes
        ],
        "source_physical_edge_snapshots": [
            {**edge, "distance": 10.0, "observed_time": time} for time in times
        ],
        "source_task_snapshots": [],
        "source_offload_actions": [],
        "source_return_actions": [],
        "source_rb_actions": [],
        "source_cpu_actions": [],
        "source_transfer_events": [],
    }


def write_fixture(root: Path) -> None:
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "PI-JWM-AirFogSim-formal-dataset-v1",
                "history_steps": 2,
                "horizon_steps": 1,
                "trajectory_count": 3,
                "unlocked_trajectory_count": 2,
                "locked_test_trajectory_count": 1,
            }
        ),
        encoding="utf-8",
    )
    trajectories = [
        ("train_0", 0, "train"),
        ("validation_0", 1, "validation"),
        ("locked_0", 2, "locked_test"),
    ]
    with (root / "trajectory_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["trajectory_id", "seed", "split"]
        )
        writer.writeheader()
        for trajectory_id, seed, split in trajectories:
            writer.writerow(
                {"trajectory_id": trajectory_id, "seed": seed, "split": split}
            )
    with (root / "window_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "seed",
                "split",
                "input_start_index",
                "input_end_index",
                "label_start_index",
                "label_end_index",
            ],
        )
        writer.writeheader()
        for seed, split in ((0, "train"), (1, "validation")):
            writer.writerow(
                {
                    "sample_id": f"seed{seed:03d}::window000000",
                    "seed": seed,
                    "split": split,
                    "input_start_index": 0,
                    "input_end_index": 2,
                    "label_start_index": 2,
                    "label_end_index": 3,
                }
            )


class BuildFormalAirFogSimTensorTests(unittest.TestCase):
    def test_tensorizes_only_unlocked_trajectories_with_train_only_stats(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            output = Path(output_temp)
            write_fixture(source)

            result = subject.build_formal_tensor_dataset(
                source_dir=source,
                output_dir=output,
                graph_loader=lambda row: fake_graph(int(row["seed"])),
            )

            stats = json.loads(
                (output / "normalization_stats.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            with np.load(
                output / "seed_000" / "trajectory_tensors.npz", allow_pickle=False
            ) as arrays:
                action_width = arrays["task_action"].shape[-1]
                self.assertIn("task_dag_state", arrays.files)

            self.assertTrue(result["formal_tensor_ready"])
            self.assertFalse(result["formal_training_ready"])
            self.assertEqual("train", stats["source_split"])
            self.assertEqual(8, action_width)
            self.assertTrue(validation["checks"]["locked_test_not_tensorized"])

    def test_unlocked_only_refresh_does_not_require_or_read_locked_test_directory(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            write_fixture(source)
            locked_dir = source / "locked_test"
            locked_dir.mkdir()
            (locked_dir / "seed_999").mkdir()
            output = Path(output_temp)
            result = subject.build_formal_tensor_dataset(
                source_dir=source,
                output_dir=output,
                include_locked_test=False,
                graph_loader=lambda row: fake_graph(int(row["seed"])),
            )
            self.assertTrue(result["formal_tensor_ready"])
            self.assertEqual(2, result["unlocked_trajectory_count"])
            self.assertEqual(0, result["locked_test_trajectory_count"])
            self.assertFalse((output / "locked_test").exists())
            self.assertFalse((output / "seed_002").exists())

    def test_explicit_horizon_rebuilds_window_index_from_verified_trajectory(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            output = Path(output_temp)
            write_fixture(source)

            result = subject.build_formal_tensor_dataset(
                source_dir=source,
                output_dir=output,
                include_locked_test=False,
                horizon_steps=2,
                graph_loader=lambda row: fake_graph(int(row["seed"])),
            )

            contract = json.loads((output / "tensor_contract.json").read_text(encoding="utf-8"))
            with (output / "window_index.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(result["formal_tensor_ready"])
            self.assertEqual(2, contract["horizon_steps"])
            self.assertEqual(4, len(rows))
            self.assertTrue(all(int(row["label_end_index"]) - int(row["label_start_index"]) == 2 for row in rows))

    def test_reuse_existing_seed_tensors_skips_graph_loader(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            output = Path(output_temp)
            write_fixture(source)
            subject.build_formal_tensor_dataset(
                source_dir=source,
                output_dir=output,
                include_locked_test=False,
                graph_loader=lambda row: fake_graph(int(row["seed"])),
            )

            def unexpected_graph_loader(row):
                raise AssertionError("reuse mode must not load source graphs")

            result = subject.build_formal_tensor_dataset(
                source_dir=source,
                output_dir=output,
                include_locked_test=False,
                graph_loader=unexpected_graph_loader,
                reuse_existing=True,
            )
            self.assertTrue(result["formal_tensor_ready"])
            self.assertEqual(2, result["unlocked_trajectory_count"])
            self.assertEqual(6, result["window_count"])

    def test_formal_graph_normalizes_legacy_return_action_field(self):
        graph = fake_graph(0)
        graph["task_nodes"] = [{"id": "task-0", "task_node_id": "vehicle_0"}]
        graph["source_task_snapshots"] = [
            {
                "id": "task-0",
                "observed_time": time,
                "arrival_time": 0.0,
                "lifecycle_state": "waiting_to_return",
                "current_node_id": "vehicle_0",
                "return_destination_id": "RSU_0",
            }
            for time in (0.1, 0.2, 0.3)
        ]
        graph["source_return_actions"] = [
            {
                "task_id": "task-0",
                "source_node_id": "vehicle_0",
                "target_node_id": "RSU_0",
                "time": 0.1,
            }
        ]
        contract = infer_tensor_contract([graph], history_steps=1, horizon_steps=1)
        arrays, report = tensorize_formal_graph(graph, contract)
        self.assertEqual(1, report["legacy_return_action_field_count"])
        self.assertTrue(arrays["task_action_present"].any())

    def test_tensorizes_explicit_action_sources_flow_task_mapping_and_slot(self):
        graph = fake_graph(0)
        graph["task_nodes"] = [{"id": "task-0", "source": "vehicle_0"}]
        graph["source_task_snapshots"] = [
            {
                "id": "task-0",
                "observed_time": time,
                "arrival_time": 0.0,
                "lifecycle_state": "waiting_to_offload",
                "source": "vehicle_0",
                "current_node_id": "vehicle_0",
                "task_size": 2.0,
                "return_size": 1.0,
                "task_cpu": 3.0,
            }
            for time in (0.1, 0.2, 0.3)
        ]
        graph["source_offload_actions"] = [{
            "task_id": "task-0",
            "source_node_id": "vehicle_0",
            "target_node_id": "RSU_0",
            "time": 0.1,
        }]
        graph["source_rb_actions"] = [{
            "task_id": "task-0",
            "current_node_id": "vehicle_0",
            "assigned_to": "RSU_0",
            "rb_count": 1,
            "time": 0.1,
        }]
        graph["source_cpu_actions"] = [{
            "task_id": "task-0",
            "node_id": "RSU_0",
            "allocated_cpu": 1.0,
            "node_cpu_capacity": 2.0,
            "dt": 0.1,
            "time": 0.1,
        }]
        graph["information_edges"] = [{
            "id": "flow::task-0::task_input::vehicle_0::RSU_0",
            "src": "agent::vehicle_0",
            "dst": "agent::RSU_0",
            "flow_type": "task_input",
            "task_id": "task-0",
            "total_data": 2.0,
            "first_time": 0.1,
        }]
        contract = infer_tensor_contract([graph], history_steps=1, horizon_steps=1)
        arrays, report = tensorize_formal_graph(graph, contract)
        vehicle_index = report["node_vocab"].index("vehicle_0")
        rsu_index = report["node_vocab"].index("RSU_0")
        self.assertEqual(vehicle_index, arrays["task_action_source_node_index"][0, 0, 0])
        self.assertEqual(vehicle_index, arrays["task_action_source_node_index"][0, 0, 2])
        self.assertEqual(rsu_index, arrays["task_action_source_node_index"][0, 0, 3])
        self.assertEqual(0, arrays["flow_task_index"][0])
        self.assertAlmostEqual(0.1, float(arrays["slot_seconds"]), places=6)

    def test_formal_tensorization_restores_runtime_rb_targets_when_direct_rows_absent(self):
        graph = fake_graph(0)
        graph["task_nodes"] = [{"id": "Task_2"}]
        graph["source_rb_actions"] = [{
            "task_id": "Task_2",
            "current_node_id": "vehicle_0",
            "assigned_to": "RSU_0",
            "time": 0.2,
            "rb_indices": [7],
            "n_rb": 50,
        }]
        graph["source_transfer_events"] = [{
            "task_id": "Task_2",
            "source": "vehicle_0",
            "target": "RSU_0",
            "time": 0.2,
            "rb_indices": [7],
            "path": ["pe::vehicle_0::RSU_0"],
            "planned_capacity": 1.25,
        }]

        contract = infer_tensor_contract([graph], history_steps=1, horizon_steps=1)
        arrays, report = tensorize_formal_graph(graph, contract)

        self.assertEqual(50, contract.n_rb)
        self.assertEqual((5, 1, 50), arrays["link_rate_by_rb"].shape)
        self.assertTrue(arrays["link_rate_by_rb_mask"][1, 0, 7])
        self.assertAlmostEqual(12.5, float(arrays["link_rate_by_rb"][1, 0, 7]))
        self.assertEqual(
            "derived_from_action_and_runtime_event",
            report["per_rb_observation_source"]["source"],
        )

    def test_rejects_offload_action_without_explicit_source(self):
        graph = fake_graph(0)
        graph["task_nodes"] = [{"id": "task-0", "source": "vehicle_0"}]
        graph["source_task_snapshots"] = [{
            "id": "task-0",
            "observed_time": time,
            "lifecycle_state": "waiting_to_offload",
            "source": "vehicle_0",
        } for time in (0.1, 0.2, 0.3)]
        graph["source_offload_actions"] = [{
            "task_id": "task-0", "target_node_id": "RSU_0", "time": 0.1,
        }]
        contract = infer_tensor_contract([graph], history_steps=1, horizon_steps=1)
        with self.assertRaisesRegex(ValueError, "source_node_id"):
            tensorize_formal_graph(graph, contract)


if __name__ == "__main__":
    unittest.main()
