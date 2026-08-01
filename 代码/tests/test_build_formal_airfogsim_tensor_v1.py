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
    times = [0.1, 0.2, 0.3]
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
            self.assertFalse((output / "seed_002").exists())


if __name__ == "__main__":
    unittest.main()
