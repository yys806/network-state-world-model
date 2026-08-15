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
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_airfogsim_tensor_v2.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location("build_airfogsim_tensor_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load tensor v2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_graph(seed: int):
    value = 1.0 if seed == 0 else 100.0
    times = [0.1, 0.2, 0.3]
    nodes = [
        {"id": "vehicle_0", "kind": "vehicle"},
        {"id": "RSU_0", "kind": "rsu"},
    ]
    edge = {"id": "pe::vehicle_0::RSU_0", "src": "vehicle_0", "dst": "RSU_0", "kind": "V2I"}
    return {
        "schema_version": "PI-JWM-AirFogSim-dual-graph-v2",
        "physical_nodes": nodes,
        "physical_edges": [edge],
        "information_nodes": [],
        "information_edges": [],
        "agent_attachments": [
            {"information_node_id": "agent::vehicle_0", "physical_node_id": "vehicle_0"},
            {"information_node_id": "agent::RSU_0", "physical_node_id": "RSU_0"},
        ],
        "flow_bearers": [],
        "task_nodes": [],
        "task_dag_edges": [],
        "source_physical_node_snapshots": [
            {"id": node["id"], "kind": node["kind"], "position": [value, 0.0, 0.0], "observed_time": time}
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
        "source_transfer_events": [],
    }


def write_source_fixture(root: Path) -> None:
    summary = {
        "schema_version": "PI-JWM-AirFogSim-multiseed-dataset-v2",
        "seeds": [0, 1],
        "split_by_seed": {"0": "dev_train", "1": "dev_validation"},
        "history_steps": 2,
        "horizon_steps": 1,
    }
    (root / "dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {"sample_id": f"seed{seed:03d}::window000000", "seed": seed, "split": split, "input_start_index": 0, "input_end_index": 2, "label_start_index": 2, "label_end_index": 3}
        for seed, split in ((0, "dev_train"), (1, "dev_validation"))
    ]
    with (root / "window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BuildAirFogSimTensorV2Tests(unittest.TestCase):
    def test_builds_frozen_tensor_dataset_and_train_only_stats(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as source_temporary, tempfile.TemporaryDirectory() as output_temporary:
            source = Path(source_temporary)
            output = Path(output_temporary)
            write_source_fixture(source)

            result = subject.build_tensor_dataset(
                source_dir=source,
                output_dir=output,
                graph_loader=lambda seed: fake_graph(seed),
            )

            contract = json.loads((output / "tensor_contract.json").read_text(encoding="utf-8"))
            validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
            stats = json.loads((output / "normalization_stats.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            with np.load(output / "seed_000" / "trajectory_tensors.npz", allow_pickle=False) as arrays:
                node_shape = arrays["node_state"].shape

            self.assertTrue(result["development_tensor_ready"])
            self.assertTrue(validation["development_tensor_ready"])
            self.assertEqual(2, contract["max_nodes"])
            self.assertEqual(1, contract["max_physical_edges"])
            self.assertEqual((3, 2, 7), node_shape)
            self.assertEqual("dev_train", stats["source_split"])
            self.assertAlmostEqual(1.0, stats["features"]["node_state"]["mean"][0])
            self.assertTrue(manifest["files"]["seed_000/trajectory_tensors.npz"]["sha256"])
            self.assertTrue((output / "seed_001" / "tensor_report.json").is_file())
            self.assertTrue((output / "window_index.csv").is_file())


if __name__ == "__main__":
    unittest.main()
