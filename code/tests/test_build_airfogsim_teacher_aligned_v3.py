from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_airfogsim_teacher_graph_v3 import source_graph


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_airfogsim_teacher_aligned_v3.py"


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "build_airfogsim_teacher_aligned_v3", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(root: Path) -> None:
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "PI-JWM-AirFogSim-formal-dataset-v1",
                "history_steps": 1,
                "horizon_steps": 1,
                "trajectory_count": 3,
                "unlocked_trajectory_count": 2,
                "locked_test_trajectory_count": 1,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        ("train_0", 0, "train"),
        ("validation_0", 1, "validation"),
        ("locked_0", 2, "locked_test"),
    ]
    with (root / "trajectory_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trajectory_id",
                "seed",
                "split",
                "physical_nodes",
                "physical_edges",
                "information_edges",
                "task_nodes",
                "task_dag_edges",
            ],
        )
        writer.writeheader()
        for trajectory_id, seed, split in rows:
            writer.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "seed": seed,
                    "split": split,
                    "physical_nodes": 2,
                    "physical_edges": 1,
                    "information_edges": 1,
                    "task_nodes": 0,
                    "task_dag_edges": 0,
                }
            )
            trajectory_root = root / ("locked_test/trajectories" if split == "locked_test" else "trajectories")
            trajectory = trajectory_root / trajectory_id
            trajectory.mkdir(parents=True)
            (trajectory / "manifest.json").write_text(
                json.dumps(
                    {
                        "trajectory_id": trajectory_id,
                        "seed": seed,
                        "split": split,
                        "files": {
                            "dual_graph_v2_bundle.json": {
                                "sha256": f"hash-{trajectory_id}",
                                "size_bytes": 123,
                            }
                        },
                    }
                ),
                encoding="utf-8",
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
                    "input_end_index": 1,
                    "label_start_index": 1,
                    "label_end_index": 2,
                }
            )


class BuildTeacherAlignedV3DiscoveryTests(unittest.TestCase):
    def test_builder_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists())


class BuildTeacherAlignedV3BehaviorTests(unittest.TestCase):
    def test_builds_only_unlocked_tensors_and_keeps_train_only_stats(self):
        subject = load_subject()
        self.assertTrue(hasattr(subject, "build_teacher_aligned_dataset"))
        loaded: list[str] = []

        def graph_loader(row):
            self.assertNotEqual("locked_test", row["split"])
            loaded.append(row["trajectory_id"])
            graph = source_graph()
            graph["trajectory_id"] = row["trajectory_id"]
            offset = 0.0 if row["split"] == "train" else 100.0
            for node in graph["source_physical_node_snapshots"]:
                node["position"][0] += offset
            return graph

        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            output = Path(output_temp)
            write_fixture(source)

            result = subject.build_teacher_aligned_dataset(
                source_dir=source,
                output_dir=output,
                graph_loader=graph_loader,
            )

            stats = json.loads(
                (output / "normalization_stats.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            contract = json.loads(
                (output / "tensor_contract.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            locked = json.loads(
                (output / "locked_test_integrity.json").read_text(encoding="utf-8")
            )

            self.assertEqual(["train_0", "validation_0"], loaded)
            self.assertTrue(result["teacher_aligned_graph_tensor_ready"])
            self.assertFalse(result["airfogsim_rerun_required"])
            self.assertEqual("train", stats["source_split"])
            self.assertEqual(1, locked["trajectory_count"])
            self.assertFalse((output / "seed_002").exists())
            self.assertNotIn("csi_mean", contract["physical_edge_features"])
            self.assertIn("pre.csi_mean", contract["information_edge_features"])
            self.assertTrue(validation["checks"]["locked_test_labels_not_read"])
            self.assertTrue((output / "manifest.json").exists())
            self.assertIn("seed_000/manifest.json", manifest["files"])

    def test_train_statistics_are_streamed_without_global_concatenation(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as output_temp:
            output = Path(output_temp)
            rows = []
            for seed, value in ((0, 1.0), (1, 3.0)):
                seed_dir = output / f"seed_{seed:03d}"
                seed_dir.mkdir(parents=True)
                arrays = {
                    "physical_node_state": np.full((2, 1, 1), value, np.float32),
                    "physical_node_feature_mask": np.ones((2, 1, 1), bool),
                    "physical_edge_state": np.full((2, 1, 1), value, np.float32),
                    "physical_edge_feature_mask": np.ones((2, 1, 1), bool),
                    "information_node_state": np.full((2, 1, 1), value, np.float32),
                    "information_node_feature_mask": np.ones((2, 1, 1), bool),
                    "information_edge_state": np.full((2, 1, 1), value, np.float32),
                    "information_edge_feature_mask": np.ones((2, 1, 1), bool),
                    "data_flow_state": np.full((2, 1, 1), value, np.float32),
                    "data_flow_present": np.ones((2, 1), bool),
                    "task_state": np.full((2, 1, 1), value, np.float32),
                    "task_present": np.ones((2, 1), bool),
                }
                np.savez_compressed(seed_dir / "trajectory_tensors.npz", **arrays)
                rows.append({"seed": str(seed)})

            with patch.object(
                subject.np,
                "concatenate",
                side_effect=AssertionError("global concatenation is forbidden"),
            ):
                stats = subject._fit_train_only_stats(output, rows)

            physical = stats["features"]["physical_node_state"]
            self.assertEqual([4], physical["count"])
            self.assertEqual([2.0], physical["mean"])
            self.assertEqual([1.0], physical["scale"])

    def test_resume_reuses_valid_seed_outputs_without_loading_graphs(self):
        subject = load_subject()

        def graph_loader(row):
            graph = source_graph()
            graph["trajectory_id"] = row["trajectory_id"]
            return graph

        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source = Path(source_temp)
            output = Path(output_temp)
            write_fixture(source)
            subject.build_teacher_aligned_dataset(
                source_dir=source,
                output_dir=output,
                graph_loader=graph_loader,
            )
            for name in (
                "locked_test_integrity.json",
                "trajectory_index.csv",
                "normalization_stats.json",
                "validation_report.json",
                "dataset_summary.json",
                "manifest.json",
            ):
                (output / name).unlink()

            def forbidden_loader(_row):
                raise AssertionError("completed seed output must not be rebuilt")

            result = subject.build_teacher_aligned_dataset(
                source_dir=source,
                output_dir=output,
                graph_loader=forbidden_loader,
                resume=True,
            )

            self.assertTrue(result["teacher_aligned_graph_tensor_ready"])
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
