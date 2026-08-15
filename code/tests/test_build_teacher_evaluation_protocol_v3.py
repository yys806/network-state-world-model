from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _arrays() -> dict[str, np.ndarray]:
    time_count = 3
    arrays = {
        "time": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        "physical_node_state": np.zeros((time_count, 1, 9), dtype=np.float32),
        "physical_node_feature_mask": np.ones((time_count, 1, 9), dtype=bool),
        "physical_node_present": np.ones((time_count, 1), dtype=bool),
        "physical_edge_state": np.zeros((time_count, 1, 7), dtype=np.float32),
        "physical_edge_feature_mask": np.ones((time_count, 1, 7), dtype=bool),
        "physical_edge_present": np.ones((time_count, 1), dtype=bool),
        "information_node_state": np.zeros((time_count, 1, 7), dtype=np.float32),
        "information_node_feature_mask": np.ones((time_count, 1, 7), dtype=bool),
        "information_node_present": np.ones((time_count, 1), dtype=bool),
        "information_edge_state": np.zeros((time_count, 1, 18), dtype=np.float32),
        "information_edge_feature_mask": np.ones((time_count, 1, 18), dtype=bool),
        "information_edge_present": np.ones((time_count, 1), dtype=bool),
        "data_flow_state": np.zeros((time_count, 1, 5), dtype=np.float32),
        "data_flow_present": np.asarray([[0], [1], [1]], dtype=bool),
        "data_flow_valid": np.asarray([1], dtype=bool),
        "task_state": np.zeros((time_count, 1, 8), dtype=np.float32),
        "task_present": np.asarray([[0], [1], [1]], dtype=bool),
        "task_valid": np.asarray([1], dtype=bool),
        "task_lifecycle_index": np.asarray([[-1], [1], [2]], dtype=np.int16),
        "task_dag_state": np.zeros((time_count, 1, 3), dtype=np.float32),
        "task_dag_state_present": np.ones((time_count, 1), dtype=bool),
    }
    arrays["physical_node_state"][:, 0, 0] = [1.0, 2.0, 4.0]
    arrays["information_edge_state"][:, 0, 11] = [0.0, 1.0, 0.0]
    arrays["information_edge_state"][:, 0, 12] = [0.0, 10.0, 0.0]
    return arrays


def _write_fixture(root: Path) -> tuple[Path, Path]:
    from pi_jwm.evaluation_protocol_v3 import build_factual_metric_mapping
    from pi_jwm.airfogsim_teacher_tensor_v3 import (
        INFORMATION_EDGE_FEATURES,
        INFORMATION_NODE_FEATURES,
        PHYSICAL_EDGE_FEATURES,
        PHYSICAL_NODE_FEATURES,
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_manifest(directory: Path, paths: list[Path]) -> None:
        payload = {
            "files": {
                path.relative_to(directory).as_posix(): {
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in paths
            }
        }
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = root / "dataset"
    dataset.mkdir()
    rows = [
        {
            "seed": "1",
            "split": "train",
            "trajectory_id": "scenario__r01",
            "v3_status": "materialized",
            "v3_seed_dir": "seed_001",
        },
        {
            "seed": "2",
            "split": "validation",
            "trajectory_id": "scenario__r02",
            "v3_status": "materialized",
            "v3_seed_dir": "seed_002",
        },
        {
            "seed": "3",
            "split": "calibration",
            "trajectory_id": "scenario__r03",
            "v3_status": "materialized",
            "v3_seed_dir": "seed_003",
        },
        {
            "seed": "9",
            "split": "locked_test",
            "trajectory_id": "scenario__r09",
            "v3_status": "locked_integrity_only",
            "v3_seed_dir": "",
        },
    ]
    with (dataset / "trajectory_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:3]:
        seed_dir = dataset / row["v3_seed_dir"]
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "trajectory_tensors.npz", **_arrays())
    locked = {
        "trajectory_count": 1,
        "label_content_read": False,
        "tensorized": False,
        "trajectories": [
            {
                "seed": 9,
                "split": "locked_test",
                "trajectory_id": "scenario__r09",
                "label_content_read": False,
                "tensorized": False,
                "source_bundle_sha256": "a" * 64,
            }
        ],
    }
    (dataset / "locked_test_integrity.json").write_text(
        json.dumps(locked), encoding="utf-8"
    )
    (dataset / "dataset_summary.json").write_text(
        json.dumps(
            {"unlocked_trajectory_count": 3, "locked_test_trajectory_count": 1}
        ),
        encoding="utf-8",
    )
    normalization = {
        "source_split": "train",
        "features": {
            "physical_node_state": {"mean": [0.0] * 9, "scale": [1.0] * 9},
            "physical_edge_state": {"mean": [0.0] * 7, "scale": [1.0] * 7},
            "information_node_state": {"mean": [0.0] * 7, "scale": [1.0] * 7},
            "information_edge_state": {"mean": [0.0] * 18, "scale": [1.0] * 18},
            "data_flow_state": {"mean": [0.0] * 5, "scale": [1.0] * 5},
            "task_state": {"mean": [0.0] * 8, "scale": [1.0] * 8},
        },
    }
    (dataset / "normalization_stats.json").write_text(
        json.dumps(normalization), encoding="utf-8"
    )
    (dataset / "tensor_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "PIJWM-DG-Contract-v3-tensor",
                "physical_node_features": list(PHYSICAL_NODE_FEATURES),
                "physical_edge_features": list(PHYSICAL_EDGE_FEATURES),
                "information_node_features": list(INFORMATION_NODE_FEATURES),
                "information_edge_features": list(INFORMATION_EDGE_FEATURES),
            }
        ),
        encoding="utf-8",
    )
    (dataset / "protocol.json").write_text(
        json.dumps(
            {
                "schema_version": "PIJWM-DG-Contract-v3",
                "framework": "PI-JWM",
                "simulator_role": "AirFogSim is a reusable simulator/data source only",
                "physical_edge_rule": "complete_directed_spatial_relation",
                "missing_value_rule": (
                    "numeric zero plus false feature mask; zero alone is never evidence"
                ),
                "deprecated": ["wireless channel fields on physical edges"],
            }
        ),
        encoding="utf-8",
    )
    dataset_inputs = [
        dataset / "trajectory_index.csv",
        dataset / "dataset_summary.json",
        dataset / "normalization_stats.json",
        dataset / "locked_test_integrity.json",
        dataset / "tensor_contract.json",
        dataset / "protocol.json",
    ] + [dataset / row["v3_seed_dir"] / "trajectory_tensors.npz" for row in rows[:3]]
    write_manifest(dataset, dataset_inputs)

    factual_dir = root / "formal_source"
    factual_dir.mkdir()
    metric_csv = factual_dir / "metrics_by_trajectory.csv"
    with metric_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "denominator", "name", "numerator", "reason", "sample_count",
                "seed", "source", "split", "status", "unit", "value",
            ],
        )
        writer.writeheader()
        for seed, split in ((1, "train"), (2, "validation"), (3, "calibration")):
            for mapping in build_factual_metric_mapping():
                writer.writerow(
                    {
                        "denominator": 10,
                        "name": mapping["source_metric_name"],
                        "numerator": 7,
                        "reason": "",
                        "sample_count": 10,
                        "seed": seed,
                        "source": "AirFogSim fixture sidecar",
                        "split": split,
                        "status": "available",
                        "unit": "ratio",
                        "value": 0.9 if mapping["source_metric_name"] == "physical_link_active_ratio" else 0.7,
                    }
                )
    factual_validation = factual_dir / "validation_report.json"
    factual_validation.write_text(
        json.dumps({"checks": {"locked_test_excluded_from_metrics": True}}),
        encoding="utf-8",
    )
    write_manifest(factual_dir, [metric_csv, factual_validation])
    return dataset, metric_csv


class BuildTeacherEvaluationProtocolV3Tests(unittest.TestCase):
    def test_builds_auditable_bundle_without_opening_locked_labels(self):
        from pi_jwm.evaluation_bundle_v3 import build_evaluation_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, metric_csv = _write_fixture(root)
            output = root / "output"
            result = build_evaluation_bundle(dataset, metric_csv, output)

            self.assertTrue(result["evaluation_protocol_ready"])
            validation = json.loads((output / "validation_report.json").read_text("utf-8"))
            self.assertEqual(3, validation["evaluated_trajectory_count"])
            self.assertEqual(1, validation["locked_trajectory_count"])
            self.assertTrue(validation["checks"]["locked_labels_remained_sealed"])
            baseline = json.loads((output / "baseline_metrics.json").read_text("utf-8"))
            self.assertEqual(6, len(baseline["trajectory_reports"]))
            self.assertEqual(
                {"zero_state", "last_persistence"},
                {row["method"] for row in baseline["trajectory_reports"]},
            )
            for report in baseline["trajectory_reports"]:
                for metric in report["metrics"].values():
                    if metric["status"] == "computed":
                        self.assertTrue(math.isfinite(float(metric["value"])))
                    else:
                        self.assertEqual("not_computable", metric["status"])
            availability = json.loads(
                (output / "factual_system_metric_availability.json").read_text("utf-8")
            )
            self.assertEqual(22, availability["metric_count"])
            self.assertEqual(3, availability["trajectory_count"])
            canonical = json.loads((output / "evaluation_rows.json").read_text("utf-8"))
            template = json.loads((output / "report_template.json").read_text("utf-8"))
            self.assertTrue(
                all(
                    set(template["required_result_columns"]) <= set(row)
                    for row in canonical["rows"]
                )
            )
            active_rows = [
                row
                for row in canonical["rows"]
                if row["source_role"] == "factual_system_outcome"
                and row["metric_id"] == "system.information_link_active_ratio"
            ]
            self.assertEqual(3, len(active_rows))
            self.assertTrue(all(abs(row["value"] - 1.0 / 3.0) < 1e-12 for row in active_rows))
            self.assertTrue(
                all("physical" not in " ".join(row["source_fields"]).lower() for row in active_rows)
            )
            action_regret = next(
                row
                for row in canonical["rows"]
                if row["source_role"] == "factual_system_outcome"
                and row["metric_id"] == "decision.action_regret"
            )
            self.assertEqual("utility", action_regret["unit"])
            provenance = json.loads((output / "input_provenance.json").read_text("utf-8"))
            self.assertTrue(
                all(row["verified"] for row in provenance["verified_inputs"].values())
            )
            self.assertIn("formal_airfogsim_graph_v1.py", provenance["code_files"])
            evaluation_stats = json.loads(
                (output / "evaluation_normalization_stats.json").read_text("utf-8")
            )
            self.assertEqual(
                3,
                len(evaluation_stats["features"]["task_dag_state"]["mean"]),
                "R1 freezes task_dag_state as parent_count, unfinished_parent_count, release_ready",
            )
            checkpoint_scales = json.loads(
                (output / "checkpoint_selection_scales.json").read_text("utf-8")
            )
            self.assertEqual(
                evaluation_stats["features"]["task_dag_state"]["scale"][1],
                checkpoint_scales["scales"]["state.dag.unfinished_parent_count.mae"],
            )
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertNotIn("manifest.json", manifest["files"])
            self.assertTrue(manifest["evaluation_protocol_ready"])

    def test_refuses_nonempty_output_directory(self):
        from pi_jwm.evaluation_bundle_v3 import build_evaluation_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, metric_csv = _write_fixture(root)
            output = root / "output"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                build_evaluation_bundle(dataset, metric_csv, output)

    def test_rejects_a_tensor_changed_after_the_source_manifest_was_frozen(self):
        from pi_jwm.evaluation_bundle_v3 import build_evaluation_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, metric_csv = _write_fixture(root)
            with (dataset / "seed_001" / "trajectory_tensors.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_evaluation_bundle(dataset, metric_csv, root / "output")

    def test_rejects_a_reordered_tensor_contract_even_when_its_hash_is_updated(self):
        from pi_jwm.evaluation_bundle_v3 import build_evaluation_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, metric_csv = _write_fixture(root)
            contract_path = dataset / "tensor_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["physical_node_features"][0:2] = reversed(
                contract["physical_node_features"][0:2]
            )
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["tensor_contract.json"] = {
                "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                "size_bytes": contract_path.stat().st_size,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                build_evaluation_bundle(dataset, metric_csv, root / "output")


if __name__ == "__main__":
    unittest.main()
