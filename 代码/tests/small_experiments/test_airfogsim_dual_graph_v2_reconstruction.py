from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "small_experiments" / "airfogsim_dual_graph_v2_reconstruction.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location("airfogsim_dual_graph_v2_reconstruction", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load v2 reconstruction script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exp03_fixture():
    return {
        "physical_nodes": [
            {"id": "UAV_0", "kind": "uav"},
            {"id": "RSU_0", "kind": "rsu"},
        ],
        "physical_edges": [
            {"id": "pe::UAV_0::RSU_0", "src": "UAV_0", "dst": "RSU_0", "kind": "U2I"}
        ],
        "physical_node_snapshots": [
            {"id": "UAV_0", "kind": "uav", "observed_time": 1.0},
            {"id": "RSU_0", "kind": "rsu", "observed_time": 1.0},
            {"id": "UAV_0", "kind": "uav", "observed_time": 2.0},
            {"id": "RSU_0", "kind": "rsu", "observed_time": 2.0},
        ],
        "physical_edge_snapshots": [
            {
                "id": "pe::UAV_0::RSU_0",
                "src": "UAV_0",
                "dst": "RSU_0",
                "kind": "U2I",
                "observed_time": 1.0,
            },
            {
                "id": "pe::UAV_0::RSU_0",
                "src": "UAV_0",
                "dst": "RSU_0",
                "kind": "U2I",
                "observed_time": 2.0,
            },
        ],
        "information_nodes": [
            {
                "id": "Task_1",
                "source": "UAV_0",
                "host": "UAV_0",
                "exec": "RSU_0",
                "ret": "UAV_0",
                "task_size": 0.4,
                "return_size": 0.1,
                "arrival_time": 0.5,
                "deadline": 5.0,
                "lifecycle_state": "offloading",
            },
            {
                "id": "Task_2",
                "source": "UAV_0",
                "host": "UAV_0",
                "exec": "RSU_0",
                "ret": "UAV_0",
                "task_size": 0.3,
                "return_size": 0.1,
                "arrival_time": 0.6,
                "deadline": 5.0,
                "lifecycle_state": "to_generate",
            },
        ],
        "information_edges": [
            {
                "id": "ie::Task_1::Task_2",
                "src": "Task_1",
                "dst": "Task_2",
                "data_mb": None,
                "semantic": "precedence_only",
            }
        ],
        "transfer_events": [
            {
                "event_id": "event::Task_1::offload::0::1.000000",
                "task_id": "Task_1",
                "phase": "offload",
                "source": "UAV_0",
                "target": "RSU_0",
                "path": ["pe::UAV_0::RSU_0"],
                "rb_indices": [0, 1],
                "remaining_before": 0.4,
                "delivered_data": 0.4,
                "flow_completed": True,
                "time": 1.0,
                "evidence": "direct_runtime_channel_event",
            }
        ],
        "dependency_flows": [{"dependency_flow_id": "legacy-flow-must-not-be-used"}],
    }


def exp04_fixture():
    return {
        "task_ledger": [{"record_id": "transfer-1", "kind": "communication"}],
        "dependency_ledger": [{"record_id": "legacy-dependency"}],
        "rb_ledger": [{"record_id": "rb-1", "n_rb": 4, "rb_indices": [0, 1]}],
        "cpu_ledger": [{"record_id": "cpu-1", "allocated_cpu": 2.0}],
        "uav_energy_ledger": [
            {"record_id": "energy-1", "uav_id": "UAV_0", "energy_before": 10.0, "energy_after": 9.5}
        ],
    }


class ReconstructionTests(unittest.TestCase):
    def test_frozen_old_evidence_is_relabelled_into_new_semantics(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exp03_path = root / "exp03.json"
            exp04_path = root / "exp04.json"
            output_dir = root / "output"
            exp03_path.write_text(json.dumps(exp03_fixture()), encoding="utf-8")
            exp04_path.write_text(json.dumps(exp04_fixture()), encoding="utf-8")

            result = subject.reconstruct_from_artifacts(
                exp03_bundle_path=exp03_path,
                exp04_bundle_path=exp04_path,
                output_dir=output_dir,
                trajectory_id="fixture-seed0",
                sample_task_id="Task_1",
                sample_time=1.0,
            )

            bundle = json.loads((output_dir / "dual_graph_v2_bundle.json").read_text(encoding="utf-8"))
            metric_inputs = json.loads((output_dir / "metric_input_manifest.json").read_text(encoding="utf-8"))
            sample = json.loads((output_dir / "single_slot_sample.json").read_text(encoding="utf-8"))
            metrics_report = json.loads((output_dir / "metric_results.json").read_text(encoding="utf-8"))

            self.assertTrue(result["dual_graph_v2_ready"])
            self.assertEqual({"agent::UAV_0", "agent::RSU_0"}, {row["id"] for row in bundle["information_nodes"]})
            self.assertEqual({"task_input"}, {row["flow_type"] for row in bundle["information_edges"]})
            self.assertEqual({"ie::Task_1::Task_2"}, {row["id"] for row in bundle["task_dag_edges"]})
            self.assertNotIn("legacy-flow-must-not-be-used", json.dumps(bundle))
            self.assertFalse(metric_inputs["legacy_dependency_ledger_used"])
            self.assertEqual("Task_1", sample["task"]["id"])
            self.assertEqual(1.0, sample["event"]["time"])
            self.assertEqual("available", sample["status"])
            self.assertTrue(sample["strict_same_time"])
            self.assertTrue(all(row["observed_time"] == 1.0 for row in sample["physical_nodes"]))
            self.assertTrue(all(row["observed_time"] == 1.0 for row in sample["physical_edges"]))
            self.assertEqual("service_loss", metrics_report["optimization_objectives"]["primary"]["name"])
            self.assertIn("action_regret", {row["name"] for row in metrics_report["metrics"]})

    def test_sample_without_exact_physical_snapshots_is_not_reported_as_strict(self):
        subject = load_subject()
        fixture = exp03_fixture()
        fixture.pop("physical_node_snapshots")
        fixture.pop("physical_edge_snapshots")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exp03_path = root / "exp03.json"
            exp04_path = root / "exp04.json"
            output_dir = root / "output"
            exp03_path.write_text(json.dumps(fixture), encoding="utf-8")
            exp04_path.write_text(json.dumps(exp04_fixture()), encoding="utf-8")

            subject.reconstruct_from_artifacts(
                exp03_path,
                exp04_path,
                output_dir,
                sample_task_id="Task_1",
                sample_time=1.0,
            )

            sample = json.loads((output_dir / "single_slot_sample.json").read_text(encoding="utf-8"))
            self.assertEqual("missing_exact_physical_snapshot", sample["status"])
            self.assertFalse(sample["strict_same_time"])

    def test_writer_produces_a_compact_auditable_file_set(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exp03_path = root / "exp03.json"
            exp04_path = root / "exp04.json"
            output_dir = root / "output"
            exp03_path.write_text(json.dumps(exp03_fixture()), encoding="utf-8")
            exp04_path.write_text(json.dumps(exp04_fixture()), encoding="utf-8")

            subject.reconstruct_from_artifacts(exp03_path, exp04_path, output_dir)

            self.assertEqual(
                {
                    "REPORT.md",
                    "agent_attachments.csv",
                    "dual_graph_v2_bundle.json",
                    "flow_bearers.csv",
                    "information_edges.csv",
                    "information_nodes.csv",
                    "manifest.json",
                    "metric_input_bundle.json",
                    "metric_input_manifest.json",
                    "metric_results.csv",
                    "metric_results.json",
                    "single_slot_sample.json",
                    "task_dag_edges.csv",
                    "task_nodes.csv",
                    "validation_report.json",
                },
                {path.name for path in output_dir.iterdir()},
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(14, len(manifest["files"]))
            self.assertIn("exp03_bundle_sha256", manifest["sources"])
            self.assertIn("exp04_bundle_sha256", manifest["sources"])


if __name__ == "__main__":
    unittest.main()
