from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_formal_airfogsim_dataset_v1.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "build_formal_airfogsim_dataset_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load formal dataset builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_runtime(spec, max_time):
    times = [round(index * 0.1, 1) for index in range(1, 13)]
    task_id = f"Task_{spec.seed}"
    parent_id = f"Parent_{spec.seed}"
    trajectory_id = spec.trajectory_id
    nodes = [
        {"id": "vehicle_0", "kind": "vehicle", "trajectory_id": trajectory_id},
        {"id": "RSU_0", "kind": "rsu", "trajectory_id": trajectory_id},
    ]
    edge = {
        "id": "pe::vehicle_0::RSU_0",
        "src": "vehicle_0",
        "dst": "RSU_0",
        "kind": "V2I",
        "trajectory_id": trajectory_id,
    }
    task_records = [
        {
            "id": parent_id,
            "source": "vehicle_0",
            "host": "RSU_0",
            "exec": "RSU_0",
            "ret": "vehicle_0",
            "task_size": 1.0,
            "return_size": 0.1,
            "task_cpu": 2.0,
            "arrival_time": 0.1,
            "deadline": 2.0,
            "deadline_time": 2.1,
            "priority": 1.0,
            "terminal_status": "completed",
            "completion_time": 0.5,
            "task_delay": 0.4,
        },
        {
            "id": task_id,
            "source": "vehicle_0",
            "host": "RSU_0",
            "exec": "RSU_0",
            "ret": "vehicle_0",
            "task_size": 1.0,
            "return_size": 0.1,
            "task_cpu": 2.0,
            "arrival_time": 0.2,
            "deadline": 2.0,
            "deadline_time": 2.2,
            "priority": 1.0,
            "terminal_status": "completed",
            "completion_time": 0.8,
            "task_delay": 0.6,
        },
    ]
    task_snapshots = [
        {
            **task,
            "observed_time": time,
            "lifecycle_state": "finished" if time >= task["completion_time"] else "computing",
            "computed_size": task["task_cpu"] if time >= task["completion_time"] else 0.0,
        }
        for time in times
        for task in task_records
    ]
    cpu_row = {
        "record_id": f"cpu::{task_id}::0.600000",
        "kind": "compute",
        "time": 0.6,
        "node_id": "RSU_0",
        "task_id": task_id,
        "allocated_cpu": 10.0,
        "node_cpu_capacity": 10.0,
        "allocated_fraction": 1.0,
        "dt": 0.1,
        "task_cpu": 2.0,
        "computed_before": 0.0,
        "computed_after": 1.0,
        "remaining_before": 2.0,
        "delivered_data": 1.0,
        "remaining_after": 1.0,
        "policy_id": spec.cpu_policy,
        "policy_weight": 1.0,
        "deadline_remaining": 1.6,
        "queue_size": 1,
    }
    source = {
        "physical_nodes": nodes,
        "physical_edges": [edge],
        "physical_node_snapshots": [
            {
                **node,
                "position": [0.0, 0.0, 0.0],
                "speed": 0.0,
                "acceleration": 0.0,
                "cpu": 10.0,
                "storage": 1.0,
                "observed_time": time,
            }
            for time in times
            for node in nodes
        ],
        "physical_edge_snapshots": [
            {
                **edge,
                "distance": 1.0,
                "csi_mean": 1.0,
                "rate_sum": 0.0,
                "active_task_count": 0,
                "allocated_rb_count": 0,
                "observed_time": time,
            }
            for time in times
        ],
        "information_nodes": task_records,
        "information_edges": [
            {
                "id": f"dag::{parent_id}::{task_id}",
                "src": parent_id,
                "dst": task_id,
                "time": 0.1,
                "data_mb": None,
                "semantic": "precedence_only",
            }
        ],
        "task_snapshots": task_snapshots,
        "offload_actions": [],
        "return_actions": [],
        "rb_actions": [],
        "transfer_events": [],
        "dependency_flows": [],
        "ep_relations": [],
    }
    return {
        "config": {
            "seed": spec.seed,
            "max_time": max_time,
            "pi_jwm_formal_v1": spec.to_dict(),
        },
        "bundle": {
            "task_ledger": [cpu_row],
            "dependency_ledger": [],
            "rb_ledger": [],
            "cpu_ledger": [cpu_row],
            "uav_energy_ledger": [],
        },
        "source_bundle": source,
        "runtime_summary": {"seed": spec.seed, "steps": len(times)},
    }


def fake_resource_validator(bundle):
    return {"conservation_ready": True, "failed_gates": [], "gates": {}}


class BuildFormalAirFogSimDatasetTests(unittest.TestCase):
    def test_builder_writes_60_trajectory_protocol_and_separate_locked_windows(self):
        subject = load_subject()
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = subject.build_formal_dataset(
                output_dir=output,
                specs=build_formal_trajectory_specs(),
                runtime_runner=fake_runtime,
                resource_validator=fake_resource_validator,
                max_time=30.0,
                history_steps=8,
                horizon_steps=3,
                required_physical_directions={"V2I"},
            )

            summary = json.loads(
                (output / "dataset_summary.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

            self.assertTrue(result["formal_dataset_ready"])
            self.assertFalse(result["formal_training_ready"])
            self.assertEqual(60, result["trajectory_count"])
            self.assertEqual(54, summary["unlocked_trajectory_count"])
            self.assertEqual(6, summary["locked_test_trajectory_count"])
            self.assertEqual(108, summary["window_count"])
            self.assertEqual(12, summary["locked_test_window_count"])
            self.assertTrue(validation["checks"]["cpu_policy_trace_valid"])
            self.assertTrue(validation["checks"]["dag_precedence_only"])
            self.assertNotIn("locked_test", summary["metric_splits"])
            self.assertTrue(
                (output / "locked_test" / "window_index.csv").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "locked_test"
                    / "trajectories"
                    / "load_low__density_sparse__r09"
                    / "manifest.json"
                ).is_file()
            )
            self.assertTrue(manifest["files"]["dataset_summary.json"]["sha256"])
            self.assertTrue(manifest["generation_completed"])
            self.assertTrue(manifest["field_masks_valid"])
            self.assertTrue(manifest["splits_frozen"])
            self.assertTrue(manifest["source_manifest_present"])

    def test_completed_trajectories_are_verified_and_reused(self):
        subject = load_subject()
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        specs = build_formal_trajectory_specs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            subject.build_formal_dataset(
                output_dir=output,
                specs=specs,
                runtime_runner=fake_runtime,
                resource_validator=fake_resource_validator,
                max_time=30.0,
                history_steps=8,
                horizon_steps=3,
                required_physical_directions={"V2I"},
            )

            def fail_if_called(spec, max_time):
                raise AssertionError(f"trajectory unexpectedly reran: {spec.trajectory_id}")

            result = subject.build_formal_dataset(
                output_dir=output,
                specs=specs,
                runtime_runner=fail_if_called,
                resource_validator=fake_resource_validator,
                max_time=30.0,
                history_steps=8,
                horizon_steps=3,
                required_physical_directions={"V2I"},
            )

            self.assertEqual(60, result["reused_trajectory_count"])
            self.assertEqual(0, result["generated_trajectory_count"])
            self.assertTrue(result["formal_dataset_ready"])


if __name__ == "__main__":
    unittest.main()
