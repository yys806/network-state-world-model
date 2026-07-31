from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "build_airfogsim_multiseed_v2.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location("build_airfogsim_multiseed_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load multiseed v2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_runtime(seed: int, max_time: float):
    trajectory_id = f"seed_{seed}"
    snapshots = []
    edge_snapshots = []
    for index in range(1, 13):
        time = round(index * 0.1, 1)
        snapshots.extend(
            [
                {"id": "p0", "kind": "vehicle", "cpu": 2.0, "observed_time": time},
                {"id": "p1", "kind": "rsu", "cpu": 10.0, "observed_time": time},
                {"id": "cloud0", "kind": "cloud", "cpu": 100.0, "observed_time": time},
            ]
        )
        edge_snapshots.append(
            {
                "id": "pe::p0::p1",
                "src": "p0",
                "dst": "p1",
                "kind": "V2I",
                "observed_time": time,
                "active_task_count": int(time == 0.5),
            }
        )
    source_bundle = {
        "physical_nodes": snapshots[-3:],
        "physical_edges": [edge_snapshots[-1]],
        "physical_node_snapshots": snapshots,
        "physical_edge_snapshots": edge_snapshots,
        "information_nodes": [
            {
                "id": f"Task_{seed}",
                "source": "p0",
                "host": "p1",
                "exec": "p1",
                "ret": "p0",
                "task_size": 1.0,
                "return_size": 0.0,
                "arrival_time": 0.1,
                "deadline": 1.0,
                "deadline_time": 1.1,
                "priority": 1.0,
                "terminal_status": "completed",
                "task_delay": 0.4,
                "observed_time": 1.2,
            }
        ],
        "information_edges": [],
        "task_snapshots": [
            {
                "id": f"Task_{seed}",
                "terminal_status": "active" if index < 5 else "completed",
                "observed_time": round(index * 0.1, 1),
            }
            for index in range(1, 13)
        ],
        "offload_actions": [
            {
                "task_id": f"Task_{seed}",
                "source_node_id": "p0",
                "target_node_id": "p1",
                "time": 0.1,
            }
        ],
        "return_actions": [],
        "rb_actions": [
            {
                "task_id": f"Task_{seed}",
                "current_node_id": "p0",
                "assigned_to": "p1",
                "rb_indices": "0",
                "time": 0.5,
            }
        ],
        "transfer_events": [
            {
                "event_id": f"event::{seed}",
                "task_id": f"Task_{seed}",
                "phase": "offload",
                "source": "p0",
                "target": "p1",
                "path": ["pe::p0::p1"],
                "remaining_before": 1.0,
                "delivered_data": 1.0,
                "flow_completed": True,
                "time": 0.5,
            }
        ],
    }
    resource_bundle = {
        "task_ledger": [
            {
                "record_id": f"task::{seed}",
                "remaining_before": 1.0,
                "delivered_data": 1.0,
                "remaining_after": 0.0,
            }
        ],
        "dependency_ledger": [],
        "rb_ledger": [
            {"record_id": f"rb::{seed}", "time": 0.5, "n_rb": 4, "rb_indices": [0]}
        ],
        "cpu_ledger": [
            {
                "record_id": f"cpu::{seed}",
                "time": 0.6,
                "node_id": "p1",
                "allocated_cpu": 1.0,
                "node_cpu_capacity": 10.0,
                "dt": 0.1,
            }
        ],
        "uav_energy_ledger": [],
    }
    return {
        "config": {"seed": seed, "max_time": max_time},
        "bundle": resource_bundle,
        "source_bundle": source_bundle,
        "runtime_summary": {"seed": seed, "steps": 12, "max_time": max_time},
    }


class BuildAirFogSimMultiseedV2Tests(unittest.TestCase):
    def test_builds_seed_isolated_development_dataset(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = subject.build_multiseed_dataset(
                output_dir=output_dir,
                seeds=[0, 1],
                max_time=1.2,
                history_steps=3,
                horizon_steps=2,
                runtime_runner=fake_runtime,
                resource_validator=lambda bundle: {
                    "conservation_ready": True,
                    "failed_gates": [],
                    "gates": {},
                },
                required_physical_directions={"V2I"},
            )

            summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
            validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
            windows = (output_dir / "window_index.csv").read_text(encoding="utf-8-sig").splitlines()

            self.assertTrue(result["development_dataset_ready"])
            self.assertFalse(result["formal_training_ready"])
            self.assertTrue(validation["development_dataset_ready"])
            self.assertEqual(16, summary["window_count"])
            self.assertEqual(17, len(windows))
            self.assertEqual({"dev_train", "dev_validation"}, set(summary["split_by_seed"].values()))
            self.assertEqual(12, summary["seed_summaries"][0]["task_snapshot_count"])
            self.assertEqual(1, summary["seed_summaries"][0]["offload_action_count"])
            self.assertAlmostEqual(1.0, summary["key_metric_summary"]["task_completion_rate"]["mean"])
            self.assertTrue((output_dir / "seed_000" / "dual_graph_v2_bundle.json").is_file())
            self.assertTrue((output_dir / "seed_001" / "metric_results.json").is_file())
            graph = json.loads(
                (output_dir / "seed_000" / "dual_graph_v2_bundle.json").read_text(encoding="utf-8")
            )
            self.assertFalse(any(row["kind"] == "cloud" for row in graph["physical_nodes"]))
            self.assertFalse(
                any(row["kind"] == "cloud" for row in graph["source_physical_node_snapshots"])
            )
            self.assertEqual(12, len(graph["source_task_snapshots"]))
            self.assertEqual(1, len(graph["source_offload_actions"]))
            self.assertEqual(1, len(graph["source_rb_actions"]))
            self.assertTrue(validation["checks"]["time_aligned_task_snapshots"])
            self.assertTrue(validation["checks"]["action_ledgers_frozen"])
            self.assertTrue(validation["checks"]["required_physical_directions_covered"])
            self.assertIn("variable_graph_tensorization_not_frozen", summary["formal_training_blockers"])


if __name__ == "__main__":
    unittest.main()
