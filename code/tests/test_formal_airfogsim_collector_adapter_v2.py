from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_collector_adapter_v2 import (  # noqa: E402
    CollectorBundleContractError,
    build_formal_bundles,
)


def _frame(*, trajectory_id: str = "traj-0") -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "frame_index": 0,
        "decision_snapshot": {
            "simulation_time": 0.0,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "execution_snapshot": {
            "simulation_time": 0.1,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "outcome_snapshot": {
            "simulation_time": 0.1,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "action": {
            "frame_index": 0,
            "decisions": [],
            "flows": [],
            "hops": [],
            "rb_allocations": [],
        },
        "lifecycle_rows": [],
        "transfer_rows": [],
        "cpu_rows": [],
        "energy_rows": [],
    }


class FormalCollectorAdapterTests(unittest.TestCase):
    def test_physical_state_stream_uses_one_decision_snapshot_per_time(self):
        source, resource = build_formal_bundles(
            [_frame()],
            task_records=[],
            n_rb=4,
        )

        self.assertEqual("traj-0", source["trajectory_id"])
        self.assertEqual("v0", source["physical_nodes"][0]["id"])
        self.assertEqual(1, len(source["physical_node_snapshots"]))
        self.assertEqual(0, len(source["physical_edge_snapshots"]))
        self.assertEqual(
            {(0.0, "v0")},
            {
                (float(row["observed_time"]), str(row["id"]))
                for row in source["physical_node_snapshots"]
            },
        )
        self.assertIn("task_ledger", resource)
        self.assertIn("rb_ledger", resource)
        self.assertIn("cpu_ledger", resource)
        self.assertIn("uav_energy_ledger", resource)

    def test_missing_outcome_snapshot_blocks_formal_bundle(self):
        frame = _frame()
        frame["outcome_snapshot"] = None

        with self.assertRaises(CollectorBundleContractError):
            build_formal_bundles([frame], task_records=[], n_rb=4)

    def test_task_state_stream_uses_decision_time_and_keeps_outcome_evidence_separate(self):
        frame = _frame()
        frame["decision_snapshot"]["tasks"] = [
            {
                "task_id": "task-0",
                "lifecycle": "waiting_to_offload",
                "current_node_id": "v0",
                "route_nodes": [],
                "return_destination_id": "v0",
                "arrival_time": 0.0,
            }
        ]
        frame["outcome_snapshot"]["tasks"] = [
            {
                "task_id": "task-0",
                "lifecycle": "offloading",
                "current_node_id": "v0",
                "route_nodes": ["v0"],
                "return_destination_id": "v0",
                "arrival_time": 0.0,
            }
        ]

        source, _ = build_formal_bundles(
            [frame],
            task_records=[{"id": "task-0"}],
            n_rb=4,
        )

        self.assertEqual([(0.0, "task-0")], [
            (float(row["observed_time"]), str(row["id"]))
            for row in source["task_snapshots"]
        ])
        self.assertEqual([(0.1, "task-0")], [
            (float(row["observed_time"]), str(row["id"]))
            for row in source["outcome_task_snapshots"]
        ])

    def test_return_actions_use_return_target_id_contract_field(self):
        frame = _frame()
        frame["decision_snapshot"]["tasks"] = [
            {
                "task_id": "task-0",
                "lifecycle": "waiting_to_return",
                "current_node_id": "v0",
                "route_nodes": [],
                "return_destination_id": "v0",
                "arrival_time": 0.0,
            }
        ]
        frame["action"]["decisions"] = [
            {
                "task_id": "task-0",
                "lifecycle": "waiting_to_return",
                "selected": True,
                "target_node_id": "v0",
                "route_nodes": ["v0"],
            }
        ]

        source, _ = build_formal_bundles(
            [frame],
            task_records=[{"id": "task-0"}],
            n_rb=4,
        )

        self.assertEqual(1, len(source["return_actions"]))
        self.assertEqual("v0", source["return_actions"][0]["return_target_id"])
        self.assertNotIn("target_node_id", source["return_actions"][0])

    def test_decision_snapshot_preserves_dynamic_task_and_edge_observations(self):
        frame = _frame()
        frame["decision_snapshot"]["nodes"].append(
            {
                "node_id": "v1",
                "node_type": "vehicle",
                "present": True,
                "position": [4.0, 6.0, 0.0],
            }
        )
        frame["decision_snapshot"]["physical_edges"] = [
            {
                "edge_id": "physical::v0::v1::V2V",
                "source_id": "v0",
                "target_id": "v1",
                "edge_type": "V2V",
                "present": True,
            }
        ]
        frame["decision_snapshot"]["channel_rows"] = [
            {
                "physical_edge_id": "physical::v0::v1::V2V",
                "channel_attenuation_db": [10.0, 12.0],
                "observed_mask": True,
                "capture_phase": "decision",
            }
        ]
        frame["decision_snapshot"]["tasks"] = [
            {
                "task_id": "task-0",
                "lifecycle_state": "waiting_to_offload",
                "current_node_id": "v0",
                "route_nodes": [],
                "return_destination_id": "v0",
                "arrival_time": 0.0,
                "task_size": 100.0,
                "return_size": 20.0,
                "task_cpu": 3.0,
                "deadline": 25.0,
                "priority": 2.0,
                "in_stage_transmitted_size": 7.0,
                "computed_size": 9.0,
                "task_delay": 0.0,
                "source": "v0",
                "host": "v0",
                "exec": "v1",
                "ret": "v0",
            }
        ]

        source, _ = build_formal_bundles(
            [frame],
            task_records=[{"id": "task-0"}],
            n_rb=4,
        )

        task = source["task_snapshots"][0]
        self.assertEqual("waiting_to_offload", task["lifecycle_state"])
        self.assertEqual(100.0, task["task_size"])
        self.assertEqual("v1", task["exec"])
        self.assertEqual(25.0, task["deadline_time"])
        self.assertEqual(1.0, task["task_feature_mask"][3])
        edge = source["physical_edge_snapshots"][0]
        self.assertEqual(11.0, edge["csi_mean"])
        self.assertEqual(1.0, edge["physical_edge_feature_mask"][1])
        self.assertIsNone(edge["rate_sum"])
        self.assertEqual(0.0, edge["physical_edge_feature_mask"][2])

    def test_transfer_rows_are_preserved_as_direct_per_rb_observations(self):
        frame = _frame()
        frame["decision_snapshot"]["nodes"].append(
            {
                "node_id": "r0",
                "node_type": "rsu",
                "present": True,
                "position": [4.0, 6.0, 0.0],
            }
        )
        frame["action"]["flows"] = [
            {
                "flow_id": "flow-0",
                "task_id": "task-0",
                "phase": "offload",
            }
        ]
        frame["action"]["hops"] = [
            {
                "hop_id": "hop-0",
                "flow_id": "flow-0",
                "source_id": "v0",
                "target_id": "r0",
                "physical_edge_id": "physical::v0::r0::V2I",
                "transport": "wireless",
            }
        ]
        frame["transfer_rows"] = [
            {
                "flow_id": "flow-0",
                "hop_id": "hop-0",
                "physical_edge_id": "physical::v0::r0::V2I",
                "rb_index": 1,
                "rate_per_s": 12.5,
                "planned_capacity": 1.25,
                "remaining_before": 5.0,
                "delivered_data": 1.25,
                "time": 0.1,
                "observed_mask": True,
                "capture_phase": "after_fast_fading_before_transfer",
                "temporal_role": "outcome_only_not_same_frame_decision_input",
                "source_method": "AirFogSim_direct_per_rb_runtime_arrays",
            }
        ]

        source, _ = build_formal_bundles(
            [frame],
            task_records=[],
            n_rb=4,
        )

        self.assertEqual(1, len(source["source_rb_observations"]))
        row = source["source_rb_observations"][0]
        self.assertEqual(12.5, row["rate_per_s"])
        self.assertEqual("physical::v0::r0::V2I", row["physical_edge_id"])
        self.assertEqual("outcome", row["source_phase"])

    def test_missing_per_rb_rows_get_outcome_time_and_keep_null_mask(self):
        frame = _frame()
        frame["action"]["flows"] = [{"flow_id": "flow-0", "task_id": "task-0", "phase": "offload"}]
        frame["action"]["hops"] = [{
            "hop_id": "hop-0", "flow_id": "flow-0", "source_id": "v0", "target_id": "v0",
            "physical_edge_id": "physical::v0::v0::V2V", "transport": "wireless",
        }]
        frame["transfer_rows"] = [{
            "flow_id": "flow-0", "hop_id": "hop-0", "physical_edge_id": "physical::v0::v0::V2V",
            "rb_index": 0, "observed_mask": False, "rate_per_s": None,
            "missing_reason": "runtime_channel_row_unavailable",
            "capture_phase": "execution", "temporal_role": "runtime_outcome_missing",
        }]
        source, _ = build_formal_bundles([frame], task_records=[], n_rb=2)
        row = source["source_rb_observations"][0]
        self.assertEqual(0.1, row["time"])
        self.assertFalse(row["observed_mask"])
        self.assertIsNone(row["rate_per_s"])


if __name__ == "__main__":
    unittest.main()
