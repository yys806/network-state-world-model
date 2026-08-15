from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def fake_graph(task_ids=("Task_2", "Task_10")):
    times = [0.1, 0.2, 0.3]
    node_snapshots = []
    edge_snapshots = []
    task_snapshots = []
    for time in times:
        node_snapshots.extend(
            [
                {
                    "id": "vehicle_0",
                    "kind": "vehicle",
                    "position": [float(time), 0.0, 0.0],
                    "speed": 1.0,
                    "acceleration": 0.0,
                    "cpu": 2.0,
                    "storage": 1.0,
                    "observed_time": time,
                },
                {
                    "id": "RSU_0",
                    "kind": "rsu",
                    "position": [1.0, 0.0, 0.0],
                    "speed": 0.0,
                    "acceleration": 0.0,
                    "cpu": 10.0,
                    "storage": 2.0,
                    "observed_time": time,
                },
            ]
        )
        edge_snapshots.append(
            {
                "id": "pe::vehicle_0::RSU_0",
                "src": "vehicle_0",
                "dst": "RSU_0",
                "kind": "V2I",
                "distance": 1.0,
                "csi_mean": 2.0,
                "rate_sum": 3.0 if time == 0.2 else 0.0,
                "active_task_count": int(time == 0.2),
                "allocated_rb_count": int(time == 0.2),
                "observed_time": time,
            }
        )
        for task_index, task_id in enumerate(task_ids):
            arrival = 0.2 if task_index == 0 else 0.1
            task_snapshots.append(
                {
                    "id": task_id,
                    "source": "vehicle_0",
                    "host": "RSU_0",
                    "exec": "RSU_0",
                    "ret": "vehicle_0",
                    "task_size": 1.0 + task_index,
                    "return_size": 0.2,
                    "task_cpu": 0.5,
                    "deadline": 2.0,
                    "deadline_time": arrival + 2.0,
                    "priority": 1.0,
                    "in_stage_transmitted_size": 0.0,
                    "computed_size": 0.0,
                    "task_delay": None,
                    "arrival_time": arrival,
                    "lifecycle_state": "to_generate" if time < arrival else "to_offload",
                    "observed_time": time,
                }
            )
    task_nodes = [
        {
            "id": task_id,
            "source": "vehicle_0",
            "host": "RSU_0",
            "exec": "RSU_0",
            "ret": "vehicle_0",
            "task_size": 1.0 + index,
            "return_size": 0.2,
            "task_cpu": 0.5,
            "deadline": 2.0,
            "deadline_time": 2.2,
            "priority": 1.0,
            "arrival_time": 0.2 if index == 0 else 0.1,
            "terminal_status": "pending",
        }
        for index, task_id in enumerate(task_ids)
    ]
    flow = {
        "id": "flow::Task_2::task_input::vehicle_0::RSU_0",
        "src": "agent::vehicle_0",
        "dst": "agent::RSU_0",
        "flow_type": "task_input",
        "task_id": "Task_2",
        "total_data": 1.0,
        "remaining_data": 0.6,
        "delivered_data": 0.4,
        "status": "active",
        "first_time": 0.2,
        "last_time": 0.2,
    }
    return {
        "physical_nodes": [node_snapshots[0], node_snapshots[1]],
        "physical_edges": [edge_snapshots[0]],
        "information_nodes": [
            {"id": "agent::vehicle_0", "physical_node_id": "vehicle_0"},
            {"id": "agent::RSU_0", "physical_node_id": "RSU_0"},
        ],
        "information_edges": [flow],
        "task_nodes": task_nodes,
        "task_dag_edges": [],
        "agent_attachments": [],
        "flow_bearers": [],
        "source_physical_node_snapshots": node_snapshots,
        "source_physical_edge_snapshots": edge_snapshots,
        "source_task_snapshots": task_snapshots,
        "source_transfer_events": [
            {
                "event_id": "event::Task_2::offload::0::0.2",
                "task_id": "Task_2",
                "phase": "offload",
                "source": "vehicle_0",
                "target": "RSU_0",
                "path": ["pe::vehicle_0::RSU_0"],
                "delivered_data": 0.4,
                "remaining_before": 1.0,
                "flow_completed": False,
                "time": 0.2,
            }
        ],
        "source_offload_actions": [
            {
                "task_id": "Task_2",
                "source_node_id": "vehicle_0",
                "target_node_id": "RSU_0",
                "time": 0.2,
            }
        ],
        "source_return_actions": [],
        "source_rb_actions": [
            {
                "task_id": "Task_2",
                "current_node_id": "vehicle_0",
                "assigned_to": "RSU_0",
                "rb_count": 2,
                "rb_indices": "0 1",
                "time": 0.2,
            }
        ],
    }


class AirFogSimTensorV2Tests(unittest.TestCase):
    def test_natural_task_order_and_padding_contract(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph

        graph = fake_graph()
        contract = dataclasses.replace(infer_tensor_contract([graph]), max_tasks=3)
        arrays, report = tensorize_seed_graph(graph, contract)

        self.assertEqual(["Task_2", "Task_10"], report["task_vocab"])
        self.assertTrue(np.all(arrays["task_state"][:, 2:] == 0.0))
        self.assertTrue(np.all(arrays["task_node_index"][:, 2:] == -1))
        self.assertEqual([True, True, False], arrays["task_valid"].tolist())
        self.assertEqual((3, 2, 7), arrays["node_state"].shape)

    def test_hides_to_generate_task_before_arrival(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph

        graph = fake_graph(task_ids=("Task_2",))
        arrays, _ = tensorize_seed_graph(graph, infer_tensor_contract([graph]))

        self.assertEqual(0, arrays["task_present"][0, 0])
        self.assertTrue(np.all(arrays["task_state"][0, 0] == 0.0))
        self.assertEqual(1, arrays["task_present"][1, 0])

    def test_reconstructs_flow_from_action_and_direct_events(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph

        graph = fake_graph(task_ids=("Task_2",))
        arrays, report = tensorize_seed_graph(graph, infer_tensor_contract([graph]))

        self.assertEqual([0, 1, 1], arrays["flow_present"][:, 0].tolist())
        np.testing.assert_allclose([0.0, 0.4, 0.0], arrays["flow_delivered_this_slot"][:, 0])
        self.assertTrue(np.all(arrays["flow_state"][0, 0] == 0.0))
        self.assertEqual(0, report["flow_creation_fallback_count"])

    def test_completion_slot_remains_observable_when_action_log_is_late(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph

        graph = fake_graph()
        graph["source_offload_actions"][0]["time"] = 0.3
        graph["source_transfer_events"][0]["flow_completed"] = True
        arrays, _ = tensorize_seed_graph(graph, infer_tensor_contract([graph]))
        self.assertEqual([False, True, False], arrays["flow_present"][:, 0].tolist())
        self.assertEqual([False, True, True], arrays["flow_completed"][:, 0].tolist())
        self.assertTrue(np.all(arrays["flow_state"][2, 0] == 0.0))

    def test_invalid_reference_is_rejected(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract, tensorize_seed_graph

        graph = fake_graph(task_ids=("Task_2",))
        graph["source_offload_actions"][0]["target_node_id"] = "missing"
        with self.assertRaises(ValueError):
            tensorize_seed_graph(graph, infer_tensor_contract([graph]))


if __name__ == "__main__":
    unittest.main()
