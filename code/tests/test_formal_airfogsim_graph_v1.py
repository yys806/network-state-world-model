from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def formal_graph():
    times = (0.1, 0.2, 0.3)
    nodes = [
        {"id": "vehicle_0", "kind": "vehicle"},
        {"id": "RSU_0", "kind": "rsu"},
    ]
    edges = [
        {
            "id": "pe::vehicle_0::RSU_0",
            "src": "vehicle_0",
            "dst": "RSU_0",
            "kind": "V2I",
        }
    ]
    task_nodes = [
        {
            "id": task_id,
            "source": "vehicle_0",
            "host": "RSU_0",
            "exec": "RSU_0",
            "ret": "vehicle_0",
            "task_size": 1.0,
            "return_size": 0.1,
            "task_cpu": 4.0,
            "deadline": 2.0,
            "deadline_time": 2.1,
            "priority": 1.0,
            "arrival_time": 0.1,
            "terminal_status": "completed" if task_id == "Task_1" else "pending",
        }
        for task_id in ("Task_1", "Task_2")
    ]
    task_snapshots = []
    for time in times:
        task_snapshots.extend(
            [
                {
                    **task_nodes[0],
                    "lifecycle_state": "to_offload" if time == 0.1 else "finished",
                    "computed_size": 4.0 if time >= 0.2 else 0.0,
                    "observed_time": time,
                },
                {
                    **task_nodes[1],
                    "lifecycle_state": "to_generate" if time == 0.1 else "computing",
                    "computed_size": 0.0,
                    "observed_time": time,
                },
            ]
        )
    return {
        "physical_nodes": nodes,
        "physical_edges": edges,
        "information_nodes": [],
        "information_edges": [],
        "task_nodes": task_nodes,
        "task_dag_edges": [
            {
                "id": "dag::Task_1::Task_2",
                "src": "Task_1",
                "dst": "Task_2",
                "time": 0.1,
                "data_mb": None,
                "semantic": "precedence_only",
            }
        ],
        "agent_attachments": [],
        "flow_bearers": [],
        "source_physical_node_snapshots": [
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
        "source_physical_edge_snapshots": [
            {
                **edges[0],
                "distance": 1.0,
                "csi_mean": 1.0,
                "rate_sum": 0.0,
                "active_task_count": 0,
                "allocated_rb_count": 0,
                "observed_time": time,
            }
            for time in times
        ],
        "source_task_snapshots": task_snapshots,
        "source_transfer_events": [],
        "source_offload_actions": [],
        "source_return_actions": [],
        "source_rb_actions": [],
        "source_cpu_actions": [
            {
                "time": 0.2,
                "task_id": "Task_2",
                "node_id": "RSU_0",
                "allocated_cpu": 3.0,
                "node_cpu_capacity": 10.0,
                "allocated_fraction": 0.3,
                "policy_id": "deadline_aware",
            }
        ],
    }


class FormalAirFogSimGraphTests(unittest.TestCase):
    def test_cpu_action_time_tolerates_float32_grid_rounding(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract
        from pi_jwm.formal_airfogsim_graph_v1 import (
            FORMAL_ACTION_FEATURES,
            tensorize_formal_graph,
        )

        graph = formal_graph()
        replacements = {0.1: 16.1, 0.2: 16.2, 0.3: 16.3}
        for key in (
            "source_physical_node_snapshots",
            "source_physical_edge_snapshots",
            "source_task_snapshots",
        ):
            for row in graph[key]:
                row["observed_time"] = replacements[row["observed_time"]]
        graph["source_cpu_actions"][0]["time"] = 16.2
        graph["task_dag_edges"][0]["time"] = 16.1

        arrays, report = tensorize_formal_graph(
            graph, infer_tensor_contract([graph])
        )
        task_index = report["task_vocab"].index("Task_2")

        self.assertEqual(
            1.0,
            arrays["task_action"][
                1, task_index, FORMAL_ACTION_FEATURES.index("cpu")
            ],
        )

    def test_tensor_contains_cpu_action_and_time_observed_dag_state(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract
        from pi_jwm.formal_airfogsim_graph_v1 import (
            FORMAL_ACTION_FEATURES,
            FORMAL_DAG_STATE_FEATURES,
            tensorize_formal_graph,
        )

        graph = formal_graph()
        arrays, report = tensorize_formal_graph(
            graph,
            infer_tensor_contract([graph], history_steps=2, horizon_steps=1),
        )
        task_index = report["task_vocab"].index("Task_2")

        self.assertEqual(8, arrays["task_action"].shape[-1])
        self.assertEqual(
            1.0,
            arrays["task_action"][1, task_index, FORMAL_ACTION_FEATURES.index("cpu")],
        )
        self.assertEqual(
            3.0,
            arrays["task_action"][
                1, task_index, FORMAL_ACTION_FEATURES.index("cpu_allocated")
            ],
        )
        self.assertEqual(0.3, arrays["task_action"][1, task_index, -1])
        self.assertEqual(3, arrays["task_dag_state"].shape[-1])
        self.assertEqual(
            1.0,
            arrays["task_dag_state"][
                0, task_index, FORMAL_DAG_STATE_FEATURES.index("parent_count")
            ],
        )
        self.assertEqual(
            1.0,
            arrays["task_dag_state"][
                0, task_index, FORMAL_DAG_STATE_FEATURES.index("unfinished_parent_count")
            ],
        )
        self.assertEqual(
            0.0,
            arrays["task_dag_state"][
                0, task_index, FORMAL_DAG_STATE_FEATURES.index("release_ready")
            ],
        )
        self.assertEqual(1.0, arrays["task_dag_state"][1, task_index, -1])
        self.assertTrue(np.all(arrays["dag_edge_present"][:, 0]))
        self.assertEqual("PI-JWM-AirFogSim-formal-tensor-v1", report["schema_version"])

    def test_dag_edge_is_hidden_before_airfogsim_observation_time(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract
        from pi_jwm.formal_airfogsim_graph_v1 import tensorize_formal_graph

        graph = formal_graph()
        graph["task_dag_edges"][0]["time"] = 0.2
        arrays, _ = tensorize_formal_graph(graph, infer_tensor_contract([graph]))

        self.assertEqual([False, True, True], arrays["dag_edge_present"][:, 0].tolist())

    def test_formal_graph_rejects_explicit_dag_payload(self):
        from pi_jwm.formal_airfogsim_graph_v1 import validate_formal_graph_boundary

        graph = formal_graph()
        graph["task_dag_edges"][0]["data_mb"] = 0.5

        with self.assertRaisesRegex(ValueError, "DAG payload"):
            validate_formal_graph_boundary(graph)


if __name__ == "__main__":
    unittest.main()
