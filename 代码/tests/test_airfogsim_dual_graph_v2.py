from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def base_records():
    physical_nodes = [
        {"id": "vehicle_0", "kind": "vehicle", "observed_time": 2.6},
        {"id": "UAV_0", "kind": "uav", "observed_time": 2.6},
        {"id": "RSU_0", "kind": "rsu", "observed_time": 2.6},
    ]
    physical_edges = [
        {"id": "pe::vehicle_0::UAV_0", "src": "vehicle_0", "dst": "UAV_0", "kind": "V2U"},
        {"id": "pe::UAV_0::RSU_0", "src": "UAV_0", "dst": "RSU_0", "kind": "U2I"},
        {"id": "pe::RSU_0::vehicle_0", "src": "RSU_0", "dst": "vehicle_0", "kind": "I2V"},
    ]
    tasks = [
        {
            "id": "Task_1",
            "source": "vehicle_0",
            "host": "vehicle_0",
            "exec": "RSU_0",
            "ret": "vehicle_0",
            "task_size": 1.0,
            "return_size": 0.2,
            "lifecycle_state": "offloading",
        },
        {
            "id": "Task_2",
            "source": "vehicle_0",
            "host": "vehicle_0",
            "exec": "UAV_0",
            "ret": "vehicle_0",
            "task_size": 0.8,
            "return_size": 0.1,
            "lifecycle_state": "to_generate",
        },
    ]
    dag_edges = [
        {
            "id": "dag::Task_1::Task_2",
            "src": "Task_1",
            "dst": "Task_2",
            "data_mb": None,
            "payload_status": "not_modeled",
        }
    ]
    transfer_events = [
        {
            "event_id": "event::Task_1::offload::0::2.600000",
            "task_id": "Task_1",
            "phase": "offload",
            "source": "vehicle_0",
            "target": "RSU_0",
            "path": ["pe::vehicle_0::UAV_0", "pe::UAV_0::RSU_0"],
            "rb_indices": [0, 1],
            "remaining_before": 1.0,
            "delivered_data": 0.4,
            "flow_completed": False,
            "time": 2.6,
            "evidence": "direct_runtime_channel_event",
        },
        {
            "event_id": "event::Task_1::return::0::2.800000",
            "task_id": "Task_1",
            "phase": "return",
            "source": "RSU_0",
            "target": "vehicle_0",
            "path": ["pe::RSU_0::vehicle_0"],
            "rb_indices": [2],
            "remaining_before": 0.2,
            "delivered_data": 0.2,
            "flow_completed": True,
            "time": 2.8,
            "evidence": "direct_runtime_channel_event",
        },
    ]
    return physical_nodes, physical_edges, tasks, dag_edges, transfer_events


class DualGraphV2BuildTests(unittest.TestCase):
    def test_disconnected_cloud_is_excluded_from_physical_and_information_graphs(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        bundle = build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=[
                {"id": "vehicle_0", "kind": "vehicle"},
                {"id": "RSU_0", "kind": "rsu"},
                {"id": "cloudServer_0", "kind": "cloud"},
            ],
            physical_edges=[
                {"id": "pe::vehicle_0::RSU_0", "src": "vehicle_0", "dst": "RSU_0", "kind": "V2I"}
            ],
            task_records=[],
            dag_edges=[],
            transfer_events=[],
        )

        self.assertEqual({"vehicle_0", "RSU_0"}, {row["id"] for row in bundle["physical_nodes"]})
        self.assertNotIn("agent::cloudServer_0", {row["id"] for row in bundle["information_nodes"]})

    def test_agents_and_real_flows_replace_task_nodes_and_dag_edges(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        nodes, edges, tasks, dags, events = base_records()
        bundle = build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=events,
        )

        self.assertEqual(
            {"agent::vehicle_0", "agent::UAV_0", "agent::RSU_0"},
            {row["id"] for row in bundle["information_nodes"]},
        )
        self.assertFalse({"Task_1", "Task_2"} & {row["id"] for row in bundle["information_nodes"]})
        self.assertEqual({"Task_1", "Task_2"}, {row["id"] for row in bundle["task_nodes"]})
        self.assertEqual({"dag::Task_1::Task_2"}, {row["id"] for row in bundle["task_dag_edges"]})

        flows = {row["flow_type"]: row for row in bundle["information_edges"]}
        self.assertEqual({"task_input", "result_return"}, set(flows))
        self.assertEqual("agent::vehicle_0", flows["task_input"]["src"])
        self.assertEqual("agent::RSU_0", flows["task_input"]["dst"])
        self.assertEqual("agent::RSU_0", flows["result_return"]["src"])
        self.assertEqual("agent::vehicle_0", flows["result_return"]["dst"])
        self.assertEqual(0.6, flows["task_input"]["remaining_data"])
        self.assertEqual("completed", flows["result_return"]["status"])

    def test_every_agent_has_one_attachment_and_every_observed_hop_has_a_bearer(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        nodes, edges, tasks, dags, events = base_records()
        bundle = build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=events,
        )

        self.assertEqual(3, len(bundle["agent_attachments"]))
        self.assertTrue(all(row["value"] == 1 for row in bundle["agent_attachments"]))
        self.assertEqual(
            {
                "pe::vehicle_0::UAV_0",
                "pe::UAV_0::RSU_0",
                "pe::RSU_0::vehicle_0",
            },
            {row["physical_edge_id"] for row in bundle["flow_bearers"]},
        )

    def test_precedence_without_payload_does_not_create_dependency_flow(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        nodes, edges, tasks, dags, events = base_records()
        bundle = build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=events,
        )

        self.assertNotIn(
            "dependency_data",
            {row["flow_type"] for row in bundle["information_edges"]},
        )
        self.assertEqual("not_modeled", bundle["task_dag_edges"][0]["payload_status"])

    def test_explicit_cross_agent_dependency_payload_creates_waiting_flow(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        nodes, edges, tasks, dags, events = base_records()
        dags[0]["data_mb"] = 0.3
        dags[0]["payload_status"] = "pi_jwm_explicit"
        bundle = build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=events,
        )

        dependency = next(
            row for row in bundle["information_edges"] if row["flow_type"] == "dependency_data"
        )
        self.assertEqual("agent::RSU_0", dependency["src"])
        self.assertEqual("agent::UAV_0", dependency["dst"])
        self.assertEqual(0.3, dependency["total_data"])
        self.assertEqual(0.3, dependency["remaining_data"])
        self.assertEqual("pending", dependency["status"])
        self.assertFalse(
            any(row["flow_id"] == dependency["id"] for row in bundle["flow_bearers"])
        )


class DualGraphV2ValidationTests(unittest.TestCase):
    def build_valid_bundle(self):
        from pi_jwm.airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle

        nodes, edges, tasks, dags, events = base_records()
        return build_dual_graph_v2_bundle(
            trajectory_id="seed0",
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=events,
        )

    def test_valid_bundle_passes_new_semantic_gates(self):
        from pi_jwm.airfogsim_dual_graph_v2 import validate_dual_graph_v2_bundle

        report = validate_dual_graph_v2_bundle(self.build_valid_bundle())

        self.assertTrue(report["dual_graph_v2_ready"])
        self.assertEqual([], report["failed_checks"])

    def test_invalid_attachment_and_missing_bearer_edge_are_rejected(self):
        from pi_jwm.airfogsim_dual_graph_v2 import validate_dual_graph_v2_bundle

        bundle = self.build_valid_bundle()
        bundle["agent_attachments"].append(dict(bundle["agent_attachments"][0]))
        bundle["flow_bearers"][0]["physical_edge_id"] = "pe::missing::edge"

        report = validate_dual_graph_v2_bundle(bundle)

        self.assertFalse(report["dual_graph_v2_ready"])
        self.assertIn("unique_agent_attachment", report["failed_checks"])
        self.assertIn("bearer_edges_exist", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
