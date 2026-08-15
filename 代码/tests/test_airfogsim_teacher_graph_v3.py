from __future__ import annotations

import copy
import importlib
import importlib.util
import unittest


def source_graph() -> dict:
    times = (0.1, 0.2)
    nodes = [
        {"id": "vehicle_0", "kind": "vehicle"},
        {"id": "RSU_0", "kind": "rsu"},
    ]
    node_snapshots = []
    for time in times:
        node_snapshots.extend(
            [
                {
                    **nodes[0],
                    "observed_time": time,
                    "position": [time, 0.0, 0.0],
                    "speed": 1.0,
                    "acceleration": 0.0,
                    "cpu": 3.0,
                    "storage": 1.0,
                },
                {
                    **nodes[1],
                    "observed_time": time,
                    "position": [10.0, 0.0, 0.0],
                    "speed": 0.0,
                    "acceleration": 0.0,
                    "cpu": 10.0,
                    "storage": 1.0,
                },
            ]
        )
    channel_edge = {
        "id": "pe::vehicle_0::RSU_0",
        "src": "vehicle_0",
        "dst": "RSU_0",
        "kind": "V2I",
    }

    return {
        "schema_version": "PI-JWM-AirFogSim-dual-graph-v2",
        "trajectory_id": "fixture",
        "physical_nodes": nodes,
        "physical_edges": [channel_edge],
        "information_nodes": [
            {
                "id": "agent::vehicle_0",
                "physical_node_id": "vehicle_0",
            },
            {"id": "agent::RSU_0", "physical_node_id": "RSU_0"},
        ],
        "information_edges": [
            {
                "id": "flow::Task_1::task_input::vehicle_0::RSU_0",
                "src": "agent::vehicle_0",
                "dst": "agent::RSU_0",
                "flow_type": "task_input",
                "task_id": "Task_1",
                "total_data": 1.0,
                "first_time": 0.1,
            }
        ],
        "source_physical_node_snapshots": node_snapshots,
        "source_physical_edge_snapshots": [
            {
                **channel_edge,
                "observed_time": time,
                "distance": 10.0 - time,
                "csi_mean": 12.0 + time,
                "allocated_rb_count": int(time == 0.2),
                "active_task_count": int(time == 0.2),
                "rate_sum": 5.0 if time == 0.2 else 0.0,
            }
            for time in times
        ],
        "source_transfer_events": [
            {
                "event_id": "event::Task_1::offload::0::0.2",
                "task_id": "Task_1",
                "phase": "offload",
                "source": "vehicle_0",
                "target": "RSU_0",
                "time": 0.2,
                "delivered_data": 1.0,
                "flow_completed": True,
                "path": ["pe::vehicle_0::RSU_0"],
            }
        ],
        "source_task_snapshots": [],
        "source_offload_actions": [],
        "source_return_actions": [],
        "source_rb_actions": [],
        "source_cpu_actions": [],
        "task_nodes": [],
        "task_dag_edges": [],
    }


def subject():
    return importlib.import_module("pi_jwm.airfogsim_teacher_graph_v3")


class TeacherAlignedGraphV3DiscoveryTests(unittest.TestCase):
    def test_teacher_aligned_graph_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pi_jwm.airfogsim_teacher_graph_v3")
        )


class TeacherAlignedGraphV3BehaviorTests(unittest.TestCase):
    def test_remap_separates_spatial_edges_from_communication_edges(self):
        module = subject()
        self.assertTrue(hasattr(module, "remap_teacher_aligned_graph"))

        graph = module.remap_teacher_aligned_graph(source_graph())

        self.assertEqual("PIJWM-DG-Contract-v3", graph["schema_version"])
        self.assertEqual(2, len(graph["physical_edges"]))
        self.assertEqual(1, len(graph["information_edges"]))
        forbidden = {"csi_mean", "allocated_rb_count", "active_task_count", "rate_sum"}
        self.assertFalse(forbidden & set(graph["physical_edges"][0]))
        self.assertFalse(
            forbidden & set(graph["source_physical_edge_snapshots"][0])
        )
        information = graph["source_information_edge_snapshots"][1]
        self.assertEqual(12.2, information["pre"]["csi_mean"])
        self.assertEqual(1, information["action"]["allocated_rb_count"])
        self.assertEqual(1, information["outcome"]["active_task_count"])
        self.assertEqual(5.0, information["outcome"]["rate_sum"])

    def test_remap_builds_unique_cip_cep_and_cfl_relations(self):
        module = subject()
        self.assertTrue(hasattr(module, "remap_teacher_aligned_graph"))
        self.assertTrue(hasattr(module, "validate_teacher_aligned_graph"))

        graph = module.remap_teacher_aligned_graph(source_graph())
        report = module.validate_teacher_aligned_graph(graph)

        self.assertTrue(report["teacher_aligned_graph_valid"])
        self.assertEqual(2, len(graph["cip_relations"]))
        self.assertEqual(1, len(graph["cep_relations"]))
        self.assertEqual(1, len(graph["cfl_relations"]))
        self.assertEqual(
            "information_edge::vehicle_0::RSU_0::V2I",
            graph["cfl_relations"][0]["information_edge_id"],
        )

    def test_source_audit_does_not_require_rerun_for_optional_missing_fields(self):
        module = subject()
        self.assertTrue(hasattr(module, "audit_v3_source_fields"))

        audit = module.audit_v3_source_fields(source_graph())

        self.assertEqual([], audit["required_missing"])
        self.assertIn("sinr", audit["optional_missing"])
        self.assertFalse(audit["airfogsim_rerun_required"])

    def test_validator_rejects_cep_endpoint_mismatch(self):
        module = subject()
        self.assertTrue(hasattr(module, "remap_teacher_aligned_graph"))
        self.assertTrue(hasattr(module, "validate_teacher_aligned_graph"))

        graph = module.remap_teacher_aligned_graph(source_graph())
        broken = copy.deepcopy(graph)
        broken["cep_relations"][0]["physical_edge_id"] = (
            "physical_edge::RSU_0::vehicle_0"
        )

        with self.assertRaisesRegex(ValueError, "CEP endpoint mismatch"):
            module.validate_teacher_aligned_graph(broken)


if __name__ == "__main__":
    unittest.main()
