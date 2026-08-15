import unittest

import numpy as np

from pi_jwm.airfogsim_teacher_tensor_v3 import TeacherAlignedTensorContract
from pi_jwm.r3_preflight_data import make_explicit_batch
from pi_jwm.r6_online_observation import (
    OnlineDualGraphHistory,
    build_online_teacher_arrays,
    make_online_inference_payload,
)


def _node(node_id, kind, time, x):
    return {
        "trajectory_id": "online-train",
        "id": node_id,
        "kind": kind,
        "observed_time": time,
        "position": [x, 0.0, 0.0],
        "speed": 0.0,
        "acceleration": 0.0,
        "cpu": 10.0,
        "storage": 1.0,
        "evidence": "direct",
    }


class R6OnlineObservationTest(unittest.TestCase):
    def test_history_builds_source_graph_from_live_frames_and_actual_actions(self):
        history = OnlineDualGraphHistory("online-train")
        for time in (0.1, 0.2):
            history.append_frame(
                physical_nodes=[_node("UAV_0", "uav", time, time)],
                physical_edges=[],
                task_records=[],
                dag_edges=[],
                transfer_events=[],
                offload_actions=[
                    {
                        "time": time,
                        "task_id": "Task_1",
                        "source_node_id": "UAV_0",
                        "target_node_id": "UAV_0",
                    }
                ],
                return_actions=[],
                rb_actions=[],
                cpu_actions=[],
            )

        source = history.build_source_graph()

        self.assertEqual("online-train", source["trajectory_id"])
        self.assertEqual(2, len(source["source_physical_node_snapshots"]))
        self.assertAlmostEqual(0.2, source["physical_nodes"][0]["observed_time"])
        self.assertEqual(2, len(source["source_offload_actions"]))
        self.assertEqual(2, history.frame_count)
        self.assertAlmostEqual(0.2, history.last_observed_time)

    def test_bounded_history_keeps_only_the_frozen_eight_step_context(self):
        history = OnlineDualGraphHistory("online-train", max_frames=8)
        for index in range(1, 11):
            time = round(index / 10.0, 1)
            history.append_frame(
                physical_nodes=[_node("UAV_0", "uav", time, time)],
                physical_edges=[],
                task_records=[],
                dag_edges=[],
                transfer_events=[],
                offload_actions=[{"time": time, "task_id": f"Task_{index}"}],
                return_actions=[],
                rb_actions=[],
                cpu_actions=[],
            )

        source = history.build_source_graph()

        self.assertEqual(8, history.frame_count)
        self.assertEqual(8, len(source["source_physical_node_snapshots"]))
        self.assertAlmostEqual(0.3, source["source_physical_node_snapshots"][0]["observed_time"])
        self.assertEqual("Task_3", source["source_offload_actions"][0]["task_id"])

    def test_recent_vocabulary_keeps_departed_endpoint_for_historical_edges(self):
        history = OnlineDualGraphHistory("online-train", max_frames=8)
        first_nodes = [
            _node("UAV_0", "uav", 0.1, 0.0),
            _node("vehicle_0", "vehicle", 0.1, 1.0),
        ]
        edge = {
            "trajectory_id": "online-train",
            "id": "pe::vehicle_0::UAV_0",
            "src": "vehicle_0",
            "dst": "UAV_0",
            "kind": "V2U",
            "observed_time": 0.1,
        }
        history.append_frame(
            physical_nodes=first_nodes,
            physical_edges=[edge],
            task_records=[], dag_edges=[], transfer_events=[], offload_actions=[],
            return_actions=[], rb_actions=[], cpu_actions=[],
        )
        history.append_frame(
            physical_nodes=[_node("UAV_0", "uav", 0.2, 0.0)],
            physical_edges=[],
            task_records=[], dag_edges=[], transfer_events=[], offload_actions=[],
            return_actions=[], rb_actions=[], cpu_actions=[],
        )

        source = history.build_source_graph()

        self.assertEqual({"UAV_0", "vehicle_0"}, {row["id"] for row in source["physical_nodes"]})
        self.assertEqual({"pe::vehicle_0::UAV_0"}, {row["id"] for row in source["physical_edges"]})

    def test_filtered_network_attached_nodes_do_not_leave_orphan_snapshots(self):
        history = OnlineDualGraphHistory("online-train")
        history.append_frame(
            physical_nodes=[
                _node("UAV_0", "uav", 0.1, 0.0),
                _node("cloud_0", "cloud", 0.1, 10.0),
            ],
            physical_edges=[], task_records=[], dag_edges=[], transfer_events=[],
            offload_actions=[], return_actions=[], rb_actions=[], cpu_actions=[],
        )

        source = history.build_source_graph()

        self.assertEqual({"UAV_0"}, {row["id"] for row in source["physical_nodes"]})
        self.assertEqual(
            {"UAV_0"},
            {row["id"] for row in source["source_physical_node_snapshots"]},
        )

    def test_live_source_is_remapped_to_strict_dual_graph_tensor(self):
        times = [round(0.1 * index, 1) for index in range(1, 9)]
        node_snapshots = [
            _node(node_id, kind, time, x)
            for time in times
            for node_id, kind, x in (("UAV_0", "uav", time), ("RSU_0", "rsu", 10.0))
        ]
        link_snapshots = [
            {
                "trajectory_id": "online-train",
                "id": "pe::UAV_0::RSU_0",
                "src": "UAV_0",
                "dst": "RSU_0",
                "kind": "U2I",
                "observed_time": time,
                "distance": 10.0 - time,
                "csi_mean": 2.0,
                "rate_sum": 3.0,
                "active_task_count": 0,
                "allocated_rb_count": 0,
                "evidence": "direct",
            }
            for time in times
        ]
        source = {
            "trajectory_id": "online-train",
            "physical_nodes": [node_snapshots[-2], node_snapshots[-1]],
            "source_physical_node_snapshots": node_snapshots,
            "source_physical_edge_snapshots": link_snapshots,
            "information_edges": [],
            "task_nodes": [],
            "task_dag_edges": [],
            "source_task_snapshots": [],
            "source_transfer_events": [],
            "source_offload_actions": [],
            "source_return_actions": [],
            "source_rb_actions": [],
            "source_cpu_actions": [],
        }
        contract = TeacherAlignedTensorContract(
            max_nodes=2,
            max_physical_edges=2,
            max_information_edges=2,
            max_flows=1,
            max_tasks=1,
            max_dag_edges=1,
        )

        arrays, report = build_online_teacher_arrays(source, contract=contract)

        self.assertTrue(report["validation"]["teacher_aligned_tensor_valid"])
        self.assertEqual(arrays["physical_node_state"].shape, (8, 2, 9))
        self.assertEqual(arrays["information_edge_state"].shape, (8, 2, 18))
        self.assertEqual(int(arrays["information_edge_present"].sum()), 8)

    def test_inference_payload_uses_only_last_history_and_duplicates_current_as_placeholder_target(self):
        times = [round(0.1 * index, 1) for index in range(1, 10)]
        source = {
            "trajectory_id": "online-train",
            "physical_nodes": [_node("UAV_0", "uav", times[-1], times[-1])],
            "source_physical_node_snapshots": [
                _node("UAV_0", "uav", time, time) for time in times
            ],
            "source_physical_edge_snapshots": [],
            "information_edges": [],
            "task_nodes": [],
            "task_dag_edges": [],
            "source_task_snapshots": [],
            "source_transfer_events": [],
            "source_offload_actions": [],
            "source_return_actions": [],
            "source_rb_actions": [],
            "source_cpu_actions": [],
        }
        contract = TeacherAlignedTensorContract(
            max_nodes=1,
            max_physical_edges=1,
            max_information_edges=1,
            max_flows=1,
            max_tasks=1,
            max_dag_edges=1,
        )
        arrays, _ = build_online_teacher_arrays(source, contract=contract)

        payload = make_online_inference_payload(
            arrays,
            trajectory_id="online-train",
            environment_seed=0,
            split="train",
            history_steps=8,
        )

        self.assertEqual(payload["history"]["physical_node_state"].shape[0], 8)
        self.assertAlmostEqual(float(payload["history_time"][0]), 0.2, places=6)
        self.assertAlmostEqual(float(payload["history_time"][-1]), 0.9, places=6)
        np.testing.assert_array_equal(
            payload["target"]["physical_node_state"][0],
            payload["history"]["physical_node_state"][-1],
        )
        self.assertFalse(payload["future_action"]["task_action_present"].any())
        self.assertTrue((payload["future_action"]["task_action_information_node_index"] == -1).all())
        normalization = {
            "source_split": "train",
            "features": {
                name: {
                    "mean": [0.0] * value.shape[-1],
                    "scale": [1.0] * value.shape[-1],
                }
                for name, value in payload["history"].items()
                if name.endswith("_state")
            },
        }
        batch = make_explicit_batch(payload, normalization, device="cpu")
        self.assertEqual(batch.metadata["state_source"], "online_airfogsim_strict_dual_graph")


if __name__ == "__main__":
    unittest.main()
