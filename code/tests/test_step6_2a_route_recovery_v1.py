"""Route recovery acceptance tests. The first test must fail on legacy 4.4."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts")]
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig, StructuredRSSMWorldModel
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from build_step5_1d_unified_model_chain_v1 import build_state, build_action
import json


class RouteRecoveryTest(unittest.TestCase):
    def test_hop_completion_rebinds_current_comm_relation(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        state["route_node_indices"][0, 0] = torch.tensor([1, 0, -1, -1])
        state["route_node_mask"][0, 0] = torch.tensor([True, True, False, False])
        state["flow_destination_index"][0, 0] = 0
        state["hop_remaining"][0, 0] = .001
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        future, _ = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertEqual(int(future["current_holder_index"][0, 0]), 1)
        self.assertEqual(int(future["carrying_hop_destination_index"][0, 0]), 0)
        self.assertEqual(int(future["flow_comm_relation_index"][0, 0]), 1)

    def test_two_hop_partial_and_terminal_lifecycle(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        state["hop_remaining"][0, 0] = 1_000_000.0
        partial, trace = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertGreater(float(trace["delivered_bytes"][0, 0]), 0)
        self.assertEqual(int(partial["current_holder_index"][0, 0]), 0)
        self.assertEqual(int(partial["current_hop_index"][0, 0]), 0)
        self.assertEqual(int(partial["carrying_hop_destination_index"][0, 0]), 1)
        self.assertLess(float(partial["hop_remaining"][0, 0]), 1_000_000.0)
        self.assertEqual(float(partial["flow_remaining"][0, 0]), 5000.0)
        state["hop_remaining"][0, 0] = .001
        first, _ = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertEqual(int(first["current_holder_index"][0, 0]), 1)
        self.assertEqual(int(first["current_hop_index"][0, 0]), 1)
        self.assertEqual(int(first["carrying_hop_destination_index"][0, 0]), 2)
        self.assertEqual(float(first["flow_remaining"][0, 0]), 5000.0)
        self.assertEqual(float(first["hop_remaining"][0, 0]), 5000.0)
        first["hop_remaining"][0, 0] = .001
        first["flow_remaining"][0, 0] = .001
        first["comm_target_index"][0, 1] = 2
        first["flow_comm_relation_index"][0, 0] = 1
        action["comm_allocation_mask"][0, 1, 0] = True
        terminal, trace = world.deterministic_transition(first, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertGreater(float(trace["delivered_bytes"][0, 0]), 0)
        self.assertEqual(float(terminal["flow_remaining"][0, 0]), 0)
        self.assertFalse(bool(terminal["flow_presence"][0, 0]))
        self.assertFalse(bool(terminal["carrying_active"][0, 0]))

    def test_real_two_hop_input_tensor_state_rule_alignment(self):
        base = ROOT / "code/artifacts/protocols"
        samples = json.loads((base / "pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/model_ready_flow_samples.json").read_text(encoding="utf-8"))
        tensor = load_flow_tensor_batch(base / "pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz")
        graph_package = load_typed_dual_graph_batch(base / "pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/typed_dual_graph_batch.npz")
        self.assertIn("real-multihop-cross-slot", samples[1]["metadata"]["sample_id"])
        raw = samples[1]["history"][-1]["carrying_states"][0]
        self.assertEqual(raw["route"], ["RSU_0", "cloudServer_4"])
        self.assertEqual(raw["current_holder"], "RSU_0")
        self.assertEqual(raw["current_hop_index"], 1)
        state, graph = build_state(tensor, graph_package, 1)
        fi = int(raw["flow_index"])
        self.assertEqual(state["route_node_indices"][0, fi, :2].tolist(), raw["route_node_indices"])
        self.assertEqual(int(state["current_holder_index"][0, fi]), raw["current_holder_index"])
        self.assertEqual(int(state["current_hop_index"][0, fi]), raw["current_hop_index"])
        self.assertEqual(int(state["carrying_hop_destination_index"][0, fi]), raw["hop_destination_index"])
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=state["rb_active_mask"].shape[-1]))
        action, _ = build_action(samples[1], state, 0)
        state["hop_remaining"][0, fi] = .000001
        state["flow_remaining"][0, fi] = .000001
        learned = {"vehicle_motion": torch.zeros((*state["position"].shape[:2], 4)), "csi": state["csi"].clone()}
        future, trace = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertGreater(float(trace["delivered_bytes"][0, fi]), 0)
        self.assertFalse(bool(future["flow_presence"][0, fi]))
        self.assertFalse(bool(future["carrying_active"][0, fi]))
        self.assertEqual(float(future["flow_remaining"][0, fi]), 0)

    def test_destination_list_intermediate_hop_advances_holder_and_target(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        # Frozen Causal Flow Ledger: A=0, route=[B=1,C=2], no holder in route.
        state["route_node_indices"][0, 0] = torch.tensor([1, 2, -1, -1])
        state["route_node_mask"][0, 0] = torch.tensor([True, True, False, False])
        state["flow_destination_index"][0, 0] = 2
        state["hop_remaining"][0, 0] = .001
        action["route_flow_index"].fill_(-1)
        action["route_task_index"].fill_(-1)
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        future, trace = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertGreater(float(trace["delivered_bytes"][0, 0]), 0)
        self.assertEqual(int(future["current_hop_index"][0, 0]), 1)
        self.assertEqual(int(future["current_holder_index"][0, 0]), 1)
        self.assertEqual(int(future["carrying_hop_source_index"][0, 0]), 1)
        self.assertEqual(int(future["carrying_hop_destination_index"][0, 0]), 2)
        self.assertEqual(float(future["hop_remaining"][0, 0]), 5000)
        self.assertEqual(float(future["flow_remaining"][0, 0]), 5000)

    def test_same_destination_reroute_writes_full_destination_list(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        state["route_node_indices"][0, 0] = torch.tensor([1, 2, -1, -1])
        state["route_node_mask"][0, 0] = torch.tensor([True, True, False, False])
        action["route_values"][0, 0] = torch.tensor([0., 3., 2., 1.])
        action["route_flow_index"][0, 0] = 0
        action["route_task_index"][0, 0] = 0
        action["comm_allocation_mask"].zero_()
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        metadata = ({"batch_index": 0, "action_row": 0, "flow_index": 0,
                     "task_index": 0, "current_holder_index": 0,
                     "route_revision_before": 0, "route_node_indices": (3, 2)},)
        future, _ = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None,
            route_rule_metadata=metadata)
        self.assertEqual(future["route_node_indices"][0, 0, :2].tolist(), [3, 2])
        self.assertEqual(future["route_node_mask"][0, 0].tolist(), [True, True, False, False])
        self.assertEqual(int(future["current_hop_index"][0, 0]), 0)
        self.assertEqual(int(future["current_holder_index"][0, 0]), 0)
        self.assertEqual(int(future["carrying_hop_destination_index"][0, 0]), 3)
        self.assertEqual(int(future["flow_route_revision"][0, 0]), 1)
        self.assertEqual(int(future["flow_identity_index"][0, 0]), 0)

    def test_destination_change_does_not_fake_a_new_epoch(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        action["route_flow_index"][0, 0] = 0
        action["route_task_index"][0, 0] = 0
        action["route_values"][0, 0, :2] = torch.tensor([0., 3.])
        metadata = ({"batch_index": 0, "action_row": 0, "flow_index": 0,
                     "task_index": 0, "current_holder_index": 0,
                     "route_revision_before": 0, "route_node_indices": (3,)},)
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_BY_FIXED_OBJECT_SUPPORT"):
            world.deterministic_transition(state, action, learned, graph=graph,
                service_mode="expectation", generator=None, route_rule_metadata=metadata)


if __name__ == "__main__":
    unittest.main()
