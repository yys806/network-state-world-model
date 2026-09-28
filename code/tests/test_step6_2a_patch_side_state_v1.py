import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from pi_jwm.step6_2a_planner_objective_side_state_v1 import (
    PlannerRouteCausalSideState, PlannerTaskCausalSideState, advance_objective_side_state,
    check_route_model_alignment, common_support_horizon, deadline_violation,
    planner_derived_return_birth_required, prepare_objective_side_state, support_horizon,
    transmission_burden,
)
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig, StructuredRSSMWorldModel
from pi_jwm.step6_2a_throughput_metric_v1 import extract_real_throughput


def model(route=(0, 1, 2), index=0, revision=0, hop=10, e2e=10):
    return {
        "flow_identity_index": torch.tensor([[0]]), "flow_destination_index": torch.tensor([[2]]),
        "flow_route_revision": torch.tensor([[revision]]), "current_hop_index": torch.tensor([[index]]),
        "current_holder_index": torch.tensor([[route[index]]]),
        "carrying_hop_source_index": torch.tensor([[route[index]]]),
        "carrying_hop_destination_index": torch.tensor([[route[index+1]]]),
        "hop_remaining": torch.tensor([[float(hop)]]), "flow_remaining": torch.tensor([[float(e2e)]]),
        "task_presence": torch.tensor([[True]]), "rb_active_mask": torch.tensor([[True, True]]),
    }


def route(nodes=(0, 1, 2), index=0, revision=0):
    return PlannerRouteCausalSideState("flow::Task_1::Input::0", 0, "Task_1", "Input", 0,
                                      2, tuple(nodes), index, revision, 1.9)


def prepared():
    decision = {"trajectory_id": "t", "frame_index": 1, "capture_event_id": "c",
                "simulation_time_s": 1.9,
                "tasks": [{"task_id": "Task_1", "arrival_time_s": .7,
                           "required_returned_size": .2, "return_destination_id": "D",
                           "current_node_id": "S", "lifecycle": "computing"}],
                "entities": [{"entity_type": "uav"}],
                "node_cpu_capacity_observation_rows": [{"capacity_per_s": 5, "observed_mask": True}]}
    sidecar = {"trajectory_id": "t", "frame_index": 1, "capture_event_id": "c",
               "simulation_time_s": 1.9, "source_time_s": 1.9, "alignment_passed": True,
               "tasks": [{"task_id": "Task_1", "arrival_time_s": .7, "deadline_s": 2}]}
    return decision, sidecar


class SideStateTest(unittest.TestCase):
    def test_deadline_equality_and_tolerance(self):
        task = PlannerTaskCausalSideState("x", 0, 1, 0, 2, 3, 0, 2, False, False, 1)
        self.assertFalse(deadline_violation(task, 3))
        self.assertTrue(deadline_violation(task, 3.00001))
        self.assertFalse(deadline_violation(task, 3.000005, final_completion_time_s=3.000005, no_return_completion=True))
        self.assertTrue(deadline_violation(task, 3.00002, final_completion_time_s=3.00002, no_return_completion=True))
        self.assertFalse(deadline_violation(task, 4, final_completion_time_s=3))

    def test_prepare_and_anchor_frozen_effort(self):
        decision, sidecar = prepared()
        current = prepare_objective_side_state(decision=decision, deadline_sidecar=sidecar,
                  task_slots={"Task_1": 0}, physical_slots={"S": 0, "D": 2},
                  routes=[route()], model_state=model())
        self.assertEqual(current.effort_component_mask, (True, True, True))
        self.assertEqual(current.effort_denominators, (2, 5, 15))
        self.assertTrue(current.tasks[0].requires_return)
        advanced = advance_objective_side_state(current, predicted_state=model(index=1, hop=8), slot_duration_s=.1)
        self.assertEqual(advanced.effort_denominators, current.effort_denominators)
        self.assertAlmostEqual(advanced.tasks[0].absolute_deadline_s, 2.7)
        self.assertAlmostEqual(advanced.current_time_s, 2.0)

    def test_future_fields_and_unaligned_sidecar_rejected(self):
        decision, sidecar = prepared()
        for altered in ({**sidecar, "future_target": {}}, {**sidecar, "alignment_passed": False},
                        {**sidecar, "source_time_s": 2.0}):
            with self.assertRaises(ValueError):
                prepare_objective_side_state(decision=decision, deadline_sidecar=altered,
                    task_slots={"Task_1": 0}, physical_slots={"S": 0, "D": 2},
                    routes=[route()], model_state=model())

    def test_burden_intermediate_and_terminal(self):
        self.assertEqual(transmission_burden(route(), model(hop=8, e2e=10)), 18)
        self.assertEqual(transmission_burden(route(index=1), model(index=1, hop=10, e2e=10)), 10)
        self.assertEqual(transmission_burden(route(index=1), model(index=1, hop=6, e2e=6)), 6)
        with self.assertRaises(ValueError):
            check_route_model_alignment(route(nodes=(0, 3, 2), revision=1), model(revision=1))

    def test_return_birth_and_support_off_by_one(self):
        decision, sidecar = prepared()
        current = prepare_objective_side_state(decision=decision, deadline_sidecar=sidecar,
                  task_slots={"Task_1": 0}, physical_slots={"S": 0, "D": 2},
                  routes=[route()], model_state=model())
        task = current.tasks[0]
        self.assertTrue(planner_derived_return_birth_required(task, True, False, 0))
        self.assertFalse(planner_derived_return_birth_required(task, True, False, 2))
        self.assertFalse(planner_derived_return_birth_required(task, True, True, 0))
        self.assertEqual(support_horizon(3, 4), 2)
        self.assertEqual(common_support_horizon([4, 2, 3]), 2)
        with self.assertRaisesRegex(ValueError, "OBJECTIVE_UNSCOREABLE"):
            common_support_horizon([4, support_horizon(1, 4)])

    def test_route_reroute_stale_model_is_detected(self):
        decision, sidecar = prepared()
        current = prepare_objective_side_state(decision=decision, deadline_sidecar=sidecar,
                  task_slots={"Task_1": 0}, physical_slots={"S": 0, "D": 2},
                  routes=[route()], model_state=model())
        # The trained model can update endpoints/revision, but a later hop
        # completion still reads its unchanged route_node_indices.
        predicted = model(route=(0, 3, 2), revision=1)
        updated = advance_objective_side_state(current, predicted_state=predicted,
            slot_duration_s=.1, route_actions=[{"flow_index": 0, "route_node_indices": [0, 3, 2]}])
        self.assertEqual(updated.routes[0].remaining_hops_after_current, 1)
        with self.assertRaises(ValueError):
            advance_objective_side_state(updated, predicted_state=model(index=1, revision=1), slot_duration_s=.1)

    def test_actual_model_cannot_advance_destination_only_two_hop_route(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        # 4.2C-C stores remaining destinations [next, final], not [holder, next, final].
        state["route_node_indices"][0, 0] = torch.tensor([1, 2, -1, -1])
        state["route_node_mask"][0, 0] = torch.tensor([True, True, False, False])
        state["hop_remaining"][0, 0] = .001
        action["route_flow_index"].fill_(-1)
        action["route_task_index"].fill_(-1)
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        future, diagnostic = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertGreater(float(diagnostic["delivered_bytes"][0, 0]), 0)
        self.assertEqual(float(future["hop_remaining"][0, 0]), 0)
        self.assertEqual(int(future["current_hop_index"][0, 0]), 0)  # blocker, not desired semantics
        self.assertEqual(int(future["current_holder_index"][0, 0]), 0)
        self.assertEqual(float(future["flow_remaining"][0, 0]), 5000)

    def test_actual_route_action_keeps_stale_array_and_ignores_hop_count_in_rule(self):
        world = StructuredRSSMWorldModel(StructuredRSSMConfig(
            d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = world.synthetic_fixture()
        state["carrying_active"][0, 0] = False  # isolate the Route rule
        action["route_values"][0, 0] = torch.tensor([0., 3., 2., 2.])
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        first, _ = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        self.assertEqual(int(first["carrying_hop_destination_index"][0, 0]), 3)
        self.assertEqual(int(first["flow_route_revision"][0, 0]), 1)
        self.assertTrue(torch.equal(first["route_node_indices"], state["route_node_indices"]))
        self.assertTrue(torch.equal(first["route_node_mask"], state["route_node_mask"]))
        action["route_values"][0, 0, 3] = 99.
        second, _ = world.deterministic_transition(state, action, learned,
            graph=graph, service_mode="expectation", generator=None)
        for key in ("route_node_indices", "route_node_mask", "current_hop_index",
                    "carrying_hop_source_index", "carrying_hop_destination_index", "flow_route_revision"):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_shared_real_throughput_uses_ledger_e2e_delta(self):
        flows = {"f": {"total_data": 1.0, "logical_destination": "D"}}
        events = [
            {"flow_id": "f", "target_id": "B", "delivered_data": 1.0, "e2e_remaining_after": 1.0},
            {"flow_id": "f", "target_id": "C", "delivered_data": 1.0, "e2e_remaining_after": 1.0},
            {"flow_id": "f", "target_id": "D", "delivered_data": 1.2, "e2e_remaining_after": 0.0},
        ]
        measured = extract_real_throughput(events, elapsed_s=2, flow_definitions=flows)
        self.assertEqual(measured["END_TO_END_USEFUL_THROUGHPUT"], .5)
        self.assertEqual(measured["NETWORK_SERVICE_THROUGHPUT"], 1.6)
        self.assertEqual(measured["e2e_useful_bytes"], 1.0)
        with self.assertRaises(ValueError):
            extract_real_throughput([{**events[0], "e2e_remaining_after": .5}],
                elapsed_s=2, flow_definitions=flows)


if __name__ == "__main__":
    unittest.main()
