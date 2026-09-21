from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from pi_jwm.step4_4_structured_rssm_world_model_v1 import (
    StructuredRSSMConfig,
    StructuredRSSMWorldModel,
    REQUIRED_ACCEPTANCE_CHECKS,
    apply_known_stochastic_wireless_service,
    bind_existing_return_flows,
    derive_wired_active_membership,
    rayleigh_outage_probability,
    resolve_wireless_sinr_db,
    save_world_model_package,
    load_world_model_package,
    validate_world_model_acceptance,
    wired_fair_share_service,
    wireless_nominal_rate_mbps,
    world_model_digest,
    ADDITIONAL_STRUCTURAL_CHECKS,
)


class Step44ServiceRuleTests(unittest.TestCase):
    def test_nominal_rate_and_rayleigh_outage_match_audited_formulas(self):
        sinr_db = torch.tensor([0.0, 10.0, 20.0])
        expected_rate = 2.0 * torch.log2(1.0 + torch.pow(10.0, sinr_db / 10.0))
        self.assertTrue(torch.allclose(wireless_nominal_rate_mbps(sinr_db, 2.0), expected_rate))
        clamped = sinr_db.clamp_min(1e-9)
        self.assertTrue(torch.allclose(rayleigh_outage_probability(sinr_db, 1.0), 1.0 - torch.exp(-1.0 / clamped)))

    def test_expectation_and_seeded_sample_are_explicit(self):
        sinr = torch.full((64,), 1.0)
        expected = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=2.0, snr_threshold=1.0, mode="expectation")
        self.assertTrue(expected["expected_service_approximation"])
        self.assertTrue(torch.allclose(expected["actual_rate_mbps"], (1.0 - expected["outage_probability"]) * expected["nominal_rate_mbps"]))
        a = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=2.0, snr_threshold=1.0, mode="sample", generator=torch.Generator().manual_seed(7))
        b = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=2.0, snr_threshold=1.0, mode="sample", generator=torch.Generator().manual_seed(7))
        c = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=2.0, snr_threshold=1.0, mode="sample", generator=torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(a["outage"], b["outage"]))
        self.assertFalse(torch.equal(a["outage"], c["outage"]))
        self.assertTrue(torch.equal(a["nominal_rate_mbps"], c["nominal_rate_mbps"]))
        self.assertTrue(torch.all(a["actual_rate_mbps"][a["outage"]] == 0))
        self.assertTrue(torch.allclose(a["actual_rate_mbps"][~a["outage"]], a["nominal_rate_mbps"][~a["outage"]]))

    def test_wired_membership_and_bytes_rule(self):
        carrying = {
            "active": torch.tensor([[True, True, False]]),
            "hop_source_index": torch.tensor([[1, 1, 1]]),
            "hop_destination_index": torch.tensor([[2, 2, 2]]),
            "flow_index": torch.tensor([[3, 4, 5]]),
        }
        membership = derive_wired_active_membership(carrying, {(1, 2)})
        self.assertEqual(membership[(0, 1, 2)], (3, 4))
        delivered = wired_fair_share_service(torch.tensor([100.0, 1_000_000.0]), capacity_mbps=1.0, slot_duration_s=0.1, active_count=2)
        self.assertTrue(torch.equal(delivered, torch.tensor([100.0, 6250.0])))

    def test_complete_allocation_changes_interference_not_signal_rule(self):
        csi = torch.full((1, 2, 2), 80.0)
        src, dst = torch.tensor([[0, 1]]), torch.tensor([[1, 0]])
        types = torch.tensor([[2, 3]])
        no_collision = resolve_wireless_sinr_db(csi, src, dst, types, types.flip(1), torch.tensor([[[True, False], [False, True]]]), noise_power_mw=1e-9, v2v_tx_power_dbm=23, vehicle_to_infra_tx_power_dbm=26, uav_or_rsu_tx_power_dbm=29)
        collision = resolve_wireless_sinr_db(csi, src, dst, types, types.flip(1), torch.ones((1, 2, 2), dtype=torch.bool), noise_power_mw=1e-9, v2v_tx_power_dbm=23, vehicle_to_infra_tx_power_dbm=26, uav_or_rsu_tx_power_dbm=29)
        self.assertLessEqual(float(collision[0, 0, 0]), float(no_collision[0, 0, 0]) + 1e-6)
        self.assertLessEqual(float(collision[0, 1, 1]), float(no_collision[0, 1, 1]) + 1e-6)

    def test_sample_requires_explicit_generator(self):
        with self.assertRaises(ValueError):
            apply_known_stochastic_wireless_service(torch.ones(2), rb_bandwidth_mhz=2.0, snr_threshold=10.0, mode="sample")


class Step44ArchitectureTests(unittest.TestCase):
    def test_existing_return_flow_binding_uses_task_and_type_identity(self):
        mapping = bind_existing_return_flows(
            task_presence=torch.tensor([[True, True]]),
            flow_known=torch.tensor([[True, True, True]]),
            flow_task_index=torch.tensor([[0, 0, 1]]),
            flow_type_index=torch.tensor([[2, 3, 3]]),
        )
        self.assertTrue(torch.equal(mapping, torch.tensor([[1, 2]])))

    def test_computation_finished_waits_for_existing_return_flow(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = model.synthetic_fixture()
        state["flow_task_index"] = torch.tensor([[0, 0]])
        state["flow_known"] = torch.tensor([[True, True]])
        state["flow_type_index"] = torch.tensor([[2, 3]])
        state["return_flow_index"] = bind_existing_return_flows(
            state["task_presence"], state["flow_known"], state["flow_task_index"], state["flow_type_index"]
        )
        state["task_requires_return"] = state["return_flow_index"] >= 0
        state["task_work_remaining"][0, 0] = 0.0
        state["flow_presence"][0, 1] = True
        state["carrying_active"][0, 1] = False
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        next_state, _ = model.deterministic_transition(state, action, learned, graph=graph, service_mode="expectation", generator=None)
        self.assertFalse(bool(next_state["task_completed"][0, 0]))
        self.assertFalse(bool(next_state["return_birth_required"][0, 0]))

        completed_return = {key: value.clone() for key, value in state.items()}
        completed_return["flow_presence"][0, 1] = False
        next_state, _ = model.deterministic_transition(completed_return, action, learned, graph=graph, service_mode="expectation", generator=None)
        self.assertTrue(bool(next_state["task_completed"][0, 0]))

    def test_missing_required_return_support_blocks_final_completion(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = model.synthetic_fixture()
        state["task_work_remaining"][0, 0] = 0.0
        state["task_requires_return"][0, 0] = True
        state["return_flow_index"][0, 0] = -1
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        next_state, _ = model.deterministic_transition(state, action, learned, graph=graph, service_mode="expectation", generator=None)
        self.assertFalse(bool(next_state["task_completed"][0, 0]))
        self.assertTrue(bool(next_state["return_birth_required"][0, 0]))
        self.assertTrue(bool(next_state["final_completion_blocked_by_fixed_support"][0, 0]))

    def test_rollout_declares_fixed_support_and_no_future_return_birth(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        out = model.rollout(zpi, state, graph, [action], future_target={"return_flow": torch.tensor([99])}, prior_mode="mean", service_mode="expectation")
        self.assertTrue(out["diagnostics"]["fixed_current_object_support"])
        self.assertFalse(out["diagnostics"]["future_return_birth_supported"])

    def test_hop_cap_and_partial_progress_are_distinct_from_flow_remaining(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        state["flow_remaining"][0, 0] = 1000.0
        state["hop_remaining"][0, 0] = 10.0
        state["hop_progress"][0, 0] = 0.0
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        next_state, trace = model.deterministic_transition(state, action, learned, graph=graph, service_mode="expectation", generator=None)
        self.assertLessEqual(float(trace["delivered_bytes"][0, 0]), 10.0)
        self.assertTrue(float(next_state["hop_progress"][0, 0]) <= 10.0)
        self.assertTrue(float(next_state["hop_remaining"][0, 0]) >= 0.0)

    def test_unchanged_route_does_not_increment_revision_and_flow_embeddings_are_semantic(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        _, state, graph, action = model.synthetic_fixture()
        action["route_values"][0, 0, :2] = torch.tensor([0.0, 1.0])
        learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": state["csi"].clone()}
        next_state, _ = model.deterministic_transition(state, action, learned, graph=graph, service_mode="expectation", generator=None)
        self.assertTrue(torch.equal(next_state["flow_route_revision"], state["flow_route_revision"]))
        base = model._state_features("flow", state)
        changed = {k: v.clone() for k, v in state.items()}
        changed["flow_type_index"][0, 0] += 1
        self.assertFalse(torch.equal(base, model._state_features("flow", changed)))

    def test_structural_acceptance_checks_are_declared(self):
        expected = {
            "no_raw_index_learned_feature", "categorical_embedding_semantics", "graph_layers_effective",
            "future_topology_matches_builder_policy", "no_unintended_self_physical_edges", "carrying_state_complete",
            "hop_service_capped_by_hop_remaining", "hop_advancement", "dynamic_flow_comm_mapping",
            "flow_completion_presence_sync", "route_revision_semantics", "task_lifecycle_rule_complete",
            "dag_dynamic_rule", "comm_endpoint_presence_validity", "task_agent_dynamic_validity",
            "strong_recursive_counterfactual", "existing_return_flow_typed_binding",
            "future_return_birth_unsupported", "computation_finished_not_final_without_return",
            "input_flow_not_return_substitute",
        }
        self.assertEqual(set(ADDITIONAL_STRUCTURAL_CHECKS), expected)

    def test_graph_layers_really_change_processor_depth(self):
        one = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        two = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=2, n_comm_rb=4))
        self.assertEqual(len(one.dynamics_processor_layers), 1)
        self.assertEqual(len(two.dynamics_processor_layers), 2)
        self.assertNotEqual(world_model_digest(one.dynamics_processor_layers.state_dict()), world_model_digest(two.dynamics_processor_layers.state_dict()))

    def test_flow_state_contains_carrying_progress_and_hop_advances(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        for key in ("current_holder_index", "current_hop_index", "hop_progress", "hop_remaining", "route_node_indices", "route_node_mask"):
            self.assertIn(key, state)
        state["flow_remaining"][0, 0] = 1000.0
        state["hop_remaining"][0, 0] = 1.0
        state["route_node_indices"][0, 0] = torch.tensor([0, 1, 2, 3])
        state["route_node_mask"][0, 0] = True
        out = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        self.assertEqual(int(out["states"][0]["current_hop_index"][0, 0]), 1)
        self.assertTrue(bool(out["states"][0]["carrying_active"][0, 0]))
    def test_structured_layout_and_forbidden_heads(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        modules = dict(model.named_modules())
        for name in ("phy_init", "agent_init", "comm_init", "flow_init", "task_init", "phy_prior", "phy_posterior", "comm_prior", "comm_posterior", "vehicle_decoder", "csi_decoder"):
            self.assertIn(name, modules)
        for forbidden in ("outage_head", "rate_head", "service_head", "service_residual", "flow_remaining_head", "task_progress_head", "acceleration_decoder"):
            self.assertNotIn(forbidden, modules)
        self.assertFalse(model.contract["learned_service_residual"])
        self.assertEqual(model.contract["stochastic_families"], ["physical_vehicle", "communication"])

    def test_prior_posterior_and_local_action_routing(self):
        cfg = StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4)
        model = StructuredRSSMWorldModel(cfg)
        zpi, state, graph, action = model.synthetic_fixture(batch_size=1)
        latent = model.initialize_latent(zpi, state)
        self.assertEqual(set(latent["h"]), {"physical", "agent", "communication", "flow", "task"})
        self.assertEqual(set(latent["z"]), {"physical", "communication"})
        routed = model.route_actions(action, state, graph)
        changed = copy.deepcopy(action)
        changed["mobility_values"] = action["mobility_values"].clone()
        changed["mobility_values"][0, 0, 0] += 1
        rerouted = model.route_actions(changed, state, graph)
        target = int(action["mobility_entity_index"][0, 0])
        other = torch.arange(state["entity_presence"].shape[1]) != target
        self.assertFalse(torch.equal(routed["physical"][0, target], rerouted["physical"][0, target]))
        self.assertTrue(torch.equal(routed["physical"][0, other], rerouted["physical"][0, other]))
        bad = copy.deepcopy(action)
        bad["mobility_entity_index"] = torch.tensor([[99]])
        with self.assertRaises(ValueError):
            model.route_actions(bad, state, graph)

    def test_recursive_prior_rollout_dynamic_graph_and_no_target_leakage(self):
        cfg = StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4, rollout_horizon=2)
        model = StructuredRSSMWorldModel(cfg)
        zpi, state, graph, action = model.synthetic_fixture(batch_size=1)
        target = {"future_secret": torch.tensor([1.0])}
        out1 = model.rollout(zpi, state, graph, [action, action], future_target=target, prior_mode="mean", service_mode="expectation")
        target["future_secret"] = torch.tensor([9999.0])
        out2 = model.rollout(zpi, state, graph, [action, action], future_target=target, prior_mode="mean", service_mode="expectation")
        self.assertTrue(torch.equal(out1["states"][-1]["position"], out2["states"][-1]["position"]))
        self.assertTrue(out1["diagnostics"]["future_target_ignored"])
        self.assertTrue(out1["diagnostics"]["recursive_state_feedback"])
        self.assertFalse(torch.equal(out1["graphs"][0]["physical_relation_features"], out1["graphs"][1]["physical_relation_features"]))
        self.assertTrue(out1["diagnostics"]["expected_service_approximation"])

    def test_posterior_teacher_can_change_without_entering_prior_rollout(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        base = model.initialize_latent(zpi, state)
        changed = copy.deepcopy(zpi); changed["physical"]["node_latent"] = changed["physical"]["node_latent"] + 2
        teacher = model.initialize_latent(changed, state)
        self.assertFalse(torch.equal(base["posterior"]["physical"]["mean"], teacher["posterior"]["physical"]["mean"]))
        out = model.rollout(zpi, state, graph, [action], future_target=changed, prior_mode="mean", service_mode="expectation")
        reference = model.rollout(zpi, state, graph, [action], future_target=None, prior_mode="mean", service_mode="expectation")
        self.assertEqual(world_model_digest(out), world_model_digest(reference))

    def test_vehicle_uav_static_stochastic_eligibility(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, _, _ = model.synthetic_fixture()
        latent = model.initialize_latent(zpi, state)
        self.assertTrue(torch.all(latent["z"]["physical"][~state["vehicle_mask"]] == 0))
        self.assertTrue(torch.any(latent["z"]["physical"][state["vehicle_mask"]] != 0))

    def test_intermediate_hop_does_not_reduce_e2e_remaining(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        state["flow_destination_index"][0, 0] = 3
        out = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        trace = out["traces"][0]["rule"]
        self.assertGreaterEqual(float(trace["delivered_bytes"][0, 0].detach()), 0.0)
        self.assertEqual(float(trace["e2e_reduction_bytes"][0, 0].detach()), 0.0)

    def test_route_updates_carrying_and_task_agent_but_preserves_flow_identity(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        identity = state["flow_identity_index"].clone()
        action["route_values"][0, 0, :2] = torch.tensor([1.0, 0.0])
        out = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        self.assertTrue(torch.equal(out["states"][0]["flow_identity_index"], identity))
        self.assertEqual(int(out["states"][0]["carrying_hop_destination_index"][0, 0]), 0)
        self.assertEqual(int(out["graphs"][0]["task_agent_agent_index"][0, 0]), 0)

    def test_package_round_trip(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        before = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"; save_world_model_package(model, path)
            after = load_world_model_package(path).rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        self.assertEqual(world_model_digest(before), world_model_digest(after))

    def test_acceptance_validator_ands_every_required_check(self):
        scope = {name: False for name in ("loss", "optimizer", "training", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim")}
        receipt = {"required_checks": {name: True for name in REQUIRED_ACCEPTANCE_CHECKS}, "scope": scope, "evidence_class": "UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE", "passed": True}
        self.assertTrue(validate_world_model_acceptance(receipt)["passed"])
        receipt["required_checks"]["no_target_leakage"] = False
        receipt["passed"] = False
        self.assertFalse(validate_world_model_acceptance(receipt)["passed"])

    def test_sample_rollout_seed_replay(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        a = model.rollout(zpi, state, graph, [action], prior_mode="sample", service_mode="sample", generator=torch.Generator().manual_seed(91))
        b = model.rollout(zpi, state, graph, [action], prior_mode="sample", service_mode="sample", generator=torch.Generator().manual_seed(91))
        self.assertEqual(world_model_digest(a), world_model_digest(b))

    def test_complete_comm_action_is_written_before_service(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        action["comm_allocation_mask"].zero_(); action["comm_allocation_mask"][0, 1, 2] = True
        out = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        self.assertTrue(torch.equal(out["states"][0]["rb_active_mask"], action["comm_allocation_mask"]))
        self.assertEqual(int(torch.count_nonzero(out["traces"][0]["rule"]["wireless"]["nominal_rate_mbps"])), 1)

    def test_absent_relation_cannot_become_csi_valid(self):
        model = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8, d_z=3, mlp_width=12, graph_layers=1, n_comm_rb=4))
        zpi, state, graph, action = model.synthetic_fixture()
        state["comm_presence"][0, 0] = False
        state["comm_validity"][0, 0] = True
        state["csi_mask"][0, 0] = True
        action["comm_relation_index"] = torch.tensor([[1]])
        out = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
        self.assertFalse(bool(out["states"][0]["csi_mask"][0, 0].any()))
        self.assertFalse(bool(out["graphs"][0]["geo_comm_validity"][0, 0]))


if __name__ == "__main__":
    unittest.main()
