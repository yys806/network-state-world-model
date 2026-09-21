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
    derive_wired_active_membership,
    rayleigh_outage_probability,
    resolve_wireless_sinr_db,
    save_world_model_package,
    load_world_model_package,
    validate_world_model_acceptance,
    wired_fair_share_service,
    wireless_nominal_rate_mbps,
    world_model_digest,
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
