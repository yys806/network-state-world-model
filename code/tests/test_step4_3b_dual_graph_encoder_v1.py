import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import PhysicalTopologyConfig, build_typed_dual_graph_batch, load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import (
    REQUIRED_ACCEPTANCE_CHECKS,
    DualGraphEncoderConfig,
    PIJointGraphEncoder,
    build_encoder_input_audit,
    fit_encoder_normalization_stats,
    load_encoder_package,
    output_semantic_digest,
    save_encoder_package,
    validate_encoder_acceptance,
    validate_encoder_architecture_checks,
    validate_encoder_contract_checks,
)


ROOT = Path(__file__).resolve().parents[2]
TENSOR_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz"
GRAPH_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/typed_dual_graph_batch.npz"
UPSTREAM_STATS = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/train_normalization_stats.json"


class Step43BDualGraphEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensor = load_flow_tensor_batch(TENSOR_PATH)
        cls.graph = load_typed_dual_graph_batch(GRAPH_PATH)
        cls.stats = fit_encoder_normalization_stats(cls.tensor, cls.graph, UPSTREAM_STATS)
        cls.config = DualGraphEncoderConfig(d_h=16, mlp_hidden_width=24, type_embedding_dim=4, lifecycle_embedding_dim=4, direction_embedding_dim=4, graph_layers=2, initialization_seed=431)

    def model(self):
        return PIJointGraphEncoder(self.config, self.tensor["contract"], self.graph["contract"], self.stats)

    def test_input_audit_is_machine_readable_and_truthful(self):
        audit = build_encoder_input_audit(self.tensor, self.graph, UPSTREAM_STATS)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["continuous_inputs"]["logical_flow"]["learned_fields"], ["logical.total_data", "logical.e2e_remaining"])
        self.assertEqual(audit["continuous_inputs"]["logical_flow"]["excluded_redundant_fields"], ["logical.e2e_delivered"])
        self.assertEqual(audit["unavailable_fields"]["agent.dynamic_available_cpu"], "UPSTREAM_FIELD_NOT_AVAILABLE")
        self.assertFalse(audit["current_non_flow_tensor_features_are_pre_normalized"])

    def test_forward_outputs_aligned_z_pi_only(self):
        model = self.model()
        out = model(self.tensor, self.graph)
        self.assertEqual(out["schema_version"], "PI-JWM-Joint-Graph-Representation-v1-step4.3B")
        self.assertNotIn("global_latent", out)
        self.assertNotIn("world_model_latent", out)
        self.assertEqual(tuple(out["physical"]["node_latent"].shape), (5, 8, 16))
        self.assertEqual(tuple(out["information"]["agent_latent"].shape), (5, 8, 16))
        self.assertEqual(tuple(out["information"]["task_latent"].shape), (5, 7, 16))
        self.assertEqual(tuple(out["information"]["flow_relation_latent"].shape), (5, 4, 16))
        self.assertTrue(validate_encoder_contract_checks(model, out, self.tensor, self.graph)["passed"])

    def test_type_specific_encoders_and_grus_do_not_share_parameters(self):
        model = self.model()
        encoders = [model.physical_node_encoder, model.agent_encoder, model.task_encoder, model.physical_relation_encoder, model.comm_relation_encoder, model.logical_flow_encoder, model.carrying_encoder, model.flow_fuse_encoder]
        self.assertEqual(len({id(next(module.parameters())) for module in encoders}), len(encoders))
        grus = [model.physical_gru, model.agent_gru, model.task_gru, model.flow_gru]
        self.assertEqual(len({id(next(module.parameters())) for module in grus}), 4)
        names = dict(model.named_modules())
        self.assertNotIn("physical_relation_gru", names)
        self.assertNotIn("comm_gru", names)

    def test_missing_placeholder_is_ignored_but_real_zero_is_not_missing(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        candidates = np.argwhere(~changed["entity_feature_mask"] & changed["entity_presence"][..., None])
        self.assertGreater(len(candidates), 0)
        idx = tuple(int(x) for x in candidates[0])
        changed["entity_raw_features"][idx] = 999999.0
        same = model(changed, self.graph, validate_inputs=False)
        self.assertTrue(torch.equal(baseline["physical"]["node_latent"], same["physical"]["node_latent"]))
        changed["entity_feature_mask"][idx] = True
        changed["entity_raw_features"][idx] = 0.0
        real_zero = model(changed, self.graph, validate_inputs=False)
        self.assertFalse(torch.equal(baseline["physical"]["node_latent"], real_zero["physical"]["node_latent"]))

    def test_presence_gated_history_and_temporal_causality(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        valid = np.argwhere(changed["entity_presence"][:, 0, :] & changed["entity_position_mask"][:, 0, :, :].all(axis=-1))[0]
        bi, ei = map(int, valid)
        changed["entity_position_raw"][bi, 0, ei, 0] += 100.0
        causal = model(changed, self.graph, validate_inputs=False)
        self.assertFalse(torch.equal(baseline["physical"]["node_latent"][bi, ei], causal["physical"]["node_latent"][bi, ei]))
        absent_base = copy.deepcopy(self.tensor)
        bi2, ei2 = map(int, valid)
        absent_base["entity_presence"][bi2, 0, ei2] = False
        absent = copy.deepcopy(absent_base)
        absent["entity_position_raw"][bi2, 0, ei2, :] = 777777.0
        absent["entity_raw_features"][bi2, 0, ei2, :] = 777777.0
        ignored_base = model(absent_base, self.graph, validate_inputs=False)
        ignored = model(absent, self.graph, validate_inputs=False)
        self.assertTrue(torch.equal(ignored_base["physical"]["node_latent"], ignored["physical"]["node_latent"]))

    def test_future_target_counterfactual_does_not_change_output(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        for key, value in list(changed.items()):
            if key.startswith("target_") and isinstance(value, np.ndarray):
                changed[key] = (~value) if value.dtype == bool else value + 991
        counterfactual = model(changed, self.graph, validate_inputs=False)
        self.assertEqual(output_semantic_digest(baseline), output_semantic_digest(counterfactual))

    def test_flow_excludes_delivered_epoch_and_indices_from_learned_input(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        changed["logical_flow_raw_features"][..., 1] += 999.0
        changed["logical_flow_features"][..., 1] -= 333.0
        changed["logical_flow_epoch"] += 9
        changed["logical_flow_index"] += 17
        out = model(changed, self.graph, validate_inputs=False)
        self.assertTrue(torch.equal(baseline["information"]["flow_relation_latent"], out["information"]["flow_relation_latent"]))

    def test_flow_processor_uses_logical_not_carrying_hop_endpoints(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        hop_changed = copy.deepcopy(self.graph)
        active = np.argwhere(hop_changed["blocks"]["flow_carrying_state"]["active"])
        self.assertGreater(len(active), 0)
        bi, fi = map(int, active[0])
        hop_changed["blocks"]["flow_carrying_state"]["hop_destination_index"][bi, fi] = 0
        hop_out = model(self.tensor, hop_changed, validate_inputs=False)
        self.assertTrue(torch.equal(baseline["information"]["flow_relation_latent"], hop_out["information"]["flow_relation_latent"]))
        logical_changed = copy.deepcopy(self.graph)
        destination = logical_changed["blocks"]["flow_relations"]["destination_index"]
        destination[bi, fi] = (int(destination[bi, fi]) + 1) % self.tensor["entity_presence"].shape[2]
        logical_out = model(self.tensor, logical_changed, validate_inputs=False)
        self.assertFalse(torch.equal(baseline["information"]["flow_relation_latent"][bi, fi], logical_out["information"]["flow_relation_latent"][bi, fi]))

    def test_carrying_is_one_fused_side_branch_not_second_relation(self):
        model = self.model().eval()
        out = model(self.tensor, self.graph)
        self.assertIn("flow_relation_latent", out["information"])
        self.assertNotIn("carrying_relation_latent", out["information"])
        self.assertIsNot(model.logical_flow_encoder, model.carrying_encoder)
        self.assertIsNot(model.logical_flow_encoder, model.flow_fuse_encoder)
        self.assertIn("flow_carrying_state", out["structural"])

    def test_direction_contract_and_family_specific_processors(self):
        model = self.model()
        self.assertTrue(model.config.enable_reverse_comm)
        self.assertTrue(model.config.enable_reverse_flow)
        self.assertTrue(model.config.enable_reverse_task_agent)
        self.assertFalse(model.config.enable_reverse_dag)
        processors = list(model.relation_processors.values())
        self.assertEqual(len({id(next(module.parameters())) for module in processors}), 5)
        self.assertEqual(model.config.aggregator, "masked_mean")
        self.assertEqual(model.config.update_rule, "residual_layernorm")

    def test_architecture_tamper_rejects_shared_encoder_comm_gru_and_shortcut(self):
        shared = self.model()
        shared.agent_encoder = shared.physical_node_encoder
        self.assertFalse(validate_encoder_architecture_checks(shared)["passed"])
        temporal_comm = self.model()
        temporal_comm.comm_gru = torch.nn.GRUCell(self.config.d_h, self.config.d_h)
        self.assertFalse(validate_encoder_architecture_checks(temporal_comm)["passed"])
        shortcut = self.model()
        shortcut.agent_to_physical = torch.nn.Linear(self.config.d_h, self.config.d_h)
        self.assertFalse(validate_encoder_architecture_checks(shortcut)["passed"])

    def test_reverse_direction_contract_tamper_is_rejected(self):
        for field in ("enable_reverse_comm", "enable_reverse_flow", "enable_reverse_task_agent"):
            changed = PIJointGraphEncoder(replace(self.config, **{field: False}), self.tensor["contract"], self.graph["contract"], self.stats)
            self.assertFalse(validate_encoder_architecture_checks(changed)["passed"])
        with self.assertRaises(ValueError):
            replace(self.config, enable_reverse_dag=True)

    def test_p2a_p2c_masks_gates_and_no_matching_physical_edge_requirement(self):
        model = self.model().eval()
        out = model(self.tensor, self.graph)
        self.assertTrue(torch.all((out["diagnostics"]["p2a_gate"] >= 0) & (out["diagnostics"]["p2a_gate"] <= 1)))
        self.assertTrue(torch.all((out["diagnostics"]["p2c_gate"] >= 0) & (out["diagnostics"]["p2c_gate"] <= 1)))
        comm_type = torch.as_tensor(self.graph["blocks"]["comm_relations"]["relation_type_index"])
        self.assertTrue(torch.all(out["diagnostics"]["p2c_message"][comm_type == 3] == 0))
        invalid = ~torch.as_tensor(self.graph["blocks"]["geo_comm_relations"]["validity"])
        self.assertTrue(torch.all(out["diagnostics"]["p2c_message"][invalid] == 0))
        changed_graph = copy.deepcopy(self.graph)
        changed_graph["blocks"]["physical_relations"]["presence"][:] = False
        changed_graph["blocks"]["physical_relations"]["validity"][:] = False
        no_edges = model(self.tensor, changed_graph, validate_inputs=False)
        self.assertTrue(torch.equal(out["diagnostics"]["p2c_initial_message"], no_edges["diagnostics"]["p2c_initial_message"]))

    def test_padding_object_does_not_influence_active_latents(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        padding = np.argwhere(~changed["task_presence"])
        self.assertGreater(len(padding), 0)
        bi, hi, ti = map(int, padding[0])
        changed["task_raw_features"][bi, hi, ti, :] = 123456.0
        changed["task_history_extended_raw_features"][bi, hi, ti, :] = 123456.0
        out = model(changed, self.graph, validate_inputs=False)
        active = torch.as_tensor(self.graph["blocks"]["task_nodes"]["presence"])
        self.assertTrue(torch.equal(baseline["information"]["task_latent"][active], out["information"]["task_latent"][active]))

    def test_entity_index_permutation_is_equivariant(self):
        model = self.model().eval()
        baseline = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        n_entity = changed["entity_presence"].shape[2]
        permutation = np.arange(n_entity - 1, -1, -1)
        old_to_new = np.empty(n_entity, dtype=np.int64)
        old_to_new[permutation] = np.arange(n_entity)
        for key in ("entity_raw_features", "entity_features", "entity_feature_mask", "entity_presence", "entity_type_index", "entity_position_raw", "entity_position", "entity_position_mask"):
            changed[key] = changed[key][:, :, permutation].copy()
        for key in ("agent_cpu_capacity_raw", "agent_cpu_capacity", "agent_cpu_capacity_mask"):
            changed[key] = changed[key][:, permutation].copy()
        for key in ("comm_source_index", "comm_target_index", "task_agent_agent_index", "logical_flow_source_index", "logical_flow_destination_index", "carrying_holder_index", "carrying_hop_source_index", "carrying_hop_destination_index", "route_node_indices"):
            values = changed[key]
            mask = values >= 0
            values[mask] = old_to_new[values[mask]]
        n_task = changed["task_presence"].shape[2]
        task_permutation = np.arange(n_task - 1, -1, -1)
        task_old_to_new = np.empty(n_task, dtype=np.int64)
        task_old_to_new[task_permutation] = np.arange(n_task)
        for key in ("task_raw_features", "task_features", "task_feature_mask", "task_presence", "task_lifecycle_index", "task_history_extended_raw_features", "task_history_extended_features", "task_history_extended_feature_mask"):
            changed[key] = changed[key][:, :, task_permutation].copy()
        for key in ("task_agent_task_index", "logical_flow_task_index"):
            values = changed[key]
            mask = values >= 0
            values[mask] = task_old_to_new[values[mask]]
        mask = changed["dag_edges"] >= 0
        changed["dag_edges"][mask] = task_old_to_new[changed["dag_edges"][mask]]
        n_flow = changed["logical_flow_presence"].shape[2]
        flow_permutation = np.arange(n_flow - 1, -1, -1)
        for key in ("logical_flow_known_mask", "logical_flow_presence", "logical_flow_index", "logical_flow_task_index", "logical_flow_type_index", "logical_flow_status_index", "logical_flow_epoch", "logical_flow_source_index", "logical_flow_destination_index", "logical_flow_raw_features", "logical_flow_features", "logical_flow_feature_mask", "carrying_known_mask", "carrying_active", "carrying_route_revision", "carrying_current_hop_index", "carrying_holder_index", "carrying_hop_source_index", "carrying_hop_destination_index", "carrying_raw_features", "carrying_features", "carrying_feature_mask", "route_node_indices", "route_node_mask"):
            changed[key] = changed[key][:, :, flow_permutation].copy()
        flow_old_to_new = np.empty(n_flow, dtype=np.int64)
        flow_old_to_new[flow_permutation] = np.arange(n_flow)
        for static in changed["sample_static"]:
            indices = static["input_entity_index"]
            for object_id, old_index in list(indices["physical"].items()):
                indices["physical"][object_id] = int(old_to_new[old_index])
            for object_id, old_index in list(indices["task"].items()):
                indices["task"][object_id] = int(task_old_to_new[old_index])
            for object_id, old_index in list(indices["logical_flow"].items()):
                indices["logical_flow"][object_id] = int(flow_old_to_new[old_index])
        permuted_graph = build_typed_dual_graph_batch(changed, PhysicalTopologyConfig())
        actual = model(changed, permuted_graph)
        self.assertTrue(torch.allclose(baseline["physical"]["node_latent"][:, permutation], actual["physical"]["node_latent"], atol=1e-6))
        self.assertTrue(torch.allclose(baseline["information"]["agent_latent"][:, permutation], actual["information"]["agent_latent"], atol=1e-6))
        self.assertTrue(torch.allclose(baseline["information"]["task_latent"][:, task_permutation], actual["information"]["task_latent"], atol=1e-6))
        self.assertTrue(torch.allclose(baseline["information"]["flow_relation_latent"][:, flow_permutation], actual["information"]["flow_relation_latent"], atol=1e-6))

    def test_deterministic_initialization_and_forward(self):
        a, b = self.model().eval(), self.model().eval()
        self.assertEqual(output_semantic_digest(a(self.tensor, self.graph)), output_semantic_digest(b(self.tensor, self.graph)))
        for left, right in zip(a.state_dict().values(), b.state_dict().values()):
            self.assertTrue(torch.equal(left, right))

    def test_state_dict_config_round_trip(self):
        model = self.model().eval()
        expected = model(self.tensor, self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pt"
            save_encoder_package(model, path)
            loaded = load_encoder_package(path, self.tensor["contract"], self.graph["contract"]).eval()
            actual = loaded(self.tensor, self.graph)
        self.assertEqual(output_semantic_digest(expected), output_semantic_digest(actual))

    def test_cpu_autograd_smoke_without_optimizer(self):
        model = self.model()
        out = model(self.tensor, self.graph)
        loss = sum(value.sum() for section in (out["physical"], out["information"]) for key, value in section.items() if key.endswith("latent"))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_receipt_tamper_forces_failure(self):
        required = {name: True for name in REQUIRED_ACCEPTANCE_CHECKS}
        required["future_target_isolation"] = False
        receipt = {"required_checks": required, "passed": True, "scope": {name: False for name in ("rssm", "world_model", "future_action", "prediction", "loss", "planner", "training", "optimizer", "gpu", "locked_test", "formal_dataset")}}
        self.assertFalse(validate_encoder_acceptance(receipt)["passed"])


if __name__ == "__main__":
    unittest.main()
