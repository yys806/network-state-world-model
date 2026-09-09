from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _stats() -> dict:
    widths = {"node": 7, "physical_edge": 5, "flow": 5, "task": 8}
    return {"features": {f"{name}_state": {"mean": [0.0] * width, "scale": [1.0] * width} for name, width in widths.items()}}


def _rule_inputs(task_count: int = 2) -> tuple[dict, dict]:
    previous = {
        "node": torch.zeros(1, 2, 7),
        "physical_edge": torch.zeros(1, 1, 5),
        "flow": torch.zeros(1, 2, 5),
        "task": torch.zeros(1, task_count, 8),
    }
    previous["node"][..., 5] = torch.tensor([[10.0, 10.0]])
    previous["flow"][0, :, 0:2] = 1.0
    previous["task"][0, :, 0] = 1.0
    previous["task"][0, :, 1] = 0.5
    previous["task"][0, :, 2] = 0.8
    previous["task"][0, :, 3] = 1.0
    return previous, {name: value.clone() + 9.0 for name, value in previous.items()}


def _static(*, flow_tasks, flow_types, dag_index=None, dag_valid=None) -> dict:
    return {
        "physical_edge_endpoint_index": torch.tensor([[[0, 1]]]),
        "flow_task_index": torch.tensor([flow_tasks]),
        "flow_type_index": torch.tensor([flow_types]),
        "flow_endpoint_index": torch.tensor([[[0, 1], [0, 1]]]),
        "flow_valid": torch.tensor([[index >= 0 for index in flow_tasks]]),
        "task_valid": torch.tensor([[True, True]]),
        "dag_edge_index": dag_index if dag_index is not None else torch.tensor([[[-1], [-1]]]),
        "dag_edge_valid": dag_valid if dag_valid is not None else torch.tensor([[False]]),
        "slot_seconds": torch.tensor([0.1]),
    }


class DeterministicRuleLayerV1Tests(unittest.TestCase):
    def test_rollout_audit_accepts_rule_layer_step_with_conserved_flow_and_cpu(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer
        try:
            from pi_jwm.formal_rule_rollout_audit_v1 import audit_rule_step
        except ModuleNotFoundError:
            audit_rule_step = None

        self.assertIsNotNone(audit_rule_step, "rollout rule-step audit is not implemented")

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action=torch.tensor([[[1.0, 1.0, 0.0, 2.0, 0.5, 1.0, 0.0, 1.0], [0.0] * 8]]),
            action_present=torch.tensor([[True, False]]),
            source_node_index=torch.tensor([[[0, -1, 0, -1], [-1] * 4]]),
            target_node_index=torch.tensor([[[1, -1, 1, -1], [-1] * 4]]),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[0, 0]]),
            static=_static(flow_tasks=[0, 1], flow_types=[0, 0]),
            service_outcome={"edge_rate": torch.tensor([[3.0]]), "flow_delivered": torch.tensor([[0.25, 0.0]])},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )

        report = audit_rule_step(
            result=result,
            previous_states=previous,
            static=_static(flow_tasks=[0, 1], flow_types=[0, 0]),
            n_rb=4,
        )

        self.assertTrue(report["passed"])
        self.assertEqual([], report["violations"])

    def test_created_flow_resets_unobserved_slot_cumulative_data(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer
        from pi_jwm.formal_rule_rollout_audit_v1 import audit_rule_step

        stats = _stats()
        stats["features"]["flow_state"]["mean"][2] = 0.25
        stats["features"]["flow_state"]["mean"][4] = 7.0
        layer = DeterministicRuleLayer(stats, n_rb=4)
        previous_physical, proposed_physical = _rule_inputs()
        previous = {
            name: layer.normalization.to_normalized(name, value)
            for name, value in previous_physical.items()
        }
        proposed = {
            name: layer.normalization.to_normalized(name, value)
            for name, value in proposed_physical.items()
        }
        # A zero normalized padding slot maps to the train-set mean in physical units.
        previous["flow"].zero_()
        static = _static(flow_tasks=[0, 1], flow_types=[0, 0])
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action=torch.tensor([[[1.0, 1.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 8]]),
            action_present=torch.tensor([[True, False]]),
            source_node_index=torch.tensor([[[0, -1, 0, -1], [-1] * 4]]),
            target_node_index=torch.tensor([[[1, -1, 1, -1], [-1] * 4]]),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[0, 0]]),
            static=static,
            service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.tensor([[0.25, 0.0]])},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )

        self.assertAlmostEqual(0.25, result.states["flow"][0, 0, 2].item())
        self.assertAlmostEqual(0.1, result.states["flow"][0, 0, 4].item())
        report = audit_rule_step(result=result, previous_states={name: layer.normalization.to_physical(name, value) for name, value in previous.items()}, static=static, n_rb=4)
        self.assertTrue(report["passed"])

    def test_rollout_audit_summary_reports_every_rule_step_and_violation_counts(self):
        from pi_jwm import formal_rule_rollout_audit_v1 as audit_module

        summarize = getattr(audit_module, "summarize_rule_rollout_reports", None)
        self.assertIsNotNone(summarize, "rollout audit summarizer is not implemented")
        report = summarize(
            [
                {"passed": True, "violations": [], "details": {"active_flow_count": 2}},
                {"passed": False, "violations": ["cpu_capacity"], "details": {"active_flow_count": 1}},
            ],
            expected_step_count=2,
        )

        self.assertFalse(report["audit_passed"])
        self.assertEqual(2, report["observed_step_count"])
        self.assertEqual({"cpu_capacity": 1}, report["violation_counts"])

    def test_rollout_audit_captures_each_model_rule_step(self):
        from pi_jwm import formal_rule_rollout_audit_v1 as audit_module
        from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
        from test_formal_dual_graph_world_model_v1 import fake_formal_batch

        audit_rollout = getattr(audit_module, "audit_rule_rollout", None)
        self.assertIsNotNone(audit_rollout, "model rule-step capture is not implemented")
        batch = fake_formal_batch()
        batch["history"]["flow_state"].zero_()
        batch["history"]["flow_state"][:, -1, :, 0:2] = 1.0
        batch["history"]["flow_present"][:, -1].fill_(True)
        batch["history"]["task_state"].zero_()
        batch["history"]["task_state"][:, -1, :, 0] = 1.0
        batch["history"]["task_state"][:, -1, :, 2:4] = 1.0
        batch["history"]["task_lifecycle_index"][:, -1].zero_()
        batch["future_action"]["task_action_source_node_index"] = torch.full((2, 2, 3, 4), -1, dtype=torch.long)
        batch["static"]["flow_task_index"] = torch.tensor([[0, 0], [0, 0]])
        batch["static"]["flow_type_index"] = torch.zeros(2, 2, dtype=torch.long)
        batch["static"]["flow_valid"] = torch.ones(2, 2, dtype=torch.bool)
        batch["static"]["slot_seconds"] = torch.tensor([0.1, 0.1])
        model = FormalDualGraphWorldModel(FormalWorldModelConfig(
            mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2,
            deterministic_rule_layer=True, rule_layer_stats=_stats(), n_rb=4,
        ))

        _, report = audit_rollout(model, batch, n_rb=4)

        self.assertTrue(report["audit_passed"])
        self.assertEqual(2, report["observed_step_count"])

    def test_rollout_audit_converts_hooked_normalized_previous_states_to_physical_units(self):
        from pi_jwm import formal_rule_rollout_audit_v1 as audit_module
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer

        stats = _stats()
        stats["features"]["node_state"]["mean"][5] = 100.0
        stats["features"]["node_state"]["scale"][5] = 10.0
        layer = DeterministicRuleLayer(stats, n_rb=4)
        previous_physical, proposed_physical = _rule_inputs()
        previous_normalized = {
            name: layer.normalization.to_normalized(name, value)
            for name, value in previous_physical.items()
        }
        proposed_normalized = {
            name: layer.normalization.to_normalized(name, value)
            for name, value in proposed_physical.items()
        }
        static = _static(flow_tasks=[-1, -1], flow_types=[-1, -1])

        class _OneStepModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deterministic_rules = layer
                self.config = type("Config", (), {"horizon_steps": 1})()

            def forward(self, _batch):
                self.deterministic_rules(
                    proposed_normalized,
                    previous_states=previous_normalized,
                    previous_task_present=torch.tensor([[True, True]]),
                    previous_flow_present=torch.tensor([[False, False]]),
                    action=torch.zeros(1, 2, 8),
                    action_present=torch.zeros(1, 2, dtype=torch.bool),
                    source_node_index=torch.full((1, 2, 4), -1),
                    target_node_index=torch.full((1, 2, 4), -1),
                    task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
                    previous_lifecycle_index=torch.tensor([[1, 1]]),
                    static=static,
                    service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.zeros(1, 2)},
                    lifecycle_logits=torch.zeros(1, 2, 5),
                )
                return {}

        _, report = audit_module.audit_rule_rollout(_OneStepModel(), {}, n_rb=4)

        self.assertTrue(report["audit_passed"])
        self.assertEqual({}, report["violation_counts"])

    def test_rollout_audit_accepts_failed_lifecycle_state(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer
        from pi_jwm.formal_rule_rollout_audit_v1 import audit_rule_step

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        previous["task"][..., 3] = 0.0
        static = _static(flow_tasks=[-1, -1], flow_types=[-1, -1])
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action=torch.zeros(1, 2, 8),
            action_present=torch.zeros(1, 2, dtype=torch.bool),
            source_node_index=torch.full((1, 2, 4), -1),
            target_node_index=torch.full((1, 2, 4), -1),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[0, 0]]),
            static=static,
            service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.zeros(1, 2)},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )

        self.assertEqual(4, result.lifecycle_index[0, 0].item())
        report = audit_rule_step(result=result, previous_states=previous, static=static, n_rb=4)

        self.assertTrue(report["passed"])

    def test_rollout_audit_excludes_not_present_tasks_from_dag_readiness_check(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer
        from pi_jwm.formal_rule_rollout_audit_v1 import audit_rule_step

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        static = _static(
            flow_tasks=[-1, -1],
            flow_types=[-1, -1],
            dag_index=torch.tensor([[[0], [1]]]),
            dag_valid=torch.tensor([[True]]),
        )
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, False]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action=torch.zeros(1, 2, 8),
            action_present=torch.zeros(1, 2, dtype=torch.bool),
            source_node_index=torch.full((1, 2, 4), -1),
            target_node_index=torch.full((1, 2, 4), -1),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[3, -1]]),
            static=static,
            service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.zeros(1, 2)},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )

        report = audit_rule_step(result=result, previous_states=previous, static=static, n_rb=4)

        self.assertTrue(report["passed"])

    def test_resolves_only_explicit_action_endpoints(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import resolve_action_endpoints

        source = torch.tensor([[[[0, -1, 2, -1]]]])
        target = torch.tensor([[[[1, -1, 3, -1]]]])
        resolved_source, resolved_target = resolve_action_endpoints(source, target)
        torch.testing.assert_close(resolved_source, source)
        torch.testing.assert_close(resolved_target, target)
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_action_endpoints(torch.tensor([[[0, 1, 2, 3]]]), target)

    def test_flow_service_not_action_occurrence_updates_task_progress(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        previous["flow"][0, 0].zero_()
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action=torch.tensor([[[1.0, 1.0, 0.0, 2.0, 0.5, 1.0, 99.0, 1.0], [0.0] * 8]]),
            action_present=torch.tensor([[True, False]]),
            source_node_index=torch.tensor([[[0, -1, 0, -1], [-1] * 4]]),
            target_node_index=torch.tensor([[[1, -1, 1, -1], [-1] * 4]]),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[0, 0]]),
            static=_static(flow_tasks=[0, 1], flow_types=[0, 0]),
            service_outcome={"edge_rate": torch.tensor([[3.0]]), "flow_delivered": torch.tensor([[0.25, 0.0]])},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )
        self.assertAlmostEqual(0.75, result.states["flow"][0, 0, 1].item())
        self.assertAlmostEqual(0.25, result.states["task"][0, 0, 5].item())
        self.assertNotEqual(1.0, result.states["task"][0, 0, 5].item())
        self.assertEqual(2.0, result.states["physical_edge"][0, 0, 4].item())
        self.assertFalse(result.masks["physical_edge"][0, 0, 2])
        self.assertTrue(result.service_masks["physical_edge"][0, 0, 2])
        self.assertTrue(result.masks["physical_edge"][0, 0, 4])

    def test_cpu_inner_rule_is_capped_work_conserving_and_ignores_logged_cpu_action(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        task_nodes = torch.tensor([[[0, 1, 1, 0], [0, 1, 1, 0]]])
        common = dict(
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[False, False]]),
            action_present=torch.tensor([[True, True]]),
            source_node_index=torch.full((1, 2, 4), -1),
            target_node_index=torch.full((1, 2, 4), -1),
            task_node_index=task_nodes,
            previous_lifecycle_index=torch.tensor([[1, 1]]),
            static=_static(flow_tasks=[-1, -1], flow_types=[-1, -1]),
            service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.zeros(1, 2)},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )
        first = layer(proposed, action=torch.tensor([[[0, 0, 0, 0, 0, 1, 999, 1], [0, 0, 0, 0, 0, 1, 0, 0]]], dtype=torch.float32), **common)
        second = layer(proposed, action=torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 123, 1]]], dtype=torch.float32), **common)
        torch.testing.assert_close(first.cpu_allocation, torch.tensor([[5.0, 5.0]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first.cpu_served, torch.tensor([[0.5, 0.5]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first.cpu_allocation, second.cpu_allocation)
        torch.testing.assert_close(first.states["task"], second.states["task"])
        self.assertAlmostEqual(0.5, first.states["task"][0, 0, 6].item())
        self.assertAlmostEqual(proposed["node"][0, 1, 5].item(), first.states["node"][0, 1, 5].item())
        self.assertFalse(first.masks["node"].any())

    def test_lifecycle_and_dag_release_follow_recursive_conservation(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import DeterministicRuleLayer

        layer = DeterministicRuleLayer(_stats(), n_rb=4)
        previous, proposed = _rule_inputs()
        previous["task"][0, 0, 5] = 0.9
        previous["task"][0, 0, 2] = 2.0
        result = layer(
            proposed,
            previous_states=previous,
            previous_task_present=torch.tensor([[True, True]]),
            previous_flow_present=torch.tensor([[True, True]]),
            action=torch.zeros(1, 2, 8),
            action_present=torch.zeros(1, 2, dtype=torch.bool),
            source_node_index=torch.full((1, 2, 4), -1),
            target_node_index=torch.full((1, 2, 4), -1),
            task_node_index=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]]),
            previous_lifecycle_index=torch.tensor([[0, 0]]),
            static=_static(flow_tasks=[0, 1], flow_types=[0, 0], dag_index=torch.tensor([[[0], [1]]]), dag_valid=torch.tensor([[True]])),
            service_outcome={"edge_rate": torch.zeros(1, 1), "flow_delivered": torch.tensor([[0.1, 1.0]])},
            lifecycle_logits=torch.zeros(1, 2, 5),
        )
        self.assertEqual(1, result.lifecycle_index[0, 0].item())
        self.assertEqual(0, result.lifecycle_index[0, 1].item())
        self.assertEqual(0.0, result.dag_state[0, 1, 2].item())

    def test_normalized_physical_round_trip_is_exact(self):
        from pi_jwm.formal_deterministic_rule_layer_v1 import NormalizationAdapter

        adapter = NormalizationAdapter(_stats())
        value = torch.randn(2, 3, 7)
        torch.testing.assert_close(adapter.to_normalized("node", adapter.to_physical("node", value)), value)

    def test_model_current_step_output_contains_rule_update(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
        from test_formal_dual_graph_world_model_v1 import fake_formal_batch

        batch = fake_formal_batch()
        batch["history"]["flow_state"].zero_()
        batch["history"]["flow_state"][:, -1, :, 0:2] = 1.0
        batch["history"]["flow_present"][:, -1].fill_(True)
        batch["history"]["task_state"].zero_()
        batch["history"]["task_state"][:, -1, :, 0] = 1.0
        batch["history"]["task_state"][:, -1, :, 2:4] = 1.0
        batch["history"]["task_lifecycle_index"][:, -1].zero_()
        batch["future_action"]["task_action_source_node_index"] = torch.full((2, 2, 3, 4), -1, dtype=torch.long)
        batch["static"]["flow_task_index"] = torch.tensor([[0, 0], [0, 0]])
        batch["static"]["flow_type_index"] = torch.zeros(2, 2, dtype=torch.long)
        batch["static"]["flow_valid"] = torch.ones(2, 2, dtype=torch.bool)
        batch["static"]["slot_seconds"] = torch.tensor([0.1, 0.1])
        model = FormalDualGraphWorldModel(FormalWorldModelConfig(mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2, deterministic_rule_layer=True, rule_layer_stats=_stats(), n_rb=4))
        torch.nn.init.zeros_(model.flow_service_head.weight)
        torch.nn.init.constant_(model.flow_service_head.bias, math.log(0.25 / 0.75))
        output = model(batch)
        torch.testing.assert_close(output["flow_state_mean"][:, 0, :, 1], torch.full((2, 2), 0.75), atol=1e-5, rtol=1e-5)
        self.assertIn("deterministic_flow_mask", output)
        self.assertIn("service_flow_delivered", output)
        output["flow_state_mean"].square().mean().backward()
        self.assertIsNotNone(model.flow_service_head.weight.grad)
        self.assertGreater(model.flow_service_head.weight.grad.abs().sum().item(), 0.0)

    def test_model_feedback_contains_only_rule_correction_not_full_state(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
        from test_formal_dual_graph_world_model_v1 import fake_formal_batch

        batch = fake_formal_batch()
        batch["future_action"]["task_action_source_node_index"] = torch.full((2, 2, 3, 4), -1, dtype=torch.long)
        batch["static"]["flow_task_index"] = torch.tensor([[0, 0], [0, 0]])
        batch["static"]["flow_type_index"] = torch.zeros(2, 2, dtype=torch.long)
        batch["static"]["slot_seconds"] = torch.tensor([0.1, 0.1])
        model = FormalDualGraphWorldModel(FormalWorldModelConfig(
            mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2,
            deterministic_rule_layer=True, rule_layer_stats=_stats(), n_rb=4,
        ))
        captured = []
        handle = model.state_feedback["node"].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        model(batch)
        handle.remove()
        self.assertEqual(1, len(captured))
        torch.testing.assert_close(captured[0], torch.zeros_like(captured[0]))

    def test_physical_edge_rule_feedback_is_injected_as_gru_input_message(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
        from test_formal_dual_graph_world_model_v1 import fake_formal_batch

        batch = fake_formal_batch()
        batch["future_action"]["task_action_source_node_index"] = torch.full((2, 2, 3, 4), -1, dtype=torch.long)
        batch["static"]["flow_task_index"] = torch.tensor([[0, 0], [0, 0]])
        batch["static"]["flow_type_index"] = torch.zeros(2, 2, dtype=torch.long)
        batch["static"]["slot_seconds"] = torch.tensor([0.1, 0.1])
        model = FormalDualGraphWorldModel(FormalWorldModelConfig(
            mode="coupled_dual_gnn", hidden_dim=8, history_steps=3, horizon_steps=2,
            deterministic_rule_layer=True, rule_layer_stats=_stats(), n_rb=4,
        ))
        normal_transitions, normal_projections = [], []

        normal_transition_handle = model.edge_transition.register_forward_pre_hook(
            lambda _module, inputs: normal_transitions.append(
                tuple(value.detach().clone() for value in inputs)
            )
        )
        projection_handle = model.state_feedback["physical_edge"].register_forward_hook(
            lambda _module, _inputs, output: normal_projections.append(output.detach().clone())
        )
        model(batch)
        projection_handle.remove()
        normal_transition_handle.remove()

        suppressed_transitions = []
        suppressed_transition_handle = model.edge_transition.register_forward_pre_hook(
            lambda _module, inputs: suppressed_transitions.append(
                tuple(value.detach().clone() for value in inputs)
            )
        )
        suppress_projection_handle = model.state_feedback["physical_edge"].register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        model(batch)
        suppress_projection_handle.remove()
        suppressed_transition_handle.remove()

        self.assertEqual(2, len(normal_transitions))
        self.assertEqual(2, len(suppressed_transitions))
        self.assertEqual(1, len(normal_projections))
        projection = normal_projections[0]
        self.assertGreater(projection.abs().max().item(), 0.0)
        normal_input, normal_hidden = normal_transitions[1]
        suppressed_input, suppressed_hidden = suppressed_transitions[1]
        torch.testing.assert_close(normal_hidden, suppressed_hidden)
        torch.testing.assert_close(normal_input - suppressed_input, projection.reshape_as(normal_input))


if __name__ == "__main__":
    unittest.main()
