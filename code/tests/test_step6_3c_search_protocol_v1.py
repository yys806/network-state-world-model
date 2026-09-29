"""Search-independent prefix and B_WM accounting contract tests."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "tests"))

from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_1_trained_candidate_rollout_v1 import OneStepRolloutResult
from pi_jwm.step6_3b_candidate_grammar_v1 import CommBlockChoice, StructuredStepChoice, bind_structured_step
from pi_jwm.step6_3c_search_protocol_v1 import (
    SearchNode, TransitionBudgetAccountant, advance_one_step,
)


class ProtocolTests(unittest.TestCase):
    def test_cache_key_and_budget_use_candidate_steps_not_python_calls(self):
        context, domain, state, control, catalog = fixture()
        bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                     control, catalog)
        latent = {"z": torch.zeros((1, 2))}
        graph = {"g": torch.zeros((1, 1))}
        node = SearchNode.from_anchor(latent, state, graph, control,
                                      context.causal_provenance)
        budget = TransitionBudgetAccountant(1)
        calls = []

        def transition():
            calls.append(True)
            return OneStepRolloutResult(latent, state, graph, control, {}, {}, {},
                                        node.fingerprints, node.fingerprints)

        budget.proposed(True)
        result, hit = budget.evaluate(node, bound, transition)
        self.assertFalse(hit)
        child = node.advance(bound, result, cache_hit=hit)
        self.assertEqual(child.depth, 1)
        self.assertEqual(child.evaluated_transition_count, 1)
        budget.proposed(True)
        _, hit = budget.evaluate(node, bound, transition)
        self.assertTrue(hit)
        self.assertEqual(len(calls), 1)
        other = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 1, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                     control, catalog)
        budget.proposed(True)
        with self.assertRaisesRegex(RuntimeError, "B_WM_EXHAUSTED"):
            budget.evaluate(node, other, transition)
        budget.proposed(False)
        budget.complete()
        budget.dead_end()
        receipt = budget.receipt()
        self.assertEqual(receipt["N_proposed_steps"], 4)
        self.assertEqual(receipt["N_admitted_steps"], 3)
        self.assertEqual(receipt["N_rejected_steps"], 1)
        self.assertEqual(receipt["N_unique_transition_evals"], 1)
        self.assertEqual(receipt["N_cache_hits"], 1)
        self.assertEqual(receipt["N_complete_sequences"], 1)
        self.assertEqual(receipt["N_dead_end_branches"], 1)
        self.assertIn("wall_clock_seconds_diagnostic_only", receipt)

    def test_parent_fingerprint_is_part_of_cache_identity(self):
        context, domain, state, control, catalog = fixture()
        bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                     control, catalog)
        node = SearchNode.from_anchor({"z": torch.zeros((1, 2))}, state,
                                      {"g": torch.zeros((1, 1))}, control,
                                      context.causal_provenance)
        changed = SearchNode.from_anchor({"z": torch.ones((1, 2))}, state,
                                         {"g": torch.zeros((1, 1))}, control,
                                         context.causal_provenance)
        budget = TransitionBudgetAccountant(2)
        self.assertNotEqual(budget.cache_key(node, bound), budget.cache_key(changed, bound))
        node.latent["z"][0, 0] = 9.0
        with self.assertRaisesRegex(ValueError, "mutated after fingerprinting"):
            budget.cache_key(node, bound)

    def test_batch_forward_call_charges_each_unique_candidate_step(self):
        context, domain, state, control, catalog = fixture()
        bound0 = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                      control, catalog)
        bound1 = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 1, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                      control, catalog)
        latent = {"z": torch.zeros((1, 2))}
        graph = {"g": torch.zeros((1, 1))}
        node = SearchNode.from_anchor(latent, state, graph, control,
                                      context.causal_provenance)
        budget = TransitionBudgetAccountant(2)
        for _ in range(3):
            budget.proposed(True)
        calls = []

        def batch(requests):
            calls.append(len(requests))
            return [OneStepRolloutResult(latent, state, graph, control, {}, {}, {},
                                         n.fingerprints, n.fingerprints) for n, _ in requests]

        outcomes = budget.evaluate_batch(((node, bound0), (node, bound1), (node, bound0)), batch)
        self.assertEqual(calls, [2])
        self.assertEqual([hit for _, hit in outcomes], [False, False, True])
        self.assertEqual(budget.n_unique_transition_evals, 2)
        self.assertEqual(budget.n_cache_hits, 1)
        self.assertEqual(budget.n_admitted_steps, 3)

    def test_shared_advance_interface_updates_prefix_and_budget(self):
        context, domain, state, control, catalog = fixture()
        bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                              "SCALE_1.0", "PROFILE_HOLD"), context, domain, state,
                                     control, catalog)
        latent = {"z": torch.zeros((1, 2))}
        graph = {"g": torch.zeros((1, 1))}
        node = SearchNode.from_anchor(latent, state, graph, control,
                                      context.causal_provenance)
        output = OneStepRolloutResult(latent, state, graph, control, {}, {}, {},
                                      node.fingerprints, node.fingerprints)
        budget = TransitionBudgetAccountant(1)
        with patch("pi_jwm.step6_3c_search_protocol_v1.rollout_one_step", return_value=output) as step:
            child = advance_one_step(object(), context, domain, node, bound, budget)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(child.depth, 1)
            self.assertEqual(child.action_prefix, (bound.action,))
            self.assertEqual(child.structural_signature_prefix, (bound.structural_signature,))
            self.assertEqual(budget.n_unique_transition_evals, 1)


if __name__ == "__main__":
    unittest.main()
