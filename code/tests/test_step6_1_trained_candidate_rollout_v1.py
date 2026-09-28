"""Contract tests; real trained-checkpoint evidence is produced by the Step 6.1 runner."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts"), str(ROOT / "code" / "tests")]

from test_step6_0c_planner_action_domain_v1 import fixture
from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, PlannerCandidateContext,
    compile_candidate,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import annotate_domain, rule_fallback_v1
from pi_jwm.step6_1_trained_candidate_rollout_v1 import CandidateRolloutTrace, audit_recursive_link, compile_candidate_step, fingerprint


class TrainedRolloutContractTests(unittest.TestCase):
    def test_each_step_matches_frozen_eleven_tensor_adapter(self):
        base, _, domain = fixture()
        control = rule_fallback_v1(base, domain, 4)
        comp = {"node_id": "edge", "task_id": "task", "allocated_cpu_per_s": 2.5}
        steps = (CandidateActionStep(comp=(comp,), mob=control.steps[0].mob), *control.steps[1:])
        candidate = annotate_domain(CandidateActionSequence(steps, "comp", Backend.RULE_FALLBACK, 7,
            base.causal_provenance), base, domain)
        states = (base.current_state,) * 4
        expected, mappings = compile_candidate(candidate, base, states, planner_domain_context=domain)
        for index in range(4):
            actual, mapping = compile_candidate_step(candidate, base, states[index], index,
                                                       planner_domain_context=domain)
            self.assertEqual(set(actual), set(expected[index]))
            self.assertEqual(len(actual), 11)
            for key in actual:
                self.assertTrue(torch.equal(actual[key], expected[index][key]), (index, key))
            self.assertEqual(mapping, mappings[index])

    def test_fingerprint_detects_recursive_input_replacement(self):
        start = {"x": torch.tensor([[1.0]])}
        next_state = {"x": torch.tensor([[2.0]])}
        self.assertNotEqual(fingerprint(start), fingerprint(next_state))
        self.assertEqual(fingerprint(next_state), fingerprint({"x": next_state["x"].clone()}))
        h0 = {"state": "s0", "graph": "g0", "latent": "z0"}
        h1 = {"state": "s1", "graph": "g1", "latent": "z1"}
        trace = CandidateRolloutTrace("test", h0, (h0, h1), (h1, h1), (), (), (), (), (), ())
        audit_recursive_link(trace, 1)
        with self.assertRaisesRegex(ValueError, "recursive input"):
            audit_recursive_link(trace, 1, substituted_input=h0)


if __name__ == "__main__":
    unittest.main()
