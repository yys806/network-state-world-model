"""Exhaustive small H4 oracle for the shared search implementation."""
import random
import sys
import unittest
import subprocess
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/tests"))
from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_1_trained_candidate_rollout_v1 import OneStepRolloutResult, fingerprint
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget


class ExactOracleTests(unittest.TestCase):
    def test_all_three_methods_recover_exhaustive_h4_optimum(self):
        context, operational, state, control, _ = fixture()
        signature = ("rows:1", "tasks:1;alpha:0.5", "HOLD,HOLD")
        catalog = TrainStructuralSupportCatalog("synthetic", frozenset((signature,)),
            tuple(frozenset((signature[i],)) for i in range(3)),
            (frozenset(((signature,),)), frozenset(), frozenset(), frozenset()),
            frozenset(), frozenset(((0, 1), (1, 1))),
            {"rows:1": frozenset({1})})
        domain = CandidateDomain.from_state(context, operational, state, control, catalog)
        self.assertEqual(domain.exact_unique_single_step_count, 2)
        # Exhaustive H4 space is {0,1}^4. The five-part optimum is all zeros.
        expected = min(((sum((number >> bit) & 1 for bit in range(4)), 0., 0., 0., 0.)
                        for number in range(16)))
        latent = {"path": torch.zeros((1, 1), dtype=torch.float32)}
        graph = {"oracle": torch.zeros((1, 1), dtype=torch.float32)}
        fingerprints = {"state": fingerprint(state), "graph": fingerprint(graph),
                        "latent": fingerprint(latent), "anchor": "synthetic_anchor"}
        anchor = SimpleNamespace(context=context, domain=operational, state=state,
                                 graph=graph, latent=latent, fingerprints=fingerprints)

        def transition(node, bound):
            start = int(bound.action.comm[0]["rb_indices"][0])
            next_latent = {"path": node.latent["path"] * 3 + start + 1}
            output = {"latent": fingerprint(next_latent), "state": fingerprint(state),
                      "graph": fingerprint(graph)}
            return OneStepRolloutResult(next_latent, state, graph,
                bound.next_mobility_control, {}, {}, {}, node.fingerprints, output)

        def score(candidate, trace):
            self.assertEqual(len(trace.states), 4)
            self.assertTrue(all(not step.route for step in candidate.steps))
            total_start = sum(int(step.comm[0]["rb_indices"][0]) for step in candidate.steps)
            return SimpleNamespace(objective_tuple=(total_start, 0., 0., 0., 0.),
                                   candidate_fingerprint=candidate.fingerprint)

        for method, config in (("HRS", {}),
                               ("S-CEM", {"iterations": 3, "elite_ratio": 0.1}),
                               ("MH-CEM", {"iterations": 3, "elite_ratio": 0.1})):
            for batch_size in (1, 4):
                with self.subTest(method=method, batch_size=batch_size):
                    result = solve_fixed_budget(method=method, seed=6301, b_wm=30,
                        anchor=anchor, catalog=catalog, transition=transition,
                        transition_batch=(lambda requests: tuple(transition(n, b)
                            for n, b in requests)) if batch_size > 1 else None,
                        batch_size=batch_size, score_h4=score, **config)
                    self.assertEqual(result.best_objective, expected)
                    self.assertEqual(result.winner_sequence.fingerprint, result.best_fingerprint)
                    self.assertEqual(result.winner_first_action, result.winner_sequence.steps[0])
                    self.assertEqual(result.budget_receipt["N_unique_transition_evals"], 30)
                    self.assertEqual(result.h4_scoreable_count, 16)
                    self.assertGreater(result.budget_receipt["N_cache_hits"], 0)
                    name = 'pi_jwm._step64b_previous_solver_oracle'
                    old = types.ModuleType(name)
                    sys.modules[name] = old
                    old_source = subprocess.check_output(['git', 'show',
                        'f9b7c05fb016403b776065074de8d791126661f0:code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py'], cwd=ROOT, text=True)
                    exec(compile(old_source, name, 'exec'), old.__dict__)
                    previous = old.solve_fixed_budget(method=method, seed=6301, b_wm=30,
                        anchor=anchor, catalog=catalog, transition=transition,
                        transition_batch=(lambda requests: tuple(transition(n,b) for n,b in requests)) if batch_size>1 else None,
                        batch_size=batch_size, score_h4=score, **config)
                    left,right=asdict(previous),asdict(result)
                    right.pop('winner_sequence')
                    for row in (left,right):
                        row['budget_receipt'].pop('wall_clock_seconds_diagnostic_only',None)
                    self.assertEqual(left,right)

        failed=solve_fixed_budget(method='MH-CEM',seed=6301,b_wm=16,
            anchor=anchor,catalog=catalog,transition=transition,
            score_h4=lambda candidate,trace:None,iterations=4,elite_ratio=0.1,batch_size=1)
        self.assertIsNone(failed.winner_sequence)
        self.assertIsNone(failed.winner_first_action)
        self.assertIsNone(failed.best_objective)
        self.assertEqual(failed.h4_scoreable_count,0)


if __name__ == "__main__":
    unittest.main()
