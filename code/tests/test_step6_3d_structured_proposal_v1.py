"""Small-domain checks of the shared structured proposal path."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/tests"))
from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3d_structured_proposal_v1 import (
    SparseCategorical, StructuredProposalDistribution, sample_uniform_structured_step,
)


class StructuredProposalTests(unittest.TestCase):
    def test_cem_update_equation_and_method_layer_isolation(self):
        for method, expected in (("HRS", set()), ("S-CEM", {"mode", "count"}),
                                 ("MH-CEM", {"mode", "count", "subset", "assignment", "start"})):
            proposal = StructuredProposalDistribution(method)
            for layer in ("mode", "count", "subset", "assignment", "start"):
                table = proposal.table(layer, (layer, "mask"))
                table.sample(random.Random(1), lambda rng: "a")
            before = {key: value.snapshot() for key, value in proposal.tables.items()}
            elite = [{"draws": [((layer, "mask"), "b") for layer in
                                  ("mode", "count", "subset", "assignment", "start")]}]
            proposal.update_from_elites(elite)
            changed = {key[0] for key, value in proposal.tables.items()
                       if value.snapshot() != before[key]}
            self.assertEqual(changed, expected)
        table = SparseCategorical()
        table.update({"b": 1.0})
        self.assertAlmostEqual(table.prior_weight, 0.525)
        self.assertAlmostEqual(table.sparse_weights["b"], 0.475)

    def test_all_samples_are_canonical_members_of_exact_domain(self):
        context, domain, state, control, _ = fixture()
        signatures = (("rows:1,1", "tasks:1;alpha:0.5", "HOLD,HOLD"),)
        tiny = TrainStructuralSupportCatalog("synthetic", frozenset(signatures),
            tuple(frozenset(sig[i] for sig in signatures) for i in range(3)),
            (frozenset((sig,) for sig in signatures), frozenset(),
             frozenset(), frozenset()), frozenset(),
            frozenset(((0, 1), (1, 1))), {"rows:1,1": frozenset({1})})
        current = CandidateDomain.from_state(context, domain, state, control, tiny)
        expected = {str(bound.action.frame()) for bound in current.iter_bound()}
        self.assertGreater(len(expected), 1)
        rng = random.Random(6301)
        observed = set()
        for _ in range(100):
            bound, path = sample_uniform_structured_step(current, rng)
            self.assertIn(str(bound.action.frame()), expected)
            self.assertTrue(bound.support.formal_pool_admitted)
            self.assertEqual(bound.action.route, ())
            self.assertEqual(path["selected_task_count"], 1)
            observed.add(str(bound.action.frame()))
        self.assertEqual(observed, expected)

    def test_predicted_state_mask_removes_unbindable_comm_mode(self):
        context, domain, state, control, _ = fixture()
        signatures = (("rows:1", "tasks:1;alpha:0.5", "HOLD,HOLD"),
                      ("NOOP", "tasks:1;alpha:0.5", "HOLD,HOLD"))
        tiny = TrainStructuralSupportCatalog("synthetic", frozenset(signatures),
            tuple(frozenset(sig[i] for sig in signatures) for i in range(3)),
            (frozenset((sig,) for sig in signatures), frozenset(),
             frozenset(), frozenset()), frozenset(), frozenset(((0, 1),)),
            {"rows:1": frozenset({1}), "NOOP": frozenset({0})})
        predicted = {key: value.clone() for key, value in state.items()}
        predicted["flow_presence"][:] = False
        predicted["carrying_active"][:] = False
        current = CandidateDomain.from_state(context, domain, predicted, control, tiny)
        self.assertEqual([mode.signature[0] for mode in current.modes], ["NOOP"])
        proposal = StructuredProposalDistribution("MH-CEM")
        for _ in range(10):
            bound, _ = sample_uniform_structured_step(current, random.Random(_))
            self.assertEqual(bound.action.comm, ())


if __name__ == "__main__":
    unittest.main()
