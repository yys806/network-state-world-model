"""Symbolic versus brute-force canonical candidate counts, CPU only."""
import sys
import unittest
from itertools import combinations_with_replacement
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "tests"))

from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain, count_comm_multisets
from pi_jwm.step6_3b_candidate_grammar_v1 import admit_structured_candidate
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog


class CandidateDomainTests(unittest.TestCase):
    def test_symbolic_comm_multiset_count_equals_bruteforce(self):
        starts = {1: 2, 2: 1, 3: 0}
        for task_count in range(4):
            for widths in ((), (1,), (1, 1), (1, 2), (1, 1, 1), (2, 2, 2, 2)):
                options_by_width = {w: [(task, start, w)
                                        for task in range(task_count) for start in range(starts[w])]
                                    for w in set(widths)}
                unique = set()
                groups = sorted((w, widths.count(w)) for w in set(widths))

                def visit(index, rows):
                    if index == len(groups):
                        if {row[0] for row in rows} == set(range(task_count)):
                            unique.add(tuple(sorted(rows)))
                        return
                    width, multiplicity = groups[index]
                    for selection in combinations_with_replacement(options_by_width[width], multiplicity):
                        visit(index + 1, rows + selection)

                visit(0, ())
                self.assertEqual(count_comm_multisets(task_count, widths, starts), len(unique),
                                 (task_count, widths))

    def test_real_train_catalog_modes_are_lazy_and_bind_through_6_3b(self):
        context, domain, state, control, catalog = fixture()
        candidate_domain = CandidateDomain.from_state(context, domain, state, control, catalog)
        self.assertFalse(candidate_domain.is_empty)
        self.assertGreater(candidate_domain.exact_unique_single_step_count, len(candidate_domain.modes))
        first = next(candidate_domain.iter_bound())
        self.assertTrue(first.support.formal_pool_admitted)
        self.assertEqual(first.action.route, ())
        self.assertEqual(set(first.binding["wireless_task_to_relation"]), {"input"})

    def test_joint_incompatible_state_is_dead_end_not_fallback(self):
        context, domain, state, control, catalog = fixture()
        signature = ("NOOP", "NOOP", "HOLD,HOLD")
        tiny = TrainStructuralSupportCatalog("synthetic", frozenset((signature,)),
            tuple(frozenset((signature[i],)) for i in range(3)),
            (frozenset(((signature,),)), frozenset(), frozenset(), frozenset()),
            frozenset(), frozenset(((0, 1),)))
        candidate_domain = CandidateDomain.from_state(context, domain, state, control, tiny)
        self.assertTrue(candidate_domain.is_empty)
        self.assertEqual(candidate_domain.exact_unique_single_step_count, 0)
        self.assertEqual(candidate_domain.empty_reason,
                         "NO_TRAIN_OBSERVED_JOINT_STRUCTURE_COMPATIBLE")
        self.assertEqual(list(candidate_domain.iter_bound()), [])

    def test_small_full_domain_symbolic_equals_unique_canonical_bound_steps(self):
        context, domain, state, control, _ = fixture()
        signatures = (("rows:1", "tasks:1;alpha:0.5", "HOLD,HOLD"),
                      ("rows:1", "tasks:1;alpha:1.0", "HOLD,HOLD"))
        tiny = TrainStructuralSupportCatalog("synthetic", frozenset(signatures),
            tuple(frozenset(sig[i] for sig in signatures) for i in range(3)),
            (frozenset((sig,) for sig in signatures), frozenset(),
             frozenset(), frozenset()), frozenset(),
            frozenset(((0, 1), (1, 1))))
        candidate_domain = CandidateDomain.from_state(context, domain, state, control, tiny)
        concrete = list(candidate_domain.iter_bound())
        unique_frames = {str(bound.action.frame()) for bound in concrete}
        self.assertEqual(candidate_domain.exact_unique_single_step_count, 4)
        self.assertEqual(len(concrete), len(unique_frames))
        self.assertEqual(len(concrete), candidate_domain.exact_unique_single_step_count)
        for bound in concrete:
            sequence = CandidateActionSequence((bound.action,), "tiny", Backend.SEARCH,
                                               0, context.causal_provenance)
            self.assertTrue(admit_structured_candidate(sequence, context, domain,
                            (state,), tiny).admitted)


if __name__ == "__main__":
    unittest.main()
