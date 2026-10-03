"""CPU oracles for full-cohort coverage, reuse and objective attribution."""
import unittest
from pi_jwm.step6_4d_full_cohort_b512_v1 import plan_cases, component_analysis, require_same_sources


class ExpansionTests(unittest.TestCase):
    def test_coverage_reuses_exact_48_and_rejects_unknown_or_duplicate(self):
        anchors = [f'a{i}' for i in range(64)]
        reused = {(a, s) for a in anchors[:16] for s in (6311, 6312, 6313)}
        self.assertEqual(len(plan_cases(anchors, reused)), 272)
        self.assertTrue(all(s in range(6311, 6316) for a, s in plan_cases(anchors, reused)))
        for bad in [reused | {('unknown', 6311)}, reused | {('a0', 6301)}]:
            with self.assertRaises(ValueError): plan_cases(anchors, bad)
        with self.assertRaises(ValueError): plan_cases(anchors[:-1] + ['a0'], reused)

    def test_first_difference_respects_priority_and_none(self):
        base = [0, 0, 0, 0, 0]
        left, right = {}, {}
        for i in range(5):
            bad = base.copy(); bad[i] = 1
            left[(f'loss{i}', 6311)] = bad; right[(f'loss{i}', 6311)] = base
            left[(f'win{i}', 6311)] = base; right[(f'win{i}', 6311)] = bad
        left[('equal', 6311)] = base; right[('equal', 6311)] = base
        left[('none', 6311)] = None; right[('none', 6311)] = base
        # Later components cannot reverse an earlier, strict improvement.
        left[('priority', 6311)] = [0, 0, 0, 999, 999]
        right[('priority', 6311)] = [1, 0, 0, 0, 0]
        v = component_analysis(left, right)
        self.assertEqual(v['PRIMARY_OBJECTIVE_DEGRADATION_COUNT'], 3)
        self.assertEqual(v['BURDEN_EFFORT_ONLY_DEGRADATION_COUNT'], 2)
        self.assertEqual(v['all_equal'], 1)
        self.assertEqual(v['components']['N_DDL'], {'B512_better': 2, 'B1024_better': 1})
        self.assertEqual(v['both_scoreable_count'], 12)
        right.pop(('none', 6311))
        with self.assertRaises(ValueError): component_analysis(left, right)

    def test_reuse_requires_every_historical_source_byte(self):
        require_same_sources({'solver': 'a', 'scorer': 'b'}, {'solver': 'a', 'scorer': 'b', 'new_runner': 'c'})
        for current in [{'solver': 'wrong', 'scorer': 'b'}, {'solver': 'a'}]:
            with self.assertRaises(ValueError): require_same_sources({'solver': 'a', 'scorer': 'b'}, current)

    def test_nonfinite_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            component_analysis({('a', 6311): [0, 0, float('nan'), 0, 0]},
                               {('a', 6311): [0, 0, 0, 0, 0]})


if __name__ == '__main__': unittest.main()
