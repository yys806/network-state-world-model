import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11BasePolicyTournamentTest(unittest.TestCase):
    def test_advantage_weighted_scores_emphasize_positive_advantage(self):
        from compare_v11_base_policy_candidates import advantage_weighted_scores

        behavior = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        advantage = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

        scores = advantage_weighted_scores(behavior, advantage, temperature=1.0, max_weight=10.0)

        self.assertLess(float(scores[0]), float(scores[1]))
        self.assertLess(float(scores[1]), float(scores[2]))
        self.assertLessEqual(float(scores[2]), 10.0)

    def test_expectile_baseline_is_pulled_toward_upper_tail(self):
        from compare_v11_base_policy_candidates import expectile_baseline

        values = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        mean = float(np.mean(values))
        baseline = expectile_baseline(values, expectile=0.8, iterations=200)

        self.assertGreater(baseline, mean)
        self.assertLess(baseline, 10.0)

    def test_candidate_family_keeps_identity_first(self):
        from compare_v11_base_policy_candidates import candidate_families

        families = candidate_families()

        self.assertEqual(families[0], 'identity')
        self.assertIn('ranked_rf', families)
        self.assertIn('awr_selector', families)
        self.assertIn('iql_expectile_selector', families)


if __name__ == '__main__':
    unittest.main()
