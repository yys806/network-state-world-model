import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DecisionAlignedTargetsTest(unittest.TestCase):
    def test_targets_use_ranked_default_sse_and_masked_oracle(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.asarray(
                [[100.0, 80.0, 20.0], [9.0, 9.0, 12.0]], dtype=np.float32
            ),
            active_count=np.asarray([2, 1]),
            default_index=1,
        )
        mask = np.asarray([[True, True, True], [True, True, False]])

        targets = build_decision_aligned_targets(outcome, mask, weight_cap=5.0)

        np.testing.assert_allclose(targets.candidate_benefit[0], [-20.0, 0.0, 60.0])
        self.assertTrue(np.isnan(targets.candidate_benefit[1, 2]))
        np.testing.assert_allclose(targets.opportunity, [60.0, 0.0])
        self.assertEqual(targets.benefit_scale, 60.0)
        self.assertEqual(targets.positive_opportunity.tolist(), [True, False])

    def test_zero_active_rows_are_audited_but_not_trainable(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.zeros((1, 2), dtype=np.float32),
            active_count=np.zeros((1,), dtype=np.int64),
            default_index=1,
        )

        targets = build_decision_aligned_targets(
            outcome, np.ones((1, 2), dtype=bool)
        )

        self.assertFalse(targets.valid_sample[0])
        self.assertEqual(targets.sample_weight[0], 0.0)

    def test_weight_cap_limits_high_gain_outlier(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.asarray([[9.0, 10.0], [0.0, 1000.0]], dtype=np.float32),
            active_count=np.ones(2, dtype=np.int64),
            default_index=1,
        )

        targets = build_decision_aligned_targets(
            outcome,
            np.ones((2, 2), dtype=bool),
            weight_cap=5.0,
            benefit_scale=1.0,
        )

        self.assertEqual(float(targets.sample_weight.max()), 5.25)

    def test_default_candidate_must_be_available(self):
        from pi_jwm.v11_objective_aligned_selector import build_decision_aligned_targets
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.ones((1, 2), dtype=np.float32),
            active_count=np.ones(1, dtype=np.int64),
            default_index=1,
        )

        with self.assertRaisesRegex(ValueError, "ranked default"):
            build_decision_aligned_targets(
                outcome, np.asarray([[True, False]], dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
