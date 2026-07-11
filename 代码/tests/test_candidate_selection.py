from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pi_jwm.evaluation.candidate_selection import (  # noqa: E402
    choice_rmse_from_sample_sse,
    choose_best_single_by_sample_sse,
    mix_actions_by_sample,
    sample_active_sse,
    sample_rmse_from_sse,
)


class CandidateSelectionTest(unittest.TestCase):
    def test_sample_active_sse_ignores_inactive_entries(self) -> None:
        predictions = {
            "link_rate_pred": np.array([[[[2.0], [100.0]]], [[[5.0], [9.0]]]]),
            "link_rate_true": np.array([[[[1.0], [0.0]]], [[[3.0], [8.0]]]]),
            "link_activity_true": np.array([[[[1.0], [0.0]]], [[[1.0], [1.0]]]]),
        }

        sse, count = sample_active_sse(predictions)

        np.testing.assert_allclose(sse, [1.0, 5.0])
        np.testing.assert_array_equal(count, [1, 2])
        np.testing.assert_allclose(sample_rmse_from_sse(sse, count), [1.0, np.sqrt(2.5)])

    def test_zero_active_samples_do_not_affect_aggregate_candidate_choice(self) -> None:
        sample_sse = np.array([[0.0, 1000.0], [9.0, 4.0], [25.0, 9.0]])
        active_count = np.array([0, 1, 1])

        best_index, rmse = choose_best_single_by_sample_sse(sample_sse, active_count)

        self.assertEqual(best_index, 1)
        np.testing.assert_allclose(rmse, [np.sqrt(17.0), np.sqrt(6.5)])
        self.assertAlmostEqual(
            choice_rmse_from_sample_sse(sample_sse, active_count, np.array([0, 1, 0])),
            np.sqrt(14.5),
        )

    def test_mix_actions_by_sample_is_deterministic(self) -> None:
        candidates = [
            np.zeros((3, 2, 1, 1), dtype=np.float32),
            np.ones((3, 2, 1, 1), dtype=np.float32),
        ]
        choice = np.array([1, 0, 1], dtype=np.int64)

        first = mix_actions_by_sample(candidates, choice)
        second = mix_actions_by_sample(candidates, choice)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0, 0, 0], [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
