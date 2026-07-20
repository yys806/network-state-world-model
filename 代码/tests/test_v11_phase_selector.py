import unittest

import numpy as np


class PhaseSelectorTest(unittest.TestCase):
    def _dataset(self):
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        sample_ids = np.asarray([0, 390, 1, 391], dtype=np.int64)
        batch = CandidateBatch(
            context=np.zeros((4, 1), dtype=np.float32),
            candidate_features=np.zeros((4, 3, 2), dtype=np.float32),
            candidate_mask=np.ones((4, 3), dtype=bool),
            stage=np.asarray(["offload", "offload", "compute", "compute"]),
            feature_names=("predicted_task_delta_0", "predicted_energy_proxy"),
            candidate_names=("identity", "ranked", "repair"),
            context_feature_names=("state_task_num_tasks_last",),
        )
        outcome = CandidateOutcome(
            active_sse=np.asarray(
                [[12.0, 10.0, 4.0], [11.0, 10.0, 6.0], [9.0, 10.0, 12.0], [8.0, 10.0, 14.0]],
                dtype=np.float32,
            ),
            active_count=np.ones(4, dtype=np.int64),
            default_index=1,
        )
        return batch, outcome, sample_ids

    def test_statistics_pool_same_phase_across_seeds(self):
        from pi_jwm.v11_phase_selector import fit_phase_candidate_statistics

        batch, outcome, sample_ids = self._dataset()
        stats = fit_phase_candidate_statistics(batch, outcome, sample_ids, episode_length=390)
        self.assertEqual(stats.count[0, 2], 2)
        self.assertAlmostEqual(float(stats.mean_benefit[0, 2]), 5.0)
        self.assertAlmostEqual(float(stats.positive_rate[0, 2]), 1.0)
        self.assertEqual(stats.count[1, 2], 2)
        self.assertLess(float(stats.mean_benefit[1, 2]), 0.0)

    def test_selection_uses_deployable_mask_and_pareto_gate(self):
        from pi_jwm.v11_phase_selector import (
            PhaseSelectorConfig,
            fit_phase_candidate_statistics,
            select_phase_candidates,
        )

        batch, outcome, sample_ids = self._dataset()
        stats = fit_phase_candidate_statistics(batch, outcome, sample_ids)
        query_ids = np.asarray([0, 2], dtype=np.int64)
        query = type(batch)(
            context=batch.context[:2],
            candidate_features=batch.candidate_features[:2],
            candidate_mask=np.ones((2, 3), dtype=bool),
            stage=batch.stage[:2],
            feature_names=batch.feature_names,
            candidate_names=batch.candidate_names,
            context_feature_names=batch.context_feature_names,
        )
        pareto = np.asarray([[True, True, False], [True, True, True]], dtype=bool)
        pareto[1, 0] = False
        with np.errstate(invalid="raise"):
            choice = select_phase_candidates(
                stats,
                query,
                query_ids,
                PhaseSelectorConfig(0.0, 0.5, 0.0, 1),
                pareto_allowed=pareto,
            )
        np.testing.assert_array_equal(choice, np.asarray([1, 1]))

    def test_calibration_selects_only_safe_thresholds(self):
        from pi_jwm.v11_phase_selector import (
            calibrate_phase_selector,
            fit_phase_candidate_statistics,
        )

        batch, outcome, sample_ids = self._dataset()
        stats = fit_phase_candidate_statistics(batch, outcome, sample_ids)
        calibrated = calibrate_phase_selector(
            stats,
            batch,
            outcome,
            sample_ids,
            pareto_allowed=np.ones((4, 3), dtype=bool),
            z_values=(0.0,),
            positive_rate_values=(0.5, 1.0),
            minimum_mean_values=(0.0,),
            min_count=1,
        )
        self.assertEqual(calibrated.status, "safe_threshold")
        self.assertGreaterEqual(calibrated.positive_precision, 0.65)
        self.assertLessEqual(calibrated.negative_selection_rate, 0.20)
        self.assertGreater(calibrated.executed_count, 0)


if __name__ == "__main__":
    unittest.main()
