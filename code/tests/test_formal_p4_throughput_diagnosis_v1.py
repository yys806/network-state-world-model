import unittest

import numpy as np

from run_formal_p4_throughput_diagnosis_v1 import summarize_stepwise_totals


class ThroughputDiagnosisTests(unittest.TestCase):
    def test_summarize_stepwise_totals_reports_bias_and_mae_per_step(self):
        predicted = np.array([[[1.0, 3.0], [2.0, 2.0]]], dtype=np.float64)
        target = np.array([[[2.0, 2.0], [1.0, 1.0]]], dtype=np.float64)
        persistence = np.array([[[2.0, 1.0], [1.0, 3.0]]], dtype=np.float64)
        observed = np.array([[[True, True], [True, False]]])

        result = summarize_stepwise_totals(predicted, target, persistence, observed)

        self.assertEqual(result["steps"][0]["count"], 1)
        self.assertAlmostEqual(result["steps"][0]["learned_total_mae"], 0.0)
        self.assertAlmostEqual(result["steps"][0]["persistence_total_mae"], 1.0)
        self.assertAlmostEqual(result["steps"][0]["learned_bias"], 0.0)
        self.assertAlmostEqual(result["steps"][0]["persistence_bias"], -1.0)
        self.assertAlmostEqual(result["steps"][1]["learned_total_mae"], 1.0)
        self.assertAlmostEqual(result["steps"][1]["persistence_total_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
