from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalBinaryCalibrationV1Tests(unittest.TestCase):
    def test_inverts_the_positive_weighted_bce_probability(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            correct_positive_weighted_probability,
        )

        posterior = torch.tensor([0.01, 0.25, 0.50, 0.80], dtype=torch.float32)
        pos_weight = 50.0
        weighted_score = posterior * pos_weight / (
            1.0 - posterior + posterior * pos_weight
        )

        recovered = correct_positive_weighted_probability(weighted_score, pos_weight)

        torch.testing.assert_close(recovered, posterior, atol=1e-6, rtol=0.0)

    def test_inverse_temperature_uses_analytic_inversion_at_identity(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            probabilities_from_weighted_logits,
        )

        logits = torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64)
        calibration = InverseTemperatureCalibration(50.0, 0.0)

        self.assertEqual(calibration.temperature, 1.0)
        torch.testing.assert_close(
            probabilities_from_weighted_logits(logits, calibration),
            torch.sigmoid(logits - torch.log(torch.tensor(50.0, dtype=torch.float64))),
            atol=1e-12,
            rtol=0.0,
        )

    def test_mapped_thresholds_preserve_legacy_decisions_and_raw_tie_break(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            choose_legacy_threshold,
            map_raw_threshold,
        )

        calibration = InverseTemperatureCalibration(50.0, 0.0)
        raw_scores = torch.tensor([0.8, 0.2], dtype=torch.float64)
        labels = torch.tensor([True, False])
        result = choose_legacy_threshold(raw_scores, labels, calibration)

        self.assertEqual(result["selected"]["raw_threshold"], 0.5)
        self.assertTrue(result["decision_equivalence_to_legacy"])
        self.assertEqual(len(result["candidates"]), 5)
        mapped = map_raw_threshold(0.5, calibration)
        self.assertGreater(mapped, 0.0)
        self.assertLess(mapped, 1.0)

    def test_fit_inverse_temperature_uses_unweighted_calibration_nll(self):
        from pi_jwm.formal_binary_calibration_v1 import fit_inverse_temperature

        logits = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1, 1])
        calibration, report = fit_inverse_temperature(logits, labels, pos_weight=50.0)

        self.assertEqual(report["fit_split"], "calibration")
        self.assertEqual(report["objective"], "unweighted_bernoulli_nll")
        self.assertEqual(report["temperature"], calibration.temperature)
        self.assertLessEqual(report["fitted_nll"], report["identity_nll"] + 1e-12)

    def test_fit_allows_identity_temperature(self):
        from pi_jwm.formal_binary_calibration_v1 import fit_inverse_temperature

        calibration, report = fit_inverse_temperature(
            torch.full((4,), torch.log(torch.tensor(2.0))),
            torch.tensor([0, 1, 0, 1]),
            pos_weight=2.0,
        )

        self.assertEqual(calibration.temperature, 1.0)
        self.assertEqual(report["temperature"], 1.0)

    def test_probability_metrics_obey_fixed_ece_boundaries(self):
        from pi_jwm.formal_binary_calibration_v1 import binary_probability_metrics

        metrics = binary_probability_metrics(
            torch.tensor([0.0, 1.0, 1.0 / 15.0, 14.0 / 15.0]),
            torch.tensor([0, 1, 0, 1]),
        )

        self.assertEqual(metrics["count"], 4)
        self.assertEqual(metrics["ece_bin_count"], 15)
        self.assertEqual(metrics["ece_bins"][0]["count"], 1)
        self.assertEqual(metrics["ece_bins"][1]["count"], 1)
        self.assertEqual(metrics["ece_bins"][14]["count"], 2)

    def test_new_apis_reject_all_specified_invalid_inputs(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            binary_probability_metrics,
            choose_legacy_threshold,
            fit_inverse_temperature,
            map_raw_threshold,
            probabilities_from_weighted_logits,
            unweighted_logits,
        )

        valid = InverseTemperatureCalibration(2.0, 0.0)
        with self.assertRaises(ValueError):
            InverseTemperatureCalibration(0.0, 0.0)
        with self.assertRaises(ValueError):
            InverseTemperatureCalibration(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            unweighted_logits(torch.tensor([float("inf")]), valid)
        with self.assertRaises(ValueError):
            probabilities_from_weighted_logits(torch.tensor([float("nan")]), valid)
        for raw_threshold in (0.0, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                map_raw_threshold(raw_threshold, valid)
        for logits, labels, weight, split in (
            (torch.tensor([]), torch.tensor([]), 1.0, "calibration"),
            (torch.tensor([0.0]), torch.tensor([0.0, 1.0]), 1.0, "calibration"),
            (torch.tensor([float("nan")]), torch.tensor([0.0]), 1.0, "calibration"),
            (torch.tensor([0.0]), torch.tensor([2.0]), 1.0, "calibration"),
            (torch.tensor([0.0]), torch.tensor([0.0]), 1.0, "validation"),
        ):
            with self.assertRaises(ValueError):
                fit_inverse_temperature(logits, labels, pos_weight=weight, fit_split=split)
        for probabilities, labels in (
            (torch.tensor([]), torch.tensor([])),
            (torch.tensor([0.2]), torch.tensor([0.0, 1.0])),
            (torch.tensor([float("inf")]), torch.tensor([0.0])),
            (torch.tensor([1.1]), torch.tensor([0.0])),
            (torch.tensor([0.2]), torch.tensor([2.0])),
        ):
            with self.assertRaises(ValueError):
                binary_probability_metrics(probabilities, labels)
        with self.assertRaises(ValueError):
            choose_legacy_threshold(torch.tensor([1.1]), torch.tensor([0]), valid)

    def test_calibration_instance_methods_are_frozen_and_match_wrappers(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            map_raw_threshold,
            probabilities_from_weighted_logits,
            unweighted_logits,
        )

        calibration = InverseTemperatureCalibration(2.0, 1e-12)
        with self.assertRaises(FrozenInstanceError):
            calibration.pos_weight = 3.0
        logits = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(
            calibration.unweighted_logits(logits), unweighted_logits(logits, calibration)
        )
        torch.testing.assert_close(
            calibration.probabilities_from_weighted_logits(logits),
            probabilities_from_weighted_logits(logits, calibration),
        )
        self.assertEqual(
            calibration.map_raw_threshold(0.5), map_raw_threshold(0.5, calibration)
        )
        self.assertEqual(calibration.temperature, 1.0)
        self.assertGreater(InverseTemperatureCalibration(2.0, 1.1e-12).temperature, 1.0)
        self.assertLess(InverseTemperatureCalibration(2.0, -1.1e-12).temperature, 1.0)

    def test_extreme_temperatures_and_underflowed_mapping_fail_as_value_errors(self):
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration

        for log_temperature in (float("nan"), 1000.0, -1000.0):
            with self.assertRaises(ValueError):
                InverseTemperatureCalibration(2.0, log_temperature)
        with self.assertRaises(ValueError):
            InverseTemperatureCalibration(50.0, -10.0).map_raw_threshold(0.1)

    def test_mapping_is_strictly_monotonic_in_the_safe_range(self):
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration

        calibration = InverseTemperatureCalibration(50.0, math.log(2.0))
        thresholds = [calibration.map_raw_threshold(value) for value in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertEqual(len(set(thresholds)), len(thresholds))

    def test_fit_report_proves_frozen_cpu_float64_lbfgs_contract_and_mean_bce(self):
        from pi_jwm.formal_binary_calibration_v1 import fit_inverse_temperature

        weighted_logits = torch.log(torch.tensor([2.0, 2.0, 2.0, 2.0]))
        labels = torch.tensor([0, 1, 0, 1])
        _, report = fit_inverse_temperature(weighted_logits, labels, pos_weight=2.0)
        expected_identity_nll = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(4, dtype=torch.float64), labels.to(torch.float64), reduction="mean"
        ).item()

        self.assertEqual(report["optimizer"], "LBFGS")
        self.assertEqual(report["dtype"], "float64")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["max_iter"], 100)
        self.assertEqual(report["tolerance_grad"], 1e-12)
        self.assertEqual(report["tolerance_change"], 1e-12)
        self.assertEqual(report["line_search_fn"], "strong_wolfe")
        self.assertAlmostEqual(report["identity_nll"], expected_identity_nll, places=14)

    def test_probability_metrics_report_exact_nll_brier_and_ece(self):
        from pi_jwm.formal_binary_calibration_v1 import binary_probability_metrics

        probabilities = torch.tensor(
            [0.0, 1.0, 1.0 / 15.0, 14.0 / 15.0], dtype=torch.float64
        )
        labels = torch.tensor([0, 1, 0, 1])
        metrics = binary_probability_metrics(probabilities, labels)
        expected_nll = torch.nn.functional.binary_cross_entropy(
            probabilities.to(torch.float64).clamp(1e-7, 1.0 - 1e-7),
            labels.to(torch.float64),
        ).item()

        self.assertAlmostEqual(metrics["nll"], expected_nll, places=14)
        self.assertAlmostEqual(metrics["brier"], 1.0 / 450.0, places=14)
        self.assertAlmostEqual(metrics["ece"], 1.0 / 30.0, places=12)

    def test_each_returned_candidate_recomputes_legacy_probability_equivalence(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            choose_legacy_threshold,
        )

        raw_scores = torch.tensor([0.12, 0.3, 0.5, 0.7, 0.95], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 1])
        calibration = InverseTemperatureCalibration(50.0, math.log(1.5))
        result = choose_legacy_threshold(raw_scores, labels, calibration)
        raw_logits = torch.logit(raw_scores)
        probabilities = calibration.probabilities_from_weighted_logits(raw_logits)
        for candidate in result["candidates"]:
            raw_decision = raw_scores >= candidate["raw_threshold"]
            probability_decision = probabilities >= candidate["probability_threshold"]
            self.assertTrue(torch.equal(raw_decision, probability_decision))

    def test_all_tensor_entry_points_reject_complex_inputs(self):
        from pi_jwm.formal_binary_calibration_v1 import (
            InverseTemperatureCalibration,
            binary_probability_metrics,
            choose_legacy_threshold,
            fit_inverse_temperature,
            probabilities_from_weighted_logits,
            unweighted_logits,
        )

        calibration = InverseTemperatureCalibration(2.0, 0.0)
        complex_values = torch.tensor([1.0 + 0.0j])
        labels = torch.tensor([1])
        for callback in (
            lambda: unweighted_logits(complex_values, calibration),
            lambda: probabilities_from_weighted_logits(complex_values, calibration),
            lambda: fit_inverse_temperature(complex_values, labels, pos_weight=2.0),
            lambda: binary_probability_metrics(complex_values, labels),
            lambda: choose_legacy_threshold(complex_values, labels, calibration),
        ):
            with self.assertRaises(ValueError):
                callback()

    def test_legacy_positive_weight_correction_rejects_invalid_weights(self):
        from pi_jwm.formal_binary_calibration_v1 import correct_positive_weighted_probability

        for pos_weight in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0):
            with self.assertRaises(ValueError):
                correct_positive_weighted_probability(torch.tensor([0.5]), pos_weight)

    def test_legacy_positive_weight_correction_preserves_tiny_positive_weight_inverse(self):
        from pi_jwm.formal_binary_calibration_v1 import correct_positive_weighted_probability

        pos_weight = 1e-20
        posterior = torch.tensor([0.5], dtype=torch.float64)
        weighted_score = posterior * pos_weight / (1.0 - posterior + posterior * pos_weight)

        recovered = correct_positive_weighted_probability(weighted_score, pos_weight)
        torch.testing.assert_close(recovered, posterior, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
