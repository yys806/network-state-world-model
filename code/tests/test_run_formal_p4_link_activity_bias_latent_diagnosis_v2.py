from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class LinkActivityBiasLatentDiagnosisV2Tests(unittest.TestCase):
    def test_probability_to_logit_for_frozen_threshold(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import probability_to_logit

        self.assertAlmostEqual(probability_to_logit(0.9), 2.1972245773362196, places=15)

    def test_raw_minus_pre_bias_equals_checkpoint_bias(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import summarize_negative_logits

        raw = np.array([[-2.0, 2.5], [-1.0, 3.0]], dtype=np.float64)
        target = np.zeros_like(raw, dtype=bool)
        valid = np.ones_like(target, dtype=bool)
        result = summarize_negative_logits(raw, target, valid, bias=0.25, probability_threshold=0.9)

        self.assertLessEqual(result["invariant_error_abs"], 1e-12)
        self.assertEqual(result["raw_false_positive_at_frozen_threshold"]["count"], 2)

    def test_fixed_bias_preserves_horizon_q99_increment(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import summarize_negative_logits

        target = np.zeros((1, 100), dtype=bool)
        valid = np.ones_like(target, dtype=bool)
        h1 = summarize_negative_logits(np.linspace(-3.0, 1.0, 100)[None, :], target, valid, bias=0.25)
        h20 = summarize_negative_logits(np.linspace(-3.0, 3.0, 100)[None, :], target, valid, bias=0.25)

        raw_increment = h20["raw_logit"]["q99"] - h1["raw_logit"]["q99"]
        pre_bias_increment = h20["pre_bias_logit"]["q99"] - h1["pre_bias_logit"]["q99"]
        self.assertAlmostEqual(raw_increment, pre_bias_increment, places=12)

    def test_positive_and_invalid_edges_are_excluded_from_negative_statistics(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import summarize_negative_logits

        raw = np.array([[3.0, 3.0, 3.0]], dtype=np.float64)
        target = np.array([[True, False, False]])
        valid = np.array([[True, True, False]])
        result = summarize_negative_logits(raw, target, valid, bias=0.1, probability_threshold=0.9)

        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["raw_false_positive_at_frozen_threshold"]["count"], 1)

    def test_frozen_threshold_false_positive_count_is_repeatable(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import summarize_negative_logits

        raw = np.array([[2.2, 2.1972245773362196, 2.19, -1.0]], dtype=np.float64)
        target = np.zeros_like(raw, dtype=bool)
        valid = np.ones_like(target, dtype=bool)
        first = summarize_negative_logits(raw, target, valid, bias=0.0, probability_threshold=0.9)
        second = summarize_negative_logits(raw, target, valid, bias=0.0, probability_threshold=0.9)

        self.assertEqual(first["raw_false_positive_at_frozen_threshold"], second["raw_false_positive_at_frozen_threshold"])
        self.assertEqual(first["raw_false_positive_at_frozen_threshold"]["count"], 2)
        self.assertEqual(first["raw_false_positive_at_frozen_threshold"]["rate"], 0.5)

    def test_only_validation_and_calibration_splits_are_accepted(self):
        from run_formal_p4_link_activity_bias_latent_diagnosis_v2 import validate_split

        self.assertEqual(validate_split("validation"), "validation")
        self.assertEqual(validate_split("calibration"), "calibration")
        with self.assertRaises(ValueError):
            validate_split("train")
        with self.assertRaises(ValueError):
            validate_split("locked_test")


if __name__ == "__main__":
    unittest.main()
