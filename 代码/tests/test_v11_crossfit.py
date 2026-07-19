import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SeedCrossfitProtocolTest(unittest.TestCase):
    def test_fixed_round_robin_folds_cover_train_without_overlap(self):
        from pi_jwm.v11_crossfit import (
            audit_seed_crossfit_folds,
            build_seed_crossfit_folds,
        )
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        folds = build_seed_crossfit_folds(DEFAULT_SELECTOR_SEEDS["train"])

        self.assertEqual(
            [fold.held_out_seeds for fold in folds],
            [
                (0, 5, 10, 15, 24, 29, 34, 39),
                (1, 6, 11, 20, 25, 30, 35, 40),
                (2, 7, 12, 21, 26, 31, 36, 41),
                (3, 8, 13, 22, 27, 32, 37, 42),
                (4, 9, 14, 23, 28, 33, 38, 43),
            ],
        )
        self.assertTrue(
            audit_seed_crossfit_folds(folds, DEFAULT_SELECTOR_SEEDS)["passed"]
        )

    def test_protocol_digest_is_canonical_and_has_no_execution_fold(self):
        from pi_jwm.v11_crossfit import build_crossfit_protocol_manifest

        left = build_crossfit_protocol_manifest({"rf_trees": 160, "schema": 6})
        right = build_crossfit_protocol_manifest({"schema": 6, "rf_trees": 160})

        self.assertEqual(
            left["crossfit_protocol_digest"], right["crossfit_protocol_digest"]
        )
        self.assertNotIn("execution_fold", left["crossfit_protocol_payload"])


class CrossfitExecutionTest(unittest.TestCase):
    @staticmethod
    def _sample_seed():
        all_seeds = (
            list(range(16))
            + list(range(20, 60))
            + list(range(60, 70))
        )
        return np.repeat(np.asarray(all_seeds, dtype=np.int64), 2)

    def test_train_fold_helper_never_reads_held_out_seed(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(sample_seed, ("train",), fold_id=0)

        self.assertEqual(
            set(sample_seed[result.label_indices["train"]]),
            set(result.held_out_seeds),
        )
        self.assertFalse(
            set(sample_seed[result.helper_train_indices]) & set(result.held_out_seeds)
        )
        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(result.helper_train_seeds),
        )

    def test_eval_uses_full_train_helper_and_rejects_fold_id(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(
            sample_seed, ("calibration", "validation"), fold_id=None
        )

        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(DEFAULT_SELECTOR_SEEDS["train"]),
        )
        with self.assertRaisesRegex(ValueError, "fold"):
            resolve_crossfit_execution(sample_seed, ("validation",), fold_id=1)

    def test_locked_evaluation_split_uses_no_locked_seed_for_helper_training(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(
            sample_seed, ("external_holdout",), fold_id=None
        )

        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(DEFAULT_SELECTOR_SEEDS["train"]),
        )
        self.assertEqual(
            set(sample_seed[result.label_indices["external_holdout"]]),
            set(DEFAULT_SELECTOR_SEEDS["external_holdout"]),
        )


if __name__ == "__main__":
    unittest.main()
