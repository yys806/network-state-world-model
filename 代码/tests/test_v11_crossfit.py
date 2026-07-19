import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
