"""CPU-only STEP 6.3B structural support tests; no optimizer or model."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog


class StructuralSupportTests(unittest.TestCase):
    def test_train_catalog_counts_and_axes(self):
        path = (ROOT / "code/artifacts/protocols/"
                "pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json")
        catalog = TrainStructuralSupportCatalog.from_json(path)
        self.assertEqual(len(catalog.joint), 251)
        self.assertEqual(tuple(len(x) for x in catalog.temporal_prefixes), (244, 1165, 2070, 2719))
        self.assertEqual(len(catalog.comm_start_width_pairs), 145)
        self.assertNotIn((0, 2), catalog.comm_start_width_pairs)

    def test_joint_composition_shift_separate_from_temporal_label(self):
        a = ("rows:1", "NOOP", "HOLD,HOLD")
        b = ("NOOP", "tasks:1;alpha:1.0", "HOLD,HOLD")
        catalog = TrainStructuralSupportCatalog(
            "synthetic", frozenset((a, b)),
            tuple(frozenset(signature[i] for signature in (a, b)) for i in range(3)),
            (frozenset(((a,), (b,))), frozenset(((a, b),)), frozenset(), frozenset()),
            frozenset(((a, b),)),
            frozenset(((0, 1),)))
        observed = catalog.label(a)
        self.assertTrue(observed.formal_pool_admitted)
        self.assertEqual(observed.temporal_prefix, "TRAIN_OBSERVED_PREFIX")
        unseen_combo = (a[0], b[1], a[2])
        label = catalog.label(unseen_combo)
        self.assertEqual(set(label.family_marginal.values()), {"TRAIN_OBSERVED"})
        self.assertFalse(label.formal_pool_admitted)
        self.assertIn("JOINT_STRUCTURE_UNSEEN_IN_TRAIN", label.reason_codes)
        unseen_time = catalog.label(a, prior=(a,))
        self.assertTrue(unseen_time.formal_pool_admitted)
        self.assertEqual(unseen_time.temporal_prefix, "TRAIN_UNSEEN_PREFIX")
        self.assertEqual(unseen_time.temporal_adjacent, "TRAIN_UNSEEN_ADJACENT")


if __name__ == "__main__":
    unittest.main()
