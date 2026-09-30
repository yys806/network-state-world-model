import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from pi_jwm.step6_3d_objective_anchor_patch_v1 import replace_empty_cohorts


class ObjectiveAnchorPatchTests(unittest.TestCase):
    def test_replaces_only_zero_cohort_with_first_hash_eligible_in_same_stratum(self):
        rows = [dict(sample_id=name, concrete_count=count, eligible_compute_tasks=comp)
                for name, count, comp in (("old0", 3, 0), ("old1", 4, 1),
                                          ("a", 5, 0), ("b", 6, 0), ("c", 7, 0),
                                          ("wrong", 5, 1), ("empty", 0, 0))]
        old = {"log_cardinality_quartile_cutoffs": [10., 11., 12.],
               "selected": [{"sample_id": "old0", "concrete_count": 3,
                             "eligible_compute_tasks": 0, "stratum": "q0:comp0"},
                            {"sample_id": "old1", "concrete_count": 4,
                             "eligible_compute_tasks": 1, "stratum": "q0:comp1"}]}
        ordered = sorted(("a", "b", "c"), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        seen = []
        def cohort(sample_id):
            seen.append(sample_id)
            return int(sample_id != ordered[0])
        result = replace_empty_cohorts(old, rows, {"old0": 0, "old1": 2}, cohort)
        self.assertEqual([row["sample_id"] for row in result["selected"]], [ordered[1], "old1"])
        self.assertEqual(seen, ordered[:2])
        self.assertEqual(result["replacements"][0]["earlier_hash_candidates_with_empty_cohort"],
                         [ordered[0]])

    def test_rejects_cohort_identity_mismatch(self):
        old = {"log_cardinality_quartile_cutoffs": [1., 2., 3.],
               "selected": [{"sample_id": "x", "concrete_count": 2,
                             "eligible_compute_tasks": 0, "stratum": "q1:comp0"}]}
        with self.assertRaises(ValueError):
            replace_empty_cohorts(old, (), {}, lambda _: 1)


if __name__ == "__main__":
    unittest.main()
