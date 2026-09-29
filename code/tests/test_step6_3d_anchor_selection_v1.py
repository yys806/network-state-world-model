import unittest

from pi_jwm.step6_3d_anchor_selection_v1 import select_stratified_anchors


class AnchorSelectionTests(unittest.TestCase):
    def test_nonempty_deterministic_strata_and_empty_separation(self):
        rows = [
            {"sample_id": f"train::{i:03d}", "concrete_count": (i + 1) * 10,
             "eligible_compute_tasks": i % 2} for i in range(80)
        ]
        rows += [{"sample_id": "train::empty", "concrete_count": 0,
                  "eligible_compute_tasks": 0}]
        first = select_stratified_anchors(rows, 32)
        second = select_stratified_anchors(list(reversed(rows)), 32)
        self.assertEqual(first, second)
        self.assertEqual(len(first["selected"]), 32)
        self.assertEqual(first["empty_sample_ids"], ["train::empty"])
        self.assertEqual(len({row["sample_id"] for row in first["selected"]}), 32)
        self.assertTrue(all(row["concrete_count"] > 0 for row in first["selected"]))
        self.assertEqual(len(first["strata"]), 8)

    def test_rejects_duplicate_and_insufficient_anchors(self):
        rows = [{"sample_id": "a", "concrete_count": 3, "eligible_compute_tasks": 0}]
        with self.assertRaises(ValueError):
            select_stratified_anchors(rows, 2)
        with self.assertRaises(ValueError):
            select_stratified_anchors(rows * 2, 1)


if __name__ == "__main__":
    unittest.main()
