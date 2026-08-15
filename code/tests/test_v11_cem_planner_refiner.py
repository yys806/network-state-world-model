import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11CemPlannerRefinerTest(unittest.TestCase):
    def test_cem_refine_scores_preserves_shape_and_finiteness(self):
        from compare_v11_cem_planner_refiner import cem_refine_scores

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
            ],
            dtype=np.int64,
        )
        base = np.array([0.1, 0.9, 0.2, 0.7, 0.1, 0.2], dtype=np.float32)

        refined = cem_refine_scores(
            coords,
            base,
            top_k=1,
            iterations=2,
            samples_per_group=8,
            elite_frac=0.25,
            noise_std=0.1,
            momentum=0.7,
            seed=7,
        )

        self.assertEqual(refined.shape, base.shape)
        self.assertTrue(np.all(np.isfinite(refined)))
        self.assertGreaterEqual(float(refined.min()), 0.0)
        self.assertLessEqual(float(refined.max()), 1.0)
        self.assertEqual(int(np.argmax(refined[:3])), 1)
        self.assertEqual(int(np.argmax(refined[3:])), 0)


if __name__ == '__main__':
    unittest.main()
