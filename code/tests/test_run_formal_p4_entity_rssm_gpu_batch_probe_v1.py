from __future__ import annotations

import unittest


class FormalP4EntityRSSMGpuBatchProbeTests(unittest.TestCase):
    def test_selects_largest_completed_batch_under_memory_limit(self):
        from run_formal_p4_entity_rssm_gpu_batch_probe_v1 import select_largest_feasible_batch

        rows = [
            {"batch_size": 8, "completed": False, "memory_fraction": 0.9},
            {"batch_size": 4, "completed": True, "memory_fraction": 0.86},
            {"batch_size": 2, "completed": True, "memory_fraction": 0.8},
            {"batch_size": 1, "completed": True, "memory_fraction": 0.5},
        ]
        self.assertEqual(2, select_largest_feasible_batch(rows))

    def test_rejects_when_no_candidate_is_safe(self):
        from run_formal_p4_entity_rssm_gpu_batch_probe_v1 import select_largest_feasible_batch

        with self.assertRaisesRegex(RuntimeError, "no candidate"):
            select_largest_feasible_batch(
                [{"batch_size": 1, "completed": True, "memory_fraction": 0.9}]
            )


if __name__ == "__main__":
    unittest.main()
