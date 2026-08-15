import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11AdaptiveCandidateOracleTest(unittest.TestCase):
    def test_oracle_select_by_sample_chooses_lowest_active_sse(self):
        from diagnose_v11_adaptive_candidate_oracle import oracle_select_by_sample

        rates = np.asarray(
            [
                [[[1.0, 10.0]], [[5.0, 0.0]]],
                [[[3.0, 2.0]], [[1.0, 8.0]]],
            ],
            dtype=np.float32,
        )
        truth = np.asarray([[[2.0, 2.0]], [[1.0, 10.0]]], dtype=np.float32)
        active = np.asarray([[[1, 1]], [[1, 1]]], dtype=bool)

        selected, indices, sse = oracle_select_by_sample(rates, truth, active)

        self.assertEqual(indices.tolist(), [1, 1])
        self.assertEqual(tuple(selected.shape), tuple(truth.shape))
        np.testing.assert_allclose(sse, [[65.0, 116.0], [1.0, 4.0]])

    def test_oracle_select_by_step_can_switch_within_sample(self):
        from diagnose_v11_adaptive_candidate_oracle import oracle_select_by_step

        rates = np.asarray(
            [
                [[1.0, 1.0], [10.0, 10.0]],
                [[10.0, 10.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ).reshape(2, 1, 2, 2)
        truth = np.asarray([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
        active = np.ones_like(truth, dtype=bool)

        selected, indices = oracle_select_by_step(rates, truth, active)

        self.assertEqual(indices.tolist(), [[0, 1]])
        np.testing.assert_allclose(selected, truth)

    def test_squeeze_last_channel_accepts_singleton_last_dim(self):
        from diagnose_v11_adaptive_candidate_oracle import squeeze_last_channel

        arr = np.zeros((2, 3, 4, 1), dtype=np.float32)

        self.assertEqual(squeeze_last_channel(arr).shape, (2, 3, 4))

    def test_parse_point_spec_accepts_named_and_unnamed_specs(self):
        from diagnose_v11_adaptive_candidate_oracle import parse_point_spec

        named = parse_point_spec("low:0.34:0.8")
        unnamed = parse_point_spec("0.4:1.0")

        self.assertEqual(named.name, "low")
        self.assertAlmostEqual(named.threshold, 0.34)
        self.assertAlmostEqual(named.value_scale, 0.8)
        self.assertEqual(unnamed.name, "point_thr0p4_scale1")
        self.assertAlmostEqual(unnamed.threshold, 0.4)
        self.assertAlmostEqual(unnamed.value_scale, 1.0)

    def test_candidate_specs_can_disable_rules(self):
        from diagnose_v11_adaptive_candidate_oracle import candidate_specs

        specs = candidate_specs(SimpleNamespace(disable_rules=True, rule=None))

        self.assertEqual(specs, [])

    def test_candidate_selector_rows_include_point_candidates(self):
        from diagnose_v11_adaptive_candidate_oracle import candidate_selector_rows

        rows = [
            {"selector": "candidate_point", "name": "p"},
            {"selector": "candidate", "name": "r"},
            {"selector": "oracle_step", "name": "o"},
        ]

        self.assertEqual([row["name"] for row in candidate_selector_rows(rows)], ["p", "r"])


if __name__ == "__main__":
    unittest.main()
