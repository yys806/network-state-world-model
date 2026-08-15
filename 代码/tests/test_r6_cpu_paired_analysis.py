from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_cpu_paired_analysis import (  # noqa: E402
    aggregate_paired_metric_rows,
    build_pair_key,
    compute_paired_deltas,
    validate_pair_records,
)


class R6PairedAnalysisTest(unittest.TestCase):
    def test_pair_key_requires_same_scenario_seed_and_config(self) -> None:
        left = {
            "scenario_id": "load_low__density_sparse",
            "seed": 7,
            "split": "validation",
            "config_fingerprint": "abc",
            "policy_id": "equal_share",
        }
        right = dict(left, policy_id="local_search")
        self.assertEqual(build_pair_key(left), build_pair_key(right))
        with self.assertRaisesRegex(ValueError, "config"):
            validate_pair_records([left, right, dict(left, config_fingerprint="different")])

    def test_pair_validation_rejects_locked_test_and_incomplete_groups(self) -> None:
        records = [
            {
                "scenario_id": "s",
                "seed": 1,
                "split": "validation",
                "config_fingerprint": "x",
                "policy_id": policy,
            }
            for policy in ("equal_share", "deadline_aware", "feasible_exploration", "local_search")
        ]
        result = validate_pair_records(records)
        self.assertEqual(result["complete_pair_count"], 1)
        with self.assertRaisesRegex(ValueError, "locked-test"):
            validate_pair_records([dict(records[0], split="locked_test")])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_pair_records(records[:2])

    def test_aggregation_keeps_not_computable_without_zero_fill(self) -> None:
        rows = [
            {"pair_key": "s|1|x", "policy_id": "equal_share", "metric_id": "action_regret", "status": "not_computable", "value": None},
            {"pair_key": "s|1|x", "policy_id": "local_search", "metric_id": "action_regret", "status": "not_computable", "value": None},
        ]
        result = aggregate_paired_metric_rows(rows)
        self.assertEqual(result[0]["status"], "not_computable")
        self.assertIsNone(result[0]["mean"])

    def test_paired_delta_requires_complete_available_factual_values(self) -> None:
        rows = [
            {"pair_key": "p", "policy_id": "equal_share", "metric_id": "delay", "status": "available", "value": 10.0},
            {"pair_key": "p", "policy_id": "local_search", "metric_id": "delay", "status": "available", "value": 7.0},
        ]
        result = compute_paired_deltas(rows)
        self.assertEqual(result[0]["status"], "available")
        self.assertAlmostEqual(result[0]["delta_vs_equal_share"], -3.0)
        self.assertEqual(result[0]["reference_policy"], "equal_share")


if __name__ == "__main__":
    unittest.main()
