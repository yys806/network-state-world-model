from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalAirFogSimProtocolTests(unittest.TestCase):
    def test_default_protocol_has_balanced_60_trajectories(self):
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        specs = build_formal_trajectory_specs()

        self.assertEqual(60, len(specs))
        self.assertEqual(
            {"train": 36, "validation": 12, "calibration": 6, "locked_test": 6},
            Counter(row.split for row in specs),
        )
        self.assertEqual(
            {"equal_share": 20, "deadline_aware": 20, "feasible_exploration": 20},
            Counter(row.cpu_policy for row in specs),
        )

    def test_each_scenario_has_ten_trajectories_and_all_splits(self):
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        specs = build_formal_trajectory_specs()
        by_scenario = Counter(row.scenario.scenario_id for row in specs)

        self.assertEqual(6, len(by_scenario))
        self.assertTrue(all(count == 10 for count in by_scenario.values()))
        for scenario_id in by_scenario:
            splits = Counter(
                row.split for row in specs if row.scenario.scenario_id == scenario_id
            )
            self.assertEqual(
                {"train": 6, "validation": 2, "calibration": 1, "locked_test": 1},
                splits,
            )

    def test_protocol_validation_rejects_duplicate_seed(self):
        from pi_jwm.formal_airfogsim_dataset_v1 import (
            build_formal_trajectory_specs,
            validate_formal_protocol,
        )

        specs = build_formal_trajectory_specs()
        broken = [specs[0], specs[0], *specs[2:]]
        report = validate_formal_protocol(broken)

        self.assertFalse(report["protocol_valid"])
        self.assertIn("trajectory_seeds_unique", report["failed_checks"])
        self.assertIn("trajectory_ids_unique", report["failed_checks"])

    def test_locked_test_access_requires_explicit_unlock(self):
        from pi_jwm.formal_airfogsim_dataset_v1 import require_split_access

        require_split_access("train")
        require_split_access("locked_test", allow_locked_test=True)
        with self.assertRaises(PermissionError):
            require_split_access("locked_test")
        with self.assertRaises(ValueError):
            require_split_access("unknown")

    def test_scenario_overrides_only_declared_config_fields(self):
        from pi_jwm.formal_airfogsim_dataset_v1 import (
            DEFAULT_SCENARIOS,
            apply_formal_scenario_overrides,
        )

        base = {
            "traffic": {
                "max_n_vehicles": 5,
                "arrival_lambda": 0.1,
                "max_n_UAVs": 2,
            },
            "task": {"task_generation_kwargs": {"lambda": 9.0}},
            "task_profile": {
                "vehicle": {"lambda": 9.0, "dag_edge_prob": 0.6},
                "uav": {"lambda": 9.0, "dag_edge_prob": 0.6},
            },
            "untouched": {"value": 17},
        }
        scenario = DEFAULT_SCENARIOS[-1]
        configured = apply_formal_scenario_overrides(base, scenario)

        self.assertEqual(scenario.max_vehicles, configured["traffic"]["max_n_vehicles"])
        self.assertEqual(
            scenario.vehicle_arrival_lambda,
            configured["traffic"]["arrival_lambda"],
        )
        self.assertEqual(
            scenario.task_lambda,
            configured["task"]["task_generation_kwargs"]["lambda"],
        )
        self.assertEqual(
            scenario.task_lambda,
            configured["task_profile"]["vehicle"]["lambda"],
        )
        self.assertEqual(
            scenario.task_lambda,
            configured["task_profile"]["uav"]["lambda"],
        )
        self.assertEqual({"value": 17}, configured["untouched"])
        self.assertEqual(5, base["traffic"]["max_n_vehicles"])
        self.assertEqual(copy.deepcopy(base), base)


if __name__ == "__main__":
    unittest.main()
