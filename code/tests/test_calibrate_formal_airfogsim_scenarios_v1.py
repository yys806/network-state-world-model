from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "calibrate_formal_airfogsim_scenarios_v1.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "calibrate_formal_airfogsim_scenarios_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load formal scenario calibration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FormalScenarioCalibrationTests(unittest.TestCase):
    def test_summary_accepts_monotonic_load_and_density_evidence(self):
        subject = load_subject()
        rows = []
        for load_index, load in enumerate(("low", "medium", "high"), start=1):
            for density, nodes in (("sparse", 10), ("dense", 20)):
                for seed in (900, 901):
                    rows.append(
                        {
                            "scenario_id": f"load_{load}__density_{density}",
                            "load_level": load,
                            "density_level": density,
                            "seed": seed,
                            "task_count": load_index * 5,
                            "mean_concurrent_tasks": float(load_index),
                            "physical_node_count": nodes,
                            "physical_edge_count": nodes * 2,
                            "link_active_rate": 0.25,
                            "cpu_utilization": 0.5,
                        }
                    )

        report = subject.summarize_calibration(rows, expected_seeds=(900, 901))

        self.assertTrue(report["load_task_count_monotonic"])
        self.assertTrue(report["density_node_count_monotonic"])
        self.assertTrue(report["calibration_ready"])
        self.assertEqual(6, len(report["scenario_summaries"]))

    def test_summary_rejects_missing_repetition_and_reversed_load(self):
        subject = load_subject()
        rows = [
            {
                "scenario_id": "load_low__density_sparse",
                "load_level": "low",
                "density_level": "sparse",
                "seed": 900,
                "task_count": 20,
                "mean_concurrent_tasks": 1.0,
                "physical_node_count": 10,
                "physical_edge_count": 20,
                "link_active_rate": 0.25,
                "cpu_utilization": 0.5,
            },
            {
                "scenario_id": "load_medium__density_sparse",
                "load_level": "medium",
                "density_level": "sparse",
                "seed": 900,
                "task_count": 10,
                "mean_concurrent_tasks": 1.0,
                "physical_node_count": 10,
                "physical_edge_count": 20,
                "link_active_rate": 0.25,
                "cpu_utilization": 0.5,
            },
        ]

        report = subject.summarize_calibration(rows, expected_seeds=(900, 901))

        self.assertFalse(report["load_task_count_monotonic"])
        self.assertFalse(report["two_repetitions_per_scenario"])
        self.assertFalse(report["calibration_ready"])


if __name__ == "__main__":
    unittest.main()
