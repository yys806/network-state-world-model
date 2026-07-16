import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def load_module():
    path = SCRIPTS_ROOT / "analyze_pi_jwm_horizon_stage_diagnostic.py"
    spec = importlib.util.spec_from_file_location("pi_jwm_horizon_stage_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HorizonStageAnalysisTest(unittest.TestCase):
    def test_alignment_rejects_different_candidate_sets(self):
        module = load_module()
        base = pd.DataFrame(
            [{"seed": 0, "decision_time": 1.0, "candidate_id": "default", "action_family": "default"}]
        )
        changed = pd.DataFrame(
            [{"seed": 0, "decision_time": 1.0, "candidate_id": "rb", "action_family": "rb_count"}]
        )

        with self.assertRaisesRegex(ValueError, "candidate alignment"):
            module.validate_candidate_alignment({3: base, 10: changed})

    def test_stage_family_summary_reports_per_step_effect(self):
        module = load_module()
        candidates = {
            10: pd.DataFrame(
                [
                    {
                        "seed": 0,
                        "decision_time": 1.0,
                        "candidate_id": "rb",
                        "action_family": "rb_count",
                        "decision_stage": "offload_rb",
                    }
                ]
            )
        }
        effects = {
            10: pd.DataFrame(
                [
                    {
                        "seed": 0,
                        "decision_time": 1.0,
                        "candidate_id": "rb",
                        "action_family": "rb_count",
                        "effect_task_utility": -2.0,
                        "effect_energy_total": -1.0,
                        "effect_throughput_delta": -5.0,
                    }
                ]
            )
        }

        result = module.build_stage_family_summary(candidates, effects)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["decision_stage"], "offload_rb")
        self.assertAlmostEqual(result.iloc[0]["mean_effect_task_utility_per_step"], -0.2)
        self.assertAlmostEqual(result.iloc[0]["mean_effect_energy_total_per_step"], -0.1)

    def test_horizon_summary_keeps_nontrivial_ratio_and_audit_boundary(self):
        module = load_module()
        summaries = {
            3: {
                "num_seeds": 5,
                "num_step_rows": 198,
                "num_candidates": 66,
                "num_decision_groups": 15,
                "num_nontrivial_groups": 7,
                "quality_audit": {"passed": True},
            }
        }
        group_audits = {
            3: pd.DataFrame([{"utility_spread": 0.0}, {"utility_spread": 1.0}])
        }

        result = module.build_horizon_summary(summaries, group_audits)

        self.assertAlmostEqual(result.iloc[0]["nontrivial_group_ratio"], 7 / 15)
        self.assertTrue(bool(result.iloc[0]["quality_audit_passed"]))
        self.assertAlmostEqual(result.iloc[0]["mean_utility_spread"], 0.5)


if __name__ == "__main__":
    unittest.main()
