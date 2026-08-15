from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_10k_analysis import (  # noqa: E402
    analyze_r6_10k_records,
    write_r6_10k_analysis_bundle,
)


METHODS = ("actor_critic", "ppo_clipped")
MODES = ("explicit_only", "latent_only", "explicit_latent")
SEEDS = (11, 12, 13)


def _record(method: str, mode: str, seed: int) -> dict:
    method_offset = 0.01 if method == "ppo_clipped" else 0.0
    mode_offset = {
        "explicit_only": 0.0,
        "latent_only": 0.02,
        "explicit_latent": 0.03,
    }[mode]
    validation_return = 0.20 + method_offset + mode_offset + 0.001 * (seed - 12)
    latency = 0.50 - method_offset - mode_offset - 0.001 * (seed - 12)
    reports = []
    for update, environment_step in enumerate((128, 256, 10_000), start=1):
        reports.append(
            {
                "environment_step": environment_step,
                "objective_id": method,
                "total_loss": 4.0 / update,
                "policy_loss": 2.0 / update,
                "value_loss": 3.0 / update,
                "entropy": 1.7,
                "gradient_norm": 2.0,
                "ratio_min": 0.9,
                "ratio_max": 1.1,
                "parameter_changed": True,
            }
        )
    return {
        "run_id": f"{method}__{mode}__seed_{seed}",
        "formal": True,
        "status": "complete",
        "environment_steps": 10_000,
        "update_count": 3,
        "reports": reports,
        "validation_reports": [
            {
                "environment_step": 10_000,
                "validation_return": validation_return,
                "on_time_completion_rate": 1.0,
                "mean_latency": latency,
                "hard_violation_count": 0,
                "validation_step_count": 768,
                "validation_trajectory_count": 12,
            }
        ],
        "checkpoint_reload_verified": True,
        "world_model_updated": False,
        "locked_test_accessed": False,
        "hard_violation_count": 0,
        "state_source": "online_airfogsim_strict_dual_graph",
        "candidate_selection_counts": {
            "airfogsim_default": 2_000,
            "deadline_first": 1_600,
            "energy_conservative": 1_600,
            "load_balance": 1_600,
            "priority_first": 1_600,
            "rate_aware": 1_600,
        },
        "nondefault_selection_count": 8_000,
        "distinct_explicit_state_count": 9_000,
        "elapsed_seconds": 100.0,
    }


def _matrix() -> list[dict]:
    return [
        _record(method, mode, seed)
        for method in METHODS
        for mode in MODES
        for seed in SEEDS
    ]


class R610kGateAnalysisTest(unittest.TestCase):
    def test_rejects_incomplete_or_locked_test_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            analyze_r6_10k_records(
                _matrix()[:-1],
                expected_methods=METHODS,
                expected_state_modes=MODES,
                expected_seeds=SEEDS,
                target_environment_steps=10_000,
            )

        unsafe = _matrix()
        unsafe[0]["locked_test_accessed"] = True
        with self.assertRaisesRegex(ValueError, "locked-test"):
            analyze_r6_10k_records(
                unsafe,
                expected_methods=METHODS,
                expected_state_modes=MODES,
                expected_seeds=SEEDS,
                target_environment_steps=10_000,
            )

    def test_rejects_nonfinite_training_diagnostic(self) -> None:
        records = _matrix()
        records[0]["reports"][1]["total_loss"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_r6_10k_records(
                records,
                expected_methods=METHODS,
                expected_state_modes=MODES,
                expected_seeds=SEEDS,
                target_environment_steps=10_000,
            )

    def test_rejects_unknown_candidate_id_even_when_count_total_matches(self) -> None:
        records = _matrix()
        counts = records[0]["candidate_selection_counts"]
        counts["unknown_candidate"] = counts.pop("rate_aware")
        with self.assertRaisesRegex(ValueError, "candidate IDs"):
            analyze_r6_10k_records(
                records,
                expected_methods=METHODS,
                expected_state_modes=MODES,
                expected_seeds=SEEDS,
                target_environment_steps=10_000,
            )

    def test_pairing_respects_metric_direction_and_seed_alignment(self) -> None:
        analysis = analyze_r6_10k_records(
            _matrix(),
            expected_methods=METHODS,
            expected_state_modes=MODES,
            expected_seeds=SEEDS,
            target_environment_steps=10_000,
        )

        policy = analysis["policy_paired_by_state_mode"]["explicit_only"]
        self.assertEqual(policy["validation_return"]["wins"], 3)
        self.assertEqual(policy["mean_latency"]["wins"], 3)
        state = analysis["state_paired_by_method"]["actor_critic"]
        self.assertEqual(
            state["explicit_latent_vs_explicit_only"]["validation_return"]["wins"],
            3,
        )
        self.assertEqual(
            state["explicit_latent_vs_explicit_only"]["mean_latency"]["wins"],
            3,
        )

    def test_health_gate_continues_all_runs_without_declaring_winner(self) -> None:
        analysis = analyze_r6_10k_records(
            _matrix(),
            expected_methods=METHODS,
            expected_state_modes=MODES,
            expected_seeds=SEEDS,
            target_environment_steps=10_000,
        )

        self.assertNotIn("winner", analysis)
        gate = analysis["continuation_gate"]
        self.assertEqual(gate["status"], "pass_continue_full_frozen_matrix")
        self.assertTrue(gate["continue_to_full_budget"])
        self.assertEqual(gate["recommended_configuration_count"], 6)
        self.assertEqual(gate["recommended_run_count"], 18)
        self.assertEqual(analysis["integrity"]["validation_points_per_run"], [1])
        self.assertEqual(analysis["claim_boundary"], "health_gate_not_final_selection")

    def test_writer_emits_self_verifying_bundle_and_curves(self) -> None:
        analysis = analyze_r6_10k_records(
            _matrix(),
            expected_methods=METHODS,
            expected_state_modes=MODES,
            expected_seeds=SEEDS,
            target_environment_steps=10_000,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle"
            write_r6_10k_analysis_bundle(
                analysis,
                output,
                input_binding={"run_records_sha256": "abc"},
            )

            expected = {
                "analysis.json",
                "run_metrics.csv",
                "configuration_summary.csv",
                "policy_paired.csv",
                "state_paired.csv",
                "training_curve.csv",
                "training_diagnostics.csv",
                "training_objective_curves.png",
                "validation_return_latency.png",
                "continuation_gate.json",
                "README.md",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_entry_count"], 11)
            self.assertNotIn('"winner"', (output / "analysis.json").read_text(encoding="utf-8"))
            curve_lines = (output / "training_curve.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(curve_lines), 1 + 18 * 3)

    def test_writer_preserves_planning_files_and_audit_archive(self) -> None:
        analysis = analyze_r6_10k_records(
            _matrix(),
            expected_methods=METHODS,
            expected_state_modes=MODES,
            expected_seeds=SEEDS,
            target_environment_steps=10_000,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle"
            output.mkdir()
            for name in ("task_plan.md", "findings.md", "progress.md"):
                (output / name).write_text(name, encoding="utf-8")
            archive = output / "audit_previous_bundle"
            archive.mkdir()
            (archive / "analysis.json").write_text("{}", encoding="utf-8")

            write_r6_10k_analysis_bundle(
                analysis,
                output,
                input_binding={"run_records_sha256": "abc"},
            )

            self.assertTrue((archive / "analysis.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
