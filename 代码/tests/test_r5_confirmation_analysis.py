from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_analysis import MetricSpec  # noqa: E402
from pi_jwm.r5_confirmation_analysis import (  # noqa: E402
    analyze_confirmation_reports,
    analyze_horizon_records,
    evaluate_model_by_horizon_cpu,
    freeze_r6_candidate_set,
    merge_confirmation_reports,
    validate_window_schedule,
    window_identity,
    write_confirmation_analysis_bundle,
)


SEEDS = (11, 12, 13)
COMBINATIONS = ("B", "F", "G", "H", "J")
METRICS = (
    MetricSpec("protocol_score", "lower"),
    MetricSpec("state.information_edge.rate.rmse", "lower"),
    MetricSpec("link.active_only_rate.mae", "lower"),
    MetricSpec("task.lifecycle.macro_f1", "higher"),
    MetricSpec("selection.required_continuous.normalized_error", "lower"),
)


def _report(
    combination: str,
    seed: int,
    *,
    protocol: float,
    rate: float,
    active_rate: float,
    task_f1: float,
    continuous: float,
) -> dict:
    metrics = {
        "state.information_edge.rate.rmse": {"status": "computed", "value": rate},
        "link.active_only_rate.mae": {"status": "computed", "value": active_rate},
        "task.lifecycle.macro_f1": {"status": "computed", "value": task_f1},
        "selection.required_continuous.normalized_error": {
            "status": "computed",
            "value": continuous,
        },
    }
    return {
        "combination_id": combination,
        "training_seed": seed,
        "status": "completed",
        "locked_test_accessed": False,
        "best_validation_protocol_score": protocol,
        "final_validation": {"validation_protocol_score": protocol, "metrics": metrics},
        "calibration": {"validation_protocol_score": protocol + 0.1, "metrics": metrics},
    }


def _matrix() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        rows.append(
            _report(
                "B",
                seed,
                protocol=4.5 + 0.01 * (seed - 12),
                rate=30.0,
                active_rate=400.0,
                task_f1=0.15,
                continuous=0.50,
            )
        )
        rows.append(
            _report(
                "F",
                seed,
                protocol=4.7,
                rate=31.0,
                active_rate=410.0,
                task_f1=0.14,
                continuous=0.52,
            )
        )
        rows.append(
            _report(
                "G",
                seed,
                protocol=4.55,
                rate=30.2,
                active_rate=402.0,
                task_f1=0.20,
                continuous=0.51,
            )
        )
        rows.append(
            _report(
                "H",
                seed,
                protocol=4.6,
                rate=30.3,
                active_rate=403.0,
                task_f1=0.15,
                continuous=0.50,
            )
        )
        rows.append(
            _report(
                "J",
                seed,
                protocol=4.2 + 0.01 * (seed - 12),
                rate=42.0,
                active_rate=405.0,
                task_f1=0.09,
                continuous=0.40,
            )
        )
    return rows


def _horizon_records() -> list[dict]:
    records: list[dict] = []
    report_by_key = {
        (row["combination_id"], row["training_seed"]): row for row in _matrix()
    }
    for combination in COMBINATIONS:
        for seed in SEEDS:
            report = report_by_key[(combination, seed)]
            for horizon in (1, 5, 20):
                records.append(
                    {
                        "combination_id": combination,
                        "training_seed": seed,
                        "horizon_steps": horizon,
                        "window_count": 12,
                        "validation_protocol_score": report[
                            "best_validation_protocol_score"
                        ],
                        "metrics": report["final_validation"]["metrics"],
                    }
                )
    return records


class R5ConfirmationAnalysisTest(unittest.TestCase):
    def test_cpu_horizon_evaluator_rejects_empty_input_before_model_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires validation windows"):
            evaluate_model_by_horizon_cpu(
                object(),  # type: ignore[arg-type]
                [],
                {},
                {},
            )

    def test_window_identity_ignores_machine_specific_tensor_path(self) -> None:
        stored = {
            "tensor_path": "/remote/data/a.pt",
            "environment_seed": 7,
            "history_start": 0,
            "history_end": 8,
            "target_start": 8,
            "target_end": 13,
            "horizon_steps": 5,
            "split": "validation",
        }
        local = dict(stored, tensor_path="D:/local/data/a.pt")

        self.assertEqual(window_identity(stored), window_identity(local))
        validate_window_schedule([local], [stored])

    def test_merge_requires_exact_complete_locked_test_safe_matrix(self) -> None:
        matrix = _matrix()
        merged = merge_confirmation_reports(
            matrix[3:],
            matrix[:3],
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
        )
        self.assertEqual(len(merged), 15)

        unsafe = [dict(row) for row in matrix]
        unsafe[0]["locked_test_accessed"] = True
        with self.assertRaisesRegex(ValueError, "locked-test"):
            merge_confirmation_reports(
                unsafe,
                [],
                expected_combinations=COMBINATIONS,
                expected_seeds=SEEDS,
            )

    def test_analysis_is_generic_against_B_and_preserves_metric_direction(self) -> None:
        analysis = analyze_confirmation_reports(
            _matrix(),
            metric_specs=METRICS,
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
            reference_combination="B",
        )

        self.assertEqual(analysis["integrity"]["completed_run_count"], 15)
        self.assertEqual(
            analysis["paired_vs_reference"]["J"]["validation.protocol_score"]["wins"],
            3,
        )
        self.assertEqual(
            analysis["paired_vs_reference"]["J"]
            ["validation.state.information_edge.rate.rmse"]["losses"],
            3,
        )
        self.assertNotIn("winner", analysis)

    def test_horizon_analysis_rejects_missing_cell_in_three_by_three_matrix(self) -> None:
        records = _horizon_records()
        with self.assertRaisesRegex(ValueError, "horizon matrix"):
            analyze_horizon_records(
                records[:-1],
                metric_specs=METRICS,
                expected_combinations=COMBINATIONS,
                expected_seeds=SEEDS,
                expected_horizons=(1, 5, 20),
                reference_combination="B",
            )

    def test_candidate_freeze_applies_predeclared_gates_without_finalizing_method(self) -> None:
        aggregate = analyze_confirmation_reports(
            _matrix(),
            metric_specs=METRICS,
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
            reference_combination="B",
        )
        horizons = analyze_horizon_records(
            _horizon_records(),
            metric_specs=METRICS,
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
            expected_horizons=(1, 5, 20),
            reference_combination="B",
        )

        frozen = freeze_r6_candidate_set(
            aggregate,
            horizons,
            expected_combinations=COMBINATIONS,
            reference_combination="B",
            ablation_combinations=("F",),
        )

        self.assertEqual(frozen["primary_working_candidate"], "B")
        self.assertIn("G", frozen["task_lifecycle_specialists"])
        self.assertIn("J", frozen["continuous_state_specialists"])
        self.assertNotIn("J", frozen["overall_challengers"])
        self.assertEqual(frozen["ablation_controls"], ["F"])
        self.assertTrue(frozen["r5_1_candidate_set_frozen"])
        self.assertTrue(frozen["r6_cpu_preflight_ready"])
        self.assertFalse(frozen["r6_gpu_strategy_training_ready"])
        self.assertFalse(frozen["final_method_frozen"])

    def test_writer_emits_self_verifying_bundle_and_no_winner_claim(self) -> None:
        aggregate = analyze_confirmation_reports(
            _matrix(),
            metric_specs=METRICS,
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
            reference_combination="B",
        )
        horizons = analyze_horizon_records(
            _horizon_records(),
            metric_specs=METRICS,
            expected_combinations=COMBINATIONS,
            expected_seeds=SEEDS,
            expected_horizons=(1, 5, 20),
            reference_combination="B",
        )
        frozen = freeze_r6_candidate_set(
            aggregate,
            horizons,
            expected_combinations=COMBINATIONS,
            reference_combination="B",
            ablation_combinations=("F",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle"
            write_confirmation_analysis_bundle(
                output,
                aggregate_analysis=aggregate,
                horizon_analysis=horizons,
                horizon_records=_horizon_records(),
                candidate_freeze=frozen,
                input_binding={"confirmation_manifest_sha256": "abc"},
            )

            expected = {
                "analysis.json",
                "aggregate_summary.csv",
                "paired_vs_B.csv",
                "horizon_run_metrics.csv",
                "horizon_summary.csv",
                "horizon_paired_vs_B.csv",
                "candidate_freeze.json",
                "README.md",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_entry_count"], 8)
            self.assertNotIn('"winner"', (output / "analysis.json").read_text())


if __name__ == "__main__":
    unittest.main()
