import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_analysis import (  # noqa: E402
    MetricSpec,
    analyze_reports,
    audit_r2_metric_coverage,
    load_complete_report_matrix,
    paired_comparison,
    write_analysis_bundle,
)


def _report(combination: str, seed: int, lower: float, higher: float) -> dict:
    metrics = {
        "lower": {"status": "computed", "value": lower},
        "higher": {"status": "computed", "value": higher},
    }
    return {
        "combination_id": combination,
        "training_seed": seed,
        "status": "complete",
        "best_epoch": 3,
        "epochs_executed": 8,
        "best_validation_protocol_score": lower,
        "final_validation": {"metrics": metrics, "validation_protocol_score": lower},
        "calibration": {"metrics": metrics, "validation_protocol_score": lower},
        "checkpoint_reproduction_score_delta": 0.0,
        "locked_test_accessed": False,
        "runtime_seconds": 1.0,
        "peak_cuda_memory_bytes": 10,
        "parameter_count": 20,
    }


class R5MultiSeedAnalysisTest(unittest.TestCase):
    def test_r2_coverage_distinguishes_computed_and_deferred_metrics(self) -> None:
        reports = [
            _report(combination, seed, 1.0, 2.0)
            for combination in ("A", "B")
            for seed in (1, 2, 3)
        ]
        registry = [
            {"metric_id": "lower", "layer": "state_prediction"},
            {"metric_id": "system.latency.p95", "layer": "system"},
            {"metric_id": "deployment.inference_latency.p95", "layer": "deployment"},
        ]

        coverage = audit_r2_metric_coverage(reports, registry)

        self.assertEqual(coverage[0]["coverage_status"], "computed_all_runs")
        self.assertEqual(coverage[0]["computed_run_count"], 6)
        self.assertEqual(
            coverage[1]["coverage_status"],
            "requires_policy_execution",
        )
        self.assertEqual(coverage[1]["computed_run_count"], 0)
        self.assertEqual(
            coverage[2]["coverage_status"],
            "requires_timed_post_evaluation",
        )

    def test_paired_benefit_respects_direction_and_seed_alignment(self) -> None:
        baseline = {1: 5.0, 2: 6.0, 3: 7.0}
        candidate_lower = {3: 6.0, 1: 4.0, 2: 5.0}
        candidate_higher = {2: 7.0, 3: 8.0, 1: 6.0}

        lower = paired_comparison(
            baseline,
            candidate_lower,
            MetricSpec("lower", "lower"),
        )
        higher = paired_comparison(
            baseline,
            candidate_higher,
            MetricSpec("higher", "higher"),
        )

        self.assertEqual(lower["benefit_by_seed"], {"1": 1.0, "2": 1.0, "3": 1.0})
        self.assertEqual(higher["benefit_by_seed"], {"1": 1.0, "2": 1.0, "3": 1.0})
        self.assertEqual(lower["wins"], 3)
        self.assertEqual(lower["losses"], 0)
        self.assertAlmostEqual(lower["exact_sign_flip_p_two_sided"], 0.25)

    def test_loader_rejects_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "combinations" / "A" / "seed_1"
            path.mkdir(parents=True)
            (path / "run_report.json").write_text(json.dumps(_report("A", 1, 1.0, 1.0)))

            with self.assertRaisesRegex(ValueError, "report matrix"):
                load_complete_report_matrix(
                    root,
                    expected_combinations=("A", "B"),
                    expected_seeds=(1, 2, 3),
                )

    def test_analysis_is_descriptive_and_flags_budget_censoring(self) -> None:
        reports = []
        for combination, offset in (("A", 0.0), ("B", -1.0)):
            for seed in (1, 2, 3):
                report = _report(combination, seed, 5.0 + seed + offset, 5.0 + seed - offset)
                if combination == "B" and seed == 1:
                    report["best_epoch"] = 100
                    report["epochs_executed"] = 100
                reports.append(report)

        analysis = analyze_reports(
            reports,
            metric_specs=(MetricSpec("lower", "lower"), MetricSpec("higher", "higher")),
            expected_combinations=("A", "B"),
            expected_seeds=(1, 2, 3),
            max_epochs=100,
        )

        self.assertNotIn("winner", analysis)
        self.assertEqual(analysis["selection_status"], "descriptive_only")
        self.assertEqual(analysis["integrity"]["completed_run_count"], 6)
        self.assertEqual(analysis["convergence"]["B"]["budget_censored_run_count"], 1)
        self.assertEqual(
            analysis["paired_vs_A"]["B"]["validation.lower"]["wins"],
            3,
        )
        self.assertEqual(analysis["paired_vs_B"], {})

    def test_writer_emits_a_self_verifying_bundle_without_winner(self) -> None:
        reports = [
            _report(combination, seed, 5.0 + seed + offset, 5.0 + seed - offset)
            for combination, offset in (("A", 0.0), ("B", -1.0))
            for seed in (1, 2, 3)
        ]
        analysis = analyze_reports(
            reports,
            metric_specs=(MetricSpec("lower", "lower"), MetricSpec("higher", "higher")),
            expected_combinations=("A", "B"),
            expected_seeds=(1, 2, 3),
            max_epochs=100,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle"
            write_analysis_bundle(
                analysis,
                output,
                input_binding={"formal_training_manifest_sha256": "abc"},
                combination_labels={"A": "baseline", "B": "candidate"},
            )

            expected = {
                "analysis.json",
                "combination_summary.csv",
                "paired_vs_A.csv",
                "paired_vs_B.csv",
                "convergence.csv",
                "README.md",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["manifest_entry_count"], 6)
            self.assertEqual(manifest["input_binding"]["formal_training_manifest_sha256"], "abc")
            self.assertNotIn('"winner"', (output / "analysis.json").read_text())
            self.assertIn("candidate", (output / "README.md").read_text())

    def test_analysis_compares_added_modules_against_graph_rssm(self) -> None:
        reports = []
        for combination, offset in (("A", 0.0), ("B", -1.0), ("C", -2.0)):
            reports.extend(
                _report(combination, seed, 5.0 + seed + offset, 5.0 + seed - offset)
                for seed in (1, 2, 3)
            )
        analysis = analyze_reports(
            reports,
            metric_specs=(MetricSpec("lower", "lower"), MetricSpec("higher", "higher")),
            expected_combinations=("A", "B", "C"),
            expected_seeds=(1, 2, 3),
            max_epochs=100,
        )

        self.assertEqual(
            analysis["paired_vs_B"]["C"]["validation.lower"]["wins"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
