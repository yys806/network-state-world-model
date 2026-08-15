import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import analyze_r5_multi_seed  # noqa: E402


METRICS = tuple(
    metric.metric_id
    for metric in analyze_r5_multi_seed.METRIC_SPECS
    if metric.metric_id != "protocol_score"
)


def _report(combination: str, seed: int, value: float) -> dict:
    metrics = {metric: {"status": "computed", "value": value} for metric in METRICS}
    return {
        "combination_id": combination,
        "training_seed": seed,
        "status": "completed",
        "best_epoch": 1,
        "epochs_executed": 11,
        "best_validation_protocol_score": value,
        "final_validation": {"metrics": metrics, "validation_protocol_score": value},
        "calibration": {"metrics": metrics, "validation_protocol_score": value},
        "checkpoint_reproduction_score_delta": 0.0,
        "locked_test_accessed": False,
        "runtime_seconds": 1.0,
        "peak_cuda_memory_bytes": 10,
        "parameter_count": 20,
    }


class AnalyzeR5MultiSeedCliTest(unittest.TestCase):
    def test_metric_specs_cover_every_metric_saved_by_formal_reports(self) -> None:
        expected = {
            "event.information_link_activity.auprc",
            "link.active_only_rate.mae",
            "selection.required_continuous.normalized_error",
            "state.dag.unfinished_parent_count.mae",
            "state.flow.remaining_data.mae",
            "state.information_edge.rate.rmse",
            "state.information_node.cpu_backlog.mae",
            "state.information_node.queue.mae",
            "state.physical_edge.distance.rmse",
            "state.physical_edge.relative_speed.rmse",
            "state.physical_node.motion.rmse",
            "state.physical_node.position.rmse",
            "state.task.deadline_remaining.mae",
            "task.lifecycle.macro_f1",
        }

        self.assertEqual(
            {metric.metric_id for metric in analyze_r5_multi_seed.METRIC_SPECS}
            - {"protocol_score"},
            expected,
        )

    def test_main_writes_formal_analysis_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "formal"
            root.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text('{"schema_version":"source"}\n')
            (root / "training_summary.json").write_text(
                json.dumps(
                    {
                        "r5_gpu_training_complete": True,
                        "completed_run_count": 15,
                        "expected_run_count": 15,
                        "failed_run_count": 0,
                        "locked_test_accessed": False,
                    }
                )
            )
            for combination_index, combination in enumerate("ABCDE"):
                for seed in (20260803, 20260804, 20260805):
                    target = root / "combinations" / combination / f"seed_{seed}"
                    target.mkdir(parents=True)
                    (target / "run_report.json").write_text(
                        json.dumps(_report(combination, seed, 5.0 + combination_index))
                    )
            output = Path(tmp) / "analysis"

            exit_code = analyze_r5_multi_seed.main(
                ["--input-dir", str(root), "--output-dir", str(output)]
            )

            self.assertEqual(exit_code, 0)
            result_manifest = json.loads((output / "manifest.json").read_text())
            expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(
                result_manifest["input_binding"]["formal_training_manifest_sha256"],
                expected_hash,
            )
            analysis = json.loads((output / "analysis.json").read_text())
            self.assertEqual(analysis["integrity"]["completed_run_count"], 15)
            self.assertTrue((output / "r2_metric_coverage.csv").is_file())


if __name__ == "__main__":
    unittest.main()
