from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_analysis import (  # noqa: E402
    MetricSpec,
    analyze_reports,
    audit_r2_metric_coverage,
    load_complete_report_matrix,
    write_analysis_bundle,
)
from pi_jwm.evaluation_protocol_v3 import build_metric_registry  # noqa: E402


COMBINATIONS = ("A", "B", "C", "D", "E")
TRAINING_SEEDS = (20260803, 20260804, 20260805)
COMBINATION_LABELS = {
    "A": "Graph-GRU reference",
    "B": "Graph-RSSM",
    "C": "Graph-RSSM + heteroscedastic head",
    "D": "Graph-RSSM + explicit DAG message",
    "E": "Graph-RSSM + soft presence",
}
METRIC_SPECS = (
    MetricSpec("protocol_score", "lower"),
    MetricSpec("state.physical_node.position.rmse", "lower"),
    MetricSpec("state.physical_node.motion.rmse", "lower"),
    MetricSpec("state.physical_edge.distance.rmse", "lower"),
    MetricSpec("state.physical_edge.relative_speed.rmse", "lower"),
    MetricSpec("state.information_node.queue.mae", "lower"),
    MetricSpec("state.information_node.cpu_backlog.mae", "lower"),
    MetricSpec("state.information_edge.rate.rmse", "lower"),
    MetricSpec("link.active_only_rate.mae", "lower"),
    MetricSpec("state.flow.remaining_data.mae", "lower"),
    MetricSpec("state.task.deadline_remaining.mae", "lower"),
    MetricSpec("state.dag.unfinished_parent_count.mae", "lower"),
    MetricSpec("selection.required_continuous.normalized_error", "lower"),
    MetricSpec("event.information_link_activity.auprc", "higher"),
    MetricSpec("task.lifecycle.macro_f1", "higher"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_training_summary(input_dir: Path) -> dict:
    summary_path = input_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "r5_gpu_training_complete": True,
        "completed_run_count": 15,
        "expected_run_count": 15,
        "failed_run_count": 0,
        "locked_test_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"R5 formal training summary is not complete: {mismatches}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze PI-JWM R5 formal multi-seed results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    _validate_training_summary(input_dir)
    source_manifest = input_dir / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"R5 formal manifest is missing: {source_manifest}")
    reports = load_complete_report_matrix(
        input_dir,
        expected_combinations=COMBINATIONS,
        expected_seeds=TRAINING_SEEDS,
    )
    analysis = analyze_reports(
        reports,
        metric_specs=METRIC_SPECS,
        expected_combinations=COMBINATIONS,
        expected_seeds=TRAINING_SEEDS,
        max_epochs=100,
    )
    input_binding = {
        "formal_training_manifest_sha256": _sha256(source_manifest),
        "formal_training_summary_sha256": _sha256(input_dir / "training_summary.json"),
    }
    analysis["input_binding"] = input_binding
    analysis["combination_labels"] = COMBINATION_LABELS
    analysis["r2_metric_coverage"] = audit_r2_metric_coverage(
        reports,
        build_metric_registry(),
    )
    write_analysis_bundle(
        analysis,
        output_dir,
        input_binding=input_binding,
        combination_labels=COMBINATION_LABELS,
    )
    print(json.dumps({"output_dir": str(output_dir), **analysis["integrity"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
