from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from pi_jwm.r5_analysis import MetricSpec, _describe, paired_comparison
from pi_jwm.r3_preflight_data import R3Window, load_r3_window, make_explicit_batch
from pi_jwm.r4_gpu_screening import R4ValidationAccumulator, collate_explicit_batches


R5_CONFIRMATION_ANALYSIS_SCHEMA = "PIJWM-R5-Module-Confirmation-Analysis-v1"
R6_CANDIDATE_FREEZE_SCHEMA = "PIJWM-R6-Working-Candidate-Freeze-v1"


def window_identity(value: Mapping[str, object]) -> tuple[object, ...]:
    """Return the machine-independent identity frozen by the R5 protocol."""

    fields = (
        "environment_seed",
        "history_start",
        "history_end",
        "target_start",
        "target_end",
        "horizon_steps",
        "split",
    )
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"window identity is missing fields: {missing}")
    return tuple(value[field] for field in fields)


def validate_window_schedule(
    actual: Sequence[Mapping[str, object]],
    stored: Sequence[Mapping[str, object]],
) -> None:
    actual_ids = [window_identity(value) for value in actual]
    stored_ids = [window_identity(value) for value in stored]
    if actual_ids != stored_ids:
        raise ValueError("rebuilt validation window schedule differs from the frozen R5 schedule")


def evaluate_model_by_horizon_cpu(
    model: torch.nn.Module,
    windows: Sequence[R3Window],
    normalization_stats: Mapping[str, object],
    selection_scales: Mapping[str, float],
    *,
    micro_batch_size: int = 4,
) -> dict[str, object]:
    """Replay one frozen validation schedule once and pool both overall and per horizon."""

    if not windows:
        raise ValueError("CPU horizon evaluation requires validation windows")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if any(window.split != "validation" for window in windows):
        raise ValueError("CPU horizon evaluation accepts validation windows only")
    horizons = sorted({int(window.horizon_steps) for window in windows})
    overall = R4ValidationAccumulator(
        normalization_stats,
        selection_scales=selection_scales,
    )
    per_horizon = {
        horizon: R4ValidationAccumulator(
            normalization_stats,
            selection_scales=selection_scales,
        )
        for horizon in horizons
    }
    model = model.to(torch.device("cpu"))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for horizon in horizons:
            selected = [window for window in windows if int(window.horizon_steps) == horizon]
            for start in range(0, len(selected), micro_batch_size):
                batch = collate_explicit_batches(
                    [
                        make_explicit_batch(
                            load_r3_window(window),
                            normalization_stats,
                            device="cpu",
                        )
                        for window in selected[start : start + micro_batch_size]
                    ]
                )
                output = model(batch, rollout_steps=horizon)
                overall.update(output, batch)
                per_horizon[horizon].update(output, batch)
    runtime = time.perf_counter() - started
    overall_report = overall.finalize()
    overall_report["runtime_seconds"] = runtime
    overall_report["mean_window_latency_ms"] = 1000.0 * runtime / max(
        int(overall_report["window_count"]), 1
    )
    horizon_reports = {
        str(horizon): accumulator.finalize()
        for horizon, accumulator in per_horizon.items()
    }
    return {
        "device": "cpu",
        "overall": overall_report,
        "per_horizon": horizon_reports,
    }


def _validate_report_matrix(
    reports: Iterable[Mapping[str, object]],
    *,
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, object]]:
    expected = {
        (str(combination), int(seed))
        for combination in expected_combinations
        for seed in expected_seeds
    }
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for raw in reports:
        report = dict(raw)
        key = (str(report.get("combination_id", "")), int(report.get("training_seed", -1)))
        if key in by_key:
            raise ValueError(f"duplicate confirmation report matrix entry: {key}")
        if report.get("status") not in {"complete", "completed"}:
            raise ValueError(f"confirmation report is not complete: {key}")
        if report.get("locked_test_accessed") is not False:
            raise ValueError(f"confirmation report accessed locked-test: {key}")
        by_key[key] = report
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(
            f"confirmation report matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    return [by_key[key] for key in sorted(by_key)]


def merge_confirmation_reports(
    new_reports: Iterable[Mapping[str, object]],
    reused_reports: Iterable[Mapping[str, object]],
    *,
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, object]]:
    return _validate_report_matrix(
        [*new_reports, *reused_reports],
        expected_combinations=expected_combinations,
        expected_seeds=expected_seeds,
    )


def _report_metric_value(
    report: Mapping[str, object],
    split: str,
    metric: MetricSpec,
) -> float:
    if split not in {"validation", "calibration"}:
        raise ValueError(f"unknown report split: {split}")
    if metric.metric_id == "protocol_score":
        value = (
            report["best_validation_protocol_score"]
            if split == "validation"
            else report["calibration"]["validation_protocol_score"]  # type: ignore[index]
        )
    else:
        section = report["final_validation"] if split == "validation" else report["calibration"]
        record = section["metrics"][metric.metric_id]  # type: ignore[index]
        if record.get("status") != "computed":  # type: ignore[union-attr]
            raise ValueError(f"metric is not computed: {split}.{metric.metric_id}")
        value = record["value"]  # type: ignore[index]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"metric is not finite: {split}.{metric.metric_id}")
    return numeric


def analyze_confirmation_reports(
    reports: Iterable[Mapping[str, object]],
    *,
    metric_specs: Sequence[MetricSpec],
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
    reference_combination: str,
) -> dict[str, object]:
    if reference_combination not in expected_combinations:
        raise ValueError("reference combination is absent from expected combinations")
    validated = _validate_report_matrix(
        reports,
        expected_combinations=expected_combinations,
        expected_seeds=expected_seeds,
    )
    by_key = {
        (str(report["combination_id"]), int(report["training_seed"])): report
        for report in validated
    }
    summary: dict[str, object] = {}
    for combination in expected_combinations:
        reports_for_combination = [
            by_key[(str(combination), int(seed))] for seed in expected_seeds
        ]
        summary[str(combination)] = {
            split: {
                metric.metric_id: _describe(
                    [
                        _report_metric_value(report, split, metric)
                        for report in reports_for_combination
                    ]
                )
                for metric in metric_specs
            }
            for split in ("validation", "calibration")
        }

    paired: dict[str, object] = {}
    for combination in expected_combinations:
        if combination == reference_combination:
            continue
        comparisons: dict[str, object] = {}
        for split in ("validation", "calibration"):
            for metric in metric_specs:
                baseline = {
                    int(seed): _report_metric_value(
                        by_key[(reference_combination, int(seed))], split, metric
                    )
                    for seed in expected_seeds
                }
                candidate = {
                    int(seed): _report_metric_value(
                        by_key[(str(combination), int(seed))], split, metric
                    )
                    for seed in expected_seeds
                }
                comparisons[f"{split}.{metric.metric_id}"] = paired_comparison(
                    baseline,
                    candidate,
                    metric,
                )
        paired[str(combination)] = comparisons

    return {
        "schema_version": R5_CONFIRMATION_ANALYSIS_SCHEMA,
        "selection_status": "working_candidate_diagnostics_only",
        "claim_boundary": (
            "three-seed validation/calibration evidence can freeze R6 working candidates; "
            "it cannot select the final PI-JWM method or expose locked-test"
        ),
        "reference_combination": reference_combination,
        "integrity": {
            "completed_run_count": len(validated),
            "expected_run_count": len(expected_combinations) * len(expected_seeds),
            "failed_run_count": 0,
            "locked_test_accessed": False,
        },
        "metric_directions": {metric.metric_id: metric.direction for metric in metric_specs},
        "combination_summary": summary,
        "paired_vs_reference": paired,
    }


def _horizon_metric_value(record: Mapping[str, object], metric: MetricSpec) -> float:
    if metric.metric_id == "protocol_score":
        value = record["validation_protocol_score"]
    else:
        metric_record = record["metrics"][metric.metric_id]  # type: ignore[index]
        if metric_record.get("status") != "computed":  # type: ignore[union-attr]
            raise ValueError(
                f"horizon metric is not computed: {record.get('combination_id')}/"
                f"{record.get('training_seed')}/{record.get('horizon_steps')}/"
                f"{metric.metric_id}"
            )
        value = metric_record["value"]  # type: ignore[index]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"horizon metric is not finite: {metric.metric_id}")
    return numeric


def analyze_horizon_records(
    records: Iterable[Mapping[str, object]],
    *,
    metric_specs: Sequence[MetricSpec],
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
    expected_horizons: Sequence[int],
    reference_combination: str,
) -> dict[str, object]:
    expected = {
        (str(combination), int(seed), int(horizon))
        for combination in expected_combinations
        for seed in expected_seeds
        for horizon in expected_horizons
    }
    by_key: dict[tuple[str, int, int], dict[str, object]] = {}
    for raw in records:
        record = dict(raw)
        key = (
            str(record.get("combination_id", "")),
            int(record.get("training_seed", -1)),
            int(record.get("horizon_steps", -1)),
        )
        if key in by_key:
            raise ValueError(f"duplicate horizon matrix entry: {key}")
        if int(record.get("window_count", 0)) <= 0:
            raise ValueError(f"horizon matrix entry has no windows: {key}")
        for metric in metric_specs:
            _horizon_metric_value(record, metric)
        by_key[key] = record
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(f"horizon matrix mismatch: missing={missing}, unexpected={unexpected}")

    summary: dict[str, object] = {}
    for combination in expected_combinations:
        summary[str(combination)] = {
            str(horizon): {
                metric.metric_id: _describe(
                    [
                        _horizon_metric_value(
                            by_key[(str(combination), int(seed), int(horizon))], metric
                        )
                        for seed in expected_seeds
                    ]
                )
                for metric in metric_specs
            }
            for horizon in expected_horizons
        }

    paired: dict[str, object] = {}
    for combination in expected_combinations:
        if combination == reference_combination:
            continue
        comparisons: dict[str, object] = {}
        for horizon in expected_horizons:
            for metric in metric_specs:
                baseline = {
                    int(seed): _horizon_metric_value(
                        by_key[(reference_combination, int(seed), int(horizon))], metric
                    )
                    for seed in expected_seeds
                }
                candidate = {
                    int(seed): _horizon_metric_value(
                        by_key[(str(combination), int(seed), int(horizon))], metric
                    )
                    for seed in expected_seeds
                }
                comparisons[f"h{int(horizon)}.{metric.metric_id}"] = paired_comparison(
                    baseline,
                    candidate,
                    metric,
                )
        paired[str(combination)] = comparisons

    return {
        "schema_version": R5_CONFIRMATION_ANALYSIS_SCHEMA,
        "reference_combination": reference_combination,
        "integrity": {
            "record_count": len(by_key),
            "expected_record_count": len(expected),
            "all_seed_horizon_cells_computable": True,
            "locked_test_accessed": False,
        },
        "metric_directions": {metric.metric_id: metric.direction for metric in metric_specs},
        "horizon_summary": summary,
        "paired_vs_reference": paired,
    }


def _summary_mean(
    analysis: Mapping[str, object],
    combination: str,
    metric_id: str,
) -> float:
    return float(
        analysis["combination_summary"][combination]["validation"][metric_id]["mean"]  # type: ignore[index]
    )


def freeze_r6_candidate_set(
    aggregate_analysis: Mapping[str, object],
    horizon_analysis: Mapping[str, object],
    *,
    expected_combinations: Sequence[str],
    reference_combination: str,
    ablation_combinations: Sequence[str] = (),
) -> dict[str, object]:
    if not horizon_analysis["integrity"]["all_seed_horizon_cells_computable"]:  # type: ignore[index]
        raise ValueError("cannot freeze candidates with incomplete horizon evidence")
    expected_seed_count = int(aggregate_analysis["integrity"]["completed_run_count"]) // len(  # type: ignore[index]
        expected_combinations
    )
    baseline_rate = _summary_mean(
        aggregate_analysis, reference_combination, "state.information_edge.rate.rmse"
    )
    baseline_active_rate = _summary_mean(
        aggregate_analysis, reference_combination, "link.active_only_rate.mae"
    )
    baseline_task = _summary_mean(
        aggregate_analysis, reference_combination, "task.lifecycle.macro_f1"
    )
    baseline_continuous = _summary_mean(
        aggregate_analysis,
        reference_combination,
        "selection.required_continuous.normalized_error",
    )

    ablations = {str(value) for value in ablation_combinations}
    overall_challengers: list[str] = []
    task_specialists: list[str] = []
    continuous_specialists: list[str] = []
    decisions: dict[str, object] = {}
    comparisons = aggregate_analysis["paired_vs_reference"]  # type: ignore[index]
    for combination in expected_combinations:
        combination = str(combination)
        if combination == reference_combination:
            decisions[combination] = {
                "role": "mandatory_reference_control",
                "overall_gate_passed": True,
            }
            continue
        paired = comparisons[combination]
        protocol_pair = paired["validation.protocol_score"]
        rate_mean = _summary_mean(
            aggregate_analysis, combination, "state.information_edge.rate.rmse"
        )
        active_rate_mean = _summary_mean(
            aggregate_analysis, combination, "link.active_only_rate.mae"
        )
        task_mean = _summary_mean(
            aggregate_analysis, combination, "task.lifecycle.macro_f1"
        )
        continuous_mean = _summary_mean(
            aggregate_analysis,
            combination,
            "selection.required_continuous.normalized_error",
        )
        gate_checks = {
            "protocol_better_all_seeds": (
                int(protocol_pair["wins"]) == expected_seed_count
                and int(protocol_pair["losses"]) == 0
            ),
            "information_rate_regression_le_2pct": rate_mean <= baseline_rate * 1.02,
            "active_only_rate_regression_le_2pct": (
                active_rate_mean <= baseline_active_rate * 1.02
            ),
            "task_macro_f1_absolute_regression_le_0_02": task_mean >= baseline_task - 0.02,
            "all_seed_horizon_cells_computable": True,
        }
        overall_passed = all(gate_checks.values()) and combination not in ablations
        if overall_passed:
            overall_challengers.append(combination)

        task_pair = paired["validation.task.lifecycle.macro_f1"]
        task_specialist = (
            combination not in ablations
            and baseline_task > 0.0
            and task_mean >= baseline_task * 1.10
            and int(task_pair["wins"]) >= 2
        )
        if task_specialist:
            task_specialists.append(combination)

        continuous_pair = paired[
            "validation.selection.required_continuous.normalized_error"
        ]
        continuous_specialist = (
            combination not in ablations
            and baseline_continuous > 0.0
            and continuous_mean <= baseline_continuous * 0.90
            and int(continuous_pair["wins"]) >= 2
        )
        if continuous_specialist:
            continuous_specialists.append(combination)

        decisions[combination] = {
            "role": "ablation_control" if combination in ablations else "evaluated_candidate",
            "overall_gate_checks": gate_checks,
            "overall_gate_passed": overall_passed,
            "task_lifecycle_specialist_gate_passed": task_specialist,
            "continuous_state_specialist_gate_passed": continuous_specialist,
        }

    eligible_primary = [reference_combination, *overall_challengers]
    primary = min(
        eligible_primary,
        key=lambda combination: _summary_mean(
            aggregate_analysis, combination, "protocol_score"
        ),
    )
    return {
        "schema_version": R6_CANDIDATE_FREEZE_SCHEMA,
        "freeze_scope": "R6_working_candidate_set_only",
        "reference_control": reference_combination,
        "primary_working_candidate": primary,
        "overall_challengers": sorted(overall_challengers),
        "task_lifecycle_specialists": sorted(task_specialists),
        "continuous_state_specialists": sorted(continuous_specialists),
        "ablation_controls": sorted(ablations),
        "candidate_decisions": decisions,
        "gate_definition": {
            "overall": (
                "protocol score improves on all three paired seeds; information-edge rate RMSE "
                "and active-only rate MAE regress by at most 2%; task Macro-F1 absolute "
                "regression is at most 0.02; every seed/horizon cell is computable"
            ),
            "specialist": (
                "at least 10% mean improvement in the declared metric family and at least "
                "two paired-seed wins"
            ),
        },
        "r5_1_candidate_set_frozen": True,
        "r6_cpu_preflight_ready": True,
        "r6_gpu_strategy_training_ready": False,
        "final_method_frozen": False,
        "locked_test_accessed": False,
    }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_confirmation_analysis_bundle(
    output_dir: str | Path,
    *,
    aggregate_analysis: Mapping[str, object],
    horizon_analysis: Mapping[str, object],
    horizon_records: Sequence[Mapping[str, object]],
    candidate_freeze: Mapping[str, object],
    input_binding: Mapping[str, str],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(
            {"aggregate": aggregate_analysis, "horizon": horizon_analysis},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "candidate_freeze.json").write_text(
        json.dumps(candidate_freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate_rows: list[dict[str, object]] = []
    for combination, split_map in aggregate_analysis["combination_summary"].items():  # type: ignore[union-attr]
        for split, metric_map in split_map.items():
            for metric_id, summary in metric_map.items():
                aggregate_rows.append(
                    {
                        "combination_id": combination,
                        "split": split,
                        "metric_id": metric_id,
                        **summary,
                    }
                )
    _write_csv(
        output / "aggregate_summary.csv",
        (
            "combination_id",
            "split",
            "metric_id",
            "count",
            "mean",
            "sample_std",
            "median",
            "minimum",
            "maximum",
            "mean_ci95_low",
            "mean_ci95_high",
        ),
        aggregate_rows,
    )

    paired_rows: list[dict[str, object]] = []
    for combination, metric_map in aggregate_analysis["paired_vs_reference"].items():  # type: ignore[union-attr]
        for metric_key, result in metric_map.items():
            paired_rows.append(
                {
                    "combination_id": combination,
                    "metric_key": metric_key,
                    "direction": result["direction"],
                    "mean_benefit": result["benefit"]["mean"],
                    "mean_percent_benefit": result["mean_percent_benefit"],
                    "wins": result["wins"],
                    "ties": result["ties"],
                    "losses": result["losses"],
                    "exact_sign_flip_p_two_sided": result[
                        "exact_sign_flip_p_two_sided"
                    ],
                }
            )
    paired_fields = (
        "combination_id",
        "metric_key",
        "direction",
        "mean_benefit",
        "mean_percent_benefit",
        "wins",
        "ties",
        "losses",
        "exact_sign_flip_p_two_sided",
    )
    _write_csv(output / "paired_vs_B.csv", paired_fields, paired_rows)

    horizon_metric_rows: list[dict[str, object]] = []
    for record in horizon_records:
        horizon_metric_rows.append(
            {
                "combination_id": record["combination_id"],
                "training_seed": record["training_seed"],
                "horizon_steps": record["horizon_steps"],
                "window_count": record["window_count"],
                "metric_id": "protocol_score",
                "value": record["validation_protocol_score"],
            }
        )
        for metric_id, metric_record in record["metrics"].items():  # type: ignore[union-attr]
            horizon_metric_rows.append(
                {
                    "combination_id": record["combination_id"],
                    "training_seed": record["training_seed"],
                    "horizon_steps": record["horizon_steps"],
                    "window_count": record["window_count"],
                    "metric_id": metric_id,
                    "value": metric_record.get("value"),
                }
            )
    _write_csv(
        output / "horizon_run_metrics.csv",
        ("combination_id", "training_seed", "horizon_steps", "window_count", "metric_id", "value"),
        horizon_metric_rows,
    )

    horizon_summary_rows: list[dict[str, object]] = []
    for combination, horizon_map in horizon_analysis["horizon_summary"].items():  # type: ignore[union-attr]
        for horizon, metric_map in horizon_map.items():
            for metric_id, summary in metric_map.items():
                horizon_summary_rows.append(
                    {
                        "combination_id": combination,
                        "horizon_steps": horizon,
                        "metric_id": metric_id,
                        **summary,
                    }
                )
    _write_csv(
        output / "horizon_summary.csv",
        (
            "combination_id",
            "horizon_steps",
            "metric_id",
            "count",
            "mean",
            "sample_std",
            "median",
            "minimum",
            "maximum",
            "mean_ci95_low",
            "mean_ci95_high",
        ),
        horizon_summary_rows,
    )

    horizon_paired_rows: list[dict[str, object]] = []
    for combination, metric_map in horizon_analysis["paired_vs_reference"].items():  # type: ignore[union-attr]
        for metric_key, result in metric_map.items():
            horizon_paired_rows.append(
                {
                    "combination_id": combination,
                    "metric_key": metric_key,
                    "direction": result["direction"],
                    "mean_benefit": result["benefit"]["mean"],
                    "mean_percent_benefit": result["mean_percent_benefit"],
                    "wins": result["wins"],
                    "ties": result["ties"],
                    "losses": result["losses"],
                    "exact_sign_flip_p_two_sided": result[
                        "exact_sign_flip_p_two_sided"
                    ],
                }
            )
    _write_csv(output / "horizon_paired_vs_B.csv", paired_fields, horizon_paired_rows)

    readme = (
        "# PI-JWM R5.1 Module Confirmation Analysis\n\n"
        "This bundle contains three-seed validation/calibration analysis and validation-only "
        "1/5/20-step diagnostics. B is the mandatory reference control. Candidate roles are "
        "working inputs for R6, not a final-method or winner claim. locked-test remains sealed.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    payload_files = sorted(path for path in output.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema_version": R5_CONFIRMATION_ANALYSIS_SCHEMA,
        "manifest_entry_count": len(payload_files),
        "input_binding": dict(input_binding),
        "locked_test_accessed": False,
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in payload_files
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
