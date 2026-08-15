from __future__ import annotations

import itertools
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


R5_ANALYSIS_SCHEMA = "PIJWM-R5-Multi-Seed-Analysis-v1"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in {"lower", "higher"}:
            raise ValueError("metric direction must be 'lower' or 'higher'")


def audit_r2_metric_coverage(
    reports: Iterable[Mapping[str, object]],
    metric_registry: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Report which frozen R2 metrics are present in the R5 offline reports."""

    report_rows = [dict(report) for report in reports]
    if not report_rows:
        raise ValueError("R2 metric coverage requires at least one R5 report")
    total = len(report_rows)
    result: list[dict[str, object]] = []
    policy_layers = {"system", "resource", "safety", "decision"}
    for registry_row in metric_registry:
        metric_id = str(registry_row["metric_id"])
        layer = str(registry_row["layer"])
        computed = 0
        recorded = 0
        for report in report_rows:
            split_records = []
            for split_name in ("final_validation", "calibration"):
                split = report.get(split_name, {})
                metrics = split.get("metrics", {}) if isinstance(split, Mapping) else {}
                record = metrics.get(metric_id) if isinstance(metrics, Mapping) else None
                if isinstance(record, Mapping):
                    split_records.append(record)
            recorded += int(len(split_records) == 2)
            computed += int(
                len(split_records) == 2
                and all(record.get("status") == "computed" for record in split_records)
            )
        if computed == total:
            status = "computed_all_runs"
        elif computed:
            status = "partially_computed"
        elif metric_id == "deployment.inference_latency.p95":
            status = "requires_timed_post_evaluation"
        elif layer in policy_layers:
            status = "requires_policy_execution"
        elif layer == "uncertainty":
            status = "requires_distribution_post_evaluation"
        else:
            status = "not_recorded_by_r5_predictive_evaluation"
        result.append(
            {
                "metric_id": metric_id,
                "layer": layer,
                "coverage_status": status,
                "computed_run_count": computed,
                "recorded_run_count": recorded,
                "expected_run_count": total,
            }
        )
    return result


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("summary values must be finite")
    count = len(numeric)
    mean = statistics.fmean(numeric)
    sample_std = statistics.stdev(numeric) if count > 1 else 0.0
    t_critical = {
        1: 12.7062047364,
        2: 4.30265272991,
        3: 3.18244630528,
        4: 2.7764451052,
        5: 2.57058183564,
        6: 2.44691184879,
        7: 2.36462425101,
        8: 2.3060041352,
        9: 2.26215716285,
        10: 2.22813885196,
    }.get(count - 1, 1.95996398454)
    half_width = t_critical * sample_std / math.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "sample_std": sample_std,
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean_ci95_low": mean - half_width,
        "mean_ci95_high": mean + half_width,
    }


def _exact_sign_flip_p_two_sided(benefits: Sequence[float]) -> float:
    values = [float(value) for value in benefits]
    if not values:
        raise ValueError("sign-flip test requires paired benefits")
    observed = abs(statistics.fmean(values))
    exceedances = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        exceedances += int(permuted >= observed - 1e-12)
        total += 1
    return exceedances / total


def paired_comparison(
    baseline_by_seed: Mapping[int, float],
    candidate_by_seed: Mapping[int, float],
    metric: MetricSpec,
) -> dict[str, object]:
    baseline_seeds = set(baseline_by_seed)
    candidate_seeds = set(candidate_by_seed)
    if baseline_seeds != candidate_seeds or not baseline_seeds:
        raise ValueError("paired comparison requires identical non-empty seed sets")
    seeds = sorted(baseline_seeds)
    benefits: list[float] = []
    percent_benefits: list[float] = []
    benefit_by_seed: dict[str, float] = {}
    for seed in seeds:
        baseline = float(baseline_by_seed[seed])
        candidate = float(candidate_by_seed[seed])
        benefit = baseline - candidate if metric.direction == "lower" else candidate - baseline
        benefits.append(benefit)
        benefit_by_seed[str(seed)] = benefit
        if baseline != 0.0:
            percent_benefits.append(100.0 * benefit / abs(baseline))
    tolerance = 1e-12
    return {
        "direction": metric.direction,
        "benefit_definition": (
            "baseline_minus_candidate" if metric.direction == "lower" else "candidate_minus_baseline"
        ),
        "benefit_by_seed": benefit_by_seed,
        "benefit": _describe(benefits),
        "mean_percent_benefit": (
            statistics.fmean(percent_benefits) if len(percent_benefits) == len(benefits) else None
        ),
        "wins": sum(value > tolerance for value in benefits),
        "ties": sum(abs(value) <= tolerance for value in benefits),
        "losses": sum(value < -tolerance for value in benefits),
        "exact_sign_flip_p_two_sided": _exact_sign_flip_p_two_sided(benefits),
    }


def _validate_report_matrix(
    reports: Iterable[Mapping[str, object]],
    *,
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, object]]:
    expected = {(combination, int(seed)) for combination in expected_combinations for seed in expected_seeds}
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for raw in reports:
        report = dict(raw)
        key = (str(report.get("combination_id", "")), int(report.get("training_seed", -1)))
        if key in by_key:
            raise ValueError(f"duplicate R5 report matrix entry: {key}")
        if report.get("status") not in {"complete", "completed"}:
            raise ValueError(f"R5 report is not complete: {key}")
        if report.get("locked_test_accessed") is not False:
            raise ValueError(f"R5 report accessed locked-test: {key}")
        by_key[key] = report
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(f"R5 report matrix mismatch: missing={missing}, unexpected={unexpected}")
    return [by_key[key] for key in sorted(by_key)]


def load_complete_report_matrix(
    root: str | Path,
    *,
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, object]]:
    report_paths = sorted(Path(root).glob("combinations/*/seed_*/run_report.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    return _validate_report_matrix(
        reports,
        expected_combinations=expected_combinations,
        expected_seeds=expected_seeds,
    )


def _metric_value(report: Mapping[str, object], split: str, metric: MetricSpec) -> float:
    if split not in {"validation", "calibration"}:
        raise ValueError(f"unknown split: {split}")
    if metric.metric_id == "protocol_score":
        if split == "validation":
            value = report["best_validation_protocol_score"]
        else:
            value = report["calibration"]["validation_protocol_score"]  # type: ignore[index]
    else:
        section = report["final_validation"] if split == "validation" else report["calibration"]
        metric_record = section["metrics"][metric.metric_id]  # type: ignore[index]
        if metric_record.get("status") != "computed":  # type: ignore[union-attr]
            raise ValueError(f"metric is not computed: {split}.{metric.metric_id}")
        value = metric_record["value"]  # type: ignore[index]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"metric is not finite: {split}.{metric.metric_id}")
    return numeric


def analyze_reports(
    reports: Iterable[Mapping[str, object]],
    *,
    metric_specs: Sequence[MetricSpec],
    expected_combinations: Sequence[str],
    expected_seeds: Sequence[int],
    max_epochs: int,
) -> dict[str, object]:
    validated = _validate_report_matrix(
        reports,
        expected_combinations=expected_combinations,
        expected_seeds=expected_seeds,
    )
    by_key = {
        (str(report["combination_id"]), int(report["training_seed"])): report
        for report in validated
    }
    combination_summary: dict[str, object] = {}
    convergence: dict[str, object] = {}
    for combination in expected_combinations:
        combination_reports = [by_key[(combination, int(seed))] for seed in expected_seeds]
        split_summary: dict[str, object] = {}
        for split in ("validation", "calibration"):
            split_summary[split] = {
                metric.metric_id: _describe(
                    [_metric_value(report, split, metric) for report in combination_reports]
                )
                for metric in metric_specs
            }
        combination_summary[combination] = split_summary
        best_epochs = [int(report["best_epoch"]) for report in combination_reports]
        executed_epochs = [int(report["epochs_executed"]) for report in combination_reports]
        convergence[combination] = {
            "best_epoch": _describe(best_epochs),
            "epochs_executed": _describe(executed_epochs),
            "early_stopped_run_count": sum(epoch < max_epochs for epoch in executed_epochs),
            "max_epoch_run_count": sum(epoch == max_epochs for epoch in executed_epochs),
            "budget_censored_run_count": sum(
                best == max_epochs and executed == max_epochs
                for best, executed in zip(best_epochs, executed_epochs)
            ),
            "checkpoint_reproduction_delta_max": max(
                float(report["checkpoint_reproduction_score_delta"])
                for report in combination_reports
            ),
            "runtime_seconds": _describe(
                [float(report["runtime_seconds"]) for report in combination_reports]
            ),
            "peak_cuda_memory_bytes": max(
                int(report["peak_cuda_memory_bytes"]) for report in combination_reports
            ),
            "parameter_count": sorted(
                {int(report["parameter_count"]) for report in combination_reports}
            ),
        }

    def comparisons_against(reference: str, candidates: Sequence[str]) -> dict[str, object]:
        result: dict[str, object] = {}
        for combination in candidates:
            comparisons: dict[str, object] = {}
            for split in ("validation", "calibration"):
                for metric in metric_specs:
                    baseline = {
                        int(seed): _metric_value(by_key[(reference, int(seed))], split, metric)
                        for seed in expected_seeds
                    }
                    candidate = {
                        int(seed): _metric_value(by_key[(combination, int(seed))], split, metric)
                        for seed in expected_seeds
                    }
                    comparisons[f"{split}.{metric.metric_id}"] = paired_comparison(
                        baseline,
                        candidate,
                        metric,
                    )
            result[combination] = comparisons
        return result

    paired_vs_a = comparisons_against(
        "A",
        [combination for combination in expected_combinations if combination != "A"],
    )
    paired_vs_b = comparisons_against(
        "B",
        [
            combination
            for combination in expected_combinations
            if combination not in {"A", "B"}
        ],
    )

    return {
        "schema_version": R5_ANALYSIS_SCHEMA,
        "selection_status": "descriptive_only",
        "claim_boundary": (
            "three training seeds support descriptive paired comparison only; "
            "no automatic final-method selection"
        ),
        "integrity": {
            "completed_run_count": len(validated),
            "expected_run_count": len(expected_combinations) * len(expected_seeds),
            "failed_run_count": 0,
            "locked_test_accessed": False,
        },
        "metric_directions": {metric.metric_id: metric.direction for metric in metric_specs},
        "combination_summary": combination_summary,
        "paired_vs_A": paired_vs_a,
        "paired_vs_B": paired_vs_b,
        "convergence": convergence,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_bundle(
    analysis: Mapping[str, object],
    output_dir: str | Path,
    *,
    input_binding: Mapping[str, str],
    combination_labels: Mapping[str, str],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metric_directions = analysis["metric_directions"]  # type: ignore[index]
    summary_rows: list[dict[str, object]] = []
    combination_summary = analysis["combination_summary"]  # type: ignore[index]
    for combination, split_records in combination_summary.items():  # type: ignore[union-attr]
        for split, metric_records in split_records.items():
            for metric_id, record in metric_records.items():
                summary_rows.append(
                    {
                        "combination_id": combination,
                        "combination_label": combination_labels.get(combination, combination),
                        "split": split,
                        "metric_id": metric_id,
                        "direction": metric_directions[metric_id],
                        **record,
                    }
                )
    summary_fields = [
        "combination_id",
        "combination_label",
        "split",
        "metric_id",
        "direction",
        "count",
        "mean",
        "sample_std",
        "median",
        "minimum",
        "maximum",
        "mean_ci95_low",
        "mean_ci95_high",
    ]
    _write_csv(output / "combination_summary.csv", summary_fields, summary_rows)

    def paired_rows_for(baseline_id: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        comparison_key = f"paired_vs_{baseline_id}"
        for combination, comparisons in analysis[comparison_key].items():  # type: ignore[index,union-attr]
            for qualified_metric, record in comparisons.items():
                split, metric_id = qualified_metric.split(".", 1)
                benefit = record["benefit"]
                rows.append(
                    {
                        "combination_id": combination,
                        "combination_label": combination_labels.get(combination, combination),
                        "baseline_id": baseline_id,
                        "split": split,
                        "metric_id": metric_id,
                        "direction": record["direction"],
                        "mean_benefit": benefit["mean"],
                        "benefit_sample_std": benefit["sample_std"],
                        "benefit_ci95_low": benefit["mean_ci95_low"],
                        "benefit_ci95_high": benefit["mean_ci95_high"],
                        "mean_percent_benefit": record["mean_percent_benefit"],
                        "wins": record["wins"],
                        "ties": record["ties"],
                        "losses": record["losses"],
                        "exact_sign_flip_p_two_sided": record["exact_sign_flip_p_two_sided"],
                    }
                )
        return rows

    paired_rows = paired_rows_for("A")
    paired_fields = [
        "combination_id",
        "combination_label",
        "baseline_id",
        "split",
        "metric_id",
        "direction",
        "mean_benefit",
        "benefit_sample_std",
        "benefit_ci95_low",
        "benefit_ci95_high",
        "mean_percent_benefit",
        "wins",
        "ties",
        "losses",
        "exact_sign_flip_p_two_sided",
    ]
    _write_csv(output / "paired_vs_A.csv", paired_fields, paired_rows)
    _write_csv(output / "paired_vs_B.csv", paired_fields, paired_rows_for("B"))

    convergence_rows: list[dict[str, object]] = []
    for combination, record in analysis["convergence"].items():  # type: ignore[index,union-attr]
        convergence_rows.append(
            {
                "combination_id": combination,
                "combination_label": combination_labels.get(combination, combination),
                "best_epoch_mean": record["best_epoch"]["mean"],
                "epochs_executed_mean": record["epochs_executed"]["mean"],
                "early_stopped_run_count": record["early_stopped_run_count"],
                "max_epoch_run_count": record["max_epoch_run_count"],
                "budget_censored_run_count": record["budget_censored_run_count"],
                "checkpoint_reproduction_delta_max": record[
                    "checkpoint_reproduction_delta_max"
                ],
                "runtime_seconds_mean": record["runtime_seconds"]["mean"],
                "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
                "parameter_count": ";".join(str(value) for value in record["parameter_count"]),
            }
        )
    convergence_fields = [
        "combination_id",
        "combination_label",
        "best_epoch_mean",
        "epochs_executed_mean",
        "early_stopped_run_count",
        "max_epoch_run_count",
        "budget_censored_run_count",
        "checkpoint_reproduction_delta_max",
        "runtime_seconds_mean",
        "peak_cuda_memory_bytes",
        "parameter_count",
    ]
    _write_csv(output / "convergence.csv", convergence_fields, convergence_rows)

    if "r2_metric_coverage" in analysis:
        coverage_fields = [
            "metric_id",
            "layer",
            "coverage_status",
            "computed_run_count",
            "recorded_run_count",
            "expected_run_count",
        ]
        _write_csv(
            output / "r2_metric_coverage.csv",
            coverage_fields,
            analysis["r2_metric_coverage"],  # type: ignore[arg-type]
        )

    markdown = [
        "# PI-JWM R5 multi-seed descriptive analysis",
        "",
        f"- Selection status: `{analysis['selection_status']}`",
        f"- Claim boundary: {analysis['claim_boundary']}",
        "- Positive paired benefit always means the candidate is better than A.",
        "- With three paired seeds, the smallest attainable two-sided exact sign-flip p-value is 0.25.",
        "",
        "| ID | Combination |",
        "|---|---|",
    ]
    markdown.extend(
        f"| {combination} | {label} |" for combination, label in combination_labels.items()
    )
    markdown.extend(
        [
            "",
            "Machine-readable results are in `analysis.json`, `combination_summary.csv`, "
            "`paired_vs_A.csv`, `paired_vs_B.csv`, `convergence.csv`, and, when present, "
            "`r2_metric_coverage.csv`.",
        ]
    )
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    manifest_files: dict[str, dict[str, object]] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        content = path.read_bytes()
        manifest_files[path.name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {
        "schema_version": R5_ANALYSIS_SCHEMA,
        "manifest_entry_count": len(manifest_files),
        "input_binding": dict(input_binding),
        "files": manifest_files,
        "selection_status": analysis["selection_status"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
