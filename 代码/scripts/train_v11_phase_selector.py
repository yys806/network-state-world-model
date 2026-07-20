#!/usr/bin/env python
"""Fit and validation-audit the PI-JWM v11 phase-conditioned selector."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.v11_labeling import (
    load_candidate_interaction_cache,
    load_candidate_label_metadata,
)
from pi_jwm.v11_phase_selector import (
    build_observable_pareto_allowed,
    calibrate_phase_selector,
    fit_phase_candidate_statistics,
    select_phase_candidates,
)
from pi_jwm.v11_selector import canonical_sha256, file_sha256
from train_v11_candidate_set_selector import (
    _choice_metrics,
    classify_selector_validation_gate,
    validate_cache_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, default=390)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _result_grade(metrics: dict) -> str:
    rmse = float(metrics["validation_rmse"])
    if rmse < 200.0:
        return "A"
    if rmse < 213.160874:
        return "B"
    if float(metrics["improvement_vs_default"]) > 0.0:
        return "improved_diagnostic"
    return "not_passed"


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.train_cache.resolve(),
        "calibration": args.calibration_cache.resolve(),
        "validation": args.validation_cache.resolve(),
    }
    loaded = {split: load_candidate_interaction_cache(path) for split, path in paths.items()}
    manifests = {split: item[3] for split, item in loaded.items()}
    digest = validate_cache_protocol(manifests, required_schema_version=6)
    batches = {split: item[0] for split, item in loaded.items()}
    outcomes = {split: item[1] for split, item in loaded.items()}
    metadata = {
        split: load_candidate_label_metadata(path, expected_configuration_digest=digest)
        for split, path in paths.items()
    }
    statistics = fit_phase_candidate_statistics(
        batches["train"],
        outcomes["train"],
        metadata["train"]["sample_ids"],
        episode_length=int(args.episode_length),
    )
    positive_means = statistics.mean_benefit[
        np.isfinite(statistics.mean_benefit) & (statistics.mean_benefit > 0.0)
    ]
    minimum_mean_values = tuple(
        float(value)
        for value in np.unique(np.quantile(positive_means, np.linspace(0.0, 0.9, 37)))
    )
    calibrated = calibrate_phase_selector(
        statistics,
        batches["calibration"],
        outcomes["calibration"],
        metadata["calibration"]["sample_ids"],
        pareto_allowed=build_observable_pareto_allowed(
            batches["calibration"], outcomes["calibration"].default_index
        ),
        z_values=tuple(float(value) for value in np.linspace(0.0, 2.2, 23)),
        positive_rate_values=tuple(float(value) for value in np.arange(0.5, 0.876, 0.025)),
        minimum_mean_values=minimum_mean_values,
        min_count=5,
    )
    validation_pareto = build_observable_pareto_allowed(
        batches["validation"], outcomes["validation"].default_index
    )
    choice = select_phase_candidates(
        statistics,
        batches["validation"],
        metadata["validation"]["sample_ids"],
        calibrated.config,
        pareto_allowed=validation_pareto,
    )
    validation = _choice_metrics(
        outcomes["validation"],
        choice,
        metadata["validation"]["sample_seed"],
        batches["validation"].candidate_mask,
    )
    executed = (choice != outcomes["validation"].default_index) & (
        outcomes["validation"].active_count > 0
    )
    metrics = {
        "config_id": "phase_mean_lcb_exact",
        "validation_rmse": validation["rmse"],
        "validation_link_rmse": validation["link_rmse"],
        "validation_activity_f1": validation["activity_f1"],
        "improvement_vs_default": validation["improvement_vs_default"],
        "worst_seed_regret": validation["worst_seed_regret"],
        "improved_seed_count": validation["improved_seed_count"],
        "executed_count": int(np.sum(executed)),
        "defer_ratio": float(
            1.0 - np.sum(executed) / max(1, np.sum(outcomes["validation"].active_count > 0))
        ),
        "executed_positive_precision": validation["executed_positive_precision"],
        "negative_selection_rate": validation["negative_selection_rate"],
        "activity_f1_drop": validation["activity_f1_drop"],
        "link_rmse_relative_degradation": validation["link_rmse_relative_degradation"],
        "training_seed_std": 0.0,
        "pareto_violations": int(
            np.sum(~validation_pareto[np.arange(choice.shape[0]), choice])
        ),
    }
    validation_gate = classify_selector_validation_gate(metrics)
    grade = _result_grade(metrics)
    configuration_frozen = bool(grade == "A" and validation_gate["passed"])
    phase = np.mod(metadata["validation"]["sample_ids"], int(args.episode_length))
    trace_rows = []
    for index, candidate in enumerate(choice):
        candidate = int(candidate)
        phase_index = int(phase[index])
        realized_benefit = float(
            outcomes["validation"].active_sse[index, outcomes["validation"].default_index]
            - outcomes["validation"].active_sse[index, candidate]
        )
        trace_rows.append(
            {
                "sample_id": int(metadata["validation"]["sample_ids"][index]),
                "seed": int(metadata["validation"]["sample_seed"][index]),
                "episode_phase": phase_index,
                "candidate_index": candidate,
                "candidate_name": batches["validation"].candidate_names[candidate],
                "executed": bool(executed[index]),
                "train_mean_benefit": float(statistics.mean_benefit[phase_index, candidate]),
                "train_positive_rate": float(statistics.positive_rate[phase_index, candidate]),
                "train_count": int(statistics.count[phase_index, candidate]),
                "pareto_allowed": bool(validation_pareto[index, candidate]),
                "realized_sse_benefit": realized_benefit,
                "result_kind": "diagnostic_only",
            }
        )
    np.savez_compressed(
        output_dir / "phase_candidate_statistics.npz",
        mean_benefit=statistics.mean_benefit,
        benefit_std=statistics.benefit_std,
        positive_rate=statistics.positive_rate,
        count=statistics.count,
        default_index=np.asarray([statistics.default_index], dtype=np.int64),
        episode_length=np.asarray([statistics.episode_length], dtype=np.int64),
        candidate_names=np.asarray(statistics.candidate_names),
    )
    _write_csv(output_dir / "decision_trace_validation.csv", trace_rows)
    _write_csv(output_dir / "validation_per_seed.csv", validation["per_seed"])
    freeze_payload = {
        "configuration_digest": digest,
        "method": "phase_mean_lcb_exact",
        "calibrated": asdict(calibrated),
        "candidate_names": list(statistics.candidate_names),
        "cache_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "episode_length": int(args.episode_length),
        "pareto_rule": "observable_task_energy_proxy_non_dominated",
    }
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "method": "phase_conditioned_benefit_lcb",
        "validation_grade": grade,
        "configuration_frozen": configuration_frozen,
        "external_holdout_unlocked": configuration_frozen,
        "configuration_digest": digest,
        "selector_freeze_digest": canonical_sha256(freeze_payload),
        "calibration": asdict(calibrated),
        "validation_metrics": metrics,
        "validation_gate": validation_gate,
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "result_kind": "deployable" if configuration_frozen else "diagnostic_only",
        "runtime_seconds": time.time() - started,
        "cache_sha256": freeze_payload["cache_sha256"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "freeze_payload.json").write_text(
        json.dumps(freeze_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
