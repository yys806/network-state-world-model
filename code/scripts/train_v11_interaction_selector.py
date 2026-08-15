#!/usr/bin/env python
"""Train the PI-JWM v11 opportunity-gated token interaction selector."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.v11_interaction_selector import (
    append_episode_phase_context,
    fit_interaction_selector,
    predict_interaction_selector,
    select_interaction_candidates,
)
from pi_jwm.v11_labeling import (
    load_candidate_interaction_cache,
    load_candidate_label_metadata,
)
from pi_jwm.v11_selector import (
    _pareto_dominated,
    canonical_sha256,
    file_sha256,
    observable_pareto_deltas,
)
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--temperature", type=float, nargs="+", default=(0.1, 0.25))
    parser.add_argument("--dropout", type=float, nargs="+", default=(0.0, 0.1))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(17, 29, 41))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _legal_mask(batch, outcome) -> np.ndarray:
    legal = batch.candidate_mask.copy()
    if outcome.action_applicable is not None:
        legal &= outcome.action_applicable
    if outcome.action_applied is not None:
        legal &= outcome.action_applied
    legal[:, outcome.default_index] = batch.candidate_mask[:, outcome.default_index]
    return legal


def _pareto_allowed(batch, outcome) -> np.ndarray:
    task, energy = observable_pareto_deltas(batch, outcome.default_index)
    allowed = np.ones(batch.candidate_mask.shape, dtype=bool)
    for sample in range(allowed.shape[0]):
        allowed[sample] = ~_pareto_dominated(task[sample], energy[sample])
    allowed[:, outcome.default_index] = True
    return allowed


def _ensemble_predictions(predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "opportunity_probability": np.mean(
            np.stack([item["opportunity_probability"] for item in predictions]), axis=0
        ),
        "candidate_score": np.mean(
            np.stack([item["candidate_score"] for item in predictions]), axis=0
        ),
        "candidate_sign_probability": np.mean(
            np.stack([item["candidate_sign_probability"] for item in predictions]), axis=0
        ),
    }


def _calibrate_thresholds(batch, outcome, seed, ensemble) -> dict:
    legal = _legal_mask(batch, outcome)
    pareto = _pareto_allowed(batch, outcome)
    rows = []
    for opportunity_threshold, sign_threshold in itertools.product(
        (0.5, 0.65, 0.8, 0.9), repeat=2
    ):
        choice = select_interaction_candidates(
            ensemble["opportunity_probability"],
            ensemble["candidate_score"],
            ensemble["candidate_sign_probability"],
            legal,
            outcome.default_index,
            opportunity_threshold,
            sign_threshold,
            pareto_allowed=pareto,
        )
        metrics = _choice_metrics(outcome, choice, seed, legal)
        rows.append(
            {
                "opportunity_threshold": float(opportunity_threshold),
                "sign_threshold": float(sign_threshold),
                "rmse": metrics["rmse"],
                "executed_count": int(
                    np.sum((choice != outcome.default_index) & (outcome.active_count > 0))
                ),
                "positive_precision": metrics["executed_positive_precision"],
                "negative_selection_rate": metrics["negative_selection_rate"],
            }
        )
    safe = [
        row
        for row in rows
        if row["executed_count"] > 0
        and row["positive_precision"] >= 0.65
        and row["negative_selection_rate"] <= 0.20
    ]
    if not safe:
        return {
            "status": "no_safe_threshold",
            "opportunity_threshold": 1.1,
            "sign_threshold": 1.1,
            "rows": rows,
        }
    best = min(
        safe,
        key=lambda row: (
            float(row["rmse"]),
            -int(row["executed_count"]),
            float(row["opportunity_threshold"]),
            float(row["sign_threshold"]),
        ),
    )
    return {"status": "safe_threshold", **best, "rows": rows}


def _save_checkpoint(path: Path, fitted, metadata: dict) -> dict:
    payload = {
        "state_dict": {
            name: value.detach().cpu() for name, value in fitted.model.state_dict().items()
        },
        "candidate_dim": int(fitted.normalizer.candidate_mean.shape[0]),
        "context_dim": int(fitted.normalizer.context_mean.shape[0]),
        "token_dim": int(fitted.normalizer.token_mean.shape[0]),
        "hidden_dim": int(fitted.hidden_dim),
        "dropout": float(fitted.dropout),
        "benefit_scale": float(fitted.benefit_scale),
        "candidate_mean": torch.from_numpy(fitted.normalizer.candidate_mean),
        "candidate_scale": torch.from_numpy(fitted.normalizer.candidate_scale),
        "context_mean": torch.from_numpy(fitted.normalizer.context_mean),
        "context_scale": torch.from_numpy(fitted.normalizer.context_scale),
        "token_mean": torch.from_numpy(fitted.normalizer.token_mean),
        "token_scale": torch.from_numpy(fitted.normalizer.token_scale),
        "history": fitted.history,
        **metadata,
    }
    torch.save(payload, path)
    return {"file": path.name, "sha256": file_sha256(path), **metadata}


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "train": args.train_cache.resolve(),
        "calibration": args.calibration_cache.resolve(),
        "validation": args.validation_cache.resolve(),
    }
    loaded = {
        split: load_candidate_interaction_cache(path) for split, path in cache_paths.items()
    }
    manifests = {split: item[3] for split, item in loaded.items()}
    configuration_digest = validate_cache_protocol(manifests, required_schema_version=6)
    batches = {split: item[0] for split, item in loaded.items()}
    outcomes = {split: item[1] for split, item in loaded.items()}
    interactions = {split: item[2] for split, item in loaded.items()}
    metadata = {
        split: load_candidate_label_metadata(
            path, expected_configuration_digest=configuration_digest
        )
        for split, path in cache_paths.items()
    }
    batches = {
        split: append_episode_phase_context(
            batches[split], metadata[split]["sample_ids"], episode_length=390
        )
        for split in batches
    }
    config_rows = []
    threshold_rows = []
    per_seed_rows = []
    checkpoint_records: dict[str, list[dict]] = {}
    for hidden_dim, temperature, dropout in itertools.product(
        args.hidden_dim, args.temperature, args.dropout
    ):
        config_id = f"interaction_h{hidden_dim}_t{temperature:g}_d{dropout:g}"
        calibration_predictions = []
        validation_predictions = []
        records = []
        for training_seed in args.training_seeds:
            fitted = fit_interaction_selector(
                batches["train"],
                outcomes["train"],
                interactions["train"],
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                temperature=float(temperature),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                seed=int(training_seed),
                device=args.device,
            )
            calibration_predictions.append(
                predict_interaction_selector(
                    fitted,
                    batches["calibration"],
                    interactions["calibration"],
                    batch_size=int(args.batch_size),
                )
            )
            validation_predictions.append(
                predict_interaction_selector(
                    fitted,
                    batches["validation"],
                    interactions["validation"],
                    batch_size=int(args.batch_size),
                )
            )
            checkpoint_path = output_dir / f"{config_id}_seed{training_seed}.pt"
            records.append(
                _save_checkpoint(
                    checkpoint_path,
                    fitted,
                    {
                        "configuration_digest": configuration_digest,
                        "training_seed": int(training_seed),
                        "temperature": float(temperature),
                    },
                )
            )
        calibration_ensemble = _ensemble_predictions(calibration_predictions)
        validation_ensemble = _ensemble_predictions(validation_predictions)
        thresholds = _calibrate_thresholds(
            batches["calibration"],
            outcomes["calibration"],
            metadata["calibration"]["sample_seed"],
            calibration_ensemble,
        )
        for row in thresholds.pop("rows"):
            threshold_rows.append({"config_id": config_id, **row})
        legal = _legal_mask(batches["validation"], outcomes["validation"])
        choice = select_interaction_candidates(
            validation_ensemble["opportunity_probability"],
            validation_ensemble["candidate_score"],
            validation_ensemble["candidate_sign_probability"],
            legal,
            outcomes["validation"].default_index,
            thresholds["opportunity_threshold"],
            thresholds["sign_threshold"],
            pareto_allowed=_pareto_allowed(
                batches["validation"], outcomes["validation"]
            ),
        )
        metrics = _choice_metrics(
            outcomes["validation"],
            choice,
            metadata["validation"]["sample_seed"],
            legal,
        )
        executed = (choice != outcomes["validation"].default_index) & (
            outcomes["validation"].active_count > 0
        )
        row = {
            "config_id": config_id,
            "hidden_dim": int(hidden_dim),
            "temperature": float(temperature),
            "dropout": float(dropout),
            "calibration_status": thresholds["status"],
            "opportunity_threshold": thresholds["opportunity_threshold"],
            "sign_threshold": thresholds["sign_threshold"],
            "validation_rmse": metrics["rmse"],
            "validation_link_rmse": metrics["link_rmse"],
            "validation_activity_f1": metrics["activity_f1"],
            "improvement_vs_default": metrics["improvement_vs_default"],
            "executed_count": int(np.sum(executed)),
            "defer_ratio": float(1.0 - np.sum(executed) / max(1, np.sum(outcomes["validation"].active_count > 0))),
            "improved_seed_count": metrics["improved_seed_count"],
            "executed_positive_precision": metrics["executed_positive_precision"],
            "negative_selection_rate": metrics["negative_selection_rate"],
            "activity_f1_drop": metrics["activity_f1_drop"],
            "link_rmse_relative_degradation": metrics["link_rmse_relative_degradation"],
            "worst_seed_regret": metrics["worst_seed_regret"],
            "training_seed_std": 0.0,
        }
        config_rows.append(row)
        checkpoint_records[config_id] = records
        for seed_row in metrics["per_seed"]:
            per_seed_rows.append({"config_id": config_id, **seed_row})
        print(json.dumps(row, ensure_ascii=False), flush=True)
    selected = min(
        config_rows,
        key=lambda row: (
            float(row["validation_rmse"]),
            float(row["worst_seed_regret"]),
            str(row["config_id"]),
        ),
    )
    gate = classify_selector_validation_gate(selected)
    freeze_payload = {
        "configuration_digest": configuration_digest,
        "selected_config": selected,
        "checkpoint_records": checkpoint_records[selected["config_id"]],
        "cache_sha256": {split: file_sha256(path) for split, path in cache_paths.items()},
        "method": "opportunity_gated_token_interaction_selector",
    }
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "method": "opportunity_gated_token_interaction_selector",
        "configuration_frozen": bool(gate["passed"]),
        "configuration_digest": configuration_digest,
        "selector_freeze_digest": canonical_sha256(freeze_payload),
        "selected_config": selected,
        "selector_validation_gate": gate,
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "result_kind": "deployable" if gate["passed"] else "diagnostic_only",
        "runtime_seconds": time.time() - started,
        "cache_sha256": freeze_payload["cache_sha256"],
    }
    _write_csv(output_dir / "selector_grid_results.csv", config_rows)
    _write_csv(output_dir / "calibration_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "validation_per_seed.csv", per_seed_rows)
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
