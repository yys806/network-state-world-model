#!/usr/bin/env python
"""Run the leakage-safe PI-JWM v11 candidate-benefit identifiability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CACHE_DIR = (
    CODE_ROOT
    / "artifacts/reports/pi_jwm_v11_selector_refinement_20260717/label_cache_schema5"
)
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT
    / "artifacts/reports/pi_jwm_v11_candidate_benefit_identifiability_20260718/formal_cpu"
)

EXPECTED_SEEDS = {
    "train": set(range(16)) | set(range(20, 44)),
    "calibration": set(range(44, 50)),
    "validation": set(range(50, 60)),
}
INTERACTION_PROTOCOL_FIELDS = (
    "token_capacity",
    "token_dimension",
    "pooled_dimension",
    "token_feature_names",
    "pooled_feature_names",
    "action_feature_names",
)


def _interaction_protocol(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    interaction = manifest.get("interaction")
    if not isinstance(interaction, Mapping):
        return None
    return {field: interaction.get(field) for field in INTERACTION_PROTOCOL_FIELDS}


def validate_audit_manifests(
    manifests: Mapping[str, Mapping[str, Any]],
    required_schema_version: int = 5,
) -> str:
    required = ("train", "calibration", "validation")
    if set(manifests) != set(required):
        raise ValueError("benefit audit requires exactly train, calibration, and validation caches")
    for split in required:
        manifest = manifests[split]
        if int(manifest.get("schema_version", -1)) != int(required_schema_version):
            raise ValueError(
                f"benefit audit requires cache schema {int(required_schema_version)}"
            )
        if str(manifest.get("split_name")) != split:
            raise ValueError(f"cache split mismatch for {split}")
        seeds = {int(value) for value in manifest.get("seed_values", ())}
        if seeds != EXPECTED_SEEDS[split]:
            raise ValueError(f"cache seed set mismatch for {split}")
    for field in ("configuration_digest", "candidate_names", "feature_names", "context_feature_names"):
        values = {
            tuple(manifests[split].get(field, ()))
            if isinstance(manifests[split].get(field), list)
            else str(manifests[split].get(field))
            for split in required
        }
        if len(values) != 1:
            raise ValueError(f"cache {field} mismatch")
    if int(required_schema_version) == 6:
        interaction_contracts = {
            json.dumps(_interaction_protocol(manifests[split]), sort_keys=True)
            for split in required
        }
        if len(interaction_contracts) != 1 or "null" in interaction_contracts:
            raise ValueError("cache interaction protocol mismatch")
    return str(manifests["train"].get("configuration_digest"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache", type=Path, default=DEFAULT_CACHE_DIR / "candidate_labels_train.npz"
    )
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=DEFAULT_CACHE_DIR / "candidate_labels_calibration.npz",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_CACHE_DIR / "candidate_labels_validation.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit-per-split", type=int, default=0)
    parser.add_argument("--group-cv-folds", type=int, default=3)
    parser.add_argument("--required-schema-version", type=int, choices=(5, 6), default=5)
    parser.add_argument(
        "--model-kinds", nargs="+", choices=("linear", "rf", "hgb", "xgb"),
        default=("linear", "rf", "hgb", "xgb"),
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=(
            "prior_only",
            "context_only",
            "candidate_only",
            "forecast_delta",
            "selected_edge",
            "full_schema_v5",
            "interaction_pooled_only",
            "full_schema_v6",
        ),
        default=(
            "prior_only",
            "context_only",
            "candidate_only",
            "forecast_delta",
            "selected_edge",
            "full_schema_v5",
        ),
    )
    return parser


def _balanced_indices(seeds: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(seeds, dtype=np.int64).reshape(-1)
    requested = int(limit)
    if requested <= 0 or requested >= values.shape[0]:
        return np.arange(values.shape[0], dtype=np.int64)
    groups = [np.flatnonzero(values == seed) for seed in np.unique(values)]
    selected: list[int] = []
    depth = 0
    while len(selected) < requested:
        added = False
        for group in groups:
            if depth < group.shape[0] and len(selected) < requested:
                selected.append(int(group[depth]))
                added = True
        if not added:
            break
        depth += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def _subset_sources(batch, outcome, metadata: Mapping[str, Any], indices: np.ndarray):
    from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

    selected = np.asarray(indices, dtype=np.int64)

    def optional(values):
        return None if values is None else values[selected]

    subset_batch = CandidateBatch(
        context=batch.context[selected],
        candidate_features=batch.candidate_features[selected],
        candidate_mask=batch.candidate_mask[selected],
        stage=batch.stage[selected],
        feature_names=batch.feature_names,
        candidate_names=batch.candidate_names,
        context_feature_names=batch.context_feature_names,
    )
    subset_outcome = CandidateOutcome(
        active_sse=outcome.active_sse[selected],
        active_count=outcome.active_count[selected],
        link_sse=optional(outcome.link_sse),
        link_count=optional(outcome.link_count),
        activity_tp=optional(outcome.activity_tp),
        activity_fp=optional(outcome.activity_fp),
        activity_fn=optional(outcome.activity_fn),
        activity_tn=optional(outcome.activity_tn),
        action_applied=optional(outcome.action_applied),
        action_applicable=optional(outcome.action_applicable),
        default_index=outcome.default_index,
        task_utility=optional(outcome.task_utility),
        energy_total=optional(outcome.energy_total),
        result_kind=outcome.result_kind,
    )
    subset_metadata = {
        "sample_ids": np.asarray(metadata["sample_ids"])[selected],
        "sample_seed": np.asarray(metadata["sample_seed"])[selected],
    }
    return subset_batch, subset_outcome, subset_metadata


def _family(candidate_name: str) -> str:
    name = str(candidate_name).lower()
    if name == "identity":
        return "identity"
    if "historical" in name:
        return "historical"
    if "compute" in name or "cpu" in name:
        return "compute"
    if "return" in name:
        return "return"
    if "offload" in name:
        return "offload"
    if "rb" in name or "ranked" in name or "benefit_residual" in name:
        return "rb"
    return "other"


def _gate(metrics: Mapping[str, Any]) -> dict[str, bool]:
    default_f1 = metrics.get("default_activity_f1")
    selected_f1 = metrics.get("activity_f1")
    f1_drop = 0.0 if default_f1 is None or selected_f1 is None else max(
        0.0, float(default_f1) - float(selected_f1)
    )
    default_link = metrics.get("default_link_rmse")
    selected_link = metrics.get("link_rmse")
    link_degradation = 0.0 if default_link in (None, 0.0) or selected_link is None else max(
        0.0, (float(selected_link) - float(default_link)) / float(default_link)
    )
    return {
        "rmse_below_best_fixed": float(metrics["active_rate_rmse"]) < 230.8556,
        "seven_of_ten_seeds": int(metrics["improved_seed_count"]) >= 7,
        "positive_precision": float(metrics["executed_positive_precision"]) >= 0.65,
        "negative_rate": float(metrics["negative_selection_rate"]) <= 0.20,
        "activity_f1": f1_drop <= 0.002,
        "link_rmse": link_degradation <= 0.02,
        "sample_rank_spearman": metrics.get("sample_rank_spearman") is not None
        and float(metrics["sample_rank_spearman"]) >= 0.20,
    }


def _seed_rows(dataset, choice: np.ndarray) -> list[dict[str, Any]]:
    rows = np.arange(choice.shape[0])
    result = []
    for seed in np.unique(dataset.sample_seed):
        keep = dataset.valid_sample & (dataset.sample_seed == seed)
        count = int(np.sum(dataset.outcome.active_count[keep]))
        selected_sse = float(np.sum(dataset.outcome.active_sse[rows[keep], choice[keep]]))
        default_sse = float(
            np.sum(dataset.outcome.active_sse[keep, dataset.outcome.default_index])
        )
        selected_rmse = float(np.sqrt(selected_sse / count)) if count else None
        default_rmse = float(np.sqrt(default_sse / count)) if count else None
        executed = keep & (choice != dataset.outcome.default_index)
        benefits = dataset.candidate_benefit[rows[executed], choice[executed]]
        result.append(
            {
                "seed": int(seed),
                "sample_count": int(np.sum(keep)),
                "active_rate_rmse": selected_rmse,
                "default_active_rate_rmse": default_rmse,
                "improvement": None if count == 0 else default_rmse - selected_rmse,
                "executed_count": int(np.sum(executed)),
                "positive_count": int(np.sum(benefits > 1e-6)),
                "negative_count": int(np.sum(benefits < -1e-6)),
            }
        )
    return result


def _trace_rows(dataset, predictions, choice: np.ndarray) -> list[dict[str, Any]]:
    result = []
    default = int(dataset.outcome.default_index)
    for index, selected in enumerate(np.asarray(choice, dtype=np.int64)):
        candidate_name = dataset.batch.candidate_names[selected]
        actual_benefit = dataset.candidate_benefit[index, selected]
        result.append(
            {
                "sample_id": int(dataset.sample_ids[index]),
                "seed": int(dataset.sample_seed[index]),
                "stage": str(dataset.batch.stage[index]),
                "selected_candidate_index": int(selected),
                "selected_candidate": candidate_name,
                "selected_family": _family(candidate_name),
                "executed": bool(dataset.valid_sample[index] and selected != default),
                "opportunity_probability": float(predictions.opportunity_probability[index])
                if dataset.valid_sample[index]
                else None,
                "candidate_sign_probability": float(
                    predictions.candidate_sign_probability[index, selected]
                )
                if dataset.valid_sample[index]
                else None,
                "predicted_benefit": float(predictions.predicted_benefit[index, selected])
                if dataset.valid_sample[index]
                else None,
                "actual_benefit": float(actual_benefit) if np.isfinite(actual_benefit) else None,
                "active_count": int(dataset.outcome.active_count[index]),
                "selected_active_sse": float(dataset.outcome.active_sse[index, selected]),
                "default_active_sse": float(dataset.outcome.active_sse[index, default]),
            }
        )
    return result


def _group_trace_rows(trace: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (stage, family), group in trace.groupby(["stage", "selected_family"], dropna=False):
        executed = group["executed"].astype(bool)
        benefits = group.loc[executed, "actual_benefit"].astype(float)
        rows.append(
            {
                "stage": stage,
                "selected_family": family,
                "sample_count": int(len(group)),
                "executed_count": int(executed.sum()),
                "positive_count": int((benefits > 1e-6).sum()),
                "negative_count": int((benefits < -1e-6).sum()),
                "mean_actual_benefit": float(benefits.mean()) if len(benefits) else None,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from pi_jwm.v11_benefit_identifiability import (
        build_benefit_audit_dataset,
        build_benefit_feature_groups,
        calibrate_safe_thresholds,
        evaluate_benefit_predictions,
        fit_benefit_audit_model,
        predict_benefit_audit_model,
        seed_group_folds,
        select_benefit_candidates,
    )
    from pi_jwm.v11_interactions import append_interaction_pooled_features
    from pi_jwm.v11_labeling import (
        load_candidate_interaction_cache,
        load_candidate_label_cache,
        load_candidate_label_metadata,
    )

    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "train": Path(args.train_cache).resolve(),
        "calibration": Path(args.calibration_cache).resolve(),
        "validation": Path(args.validation_cache).resolve(),
    }
    if int(args.required_schema_version) == 6:
        loaded = {}
        for split, path in cache_paths.items():
            base_batch, outcome, interactions, manifest = load_candidate_interaction_cache(path)
            loaded[split] = (
                append_interaction_pooled_features(base_batch, interactions),
                outcome,
                manifest,
            )
    else:
        loaded = {
            split: load_candidate_label_cache(path) for split, path in cache_paths.items()
        }
    manifests = {split: values[2] for split, values in loaded.items()}
    configuration_digest = validate_audit_manifests(
        manifests, required_schema_version=int(args.required_schema_version)
    )
    token_protocol_sha256 = None
    if int(args.required_schema_version) == 6:
        token_protocol_sha256 = hashlib.sha256(
            json.dumps(
                _interaction_protocol(manifests["train"]),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    metadata = {
        split: load_candidate_label_metadata(
            path, expected_configuration_digest=configuration_digest
        )
        for split, path in cache_paths.items()
    }
    datasets = {}
    feature_groups = {}
    for split in ("train", "calibration", "validation"):
        batch, outcome, _ = loaded[split]
        indices = _balanced_indices(
            metadata[split]["sample_seed"], int(args.sample_limit_per_split)
        )
        batch, outcome, split_metadata = _subset_sources(
            batch, outcome, metadata[split], indices
        )
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            split_metadata["sample_ids"],
            split_metadata["sample_seed"],
        )
        datasets[split] = dataset
        feature_groups[split] = build_benefit_feature_groups(dataset)

    results = []
    threshold_rows = []
    fitted_results = []
    model_status: dict[str, str] = {}
    for feature_group_name in args.feature_groups:
        train_group = feature_groups["train"][feature_group_name]
        calibration_group = feature_groups["calibration"][feature_group_name]
        validation_group = feature_groups["validation"][feature_group_name]
        if train_group.opportunity_feature_names != calibration_group.opportunity_feature_names or (
            train_group.candidate_feature_names != calibration_group.candidate_feature_names
        ):
            raise ValueError("feature group order mismatch across splits")
        for model_kind in args.model_kinds:
            result_id = f"{feature_group_name}__{model_kind}"
            try:
                fitted = fit_benefit_audit_model(
                    datasets["train"], train_group, model_kind=model_kind
                )
                calibration_predictions = predict_benefit_audit_model(
                    fitted, datasets["calibration"], calibration_group
                )
                validation_predictions = predict_benefit_audit_model(
                    fitted, datasets["validation"], validation_group
                )
                calibration = calibrate_safe_thresholds(
                    datasets["calibration"], calibration_predictions
                )
                if calibration.status == "safe_threshold":
                    validation_choice = select_benefit_candidates(
                        datasets["validation"],
                        validation_predictions,
                        calibration.opportunity_threshold,
                        calibration.sign_threshold,
                    )
                else:
                    validation_choice = np.full(
                        datasets["validation"].batch.candidate_mask.shape[0],
                        datasets["validation"].outcome.default_index,
                        dtype=np.int64,
                    )
                metrics = evaluate_benefit_predictions(
                    datasets["validation"], validation_predictions, validation_choice
                )
                gate = _gate(metrics)
                row = {
                    "result_id": result_id,
                    "feature_group": feature_group_name,
                    "model_kind": model_kind,
                    "status": "completed",
                    "opportunity_status": fitted.opportunity_status,
                    "candidate_status": fitted.candidate_status,
                    "calibration_status": calibration.status,
                    "opportunity_threshold": calibration.opportunity_threshold,
                    "sign_threshold": calibration.sign_threshold,
                    **metrics,
                    "gate_passed": all(gate.values()),
                }
                results.append(row)
                threshold_rows.append(
                    {
                        "result_id": result_id,
                        "status": calibration.status,
                        "opportunity_threshold": calibration.opportunity_threshold,
                        "sign_threshold": calibration.sign_threshold,
                        **calibration.metrics,
                    }
                )
                fitted_results.append(
                    (row, fitted, validation_predictions, validation_choice, gate)
                )
                model_status[result_id] = "completed"
            except (ImportError, ModuleNotFoundError) as error:
                results.append(
                    {
                        "result_id": result_id,
                        "feature_group": feature_group_name,
                        "model_kind": model_kind,
                        "status": "skipped_dependency_unavailable",
                        "error": str(error),
                    }
                )
                model_status[result_id] = "skipped_dependency_unavailable"
    if not fitted_results:
        raise RuntimeError("no benefit identifiability model completed")

    complexity = {"linear": 0, "hgb": 1, "rf": 2, "xgb": 3}
    passing = [item for item in fitted_results if all(item[4].values())]
    if passing:
        winner = min(
            passing,
            key=lambda item: (
                complexity[item[0]["model_kind"]],
                float(item[0]["active_rate_rmse"]),
                item[0]["feature_group"],
            ),
        )
        classification = "identifiable"
    else:
        winner = min(
            fitted_results,
            key=lambda item: (
                float(item[0]["active_rate_rmse"]),
                complexity[item[0]["model_kind"]],
                item[0]["feature_group"],
            ),
        )
        classification = "not_identifiable"
    winner_row, _, winner_predictions, winner_choice, winner_gate = winner
    cv_rows = []
    cv_fold_count = int(args.group_cv_folds)
    if cv_fold_count >= 2:
        train_dataset = datasets["train"]
        for fold_index, (fit_indices, holdout_indices) in enumerate(
            seed_group_folds(train_dataset.sample_seed, n_splits=cv_fold_count)
        ):
            fit_batch, fit_outcome, fit_metadata = _subset_sources(
                train_dataset.batch,
                train_dataset.outcome,
                {
                    "sample_ids": train_dataset.sample_ids,
                    "sample_seed": train_dataset.sample_seed,
                },
                fit_indices,
            )
            holdout_batch, holdout_outcome, holdout_metadata = _subset_sources(
                train_dataset.batch,
                train_dataset.outcome,
                {
                    "sample_ids": train_dataset.sample_ids,
                    "sample_seed": train_dataset.sample_seed,
                },
                holdout_indices,
            )
            fit_dataset = build_benefit_audit_dataset(
                fit_batch,
                fit_outcome,
                fit_metadata["sample_ids"],
                fit_metadata["sample_seed"],
            )
            holdout_dataset = build_benefit_audit_dataset(
                holdout_batch,
                holdout_outcome,
                holdout_metadata["sample_ids"],
                holdout_metadata["sample_seed"],
            )
            fit_group = build_benefit_feature_groups(fit_dataset)[winner_row["feature_group"]]
            holdout_group = build_benefit_feature_groups(holdout_dataset)[
                winner_row["feature_group"]
            ]
            fold_model = fit_benefit_audit_model(
                fit_dataset,
                fit_group,
                model_kind=winner_row["model_kind"],
                random_seed=20260718 + fold_index,
            )
            fold_calibration_predictions = predict_benefit_audit_model(
                fold_model,
                datasets["calibration"],
                feature_groups["calibration"][winner_row["feature_group"]],
            )
            fold_calibration = calibrate_safe_thresholds(
                datasets["calibration"], fold_calibration_predictions
            )
            holdout_predictions = predict_benefit_audit_model(
                fold_model, holdout_dataset, holdout_group
            )
            if fold_calibration.status == "safe_threshold":
                holdout_choice = select_benefit_candidates(
                    holdout_dataset,
                    holdout_predictions,
                    fold_calibration.opportunity_threshold,
                    fold_calibration.sign_threshold,
                )
            else:
                holdout_choice = np.full(
                    holdout_dataset.batch.candidate_mask.shape[0],
                    holdout_dataset.outcome.default_index,
                    dtype=np.int64,
                )
            fold_metrics = evaluate_benefit_predictions(
                holdout_dataset, holdout_predictions, holdout_choice
            )
            cv_rows.append(
                {
                    "fold": fold_index,
                    "fit_seeds": " ".join(
                        str(value) for value in sorted(np.unique(fit_dataset.sample_seed))
                    ),
                    "holdout_seeds": " ".join(
                        str(value) for value in sorted(np.unique(holdout_dataset.sample_seed))
                    ),
                    "calibration_status": fold_calibration.status,
                    "opportunity_threshold": fold_calibration.opportunity_threshold,
                    "sign_threshold": fold_calibration.sign_threshold,
                    **fold_metrics,
                }
            )
    trace = pd.DataFrame(
        _trace_rows(datasets["validation"], winner_predictions, winner_choice)
    )
    seed_rows = _seed_rows(datasets["validation"], winner_choice)
    family_stage_rows = _group_trace_rows(trace)

    pd.DataFrame(results).to_csv(output_dir / "feature_group_results.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_results.csv", index=False)
    pd.DataFrame(family_stage_rows).to_csv(output_dir / "family_stage_results.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "train_group_cv.csv", index=False)
    trace.to_csv(output_dir / "prediction_trace_validation.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "calibration_thresholds.csv", index=False)
    feature_manifest = {
        name: {
            "opportunity_feature_names": list(feature_groups["train"][name].opportunity_feature_names),
            "candidate_feature_names": list(feature_groups["train"][name].candidate_feature_names),
        }
        for name in args.feature_groups
    }
    (output_dir / "feature_group_manifest.json").write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        source_git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_git_sha = "unavailable"
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    command_lines = [
        "$ErrorActionPreference = 'Stop'",
        "Set-Location 'D:\\shen\\网络组\\代码'",
        "python scripts/audit_v11_candidate_benefit_identifiability.py `",
        f"  --train-cache '{cache_paths['train']}' `",
        f"  --calibration-cache '{cache_paths['calibration']}' `",
        f"  --validation-cache '{cache_paths['validation']}' `",
        f"  --output-dir '{output_dir}' `",
        f"  --required-schema-version {args.required_schema_version} `",
        f"  --model-kinds {' '.join(args.model_kinds)} `",
        f"  --feature-groups {' '.join(args.feature_groups)}",
    ]
    (output_dir / "reproduce.ps1").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "method": "candidate_benefit_identifiability_audit",
        "classification": classification,
        "result_kind": "diagnostic_only",
        "winner": winner_row,
        "hard_gate": winner_gate,
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "configuration_digest": configuration_digest,
        "source_git_sha": source_git_sha,
        "cache_sha256": {
            split: manifests[split]["cache_sha256"] for split in manifests
        },
        "cache_schema_version": int(args.required_schema_version),
        "token_protocol_sha256": token_protocol_sha256,
        "sample_limit_per_split": int(args.sample_limit_per_split),
        "model_status": model_status,
        "group_cv": {
            "fold_count": len(cv_rows),
            "active_rate_rmse_mean": float(
                np.mean([row["active_rate_rmse"] for row in cv_rows])
            )
            if cv_rows
            else None,
            "active_rate_rmse_std": float(
                np.std([row["active_rate_rmse"] for row in cv_rows])
            )
            if cv_rows
            else None,
            "candidate_sign_pr_auc_mean": float(
                np.mean([row["candidate_sign_pr_auc"] for row in cv_rows])
            )
            if cv_rows
            else None,
        },
        "runtime_seconds": float(time.time() - started),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "sha256_manifest.txt")
    (output_dir / "sha256_manifest.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
