"""Evaluate one validation-frozen PI-JWM v11 selector without retuning it."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from pi_jwm.v11_labeling import load_candidate_label_cache
from pi_jwm.v11_selector import (
    CandidateOutcome,
    aggregate_selected_metrics,
    file_sha256,
    load_fitted_selector_checkpoint,
    observable_pareto_deltas,
    predict_fitted_selector,
    select_with_defer,
    verify_selector_freeze_manifest,
)


def validate_frozen_evaluation_inputs(
    frozen_manifest: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
) -> None:
    verify_selector_freeze_manifest(frozen_manifest)
    payload = frozen_manifest["selector_freeze_payload"]
    if not bool(frozen_manifest.get("configuration_frozen")):
        raise PermissionError("selector configuration must be frozen on validation before evaluation")
    split = str(cache_manifest.get("split_name", ""))
    if split not in {"matched_test", "external_holdout"}:
        raise ValueError("frozen evaluator accepts only matched_test or external_holdout caches")
    frozen_digest = str(frozen_manifest.get("configuration_digest", ""))
    cache_digest = str(cache_manifest.get("configuration_digest", ""))
    if len(frozen_digest) != 64 or cache_digest != frozen_digest:
        raise ValueError("frozen selector and label cache configuration digest mismatch")
    if str(payload.get("configuration_digest")) != frozen_digest:
        raise ValueError("top-level configuration differs from frozen selector payload")
    payload_config = payload.get("selected_config", {})
    selected_config = frozen_manifest.get("selected_config", {})
    for key, value in payload_config.items():
        if selected_config.get(key) != value:
            raise ValueError("top-level selected config differs from frozen selector payload")
    checkpoints = frozen_manifest.get("selected_checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("frozen manifest does not contain selected checkpoints")
    records = frozen_manifest.get("selected_checkpoint_records")
    if not isinstance(records, list) or len(records) != len(checkpoints):
        raise ValueError("frozen manifest does not bind every selected checkpoint")
    if records != payload.get("checkpoint_records"):
        raise ValueError("top-level checkpoint records differ from frozen selector payload")
    if {Path(str(value)).name for value in checkpoints} != {
        str(record.get("file", "")) for record in records
    }:
        raise ValueError("frozen checkpoint records do not match selected checkpoints")


def verify_checkpoint_record(path: Path, record: Mapping[str, Any]) -> str:
    expected_name = str(record.get("file", ""))
    expected_sha = str(record.get("sha256", ""))
    if path.name != expected_name:
        raise ValueError("selected checkpoint filename does not match frozen record")
    actual_sha = file_sha256(path)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError("selected checkpoint SHA-256 mismatch")
    return actual_sha


def validate_external_evidence(
    summary: Mapping[str, Any], configuration_digest: str, selector_freeze_digest: str
) -> int:
    if str(summary.get("split_name")) != "external_holdout":
        raise ValueError("external evidence must come from external_holdout")
    if str(summary.get("configuration_digest")) != str(configuration_digest):
        raise ValueError("external evidence configuration digest mismatch")
    if str(summary.get("selector_freeze_digest")) != str(selector_freeze_digest):
        raise ValueError("external evidence selector freeze digest mismatch")
    rows = summary.get("per_seed")
    if not isinstance(rows, list):
        raise ValueError("external evidence must contain per-seed rows")
    observed = {int(row["seed"]) for row in rows}
    if observed != set(range(60, 70)) or len(rows) != 10:
        raise ValueError("external evidence must cover seeds 60-69 exactly once")
    return sum(
        float(row["active_rate_rmse"]) < float(row["default_active_rate_rmse"])
        for row in rows
    )


def validate_safety_evidence(
    audit: Mapping[str, Any], configuration_digest: str, selector_freeze_digest: str
) -> int:
    if str(audit.get("result_kind")) != "actual_airfogsim_safety_audit":
        raise ValueError("safety evidence must be an actual AirFogSim audit")
    if str(audit.get("configuration_digest")) != str(configuration_digest):
        raise ValueError("safety evidence configuration digest mismatch")
    if str(audit.get("selector_freeze_digest")) != str(selector_freeze_digest):
        raise ValueError("safety evidence selector freeze digest mismatch")
    rows = audit.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("safety evidence requires aligned physical rows")
    violations = 0
    for row in rows:
        task_delta = float(row["task_utility_delta_actual"])
        energy_delta = float(row["energy_delta_actual"])
        if not np.isfinite(task_delta) or not np.isfinite(energy_delta):
            raise ValueError("safety evidence physical deltas must be finite")
        violations += int(task_delta < 0.0 and energy_delta > 0.0)
    return violations


def record_locked_split_access(
    ledger_path: Path,
    split_name: str,
    configuration_digest: str,
    selector_freeze_digest: str,
    cache_sha256: str,
) -> dict[str, Any]:
    split = str(split_name)
    if split not in {"matched_test", "external_holdout"}:
        raise ValueError("access ledger is only for locked evaluation splits")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if ledger_path.exists():
        existing = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    key = (split, str(configuration_digest), str(selector_freeze_digest))
    if any(
        (str(row.get("split_name")), str(row.get("configuration_digest")), str(row.get("selector_freeze_digest")))
        == key
        for row in existing
    ):
        raise PermissionError(f"{split} already accessed for this frozen selector")
    record = {
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": split,
        "configuration_digest": str(configuration_digest),
        "selector_freeze_digest": str(selector_freeze_digest),
        "cache_sha256": str(cache_sha256),
    }
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def compute_acceptance_report(
    metrics: Mapping[str, Any],
    validation_seed_std: float,
    external_seed_wins: int | None,
    pareto_violations: int | None,
) -> dict[str, Any]:
    active_rmse = float(metrics["active_rate_rmse"])
    if active_rmse < 200.0:
        rmse_tier = "A"
    elif active_rmse < 213.160874:
        rmse_tier = "B"
    else:
        rmse_tier = "not_passed"
    link = metrics.get("link_rmse")
    default_link = metrics.get("default_link_rmse")
    f1 = metrics.get("activity_f1")
    default_f1 = metrics.get("default_activity_f1")
    link_degradation = (
        None
        if link is None or default_link in (None, 0)
        else float(link) / float(default_link) - 1.0
    )
    activity_f1_drop = None if f1 is None or default_f1 is None else float(default_f1) - float(f1)
    gates = {
        "validation_seed_std_le_5": float(validation_seed_std) <= 5.0,
        "external_wins_ge_7_of_10": external_seed_wins is not None and int(external_seed_wins) >= 7,
        "activity_f1_drop_le_0_002": activity_f1_drop is not None and activity_f1_drop <= 0.0020001,
        "link_rmse_degradation_le_2pct": link_degradation is not None and link_degradation <= 0.0200001,
        "pareto_violations_zero": pareto_violations is not None and int(pareto_violations) == 0,
    }
    all_gates = all(gates.values())
    if all_gates and rmse_tier in {"A", "B"}:
        final_tier = rmse_tier
    elif rmse_tier in {"A", "B"} and (
        external_seed_wins is None
        or pareto_violations is None
        or link_degradation is None
        or activity_f1_drop is None
    ):
        final_tier = "pending_external_or_safety_gate"
    elif rmse_tier in {"A", "B"}:
        final_tier = "failed_robustness_or_safety_gate"
    else:
        final_tier = "not_passed"
    return {
        "rmse_tier": rmse_tier,
        "final_tier": final_tier,
        "active_rate_rmse": active_rmse,
        "validation_seed_std": float(validation_seed_std),
        "external_seed_wins": external_seed_wins,
        "activity_f1_drop": activity_f1_drop,
        "link_rmse_relative_degradation": link_degradation,
        "pareto_violations": pareto_violations,
        "gates": gates,
    }


def _masked_oracle(outcome: CandidateOutcome, mask: np.ndarray) -> np.ndarray:
    available = np.asarray(mask, dtype=bool)
    if available.shape != outcome.active_sse.shape or np.any(available.sum(axis=1) == 0):
        raise ValueError("candidate mask is incompatible with outcome")
    return np.argmin(np.where(available, outcome.active_sse, np.inf), axis=1).astype(np.int64)


def _metrics_with_baselines(outcome: CandidateOutcome, choice: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = aggregate_selected_metrics(outcome, choice)
    default_choice = np.full(choice.shape, outcome.default_index, dtype=np.int64)
    default = aggregate_selected_metrics(outcome, default_choice)
    oracle_choice = _masked_oracle(outcome, mask)
    oracle = aggregate_selected_metrics(outcome, oracle_choice)
    return {
        **selected,
        "default_active_rate_rmse": default["active_rate_rmse"],
        "default_link_rmse": default["link_rmse"],
        "default_activity_f1": default["activity_f1"],
        "sample_oracle_active_rate_rmse": oracle["active_rate_rmse"],
        "improvement_vs_default": float(default["active_rate_rmse"] - selected["active_rate_rmse"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_candidate_family(candidate_name: str) -> str:
    name = str(candidate_name).lower()
    if name == "identity":
        return "identity"
    if name == "ranked_allocation_baseline" or name.startswith("historical_"):
        return "baseline"
    if name.startswith("offload_"):
        return "offload_rb"
    if name.startswith("compute_"):
        return "compute_cpu"
    if name.startswith("return_"):
        return "return_route"
    if name.startswith("benefit_residual"):
        return "rb_benefit_residual"
    if name.startswith("rb_repair"):
        return "rb_repair"
    return "other"


def _slice_outcome(outcome: CandidateOutcome, keep: np.ndarray) -> CandidateOutcome:
    return CandidateOutcome(
        active_sse=outcome.active_sse[keep],
        active_count=outcome.active_count[keep],
        link_sse=None if outcome.link_sse is None else outcome.link_sse[keep],
        link_count=None if outcome.link_count is None else outcome.link_count[keep],
        activity_tp=None if outcome.activity_tp is None else outcome.activity_tp[keep],
        activity_fp=None if outcome.activity_fp is None else outcome.activity_fp[keep],
        activity_fn=None if outcome.activity_fn is None else outcome.activity_fn[keep],
        activity_tn=None if outcome.activity_tn is None else outcome.activity_tn[keep],
        action_applied=None if outcome.action_applied is None else outcome.action_applied[keep],
        action_applicable=(
            None if outcome.action_applicable is None else outcome.action_applicable[keep]
        ),
        task_utility=None if outcome.task_utility is None else outcome.task_utility[keep],
        energy_total=None if outcome.energy_total is None else outcome.energy_total[keep],
        default_index=outcome.default_index,
        result_kind=outcome.result_kind,
    )


def _resolve_checkpoint(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    relative = manifest_path.parent / path.name
    if relative.exists():
        return relative
    raise FileNotFoundError(f"selected checkpoint not found: {path_value}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.frozen_manifest.resolve()
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch, outcome, cache_manifest = load_candidate_label_cache(args.cache)
    validate_frozen_evaluation_inputs(frozen, cache_manifest)
    digest = str(frozen["configuration_digest"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access_record = record_locked_split_access(
        args.output_dir / "test_access_ledger.jsonl",
        split_name=str(cache_manifest["split_name"]),
        configuration_digest=digest,
        selector_freeze_digest=str(frozen["selector_freeze_digest"]),
        cache_sha256=str(cache_manifest["cache_sha256"]),
    )
    ensemble_rank_scores = []
    ensemble_improvements = []
    ensemble_uncertainties = []
    checkpoint_metadata = []
    record_by_name = {
        str(record["file"]): record for record in frozen["selected_checkpoint_records"]
    }
    for value in frozen["selected_checkpoints"]:
        checkpoint = _resolve_checkpoint(str(value), manifest_path)
        record = record_by_name[checkpoint.name]
        checkpoint_sha = verify_checkpoint_record(checkpoint, record)
        fitted, bias, metadata = load_fitted_selector_checkpoint(
            checkpoint,
            expected_configuration_digest=digest,
            device=args.device,
        )
        if not np.isclose(float(record["calibration_bias"]), float(bias), rtol=0.0, atol=1e-12):
            raise ValueError("checkpoint calibration bias differs from frozen record")
        if not np.isclose(
            float(record["target_scale"]), float(fitted.target_scale), rtol=0.0, atol=1e-12
        ):
            raise ValueError("checkpoint target scale differs from frozen record")
        heads = predict_fitted_selector(fitted, batch)
        ensemble_rank_scores.append(heads["score"])
        ensemble_improvements.append(heads["predicted_improvement"] - float(bias))
        ensemble_uncertainties.append(heads["uncertainty"])
        checkpoint_metadata.append(
            {
                **metadata,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "calibration_bias": bias,
            }
        )
    task_delta, energy_delta = observable_pareto_deltas(batch, outcome.default_index)
    decision = select_with_defer(
        np.stack(ensemble_rank_scores, axis=0),
        default_index=outcome.default_index,
        ensemble_improvement=np.stack(ensemble_improvements, axis=0),
        ensemble_uncertainty=np.stack(ensemble_uncertainties, axis=0),
        task_delta=task_delta,
        energy_delta=energy_delta,
        candidate_mask=batch.candidate_mask,
    )
    metrics = _metrics_with_baselines(outcome, decision.candidate_index, batch.candidate_mask)
    with np.load(args.cache, allow_pickle=False) as arrays:
        sample_ids = arrays["sample_ids"].astype(np.int64)
        sample_seed = arrays["sample_seed"].astype(np.int64)
    rows = []
    selected_sse = outcome.active_sse[np.arange(sample_ids.shape[0]), decision.candidate_index]
    for index, record in enumerate(decision.to_records(sample_ids)):
        candidate = int(decision.candidate_index[index])
        count = int(outcome.active_count[index])
        rows.append(
            {
                **record,
                "seed": int(sample_seed[index]),
                "stage": str(batch.stage[index]),
                "candidate_name": batch.candidate_names[candidate],
                "action_family": infer_candidate_family(batch.candidate_names[candidate]),
                "candidate_available": bool(batch.candidate_mask[index, candidate]),
                "actual_active_rate_rmse": float(np.sqrt(selected_sse[index] / count)) if count else None,
                "predicted_task_delta_proxy": float(task_delta[index, candidate]),
                "predicted_energy_delta_proxy": float(energy_delta[index, candidate]),
                "result_kind": str(cache_manifest["split_name"]),
            }
        )
    per_seed = []
    external_wins = 0
    for seed in sorted(int(value) for value in np.unique(sample_seed)):
        keep = sample_seed == seed
        seed_metrics = _metrics_with_baselines(
            _slice_outcome(outcome, keep),
            decision.candidate_index[keep],
            batch.candidate_mask[keep],
        )
        seed_metrics["seed"] = seed
        per_seed.append(seed_metrics)
        external_wins += int(
            seed_metrics["active_rate_rmse"] < seed_metrics["default_active_rate_rmse"]
        )
    group_rows = []
    selected_families = np.asarray(
        [infer_candidate_family(batch.candidate_names[int(value)]) for value in decision.candidate_index]
    )
    for group_type, values in (("stage", batch.stage.astype(str)), ("action_family", selected_families)):
        for group_value in sorted(str(value) for value in np.unique(values)):
            keep = values == group_value
            group_metrics = _metrics_with_baselines(
                _slice_outcome(outcome, keep),
                decision.candidate_index[keep],
                batch.candidate_mask[keep],
            )
            group_rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "num_samples": int(np.sum(keep)),
                    "defer_ratio": float(np.mean(decision.deferred[keep])),
                    **group_metrics,
                }
            )
    split_name = str(cache_manifest["split_name"])
    validation_std = float(frozen.get("selected_config", {}).get("seed_std", float("inf")))
    selector_freeze_digest = str(frozen["selector_freeze_digest"])
    evidence_files: dict[str, Any] = {}
    if split_name == "external_holdout":
        external_seed_wins = external_wins
    elif args.external_summary is not None:
        external_path = args.external_summary.resolve()
        external_summary = json.loads(external_path.read_text(encoding="utf-8"))
        external_seed_wins = validate_external_evidence(
            external_summary, digest, selector_freeze_digest
        )
        evidence_files["external_summary"] = {
            "path": str(external_path),
            "sha256": file_sha256(external_path),
        }
    else:
        external_seed_wins = None
    if args.safety_audit is not None:
        safety_path = args.safety_audit.resolve()
        safety_audit = json.loads(safety_path.read_text(encoding="utf-8"))
        pareto_violations = validate_safety_evidence(
            safety_audit, digest, selector_freeze_digest
        )
        evidence_files["safety_audit"] = {
            "path": str(safety_path),
            "sha256": file_sha256(safety_path),
        }
    else:
        pareto_violations = None
    acceptance = compute_acceptance_report(
        metrics,
        validation_seed_std=validation_std,
        external_seed_wins=external_seed_wins,
        pareto_violations=pareto_violations,
    )
    _write_csv(args.output_dir / f"decision_trace_{split_name}.csv", rows)
    _write_csv(args.output_dir / f"per_seed_{split_name}.csv", per_seed)
    _write_csv(args.output_dir / f"stage_family_{split_name}.csv", group_rows)
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "split_name": split_name,
        "result_kind": split_name,
        "configuration_digest": digest,
        "selector_freeze_digest": selector_freeze_digest,
        "matched_test_accessed": split_name == "matched_test",
        "metrics": metrics,
        "per_seed": per_seed,
        "stage_family": group_rows,
        "external_seed_wins": external_wins if split_name == "external_holdout" else None,
        "acceptance": acceptance,
        "defer_ratio": float(np.mean(decision.deferred)),
        "observable_pareto_applied": task_delta is not None,
        "checkpoints": checkpoint_metadata,
        "evidence_files": evidence_files,
        "access_ledger_record": access_record,
    }
    (args.output_dir / f"summary_{split_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--external-summary", type=Path)
    parser.add_argument("--safety-audit", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
