"""Run the resumable PI-JWM R5.1 module-confirmation matrix on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r3_preflight_data import verify_r3_inputs
from pi_jwm.r4_gpu_screening import (
    build_training_window_schedule,
    build_validation_windows,
)
from pi_jwm.r5_checkpoint import load_r5_checkpoint
from pi_jwm.r5_confirmation_checkpoint import (
    CONFIRMATION_REQUIRED_BINDINGS,
    load_confirmation_checkpoint,
    save_confirmation_checkpoint,
)
from pi_jwm.r5_module_confirmation import (
    TRAINING_SEEDS,
    ConfirmationRunSpec,
    build_confirmation_matrix,
    build_confirmation_model,
    get_confirmation_config,
)
from pi_jwm.r5_protocol import REQUIRED_PUBLIC_METRIC_GATES, R5FormalProtocol, load_r5_protocol
from run_r4_gpu_screening import (
    _information_rate_stats,
    _verify_tensor_inputs,
    load_selection_scales,
)
from run_r5_gpu_training import (
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    _completed_run_is_valid,
    _input_fingerprint,
    _read_json,
    _run_dir,
    _sha256,
    _train_one_run,
    _write_json_atomic,
    _write_manifest,
    require_cuda_device,
    validate_training_splits,
)


SCHEMA_VERSION = "PIJWM-R5-Module-Confirmation-GPU-Training-v1"
RUN_STATE_SCHEMA = "PIJWM-R5-Module-Confirmation-Run-State-v1"
NEW_COMBINATIONS = tuple("FGHJ")
SOURCE_FILES = (
    SRC_ROOT / "pi_jwm" / "r3_preflight_data.py",
    SRC_ROOT / "pi_jwm" / "r3_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_module_registry.py",
    SRC_ROOT / "pi_jwm" / "r4_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_objective.py",
    SRC_ROOT / "pi_jwm" / "r5_protocol.py",
    SRC_ROOT / "pi_jwm" / "r5_world_model.py",
    SRC_ROOT / "pi_jwm" / "r5_module_confirmation.py",
    SRC_ROOT / "pi_jwm" / "r5_legacy_control.py",
    SRC_ROOT / "pi_jwm" / "r5_confirmation_checkpoint.py",
    CODE_ROOT / "scripts" / "run_r4_gpu_screening.py",
    CODE_ROOT / "scripts" / "run_r5_gpu_training.py",
    Path(__file__),
)


def build_new_run_specs(
    *,
    combination_ids: Sequence[str] = NEW_COMBINATIONS,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> tuple[ConfirmationRunSpec, ...]:
    combinations = tuple(str(value) for value in combination_ids)
    seeds = tuple(int(value) for value in training_seeds)
    if not combinations or len(set(combinations)) != len(combinations):
        raise ValueError("confirmation combination list must be non-empty and unique")
    unknown = sorted(set(combinations) - set(NEW_COMBINATIONS))
    if unknown:
        raise ValueError("unknown confirmation combination: " + ", ".join(unknown))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seed list must be non-empty and unique")
    if not set(seeds).issubset(TRAINING_SEEDS):
        raise ValueError("confirmation seed is outside the frozen protocol")
    return tuple(
        ConfirmationRunSpec(combination_id, seed, False)
        for combination_id in combinations
        for seed in seeds
    )


def require_confirmation_cuda(device: str, output_dir: str | Path) -> torch.device:
    return require_cuda_device(device, output_dir)


def _source_binding() -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"confirmation source binding is missing: {path}")
        relative = path.relative_to(CODE_ROOT).as_posix()
        value = _sha256(path)
        hashes[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest(), hashes


def _matrix_payload(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "combination_id": candidate.combination_id,
            "label": candidate.label,
            "question": candidate.question,
            "role": candidate.role,
            "components": candidate.components,
            "configuration": candidate.configuration,
        }
        for key, candidate in matrix.items()
    }


def _matrix_digest(matrix: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _matrix_payload(matrix),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_stored_protocol(payload: Mapping[str, Any], protocol: R5FormalProtocol) -> None:
    expected = {
        "training_seeds": list(protocol.training_seeds),
        "max_epochs": protocol.max_epochs,
        "patience": protocol.patience,
        "effective_batch_size": protocol.effective_batch_size,
        "minimum_improvement": protocol.minimum_improvement,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"reused B protocol mismatch: {key}")


def _window_identity(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        getter = value.get
    else:
        getter = lambda key: getattr(value, key)
    return (
        getter("environment_seed"),
        getter("history_start"),
        getter("history_end"),
        getter("target_start"),
        getter("target_end"),
        getter("horizon_steps"),
        getter("split"),
    )


def validate_reused_window_schedules(
    existing_r5_root: str | Path,
    schedules: Mapping[int, Sequence[Sequence[Any]]],
    validation_windows: Mapping[int, Sequence[Any]],
    calibration_windows: Mapping[int, Sequence[Any]],
) -> None:
    root = Path(existing_r5_root).resolve()
    stored_training = _read_json(root / "training_window_schedules.json")
    stored_validation = _read_json(root / "validation_windows.json")
    stored_calibration = _read_json(root / "calibration_windows.json")
    for seed, epochs in schedules.items():
        stored_epochs = stored_training.get(str(seed))
        if stored_epochs is None or len(stored_epochs) != len(epochs):
            raise ValueError(f"reused B training schedule length mismatch: seed {seed}")
        for index, (actual_epoch, stored_epoch) in enumerate(zip(epochs, stored_epochs), start=1):
            actual_identity = [_window_identity(window) for window in actual_epoch]
            stored_identity = [_window_identity(window) for window in stored_epoch["windows"]]
            if actual_identity != stored_identity:
                raise ValueError(f"reused B training schedule mismatch: seed {seed}, epoch {index}")
        for label, actual, stored in (
            ("validation", validation_windows[seed], stored_validation.get(str(seed))),
            ("calibration", calibration_windows[seed], stored_calibration.get(str(seed))),
        ):
            if stored is None or [_window_identity(window) for window in actual] != [_window_identity(window) for window in stored]:
                raise ValueError(f"reused B {label} window mismatch: seed {seed}")


def validate_reused_b_results(
    existing_r5_root: str | Path,
    protocol: R5FormalProtocol,
    *,
    expected_bindings: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(existing_r5_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("existing R5 manifest is missing")
    manifest = _read_json(manifest_path)
    if not manifest.get("r5_gpu_training_complete"):
        raise ValueError("existing R5 training is incomplete")
    if manifest.get("locked_test_accessed"):
        raise ValueError("existing R5 evidence accessed locked_test")
    _validate_stored_protocol(_read_json(root / "training_protocol.json"), protocol)
    provenance = _read_json(root / "input_provenance.json")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("existing R5 bindings are missing")
    if expected_bindings is not None:
        for key, expected in expected_bindings.items():
            if key == "source_code_sha256":
                continue
            if key not in bindings:
                continue
            if bindings.get(key) != expected:
                raise ValueError(f"reused B input binding mismatch: {key}")

    reports: list[dict[str, Any]] = []
    for seed in protocol.training_seeds:
        run_dir = root / "combinations" / "B" / f"seed_{seed}"
        report_path = run_dir / "run_report.json"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        for path in (report_path, checkpoint_path):
            relative = path.relative_to(root).as_posix()
            entry = manifest.get("files", {}).get(relative)
            if not isinstance(entry, Mapping):
                raise ValueError(f"reused B manifest entry is missing: {relative}")
            if int(entry.get("size_bytes", -1)) != path.stat().st_size:
                raise ValueError(f"reused B size mismatch: {relative}")
            if entry.get("sha256") != _sha256(path):
                raise ValueError(f"reused B hash mismatch: {relative}")
        report = _read_json(report_path)
        if report.get("combination_id") != "B" or int(report.get("training_seed", -1)) != seed:
            raise ValueError(f"reused B run identity mismatch: seed {seed}")
        if report.get("checkpoint_sha256") != _sha256(checkpoint_path):
            raise ValueError(f"reused B report checkpoint mismatch: seed {seed}")
        restored = load_r5_checkpoint(
            checkpoint_path,
            expected_bindings=bindings,
            expected_protocol=protocol,
        )
        if restored.model.combination_id != "B" or restored.seed != seed:
            raise ValueError(f"reused B checkpoint identity mismatch: seed {seed}")
        reused = dict(report)
        reused["evidence_origin"] = "reused_verified_r5_B"
        reused["source_manifest_sha256"] = _sha256(manifest_path)
        reports.append(reused)
    return reports


def verify_manifest_files(
    root: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("downloaded confirmation manifest is missing")
    manifest = _read_json(manifest_path)
    if require_complete and not manifest.get("r5_module_confirmation_complete"):
        raise ValueError("downloaded confirmation bundle is incomplete")
    if manifest.get("locked_test_accessed"):
        raise ValueError("downloaded confirmation bundle accessed locked_test")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("downloaded confirmation manifest has no files")
    for relative, entry in files.items():
        path = root / str(relative)
        if not path.is_file():
            raise ValueError(f"downloaded confirmation file is missing: {relative}")
        if int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"downloaded confirmation size mismatch: {relative}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"downloaded confirmation hash mismatch: {relative}")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_files_verified": True,
        "verified_file_count": len(files),
        "manifest_sha256": _sha256(manifest_path),
        "locked_test_accessed": False,
    }


def verify_smoke_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    file_report = verify_manifest_files(root, require_complete=False)
    for name in (
        "training_summary.json",
        "training_protocol.json",
        "input_provenance.json",
        "trained_run_reports.json",
        "reused_run_reports.json",
        "failed_runs.json",
    ):
        if not (root / name).is_file():
            raise FileNotFoundError(f"confirmation smoke artifact is missing: {name}")
    summary = _read_json(root / "training_summary.json")
    expected = {
        "r5_module_confirmation_complete": False,
        "smoke_only": True,
        "new_expected_run_count": 1,
        "new_completed_run_count": 1,
        "reused_run_count": 3,
        "total_evidence_run_count": 4,
        "failed_run_count": 0,
        "locked_test_accessed": False,
        "selection_status": "incomplete",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"confirmation smoke summary mismatch: {key}")
    protocol = _protocol_from_training_payload(_read_json(root / "training_protocol.json"))
    provenance = _read_json(root / "input_provenance.json")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(CONFIRMATION_REQUIRED_BINDINGS):
        raise ValueError("confirmation smoke bindings are incomplete")
    reports = _read_json(root / "trained_run_reports.json")
    reused = _read_json(root / "reused_run_reports.json")
    if len(reports) != 1 or len(reused) != 3 or _read_json(root / "failed_runs.json") != []:
        raise ValueError("confirmation smoke report counts are invalid")
    report = reports[0]
    combination = str(report.get("combination_id"))
    seed = int(report.get("training_seed", -1))
    if (combination, seed) != ("F", TRAINING_SEEDS[0]) or report.get("status") != "completed":
        raise ValueError("confirmation smoke run identity is invalid")
    checkpoint = root / "combinations" / combination / f"seed_{seed}" / "best_checkpoint.pt"
    if report.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("confirmation smoke checkpoint hash mismatch")
    restored = load_confirmation_checkpoint(
        checkpoint,
        expected_bindings=bindings,
        expected_protocol=protocol,
    )
    if restored.model.combination_id != combination or restored.seed != seed:
        raise ValueError("confirmation smoke checkpoint identity mismatch")
    _reject_locked_test_access(summary, location="smoke_summary")
    _reject_locked_test_access(provenance, location="smoke_provenance")
    _reject_locked_test_access(report, location="smoke_report")
    return {
        **file_report,
        "verified": True,
        "smoke_verified": True,
        "verified_checkpoint_count": 1,
    }


def _protocol_from_training_payload(payload: Mapping[str, Any]) -> R5FormalProtocol:
    return R5FormalProtocol(
        training_seeds=tuple(int(seed) for seed in payload["training_seeds"]),
        max_epochs=int(payload["max_epochs"]),
        patience=int(payload["patience"]),
        effective_batch_size=int(payload["effective_batch_size"]),
        minimum_improvement=float(payload["minimum_improvement"]),
    )


def _reject_locked_test_access(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "locked_test_accessed" and child is not False:
                raise ValueError(f"downloaded confirmation bundle accessed locked_test: {child_location}")
            _reject_locked_test_access(child, location=child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_locked_test_access(child, location=f"{location}[{index}]")


def verify_downloaded_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    file_report = verify_manifest_files(root)
    required = {
        "training_summary.json": "training summary",
        "training_protocol.json": "training protocol",
        "input_provenance.json": "input provenance",
        "combination_summary.json": "combination summary",
        "trained_run_reports.json": "trained run reports",
        "reused_run_reports.json": "reused run reports",
        "failed_runs.json": "failed run list",
    }
    for name, label in required.items():
        if not (root / name).is_file():
            raise FileNotFoundError(f"downloaded confirmation {label} is missing")

    summary = _read_json(root / "training_summary.json")
    expected_counts = {
        "r5_module_confirmation_complete": True,
        "smoke_only": False,
        "new_expected_run_count": 12,
        "new_completed_run_count": 12,
        "reused_run_count": 3,
        "total_evidence_run_count": 15,
        "failed_run_count": 0,
        "locked_test_accessed": False,
        "selection_status": "descriptive_only",
    }
    for key, expected in expected_counts.items():
        if summary.get(key) != expected:
            raise ValueError(f"downloaded confirmation summary mismatch: {key}")

    protocol_payload = _read_json(root / "training_protocol.json")
    protocol = _protocol_from_training_payload(protocol_payload)
    if protocol.training_seeds != TRAINING_SEEDS:
        raise ValueError("downloaded confirmation training seeds mismatch")
    if protocol_payload.get("new_combinations") != list(NEW_COMBINATIONS):
        raise ValueError("downloaded confirmation new combination list mismatch")
    if protocol_payload.get("reused_combination") != "B":
        raise ValueError("downloaded confirmation reused combination mismatch")
    if protocol_payload.get("smoke_only") is not False:
        raise ValueError("downloaded confirmation protocol is a smoke protocol")

    provenance = _read_json(root / "input_provenance.json")
    bindings = provenance.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(CONFIRMATION_REQUIRED_BINDINGS):
        raise ValueError("downloaded confirmation bindings are incomplete")
    _reject_locked_test_access(provenance, location="input_provenance")

    combination_summary = _read_json(root / "combination_summary.json")
    if set(combination_summary.get("combinations", {})) != set(("B", *NEW_COMBINATIONS)):
        raise ValueError("downloaded confirmation combination summary is incomplete")
    if combination_summary.get("selection_status") != "descriptive_only":
        raise ValueError("downloaded confirmation selection boundary mismatch")

    trained_reports = _read_json(root / "trained_run_reports.json")
    reused_reports = _read_json(root / "reused_run_reports.json")
    failed_runs = _read_json(root / "failed_runs.json")
    if not isinstance(trained_reports, list) or len(trained_reports) != 12:
        raise ValueError("downloaded confirmation trained report count mismatch")
    if not isinstance(reused_reports, list) or len(reused_reports) != 3:
        raise ValueError("downloaded confirmation reused report count mismatch")
    if failed_runs != []:
        raise ValueError("downloaded confirmation contains failed runs")

    expected_new = {
        (combination, seed)
        for combination in NEW_COMBINATIONS
        for seed in TRAINING_SEEDS
    }
    actual_new: set[tuple[str, int]] = set()
    for report in trained_reports:
        combination = str(report.get("combination_id"))
        seed = int(report.get("training_seed", -1))
        identity = (combination, seed)
        if identity not in expected_new or identity in actual_new:
            raise ValueError(f"downloaded confirmation run identity mismatch: {identity}")
        actual_new.add(identity)
        expected_run_id = f"{combination}__seed_{seed}"
        if report.get("run_id") != expected_run_id or report.get("status") != "completed":
            raise ValueError(f"downloaded confirmation report status mismatch: {expected_run_id}")
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"downloaded confirmation report schema mismatch: {expected_run_id}")
        run_dir = root / "combinations" / combination / f"seed_{seed}"
        stored_report = _read_json(run_dir / "run_report.json")
        if stored_report != report:
            raise ValueError(f"downloaded confirmation report copy mismatch: {expected_run_id}")
        checkpoint = run_dir / "best_checkpoint.pt"
        expected_checkpoint = checkpoint.relative_to(root).as_posix()
        if report.get("checkpoint") != expected_checkpoint:
            raise ValueError(f"downloaded confirmation checkpoint path mismatch: {expected_run_id}")
        if report.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError(f"downloaded confirmation checkpoint hash mismatch: {expected_run_id}")
        restored = load_confirmation_checkpoint(
            checkpoint,
            expected_bindings=bindings,
            expected_protocol=protocol,
        )
        if restored.model.combination_id != combination or restored.seed != seed:
            raise ValueError(f"downloaded confirmation checkpoint identity mismatch: {expected_run_id}")
        metrics = report.get("final_validation", {}).get("metrics", {})
        for metric_id in REQUIRED_PUBLIC_METRIC_GATES:
            if metrics.get(metric_id, {}).get("status") != "computed":
                raise ValueError(
                    f"downloaded confirmation metric is unavailable: {expected_run_id}, {metric_id}"
                )
        _reject_locked_test_access(report, location=expected_run_id)
    if actual_new != expected_new:
        raise ValueError("downloaded confirmation new-run matrix is incomplete")

    expected_reused = {("B", seed) for seed in TRAINING_SEEDS}
    actual_reused = {
        (str(report.get("combination_id")), int(report.get("training_seed", -1)))
        for report in reused_reports
    }
    if actual_reused != expected_reused:
        raise ValueError("downloaded confirmation reused B evidence is incomplete")
    for report in reused_reports:
        if report.get("status") != "completed" or report.get("evidence_origin") != "reused_verified_r5_B":
            raise ValueError("downloaded confirmation reused B evidence is invalid")
        if report.get("source_manifest_sha256") != bindings["existing_r5_manifest_sha256"]:
            raise ValueError("downloaded confirmation reused B manifest binding mismatch")
        _reject_locked_test_access(report, location=str(report.get("run_id", "B")))

    _reject_locked_test_access(summary, location="training_summary")
    _reject_locked_test_access(combination_summary, location="combination_summary")
    return {
        **file_report,
        "verified": True,
        "verified_new_run_count": len(actual_new),
        "verified_reused_run_count": len(actual_reused),
        "verified_checkpoint_count": len(actual_new),
    }


def _prepare_root(
    output_dir: Path,
    *,
    resume: bool,
    protocol: R5FormalProtocol,
    bindings: Mapping[str, str],
    input_fingerprint: str,
    matrix: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    input_report: Mapping[str, Any],
    existing_r5_root: Path,
    smoke: bool,
    micro_batch_size: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"confirmation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_payload = {
        **protocol.to_dict(),
        "schema_version": SCHEMA_VERSION,
        "learning_rate": LEARNING_RATE,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": protocol.effective_batch_size // micro_batch_size,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "validation_horizons": [1, 5, 20],
        "reused_combination": "B",
        "new_combinations": list(NEW_COMBINATIONS),
        "locked_test_accessed": False,
        "smoke_only": smoke,
    }
    matrix_payload = {"combinations": _matrix_payload(matrix)}
    provenance_payload = {
        **dict(input_report),
        "bindings": dict(bindings),
        "input_fingerprint": input_fingerprint,
        "source_files": dict(source_hashes),
        "existing_r5_root": str(existing_r5_root),
        "locked_test_accessed": False,
    }
    if resume:
        for name, expected in (
            ("training_protocol.json", protocol_payload),
            ("combination_matrix.json", matrix_payload),
            ("input_provenance.json", provenance_payload),
        ):
            path = output_dir / name
            if not path.is_file() or _read_json(path) != expected:
                raise ValueError(f"confirmation resume metadata mismatch: {name}")
        return
    _write_json_atomic(output_dir / "training_protocol.json", protocol_payload)
    _write_json_atomic(output_dir / "combination_matrix.json", matrix_payload)
    _write_json_atomic(output_dir / "input_provenance.json", provenance_payload)


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    return {
        "mean": statistics.fmean(normalized),
        "std": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
    }


def summarize_confirmation_runs(
    reports: Sequence[Mapping[str, Any]],
    protocol: R5FormalProtocol,
) -> dict[str, Any]:
    grouped = {combination: [] for combination in ("B", *NEW_COMBINATIONS)}
    for report in reports:
        grouped[str(report["combination_id"])].append(report)
    combinations: dict[str, Any] = {}
    for combination, rows in grouped.items():
        seeds = tuple(sorted(int(row["training_seed"]) for row in rows))
        if seeds != tuple(sorted(protocol.training_seeds)):
            raise ValueError(f"confirmation {combination} has incomplete seed reports")
        metrics = {
            metric_id: _mean_std(
                [float(row["final_validation"]["metrics"][metric_id]["value"]) for row in rows]
            )
            for metric_id in REQUIRED_PUBLIC_METRIC_GATES
        }
        combinations[combination] = {
            "training_seeds": list(seeds),
            "validation_protocol_score": _mean_std(
                [float(row["best_validation_protocol_score"]) for row in rows]
            ),
            "public_metrics": metrics,
            "evidence_origin": (
                "reused_verified_r5_B" if combination == "B" else "new_confirmation_training"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_status": "descriptive_only",
        "combinations": combinations,
        "claim_boundary": "same-protocol multi-seed evidence; no automatic final-method selection",
    }


def _persist_progress(
    output_dir: Path,
    trained_reports: Sequence[Mapping[str, Any]],
    reused_reports: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    failure_history: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[ConfirmationRunSpec],
    started: float,
    smoke: bool,
) -> dict[str, Any]:
    _write_json_atomic(output_dir / "trained_run_reports.json", list(trained_reports))
    _write_json_atomic(output_dir / "reused_run_reports.json", list(reused_reports))
    _write_json_atomic(output_dir / "failed_runs.json", list(failures))
    _write_json_atomic(output_dir / "failure_history.json", list(failure_history))
    complete = not smoke and len(trained_reports) == len(specs) and not failures
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r5_module_confirmation_complete": complete,
        "smoke_only": smoke,
        "new_expected_run_count": len(specs),
        "new_completed_run_count": len(trained_reports),
        "reused_run_count": len(reused_reports),
        "total_evidence_run_count": len(trained_reports) + len(reused_reports),
        "failed_run_count": len(failures),
        "locked_test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "selection_status": "descriptive_only" if complete else "incomplete",
    }
    _write_json_atomic(output_dir / "training_summary.json", summary)
    _write_manifest(output_dir, summary)
    return summary


def run_confirmation_training(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    r4_screening_root: str | Path,
    existing_r5_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    hidden_dim: int = 16,
    micro_batch_size: int = 4,
    combination_ids: Sequence[str] = NEW_COMBINATIONS,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    resume: bool = False,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    cuda_device = require_confirmation_cuda(device, output_dir)
    cuda_index = torch.cuda.current_device() if cuda_device.index is None else int(cuda_device.index)
    torch.cuda.set_device(cuda_index)
    validate_training_splits(("train", "validation", "calibration"))
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    r4_screening_root = Path(r4_screening_root).resolve()
    existing_r5_root = Path(existing_r5_root).resolve()
    output_dir = Path(output_dir).resolve()
    protocol = load_r5_protocol(evaluation_root)
    specs = build_new_run_specs(
        combination_ids=combination_ids,
        training_seeds=training_seeds,
    )
    smoke = smoke_epochs is not None
    if smoke and len(specs) != 1:
        raise ValueError("confirmation smoke must contain exactly one run")
    if micro_batch_size <= 0 or protocol.effective_batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must divide effective batch size")

    r4_summary = _read_json(r4_screening_root / "screening_summary.json")
    if not r4_summary.get("r4_gpu_screening_complete") or r4_summary.get("winner") != "graph_rssm_v1":
        raise ValueError("R4 screening binding is incomplete or has unexpected winner")
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization_stats = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    selection_scales = load_selection_scales(evaluation_root)
    rate_mean, rate_scale = _information_rate_stats(normalization_stats)
    matrix = build_confirmation_matrix(
        hidden_dim=hidden_dim,
        history_steps=8,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )
    schedules = {
        seed: build_training_window_schedule(
            dataset_root,
            epochs=protocol.max_epochs if smoke_epochs is None else smoke_epochs,
            windows_per_epoch=protocol.effective_batch_size,
            seed=seed,
        )
        for seed in sorted({spec.training_seed for spec in specs})
    }
    validation_windows = {
        seed: build_validation_windows(dataset_root, split="validation", horizons=(1, 5, 20), seed=seed)
        for seed in schedules
    }
    calibration_windows = {
        seed: build_validation_windows(dataset_root, split="calibration", horizons=(1, 5, 20), seed=seed)
        for seed in schedules
    }
    tensor_inputs = {
        str(seed): {
            "training": _verify_tensor_inputs(dataset_root, schedules[seed], ()),
            "validation": _verify_tensor_inputs(dataset_root, (), validation_windows[seed]),
            "calibration": _verify_tensor_inputs(dataset_root, (), calibration_windows[seed]),
        }
        for seed in schedules
    }
    source_digest, source_hashes = _source_binding()
    bindings = {
        **input_report["bindings"],
        "source_code_sha256": source_digest,
        "r4_screening_manifest_sha256": _sha256(r4_screening_root / "manifest.json"),
        "r5_protocol_sha256": _sha256(evaluation_root / "fair_experiment_protocol.json"),
        "existing_r5_manifest_sha256": _sha256(existing_r5_root / "manifest.json"),
        "confirmation_matrix_sha256": _matrix_digest(matrix),
    }
    if set(bindings) != set(CONFIRMATION_REQUIRED_BINDINGS):
        raise ValueError("confirmation source bindings are incomplete")
    fingerprint = _input_fingerprint(bindings)
    _prepare_root(
        output_dir,
        resume=resume,
        protocol=protocol,
        bindings=bindings,
        input_fingerprint=fingerprint,
        matrix=matrix,
        source_hashes=source_hashes,
        input_report={
            **input_report,
            "r4_screening_summary": r4_summary,
            "tensor_inputs": tensor_inputs,
        },
        existing_r5_root=existing_r5_root,
        smoke=smoke,
        micro_batch_size=micro_batch_size,
    )
    for name, values in (
        ("training_window_schedules.json", schedules),
        ("validation_windows.json", validation_windows),
        ("calibration_windows.json", calibration_windows),
    ):
        if name == "training_window_schedules.json":
            payload = {
                str(seed): [
                    {"epoch": index, "windows": [window.to_dict() for window in epoch]}
                    for index, epoch in enumerate(rows, start=1)
                ]
                for seed, rows in values.items()
            }
        else:
            payload = {str(seed): [window.to_dict() for window in rows] for seed, rows in values.items()}
        _write_json_atomic(output_dir / name, payload)
    reused_reports = validate_reused_b_results(
        existing_r5_root,
        protocol,
        expected_bindings=bindings,
    )
    if not smoke:
        validate_reused_window_schedules(
            existing_r5_root,
            schedules,
            validation_windows,
            calibration_windows,
        )

    def load_list(name: str) -> list[dict[str, Any]]:
        path = output_dir / name
        return _read_json(path) if resume and path.is_file() else []

    trained_reports = load_list("trained_run_reports.json")
    failures = load_list("failed_runs.json")
    failure_history = load_list("failure_history.json")
    started = time.perf_counter()
    for spec in specs:
        run_dir = _run_dir(output_dir, spec)
        if resume and _completed_run_is_valid(
            run_dir,
            spec,
            bindings=bindings,
            protocol=protocol,
            checkpoint_loader=load_confirmation_checkpoint,
        ):
            report = _read_json(run_dir / "run_report.json")
            trained_reports[:] = [row for row in trained_reports if row.get("run_id") != spec.run_id]
            trained_reports.append(report)
            failures[:] = [row for row in failures if row.get("run_id") != spec.run_id]
            print(json.dumps({"event": "confirmation_run_skipped", "run_id": spec.run_id}), flush=True)
            continue
        failures[:] = [row for row in failures if row.get("run_id") != spec.run_id]
        try:
            report = _train_one_run(
                spec,
                run_dir=run_dir,
                protocol=protocol,
                config=get_confirmation_config(
                    spec.combination_id,
                    hidden_dim=hidden_dim,
                    history_steps=8,
                    information_rate_mean=rate_mean,
                    information_rate_scale=rate_scale,
                ),
                bindings=bindings,
                input_fingerprint=fingerprint,
                train_schedule=schedules[spec.training_seed],
                validation_windows=validation_windows[spec.training_seed],
                calibration_windows=calibration_windows[spec.training_seed],
                normalization_stats=normalization_stats,
                selection_scales=selection_scales,
                device=cuda_device,
                micro_batch_size=micro_batch_size,
                smoke_epochs=smoke_epochs,
                resume=resume,
                run_started=time.perf_counter(),
                model_builder=build_confirmation_model,
                checkpoint_saver=save_confirmation_checkpoint,
                checkpoint_loader=load_confirmation_checkpoint,
                report_schema=SCHEMA_VERSION,
                run_state_schema=RUN_STATE_SCHEMA,
            )
            trained_reports[:] = [row for row in trained_reports if row.get("run_id") != spec.run_id]
            trained_reports.append(report)
            print(json.dumps({"event": "confirmation_run_complete", "run_id": spec.run_id}), flush=True)
        except Exception as error:
            failure = {
                "run_id": spec.run_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "locked_test_accessed": False,
            }
            failures.append(failure)
            failure_history.append(failure)
            _write_json_atomic(run_dir / "failure.json", failure)
            print(json.dumps({"event": "confirmation_run_failed", "run_id": spec.run_id, "error": str(error)}), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        _persist_progress(
            output_dir,
            trained_reports,
            reused_reports,
            failures,
            failure_history,
            specs=specs,
            started=started,
            smoke=smoke,
        )
    summary = _persist_progress(
        output_dir,
        trained_reports,
        reused_reports,
        failures,
        failure_history,
        specs=specs,
        started=started,
        smoke=smoke,
    )
    if summary["r5_module_confirmation_complete"]:
        combined = [*reused_reports, *trained_reports]
        _write_json_atomic(
            output_dir / "combination_summary.json",
            summarize_confirmation_runs(combined, protocol),
        )
        _write_manifest(output_dir, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--r4-screening-root", required=True)
    parser.add_argument("--existing-r5-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--combination", action="append", dest="combinations")
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_confirmation_training(
        args.dataset_root,
        args.evaluation_root,
        args.r4_screening_root,
        args.existing_r5_root,
        args.output_dir,
        device=args.device,
        hidden_dim=args.hidden_dim,
        micro_batch_size=args.micro_batch_size,
        combination_ids=NEW_COMBINATIONS if args.combinations is None else args.combinations,
        training_seeds=TRAINING_SEEDS if args.seeds is None else args.seeds,
        resume=args.resume,
        smoke_epochs=args.smoke_epochs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
