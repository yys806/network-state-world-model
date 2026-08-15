"""Run resumable PI-JWM R5 formal multi-seed training on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_protocol import REQUIRED_PUBLIC_METRIC_GATES, R5FormalProtocol
from pi_jwm.r5_checkpoint import (
    R5_REQUIRED_BINDINGS,
    load_r5_checkpoint,
    save_r5_checkpoint,
)
from pi_jwm.r5_protocol import (
    load_r5_protocol,
    r5_combination_matrix,
)
from pi_jwm.r5_world_model import build_r5_world_model
from pi_jwm.r3_preflight_data import (
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
    verify_r3_inputs,
)
from pi_jwm.r4_gpu_screening import (
    R4ValidationAccumulator,
    build_training_window_schedule,
    build_validation_windows,
    move_explicit_batch,
)
from pi_jwm.r4_objective import compute_r4_objective
from run_r4_gpu_screening import (
    _information_rate_stats,
    _validate_candidate,
    _verify_tensor_inputs,
    _window_batches,
    load_selection_scales,
)


SCHEMA_VERSION = "PIJWM-R5-Formal-GPU-Training-v1"
RUN_STATE_SCHEMA = "PIJWM-R5-GPU-Run-State-v1"
APPROVED_COMBINATIONS = tuple("ABCDE")
LEARNING_RATE = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
R5_SOURCE_FILES = (
    SRC_ROOT / "pi_jwm" / "airfogsim_sparse_diagnostics_v2.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_teacher_tensor_v3.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_tensor_v2.py",
    SRC_ROOT / "pi_jwm" / "formal_airfogsim_graph_v1.py",
    SRC_ROOT / "pi_jwm" / "r3_preflight_data.py",
    SRC_ROOT / "pi_jwm" / "r3_world_model.py",
    SRC_ROOT / "pi_jwm" / "r3_objective.py",
    SRC_ROOT / "pi_jwm" / "r3_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_module_registry.py",
    SRC_ROOT / "pi_jwm" / "r4_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_objective.py",
    SRC_ROOT / "pi_jwm" / "r4_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_gpu_screening.py",
    SRC_ROOT / "pi_jwm" / "r5_protocol.py",
    SRC_ROOT / "pi_jwm" / "r5_world_model.py",
    SRC_ROOT / "pi_jwm" / "r5_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "teacher_evaluation_v3.py",
    CODE_ROOT / "scripts" / "run_r4_gpu_screening.py",
    Path(__file__),
)


@dataclass(frozen=True)
class R5RunSpec:
    combination_id: str
    training_seed: int

    @property
    def run_id(self) -> str:
        return f"{self.combination_id}__seed_{self.training_seed}"


def build_run_specs(
    protocol: R5FormalProtocol,
    *,
    combination_ids: Sequence[str] = APPROVED_COMBINATIONS,
    training_seeds: Sequence[int] | None = None,
) -> tuple[R5RunSpec, ...]:
    combinations = tuple(str(value) for value in combination_ids)
    seeds = (
        protocol.training_seeds
        if training_seeds is None
        else tuple(int(value) for value in training_seeds)
    )
    if not combinations or len(set(combinations)) != len(combinations):
        raise ValueError("R5 GPU combination list must be non-empty and unique")
    unknown = sorted(set(combinations) - set(APPROVED_COMBINATIONS))
    if unknown:
        raise ValueError("unknown R5 GPU combination: " + ", ".join(unknown))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("R5 GPU seed list must be non-empty and unique")
    if not set(seeds).issubset(protocol.training_seeds):
        raise ValueError("R5 GPU seed is outside the frozen protocol")
    return tuple(
        R5RunSpec(combination_id, seed)
        for combination_id in combinations
        for seed in seeds
    )


def require_cuda_device(device: str, output_dir: str | Path) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal R5 training requires an available CUDA device")
    return requested


def validate_training_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in splits)
    if "locked_test" in normalized:
        raise ValueError("locked_test cannot be used by R5 formal training")
    if normalized != ("train", "validation", "calibration"):
        raise ValueError(
            "R5 formal training splits must be train, validation, calibration"
        )
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_binding() -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in R5_SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"R5 source binding is missing: {path}")
        relative = path.relative_to(CODE_ROOT).as_posix()
        value = _sha256(path)
        hashes[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest(), hashes


def _score_reproduced(expected: float, actual: float, tolerance: float) -> bool:
    return (
        math.isfinite(float(expected))
        and math.isfinite(float(actual))
        and abs(float(expected) - float(actual)) <= float(tolerance)
    )


def _calibration_report(
    model: torch.nn.Module,
    windows: Sequence[Any],
    normalization_stats: Mapping[str, Any],
    selection_scales: Mapping[str, float],
    *,
    device: torch.device,
    micro_batch_size: int,
    validation_seed: int,
) -> dict[str, Any]:
    return _validate_candidate(
        model,
        windows,
        normalization_stats,
        selection_scales,
        device=device,
        micro_batch_size=micro_batch_size,
        validation_seed=validation_seed,
    )


def _run_dir(output_dir: Path, spec: R5RunSpec) -> Path:
    return output_dir / "combinations" / spec.combination_id / f"seed_{spec.training_seed}"


def _input_fingerprint(bindings: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(bindings), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_manifest(root: Path, summary: Mapping[str, Any]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "manifest.json" or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json_atomic(
        root / "manifest.json",
        {
            **dict(summary),
            "files": files,
            "manifest_entry_count": len(files),
        },
    )


def _persist_progress(
    output_dir: Path,
    reports: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[R5RunSpec],
    protocol: R5FormalProtocol,
    started: float,
    smoke: bool,
) -> dict[str, Any]:
    _write_json_atomic(output_dir / "run_reports.json", list(reports))
    _write_json_atomic(output_dir / "failed_runs.json", list(failures))
    _write_json_atomic(output_dir / "failure_history.json", list(history))
    complete = (
        not smoke
        and len(reports) == len(specs)
        and not failures
        and len(specs) == len(APPROVED_COMBINATIONS) * len(protocol.training_seeds)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r5_gpu_training_complete": complete,
        "smoke_only": smoke,
        "expected_run_count": len(specs),
        "completed_run_count": len(reports),
        "failed_run_count": len(failures),
        "locked_test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "selection_status": "descriptive_only" if complete else "incomplete",
        "claim_boundary": "multi-seed training evidence; no automatic final-method selection",
    }
    _write_json_atomic(output_dir / "training_summary.json", summary)
    _write_manifest(output_dir, summary)
    return summary


def _load_existing_reports(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def load(name: str) -> list[dict[str, Any]]:
        path = output_dir / name
        return _read_json(path) if path.is_file() else []

    return load("run_reports.json"), load("failed_runs.json"), load("failure_history.json")


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
    smoke: bool,
    micro_batch_size: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"R5 output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_payload = {
        **protocol.to_dict(),
        "schema_version": SCHEMA_VERSION,
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": protocol.effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": protocol.effective_batch_size
        // micro_batch_size,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "validation_horizons": [1, 5, 20],
        "locked_test_accessed": False,
        "smoke_only": smoke,
    }
    protocol_path = output_dir / "training_protocol.json"
    provenance_path = output_dir / "input_provenance.json"
    matrix_payload = {
        "combinations": {key: value.to_dict() for key, value in matrix.items()}
    }
    if resume and protocol_path.is_file() and provenance_path.is_file():
        existing = _read_json(provenance_path)
        if existing.get("bindings") != dict(bindings):
            raise ValueError("R5 resume input binding mismatch")
        if existing.get("input_fingerprint") != input_fingerprint:
            raise ValueError("R5 resume input fingerprint mismatch")
        if _read_json(protocol_path) != protocol_payload:
            raise ValueError("R5 resume protocol or runtime budget mismatch")
        if _read_json(output_dir / "combination_matrix.json") != matrix_payload:
            raise ValueError("R5 resume combination configuration mismatch")
        return
    if resume:
        raise ValueError("R5 resume metadata is incomplete")
    _write_json_atomic(output_dir / "training_protocol.json", protocol_payload)
    _write_json_atomic(
        output_dir / "combination_matrix.json",
        matrix_payload,
    )
    _write_json_atomic(
        output_dir / "input_provenance.json",
        {
            **dict(input_report),
            "bindings": dict(bindings),
            "input_fingerprint": input_fingerprint,
            "r5_source_files": dict(source_hashes),
            "locked_test_accessed": False,
        },
    )


def _completed_run_is_valid(
    run_dir: Path,
    spec: R5RunSpec,
    *,
    bindings: Mapping[str, str],
    protocol: R5FormalProtocol,
    checkpoint_loader=load_r5_checkpoint,
) -> bool:
    report_path = run_dir / "run_report.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not report_path.is_file() or not checkpoint_path.is_file():
        return False
    report = _read_json(report_path)
    if report.get("status") != "completed":
        return False
    if report.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError(f"R5 completed run hash mismatch: {spec.run_id}")
    restored = checkpoint_loader(
        checkpoint_path,
        expected_bindings=bindings,
        expected_protocol=protocol,
    )
    if restored.model.combination_id != spec.combination_id:
        raise ValueError(f"R5 completed run combination mismatch: {spec.run_id}")
    if restored.seed != spec.training_seed:
        raise ValueError(f"R5 completed run seed mismatch: {spec.run_id}")
    return True


def _train_one_run(
    spec: R5RunSpec,
    *,
    run_dir: Path,
    protocol: R5FormalProtocol,
    config: Any,
    bindings: Mapping[str, str],
    input_fingerprint: str,
    train_schedule: Sequence[Sequence[Any]],
    validation_windows: Sequence[Any],
    calibration_windows: Sequence[Any],
    normalization_stats: Mapping[str, Any],
    selection_scales: Mapping[str, float],
    device: torch.device,
    micro_batch_size: int,
    smoke_epochs: int | None,
    resume: bool,
    run_started: float,
    model_builder=build_r5_world_model,
    checkpoint_saver=save_r5_checkpoint,
    checkpoint_loader=load_r5_checkpoint,
    report_schema: str = SCHEMA_VERSION,
    run_state_schema: str = RUN_STATE_SCHEMA,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    curve_path = run_dir / "training_curve.json"
    state_path = run_dir / "run_state.json"
    torch.manual_seed(spec.training_seed)
    torch.cuda.manual_seed_all(spec.training_seed)
    torch.cuda.reset_peak_memory_stats(device)
    model = model_builder(
        spec.combination_id,
        hidden_dim=config.hidden_dim,
        history_steps=config.history_steps,
        information_rate_mean=config.information_rate_mean,
        information_rate_scale=config.information_rate_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    curve: list[dict[str, Any]] = _read_json(curve_path) if curve_path.is_file() else []
    best_score = math.inf
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    resume_artifacts = (state_path, last_checkpoint_path, curve_path)
    if resume and any(path.is_file() for path in resume_artifacts) and not all(
        path.is_file() for path in resume_artifacts
    ):
        raise ValueError("R5 resume artifacts are incomplete")
    if resume and all(path.is_file() for path in resume_artifacts):
        state = load_resumable_run_state(
            run_dir,
            combination_id=spec.combination_id,
            training_seed=spec.training_seed,
            input_fingerprint=input_fingerprint,
            expected_schema=run_state_schema,
        )
        restored = checkpoint_loader(
            last_checkpoint_path,
            expected_bindings=bindings,
            expected_protocol=protocol,
        )
        model.load_state_dict(restored.model.state_dict(), strict=True)
        if restored.optimizer_state is None:
            raise ValueError("R5 resume checkpoint has no optimizer state")
        optimizer.load_state_dict(restored.optimizer_state)
        best_score = float(state["best_validation_protocol_score"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        start_epoch = int(state["last_epoch"]) + 1
        if len(curve) != int(state["last_epoch"]):
            raise ValueError("R5 resume curve length does not match last epoch")

    epoch_limit = protocol.max_epochs if smoke_epochs is None else int(smoke_epochs)
    if epoch_limit <= 0 or epoch_limit > protocol.max_epochs:
        raise ValueError("smoke_epochs must be between 1 and the formal epoch budget")
    for epoch in range(start_epoch, epoch_limit + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        windows = train_schedule[epoch - 1]
        micro_batches = list(
            _window_batches(
                windows,
                normalization_stats,
                micro_batch_size=micro_batch_size,
            )
        )
        for cpu_batch in micro_batches:
            batch = move_explicit_batch(cpu_batch, device)
            output = model(batch, rollout_steps=3)
            objective = compute_r4_objective(output, batch)
            (objective.total / len(micro_batches)).backward()
            epoch_loss += float(objective.total.detach().cpu().item())
            del objective, output, batch, cpu_batch
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM).item()
        )
        if not math.isfinite(gradient_norm):
            raise ValueError("R5 gradient norm is NaN or Inf")
        optimizer.step()
        validation = _validate_candidate(
            model,
            validation_windows,
            normalization_stats,
            selection_scales,
            device=device,
            micro_batch_size=micro_batch_size,
            validation_seed=spec.training_seed + 991,
        )
        score = validation.get("validation_protocol_score")
        if score is None or not math.isfinite(float(score)):
            raise ValueError("R5 validation common metric is missing")
        improved = float(score) < best_score - protocol.minimum_improvement
        if improved:
            best_score = float(score)
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_saver(
                checkpoint_path,
                model,
                optimizer,
                bindings,
                protocol,
                learning_rate=LEARNING_RATE,
                seed=spec.training_seed,
            )
        else:
            stale_epochs += 1
        checkpoint_saver(
            last_checkpoint_path,
            model,
            optimizer,
            bindings,
            protocol,
            learning_rate=LEARNING_RATE,
            seed=spec.training_seed,
        )
        row = {
            "epoch": epoch,
            "train_objective_mean": epoch_loss / len(micro_batches),
            "gradient_norm": gradient_norm,
            "validation": validation,
            "improved": improved,
            "best_epoch": best_epoch,
            "best_validation_protocol_score": best_score,
            "stale_epochs": stale_epochs,
        }
        curve.append(row)
        _write_json_atomic(curve_path, curve)
        _write_json_atomic(
            state_path,
            {
                "schema_version": run_state_schema,
                "combination_id": spec.combination_id,
                "training_seed": spec.training_seed,
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_protocol_score": best_score,
                "stale_epochs": stale_epochs,
                "last_checkpoint_sha256": _sha256(last_checkpoint_path),
                "input_fingerprint": input_fingerprint,
            },
        )
        print(
            json.dumps(
                {
                    "event": "r5_epoch",
                    "run_id": spec.run_id,
                    "epoch": epoch,
                    "score": score,
                    "best_score": best_score,
                    "stale_epochs": stale_epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stale_epochs >= protocol.patience:
            break

    if not checkpoint_path.is_file():
        raise ValueError("R5 run has no best checkpoint")
    restored = checkpoint_loader(
        checkpoint_path,
        expected_bindings=bindings,
        expected_protocol=protocol,
    )
    best_model = restored.model.to(device)
    final_validation = _validate_candidate(
        best_model,
        validation_windows,
        normalization_stats,
        selection_scales,
        device=device,
        micro_batch_size=micro_batch_size,
        validation_seed=spec.training_seed + 991,
    )
    if not _score_reproduced(
        best_score,
        float(final_validation["validation_protocol_score"]),
        protocol.minimum_improvement,
    ):
        raise ValueError("R5 restored best checkpoint does not reproduce validation score")
    calibration = _calibration_report(
        best_model,
        calibration_windows,
        normalization_stats,
        selection_scales,
        device=device,
        micro_batch_size=micro_batch_size,
        validation_seed=spec.training_seed + 1991,
    )
    report = {
        "schema_version": report_schema,
        "run_id": spec.run_id,
        "combination_id": spec.combination_id,
        "training_seed": spec.training_seed,
        "status": "completed",
        "config": asdict(config),
        "components": config.component_names(),
        "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
        "epochs_executed": len(curve),
        "best_epoch": best_epoch,
        "best_validation_protocol_score": best_score,
        "checkpoint_reproduction_score_delta": abs(
            float(final_validation["validation_protocol_score"]) - best_score
        ),
        "checkpoint_reproduction_tolerance": protocol.minimum_improvement,
        "final_validation": final_validation,
        "calibration": calibration,
        "checkpoint": checkpoint_path.relative_to(run_dir.parent.parent.parent).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - run_started,
        "locked_test_accessed": False,
    }
    _write_json_atomic(run_dir / "run_report.json", report)
    return report


def run_r5_gpu_training(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    r4_screening_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    hidden_dim: int = 16,
    micro_batch_size: int = 4,
    combination_ids: Sequence[str] = APPROVED_COMBINATIONS,
    training_seeds: Sequence[int] | None = None,
    resume: bool = False,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    cuda_device = require_cuda_device(device, output_dir)
    cuda_index = torch.cuda.current_device() if cuda_device.index is None else int(cuda_device.index)
    torch.cuda.set_device(cuda_index)
    validate_training_splits(("train", "validation", "calibration"))
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    r4_screening_root = Path(r4_screening_root).resolve()
    output_dir = Path(output_dir).resolve()
    protocol = load_r5_protocol(evaluation_root)
    specs = build_run_specs(
        protocol,
        combination_ids=combination_ids,
        training_seeds=training_seeds,
    )
    smoke = smoke_epochs is not None
    if smoke and (len(specs) != 1 or specs[0] != R5RunSpec("A", 20260803)):
        raise ValueError("R5 smoke must run only A with seed 20260803")
    if micro_batch_size <= 0 or protocol.effective_batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must divide effective batch size")
    if not (r4_screening_root / "manifest.json").is_file():
        raise FileNotFoundError("R4 screening manifest is missing")
    r4_summary = _read_json(r4_screening_root / "screening_summary.json")
    if not r4_summary.get("r4_gpu_screening_complete") or r4_summary.get("winner") != "graph_rssm_v1":
        raise ValueError("R4 screening binding is incomplete or has unexpected winner")
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization_stats = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    selection_scales = load_selection_scales(evaluation_root)
    rate_mean, rate_scale = _information_rate_stats(normalization_stats)
    matrix = r5_combination_matrix(
        hidden_dim=hidden_dim,
        history_steps=8,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )
    rows = read_trajectory_index(dataset_root)
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
        seed: build_validation_windows(
            dataset_root,
            split="validation",
            horizons=(1, 5, 20),
            seed=seed,
        )
        for seed in sorted({spec.training_seed for spec in specs})
    }
    calibration_windows = {
        seed: build_validation_windows(
            dataset_root,
            split="calibration",
            horizons=(1, 5, 20),
            seed=seed,
        )
        for seed in sorted({spec.training_seed for spec in specs})
    }
    tensor_inputs = {
        str(seed): {
            "training": _verify_tensor_inputs(dataset_root, schedules[seed], ()),
            "validation": _verify_tensor_inputs(dataset_root, (), validation_windows[seed]),
            "calibration": _verify_tensor_inputs(dataset_root, (), calibration_windows[seed]),
        }
        for seed in sorted(schedules)
    }
    source_digest, source_hashes = _source_binding()
    bindings = {
        **input_report["bindings"],
        "source_code_sha256": source_digest,
        "r4_screening_manifest_sha256": _sha256(r4_screening_root / "manifest.json"),
        "r5_protocol_sha256": _sha256(evaluation_root / "fair_experiment_protocol.json"),
    }
    if set(bindings) != set(R5_REQUIRED_BINDINGS):
        raise ValueError("R5 source bindings are incomplete")
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
        smoke=smoke,
        micro_batch_size=micro_batch_size,
    )
    _write_json_atomic(
        output_dir / "training_window_schedules.json",
        {
            str(seed): [
                {
                    "epoch": epoch,
                    "windows": [window.to_dict() for window in epoch_windows],
                }
                for epoch, epoch_windows in enumerate(schedule, start=1)
            ]
            for seed, schedule in schedules.items()
        },
    )
    _write_json_atomic(
        output_dir / "validation_windows.json",
        {
            str(seed): [window.to_dict() for window in windows]
            for seed, windows in validation_windows.items()
        },
    )
    _write_json_atomic(
        output_dir / "calibration_windows.json",
        {
            str(seed): [window.to_dict() for window in windows]
            for seed, windows in calibration_windows.items()
        },
    )
    reports, failures, history = _load_existing_reports(output_dir) if resume else ([], [], [])
    started = time.perf_counter()
    for spec in specs:
        run_dir = _run_dir(output_dir, spec)
        if resume and _completed_run_is_valid(
            run_dir,
            spec,
            bindings=bindings,
            protocol=protocol,
        ):
            completed_report = _read_json(run_dir / "run_report.json")
            reports[:] = [row for row in reports if row.get("run_id") != spec.run_id]
            reports.append(completed_report)
            failures[:] = [row for row in failures if row.get("run_id") != spec.run_id]
            print(json.dumps({"event": "r5_run_skipped", "run_id": spec.run_id}), flush=True)
            continue
        failures[:] = [row for row in failures if row.get("run_id") != spec.run_id]
        try:
            individual_started = time.perf_counter()
            report = _train_one_run(
                spec,
                run_dir=run_dir,
                protocol=protocol,
                config=matrix[spec.combination_id].config,
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
                run_started=individual_started,
            )
            reports[:] = [row for row in reports if row.get("run_id") != spec.run_id]
            reports.append(report)
            print(json.dumps({"event": "r5_run_complete", "run_id": spec.run_id}), flush=True)
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
            history.append(failure)
            _write_json_atomic(run_dir / "failure.json", failure)
            print(json.dumps({"event": "r5_run_failed", "run_id": spec.run_id, "error": str(error)}), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        _persist_progress(
            output_dir,
            reports,
            failures,
            history,
            specs=specs,
            protocol=protocol,
            started=started,
            smoke=smoke,
        )
    summary = _persist_progress(
        output_dir,
        reports,
        failures,
        history,
        specs=specs,
        protocol=protocol,
        started=started,
        smoke=smoke,
    )
    if summary["r5_gpu_training_complete"]:
        summary["combination_summary"] = summarize_completed_runs(reports, protocol)
        _write_json_atomic(output_dir / "combination_summary.json", summary["combination_summary"])
        _write_manifest(output_dir, summary)
    return summary


def load_resumable_run_state(
    run_dir: str | Path,
    *,
    combination_id: str,
    training_seed: int,
    input_fingerprint: str,
    expected_schema: str = RUN_STATE_SCHEMA,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state_path = run_dir / "run_state.json"
    checkpoint_path = run_dir / "last_checkpoint.pt"
    if not state_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("R5 resumable state or last checkpoint is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != expected_schema:
        raise ValueError("R5 resumable state schema is incompatible")
    if state.get("combination_id") != combination_id:
        raise ValueError("R5 resumable state combination mismatch")
    if int(state.get("training_seed", -1)) != int(training_seed):
        raise ValueError("R5 resumable state seed mismatch")
    if state.get("input_fingerprint") != input_fingerprint:
        raise ValueError("R5 resumable state input fingerprint mismatch")
    if state.get("last_checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("R5 resumable state checkpoint hash mismatch")
    for key in ("last_epoch", "best_epoch", "stale_epochs"):
        if int(state.get(key, -1)) < 0:
            raise ValueError(f"R5 resumable state {key} is invalid")
    score = float(state.get("best_validation_protocol_score", math.inf))
    if not math.isfinite(score):
        raise ValueError("R5 resumable state best score is invalid")
    return state


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("R5 summary values must be finite and non-empty")
    normalized = [float(value) for value in values]
    return {
        "mean": statistics.fmean(normalized),
        "std": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
    }


def summarize_completed_runs(
    reports: Sequence[Mapping[str, Any]],
    protocol: R5FormalProtocol,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        combination: [] for combination in APPROVED_COMBINATIONS
    }
    for report in reports:
        combination = str(report.get("combination_id"))
        if combination not in grouped:
            raise ValueError(f"unknown R5 completed combination: {combination}")
        grouped[combination].append(report)

    combinations: dict[str, Any] = {}
    for combination, rows in grouped.items():
        seeds = tuple(sorted(int(row["training_seed"]) for row in rows))
        if seeds != tuple(sorted(protocol.training_seeds)):
            raise ValueError(f"R5 combination {combination} has incomplete seed reports")
        metrics = {}
        for metric_id in REQUIRED_PUBLIC_METRIC_GATES:
            values = []
            for row in rows:
                metric = row["final_validation"]["metrics"][metric_id]
                if metric.get("status") != "computed":
                    raise ValueError(f"R5 metric is not computed: {metric_id}")
                values.append(float(metric["value"]))
            metrics[metric_id] = _mean_std(values)
        combinations[combination] = {
            "training_seed_count": len(rows),
            "training_seeds": list(seeds),
            "validation_protocol_score": _mean_std(
                [float(row["best_validation_protocol_score"]) for row in rows]
            ),
            "public_metrics": metrics,
            "runtime_seconds": _mean_std(
                [float(row["runtime_seconds"]) for row in rows]
            ),
            "peak_cuda_memory_bytes": {
                "max": max(int(row["peak_cuda_memory_bytes"]) for row in rows)
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_status": "descriptive_only",
        "combinations": combinations,
        "claim_boundary": (
            "multi-seed descriptive summary; no automatic final-method selection"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--r4-screening-root", required=True)
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
    summary = run_r5_gpu_training(
        args.dataset_root,
        args.evaluation_root,
        args.r4_screening_root,
        args.output_dir,
        device=args.device,
        hidden_dim=args.hidden_dim,
        micro_batch_size=args.micro_batch_size,
        combination_ids=(
            APPROVED_COMBINATIONS if args.combinations is None else args.combinations
        ),
        training_seeds=args.seeds,
        resume=args.resume,
        smoke_epochs=args.smoke_epochs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
