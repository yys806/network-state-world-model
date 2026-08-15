"""Run the frozen-budget PI-JWM R4 single-module screening on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r3_preflight_data import (
    R3Window,
    load_r3_window,
    make_explicit_batch,
    verify_r3_inputs,
)
from pi_jwm.r4_checkpoint import (
    R4TrainingBudget,
    load_r4_checkpoint,
    save_r4_checkpoint,
)
from pi_jwm.r4_gpu_screening import (
    R4ValidationAccumulator,
    build_training_window_schedule,
    build_validation_windows,
    collate_explicit_batches,
    load_frozen_screening_protocol,
    move_explicit_batch,
)
from pi_jwm.r4_module_registry import (
    candidate_matrix,
    candidate_registry,
    make_single_module_config,
    reference_r4_config,
)
from pi_jwm.r4_objective import compute_r4_objective
from pi_jwm.r4_world_model import build_r4_world_model
from pi_jwm.teacher_evaluation_v3 import SELECTION_COMPONENTS


SCHEMA_VERSION = "PIJWM-R4-Formal-GPU-Screening-v1"
LEARNING_RATE = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
R4_SOURCE_FILES = (
    SRC_ROOT / "pi_jwm" / "r3_preflight_data.py",
    SRC_ROOT / "pi_jwm" / "r3_world_model.py",
    SRC_ROOT / "pi_jwm" / "r3_objective.py",
    SRC_ROOT / "pi_jwm" / "r3_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_module_registry.py",
    SRC_ROOT / "pi_jwm" / "r4_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_objective.py",
    SRC_ROOT / "pi_jwm" / "r4_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_gpu_screening.py",
    Path(__file__),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_binding() -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in R4_SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"R4 source binding is missing: {path}")
        relative = path.relative_to(CODE_ROOT).as_posix()
        value = _sha256(path)
        hashes[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest(), hashes


def load_selection_scales(evaluation_root: str | Path) -> dict[str, float]:
    payload = _read_json(Path(evaluation_root) / "checkpoint_selection_scales.json")
    if payload.get("source_split") != "train":
        raise ValueError("checkpoint selection scales are not train-only")
    scales = {key: float(value) for key, value in payload.get("scales", {}).items()}
    if set(scales) != set(SELECTION_COMPONENTS) or any(
        not math.isfinite(value) or value <= 0.0 for value in scales.values()
    ):
        raise ValueError("checkpoint selection scales are incomplete or invalid")
    return scales


def _information_rate_stats(
    normalization_stats: Mapping[str, Any],
) -> tuple[float, float]:
    values = normalization_stats["features"]["information_edge_state"]
    mean, scale = float(values["mean"][12]), float(values["scale"][12])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("information-rate normalization is invalid")
    return mean, scale


def executable_candidate_configs(
    *,
    hidden_dim: int,
    history_steps: int,
    information_rate_mean: float,
    information_rate_scale: float,
) -> dict[str, Any]:
    configs = {
        "reference": reference_r4_config(
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        )
    }
    for name, spec in candidate_registry().items():
        if spec.status == "executable":
            configs[name] = make_single_module_config(
                spec.family,
                name,
                hidden_dim=hidden_dim,
                history_steps=history_steps,
                information_rate_mean=information_rate_mean,
                information_rate_scale=information_rate_scale,
            )
    return configs


def require_cuda_device(device: str, output_dir: str | Path) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal R4 screening requires an available CUDA device")
    return requested


def checkpoint_score_reproduced(
    expected: float, actual: float, tolerance: float
) -> bool:
    if tolerance <= 0.0:
        raise ValueError("checkpoint score tolerance must be positive")
    return (
        math.isfinite(float(expected))
        and math.isfinite(float(actual))
        and abs(float(actual) - float(expected)) <= float(tolerance)
    )


def cuda_device_index(device: str | torch.device) -> int:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("CUDA statistics require a CUDA device")
    return torch.cuda.current_device() if requested.index is None else int(requested.index)


def _window_batches(
    windows: Sequence[R3Window],
    normalization_stats: Mapping[str, Any],
    *,
    micro_batch_size: int,
):
    for start in range(0, len(windows), micro_batch_size):
        selected = windows[start : start + micro_batch_size]
        batches = [
            make_explicit_batch(load_r3_window(window), normalization_stats)
            for window in selected
        ]
        yield collate_explicit_batches(batches)


def _validate_candidate(
    model: torch.nn.Module,
    windows: Sequence[R3Window],
    normalization_stats: Mapping[str, Any],
    selection_scales: Mapping[str, float],
    *,
    device: torch.device,
    micro_batch_size: int,
    validation_seed: int,
) -> dict[str, Any]:
    model.eval()
    torch.manual_seed(validation_seed)
    torch.cuda.manual_seed_all(validation_seed)
    accumulator = R4ValidationAccumulator(
        normalization_stats,
        selection_scales=selection_scales,
    )
    started = time.perf_counter()
    with torch.no_grad():
        for horizon in sorted({window.horizon_steps for window in windows}):
            horizon_windows = [window for window in windows if window.horizon_steps == horizon]
            for cpu_batch in _window_batches(
                horizon_windows,
                normalization_stats,
                micro_batch_size=micro_batch_size,
            ):
                batch = move_explicit_batch(cpu_batch, device)
                output = model(batch, rollout_steps=horizon)
                accumulator.update(output, batch)
                del output, batch, cpu_batch
    torch.cuda.synchronize(device)
    report = accumulator.finalize()
    report["runtime_seconds"] = time.perf_counter() - started
    report["mean_window_latency_ms"] = (
        1000.0 * report["runtime_seconds"] / max(report["window_count"], 1)
    )
    return report


def _verify_tensor_inputs(
    dataset_root: Path,
    schedules: Sequence[Sequence[R3Window]],
    validation_windows: Sequence[R3Window],
) -> dict[str, Any]:
    manifest = _read_json(dataset_root / "manifest.json")
    paths = {
        window.tensor_path.resolve()
        for schedule in schedules
        for window in schedule
    } | {window.tensor_path.resolve() for window in validation_windows}
    reports = {}
    for path in sorted(paths):
        relative = path.relative_to(dataset_root).as_posix()
        entry = manifest.get("files", {}).get(relative)
        actual = _sha256(path)
        if (
            not isinstance(entry, Mapping)
            or entry.get("sha256") != actual
            or int(entry.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"R4 tensor input is not bound by the R1 manifest: {relative}")
        reports[relative] = {
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "verified": True,
        }
    return reports


def run_r4_gpu_screening(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    hidden_dim: int = 16,
    micro_batch_size: int = 4,
    candidate_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    cuda_device = require_cuda_device(device, output_dir)
    cuda_index = cuda_device_index(cuda_device)
    torch.cuda.set_device(cuda_index)
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"R4 output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_frozen_screening_protocol(evaluation_root)
    if micro_batch_size <= 0 or protocol.effective_batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must divide the frozen effective batch size")
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization_stats = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    selection_scales = load_selection_scales(evaluation_root)
    rate_mean, rate_scale = _information_rate_stats(normalization_stats)
    configs = executable_candidate_configs(
        hidden_dim=hidden_dim,
        history_steps=8,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )
    names = list(configs) if candidate_names is None else list(candidate_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("candidate_names must be non-empty and unique")
    unknown = sorted(set(names) - set(configs))
    if unknown:
        raise ValueError("unknown or non-executable R4 candidates: " + ", ".join(unknown))

    schedules = build_training_window_schedule(
        dataset_root,
        epochs=protocol.max_epochs,
        windows_per_epoch=protocol.effective_batch_size,
        seed=protocol.training_seed,
    )
    validation_windows = build_validation_windows(
        dataset_root,
        split="validation",
        horizons=(1, 5, 20),
        seed=protocol.training_seed,
    )
    tensor_inputs = _verify_tensor_inputs(dataset_root, schedules, validation_windows)
    source_digest, source_hashes = _source_binding()
    bindings = {**input_report["bindings"], "source_code_sha256": source_digest}
    budget = R4TrainingBudget(
        epochs=protocol.max_epochs,
        patience=protocol.patience,
        learning_rate=LEARNING_RATE,
        training_seed=protocol.training_seed,
    )
    _write_json(
        output_dir / "screening_protocol.json",
        {
            "schema_version": SCHEMA_VERSION,
            "budget": asdict(budget),
            "effective_batch_size": protocol.effective_batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": protocol.effective_batch_size // micro_batch_size,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "validation_horizons": [1, 5, 20],
            "validation_trajectory_count": 12,
            "locked_test_accessed": False,
            "selection_metric": "validation_protocol_score",
        },
    )
    _write_json(
        output_dir / "training_window_schedule.json",
        {
            "schema_version": SCHEMA_VERSION,
            "epochs": [
                {"epoch": index + 1, "windows": [window.to_dict() for window in windows]}
                for index, windows in enumerate(schedules)
            ],
        },
    )
    _write_json(
        output_dir / "validation_windows.json",
        [window.to_dict() for window in validation_windows],
    )
    _write_json(output_dir / "candidate_matrix.json", candidate_matrix())
    _write_json(
        output_dir / "input_provenance.json",
        {
            "bindings": bindings,
            "source_files": source_hashes,
            "tensor_inputs": tensor_inputs,
            "locked_test_accessed": False,
        },
    )

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for candidate_index, name in enumerate(names):
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = candidate_dir / "best_checkpoint.pt"
        curve: list[dict[str, Any]] = []
        candidate_started = time.perf_counter()
        print(json.dumps({"event": "candidate_start", "candidate": name, "index": candidate_index + 1, "total": len(names)}), flush=True)
        try:
            random.seed(protocol.training_seed)
            np.random.seed(protocol.training_seed)
            torch.manual_seed(protocol.training_seed)
            torch.cuda.manual_seed_all(protocol.training_seed)
            torch.cuda.reset_peak_memory_stats()
            model = build_r4_world_model(configs[name]).to(cuda_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            best_score = math.inf
            best_epoch = 0
            stale_epochs = 0
            for epoch, windows in enumerate(schedules, start=1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss = 0.0
                micro_batches = list(
                    _window_batches(
                        windows,
                        normalization_stats,
                        micro_batch_size=micro_batch_size,
                    )
                )
                for cpu_batch in micro_batches:
                    batch = move_explicit_batch(cpu_batch, cuda_device)
                    output = model(batch, rollout_steps=3)
                    objective = compute_r4_objective(output, batch)
                    (objective.total / len(micro_batches)).backward()
                    epoch_loss += float(objective.total.detach().cpu().item())
                    del objective, output, batch, cpu_batch
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM).item()
                )
                if not math.isfinite(gradient_norm):
                    raise ValueError("R4 gradient norm is NaN or Inf")
                optimizer.step()

                validation = _validate_candidate(
                    model,
                    validation_windows,
                    normalization_stats,
                    selection_scales,
                    device=cuda_device,
                    micro_batch_size=micro_batch_size,
                    validation_seed=protocol.training_seed + 991,
                )
                score = validation["validation_protocol_score"]
                if score is None or not math.isfinite(float(score)):
                    raise ValueError("candidate is ineligible because a common validation term is missing")
                improved = float(score) < best_score - protocol.minimum_improvement
                if improved:
                    best_score = float(score)
                    best_epoch = epoch
                    stale_epochs = 0
                    save_r4_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        bindings,
                        budget,
                        seed=protocol.training_seed,
                    )
                else:
                    stale_epochs += 1
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
                _write_json(candidate_dir / "training_curve.json", curve)
                _write_json(
                    output_dir / "live_status.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "candidate": name,
                        "candidate_index": candidate_index + 1,
                        "candidate_count": len(names),
                        "epoch": epoch,
                        "max_epochs": protocol.max_epochs,
                        "best_epoch": best_epoch,
                        "best_validation_protocol_score": best_score,
                        "elapsed_seconds": time.perf_counter() - run_started,
                        "locked_test_accessed": False,
                    },
                )
                print(json.dumps({"event": "epoch", "candidate": name, "epoch": epoch, "score": score, "best_score": best_score, "stale": stale_epochs}), flush=True)
                if stale_epochs >= protocol.patience:
                    break

            restored = load_r4_checkpoint(checkpoint_path, expected_bindings=bindings)
            best_model = restored.model.to(cuda_device)
            final_validation = _validate_candidate(
                best_model,
                validation_windows,
                normalization_stats,
                selection_scales,
                device=cuda_device,
                micro_batch_size=micro_batch_size,
                validation_seed=protocol.training_seed + 991,
            )
            if not checkpoint_score_reproduced(
                best_score,
                float(final_validation["validation_protocol_score"]),
                protocol.minimum_improvement,
            ):
                raise ValueError("restored best checkpoint does not reproduce its validation score")
            report = {
                "candidate": name,
                "candidate_index": candidate_index,
                "status": "completed",
                "config": asdict(configs[name]),
                "components": configs[name].component_names(),
                "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
                "epochs_executed": len(curve),
                "optimizer_steps": len(curve),
                "best_epoch": best_epoch,
                "best_validation_protocol_score": best_score,
                "checkpoint_reproduction_score_delta": abs(
                    float(final_validation["validation_protocol_score"]) - best_score
                ),
                "checkpoint_reproduction_tolerance": protocol.minimum_improvement,
                "final_validation": final_validation,
                "checkpoint": checkpoint_path.relative_to(output_dir).as_posix(),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "runtime_seconds": time.perf_counter() - candidate_started,
                "locked_test_accessed": False,
            }
            reports.append(report)
            _write_json(candidate_dir / "candidate_report.json", report)
            print(json.dumps({"event": "candidate_complete", "candidate": name, "score": best_score, "epochs": len(curve)}), flush=True)
            del model, best_model, optimizer, restored
        except Exception as error:
            failure = {
                "candidate": name,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - candidate_started,
                "locked_test_accessed": False,
            }
            failures.append(failure)
            _write_json(candidate_dir / "failure.json", failure)
            print(json.dumps({"event": "candidate_failed", "candidate": name, "error": str(error)}), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    ranked = sorted(
        reports,
        key=lambda row: (
            float(row["best_validation_protocol_score"]),
            float(row["final_validation"]["mean_window_latency_ms"]),
            str(row["candidate"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    completed_matrix = len(names) == len(configs) and not failures and len(reports) == len(configs)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r4_gpu_screening_complete": completed_matrix,
        "candidate_count": len(names),
        "completed_candidate_count": len(reports),
        "failed_candidate_count": len(failures),
        "winner": ranked[0]["candidate"] if completed_matrix and ranked else None,
        "winner_validation_protocol_score": ranked[0]["best_validation_protocol_score"] if completed_matrix and ranked else None,
        "training_seed_count": 1,
        "training_seed": protocol.training_seed,
        "locked_test_accessed": False,
        "gpu": {
            "name": torch.cuda.get_device_name(cuda_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "runtime_seconds": time.perf_counter() - run_started,
        "claim_boundary": "single-seed module screening; formal multi-seed comparison remains R6/R7",
    }
    _write_json(output_dir / "candidate_reports.json", reports)
    _write_json(output_dir / "ranked_candidates.json", ranked)
    _write_json(output_dir / "failed_candidates.json", failures)
    _write_json(output_dir / "screening_summary.json", summary)
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp"):
            relative = path.relative_to(output_dir).as_posix()
            files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        **summary,
        "files": files,
        "manifest_entry_count": len(files),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "screening_complete", **summary}), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--candidate", action="append", dest="candidates")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_r4_gpu_screening(
        args.dataset_root,
        args.evaluation_root,
        args.output_dir,
        device=args.device,
        hidden_dim=args.hidden_dim,
        micro_batch_size=args.micro_batch_size,
        candidate_names=args.candidates,
    )


if __name__ == "__main__":
    main()
