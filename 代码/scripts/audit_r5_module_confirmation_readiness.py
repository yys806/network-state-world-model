"""Audit the frozen PI-JWM R5.1 matrix before any formal GPU run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r3_preflight_data import (  # noqa: E402
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
    verify_r3_inputs,
)
from pi_jwm.r4_gpu_screening import (  # noqa: E402
    build_training_window_schedule,
    build_validation_windows,
)
from pi_jwm.r4_objective import compute_r4_objective  # noqa: E402
from pi_jwm.r5_module_confirmation import (  # noqa: E402
    TRAINING_SEEDS,
    build_confirmation_matrix,
    build_confirmation_model,
    build_confirmation_run_specs,
)
from pi_jwm.r5_protocol import load_r5_protocol  # noqa: E402
from run_r4_gpu_screening import _information_rate_stats  # noqa: E402
from run_r5_gpu_training import _read_json, _sha256, _write_json_atomic  # noqa: E402
from run_r5_module_confirmation_training import (  # noqa: E402
    validate_reused_b_results,
    validate_reused_window_schedules,
)


SCHEMA_VERSION = "PIJWM-R5-Module-Confirmation-Readiness-Audit-v1"
CORE_B_SOURCE_FILES = (
    "src/pi_jwm/r3_preflight_data.py",
    "src/pi_jwm/r3_world_model.py",
    "src/pi_jwm/r4_gpu_screening.py",
    "src/pi_jwm/r4_objective.py",
    "src/pi_jwm/r4_world_model.py",
    "src/pi_jwm/r5_protocol.py",
    "src/pi_jwm/r5_world_model.py",
    "scripts/run_r4_gpu_screening.py",
)


def _verify_frozen_bundle(
    root: Path,
    *,
    existing_manifest_sha256: str,
    rate_mean: float,
    rate_scale: float,
) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("existing_r5_manifest_sha256") != existing_manifest_sha256:
        raise ValueError("frozen matrix existing R5 manifest binding mismatch")
    files = manifest.get("files", {})
    if set(files) != {"README.md", "matrix.json", "run_specs.json", "summary.json"}:
        raise ValueError("frozen matrix manifest file set mismatch")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen matrix file is missing: {relative}")
        if path.stat().st_size != int(expected.get("size_bytes", -1)):
            raise ValueError(f"frozen matrix size mismatch: {relative}")
        if _sha256(path) != expected.get("sha256"):
            raise ValueError(f"frozen matrix hash mismatch: {relative}")
    runtime_matrix = {
        key: asdict(candidate)
        for key, candidate in build_confirmation_matrix(
            information_rate_mean=rate_mean,
            information_rate_scale=rate_scale,
        ).items()
    }
    if _read_json(root / "matrix.json") != runtime_matrix:
        raise ValueError("frozen matrix differs from the executable runtime matrix")
    runtime_specs = [asdict(spec) for spec in build_confirmation_run_specs()]
    if _read_json(root / "run_specs.json") != runtime_specs:
        raise ValueError("frozen run specifications differ from executable specifications")
    summary = _read_json(root / "summary.json")
    expected_counts = {
        "candidate_count": 5,
        "training_seed_count": 3,
        "total_run_count": 15,
        "reused_run_count": 3,
        "new_gpu_run_count": 12,
        "locked_test_accessed": False,
    }
    for key, value in expected_counts.items():
        if summary.get(key) != value:
            raise ValueError(f"frozen matrix summary mismatch: {key}")
    return {
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(files),
        "runtime_matrix_exact": True,
        "runtime_run_specs_exact": True,
    }


def _verify_core_b_sources(existing_r5_root: Path) -> dict[str, str]:
    provenance = _read_json(existing_r5_root / "input_provenance.json")
    stored = provenance.get("r5_source_files", {})
    verified: dict[str, str] = {}
    for relative in CORE_B_SOURCE_FILES:
        path = CODE_ROOT / relative
        current = _sha256(path)
        if stored.get(relative) != current:
            raise ValueError(f"core B source hash mismatch: {relative}")
        verified[relative] = current
    return verified


def _execute_model(
    combination_id: str,
    *,
    batch: Any,
    hidden_dim: int,
    rate_mean: float,
    rate_scale: float,
) -> dict[str, Any]:
    torch.manual_seed(20260806)
    model = build_confirmation_model(
        combination_id,
        hidden_dim=hidden_dim,
        history_steps=8,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    horizon = int(batch.metadata["horizon_steps"])
    output = model(batch, rollout_steps=horizon)
    objective = compute_r4_objective(output, batch)
    if not math.isfinite(float(objective.total.detach().cpu().item())):
        raise ValueError(f"{combination_id} objective is non-finite")
    objective.total.backward()
    output_tensors = [*output.predicted_explicit.values(), *output.predicted_logits.values()]
    if not output_tensors or not all(bool(torch.isfinite(value).all()) for value in output_tensors):
        raise ValueError(f"{combination_id} produced non-finite rollout tensors")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError(f"{combination_id} produced missing or non-finite gradients")
    if not any(bool(torch.count_nonzero(value)) for value in gradients):
        raise ValueError(f"{combination_id} produced only zero gradients")
    return {
        "combination_id": combination_id,
        "hidden_dim": hidden_dim,
        "rollout_horizon": horizon,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "objective_total": float(objective.total.detach().cpu().item()),
        "finite_rollout": True,
        "finite_nonzero_gradients": True,
        "components": model.component_registry(),
    }


def audit_readiness(
    *,
    dataset_root: str | Path,
    evaluation_root: str | Path,
    r4_screening_root: str | Path,
    existing_r5_root: str | Path,
    frozen_bundle_root: str | Path,
    output_dir: str | Path,
    hidden_dim: int = 16,
    combination_ids: Sequence[str] = ("F", "G", "H", "J"),
) -> dict[str, Any]:
    roots = [
        Path(value).resolve()
        for value in (
            dataset_root,
            evaluation_root,
            r4_screening_root,
            existing_r5_root,
            frozen_bundle_root,
        )
    ]
    dataset_root, evaluation_root, r4_screening_root, existing_r5_root, frozen_bundle_root = roots
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"readiness audit output directory must be empty: {output_dir}")
    combination_ids = tuple(str(value) for value in combination_ids)
    if not combination_ids or len(set(combination_ids)) != len(combination_ids):
        raise ValueError("audit combination list must be non-empty and unique")
    if not set(combination_ids).issubset({"F", "G", "H", "J"}):
        raise ValueError("audit accepts only F/G/H/J")

    torch.set_num_threads(1)
    protocol = load_r5_protocol(evaluation_root)
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    rate_mean, rate_scale = _information_rate_stats(normalization)
    r4_summary = _read_json(r4_screening_root / "screening_summary.json")
    if not r4_summary.get("r4_gpu_screening_complete") or r4_summary.get("winner") != "graph_rssm_v1":
        raise ValueError("R4 Graph-RSSM winner binding is invalid")
    existing_manifest = existing_r5_root / "manifest.json"
    frozen_report = _verify_frozen_bundle(
        frozen_bundle_root,
        existing_manifest_sha256=_sha256(existing_manifest),
        rate_mean=rate_mean,
        rate_scale=rate_scale,
    )
    core_sources = _verify_core_b_sources(existing_r5_root)
    reused = validate_reused_b_results(
        existing_r5_root,
        protocol,
        expected_bindings={
            **input_report["bindings"],
            "r4_screening_manifest_sha256": _sha256(r4_screening_root / "manifest.json"),
            "r5_protocol_sha256": _sha256(evaluation_root / "fair_experiment_protocol.json"),
        },
    )
    schedules = {
        seed: build_training_window_schedule(
            dataset_root,
            epochs=protocol.max_epochs,
            windows_per_epoch=protocol.effective_batch_size,
            seed=seed,
        )
        for seed in TRAINING_SEEDS
    }
    validation = {
        seed: build_validation_windows(
            dataset_root,
            split="validation",
            horizons=(1, 5, 20),
            seed=seed,
        )
        for seed in TRAINING_SEEDS
    }
    calibration = {
        seed: build_validation_windows(
            dataset_root,
            split="calibration",
            horizons=(1, 5, 20),
            seed=seed,
        )
        for seed in TRAINING_SEEDS
    }
    validate_reused_window_schedules(existing_r5_root, schedules, validation, calibration)

    windows = select_r3_windows(
        dataset_root,
        read_trajectory_index(dataset_root),
        split="train",
        horizons=(20,),
        history_steps=8,
        per_horizon=1,
        seed=20260806,
    )
    if len(windows) != 1:
        raise ValueError("readiness audit could not select exactly one train horizon-20 window")
    batch = make_explicit_batch(load_r3_window(windows[0]), normalization)
    model_reports = [
        _execute_model(
            combination_id,
            batch=batch,
            hidden_dim=hidden_dim,
            rate_mean=rate_mean,
            rate_scale=rate_scale,
        )
        for combination_id in combination_ids
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "ready_for_gpu_smoke": True,
        "locked_test_accessed": False,
        "claim_boundary": "CPU readiness evidence only; no convergence or superiority claim",
        "formal_hidden_dim": hidden_dim,
        "executed_model_count": len(model_reports),
        "executed_combinations": list(combination_ids),
        "real_window": windows[0].to_dict(),
        "information_rate_normalization": {"mean": rate_mean, "scale": rate_scale},
        "frozen_bundle": frozen_report,
        "verified_core_b_sources": core_sources,
        "verified_reused_b_seeds": [int(value["training_seed"]) for value in reused],
        "verified_schedule": {
            "training_seed_count": len(schedules),
            "epochs_per_seed": protocol.max_epochs,
            "windows_per_epoch": protocol.effective_batch_size,
            "validation_windows_per_seed": len(next(iter(validation.values()))),
            "calibration_windows_per_seed": len(next(iter(calibration.values()))),
            "exactly_matches_reused_B": True,
        },
        "model_execution": model_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "audit_report.json", report)
    files = {
        "audit_report.json": {
            "sha256": _sha256(output_dir / "audit_report.json"),
            "size_bytes": (output_dir / "audit_report.json").stat().st_size,
        }
    }
    _write_json_atomic(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "ready_for_gpu_smoke": True,
            "locked_test_accessed": False,
            "files": files,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--r4-screening-root", type=Path, required=True)
    parser.add_argument("--existing-r5-root", type=Path, required=True)
    parser.add_argument("--frozen-bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()
    report = audit_readiness(
        dataset_root=args.dataset_root,
        evaluation_root=args.evaluation_root,
        r4_screening_root=args.r4_screening_root,
        existing_r5_root=args.existing_r5_root,
        frozen_bundle_root=args.frozen_bundle_root,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
