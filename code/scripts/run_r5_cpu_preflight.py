"""Run the nonlocked PI-JWM R5 combination CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
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
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
    verify_r3_inputs,
)
from pi_jwm.r4_gpu_screening import R4ValidationAccumulator
from pi_jwm.r4_objective import compute_r4_objective
from pi_jwm.r5_checkpoint import load_r5_checkpoint, save_r5_checkpoint
from pi_jwm.r5_protocol import (
    REQUIRED_PUBLIC_METRIC_GATES,
    load_r5_protocol,
    r5_combination_matrix,
    validate_r5_splits,
)
from pi_jwm.r5_world_model import build_r5_world_model
from run_r4_cpu_preflight import (
    _finite_output,
    _information_rate_stats,
    _max_output_delta,
    _mutate_future_targets,
    _outputs_equal,
    _read_json,
    _selected_input_provenance,
    _sha256,
    _write_json,
    _zero_task_action,
)
from run_r4_gpu_screening import load_selection_scales


SCHEMA_VERSION = "PIJWM-R5-CPU-Combination-Preflight-v1"
LEARNING_RATE = 1.0e-4
R5_SOURCE_FILES = (
    SRC_ROOT / "pi_jwm" / "r3_preflight_data.py",
    SRC_ROOT / "pi_jwm" / "r3_world_model.py",
    SRC_ROOT / "pi_jwm" / "r3_objective.py",
    SRC_ROOT / "pi_jwm" / "r3_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_module_registry.py",
    SRC_ROOT / "pi_jwm" / "r4_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_objective.py",
    SRC_ROOT / "pi_jwm" / "r4_gpu_screening.py",
    SRC_ROOT / "pi_jwm" / "r5_protocol.py",
    SRC_ROOT / "pi_jwm" / "r5_world_model.py",
    SRC_ROOT / "pi_jwm" / "r5_checkpoint.py",
    Path(__file__),
)
MODULE_MARKERS = {
    "A": (),
    "B": ("rssm",),
    "C": ("rssm", "heteroscedastic"),
    "D": ("rssm", "dag_"),
    "E": ("rssm", "presence_heads"),
}


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


def _module_gradient_evidence(
    model: torch.nn.Module,
    combination_id: str,
) -> dict[str, bool]:
    evidence: dict[str, bool] = {}
    for marker in MODULE_MARKERS[combination_id]:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if marker in name and parameter.grad is not None
        ]
        evidence["heteroscedastic" if marker == "heteroscedastic" else marker.rstrip("_")] = (
            bool(gradients)
            and all(bool(torch.isfinite(value).all().item()) for value in gradients)
            and any(bool(torch.count_nonzero(value).item()) for value in gradients)
        )
    return evidence


def _all_public_metrics_finite(report: Mapping[str, Any]) -> bool:
    metrics = report.get("metrics", {})
    return all(
        metric_id in metrics
        and metrics[metric_id].get("status") == "computed"
        and metrics[metric_id].get("value") is not None
        and math.isfinite(float(metrics[metric_id]["value"]))
        for metric_id in REQUIRED_PUBLIC_METRIC_GATES
    )


def _protocol_payload(protocol: Any) -> dict[str, Any]:
    return {
        **protocol.to_dict(),
        "learning_rate": LEARNING_RATE,
        "public_metric_gates": list(REQUIRED_PUBLIC_METRIC_GATES),
        "selection_rule": (
            "report every gate independently; do not freeze a method from the "
            "composite score alone"
        ),
        "locked_test_accessed": False,
    }


def run_r5_cpu_preflight(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    r4_screening_root: str | Path,
    output_dir: str | Path,
    *,
    per_horizon: int = 1,
    splits: Sequence[str] = ("train", "validation", "calibration"),
    horizons: Sequence[int] = (1, 5, 20),
    hidden_dim: int = 4,
    history_steps: int = 8,
    combination_ids: Sequence[str] = ("A", "B", "C", "D", "E"),
    execution_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Exercise R5 contracts on CPU without selecting a winning combination."""

    normalized_splits = validate_r5_splits(splits)
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    r4_screening_root = Path(r4_screening_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"R5 output directory must be empty: {output_dir}")

    protocol = load_r5_protocol(evaluation_root)
    seeds = (
        protocol.training_seeds
        if execution_seeds is None
        else tuple(int(seed) for seed in execution_seeds)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("execution_seeds must be non-empty and unique")
    if not set(seeds).issubset(protocol.training_seeds):
        raise ValueError("execution seed is outside the frozen R5 protocol")

    combination_ids = tuple(str(value) for value in combination_ids)
    if not combination_ids or len(set(combination_ids)) != len(combination_ids):
        raise ValueError("combination_ids must be non-empty and unique")
    approved = {"A", "B", "C", "D", "E"}
    unknown = sorted(set(combination_ids) - approved)
    if unknown:
        raise ValueError("unknown R5 combinations: " + ", ".join(unknown))

    r4_summary = _read_json(r4_screening_root / "screening_summary.json")
    if not r4_summary.get("r4_gpu_screening_complete"):
        raise ValueError("R4 screening artifact is incomplete")
    if r4_summary.get("winner") != "graph_rssm_v1":
        raise ValueError("R4 screening winner binding is not graph_rssm_v1")
    r4_manifest_path = r4_screening_root / "manifest.json"
    if not r4_manifest_path.is_file():
        raise FileNotFoundError("R4 screening manifest is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization_stats = _read_json(
        evaluation_root / "evaluation_normalization_stats.json"
    )
    selection_scales = load_selection_scales(evaluation_root)
    rate_mean, rate_scale = _information_rate_stats(normalization_stats)
    matrix = r5_combination_matrix(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )

    rows = read_trajectory_index(dataset_root)
    windows = [
        window
        for split in normalized_splits
        for window in select_r3_windows(
            dataset_root,
            rows,
            split=split,
            horizons=horizons,
            history_steps=history_steps,
            per_horizon=per_horizon,
            seed=20260805,
        )
    ]
    batches = [
        make_explicit_batch(load_r3_window(window), normalization_stats)
        for window in windows
    ]
    train_batches = [batch for batch in batches if batch.metadata["split"] == "train"]
    validation_batches = [
        batch for batch in batches if batch.metadata["split"] == "validation"
    ]
    if not train_batches or not validation_batches:
        raise ValueError("R5 CPU preflight requires train and validation windows")
    training_batch = max(
        train_batches,
        key=lambda batch: int(batch.metadata["horizon_steps"]),
    )
    probe_batch = max(
        batches,
        key=lambda batch: int(torch.count_nonzero(batch.future_action["task_action"])),
    )

    source_digest, source_hashes = _source_binding()
    protocol_path = evaluation_root / "fair_experiment_protocol.json"
    bindings = {
        **input_report["bindings"],
        "source_code_sha256": source_digest,
        "r4_screening_manifest_sha256": _sha256(r4_manifest_path),
        "r5_protocol_sha256": _sha256(protocol_path),
    }
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, Any]] = []
    objective_reports: list[dict[str, Any]] = []
    gate_reports: list[dict[str, Any]] = []
    checkpoint_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for combination_id in combination_ids:
        for seed in seeds:
            started = time.perf_counter()
            run_id = f"{combination_id}__seed_{seed}"
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = build_r5_world_model(
                    combination_id,
                    hidden_dim=hidden_dim,
                    history_steps=history_steps,
                    information_rate_mean=rate_mean,
                    information_rate_scale=rate_scale,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                parameter_count = sum(value.numel() for value in model.parameters())

                train_horizon = int(training_batch.metadata["horizon_steps"])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_output = model(training_batch, rollout_steps=train_horizon)
                train_objective = compute_r4_objective(train_output, training_batch)
                train_objective.total.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                finite_gradients = bool(gradients) and all(
                    bool(torch.isfinite(value).all().item()) for value in gradients
                )
                nonzero_gradient = bool(gradients) and any(
                    bool(torch.count_nonzero(value).item()) for value in gradients
                )
                module_evidence = _module_gradient_evidence(model, combination_id)
                optimizer.step()

                finite_rollouts = True
                observed_horizons: set[int] = set()
                for batch in batches:
                    horizon = int(batch.metadata["horizon_steps"])
                    model.eval()
                    with torch.no_grad():
                        output = model(batch, rollout_steps=horizon)
                        objective = compute_r4_objective(output, batch)
                    finite_rollouts = finite_rollouts and _finite_output(output)
                    observed_horizons.add(horizon)
                    objective_reports.append(
                        {
                            "run_id": run_id,
                            "combination_id": combination_id,
                            "training_seed": seed,
                            **batch.metadata,
                            "total": float(objective.total.detach().cpu().item()),
                            "terms": {
                                key: asdict(value) for key, value in objective.terms.items()
                            },
                            "auxiliary_terms": {
                                key: asdict(value)
                                for key, value in objective.auxiliary_terms.items()
                            },
                        }
                    )

                accumulator = R4ValidationAccumulator(
                    normalization_stats,
                    selection_scales=selection_scales,
                )
                model.eval()
                with torch.no_grad():
                    for batch in validation_batches:
                        horizon = int(batch.metadata["horizon_steps"])
                        accumulator.update(
                            model(batch, rollout_steps=horizon),
                            batch,
                        )
                gate_report = accumulator.finalize()
                public_metrics_finite = _all_public_metrics_finite(gate_report)
                gate_reports.append(
                    {
                        "run_id": run_id,
                        "combination_id": combination_id,
                        "training_seed": seed,
                        "all_public_metrics_finite": public_metrics_finite,
                        **gate_report,
                    }
                )

                probe_horizon = int(probe_batch.metadata["horizon_steps"])
                with torch.no_grad():
                    ordinary = model(probe_batch, rollout_steps=probe_horizon)
                    changed_target = model(
                        _mutate_future_targets(probe_batch),
                        rollout_steps=probe_horizon,
                    )
                    changed_action = model(
                        _zero_task_action(probe_batch),
                        rollout_steps=probe_horizon,
                    )
                target_leakage_absent = _outputs_equal(ordinary, changed_target)
                action_delta = _max_output_delta(ordinary, changed_action)

                checkpoint_path = checkpoint_dir / f"{run_id}.pt"
                save_r5_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    bindings,
                    protocol,
                    learning_rate=LEARNING_RATE,
                    seed=seed,
                )
                restored = load_r5_checkpoint(
                    checkpoint_path,
                    expected_bindings=bindings,
                    expected_protocol=protocol,
                )
                restored.model.eval()
                with torch.no_grad():
                    restored_output = restored.model(
                        probe_batch,
                        rollout_steps=probe_horizon,
                    )
                checkpoint_exact = _outputs_equal(ordinary, restored_output)
                checks = {
                    "train_step_executed": True,
                    "finite_gradients": finite_gradients,
                    "nonzero_gradient": nonzero_gradient,
                    "required_module_gradients": all(module_evidence.values()),
                    "all_rollouts_finite": finite_rollouts,
                    "requested_horizons_executed": observed_horizons
                    == set(map(int, horizons)),
                    "future_target_leakage_absent": target_leakage_absent,
                    "action_conditioning_executable": action_delta > 0.0,
                    "all_public_metrics_finite": public_metrics_finite,
                    "strict_checkpoint_exact_roundtrip": checkpoint_exact,
                }
                passed = all(checks.values())
                run_reports.append(
                    {
                        "run_id": run_id,
                        "combination_id": combination_id,
                        "training_seed": seed,
                        "config": asdict(matrix[combination_id].config),
                        "components": matrix[combination_id].config.component_names(),
                        "parameter_count": parameter_count,
                        "runtime_seconds": time.perf_counter() - started,
                        "action_delta": action_delta,
                        "module_gradient_evidence": module_evidence,
                        "checks": checks,
                        "passed": passed,
                        "claim_boundary": "CPU execution evidence only; no method selection",
                    }
                )
                checkpoint_reports.append(
                    {
                        "run_id": run_id,
                        "path": checkpoint_path.relative_to(output_dir).as_posix(),
                        "sha256": _sha256(checkpoint_path),
                        "exact_roundtrip": checkpoint_exact,
                        "combination_id": restored.model.combination_id,
                        "protocol": restored.protocol.to_dict(),
                        "bindings": restored.bindings,
                    }
                )
                if not passed:
                    failures.append(
                        {
                            "run_id": run_id,
                            "error": "failed checks: "
                            + ", ".join(
                                key for key, value in checks.items() if not value
                            ),
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "run_id": run_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    expected_runs = len(combination_ids) * len(seeds)
    passed_runs = sum(bool(row["passed"]) for row in run_reports)
    ready = not failures and passed_runs == expected_runs
    full_budget = (
        set(combination_ids) == approved
        and len(combination_ids) == len(approved)
        and tuple(seeds) == protocol.training_seeds
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r5_cpu_preflight_ready": ready,
        "r5_gpu_ready": ready and full_budget,
        "full_matrix_and_seed_budget_run": full_budget,
        "locked_test_accessed": False,
        "combination_count": len(combination_ids),
        "combination_ids": list(combination_ids),
        "training_seed_count": len(seeds),
        "training_seeds": list(seeds),
        "expected_run_count": expected_runs,
        "completed_run_count": len(run_reports),
        "passed_run_count": passed_runs,
        "failed_run_count": len(failures),
        "window_count": len(windows),
        "splits": list(normalized_splits),
        "rollout_horizons": list(horizons),
        "claim_boundary": (
            "CPU contract and execution preflight only; no convergence, superiority, "
            "or final architecture claim."
        ),
    }
    _write_json(output_dir / "preflight_summary.json", summary)
    _write_json(output_dir / "protocol.json", _protocol_payload(protocol))
    _write_json(
        output_dir / "combination_matrix.json",
        {
            "combinations": {
                key: value.to_dict() for key, value in matrix.items()
            },
            "executed_combinations": list(combination_ids),
            "freeze_rule": "method remains unfrozen until formal multi-seed results",
        },
    )
    _write_json(
        output_dir / "selected_windows.json",
        [window.to_dict() for window in windows],
    )
    _write_json(output_dir / "combination_seed_reports.json", run_reports)
    _write_json(output_dir / "objective_reports.json", objective_reports)
    _write_json(output_dir / "validation_gate_reports.json", gate_reports)
    _write_json(output_dir / "checkpoint_reports.json", checkpoint_reports)
    _write_json(
        output_dir / "input_provenance.json",
        {
            **input_report,
            "bindings": bindings,
            "r5_source_files": source_hashes,
            "r4_screening_summary": r4_summary,
            "selected_tensor_inputs": _selected_input_provenance(
                dataset_root,
                windows,
            ),
            "information_rate_normalization": {
                "feature": "outcome.rate_sum",
                "feature_index": 12,
                "source_split": "train",
                "mean": rate_mean,
                "scale": rate_scale,
            },
        },
    )
    _write_json(
        output_dir / "gpu_handoff.json",
        {
            "ready": ready and full_budget,
            "combinations": ["A", "B", "C", "D", "E"],
            "training_seeds": list(protocol.training_seeds),
            "max_epochs": protocol.max_epochs,
            "early_stopping_patience": protocol.patience,
            "effective_batch_size": protocol.effective_batch_size,
            "minimum_improvement": protocol.minimum_improvement,
            "learning_rate": LEARNING_RATE,
            "checkpoint_split": protocol.checkpoint_split,
            "calibration_split": protocol.calibration_split,
            "locked_test_accessed": False,
            "next_stage": "R5 formal GPU multi-seed training",
        },
    )
    _write_json(output_dir / "failed_runs.json", failures)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        name = path.relative_to(output_dir).as_posix()
        files[name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "r5_cpu_preflight_ready": ready,
            "r5_gpu_ready": ready and full_budget,
            "files": files,
        },
    )
    if not ready:
        raise RuntimeError("R5 CPU preflight failed; inspect failed_runs.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3",
    )
    parser.add_argument(
        "--r4-screening-root",
        type=Path,
        default=CODE_ROOT
        / "artifacts"
        / "formal_training"
        / "pi_jwm_r4_gpu_screening_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT
        / "artifacts"
        / "preflight"
        / "pi_jwm_r5_cpu_preflight_v1",
    )
    parser.add_argument("--per-horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--combination", action="append", dest="combination_ids")
    args = parser.parse_args()
    result = run_r5_cpu_preflight(
        args.dataset_root,
        args.evaluation_root,
        args.r4_screening_root,
        args.output_dir,
        per_horizon=args.per_horizon,
        hidden_dim=args.hidden_dim,
        combination_ids=(
            ("A", "B", "C", "D", "E")
            if args.combination_ids is None
            else args.combination_ids
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
