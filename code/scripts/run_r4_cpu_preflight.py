"""Run the nonlocked PI-JWM R4 controlled-candidate CPU preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    ExplicitStateBatch,
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
    verify_r3_inputs,
)
from pi_jwm.r4_checkpoint import (
    R4TrainingBudget,
    load_r4_checkpoint,
    save_r4_checkpoint,
)
from pi_jwm.r4_module_registry import (
    candidate_matrix,
    candidate_registry,
    make_single_module_config,
    reference_r4_config,
)
from pi_jwm.r4_objective import compute_r4_objective
from pi_jwm.r4_world_model import build_r4_world_model


SCHEMA_VERSION = "PIJWM-R4-CPU-Preflight-v1"
DEFAULT_SEED = 20260804
R4_SOURCE_FILES = (
    SRC_ROOT / "pi_jwm" / "r3_preflight_data.py",
    SRC_ROOT / "pi_jwm" / "r3_world_model.py",
    SRC_ROOT / "pi_jwm" / "r3_objective.py",
    SRC_ROOT / "pi_jwm" / "r3_checkpoint.py",
    SRC_ROOT / "pi_jwm" / "r4_module_registry.py",
    SRC_ROOT / "pi_jwm" / "r4_world_model.py",
    SRC_ROOT / "pi_jwm" / "r4_objective.py",
    SRC_ROOT / "pi_jwm" / "r4_checkpoint.py",
    Path(__file__),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _r4_source_binding() -> tuple[str, dict[str, str]]:
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


def _information_rate_stats(
    normalization_stats: Mapping[str, Any],
) -> tuple[float, float]:
    feature = normalization_stats.get("features", {}).get("information_edge_state", {})
    mean = feature.get("mean")
    scale = feature.get("scale")
    if not isinstance(mean, list) or not isinstance(scale, list):
        raise ValueError("information-edge normalization statistics are missing")
    if len(mean) <= 12 or len(scale) <= 12 or float(scale[12]) <= 0.0:
        raise ValueError("information rate normalization at feature index 12 is invalid")
    return float(mean[12]), float(scale[12])


def executable_candidate_configs(
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float,
    information_rate_scale: float,
) -> dict[str, Any]:
    """Return the reference and every executable one-family R4 arm."""

    configs = {
        "reference": reference_r4_config(
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        )
    }
    for name, spec in candidate_registry().items():
        if spec.status != "executable":
            continue
        configs[name] = make_single_module_config(
            spec.family,
            name,
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        )
    return configs


def _finite_output(output: Any) -> bool:
    tensors = [
        *output.predicted_explicit.values(),
        *output.predicted_logits.values(),
        output.predicted_belief.physical_latent,
        output.predicted_belief.information_latent,
        output.predicted_belief.business_latent,
        output.predicted_belief.joint_latent,
        *getattr(output, "probabilistic_parameters", {}).values(),
    ]
    return all(bool(torch.isfinite(value).all().item()) for value in tensors)


def _output_tensors(output: Any) -> dict[str, torch.Tensor]:
    tensors = {
        **{f"explicit::{key}": value for key, value in output.predicted_explicit.items()},
        **{f"logit::{key}": value for key, value in output.predicted_logits.items()},
        "belief::physical": output.predicted_belief.physical_latent,
        "belief::information": output.predicted_belief.information_latent,
        "belief::business": output.predicted_belief.business_latent,
        "belief::joint": output.predicted_belief.joint_latent,
        **{
            f"probabilistic::{key}": value
            for key, value in getattr(output, "probabilistic_parameters", {}).items()
        },
    }
    return tensors


def _outputs_equal(left: Any, right: Any) -> bool:
    left_tensors = _output_tensors(left)
    right_tensors = _output_tensors(right)
    return left_tensors.keys() == right_tensors.keys() and all(
        torch.equal(left_tensors[key], right_tensors[key]) for key in left_tensors
    )


def _max_output_delta(left: Any, right: Any) -> float:
    left_tensors = _output_tensors(left)
    right_tensors = _output_tensors(right)
    shared = left_tensors.keys() & right_tensors.keys()
    if not shared:
        return 0.0
    return max(
        float(torch.max(torch.abs(left_tensors[key] - right_tensors[key])).item())
        for key in shared
    )


def _zero_task_action(batch: ExplicitStateBatch) -> ExplicitStateBatch:
    future_action = dict(batch.future_action)
    future_action["task_action"] = torch.zeros_like(future_action["task_action"])
    return ExplicitStateBatch(
        history=batch.history,
        history_action=batch.history_action,
        future_action=future_action,
        target=batch.target,
        static=batch.static,
        metadata=batch.metadata,
    )


def _mutate_future_targets(batch: ExplicitStateBatch) -> ExplicitStateBatch:
    changed = copy.deepcopy(batch)
    for key, value in changed.target.items():
        if torch.is_floating_point(value):
            changed.target[key] = value + 17.0
        elif value.dtype == torch.bool:
            changed.target[key] = ~value
        else:
            changed.target[key] = value + 1
    return changed


def _term_record(term: Any) -> dict[str, Any]:
    return asdict(term)


def _selected_input_provenance(
    dataset_root: Path,
    windows: Sequence[Any],
) -> dict[str, Any]:
    manifest = _read_json(dataset_root / "manifest.json")
    selected: dict[str, Any] = {}
    for window in windows:
        name = window.tensor_path.relative_to(dataset_root).as_posix()
        entry = manifest.get("files", {}).get(name)
        actual = _sha256(window.tensor_path)
        if not isinstance(entry, Mapping) or actual != entry.get("sha256"):
            raise ValueError(f"selected R4 tensor is not bound by the R1 manifest: {name}")
        selected[name] = {
            "sha256": actual,
            "size_bytes": window.tensor_path.stat().st_size,
            "verified": True,
        }
    return selected


def run_r4_cpu_preflight(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    *,
    per_horizon: int = 1,
    splits: Sequence[str] = ("train", "validation", "calibration"),
    horizons: Sequence[int] = (1, 5, 20),
    hidden_dim: int = 4,
    history_steps: int = 8,
    candidate_names: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute CPU contract checks; it does not select a winning R4 method."""

    if "locked_test" in splits:
        raise ValueError("locked_test cannot be used by the R4 CPU preflight")
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"R4 output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    normalization_stats = _read_json(
        evaluation_root / "evaluation_normalization_stats.json"
    )
    rate_mean, rate_scale = _information_rate_stats(normalization_stats)
    all_configs = executable_candidate_configs(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )
    selected_names = list(all_configs) if candidate_names is None else list(candidate_names)
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("candidate_names must be non-empty and unique")
    unknown = sorted(set(selected_names) - set(all_configs))
    if unknown:
        raise ValueError("unknown or non-executable R4 candidates: " + ", ".join(unknown))

    rows = read_trajectory_index(dataset_root)
    windows = [
        window
        for split in splits
        for window in select_r3_windows(
            dataset_root,
            rows,
            split=split,
            horizons=horizons,
            history_steps=history_steps,
            per_horizon=per_horizon,
            seed=seed,
        )
    ]
    payloads = [load_r3_window(window) for window in windows]
    batches = [make_explicit_batch(payload, normalization_stats) for payload in payloads]
    train_batches = [batch for batch in batches if batch.metadata["split"] == "train"]
    if not train_batches:
        raise ValueError("R4 CPU preflight requires at least one nonlocked train window")

    source_digest, source_hashes = _r4_source_binding()
    bindings = {**input_report["bindings"], "source_code_sha256": source_digest}
    budget = R4TrainingBudget(
        epochs=30,
        patience=5,
        learning_rate=1.0e-4,
        training_seed=seed,
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_reports: list[dict[str, Any]] = []
    objective_reports: list[dict[str, Any]] = []
    checkpoint_reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for candidate_index, name in enumerate(selected_names):
        config = all_configs[name]
        started = time.perf_counter()
        try:
            torch.manual_seed(seed)
            model = build_r4_world_model(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=budget.learning_rate)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())

            training_batch = train_batches[0]
            training_horizon = int(training_batch.metadata["horizon_steps"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            training_output = model(training_batch, rollout_steps=training_horizon)
            training_objective = compute_r4_objective(training_output, training_batch)
            training_objective.total.backward()
            gradient_tensors = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            finite_gradients = bool(gradient_tensors) and all(
                bool(torch.isfinite(value).all().item()) for value in gradient_tensors
            )
            nonzero_gradient = bool(gradient_tensors) and any(
                bool(torch.count_nonzero(value).item()) for value in gradient_tensors
            )
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
                        "candidate": name,
                        **batch.metadata,
                        "total": float(objective.total.detach().cpu().item()),
                        "terms": {
                            key: _term_record(value) for key, value in objective.terms.items()
                        },
                        "auxiliary_terms": {
                            key: _term_record(value)
                            for key, value in objective.auxiliary_terms.items()
                        },
                    }
                )

            probe_batch = batches[0]
            probe_horizon = int(probe_batch.metadata["horizon_steps"])
            model.eval()
            with torch.no_grad():
                ordinary = model(probe_batch, rollout_steps=probe_horizon)
                changed_target = model(
                    _mutate_future_targets(probe_batch), rollout_steps=probe_horizon
                )
                changed_action = model(
                    _zero_task_action(probe_batch), rollout_steps=probe_horizon
                )
            target_leakage_absent = _outputs_equal(ordinary, changed_target)
            action_delta = _max_output_delta(ordinary, changed_action)

            checkpoint_path = checkpoint_dir / f"{name}.pt"
            save_r4_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                bindings,
                budget,
                seed=seed,
            )
            restored = load_r4_checkpoint(
                checkpoint_path,
                expected_bindings=bindings,
            )
            restored.model.eval()
            with torch.no_grad():
                restored_output = restored.model(
                    probe_batch, rollout_steps=probe_horizon
                )
            checkpoint_exact = _outputs_equal(ordinary, restored_output)
            checks = {
                "train_step_executed": True,
                "finite_gradients": finite_gradients,
                "nonzero_gradient": nonzero_gradient,
                "all_rollouts_finite": finite_rollouts,
                "requested_horizons_executed": observed_horizons == set(map(int, horizons)),
                "future_target_leakage_absent": target_leakage_absent,
                "action_conditioning_executable": action_delta > 0.0,
                "strict_checkpoint_exact_roundtrip": checkpoint_exact,
            }
            passed = all(checks.values())
            candidate_reports.append(
                {
                    "candidate": name,
                    "candidate_index": candidate_index,
                    "config": asdict(config),
                    "components": config.component_names(),
                    "parameter_count": parameter_count,
                    "runtime_seconds": time.perf_counter() - started,
                    "action_delta": action_delta,
                    "checks": checks,
                    "passed": passed,
                    "claim_boundary": "execution evidence only; no superiority claim",
                }
            )
            checkpoint_reports.append(
                {
                    "candidate": name,
                    "path": checkpoint_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(checkpoint_path),
                    "exact_roundtrip": checkpoint_exact,
                    "budget": asdict(restored.budget),
                    "bindings": restored.bindings,
                }
            )
            if not passed:
                failures.append(
                    {
                        "candidate": name,
                        "error": "failed checks: "
                        + ", ".join(key for key, passed_check in checks.items() if not passed_check),
                    }
                )
        except Exception as error:  # Keep the complete matrix auditable.
            failures.append(
                {
                    "candidate": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    full_matrix = set(selected_names) == set(all_configs) and len(selected_names) == len(all_configs)
    passed_candidates = {record["candidate"] for record in candidate_reports if record["passed"]}
    ready = not failures and passed_candidates == set(selected_names)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r4_cpu_preflight_ready": ready,
        "gpu_screening_ready": ready and full_matrix,
        "full_executable_matrix_run": full_matrix,
        "locked_test_accessed": False,
        "candidate_count": len(selected_names),
        "candidates": selected_names,
        "window_count": len(windows),
        "splits": list(splits),
        "rollout_horizons": list(horizons),
        "failed_candidate_count": len(failures),
        "claim_boundary": (
            "CPU execution and contract preflight only; this artifact does not establish "
            "convergence, model superiority, or final module selection."
        ),
    }
    _write_json(output_dir / "preflight_summary.json", summary)
    _write_json(
        output_dir / "candidate_matrix.json",
        {
            "registry": candidate_matrix(),
            "selected_candidates": selected_names,
            "controlled_change_rule": "reference plus at most one changed module family",
        },
    )
    _write_json(output_dir / "selected_windows.json", [window.to_dict() for window in windows])
    _write_json(output_dir / "candidate_reports.json", candidate_reports)
    _write_json(output_dir / "objective_reports.json", objective_reports)
    _write_json(output_dir / "checkpoint_reports.json", checkpoint_reports)
    _write_json(
        output_dir / "input_provenance.json",
        {
            **input_report,
            "bindings": bindings,
            "r4_source_code_files": source_hashes,
            "selected_tensor_inputs": _selected_input_provenance(dataset_root, windows),
            "information_rate_normalization": {
                "feature": "outcome.rate_sum",
                "feature_index": 12,
                "source_split": "train",
                "mean": rate_mean,
                "scale": rate_scale,
            },
        },
    )
    _write_json(output_dir / "failed_candidates.json", failures)

    files = {}
    for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        name = path.relative_to(output_dir).as_posix()
        files[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "r4_cpu_preflight_ready": ready,
            "gpu_screening_ready": ready and full_matrix,
            "files": files,
        },
    )
    if not ready:
        raise RuntimeError("R4 CPU preflight failed; inspect failed_candidates.json")
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
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_r4_cpu_preflight_v1",
    )
    parser.add_argument("--per-horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--candidate", action="append", dest="candidate_names")
    args = parser.parse_args()
    result = run_r4_cpu_preflight(
        args.dataset_root,
        args.evaluation_root,
        args.output_dir,
        per_horizon=args.per_horizon,
        hidden_dim=args.hidden_dim,
        candidate_names=args.candidate_names,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
