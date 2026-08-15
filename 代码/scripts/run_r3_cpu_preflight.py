"""Run the nonlocked PI-JWM R3 explicit-plus-latent CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r3_checkpoint import load_r3_checkpoint, save_r3_checkpoint
from pi_jwm.r3_objective import compute_r3_objective
from pi_jwm.r3_preflight_data import (
    ExplicitStateBatch,
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
    verify_r3_inputs,
)
from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel
from pi_jwm.teacher_evaluation_v3 import METHODS, evaluate_teacher_trajectory


SCHEMA_VERSION = "PIJWM-R3-CPU-Preflight-v1"


def _write_json(path: Path, value: Any) -> None:
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


def _finite_output(output: Any) -> bool:
    tensors = [
        *output.predicted_explicit.values(),
        *output.predicted_logits.values(),
        output.predicted_belief.physical_latent,
        output.predicted_belief.information_latent,
        output.predicted_belief.business_latent,
        output.predicted_belief.joint_latent,
    ]
    return all(bool(torch.isfinite(value).all().item()) for value in tensors)


def _objective_record(
    model_name: str,
    batch: ExplicitStateBatch,
    output: Any,
) -> dict[str, Any]:
    objective = compute_r3_objective(output, batch)
    return {
        "model": model_name,
        **batch.metadata,
        "total": float(objective.total.detach().cpu().item()),
        "terms": {name: asdict(term) for name, term in objective.terms.items()},
    }


def _teacher_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    history = payload["history"]
    target = payload["target"]
    arrays = {
        key: np.concatenate((np.asarray(history[key])[-1:], np.asarray(value)), axis=0)
        for key, value in target.items()
        if key in history
    }
    arrays.update({key: np.asarray(value) for key, value in payload["static"].items()})
    arrays["time"] = np.concatenate(
        (np.asarray(payload["history_time"])[-1:], np.asarray(payload["target_time"]))
    )
    return arrays


def _metric_interface_records(
    payload: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arrays = _teacher_arrays(payload)
    records: list[dict[str, Any]] = []
    for method in METHODS:
        report = evaluate_teacher_trajectory(
            arrays,
            method=method,
            normalization_stats=normalization_stats,
        )
        computed = {
            metric_id: metric["value"]
            for metric_id, metric in report["metrics"].items()
            if metric["status"] == "computed"
        }
        records.append(
            {
                "trajectory_id": payload["window"]["trajectory_id"],
                "split": payload["window"]["split"],
                "horizon_steps": payload["window"]["horizon_steps"],
                "method": method,
                "computed_metric_count": len(computed),
                "computed_metrics": computed,
                "role": "R2 teacher-baseline interface smoke; not an R3 model score",
            }
        )
    return records


def _zero_action_batch(batch: ExplicitStateBatch) -> ExplicitStateBatch:
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


def run_r3_cpu_preflight(
    dataset_root: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    *,
    per_horizon: int = 1,
    splits: Sequence[str] = ("train", "validation", "calibration"),
    horizons: Sequence[int] = (1, 5, 20),
    hidden_dim: int = 4,
    seed: int = 20260804,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"R3 output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if "locked_test" in splits:
        raise ValueError("locked_test cannot be used by the R3 CPU preflight")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    input_report = verify_r3_inputs(dataset_root, evaluation_root)
    rows = read_trajectory_index(dataset_root)
    normalization_stats = _read_json(
        evaluation_root / "evaluation_normalization_stats.json"
    )
    windows = [
        window
        for split in splits
        for window in select_r3_windows(
            dataset_root,
            rows,
            split=split,
            horizons=horizons,
            history_steps=8,
            per_horizon=per_horizon,
            seed=seed,
        )
    ]
    payloads = [load_r3_window(window) for window in windows]
    batches = [make_explicit_batch(payload, normalization_stats) for payload in payloads]

    coupled = R3ReferenceWorldModel(
        R3ReferenceConfig(
            hidden_dim=hidden_dim,
            history_steps=8,
            use_cross_graph_coupling=True,
        )
    )
    control = R3ReferenceWorldModel(
        R3ReferenceConfig(
            hidden_dim=hidden_dim,
            history_steps=8,
            use_cross_graph_coupling=False,
        )
    )
    control.load_state_dict(coupled.state_dict(), strict=True)
    models = {"coupled_reference": coupled, "no_coupling_control": control}
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=1e-4)
        for name, model in models.items()
    }

    trained = {name: False for name in models}
    objective_records: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []
    outputs_by_case: dict[tuple[str, str, int], Any] = {}
    for model_name, model in models.items():
        optimizer = optimizers[model_name]
        for batch in batches:
            horizon = int(batch.metadata["horizon_steps"])
            key = (model_name, str(batch.metadata["trajectory_id"]), horizon)
            if batch.metadata["split"] == "train" and not trained[model_name]:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                training_output = model(batch, rollout_steps=horizon)
                training_objective = compute_r3_objective(training_output, batch)
                training_objective.total.backward()
                optimizer.step()
                trained[model_name] = True
            model.eval()
            with torch.no_grad():
                output = model(batch, rollout_steps=horizon)
            outputs_by_case[key] = output
            objective_records.append(_objective_record(model_name, batch, output))
            rollout_records.append(
                {
                    "model": model_name,
                    **batch.metadata,
                    "finite": _finite_output(output),
                    "joint_latent_shape": list(output.predicted_belief.joint_latent.shape),
                    "teacher_forcing_used": False,
                }
            )

    first_batch = batches[0]
    first_horizon = int(first_batch.metadata["horizon_steps"])
    coupled.eval()
    with torch.no_grad():
        ordinary = coupled(first_batch, rollout_steps=first_horizon)
        zero_action = coupled(
            _zero_action_batch(first_batch), rollout_steps=first_horizon
        )
    action_delta = float(
        torch.max(
            torch.abs(
                ordinary.predicted_belief.joint_latent
                - zero_action.predicted_belief.joint_latent
            )
        ).item()
    )
    coupling_probe = R3ReferenceWorldModel(
        R3ReferenceConfig(
            hidden_dim=hidden_dim,
            history_steps=8,
            use_cross_graph_coupling=False,
        )
    )
    coupling_probe.load_state_dict(coupled.state_dict(), strict=True)
    coupling_probe.eval()
    with torch.no_grad():
        no_coupling_same_parameters = coupling_probe(
            first_batch, rollout_steps=first_horizon
        )
    coupling_delta = float(
        torch.max(
            torch.abs(
                ordinary.predicted_belief.joint_latent
                - no_coupling_same_parameters.predicted_belief.joint_latent
            )
        ).item()
    )

    checkpoint_roundtrip: dict[str, bool] = {}
    for model_name, model in models.items():
        checkpoint_path = output_dir / f"{model_name}.pt"
        save_r3_checkpoint(
            checkpoint_path,
            model,
            optimizers[model_name],
            input_report["bindings"],
            seed=seed,
        )
        restored = load_r3_checkpoint(
            checkpoint_path, expected_bindings=input_report["bindings"]
        )
        restored.model.eval()
        with torch.no_grad():
            expected = model(first_batch, rollout_steps=first_horizon)
            actual = restored.model(first_batch, rollout_steps=first_horizon)
        checkpoint_roundtrip[model_name] = bool(
            torch.allclose(
                expected.predicted_belief.joint_latent,
                actual.predicted_belief.joint_latent,
            )
        )

    metric_records = [
        record
        for payload in payloads
        for record in _metric_interface_records(payload, normalization_stats)
    ]
    dataset_manifest = _read_json(dataset_root / "manifest.json")
    selected_tensor_inputs: dict[str, Any] = {}
    for window in windows:
        name = window.tensor_path.relative_to(dataset_root).as_posix()
        entry = dataset_manifest["files"].get(name)
        actual = _sha256(window.tensor_path)
        if not isinstance(entry, Mapping) or actual != entry.get("sha256"):
            raise ValueError(f"selected R3 tensor is not bound by the R1 manifest: {name}")
        selected_tensor_inputs[name] = {
            "sha256": actual,
            "size_bytes": window.tensor_path.stat().st_size,
            "verified": True,
        }

    checks = {
        "input_contracts_ready": bool(input_report["ready"]),
        "locked_test_remained_sealed": not input_report["locked_test_accessed"],
        "all_rollouts_finite": all(record["finite"] for record in rollout_records),
        "all_objectives_finite": all(
            np.isfinite(record["total"]) for record in objective_records
        ),
        "action_conditioning_executable": action_delta > 0.0,
        "cross_graph_coupling_executable": coupling_delta > 0.0,
        "strict_checkpoint_roundtrip": all(checkpoint_roundtrip.values()),
        "r2_metric_interface_executable": bool(metric_records)
        and all(record["computed_metric_count"] > 0 for record in metric_records),
        "train_step_executed_for_both_controls": all(trained.values()),
    }
    ready = all(checks.values())

    _write_json(output_dir / "selected_windows.json", [window.to_dict() for window in windows])
    _write_json(output_dir / "objective_reports.json", objective_records)
    _write_json(
        output_dir / "rollout_checks.json",
        {
            "records": rollout_records,
            "action_delta": action_delta,
            "coupling_delta": coupling_delta,
            "coupling_comparison": "same_parameters_with_vs_without_cip_cep_cfl",
            "checkpoint_roundtrip": checkpoint_roundtrip,
        },
    )
    _write_json(output_dir / "metric_interface_report.json", metric_records)
    _write_json(
        output_dir / "input_provenance.json",
        {**input_report, "selected_tensor_inputs": selected_tensor_inputs},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "r3_cpu_preflight_ready": ready,
        "locked_test_accessed": False,
        "splits": list(splits),
        "rollout_horizons": list(horizons),
        "window_count": len(windows),
        "models": list(models),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "action_delta": action_delta,
        "coupling_delta": coupling_delta,
        "claim_boundary": (
            "CPU execution and contract preflight only; this artifact is not evidence "
            "of model superiority, convergence, or final method selection."
        ),
    }
    _write_json(output_dir / "preflight_summary.json", summary)

    files = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "r3_cpu_preflight_ready": ready,
            "files": files,
        },
    )
    if not ready:
        raise RuntimeError("R3 CPU preflight failed: " + ", ".join(summary["failed_checks"]))
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
        default=CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_r3_cpu_preflight_v1",
    )
    parser.add_argument("--per-horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4)
    args = parser.parse_args()
    result = run_r3_cpu_preflight(
        args.dataset_root,
        args.evaluation_root,
        args.output_dir,
        per_horizon=args.per_horizon,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
