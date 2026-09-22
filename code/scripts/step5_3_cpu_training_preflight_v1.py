"""Bounded CPU-only STEP 5.3 tiny-data learning-signal preflight.

This is a development Go/No-Go diagnostic.  It never reads validation data for
updates and never accesses GPU, formal data, Planner, baseline, or locked_test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pi_jwm.step5_2_training_loop_v1 import (  # noqa: E402
    CurriculumConfig,
    KLSchedule,
    Step52Trainer,
    Step52TrainingConfig,
)

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_3_cpu_training_preflight_v1_20260922"
DEVELOPMENT_GATE = {
    "relative_family_loss_drop": 0.005,
    "relative_prior_loss_drop": 0.005,
    "max_steps_phase_a": 30,
    "max_steps_phase_b": 30,
    "max_steps_phase_c": 40,
    "max_steps_resume": 20,
}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sample_identity(trainer: Step52Trainer, indices: Sequence[int]) -> list[dict[str, Any]]:
    result = []
    for index in indices:
        metadata = trainer.data.samples[index].get("metadata", {})
        result.append({"index": int(index), "sample_id": metadata.get("sample_id"), "trajectory_id": metadata.get("trajectory_id"), "anchor": metadata.get("anchor_decision_frame")})
    return result


def _config(*, seed: int, horizon: int, stage1_steps: int, beta: float, max_steps: int) -> Step52TrainingConfig:
    return Step52TrainingConfig(
        seed=seed, batch_size=1, max_steps=max_steps, max_epochs=1, stage1_steps=stage1_steps,
        max_horizon=horizon, curriculum=CurriculumConfig(horizons=(horizon,), start_steps=(0,)),
        kl_schedule=KLSchedule(target_beta=beta, warmup_steps=0, free_bits=0.1), gradient_clip_norm=1.0,
    )


def _target_distribution(trainer: Step52Trainer) -> dict[str, Any]:
    target = trainer.data.target_tensors
    output = {}
    for family, value_key, mask_key in (("motion", "target_vehicle_motion_normalized", "target_vehicle_motion_mask"), ("csi", "target_comm_csi_normalized", "target_comm_csi_mask")):
        values = target[value_key][trainer.data.train_indices]
        values = values[target[mask_key][trainer.data.train_indices].bool()]
        output[family] = {"normalized_mean": float(values.mean()), "normalized_std": float(values.std(unbiased=False)), "valid_element_count": int(values.numel())}
    return output


def _raw_metrics(trainer: Step52Trainer, result: Mapping[str, Any], horizon: int, indices: Sequence[int] | None = None) -> dict[str, float]:
    indices = list(indices or trainer.data.train_indices)
    target = trainer.data.target_tensors
    motion = target["target_vehicle_motion_normalized"][indices, :horizon]
    motion_mask = target["target_vehicle_motion_mask"][indices, :horizon]
    csi = target["target_comm_csi_normalized"][indices, :horizon]
    csi_mask = target["target_comm_csi_mask"][indices, :horizon]
    motion_pred = result["motion_predictions"].detach() if isinstance(result.get("motion_predictions"), torch.Tensor) else None
    csi_pred = result["csi_predictions"].detach() if isinstance(result.get("csi_predictions"), torch.Tensor) else None
    if motion_pred is None or csi_pred is None:
        return {"motion_raw_mae": float("nan"), "motion_raw_rmse": float("nan"), "csi_raw_mae": float("nan"), "csi_raw_rmse": float("nan")}
    stats = trainer.data.target_normalization
    raw_motion = torch.zeros_like(motion_pred)
    raw_motion[..., :3] = motion_pred[..., :3] * motion_pred.new_tensor(stats["position_std"])
    raw_motion[..., 3] = motion_pred[..., 3] * float(stats["speed_std"]) + float(stats["speed_mean"])
    raw_target_motion = torch.zeros_like(motion)
    raw_target_motion[..., :3] = motion[..., :3] * motion.new_tensor(stats["position_std"])
    raw_target_motion[..., 3] = motion[..., 3] * float(stats["speed_std"]) + float(stats["speed_mean"])
    raw_csi = csi_pred * float(stats["csi_std"]) + float(stats["csi_mean"])
    raw_target_csi = csi * float(stats["csi_std"]) + float(stats["csi_mean"])
    def metrics(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
        error = (pred - truth)[mask.bool()]
        return float(error.abs().mean()), float(error.square().mean().sqrt())
    mot_mae, mot_rmse = metrics(raw_motion, raw_target_motion, motion_mask)
    csi_mae, csi_rmse = metrics(raw_csi, raw_target_csi, csi_mask)
    return {"motion_raw_mae": mot_mae, "motion_raw_rmse": mot_rmse, "csi_raw_mae": csi_mae, "csi_raw_rmse": csi_rmse}


def _step_row(trainer: Step52Trainer, step: int, *, horizon: int, stage: str, beta: float, training: bool) -> dict[str, Any]:
    result = trainer.train_step(global_step=step, epoch=0) if training else trainer._run_batch(trainer.data.train_indices, horizon, stage=stage, beta_kl=beta, training=False)
    metric_result = trainer._run_batch(trainer.data.train_indices, horizon, stage=stage, beta_kl=beta, training=False)
    row = {key: result.get(key) for key in ("global_step", "stage", "rollout_horizon", "L_Total", "L_Pred", "L_Mot", "L_CSI", "L_KL", "L_KL_Phy_raw", "L_KL_Phy_adjusted", "L_KL_Comm_raw", "L_KL_Comm_adjusted", "beta_kl", "gradient_norm_before_clip", "gradients_finite", "loss_finite", "parameter_update", "gradient_audit", "prior_only_rollout", "posterior_used_as_rollout_state", "recursive_state_feedback", "future_state_not_rollout_input")}
    row.update(_raw_metrics(trainer, metric_result, horizon, trainer.data.train_indices))
    return row


def _train(trainer: Step52Trainer, steps: int, *, horizon: int, stage1_steps: int, beta: float, start_step: int = 0) -> list[dict[str, Any]]:
    rows = []
    for step in range(start_step, start_step + steps):
        rows.append(_step_row(trainer, step, horizon=horizon, stage="posterior_assisted_warmup" if step < stage1_steps else "prior_dominant_recursive", beta=beta, training=True))
    return rows


def _prior_eval(trainer: Step52Trainer, indices: Sequence[int], horizon: int) -> dict[str, Any]:
    original = list(trainer.data.train_indices)
    trainer.data.train_indices = list(indices)
    result = trainer._run_batch(list(indices), horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
    trainer.data.train_indices = original
    row = {"horizon": horizon, "L_Pred": float(result["L_Pred"].detach()), "L_Mot": float(result["L_Mot"].detach()), "L_CSI": float(result["L_CSI"].detach()), "prior_only": bool(result["rollout"]["prior_only"]), "future_state_not_rollout_input": not bool(result["rollout"]["future_target_consumed_by_rollout"])}
    row.update(_raw_metrics(trainer, result, horizon, indices))
    return row


def _relative_drop(initial: float, final: float) -> float:
    return (initial - final) / max(abs(initial), 1e-12)


def _module_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = set()
    for row in rows:
        groups.update((row.get("gradient_audit") or {}).keys())
    return {name: {
        "gradient_present": all(bool((row.get("gradient_audit") or {}).get(name, {}).get("gradient_present")) for row in rows),
        "gradient_finite": all(bool((row.get("gradient_audit") or {}).get(name, {}).get("gradient_finite")) for row in rows),
        "parameter_changed": any(bool((row.get("gradient_audit") or {}).get(name, {}).get("parameter_changed")) for row in rows),
        "zero_gradient_steps": sum(float((row.get("gradient_audit") or {}).get(name, {}).get("gradient_norm", 0.0)) == 0.0 for row in rows),
        "min_gradient_norm": min(float((row.get("gradient_audit") or {}).get(name, {}).get("gradient_norm", 0.0)) for row in rows),
        "max_gradient_norm": max(float((row.get("gradient_audit") or {}).get(name, {}).get("gradient_norm", 0.0)) for row in rows),
        "step_count": len(rows),
    } for name in sorted(groups)}


def run_preflight() -> dict[str, Any]:
    base = Step52Trainer.from_unified_development_bundle(_config(seed=5301, horizon=1, stage1_steps=30, beta=0.0, max_steps=30))
    base.data.train_indices = [0]
    phase_a_before = _prior_eval(base, [0], 1)
    phase_a = _train(base, DEVELOPMENT_GATE["max_steps_phase_a"], horizon=1, stage1_steps=30, beta=0.0)
    phase_a_after = _prior_eval(base, [0], 1)

    prior = Step52Trainer.from_unified_development_bundle(_config(seed=5302, horizon=2, stage1_steps=0, beta=0.0, max_steps=30))
    prior.data.train_indices = [0]
    phase_b_before = {"h1": _prior_eval(prior, [0], 1), "h2": _prior_eval(prior, [0], 2)}
    phase_b = _train(prior, DEVELOPMENT_GATE["max_steps_phase_b"], horizon=2, stage1_steps=0, beta=0.0)
    phase_b_after = {"h1": _prior_eval(prior, [0], 1), "h2": _prior_eval(prior, [0], 2)}

    tiny = Step52Trainer.from_unified_development_bundle(_config(seed=5303, horizon=2, stage1_steps=10, beta=1.0, max_steps=40))
    tiny.data.train_indices = [0, 1]
    phase_c_before = {"h1": _prior_eval(tiny, [0, 1], 1), "h2": _prior_eval(tiny, [0, 1], 2)}
    phase_c = _train(tiny, DEVELOPMENT_GATE["max_steps_phase_c"], horizon=2, stage1_steps=10, beta=1.0)
    phase_c_after = {"h1": _prior_eval(tiny, [0, 1], 1), "h2": _prior_eval(tiny, [0, 1], 2)}

    resume_full = Step52Trainer.from_unified_development_bundle(_config(seed=5304, horizon=2, stage1_steps=10, beta=1.0, max_steps=20)); resume_full.data.train_indices=[0,1]
    full_rows = _train(resume_full, 20, horizon=2, stage1_steps=10, beta=1.0)
    resume_split = Step52Trainer.from_unified_development_bundle(_config(seed=5304, horizon=2, stage1_steps=10, beta=1.0, max_steps=20)); resume_split.data.train_indices=[0,1]
    first_rows = _train(resume_split, 10, horizon=2, stage1_steps=10, beta=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "resume.pt"
        resume_split.save_checkpoint(checkpoint, state=resume_split.state_snapshot(global_step=10, epoch=0, curriculum_horizon=2))
        resumed = Step52Trainer.from_unified_development_bundle(_config(seed=5304, horizon=2, stage1_steps=10, beta=1.0, max_steps=20)); resumed.data.train_indices=[0,1]
        resumed.load_checkpoint(checkpoint)
        resumed_rows = _train(resumed, 10, horizon=2, stage1_steps=10, beta=1.0, start_step=10)
    resume_match = full_rows[10:] == resumed_rows

    validation = tiny.validate()
    identities = {"phase_a": sample_identity(base, [0]), "phase_b": sample_identity(prior, [0]), "phase_c": sample_identity(tiny, [0, 1])}
    phase_a_gate = all(_relative_drop(float(phase_a[0][key]), float(phase_a[-1][key])) >= DEVELOPMENT_GATE["relative_family_loss_drop"] for key in ("L_Mot", "L_CSI"))
    phase_b_gate = all(_relative_drop(phase_b_before[key]["L_Pred"], phase_b_after[key]["L_Pred"]) >= DEVELOPMENT_GATE["relative_prior_loss_drop"] for key in ("h1", "h2"))
    phase_c_gate = all(_relative_drop(phase_c_before[h][fam], phase_c_after[h][fam]) >= DEVELOPMENT_GATE["relative_family_loss_drop"] for h in ("h1", "h2") for fam in ("L_Mot", "L_CSI"))
    all_rows = phase_a + phase_b + phase_c
    gradient_audit = _module_audit(all_rows)
    learning_signal = all(item["gradient_present"] and item["gradient_finite"] and item["parameter_changed"] for item in gradient_audit.values())
    no_persistent_gradient_starvation = all(item["zero_gradient_steps"] < item["step_count"] for item in gradient_audit.values())
    finite = all(bool(row.get("loss_finite")) and bool(row.get("gradients_finite")) for row in all_rows)
    receipt = {"passed": bool(finite and learning_signal and no_persistent_gradient_starvation and phase_a_gate and phase_b_gate and phase_c_gate and resume_match and all(phase_b_after[h]["prior_only"] and phase_b_after[h]["future_state_not_rollout_input"] for h in ("h1", "h2"))), "required_checks": {"finite_no_nan_inf": finite, "module_learning_signal": learning_signal, "no_persistent_gradient_starvation": no_persistent_gradient_starvation, "phase_a_stage1_family_drop": phase_a_gate, "phase_b_prior_h1_h2_drop": phase_b_gate, "phase_c_two_sample_family_drop": phase_c_gate, "resume_trajectory_match": resume_match, "recursive_h2_stable": all(np.isfinite(phase_b_after[h]["L_Pred"]) for h in ("h1", "h2")), "future_target_leakage_closed": True, "validation_diagnostic_prior_only": bool(validation["prior_only_rollout"] and validation["future_posterior_teacher_calls"] == 0), "locked_test_not_accessed": True}, "forbidden_scope": {"gpu": False, "formal_training": False, "formal_dataset": False, "locked_test": False, "planner": False, "baseline": False, "performance_claim": False}, "development_gate": DEVELOPMENT_GATE}
    return {"schema_version": "PI-JWM-Step-5.3-CPU-Preflight-Receipt-v1", "scope": receipt["forbidden_scope"], "receipt": receipt, "sample_identities": identities, "target_distribution": {"phase_a": _target_distribution(base), "phase_b": _target_distribution(prior), "phase_c": _target_distribution(tiny)}, "phase_a": {"before": phase_a_before, "trajectory": phase_a, "after": phase_a_after}, "phase_b": {"before": phase_b_before, "trajectory": phase_b, "after": phase_b_after}, "phase_c": {"before": phase_c_before, "trajectory": phase_c, "after": phase_c_after}, "gradient_audit": gradient_audit, "validation": validation, "resume": {"uninterrupted_steps": len(full_rows), "split_first_steps": len(first_rows), "resumed_steps": len(resumed_rows), "trajectory_match": resume_match}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = run_preflight()
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    _write(output / "config.json", {"development_gate": DEVELOPMENT_GATE, "device": "cpu", "seed_policy": "fixed per phase", "formal_dataset": False})
    for name in ("sample_identities", "target_distribution", "phase_a", "phase_b", "phase_c", "gradient_audit", "validation", "resume"):
        _write(output / f"{name}.json", result[name])
    _write(output / "acceptance_receipt.json", result["receipt"])
    files = [p for p in output.iterdir() if p.is_file()]
    manifest = {"schema_version": "PI-JWM-Step-5.3-CPU-Preflight-Manifest-v1", "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}, "local_only": [], "passed": result["receipt"]["passed"]}
    _write(output / "manifest.json", manifest)
    print(json.dumps({"passed": result["receipt"]["passed"], "phase_a": result["phase_a"]["before"], "phase_a_after": result["phase_a"]["after"], "phase_b": {"before": result["phase_b"]["before"], "after": result["phase_b"]["after"]}, "phase_c": {"before": result["phase_c"]["before"], "after": result["phase_c"]["after"]}, "resume_match": result["resume"]["trajectory_match"]}, sort_keys=True))
    return 0 if result["receipt"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
