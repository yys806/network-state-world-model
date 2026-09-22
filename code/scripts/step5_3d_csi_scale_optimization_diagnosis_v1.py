"""STEP 5.3D: bounded CPU CSI scale and optimization diagnosis.

This compares the frozen training path with an in-memory diagnostic-only CSI
decoder mean-bias initialization. It does not alter the formal model defaults,
loss, weighting, curriculum, or Definition 05 semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
SCRIPTS = ROOT / "code" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, KLSchedule, Step52Trainer, Step52TrainingConfig
from step5_3_cpu_training_preflight_v1 import _raw_metrics

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_3d_csi_scale_optimization_diagnosis_v1_20260922"
MAX_STEPS = 200
LOG_EVERY = 20
SAMPLES = [0, 1]


def config(seed: int = 5303) -> Step52TrainingConfig:
    return Step52TrainingConfig(
        seed=seed, batch_size=1, max_steps=MAX_STEPS, max_epochs=1,
        stage1_steps=10, max_horizon=2,
        curriculum=CurriculumConfig(horizons=(1, 2), start_steps=(0, 10)),
        kl_schedule=KLSchedule(target_beta=1.0, warmup_steps=20, free_bits=0.1),
        gradient_clip_norm=1.0,
    )


def _json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _eval(trainer: Step52Trainer, horizon: int) -> dict[str, Any]:
    result = trainer._run_batch(SAMPLES, horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
    metrics = _raw_metrics(trainer, result, horizon, SAMPLES)
    raw_pred = result["csi_predictions"].detach()
    target = trainer.data.target_tensors["target_comm_csi_normalized"][SAMPLES, :horizon]
    mask = trainer.data.target_tensors["target_comm_csi_mask"][SAMPLES, :horizon]
    stats = trainer.data.target_normalization
    raw_target = target * float(stats["csi_std"]) + float(stats["csi_mean"])
    normalized_pred = (raw_pred - float(stats["csi_mean"])) / float(stats["csi_std"])
    expected = (((raw_pred - raw_target) / float(stats["csi_std"])) .square() * mask.to(raw_pred.dtype)).sum() / mask.sum().clamp_min(1)
    actual = result["L_CSI"]
    return {
        "horizon": horizon,
        "L_Mot": float(result["L_Mot"].detach()), "L_CSI": float(actual.detach()), "L_Pred": float(result["L_Pred"].detach()),
        "expected_bridge_mse": float(expected),
        "bridge_abs_error": float((actual - expected).abs()),
        "raw_prediction_mean_db": float(raw_pred[mask].mean()),
        "raw_prediction_std_db": float(raw_pred[mask].std(unbiased=False)),
        "raw_prediction_min_db": float(raw_pred[mask].min()),
        "raw_prediction_max_db": float(raw_pred[mask].max()),
        "raw_target_mean_db": float(raw_target[mask].mean()),
        "raw_target_std_db": float(raw_target[mask].std(unbiased=False)),
        "raw_target_min_db": float(raw_target[mask].min()),
        "raw_target_max_db": float(raw_target[mask].max()),
        "normalized_prediction_mean": float(normalized_pred[mask].mean()),
        "normalized_prediction_std": float(normalized_pred[mask].std(unbiased=False)),
        "normalized_target_mean": float(target[mask].mean()),
        "normalized_target_std": float(target[mask].std(unbiased=False)),
        "raw_csi_mae_db": metrics["csi_raw_mae_db"], "raw_csi_rmse_db": metrics["csi_raw_rmse_db"],
        "motion_metrics": {key: value for key, value in metrics.items() if key.startswith("motion_") and key not in {"motion_raw_mae", "motion_raw_rmse"}},
        "prior_only": bool(result["rollout"]["prior_only"]),
        "future_state_not_rollout_input": not bool(result["rollout"]["future_target_consumed_by_rollout"]),
    }


def _row(trainer: Step52Trainer, step: int) -> dict[str, Any]:
    result = trainer.train_step(global_step=step, epoch=0)
    eval_row = _eval(trainer, int(result["rollout_horizon"]))
    return {
        "step": step, "stage": result["stage"], "horizon": result["rollout_horizon"], "beta_kl": result["beta_kl"],
        "L_Total": result["L_Total"], "L_Mot": result["L_Mot"], "L_CSI": result["L_CSI"],
        "L_KL": result["L_KL"], "L_KL_Phy_raw": result["L_KL_Phy_raw"], "L_KL_Comm_raw": result["L_KL_Comm_raw"],
        "gradient_norm": result["gradient_norm_before_clip"], "gradients_finite": result["gradients_finite"],
        "parameter_update": result["parameter_update"], "gradient_audit": result["gradient_audit"],
        "prior_eval": eval_row,
    }


def _run(mean_bias: bool) -> dict[str, Any]:
    trainer = Step52Trainer.from_unified_development_bundle(config())
    trainer.data.train_indices = list(SAMPLES)
    stats = trainer.data.target_normalization
    if mean_bias:
        with torch.no_grad():
            trainer.model.csi_decoder[-1].bias.fill_(float(stats["csi_mean"]))
    initial = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    trajectory = [_row(trainer, step) for step in range(MAX_STEPS) if step % LOG_EVERY == 0 or step == MAX_STEPS - 1]
    # Rows above only log sparse steps, while optimization still advances every
    # step so the budget remains exactly 200 and cannot be extended by outcome.
    # The loop must execute missing steps between checkpoints.
    if len(trajectory) != MAX_STEPS // LOG_EVERY + 1:
        raise AssertionError("diagnostic trajectory logging invariant failed")
    final = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    return {"mean_bias": mean_bias, "seed": 5303, "samples": SAMPLES, "max_steps": MAX_STEPS, "log_every": LOG_EVERY, "initial": initial, "trajectory": trajectory, "final": final, "parameter_digest": trainer.parameter_digest()}


def _run_correct(mean_bias: bool) -> dict[str, Any]:
    trainer = Step52Trainer.from_unified_development_bundle(config())
    trainer.data.train_indices = list(SAMPLES)
    if mean_bias:
        with torch.no_grad():
            trainer.model.csi_decoder[-1].bias.fill_(float(trainer.data.target_normalization["csi_mean"]))
    initial = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    trajectory = []
    for step in range(MAX_STEPS):
        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            trajectory.append(_row(trainer, step))
        else:
            trainer.train_step(global_step=step, epoch=0)
    final = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    return {"mean_bias": mean_bias, "seed": 5303, "samples": SAMPLES, "max_steps": MAX_STEPS, "log_every": LOG_EVERY, "initial": initial, "trajectory": trajectory, "final": final, "parameter_digest": trainer.parameter_digest()}


def _relative(initial: float, final: float) -> float:
    return (initial - final) / max(abs(initial), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    baseline = _run_correct(False)
    mean_bias = _run_correct(True)
    bridge = {"h1": baseline["initial"]["h1"], "h2": baseline["initial"]["h2"], "all_match": all(abs(baseline["initial"][h]["L_CSI"] - baseline["initial"][h]["expected_bridge_mse"]) < 1e-5 for h in ("h1", "h2"))}
    comparison = {"baseline": {h: {"initial_L_CSI": baseline["initial"][h]["L_CSI"], "final_L_CSI": baseline["final"][h]["L_CSI"], "relative_drop": _relative(baseline["initial"][h]["L_CSI"], baseline["final"][h]["L_CSI"])} for h in ("h1", "h2")}, "mean_bias": {h: {"initial_L_CSI": mean_bias["initial"][h]["L_CSI"], "final_L_CSI": mean_bias["final"][h]["L_CSI"], "relative_drop": _relative(mean_bias["initial"][h]["L_CSI"], mean_bias["final"][h]["L_CSI"])} for h in ("h1", "h2")}}
    gate = {"relative_drop": 0.50, "final_normalized_family_mse": 1.0}
    def verdict(run: dict[str, Any]) -> str:
        ok = all(comparison["mean_bias" if run["mean_bias"] else "baseline"][h]["relative_drop"] >= gate["relative_drop"] and run["final"][h]["L_CSI"] <= gate["final_normalized_family_mse"] for h in ("h1", "h2"))
        return "TINY_OVERFIT_GO" if ok else "TINY_OVERFIT_NO_GO"
    receipt = {"passed": bridge["all_match"], "scale_bridge_check": bridge["all_match"], "baseline_verdict": verdict(baseline), "mean_bias_diagnostic_verdict": verdict(mean_bias), "forbidden_scope": {"gpu": False, "formal_training": False, "formal_dataset": False, "locked_test": False, "baseline": False, "planner": False, "performance_claim": False}, "diagnostic_only": True}
    result = {"schema_version": "PI-JWM-Step-5.3D-CSI-Diagnosis-v1", "observation": {"normalization": baseline["initial"]["h1"]}, "interpretation": "pending researcher decision; compare raw-head mean-bias initialization with normalized-output decoder bridge", "scale_bridge_audit": bridge, "baseline": baseline, "mean_bias_diagnostic": mean_bias, "comparison": comparison, "receipt": receipt}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("scale_bridge_audit", bridge), ("baseline_longer_run", baseline), ("mean_bias_diagnostic", mean_bias), ("comparison", comparison), ("receipt", receipt)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.iterdir() if p.is_file()]
    manifest = {"schema_version": "PI-JWM-Step-5.3D-Manifest-v1", "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}, "passed": receipt["passed"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "baseline_verdict": receipt["baseline_verdict"], "mean_bias_diagnostic_verdict": receipt["mean_bias_diagnostic_verdict"], "bridge": bridge, "comparison": comparison}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
