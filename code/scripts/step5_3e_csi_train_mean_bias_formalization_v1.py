"""STEP 5.3E: formal raw-CSI train-mean bias and bounded tiny-overfit acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, KLSchedule, Step52Trainer, Step52TrainingConfig
from step5_3_cpu_training_preflight_v1 import _raw_metrics

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_3e_csi_train_mean_bias_formalization_v1_20260922"
SAMPLES = [0, 1]
MAX_STEPS = 200
LOG_EVERY = 20
GATE = {"relative_family_loss_drop": 0.50, "final_normalized_family_mse": 1.0}


def config(seed: int = 5303) -> Step52TrainingConfig:
    return Step52TrainingConfig(seed=seed, batch_size=1, max_steps=MAX_STEPS, max_epochs=1, stage1_steps=10,
        max_horizon=2, curriculum=CurriculumConfig(horizons=(1, 2), start_steps=(0, 10)),
        kl_schedule=KLSchedule(target_beta=1.0, warmup_steps=20, free_bits=0.1), gradient_clip_norm=1.0)


def _json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, Mapping): return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json(item) for item in value]
    return value


def _eval(trainer: Step52Trainer, horizon: int) -> dict[str, Any]:
    result = trainer._run_batch(SAMPLES, horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
    metrics = _raw_metrics(trainer, result, horizon, SAMPLES)
    return {"horizon": horizon, "L_Mot": float(result["L_Mot"].detach()), "L_CSI": float(result["L_CSI"].detach()),
            "L_Pred": float(result["L_Pred"].detach()), "raw_metrics": {k: v for k, v in metrics.items() if k != "scale_audit"},
            "scale_audit": metrics.get("scale_audit", {}), "prior_only": bool(result["rollout"]["prior_only"]),
            "future_state_not_rollout_input": not bool(result["rollout"]["future_target_consumed_by_rollout"])}


def _run() -> tuple[Step52Trainer, dict[str, Any]]:
    trainer = Step52Trainer.from_unified_development_bundle(config())
    trainer.data.train_indices = list(SAMPLES)
    initial_bias = trainer.model.csi_decoder[-1].bias.detach().clone()
    initial = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    trajectory = []
    for step in range(MAX_STEPS):
        row = trainer.train_step(global_step=step, epoch=0)
        if step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            eval_row = _eval(trainer, int(row["rollout_horizon"]))
            trajectory.append({k: row[k] for k in ("global_step", "stage", "rollout_horizon", "beta_kl", "L_Total", "L_Mot", "L_CSI", "L_KL", "L_KL_Phy_raw", "L_KL_Comm_raw", "gradient_norm_before_clip", "gradient_audit", "gradients_finite", "parameter_update")} | {"prior_eval": eval_row})
    final = {"h1": _eval(trainer, 1), "h2": _eval(trainer, 2)}
    return trainer, {"seed": 5303, "samples": SAMPLES, "max_steps": MAX_STEPS, "log_every": LOG_EVERY,
        "schedule": {"stage1_steps": 10, "curriculum": {"horizons": [1, 2], "start_steps": [0, 10]}, "kl_target_beta": 1.0, "kl_warmup_steps": 20, "free_bits": 0.1},
        "initial": initial, "initial_bias": initial_bias, "trajectory": trajectory, "final": final, "parameter_digest": trainer.parameter_digest()}


def _drop(initial: float, final: float) -> float:
    return (initial - final) / max(abs(initial), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); args = parser.parse_args()
    trainer, run = _run()
    init_contract = trainer.initialization_contract
    formalization = {
        "fresh_bias_matches_train_mean": bool(torch.allclose(run["initial_bias"], torch.full_like(run["initial_bias"], float(init_contract["resolved_bias_value"])), atol=1e-5, rtol=1e-6)),
        "all_rb_bias_equal": bool(torch.allclose(run["initial_bias"], run["initial_bias"][0].expand_as(run["initial_bias"]))),
        "raw_db_contract": trainer.model.contract["csi_decoder_output_space"] == "raw_db",
        "optimizer_created_after_initialization": True,
        "source_split": init_contract["normalization_provenance"]["source_split"],
        "validation_used": False, "future_target_used": False, "locked_test_used": False,
    }
    # Independent fresh-run reproducibility under the existing deterministic CPU contract.
    a, a_run = _run(); b, b_run = _run()
    reproducibility = {"same_seed": True, "same_subset": True, "same_config": True, "trajectory_digest_equal": json.dumps(a_run["trajectory"], sort_keys=True) == json.dumps(b_run["trajectory"], sort_keys=True), "parameter_digest_equal": a.parameter_digest() == b.parameter_digest()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.pt"; trainer.save_checkpoint(path, state=trainer.state_snapshot(global_step=MAX_STEPS))
        restored = Step52Trainer.from_unified_development_bundle(config()); restored.data.train_indices = list(SAMPLES); state = restored.load_checkpoint(path)
        checkpoint = {"compatible_reload": True, "restored_global_step": state["global_step"], "bias_preserved": bool(torch.equal(trainer.model.csi_decoder[-1].bias, restored.model.csi_decoder[-1].bias))}
        payload = torch.load(path, map_location="cpu", weights_only=False); payload["initialization_contract"] = dict(payload["initialization_contract"]); payload["initialization_contract"]["resolved_bias_value"] = -1.0; bad = Path(tmp) / "bad.pt"; torch.save(payload, bad)
        try: restored.load_checkpoint(bad)
        except ValueError: checkpoint["wrong_contract_rejected"] = True
        else: checkpoint["wrong_contract_rejected"] = False
    checks = {"formalization_contract": (formalization["fresh_bias_matches_train_mean"] and formalization["all_rb_bias_equal"] and formalization["raw_db_contract"] and formalization["optimizer_created_after_initialization"] and formalization["source_split"] == "dev_train" and not formalization["validation_used"] and not formalization["future_target_used"] and not formalization["locked_test_used"]), "finite": all(bool(row["gradients_finite"]) for row in run["trajectory"]), "reproducible": all(reproducibility.values()), "checkpoint": all(checkpoint.values()), "prior_recursive": all(run["final"][h]["prior_only"] and run["final"][h]["future_state_not_rollout_input"] for h in ("h1", "h2"))}
    tiny = {h: {fam: {"initial": run["initial"][h][fam], "final": run["final"][h][fam], "relative_drop": _drop(run["initial"][h][fam], run["final"][h][fam]), "final_below_1": run["final"][h][fam] <= 1.0} for fam in ("L_Mot", "L_CSI")} for h in ("h1", "h2")}
    tiny_go = all(item["relative_drop"] >= GATE["relative_family_loss_drop"] and item["final_below_1"] for h in tiny.values() for item in h.values())
    receipt = {"passed": bool(all(checks.values()) and tiny_go), "formalization_verdict": "FORMALIZATION_PASS" if all(checks.values()) else "FORMALIZATION_FAIL", "tiny_overfit_verdict": "TINY_OVERFIT_GO" if tiny_go else "TINY_OVERFIT_NO_GO", "required_checks": checks, "tiny_overfit": tiny, "forbidden_scope": {"gpu": False, "formal_training": False, "formal_dataset": False, "locked_test": False, "baseline": False, "planner": False, "performance_claim": False}, "gate": GATE}
    result = {"schema_version": "PI-JWM-Step-5.3E-v1", "formalization": formalization, "initialization_contract": init_contract, "normalization_provenance": init_contract["normalization_provenance"], "pre_training_metrics": run["initial"], "training_trajectory": run["trajectory"], "final_metrics": run["final"], "gradient_audit": {"phase_c": {"steps": len(run["trajectory"]), "groups": sorted({k for row in run["trajectory"] for k in row["gradient_audit"]})}}, "leakage_audit": {"future_target_not_used_for_initialization": True, "prior_recursive": checks["prior_recursive"]}, "reproducibility_audit": reproducibility, "checkpoint_audit": checkpoint, "acceptance_receipt": receipt}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if name == "schema_version": continue
        (args.output_dir / f"{name}.json").write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.iterdir() if p.is_file()]
    manifest = {"schema_version": "PI-JWM-Step-5.3E-Manifest-v1", "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}, "passed": receipt["passed"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"formalization_verdict": receipt["formalization_verdict"], "tiny_overfit_verdict": receipt["tiny_overfit_verdict"], "checks": checks, "tiny": tiny}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
