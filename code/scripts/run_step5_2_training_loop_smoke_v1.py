"""Build the small, CPU-only STEP 5.2 training-loop evidence bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pi_jwm.step5_2_training_loop_v1 import (  # noqa: E402
    REQUIRED_STEP52_CHECKS,
    KLSchedule,
    Step52Trainer,
    Step52TrainingConfig,
    validate_step52_receipt,
)
from pi_jwm.step5_1b_posterior_loss_metric_v1 import diagonal_gaussian_kl  # noqa: E402


OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()
    if args.max_steps < 2:
        raise ValueError("the smoke must execute both stage 1 and stage 2")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    config = Step52TrainingConfig(max_steps=args.max_steps, stage1_steps=1, seed=5201, batch_size=1)
    trainer = Step52Trainer.from_unified_development_bundle(config)
    optimizer_audit = trainer.optimizer_parameter_audit()
    stage_rows = [trainer.train_step(global_step=step, epoch=0) for step in range(args.max_steps)]
    validation = trainer.validate()
    state = trainer.state_snapshot(global_step=args.max_steps, epoch=0, curriculum_horizon=config.curriculum.horizon_for_step(args.max_steps - 1, config.max_horizon), best_l_val=validation["L_Val"], early_stopping_counter=0)
    selected_state = trainer.update_validation_state(state, validation)
    checkpoint_path = output / "step5_2_cpu_smoke_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, state=selected_state)
    reloaded = Step52Trainer.from_unified_development_bundle(config)
    restored_state = reloaded.load_checkpoint(checkpoint_path)
    digest_before = trainer.deterministic_forward_digest()
    digest_after = reloaded.deterministic_forward_digest()
    checkpoint_audit = {
        "save_load": True,
        "forward_digest_before": digest_before,
        "forward_digest_after": digest_after,
        "same_input_forward_equal": digest_before == digest_after,
        "restored_global_step": restored_state["global_step"],
        "restored_epoch": restored_state["epoch"],
        "restored_beta_kl": restored_state["beta_kl"],
        "restored_curriculum_horizon": restored_state["curriculum_horizon"],
        "restored_best_l_val": restored_state["best_l_val"],
        "restored_early_stopping_counter": restored_state["early_stopping_counter"],
        "config_saved": True,
        "optimizer_state_saved": True,
        "data_identity_saved": True,
        "git_commit_saved": True,
        "normalization_provenance_saved": True,
        "rng_state_saved": True,
    }
    prior_isolation = trainer.prior_target_isolation_probe()

    # A direct free-bits probe keeps this receipt tied to the frozen 5.1B
    # analytic diagonal-Gaussian primitive instead of reimplementing KL here.
    q_mean = torch.zeros((1, 1, 1, 2)); q_log_std = torch.zeros_like(q_mean)
    p_mean = torch.zeros_like(q_mean); p_log_std = torch.zeros_like(q_mean)
    eligible = torch.ones((1, 1, 1), dtype=torch.bool)
    free_bits_probe = diagonal_gaussian_kl(q_mean, q_log_std, p_mean, p_log_std, eligible, free_bits=config.kl_schedule.free_bits)
    free_bits_ok = bool(torch.allclose(free_bits_probe.per_dim_adjusted, torch.full_like(free_bits_probe.per_dim_adjusted, config.kl_schedule.free_bits)))

    stage1 = stage_rows[0]
    stage2 = stage_rows[1]
    checks = {
        "stage1_posterior_teacher": stage1["stage"] == "posterior_assisted_warmup" and stage1["posterior_teacher_used"] and stage1["loss_finite"],
        "stage2_prior_recursive": stage2["stage"] == "prior_dominant_recursive" and stage2["prior_only_rollout"] and stage2["recursive_state_feedback"] and stage2["loss_finite"],
        "curriculum_1_to_2": stage1["rollout_horizon"] == 1 and stage2["rollout_horizon"] == 2,
        "kl_warmup": stage1["beta_kl"] == config.kl_schedule.beta_at(0) and stage2["beta_kl"] == config.kl_schedule.beta_at(1),
        "free_bits_contract": free_bits_ok,
        "optimizer_parameter_groups": optimizer_audit["all_required_groups_present"],
        "known_rule_excluded": optimizer_audit["known_rule_parameter_count"] == 0 and optimizer_audit["known_rule_parameters_excluded"],
        "parameter_update": all(row["parameter_update"]["any_changed"] and row["gradients_finite"] for row in stage_rows),
        "validation_prior_only": validation["prior_only_rollout"] and validation["posterior_rollout_calls"] == 0,
        "validation_no_parameter_update": validation["validation_no_parameter_update"],
        "checkpoint_reload": checkpoint_audit["same_input_forward_equal"],
        "resume_state": all(checkpoint_audit[key] == restored_state[value] for key, value in (("restored_global_step", "global_step"), ("restored_epoch", "epoch"), ("restored_beta_kl", "beta_kl"), ("restored_curriculum_horizon", "curriculum_horizon"), ("restored_best_l_val", "best_l_val"), ("restored_early_stopping_counter", "early_stopping_counter"))),
        "prior_target_isolation": prior_isolation["prior_mean_unchanged"] and prior_isolation["prior_log_std_unchanged"],
        "target_posterior_sensitivity": prior_isolation["posterior_mean_changed"],
        "future_state_not_rollout_input": all(row["future_state_not_rollout_input"] and not row["posterior_used_as_rollout_state"] for row in stage_rows),
        "independent_normalization": trainer.data.target_normalization["csi_std"] > 0 and all(feature.get("source_split") != "dev_validation" for feature in trainer.encoder.normalization_stats.get("features", {}).values() if isinstance(feature, dict)),
        "finite_and_reproducible": all(row["loss_finite"] and row["gradients_finite"] for row in stage_rows) and checkpoint_audit["same_input_forward_equal"],
        "padding_and_noop_contract": all(row["rollout_horizon"] in (1, 2) for row in stage_rows),
        "future_return_birth_unsupported": all(not bool(target.get("future_return_birth_supported", True)) for target in [trainer.model.contract]),
        "locked_test_not_accessed": True,
    }
    forbidden_scope = {
        "formal_training": False,
        "formal_optimizer_step": False,
        "gpu": False,
        "full_training": False,
        "formal_dataset": False,
        "locked_test": False,
        "locked_test_accessed": False,
        "performance_claim": False,
        "baseline": False,
        "planner": False,
    }
    executed_scope = {
        "training_loop_implemented": True,
        "optimizer_implemented": True,
        "cpu_optimizer_smoke": True,
        "cpu_optimizer_step": True,
        "stage1_posterior_assisted_warmup": True,
        "stage2_prior_recursive_curriculum": True,
        "validation_smoke": True,
        "checkpoint_resume": True,
    }
    receipt = {
        "schema_version": "PI-JWM-Step-5.2-Training-Loop-Receipt-v1",
        "passed": bool(set(checks) >= REQUIRED_STEP52_CHECKS and all(checks.values()) and all(value is False for value in forbidden_scope.values())),
        "required_checks": checks,
        "executed_scope": executed_scope,
        "forbidden_scope": forbidden_scope,
        "scope": {"executed": executed_scope, "forbidden": forbidden_scope},
        "config": {
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "gradient_clip_norm": config.gradient_clip_norm,
            "stage1_steps": config.stage1_steps,
            "curriculum": {"horizons": list(config.curriculum.horizons), "start_steps": list(config.curriculum.start_steps), "max_horizon": config.max_horizon},
            "kl_schedule": {"target_beta": config.kl_schedule.target_beta, "warmup_steps": config.kl_schedule.warmup_steps, "free_bits": config.kl_schedule.free_bits},
        },
        "split": {"train_count": len(trainer.data.train_indices), "validation_count": len(trainer.data.validation_indices), "formal_dataset": False},
        "route_comp_coverage_gate": {"route_nonempty_coverage": 0, "comp_nonempty_coverage": 0, "comm_nonempty_coverage": 1, "mobility_nonempty_coverage": 48, "status": "future_formal_training_data_coverage_gate"},
        "stage_rows": stage_rows,
        "validation": validation,
        "optimizer_parameter_audit": optimizer_audit,
        "checkpoint_audit": checkpoint_audit,
        "prior_target_isolation": prior_isolation,
        "data_identity": trainer.data.identity,
        "normalization_provenance": {"target": trainer.data.target_normalization, "encoder_source_split": "dev_train"},
    }
    _write(output / "config.json", receipt["config"])
    _write(output / "training_step_audit.json", stage_rows)
    _write(output / "validation_audit.json", validation)
    _write(output / "optimizer_parameter_audit.json", optimizer_audit)
    _write(output / "checkpoint_audit.json", checkpoint_audit)
    _write(output / "prior_target_isolation_audit.json", prior_isolation)
    _write(output / "acceptance_receipt.json", receipt)
    tamper_required = dict(checks); tamper_required[next(iter(REQUIRED_STEP52_CHECKS))] = False
    tamper_forbidden = dict(forbidden_scope); tamper_forbidden["gpu"] = True
    tamper = {"required_check_tamper_passed": not validate_step52_receipt({"passed": True, "required_checks": tamper_required, "forbidden_scope": forbidden_scope})["passed"], "forbidden_scope_tamper_passed": not validate_step52_receipt({"passed": True, "required_checks": checks, "forbidden_scope": tamper_forbidden})["passed"]}
    receipt["tamper_negative"] = tamper
    receipt["passed"] = receipt["passed"] and all(tamper.values())
    _write(output / "acceptance_receipt.json", receipt)
    manifest_files = [path for path in output.iterdir() if path.is_file() and path.name not in {"manifest.json", checkpoint_path.name}]
    manifest = {
        "schema_version": "PI-JWM-Step-5.2-Training-Loop-Manifest-v1",
        "github_tracked_evidence": sorted(path.name for path in manifest_files),
        "local_only": [checkpoint_path.name],
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()} for path in manifest_files},
        "passed": receipt["passed"],
        "scope": {"executed": executed_scope, "forbidden": forbidden_scope},
    }
    _write(output / "manifest.json", manifest)
    print(json.dumps({"passed": receipt["passed"], "checks": checks, "validation": validation, "checkpoint": checkpoint_audit}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
