"""Formal v1 runner orchestration around the frozen FullFormalTrainer."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch

from pi_jwm.step5_2_training_loop_v1 import Step52Trainer, aggregate_validation_horizon_rows
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_config_artifact(path: Path) -> str:
    """Require byte identity with the accepted freeze receipt, not just equal fields."""
    digest = sha256(path)
    receipt = json.loads((path.parent / "config_freeze_receipt.json").read_text(encoding="utf-8"))
    if not receipt.get("passed") or receipt.get("config_sha256") != digest:
        raise ValueError("formal config byte SHA differs from accepted freeze receipt")
    return digest


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_seconds(completed_steps: int, validation_count: int) -> dict[str, float]:
    # H1/H2 rates are scaled estimates. Only the H4 and validation inputs were measured.
    h4_step = 25.026449158787727
    validation = 7103.09
    train = 1104 * (h4_step / 4) + 1104 * (h4_step / 2) + 3312 * h4_step
    total = train + 5 * validation
    elapsed_nominal = min(completed_steps, 1104) * h4_step / 4
    elapsed_nominal += max(0, min(completed_steps, 2208) - 1104) * h4_step / 2
    elapsed_nominal += max(0, completed_steps - 2208) * h4_step
    elapsed_nominal += validation_count * validation
    return {"training_estimate_seconds": train, "validation_estimate_seconds": 5 * validation,
            "total_estimate_seconds": total, "remaining_estimate_seconds": max(0.0, total - elapsed_nominal)}


def validate_full(trainer: FullFormalTrainer, heartbeat) -> dict[str, Any]:
    """Reuse the Step 5.2 prior-only batch path and its frozen loss aggregation."""
    trainer.eval()
    trainer.posterior_teacher_calls = trainer.posterior_rollout_calls = 0
    trainer.future_target_encoder_calls = trainer.current_posterior_calls = 0
    before = trainer.parameter_digest()
    rows: dict[int, list[dict[str, Any]]] = {}
    raw: dict[str, dict[int, dict[str, float]]] = {"motion": {}, "csi": {}}
    seen: set[str] = set()
    started = time.perf_counter()
    indices = trainer.data.validation_indices
    with torch.no_grad():
        for offset in range(0, len(indices), trainer.config.batch_size):
            batch_indices = indices[offset:offset + trainer.config.batch_size]
            batch = trainer.shards.load_batch(batch_indices)
            original = trainer.data
            try:
                trainer.data = batch
                trainer._move_data_to_device()
                result = Step52Trainer._run_batch(trainer, list(range(len(batch_indices))), 4,
                                                  stage="prior_dominant_recursive", beta_kl=0.0, training=False)
                if result["posterior_teacher_used"]:
                    raise RuntimeError("future posterior teacher used in validation")
                for row in result["horizon_rows"]:
                    rows.setdefault(int(row["horizon"]), []).append(row)
                norm = batch.target_normalization
                motion = batch.target_tensors["target_vehicle_motion_normalized"].clone()
                motion[..., :3] *= motion.new_tensor(norm["position_std"])
                motion[..., 3] = motion[..., 3] * float(norm["speed_std"]) + float(norm["speed_mean"])
                csi = batch.target_tensors["target_comm_csi_normalized"] * float(norm["csi_std"]) + float(norm["csi_mean"])
                for family, prediction, target, mask in (
                    ("motion", result["motion_predictions"], motion, result["motion_mask"]),
                    ("csi", result["csi_predictions"], csi, result["csi_mask"]),
                ):
                    for h in range(4):
                        diff = (prediction[:, h] - target[:, h])[mask[:, h]]
                        item = raw[family].setdefault(h + 1, {"sum_abs": 0.0, "sum_squared": 0.0, "count": 0})
                        item["sum_abs"] += float(diff.abs().sum())
                        item["sum_squared"] += float(diff.square().sum())
                        item["count"] += int(diff.numel())
                for index in batch_indices:
                    sample_id = trainer.shards.index[index]["metadata"]["sample_id"]
                    if sample_id in seen:
                        raise RuntimeError("duplicate validation window")
                    seen.add(sample_id)
            finally:
                trainer.data = original
            heartbeat(len(seen))
    torch.cuda.synchronize()
    per_horizon = aggregate_validation_horizon_rows(rows)
    l_val = float(np.mean([row["L_Pred"] for row in per_horizon]))
    metrics = {family: [
        {"horizon": h, "raw_mae": value["sum_abs"] / value["count"],
         "raw_rmse": math.sqrt(value["sum_squared"] / value["count"]), "valid_element_count": value["count"]}
        for h, value in sorted(horizons.items())] for family, horizons in raw.items()}
    expected = {trainer.shards.index[i]["metadata"]["sample_id"] for i in indices}
    if (seen != expected or len(seen) != 1104 or before != trainer.parameter_digest()
            or trainer.posterior_teacher_calls or trainer.future_target_encoder_calls or trainer.posterior_rollout_calls
            or not math.isfinite(l_val) or [row["horizon"] for row in per_horizon] != [1, 2, 3, 4]
            or any(len(values) != 4 or any(v["valid_element_count"] <= 0 or not math.isfinite(v["raw_mae"]) or not math.isfinite(v["raw_rmse"]) for v in values) for values in metrics.values())):
        raise RuntimeError("full prior-only validation invariant failed")
    trainer.train()
    return {"L_Val": l_val, "per_horizon": per_horizon, "raw_metrics": metrics,
            "validation_windows": len(seen), "prior_only": True, "parameter_unchanged": True,
            "future_posterior_teacher_calls": 0, "future_target_encoder_calls": 0,
            "wall_seconds": time.perf_counter() - started}


def run(manifest: Path, config_path: Path, run_dir: Path, source_sha: str, resume: Path | None = None) -> None:
    if resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        (run_dir / "train_metrics.jsonl").touch(exist_ok=False)
        (run_dir / "validation_metrics.jsonl").touch(exist_ok=False)
    elif not run_dir.is_dir() or not resume.is_file() or resume.parent != run_dir / "checkpoints":
        raise ValueError("resume must use an existing checkpoint inside the same run directory")
    start = time.time()
    config_sha = verify_frozen_config_artifact(config_path)
    protocol = formal_training_config_v1(manifest)
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    canonical = json.loads(json.dumps(protocol.as_manifest()))
    if any(frozen.get(key) != value for key, value in canonical.items()):
        raise ValueError("formal config artifact differs from programmatic frozen config")
    interface = FormalTrainingInterface.from_manifest(manifest)
    package_check = interface.verify_packages()
    if not package_check["all_present_and_matching"]:
        raise ValueError("Formal Dataset package hash mismatch")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("frozen run requires available RTX 4090")
    trainer = FullFormalTrainer.from_interface(interface, protocol.training)
    if trainer.optimizer.param_groups[0]["betas"] != (0.9, 0.999) or trainer.optimizer.param_groups[0]["eps"] != 1e-8:
        raise RuntimeError("effective AdamW defaults differ from frozen config")
    run_id = run_dir.name
    run_manifest = {"run_id": run_id, "start_time": now(), "git_commit": source_sha,
                "dataset_id": protocol.dataset_id, "dataset_manifest_sha256": interface.dataset_manifest_hash,
                "formal_config_sha256": config_sha, "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda, "pytorch": torch.__version__, "formal_training": True,
                "locked_test_accessed": False, "baseline": False, "planner": False,
                "package_hashes": {name: item["sha256"] for name, item in package_check.items() if name != "all_present_and_matching"}}
    if resume is None:
        atomic_json(run_dir / "run_manifest.json", run_manifest)
    else:
        prior = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if any(prior.get(key) != run_manifest[key] for key in ("run_id", "git_commit", "dataset_id", "dataset_manifest_sha256", "formal_config_sha256", "package_hashes")):
            raise ValueError("resume run manifest identity mismatch")
    state = trainer.state_snapshot()
    completed = 0
    validation_count = 0
    last_train: dict[str, Any] = {}
    last_validation: float | None = None
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        prior_state = payload.get("state", {})
        if (prior_state.get("formal_config_sha256") != config_sha or prior_state.get("source_git_sha") != source_sha
                or payload.get("data_identity") != trainer.data.identity):
            raise ValueError("resume config, source, or dataset identity mismatch")
        state = trainer.load_checkpoint(resume)
        completed = int(state["global_step"])
        validation_count = int(state["validation_count"])
        last_validation = state.get("last_validation_l_val")

    def heartbeat(status: str, validated: int = 0, nan_or_inf: bool = False) -> None:
        future_step = min(completed, protocol.training.max_steps - 1)
        remaining = estimate_seconds(completed, validation_count)["remaining_estimate_seconds"]
        value = {"timestamp": now(), "process_alive": status in ("RUNNING", "VALIDATING"), "pid": os.getpid(), "status": status,
                 "global_step": completed, "completed_steps": completed, "max_steps": protocol.training.max_steps,
                 "epoch": completed // protocol.steps_per_epoch, "max_epochs": protocol.training.max_epochs,
                 "stage": protocol.stage_for_step(future_step), "rollout_horizon": protocol.horizon_for_step(future_step),
                 "beta_kl": protocol.training.kl_schedule.beta_at(future_step),
                 "last_train_L_Total": last_train.get("L_Total"), "last_L_Mot": last_train.get("L_Mot"),
                 "last_L_CSI": last_train.get("L_CSI"), "last_L_KL": last_train.get("L_KL"),
                 "last_gradient_norm": last_train.get("gradient_norm_before_clip"),
                 "best_L_Val": None if math.isinf(float(state.get("best_l_val", float("inf")))) else state["best_l_val"],
                 "last_validation_L_Val": last_validation, "early_stopping_counter": state.get("early_stopping_counter", 0),
                 "elapsed_seconds": time.time() - start, "estimated_remaining_seconds": remaining,
                 "estimated_finish_time": (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(),
                 "latest_checkpoint": str(run_dir / "checkpoints/latest.pt") if (run_dir / "checkpoints/latest.pt").exists() else None,
                 "best_checkpoint": str(run_dir / "checkpoints/best.pt") if (run_dir / "checkpoints/best.pt").exists() else None,
                 "gpu_memory_allocated": torch.cuda.memory_allocated(), "gpu_memory_reserved": torch.cuda.memory_reserved(),
                 "nan_or_inf_detected": nan_or_inf, "validated_windows_current_pass": validated}
        atomic_json(run_dir / "heartbeat.json", value)
        atomic_json(run_dir / "progress.json", value)

    def checkpoint(name: str) -> None:
        path = run_dir / "checkpoints" / name
        temporary = path.with_name(path.name + ".tmp")
        trainer.save_checkpoint(temporary, state={**state, "global_step": completed,
                                "epoch": completed // protocol.steps_per_epoch, "validation_count": validation_count,
                                "formal_config_sha256": config_sha, "source_git_sha": source_sha})
        # A portable source archive need not contain .git. Bind the verified
        # source commit explicitly in the checkpoint as well as its run state.
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        payload["git_commit"] = source_sha
        torch.save(payload, temporary)
        os.replace(temporary, path)

    try:
        heartbeat("RUNNING")
        while completed < protocol.training.max_steps:
            step = completed
            result = trainer.train_step(global_step=step, epoch=step // protocol.steps_per_epoch)
            if not result["loss_finite"] or not result["gradients_finite"] or not math.isfinite(result["gradient_norm_before_clip"]):
                raise FloatingPointError("nonfinite formal optimizer step")
            completed += 1
            previous_state = state
            state = trainer.state_snapshot(global_step=min(completed, protocol.training.max_steps - 1),
                                           epoch=completed // protocol.steps_per_epoch,
                                           best_l_val=previous_state.get("best_l_val", float("inf")),
                                           early_stopping_counter=previous_state.get("early_stopping_counter", 0))
            state.update({key: previous_state[key] for key in ("last_validation_l_val", "selector_metric", "selector_uses_l_val_only") if key in previous_state})
            last_train = result
            row = {key: result[key] for key in ("global_step", "epoch", "rollout_horizon", "L_Total", "L_Mot", "L_CSI", "L_KL", "L_KL_Phy_raw", "L_KL_Comm_raw", "gradient_norm_before_clip")}
            row.update({"run_id": run_id, "completed_steps": completed, "stage": protocol.stage_for_step(step),
                        "beta_kl": protocol.training.kl_schedule.beta_at(step), "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": time.time() - start, "estimated_remaining_seconds": estimate_seconds(completed, validation_count)["remaining_estimate_seconds"]})
            append_jsonl(run_dir / "train_metrics.jsonl", row)
            heartbeat("RUNNING")
            if protocol.latest_checkpoint_due(completed):
                checkpoint("latest.pt")
                heartbeat("RUNNING")
            if protocol.validation_due(completed):
                heartbeat("VALIDATING")
                report = validate_full(trainer, lambda count: heartbeat("VALIDATING", count))
                validation_count += 1
                report.update({"run_id": run_id, "completed_steps": completed, "timestamp": now()})
                previous = float(state.get("best_l_val", float("inf")))
                state = trainer.update_validation_state(state, report)
                last_validation = report["L_Val"]
                append_jsonl(run_dir / "validation_metrics.jsonl", report)
                if report["L_Val"] < previous:
                    checkpoint("best.pt")
                checkpoint("latest.pt")
                heartbeat("RUNNING")
                if state["early_stopping_should_stop"]:
                    heartbeat("EARLY_STOPPED")
                    return
        heartbeat("COMPLETED")
    except BaseException as error:
        atomic_json(run_dir / "failure_receipt.json", {"timestamp": now(), "run_id": run_id,
                    "completed_steps": completed, "error_type": type(error).__name__, "error": str(error),
                    "latest_checkpoint_preserved": (run_dir / "checkpoints/latest.pt").exists()})
        heartbeat("FAILED", nan_or_inf=isinstance(error, FloatingPointError))
        raise
