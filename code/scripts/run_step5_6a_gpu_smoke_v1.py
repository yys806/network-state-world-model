"""STEP 5.6A CUDA smoke and full prior-only validation; never formal training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from pi_jwm.step5_2_training_loop_v1 import (
    CurriculumConfig, Step52Trainer, Step52TrainingConfig,
    aggregate_validation_horizon_rows,
)
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(batch_size: int) -> Step52TrainingConfig:
    # These are smoke-only engineering settings, not a formal training config.
    return Step52TrainingConfig(
        seed=5601, batch_size=batch_size, max_horizon=4, max_steps=4,
        max_epochs=1, stage1_steps=0,
        curriculum=CurriculumConfig(horizons=(4,), start_steps=(0,)), device="cuda",
    )


def _base(manifest: Path) -> tuple[FormalTrainingInterface, FullFormalTrainer]:
    print(json.dumps({"init": "interface_start"}), flush=True)
    interface = FormalTrainingInterface.from_manifest(manifest)
    print(json.dumps({"init": "interface_done"}), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    print(json.dumps({"init": "trainer_start"}), flush=True)
    trainer = FullFormalTrainer.from_interface(interface, _config(1))
    print(json.dumps({"init": "trainer_done"}), flush=True)
    return interface, trainer


def smoke(manifest: Path, output: Path) -> dict[str, Any]:
    interface, trainer = _base(manifest)
    environment = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    original_loader = trainer.shards.load_batch
    timing: dict[str, float] = {}
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed_loader(indices: Any) -> Any:
        began = time.perf_counter()
        value = original_loader(indices)
        timing["data_load_seconds"] = time.perf_counter() - began
        start_event.record()
        return value

    trainer.shards.load_batch = timed_loader
    results: list[dict[str, Any]] = []
    for step, batch_size in enumerate((1, 2, 4, 8)):
        trainer.config = replace(trainer.config, batch_size=batch_size)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = trainer.train_step(global_step=step, epoch=0)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "batch_size": batch_size, "success": True,
                "sample_ids": result.get("sample_ids"),
                "horizon": result["rollout_horizon"],
                "loss_finite": bool(result["loss_finite"]),
                "gradients_finite": bool(result["gradients_finite"]),
                "parameter_updated": bool(result["parameter_update"]["any_changed"]),
                "step_wall_seconds": elapsed,
                "data_load_seconds": timing["data_load_seconds"],
                "gpu_event_seconds": start_event.elapsed_time(end_event) / 1000.0,
                "samples_per_second": batch_size / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        except torch.cuda.OutOfMemoryError as exc:
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            row = {"batch_size": batch_size, "success": False, "oom": True, "error": str(exc)[:500]}
        results.append(row)
        _write(output, {"status": "partial_probe", "environment": environment, "batch_probe": results})
        print(json.dumps({"probe": row}, default=str), flush=True)
        if not row["success"]:
            break

    # Cross-trajectory H4 batch, separate from the trajectory-local sampler.
    cross = [interface.train_indices[0], interface.train_indices[92]]
    trainer.config = replace(trainer.config, batch_size=2)
    trainer.optimizer.zero_grad(set_to_none=True)
    cross_result = trainer._run_batch(cross, 4, stage="prior_dominant_recursive", beta_kl=0.0, training=True)
    cross_result["L_Total"].backward()
    cross_grad_finite = trainer._finite_gradients(p for g in trainer._optimizer_groups for p in g["params"])
    before = trainer.parameter_digest()
    trainer.optimizer.step()
    cross_changed = before != trainer.parameter_digest()
    torch.cuda.synchronize()
    groups_cuda = {g["name"]: all(p.device.type == "cuda" for p in g["params"]) for g in trainer._optimizer_groups}
    checkpoint = output.parent / "gpu_smoke_checkpoint.pt"
    state = trainer.state_snapshot(global_step=5, epoch=0, curriculum_horizon=4)
    trainer.save_checkpoint(checkpoint, state=state)
    before_reload = trainer.parameter_digest()
    loaded = trainer.load_checkpoint(checkpoint)
    reload_identical = before_reload == trainer.parameter_digest() and loaded == state
    original_identity = trainer.data.identity
    trainer.data.identity = {**original_identity, "dataset_id": "wrong-dataset"}
    try:
        try:
            trainer.load_checkpoint(checkpoint)
        except ValueError as exc:
            wrong_dataset_rejected = "data identity" in str(exc)
        else:
            wrong_dataset_rejected = False
    finally:
        trainer.data.identity = original_identity
    original_config = trainer.config
    trainer.config = replace(original_config, learning_rate=original_config.learning_rate * 2)
    try:
        try:
            trainer.load_checkpoint(checkpoint)
        except ValueError as exc:
            wrong_config_rejected = "training config" in str(exc)
        else:
            wrong_config_rejected = False
    finally:
        trainer.config = original_config
    receipt = {
        "schema_version": "PI-JWM-STEP-5.6A-GPU-Smoke-v1",
        "environment": environment, "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "dataset_identity": trainer.data.identity, "batch_probe": results,
        "cross_trajectory_batch": {
            "trajectory_ids": [trainer.shards.index[i]["metadata"]["trajectory_id"] for i in cross],
            "h4": len(cross_result["horizon_rows"]) == 4,
            "loss_finite": bool(torch.isfinite(cross_result["L_Total"])),
            "gradients_finite": cross_grad_finite, "parameter_updated": cross_changed,
        },
        "optimizer_groups_cuda": groups_cuda,
        "checkpoint_reload_identical": reload_identical,
        "wrong_dataset_rejected": wrong_dataset_rejected,
        "wrong_config_rejected": wrong_config_rejected,
        "scope": {"gpu_smoke": True, "formal_training": False, "locked_test_accessed": False,
                  "baseline": False, "planner": False, "performance_claim": False},
    }
    required_probe = [r for r in results if r["batch_size"] in (1, 2, 4)]
    optional_probe = [r for r in results if r["batch_size"] == 8]
    receipt["passed"] = bool(
        len(required_probe) == 3
        and all(r["success"] and r["loss_finite"] and r["gradients_finite"] and r["parameter_updated"] for r in required_probe)
        and len(optional_probe) == 1
        and (optional_probe[0]["success"] or optional_probe[0].get("oom") is True)
        and all(groups_cuda.values()) and reload_identical
        and wrong_dataset_rejected and wrong_config_rejected
        and all(receipt["cross_trajectory_batch"][k] for k in ("h4", "loss_finite", "gradients_finite", "parameter_updated"))
        and len(set(receipt["cross_trajectory_batch"]["trajectory_ids"])) == 2
    )
    _write(output, receipt)
    return receipt


def validate(manifest: Path, checkpoint: Path, output: Path, batch_size: int, *, shard_id: int = 0, shard_count: int = 1) -> dict[str, Any]:
    interface, trainer = _base(manifest)
    if shard_count <= 0 or not 0 <= shard_id < shard_count:
        raise ValueError("invalid validation trajectory shard")
    # The smoke checkpoint was written with batch=2. Verify exact config
    # identity first; traversal batch size is a validation-only runtime choice.
    trainer.config = replace(trainer.config, batch_size=2)
    trainer.load_checkpoint(checkpoint)
    trainer.config = replace(trainer.config, batch_size=batch_size)
    trainer.eval()
    trainer.posterior_teacher_calls = trainer.posterior_rollout_calls = 0
    trainer.future_target_encoder_calls = trainer.current_posterior_calls = 0
    before = trainer.parameter_digest()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows: dict[int, list[dict[str, Any]]] = {}
    raw: dict[str, dict[int, dict[str, float]]] = {"motion": {}, "csi": {}}
    seen: set[str] = set()
    trajectory_order = list(dict.fromkeys(interface.samples[i]["metadata"]["trajectory_id"] for i in interface.validation_indices))
    selected_trajectories = set(trajectory_order[shard_id::shard_count])
    indices = tuple(i for i in interface.validation_indices if interface.samples[i]["metadata"]["trajectory_id"] in selected_trajectories)
    data_seconds = 0.0
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            global_indices = indices[offset:offset + batch_size]
            batch_started = time.perf_counter()
            print(json.dumps({"batch_start": int(offset), "batch_size": len(global_indices)}), flush=True)
            began = time.perf_counter()
            batch = trainer.shards.load_batch(global_indices)
            load_elapsed = time.perf_counter() - began
            data_seconds += load_elapsed
            print(json.dumps({"batch_loaded": int(offset), "load_seconds": load_elapsed}), flush=True)
            original = trainer.data
            try:
                trainer.data = batch
                trainer._move_data_to_device()
                result = Step52Trainer._run_batch(
                    trainer, list(range(len(global_indices))), 4,
                    stage="prior_dominant_recursive", beta_kl=0.0, training=False,
                )
                if result["posterior_teacher_used"]:
                    raise AssertionError("validation used future posterior teacher")
                for row in result["horizon_rows"]:
                    rows.setdefault(int(row["horizon"]), []).append(row)
                params = batch.target_normalization
                target_motion = batch.target_tensors["target_vehicle_motion_normalized"].clone()
                target_motion[..., :3] *= target_motion.new_tensor(params["position_std"])
                target_motion[..., 3] = target_motion[..., 3] * float(params["speed_std"]) + float(params["speed_mean"])
                target_csi = batch.target_tensors["target_comm_csi_normalized"] * float(params["csi_std"]) + float(params["csi_mean"])
                for family, prediction, target, mask in (
                    ("motion", result["motion_predictions"], target_motion, result["motion_mask"]),
                    ("csi", result["csi_predictions"], target_csi, result["csi_mask"]),
                ):
                    for h in range(4):
                        diff = (prediction[:, h] - target[:, h])[mask[:, h]]
                        item = raw[family].setdefault(h + 1, {"sum_abs": 0.0, "sum_squared": 0.0, "count": 0})
                        item["sum_abs"] += float(diff.abs().sum())
                        item["sum_squared"] += float(diff.square().sum())
                        item["count"] += int(diff.numel())
                for i in global_indices:
                    sample_id = trainer.shards.index[i]["metadata"]["sample_id"]
                    if sample_id in seen:
                        raise AssertionError("duplicate validation sample")
                    seen.add(sample_id)
            finally:
                trainer.data = original
            print(json.dumps({"batch_done": int(offset), "batch_seconds": time.perf_counter() - batch_started}), flush=True)
            if (offset // batch_size + 1) % 5 == 0:
                print(json.dumps({"validated": len(seen), "total": len(indices)}), flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    per_horizon = aggregate_validation_horizon_rows(rows)
    metrics = {
        family: [
            {"horizon": h, "raw_mae": sums["sum_abs"] / sums["count"],
             "raw_rmse": math.sqrt(sums["sum_squared"] / sums["count"]),
             "valid_element_count": int(sums["count"])}
            for h, sums in sorted(horizons.items())
        ] for family, horizons in raw.items()
    }
    l_val = float(np.mean([r["L_Pred"] for r in per_horizon]))
    receipt = {
        "schema_version": "PI-JWM-STEP-5.6A-Full-GPU-Validation-v1",
        "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validation_shard_id": shard_id, "validation_shard_count": shard_count,
        "validation_windows": len(seen), "validation_trajectories": len(selected_trajectories),
        "sample_ids": sorted(seen), "selected_trajectory_ids": sorted(selected_trajectories),
        "prior_only": True, "horizons": [r["horizon"] for r in per_horizon],
        "future_posterior_teacher_calls": trainer.posterior_teacher_calls,
        "future_target_encoder_calls": trainer.future_target_encoder_calls,
        "posterior_rollout_calls": trainer.posterior_rollout_calls,
        "parameter_unchanged": before == trainer.parameter_digest(),
        "per_horizon_loss": per_horizon, "L_Val": l_val,
        "raw_metrics": metrics, "raw_aggregates": raw, "batch_size": batch_size,
        "wall_seconds": wall, "data_loading_seconds": data_seconds,
        "throughput_windows_per_second": len(seen) / wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": {"untrained_smoke_model": True, "formal_training": False,
                  "locked_test_accessed": False, "performance_claim": False},
    }
    receipt["passed"] = bool(
        len(seen) == len(indices) and receipt["validation_trajectories"] == len(selected_trajectories)
        and receipt["horizons"] == [1, 2, 3, 4]
        and receipt["parameter_unchanged"] and math.isfinite(l_val)
        and receipt["future_posterior_teacher_calls"] == 0
        and receipt["future_target_encoder_calls"] == 0
        and receipt["posterior_rollout_calls"] == 0
        and all(len(values) == 4 and all(x["valid_element_count"] > 0 and math.isfinite(x["raw_mae"]) and math.isfinite(x["raw_rmse"]) for x in values) for values in metrics.values())
    )
    _write(output, receipt)
    return receipt


def merge_validation(manifest: Path, parts: list[Path], output: Path) -> dict[str, Any]:
    interface = FormalTrainingInterface.from_manifest(manifest)
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    shard_count = len(receipts)
    if shard_count == 0 or {row["validation_shard_id"] for row in receipts} != set(range(shard_count)):
        raise ValueError("validation shards are incomplete or duplicated")
    expected = {interface.samples[i]["metadata"]["sample_id"] for i in interface.validation_indices}
    seen: set[str] = set()
    trajectories: set[str] = set()
    horizon_rows: dict[int, list[dict[str, Any]]] = {}
    raw: dict[str, dict[int, dict[str, float]]] = {"motion": {}, "csi": {}}
    for row in receipts:
        if not row["passed"] or row["validation_shard_count"] != shard_count or row["dataset_manifest_sha256"] != interface.dataset_manifest_hash:
            raise ValueError("validation shard failed or has incompatible identity")
        if row["checkpoint_sha256"] != receipts[0]["checkpoint_sha256"]:
            raise ValueError("validation shards used different checkpoints")
        if seen.intersection(row["sample_ids"]):
            raise ValueError("validation sample repeated across shards")
        seen.update(row["sample_ids"])
        trajectories.update(row["selected_trajectory_ids"])
        for item in row["per_horizon_loss"]:
            horizon_rows.setdefault(int(item["horizon"]), []).append(item)
        for family in raw:
            for h, values in row["raw_aggregates"][family].items():
                total = raw[family].setdefault(int(h), {"sum_abs": 0.0, "sum_squared": 0.0, "count": 0})
                for key in total:
                    total[key] += values[key]
    per_horizon = aggregate_validation_horizon_rows(horizon_rows)
    metrics = {family: [
        {"horizon": h, "raw_mae": value["sum_abs"] / value["count"],
         "raw_rmse": math.sqrt(value["sum_squared"] / value["count"]),
         "valid_element_count": int(value["count"])}
        for h, value in sorted(horizons.items())
    ] for family, horizons in raw.items()}
    # Workers may be run serially on one GPU; their sum is the measured
    # compute/traversal time. Calendar time across a user pause is excluded.
    wall = sum(row["wall_seconds"] for row in receipts)
    result = {
        "schema_version": "PI-JWM-STEP-5.6A-Merged-Full-GPU-Validation-v1",
        "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "checkpoint_sha256": receipts[0]["checkpoint_sha256"],
        "validation_windows": len(seen), "validation_trajectories": len(trajectories),
        "shards": shard_count, "batch_size_per_shard": receipts[0]["batch_size"],
        "prior_only": all(row["prior_only"] for row in receipts),
        "future_posterior_teacher_calls": sum(row["future_posterior_teacher_calls"] for row in receipts),
        "future_target_encoder_calls": sum(row["future_target_encoder_calls"] for row in receipts),
        "posterior_rollout_calls": sum(row["posterior_rollout_calls"] for row in receipts),
        "parameter_unchanged": all(row["parameter_unchanged"] for row in receipts),
        "per_horizon_loss": per_horizon,
        "L_Val": float(np.mean([item["L_Pred"] for item in per_horizon])),
        "raw_metrics": metrics,
        "validation_worker_wall_seconds_sum": wall,
        "data_loading_seconds_sum": sum(row["data_loading_seconds"] for row in receipts),
        "throughput_windows_per_second_serial": len(seen) / wall,
        "peak_allocated_bytes_per_worker": [row["peak_allocated_bytes"] for row in receipts],
        "peak_reserved_bytes_per_worker": [row["peak_reserved_bytes"] for row in receipts],
        "scope": {"untrained_smoke_model": True, "formal_training": False,
                  "locked_test_accessed": False, "performance_claim": False},
    }
    result["passed"] = bool(
        seen == expected and len(seen) == 1104 and len(trajectories) == 12
        and [item["horizon"] for item in per_horizon] == [1, 2, 3, 4]
        and result["prior_only"] and result["parameter_unchanged"]
        and result["future_posterior_teacher_calls"] == 0
        and result["future_target_encoder_calls"] == 0
        and result["posterior_rollout_calls"] == 0
        and math.isfinite(result["L_Val"])
        and all(len(values) == 4 and all(value["valid_element_count"] > 0 and math.isfinite(value["raw_mae"]) and math.isfinite(value["raw_rmse"]) for value in values) for values in metrics.values())
    )
    _write(output, result)
    return result


def reload_forward_identity(manifest: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Compare real CUDA H4 predictions across exact checkpoint reloads."""
    interface, trainer = _base(manifest)
    trainer.config = replace(trainer.config, batch_size=2)
    indices = (interface.validation_indices[0], interface.validation_indices[92])

    def run_once() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str]:
        state = trainer.load_checkpoint(checkpoint)
        trainer.eval()
        with torch.no_grad():
            result = trainer._run_batch(indices, 4, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
        return (state, result["motion_predictions"].detach().cpu().clone(),
                result["csi_predictions"].detach().cpu().clone(), trainer.parameter_digest())

    first = run_once()
    second = run_once()
    motion_delta = (first[1] - second[1]).abs()
    csi_delta = (first[2] - second[2]).abs()
    receipt = {
        "schema_version": "PI-JWM-STEP-5.6A-CUDA-Checkpoint-Forward-Identity-v1",
        "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validation_sample_ids": [interface.samples[i]["metadata"]["sample_id"] for i in indices],
        "horizon": 4,
        "device": str(trainer.device),
        "state_identical": first[0] == second[0],
        "parameter_digest_identical": first[3] == second[3],
        "motion_prediction_bitwise_identical": bool(torch.equal(first[1], second[1])),
        "csi_prediction_bitwise_identical": bool(torch.equal(first[2], second[2])),
        "motion_prediction_max_abs_delta": float(motion_delta.max()),
        "csi_prediction_max_abs_delta": float(csi_delta.max()),
        "motion_prediction_allclose_1e_6": bool(torch.allclose(first[1], second[1], rtol=1e-6, atol=1e-6)),
        "csi_prediction_allclose_1e_6": bool(torch.allclose(first[2], second[2], rtol=1e-6, atol=1e-6)),
        "future_posterior_teacher_calls": trainer.posterior_teacher_calls,
        "future_target_encoder_calls": trainer.future_target_encoder_calls,
        "scope": {"formal_training": False, "locked_test_accessed": False,
                  "performance_claim": False},
    }
    receipt["passed"] = bool(
        receipt["device"] == "cuda" and len(receipt["validation_sample_ids"]) == 2
        and receipt["state_identical"] and receipt["parameter_digest_identical"]
        and receipt["motion_prediction_allclose_1e_6"]
        and receipt["csi_prediction_allclose_1e_6"]
        and receipt["future_posterior_teacher_calls"] == 0
        and receipt["future_target_encoder_calls"] == 0
    )
    _write(output, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "validation", "merge", "reload_probe"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--partial", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.mode == "smoke":
        result = smoke(args.manifest, args.output)
    elif args.mode == "validation":
        if args.checkpoint is None:
            parser.error("validation requires --checkpoint")
        result = validate(args.manifest, args.checkpoint, args.output, args.batch_size, shard_id=args.shard_id, shard_count=args.shard_count)
    elif args.mode == "merge":
        result = merge_validation(args.manifest, args.partial, args.output)
    else:
        if args.checkpoint is None:
            parser.error("reload_probe requires --checkpoint")
        result = reload_forward_identity(args.manifest, args.checkpoint, args.output)
    print(json.dumps({"mode": args.mode, "passed": result["passed"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)
