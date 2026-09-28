"""CPU-only final acceptance for the frozen STEP 5.6B training run.

Reads local artifacts only. It never trains, contacts the server, or touches
locked_test. The bounded smoke evaluates one validation window at H1-H4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

from pi_jwm.step5_2_training_loop_v1 import Step52Trainer
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "pi_jwm_formal_train_v1_seed5601_20260924T112424Z"
SOURCE_SHA = "6e15ec2da0e3a6e0561dc821d0aaef90696a2387"
VALIDATION_STEPS = [1104, 2208, 3312, 4416, 5520]
REQUIRED_FILES = (
    "run_manifest.json", "heartbeat.json", "progress.json",
    "train_metrics.jsonl", "validation_metrics.jsonl",
    "checkpoints/best.pt", "checkpoints/latest.pt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON: {value}"))) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), f"empty JSONL: {path.name}")
    return rows


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def validate_train_rows(rows: list[dict[str, Any]], run_id: str, protocol: Any) -> None:
    require(len(rows) == 5520, "training log must contain exactly 5520 rows")
    for step, row in enumerate(rows):
        require(row["run_id"] == run_id and row["global_step"] == step
                and row["completed_steps"] == step + 1, f"training step identity mismatch at {step}")
        require(row["epoch"] == protocol.epoch_for_step(step)
                and row["stage"] == protocol.stage_for_step(step)
                and row["rollout_horizon"] == protocol.horizon_for_step(step),
                f"frozen stage/curriculum mismatch at {step}")
        require(math.isclose(row["beta_kl"], protocol.training.kl_schedule.beta_at(step), abs_tol=1e-7)
                and math.isclose(row["learning_rate"], protocol.training.learning_rate, abs_tol=1e-12),
                f"KL/learning-rate schedule mismatch at {step}")
        require(finite_tree(row) and all(math.isfinite(row[name]) for name in
                ("L_Total", "L_Mot", "L_CSI", "L_KL", "gradient_norm_before_clip")),
                f"nonfinite training metric at {step}")


def validate_validation_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    require(len(rows) == 5 and [row["completed_steps"] for row in rows] == VALIDATION_STEPS,
            "full validation steps must be exactly 1104/2208/3312/4416/5520")
    observations = []
    for row in rows:
        step = row["completed_steps"]
        require(row["run_id"] == run_id and row["validation_windows"] == 1104
                and row["prior_only"] is True and row["parameter_unchanged"] is True
                and row["future_posterior_teacher_calls"] == 0
                and row["future_target_encoder_calls"] == 0,
                f"validation provenance mismatch at {step}")
        horizons = row["per_horizon"]
        require([item["horizon"] for item in horizons] == [1, 2, 3, 4],
                f"missing validation horizon at {step}")
        for item in horizons:
            require(item["available_sample_count"] == 1104
                    and item["motion_count"] > 0 and item["csi_count"] > 0
                    and math.isclose(item["L_Pred"], 0.5 * (item["L_Mot"] + item["L_CSI"]), abs_tol=1e-9),
                    f"invalid family loss/count at {step}/H{item['horizon']}")
        recalculated = sum(item["L_Pred"] for item in horizons) / 4
        require(math.isclose(row["L_Val"], recalculated, abs_tol=1e-9)
                and finite_tree(row), f"nonfinite or inconsistent L_Val at {step}")
        raw: dict[str, list[dict[str, Any]]] = {}
        for family in ("motion", "csi"):
            values = row["raw_metrics"][family]
            require([item["horizon"] for item in values] == [1, 2, 3, 4]
                    and all(item["valid_element_count"] > 0 and item["raw_mae"] >= 0
                            and item["raw_rmse"] >= item["raw_mae"] for item in values),
                    f"invalid {family} raw metrics at {step}")
            raw[family] = [{"horizon": item["horizon"], "mae": item["raw_mae"],
                            "rmse": item["raw_rmse"], "valid_element_count": item["valid_element_count"]}
                           for item in values]
        observations.append({"completed_steps": step, "L_Val": row["L_Val"],
                             "L_Pred": [{"horizon": item["horizon"], "value": item["L_Pred"]}
                                        for item in horizons], "motion": raw["motion"], "csi": raw["csi"]})
    return observations


def checkpoint_identity(payload: dict[str, Any], run: dict[str, Any], frozen: dict[str, Any],
                        sample_index_sha256: str, best_step: int, best_l_val: float) -> None:
    state = payload["state"]
    identity = payload["data_identity"]
    require(payload["git_commit"] == run["git_commit"] == state["source_git_sha"]
            and state["formal_config_sha256"] == run["formal_config_sha256"],
            "checkpoint source/config SHA mismatch")
    require(identity["dataset_id"] == run["dataset_id"] == frozen["dataset_id"]
            and identity["dataset_manifest_hash"] == run["dataset_manifest_sha256"] == frozen["dataset_manifest_sha256"]
            and identity["formal_package_hashes"] == run["package_hashes"]
            and identity["sample_index_sha256"] == sample_index_sha256
            and identity["train_sample_count"] == 4416 and identity["validation_sample_count"] == 1104,
            "checkpoint Dataset/package/index identity mismatch")
    require(payload["config"] == frozen["training_config"]
            and payload["architecture_identity"] == {
                "rssm": frozen["training_config"]["rssm"],
                "encoder": frozen["training_config"]["encoder"]},
            "checkpoint frozen training/architecture config mismatch")
    require(state["global_step"] == best_step == 5520 and state["epoch"] == 10
            and state["validation_count"] == 5
            and math.isclose(state["best_l_val"], best_l_val, abs_tol=1e-12)
            and math.isclose(state["last_validation_l_val"], best_l_val, abs_tol=1e-12)
            and state["selector_metric"] == "L_Val" and state["selector_uses_l_val_only"] is True,
            "checkpoint final best/selector state mismatch")


def model_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, int]:
    if left.keys() != right.keys():
        return False, 0
    count = 0
    for module in left:
        if left[module].keys() != right[module].keys():
            return False, count
        for name, value in left[module].items():
            count += 1
            if not torch.equal(value, right[module][name]):
                return False, count
    return True, count


def cpu_inference_smoke(interface: FormalTrainingInterface, protocol: Any,
                        checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Translate only the runtime device to CPU; keep frozen weights/identity."""
    cpu_config = replace(protocol.training, device="cpu")
    require(payload["config"]["device"] == "cuda" and cpu_config.device == "cpu",
            "CPU replay must translate the frozen CUDA device only")
    trainer = FullFormalTrainer.from_interface(interface, cpu_config)
    require(trainer.data.identity == payload["data_identity"],
            "CPU trainer data identity differs from checkpoint")
    # FullFormalTrainer.load_checkpoint rejects the intentional device change.
    # The base loader still verifies architecture, data and normalization and
    # restores all state, after the exact frozen config was checked above.
    state = Step52Trainer.load_checkpoint(trainer, checkpoint)
    require(state["global_step"] == 5520, "CPU checkpoint reload lost final step")
    before = trainer.parameter_digest()
    selected = [interface.validation_indices[0]]
    result = trainer.validate_indices(selected)
    after = trainer.parameter_digest()
    require(result["prior_only"] is True and before == after
            and len(result["horizon_rows"]) == 4
            and [item["horizon"] for item in result["horizon_rows"]] == [1, 2, 3, 4]
            and all(finite_tree(item) for item in result["horizon_rows"])
            and trainer.posterior_teacher_calls == 0
            and trainer.future_target_encoder_calls == 0,
            "bounded CPU prior-only inference failed")
    return {"device": "cpu", "window_count": 1, "sample_id": result["sample_ids"][0],
            "horizons": [1, 2, 3, 4], "prior_only": True,
            "future_posterior_teacher_calls": 0, "future_target_encoder_calls": 0,
            "model_parameters_unchanged": True,
            "checkpoint_reloaded_with_device_only_translation": True}


def accept(run_dir: Path, dataset_manifest: Path, config_path: Path,
           output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    dataset_manifest = dataset_manifest.resolve()
    config_path = config_path.resolve()
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    require(not missing, f"missing final run files: {missing}")
    require(not list(run_dir.glob("*failure*")), "run directory contains a failure receipt")
    run = read_json(run_dir / "run_manifest.json")
    heartbeat = read_json(run_dir / "heartbeat.json")
    progress = read_json(run_dir / "progress.json")
    frozen = read_json(config_path)
    config_sha = sha256(config_path)
    require(config_sha == read_json(config_path.parent / "config_freeze_receipt.json")["config_sha256"],
            "local frozen config byte identity mismatch")
    require(run["run_id"] == RUN_ID and run["git_commit"] == SOURCE_SHA
            and run["dataset_manifest_sha256"] == sha256(dataset_manifest)
            and run["formal_config_sha256"] == config_sha
            and run["dataset_id"] == frozen["dataset_id"]
            and run["formal_training"] is True
            and run["locked_test_accessed"] is False
            and run["baseline"] is False and run["planner"] is False,
            "formal run/source/Dataset/config identity mismatch")
    git_type = subprocess.run(["git", "cat-file", "-t", SOURCE_SHA], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_SHA, "HEAD"], cwd=ROOT)
    require(git_type == "commit" and ancestry.returncode == 0,
            "training source SHA is not an ancestor of current main")
    protocol = formal_training_config_v1(dataset_manifest)
    programmatic_config = json.loads(json.dumps(protocol.as_manifest()["training_config"]))
    require(frozen["training_config"]["device"] == "cuda" and programmatic_config == frozen["training_config"],
            "programmatic frozen config and artifact disagree")
    interface = FormalTrainingInterface.from_manifest(dataset_manifest)
    require(len(interface.train_indices) == 4416 and len(interface.validation_indices) == 1104
            and interface.manifest["hashes"] == run["package_hashes"],
            "local Formal Dataset split/package manifest mismatch")

    train = read_jsonl(run_dir / "train_metrics.jsonl")
    validation = read_jsonl(run_dir / "validation_metrics.jsonl")
    validate_train_rows(train, RUN_ID, protocol)
    observations = validate_validation_rows(validation, RUN_ID)
    best_index = min(range(len(validation)), key=lambda index: validation[index]["L_Val"])
    best_step = validation[best_index]["completed_steps"]
    best_l_val = validation[best_index]["L_Val"]
    require(best_step == 5520 and all(best_l_val < row["L_Val"] for row in validation[:-1]),
            "final step is not the unique argmin L_Val")
    require(heartbeat == progress and heartbeat["status"] == "COMPLETED"
            and heartbeat["process_alive"] is False
            and heartbeat["completed_steps"] == 5520
            and heartbeat["nan_or_inf_detected"] is False
            and heartbeat["early_stopping_counter"] == 0
            and math.isclose(heartbeat["best_L_Val"], best_l_val, abs_tol=1e-12)
            and math.isclose(heartbeat["last_validation_L_Val"], best_l_val, abs_tol=1e-12),
            "completion heartbeat/progress contradict final validation")

    best_path = run_dir / "checkpoints" / "best.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    sample_index_sha = sha256(interface.package_paths["samples"] / "index.json")
    for payload in (best, latest):
        checkpoint_identity(payload, run, frozen, sample_index_sha, best_step, best_l_val)
    model_equal, tensor_count = model_states_equal(best["model_state"], latest["model_state"])
    require(model_equal and tensor_count > 0 and best["state"] == latest["state"]
            and best["data_identity"] == latest["data_identity"],
            "best/latest final-step model or state differs")
    inference = cpu_inference_smoke(interface, protocol, best_path, best)

    file_entries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(run_dir).as_posix()
            file_entries[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    require(all(name in file_entries for name in REQUIRED_FILES), "SHA256 manifest lacks required file")
    checkpoint_manifest = {
        "schema_version": "PI-JWM-STEP-5.6C-Final-Checkpoint-Manifest-v1",
        "run_id": RUN_ID, "training_source_git_sha": SOURCE_SHA,
        "dataset_id": run["dataset_id"], "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "formal_config_sha256": config_sha,
        "local_run_dir": run_dir.relative_to(ROOT).as_posix(), "files": file_entries,
        "best_step": best_step, "latest_step": latest["state"]["global_step"],
        "best_L_Val": best_l_val, "best_latest_model_state_equal": model_equal,
        "model_tensor_count": tensor_count,
    }
    observation = {
        "schema_version": "PI-JWM-STEP-5.6C-Formal-Validation-Observation-v1",
        "run_id": RUN_ID, "evidence_class": "Formal Validation Observation",
        "validation_split_windows": 1104,
        "rows": observations, "argmin_L_Val_step": best_step, "argmin_L_Val": best_l_val,
        "locked_test_accessed": False, "baseline": False, "planner_rollout": False,
        "locked_test": False, "planner": False,
        "performance_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "final_checkpoint_manifest.json"
    observation_path = output_dir / "formal_validation_observation.json"
    for path, data in ((manifest_path, checkpoint_manifest), (observation_path, observation)):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "PI-JWM-STEP-5.6C-Final-Acceptance-v1",
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID, "training_source_git_sha": SOURCE_SHA,
        "current_git_sha_at_acceptance": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                       capture_output=True, text=True, check=True).stdout.strip(),
        "dataset_id": run["dataset_id"], "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "formal_config_sha256": config_sha,
        "training_steps": len(train), "validation_steps": VALIDATION_STEPS,
        "validation_windows_per_pass": 1104,
        "argmin_L_Val_step": best_step, "argmin_L_Val": best_l_val,
        "best_checkpoint_sha256": file_entries["checkpoints/best.pt"]["sha256"],
        "latest_checkpoint_sha256": file_entries["checkpoints/latest.pt"]["sha256"],
        "best_latest_model_state_equal": model_equal, "model_tensor_count": tensor_count,
        "cpu_inference_smoke": inference,
        "final_checkpoint_manifest_sha256": sha256(manifest_path),
        "formal_validation_observation_sha256": sha256(observation_path),
        "checks": {
            "git_source_ancestor": True, "dataset_config_identity": True,
            "exact_5520_train_steps": True, "five_full_prior_only_validations": True,
            "finite_no_failure": True, "strict_final_argmin_L_Val": True,
            "best_and_latest_are_final_step": True, "checkpoint_architecture_data_config_identity": True,
            "best_latest_model_state_equal": True, "cpu_best_reload_and_h4_inference": True,
            "sha256_manifest_complete": True,
        },
        "passed": True, "step5_6b": "COMPLETE", "formal_best_checkpoint": "FROZEN",
        "scope": {"cpu_only_acceptance": True, "retraining": False, "gpu_used": False,
                  "ssh_used": False, "locked_test_accessed": False, "baseline": False,
                  "locked_test": False, "planner": False, "planner_rollout": False,
                  "performance_claim": False},
    }
    (output_dir / "final_acceptance_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "code/artifacts/formal_training" / RUN_ID)
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json")
    parser.add_argument("--formal-config", type=Path, default=ROOT / "code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/formal_training_config_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928")
    args = parser.parse_args()
    print(json.dumps(accept(args.run_dir, args.dataset_manifest, args.formal_config, args.output_dir),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
