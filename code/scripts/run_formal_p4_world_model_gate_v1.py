"""Run the bounded, non-locked P4 world-model mechanism gate on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from run_formal_dual_graph_cpu_smoke_v1 import _load_or_fit_stats, _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import move_nested_to_device


REQUIRED_HORIZONS = (1, 5, 20)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_model(run_dir: Path) -> FormalDualGraphWorldModel:
    checkpoint_path = run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or not config.get("deterministic_rule_layer"):
        raise ValueError(f"checkpoint is not rule-layer enabled: {checkpoint_path}")
    model = FormalDualGraphWorldModel(FormalWorldModelConfig(**config))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _clone_namespace(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.clone() for key, tensor in value.items()}


def _resolve_probe_horizon(model_config: dict[str, Any], tensor_horizon: int) -> int:
    model_horizon = int(model_config.get("horizon_steps", 0))
    if model_horizon <= 0:
        raise ValueError("checkpoint horizon must be positive")
    if model_horizon > int(tensor_horizon):
        raise ValueError("checkpoint horizon exceeds tensor contract")
    return model_horizon


def _slice_probe_batch(batch: dict[str, Any], horizon: int) -> dict[str, Any]:
    sliced = copy.deepcopy(batch)
    for namespace in ("future_action", "target"):
        if namespace not in batch:
            continue
        sliced[namespace] = {
            key: value[:, :horizon] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value
            for key, value in batch[namespace].items()
        }
    return sliced


def _paired_action_probe(
    *, run_dir: Path, tensor_root: Path, sample_limit: int = 16
) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _load_or_fit_stats(tensor_root)
    model = _load_model(run_dir)
    contract_horizon = int(contract["horizon_steps"])
    model_horizon = _resolve_probe_horizon(model.config.__dict__, contract_horizon)
    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=contract_horizon,
        ),
        stats=stats,
        normalize=True,
    )
    sample_ids = _read_json(run_dir / "sample_ids.json")["validation"][:sample_limit]
    loader = DataLoader(_subset_for_ids(dataset, sample_ids), batch_size=1, shuffle=False, num_workers=0)

    for batch_index, cpu_batch in enumerate(loader):
        batch = move_nested_to_device(cpu_batch, torch.device("cpu"))
        action_present = batch["future_action"]["task_action_present"].bool()
        valid = action_present[..., None].expand_as(batch["future_action"]["task_action"][:, :, :, :5])
        candidates = torch.nonzero(valid, as_tuple=False)
        if candidates.numel() == 0:
            continue
        _, step, task, feature = [int(value) for value in candidates[0].tolist()]
        baseline_batch = _slice_probe_batch({
            "history": _clone_namespace(batch["history"]),
            "future_action": _clone_namespace(batch["future_action"]),
            "target": _clone_namespace(batch["target"]),
            "static": _clone_namespace(batch["static"]),
        }, model_horizon)
        changed_batch = copy.deepcopy(baseline_batch)
        changed_batch["future_action"]["task_action"][0, step, task, feature] += 1.0
        with torch.no_grad():
            baseline = model(baseline_batch)
            changed = model(changed_batch)

        deltas = {}
        for key in ("task_state_mean", "node_state_mean", "physical_edge_state_mean", "flow_state_mean"):
            deltas[key] = float((changed[key] - baseline[key]).abs().max().item())
        max_delta = max(deltas.values())

        target_changed_batch = copy.deepcopy(baseline_batch)
        for value in target_changed_batch["target"].values():
            if value.is_floating_point():
                value.fill_(99999.0)
        with torch.no_grad():
            target_changed = model(target_changed_batch)
        target_leakage_delta = float(
            max((target_changed[key] - baseline[key]).abs().max().item() for key in deltas),
        )
        return {
            "status": "passed" if max_delta > 0.0 and target_leakage_delta == 0.0 else "failed",
            "batch_index": batch_index,
            "sample_id": sample_ids[batch_index],
            "changed_action": {"step": step, "task": task, "feature": feature, "delta": 1.0},
            "max_output_delta": max_delta,
            "output_deltas": deltas,
            "target_leakage_max_delta": target_leakage_delta,
            "seed": int(config["seed"]),
            "model_horizon": model_horizon,
        }
    return {"status": "not_computable", "reason": "no valid logged future action in the probed validation windows"}


def run_gate(
    *,
    run_dirs: list[str | Path],
    tensor_root: str | Path,
    output_dir: str | Path,
    horizon_mechanism_report: str | Path,
) -> dict[str, Any]:
    run_paths = [Path(value) for value in run_dirs]
    tensor_path = Path(tensor_root)
    output_path = Path(output_dir)
    if not run_paths:
        raise ValueError("at least one run directory is required")
    forbidden = [tensor_path, *run_paths]
    if any("locked_test" in str(path).lower() for path in forbidden):
        raise ValueError("locked_test path is forbidden")

    contract = _read_json(tensor_path / "tensor_contract.json")
    horizon_report_path = Path(horizon_mechanism_report)
    if "locked_test" in str(horizon_report_path).lower():
        raise ValueError("locked_test path is forbidden")
    horizon_report = _read_json(horizon_report_path)
    consistency_path = CODE_ROOT / "artifacts" / "audit" / "pi_jwm_formal_gpu_rule_v2_consistency_20260825" / "candidate_consistency_audit.json"
    cpu_replay_path = CODE_ROOT / "artifacts" / "audit" / "pi_jwm_formal_rule_v2_cpu_rule_rollout_multiseed_20260826_created_flow_fixed" / "rule_rollout_multiseed_audit.json"
    gpu_rollout_path = CODE_ROOT / "artifacts" / "audit" / "pi_jwm_formal_gpu_rule_v2_rollout_multiseed_20260826" / "gpu_rollout_multiseed_audit.json"
    consistency = _read_json(consistency_path)
    cpu_replay = _read_json(cpu_replay_path)
    gpu_rollout = _read_json(gpu_rollout_path)
    probes = [_paired_action_probe(run_dir=path, tensor_root=tensor_path) for path in run_paths]
    declared_horizon = int(contract["horizon_steps"])
    horizon_gate = {
        "required": list(REQUIRED_HORIZONS),
        "declared_contract_horizon": declared_horizon,
        "available_target_horizons": list(range(1, declared_horizon + 1)),
        "status": "passed" if declared_horizon >= max(REQUIRED_HORIZONS) else "blocked",
        "reason": (
            None
            if declared_horizon >= max(REQUIRED_HORIZONS)
            else "formal tensor contract has only three future target steps; k=5 and k=20 cannot be evaluated without a new data contract and retraining"
        ),
    }
    checks = {
        "candidate_consistency": bool(consistency.get("status") == "ready_for_review" and not consistency.get("critical_mismatches")),
        "cpu_rule_replay": bool(cpu_replay.get("audit_passed")),
        "gpu_rollout_artifact_nonlocked": bool(gpu_rollout.get("audit_passed") and not gpu_rollout.get("execution_policy", {}).get("locked_test_accessed", True)),
        "paired_future_action_changes_prediction": all(probe.get("status") == "passed" for probe in probes),
        "horizon_1_5_20": horizon_gate["status"] == "passed",
        "horizon_mechanism_audit": bool(
            horizon_report.get("status") == "passed"
            and horizon_report.get("tensor_contract", {}).get("history_steps") == int(contract["history_steps"])
            and horizon_report.get("tensor_contract", {}).get("horizon_steps") == declared_horizon
        ),
    }
    checkpoint_horizons = sorted(
        {int(probe["model_horizon"]) for probe in probes if "model_horizon" in probe}
    )
    next_gate = (
        "P4 mechanism gate passed. The current checkpoints were trained at horizon=3; retrain and evaluate the same rule-enabled candidate with the frozen horizon=20 contract before claiming final P4 completion or entering P5."
        if checkpoint_horizons != [declared_horizon]
        else "P4 mechanism and horizon contract passed; complete the frozen state/task/resource/uncertainty metric report before entering P5."
    )
    report = {
        "schema_version": "PI-JWM-formal-p4-world-model-gate-v1",
        "status": "passed" if all(checks.values()) else "blocked",
        "audit_passed": all(checks.values()),
        "formal_performance_claim_ready": False,
        "checks": checks,
        "action_probes": probes,
        "horizon_gate": horizon_gate,
        "checkpoint_horizons": checkpoint_horizons,
        "evidence": {
            "tensor_root": str(tensor_path.resolve()),
            "consistency_report": str(consistency_path.resolve()),
            "cpu_rule_replay_report": str(cpu_replay_path.resolve()),
            "gpu_rollout_report": str(gpu_rollout_path.resolve()),
            "run_dirs": [str(path.resolve()) for path in run_paths],
            "horizon_mechanism_report": str(horizon_report_path.resolve()),
        },
        "execution_policy": {
            "evaluation_device": "cpu",
            "gpu_execution": False,
            "training_started": False,
            "locked_test_accessed": False,
            "result_boundary": "P4 mechanism gate only; non-locked aggregate-baseline evidence.",
        },
        "next_gate": next_gate,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "p4_world_model_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-mechanism-report", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate(
        run_dirs=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        horizon_mechanism_report=args.horizon_mechanism_report,
    )
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
