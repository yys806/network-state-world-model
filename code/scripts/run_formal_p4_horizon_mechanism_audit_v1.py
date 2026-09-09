"""Audit CPU forward/conditioning mechanics for the formal P4 horizons."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_dual_graph_world_model_v1 import FormalDualGraphWorldModel, FormalWorldModelConfig
from run_formal_dual_graph_cpu_smoke_v1 import _load_or_fit_stats


REQUIRED_HORIZONS = (1, 5, 20)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    return value


def _slice_future_action(batch: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    sliced = copy.deepcopy(dict(batch))
    sliced["future_action"] = {
        key: value[:, :horizon] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value
        for key, value in batch["future_action"].items()
    }
    return sliced


def _load_checkpoint(run_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint_path = run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint lacks model_config/model_state_dict: {checkpoint_path}")
    if not bool(model_config.get("deterministic_rule_layer")):
        raise ValueError(f"checkpoint is not rule-layer enabled: {checkpoint_path}")
    return model_config, state_dict


def run_audit(*, run_dir: str | Path, tensor_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    tensor_path = Path(tensor_root)
    output_path = Path(output_dir)
    if any("locked_test" in str(path).lower() for path in (run_path, tensor_path, output_path)):
        raise ValueError("locked_test path is forbidden")

    contract = _read_json(tensor_path / "tensor_contract.json")
    if int(contract["history_steps"]) != 8 or int(contract["horizon_steps"]) < max(REQUIRED_HORIZONS):
        raise ValueError("tensor contract does not support the required P4 horizons")
    model_config_payload, state_dict = _load_checkpoint(run_path)
    stats = _load_or_fit_stats(tensor_path)
    dataset = FormalAirFogSimWindowDataset(
        tensor_path,
        split="validation",
        config=FormalWindowConfig(history_steps=8, horizon_steps=int(contract["horizon_steps"])),
        stats=stats,
        normalize=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    source_batch = None
    for candidate in loader:
        action_present = candidate["future_action"]["task_action_present"]
        if bool(action_present.any()):
            source_batch = _move_to_cpu(candidate)
            break
    if source_batch is None:
        raise ValueError("validation windows contain no logged future action")

    horizon_reports: dict[str, dict[str, Any]] = {}
    for horizon in REQUIRED_HORIZONS:
        config = FormalWorldModelConfig(**model_config_payload)
        config = replace(config, horizon_steps=horizon)
        model = FormalDualGraphWorldModel(config)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        batch = _slice_future_action(source_batch, horizon)
        with torch.no_grad():
            baseline = model(batch)
        finite = all(bool(torch.isfinite(value).all()) for value in baseline.values() if isinstance(value, torch.Tensor))
        shape_steps = {
            key: int(value.shape[1])
            for key, value in baseline.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2
        }

        changed = copy.deepcopy(batch)
        present = changed["future_action"]["task_action_present"][0]
        indices = torch.nonzero(present, as_tuple=False)
        if indices.numel() == 0:
            raise ValueError(f"no action in selected window for horizon {horizon}")
        step, task = (int(indices[0, 0]), int(indices[0, 1]))
        changed["future_action"]["task_action"][0, step, task, 0] += 1.0
        with torch.no_grad():
            perturbed = model(changed)
        output_deltas = {
            key: float((perturbed[key] - baseline[key]).abs().max().item())
            for key in baseline
            if isinstance(baseline[key], torch.Tensor)
        }
        horizon_reports[str(horizon)] = {
            "status": "passed" if finite and all(value == horizon for value in shape_steps.values()) and max(output_deltas.values(), default=0.0) > 0.0 else "failed",
            "expected_horizon": horizon,
            "output_steps": shape_steps,
            "all_tensor_outputs_finite": finite,
            "action_probe": {"step": step, "task": task, "feature": 0, "max_output_delta": max(output_deltas.values(), default=0.0)},
            "output_deltas": output_deltas,
        }

    report = {
        "schema_version": "PI-JWM-formal-p4-horizon-mechanism-audit-v1",
        "status": "passed" if all(item["status"] == "passed" for item in horizon_reports.values()) else "blocked",
        "horizons": horizon_reports,
        "tensor_contract": {"history_steps": int(contract["history_steps"]), "horizon_steps": int(contract["horizon_steps"])},
        "evidence": {"tensor_root": str(tensor_path.resolve()), "checkpoint_run": str(run_path.resolve())},
        "execution_policy": {
            "evaluation_device": "cpu",
            "gpu_execution": False,
            "training_started": False,
            "locked_test_accessed": False,
            "result_boundary": "Mechanism/interface audit only; no 1/5/20-step accuracy or final-method claim.",
        },
        "checkpoint_boundary": "Weights come from a horizon-3 training run; horizon-1/5/20 outputs are an interface audit, not retrained evidence.",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "p4_horizon_mechanism_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(run_dir=args.run_dir, tensor_root=args.tensor_root, output_dir=args.output_dir)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
