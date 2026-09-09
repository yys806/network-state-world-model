"""Choose the largest safe P4 entity-RSSM GPU batch without an optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_world_model_loss_v1 import FormalLossWeights, formal_world_model_loss
from run_formal_dual_graph_gpu_train_v1 import (
    _build_learned_model,
    _set_entity_training_stage,
    _validated_slot_seconds,
    move_nested_to_device,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def select_largest_feasible_batch(
    rows: Iterable[dict[str, Any]], *, maximum_fraction: float = 0.85
) -> int:
    for row in rows:
        if bool(row.get("completed")) and float(row.get("memory_fraction", 1.0)) <= maximum_fraction:
            return int(row["batch_size"])
    raise RuntimeError("no candidate batch size fits the GPU memory boundary")


def run_probe(*, tensor_root: str | Path, output_dir: str | Path, device: str = "cuda") -> dict[str, Any]:
    tensor_root, output_dir = Path(tensor_root), Path(output_dir)
    if any("locked_test" in str(path).lower() for path in (tensor_root, output_dir)):
        raise ValueError("locked_test path is forbidden")
    requested_device = torch.device(device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the batch probe requires an available CUDA device")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="train",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )
    slot_seconds = _validated_slot_seconds(
        {"train": dataset}, fallback=float(contract.get("slot_seconds_value", 0.1))
    )
    total_bytes = int(torch.cuda.get_device_properties(requested_device).total_memory)
    attempts = []
    for batch_size in (8, 4, 2, 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(requested_device)
        model, _ = _build_learned_model(
            "entity_aligned_dual_graph_rssm_v1",
            hidden_dim=32,
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
            use_system_energy_head=False,
            zero_init_residual_state_heads=True,
            deterministic_rule_layer=True,
            rule_layer_stats=stats,
            n_rb=int(contract["n_rb"]),
            link_pos_weight=1.0,
            link_activity_missing_prior=0.03,
            slot_seconds=slot_seconds,
            node_position_scale=stats["features"]["node_state"]["scale"][:3],
            node_motion_mean=stats["features"]["node_motion_state"]["mean"],
            node_motion_scale=stats["features"]["node_motion_state"]["scale"],
        )
        model.to(requested_device).train()
        parameters = _set_entity_training_stage(model, "rssm")
        completed = False
        error = None
        loss_value = None
        try:
            batch = next(
                iter(
                    DataLoader(
                        Subset(dataset, range(batch_size)),
                        batch_size=batch_size,
                        shuffle=False,
                    )
                )
            )
            batch = move_nested_to_device(batch, requested_device)
            prediction = model(batch)
            loss, _ = formal_world_model_loss(
                prediction,
                batch["target"],
                batch["static"],
                weights=replace(
                    FormalLossWeights(),
                    rssm_kl=0.1,
                    rssm_teacher_reconstruction=0.5,
                    rssm_overshooting=0.1,
                    rssm_kl_balance=0.8,
                ),
                class_weights={"link_activity": 1.0},
                normalization_stats=stats,
            )
            loss.backward()
            torch.cuda.synchronize(requested_device)
            completed = bool(torch.isfinite(loss)) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            )
            loss_value = float(loss.detach())
        except torch.OutOfMemoryError as exc:
            error = type(exc).__name__
            completed = False
        peak_bytes = int(torch.cuda.max_memory_allocated(requested_device))
        attempts.append(
            {
                "batch_size": batch_size,
                "completed": completed,
                "finite_loss": completed,
                "loss": loss_value,
                "peak_memory_bytes": peak_bytes,
                "memory_fraction": peak_bytes / total_bytes,
                "error": error,
                "optimizer_created": False,
                "optimizer_step_performed": False,
            }
        )
        del model
        torch.cuda.empty_cache()
    selected = select_largest_feasible_batch(attempts)
    report = {
        "schema_version": "PI-JWM-P4-entity-RSSM-GPU-batch-probe-v1",
        "status": "passed",
        "candidate_order": [8, 4, 2, 1],
        "maximum_memory_fraction": 0.85,
        "selected_batch_size": selected,
        "attempts": attempts,
        "device": torch.cuda.get_device_name(requested_device),
        "total_memory_bytes": total_bytes,
        "tensor_manifest_sha256": _sha256(tensor_root / "manifest.json"),
        "optimizer_step_performed": False,
        "gpu_execution": True,
        "locked_test_accessed": False,
        "formal_performance_claim_ready": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "gpu_batch_probe.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "PI-JWM-P4-entity-RSSM-GPU-batch-probe-manifest-v1",
        "files": {
            report_path.name: {
                "size_bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        },
        "locked_test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_probe(
        tensor_root=args.tensor_root, output_dir=args.output_dir, device=args.device
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
