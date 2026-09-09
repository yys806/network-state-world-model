"""Frozen GPU entry point for the P4 entity-aligned dual-graph RSSM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_formal_dual_graph_gpu_train_v1 import run_gpu_training


METHOD = "entity_aligned_dual_graph_rssm_v1"
SEEDS = (20260831, 20260830, 20260832)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_prerequisites(
    *,
    tensor_root: Path,
    batch_probe_dir: Path,
    consistency_audit_dir: Path,
    mode: str,
    sentinel_run_dir: Path | None,
    seed: int,
    seed31_run_dir: Path | None,
) -> int:
    if any(
        "locked_test" in str(path).lower()
        for path in (
            tensor_root,
            batch_probe_dir,
            consistency_audit_dir,
            *(() if sentinel_run_dir is None else (sentinel_run_dir,)),
            *(() if seed31_run_dir is None else (seed31_run_dir,)),
        )
    ):
        raise ValueError("locked_test path is forbidden")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    tensor_validation = _read_json(tensor_root / "validation_report.json")
    if tensor_validation.get("formal_tensor_ready") is not True:
        raise ValueError("causal-motion tensor is not ready")
    audit = _read_json(
        consistency_audit_dir / "entity_aligned_rssm_consistency_audit.json"
    )
    if audit.get("status") != "ready_for_gpu_execution_sentinel":
        raise ValueError("CPU entity-RSSM consistency audit has not passed")
    probe = _read_json(batch_probe_dir / "gpu_batch_probe.json")
    if probe.get("status") != "passed" or probe.get("optimizer_step_performed") is not False:
        raise ValueError("GPU batch probe has not passed without an optimizer step")
    tensor_manifest_sha256 = _sha256(tensor_root / "manifest.json")
    if (
        probe.get("tensor_manifest_sha256") != tensor_manifest_sha256
        or audit.get("evidence", {}).get("tensor_manifest_sha256")
        != tensor_manifest_sha256
    ):
        raise ValueError("GPU prerequisites refer to a different tensor manifest")
    batch_size = int(probe["selected_batch_size"])
    if batch_size not in (8, 4, 2, 1):
        raise ValueError("GPU batch probe selected an unsupported batch size")
    if mode == "sentinel":
        if seed != 20260831:
            raise ValueError("GPU execution sentinel is fixed to seed 20260831")
        return batch_size
    if mode != "formal":
        raise ValueError("mode must be sentinel or formal")
    if sentinel_run_dir is None:
        raise ValueError("formal training requires the completed GPU sentinel run")
    sentinel = _read_json(sentinel_run_dir / "run_summary.json")
    sentinel_config = _read_json(sentinel_run_dir / "config.json")
    if (
        sentinel.get("training_run_complete") is not True
        or sentinel.get("gpu_execution") is not True
        or sentinel.get("locked_test_accessed") is not False
        or sentinel_config.get("protocol_mode") != "gpu_execution_sentinel"
        or sentinel_config.get("batch_size") != batch_size
    ):
        raise ValueError("GPU execution sentinel evidence is incomplete or incompatible")
    if seed != 20260831:
        if seed31_run_dir is None:
            raise ValueError("follow-up seeds require the seed 20260831 formal run")
        seed31_checkpoint = torch.load(
            seed31_run_dir / "checkpoints" / f"{METHOD}__best.pt",
            map_location="cpu",
            weights_only=True,
        )
        gate = seed31_checkpoint.get("p4_gate")
        seed31_config = _read_json(seed31_run_dir / "config.json")
        if (
            seed31_checkpoint.get("method") != METHOD
            or seed31_config.get("seed") != 20260831
            or seed31_config.get("batch_size") != batch_size
            or seed31_config.get("protocol_mode") != "formal_single_seed"
            or not isinstance(gate, dict)
            or gate.get("all_numeric_gates_passed") is not True
        ):
            raise ValueError("seed 20260831 did not pass every single-seed numeric P4 gate")
    return batch_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sentinel", "formal"), required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--batch-probe-dir", type=Path, required=True)
    parser.add_argument("--consistency-audit-dir", type=Path, required=True)
    parser.add_argument("--sentinel-run-dir", type=Path)
    parser.add_argument("--seed31-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    batch_size = validate_prerequisites(
        tensor_root=args.tensor_root,
        batch_probe_dir=args.batch_probe_dir,
        consistency_audit_dir=args.consistency_audit_dir,
        mode=args.mode,
        sentinel_run_dir=args.sentinel_run_dir,
        seed=args.seed,
        seed31_run_dir=args.seed31_run_dir,
    )
    common = dict(
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        device=args.device,
        learned_methods=(METHOD,),
        seed=args.seed,
        data_seed=args.seed,
        hidden_dim=32,
        batch_size=batch_size,
        learning_rate=3e-4,
        weight_decay=1e-5,
        zero_init_residual_state_heads=True,
        deterministic_rule_layer=True,
    )
    if args.mode == "sentinel":
        summary = run_gpu_training(
            **common,
            train_limit=256,
            evaluation_limit=128,
            base_epochs=1,
            epochs=1,
            checkpoint_selection="validation_loss_v1",
            min_epochs=0,
            patience=0,
            save_every_epoch=False,
            protocol_mode="gpu_execution_sentinel",
        )
    else:
        summary = run_gpu_training(
            **common,
            train_limit=None,
            evaluation_limit=None,
            base_epochs=20,
            epochs=40,
            checkpoint_selection="p4_gate_aware_v1",
            min_epochs=20,
            patience=10,
            save_every_epoch=True,
            protocol_mode="formal_single_seed",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
