"""Decompose formal complete-RSSM node-x error into base and latent correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_tensor_v2 import NODE_FEATURES
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from run_formal_dual_graph_cpu_smoke_v1 import _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import _reload_learned_model


DEFAULT_METHOD = "complete_rssm_dual_graph_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_diagnosis(
    *,
    run_dir: str | Path,
    tensor_root: str | Path,
    output_dir: str | Path,
    method: str = DEFAULT_METHOD,
) -> dict:
    run_dir, tensor_root, output_dir = map(Path, (run_dir, tensor_root, output_dir))
    if "locked_test" in str(run_dir).lower() or "locked_test" in str(tensor_root).lower():
        raise ValueError("locked_test path is forbidden")
    summary = _read_json(run_dir / "run_summary.json")
    if summary.get("locked_test_accessed") is not False:
        raise ValueError("run does not prove the locked-test boundary")
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    sample_ids = _read_json(run_dir / "sample_ids.json")["validation"]
    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )
    loader = DataLoader(_subset_for_ids(dataset, sample_ids), batch_size=2, shuffle=False)
    checkpoint = torch.load(
        run_dir / "checkpoints" / f"{method}__best.pt",
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("method") != method:
        raise ValueError("checkpoint method does not match requested diagnosis method")
    model = _reload_learned_model(method, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    horizon = int(contract["horizon_steps"])
    accumulators = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("full_error", "base_error", "persistence_error", "correction_abs", "improved")
    }
    counts = torch.zeros(horizon, dtype=torch.float64)
    x_index = list(NODE_FEATURES).index("x")
    x_scale = float(stats["features"]["node_state"]["scale"][x_index])
    max_reconstruction_error = 0.0
    with torch.no_grad():
        for batch in loader:
            full = model(batch)
            base = model.base(
                {
                    "history": batch["history"],
                    "future_action": batch["future_action"],
                    "static": batch["static"],
                }
            )
            full_x = full["node_state_mean"][..., x_index]
            base_x = base["node_state_mean"][..., x_index]
            target_x = batch["target"]["node_state"][..., x_index]
            persistence_x = batch["history"]["node_state"][:, -1:, :, x_index].expand_as(target_x)
            valid = batch["target"]["node_present"].bool()
            correction = full_x - base_x
            max_reconstruction_error = max(
                max_reconstruction_error,
                float((base_x + correction - full_x).abs().max()),
            )
            for step in range(horizon):
                mask = valid[:, step]
                count = mask.sum()
                counts[step] += count
                full_error = (full_x[:, step] - target_x[:, step]).abs() * x_scale
                base_error = (base_x[:, step] - target_x[:, step]).abs() * x_scale
                persistence_error = (persistence_x[:, step] - target_x[:, step]).abs() * x_scale
                accumulators["full_error"][step] += full_error[mask].sum()
                accumulators["base_error"][step] += base_error[mask].sum()
                accumulators["persistence_error"][step] += persistence_error[mask].sum()
                accumulators["correction_abs"][step] += (correction[:, step].abs() * x_scale)[mask].sum()
                accumulators["improved"][step] += (full_error[mask] < base_error[mask]).sum()
    per_horizon = []
    for step in range(horizon):
        count = counts[step].clamp_min(1.0)
        per_horizon.append(
            {
                "horizon": step + 1,
                "valid_count": int(counts[step]),
                "full_node_x_mae": float(accumulators["full_error"][step] / count),
                "base_node_x_mae": float(accumulators["base_error"][step] / count),
                "persistence_node_x_mae": float(accumulators["persistence_error"][step] / count),
                "rssm_correction_abs_mean": float(accumulators["correction_abs"][step] / count),
                "rssm_correction_improved_fraction": float(accumulators["improved"][step] / count),
            }
        )
    total_count = counts.sum().clamp_min(1.0)
    overall = {
        "full_node_x_mae": float(accumulators["full_error"].sum() / total_count),
        "base_node_x_mae": float(accumulators["base_error"].sum() / total_count),
        "persistence_node_x_mae": float(accumulators["persistence_error"].sum() / total_count),
        "rssm_correction_abs_mean": float(accumulators["correction_abs"].sum() / total_count),
        "rssm_correction_improved_fraction": float(accumulators["improved"].sum() / total_count),
    }
    full_ratio = overall["full_node_x_mae"] / overall["persistence_node_x_mae"]
    base_ratio = overall["base_node_x_mae"] / overall["persistence_node_x_mae"]
    if base_ratio <= 1.25 < full_ratio:
        conclusion = "rssm_correction_sufficient_to_cross_node_x_gate"
    elif base_ratio > 1.25:
        conclusion = "formal_base_already_exceeds_node_x_gate"
    else:
        conclusion = "node_x_gate_not_reproduced"
    report = {
        "schema_version": "PI-JWM-formal-complete-RSSM-node-correction-diagnosis-v1",
        "status": "completed",
        "method": method,
        "conclusion": conclusion,
        "overall": {**overall, "full_ratio": full_ratio, "base_ratio": base_ratio},
        "per_horizon": per_horizon,
        "checks": {
            "strict_checkpoint_reload": True,
            "prediction_decomposition_exact": max_reconstruction_error == 0.0,
            "sample_ids_reused": len(sample_ids) == len(loader.dataset),
            "gpu_execution": False,
            "locked_test_accessed": False,
        },
        "result_boundary": "Read-only decomposition of one frozen checkpoint; no causal repair or performance claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "node_correction_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnosis(
        run_dir=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        method=args.method,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
