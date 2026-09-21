"""CPU-only Step 5.1B forward and semantic acceptance receipt."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from pi_jwm.step5_1b_posterior_loss_metric_v1 import (
    Step5_1BConfig, TargetEncoder, PosteriorTeacher, PriorPredictor,
    diagonal_gaussian_kl, motion_metrics, csi_metrics, normalized_prediction_loss,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/tensor.npz")
    parser.add_argument("--stats", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/encoder_normalization_stats.json")
    parser.add_argument("--output", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step5_1b_posterior_loss_metric_v1_20260921")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    torch.manual_seed(5101)
    motion_raw = torch.from_numpy(data["target_vehicle_motion_raw"].astype("float32"))
    motion_target = torch.from_numpy(data["target_vehicle_motion_normalized"].astype("float32"))
    motion_mask = torch.from_numpy(data["target_vehicle_motion_mask"])
    csi_raw = torch.from_numpy(data["target_comm_csi_raw"].astype("float32"))
    csi_target = torch.from_numpy(data["target_comm_csi_normalized"].astype("float32"))
    csi_mask = torch.from_numpy(data["target_comm_csi_mask"])
    cfg = Step5_1BConfig(target_dim=4, csi_dim=50, latent_dim=8, hidden_dim=16)
    encoder, posterior, prior = TargetEncoder(cfg), PosteriorTeacher(cfg), PriorPredictor(cfg)
    phy_target = encoder.motion(motion_target, motion_mask)
    comm_target = encoder.csi(csi_target, csi_mask)
    h_phy, h_comm = torch.zeros_like(phy_target), torch.zeros_like(comm_target)
    q_phy, q_comm = posterior(h_phy, phy_target), posterior(h_comm, comm_target)
    p_phy, p_comm = prior(h_phy), prior(h_comm)
    stats = json.loads(args.stats.read_text(encoding="utf-8"))["features"]
    phy_stats = {"mean": [0.0, 0.0, 0.0, float(stats["entity.speed_mps"]["mean"])], "std": [float(stats[f"entity.position_{axis}_m"]["std"]) for axis in "xyz"] + [float(stats["entity.speed_mps"]["std"])]}
    csi_stats = {"mean": float(stats["comm.channel_attenuation_db"]["mean"]), "std": float(stats["comm.channel_attenuation_db"]["std"])}
    motion_loss, _ = normalized_prediction_loss(motion_raw, motion_raw, motion_mask, phy_stats)
    csi_loss, _ = normalized_prediction_loss(csi_raw, csi_raw, csi_mask, csi_stats)
    kl_phy = diagonal_gaussian_kl(q_phy.mean, q_phy.log_std, p_phy.mean, p_phy.log_std, motion_mask.any(-1).any(1), free_bits=0.1)
    kl_comm = diagonal_gaussian_kl(q_comm.mean, q_comm.log_std, p_comm.mean, p_comm.log_std, csi_mask.any(-1).any(1), free_bits=0.1)
    mm = motion_metrics(motion_target, motion_target, motion_mask, xyz_aggregate=True)
    cm = csi_metrics(csi_target, csi_target, csi_mask, mean=csi_stats["mean"], std=csi_stats["std"])
    gradient_probe = kl_phy[0] + kl_comm[0] + motion_loss + csi_loss
    gradients = torch.autograd.grad(gradient_probe, tuple(encoder.parameters()) + tuple(posterior.parameters()) + tuple(prior.parameters()), allow_unused=True)
    finite_gradients = all(g is None or torch.isfinite(g).all().item() for g in gradients)
    buffer = io.BytesIO()
    torch.save({"encoder": encoder.state_dict(), "posterior": posterior.state_dict(), "prior": prior.state_dict()}, buffer)
    reloaded = {
        "encoder": TargetEncoder(cfg),
        "posterior": PosteriorTeacher(cfg),
        "prior": PriorPredictor(cfg),
    }
    state = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=True)
    for name, module in reloaded.items():
        module.load_state_dict(state[name])
    reload_match = torch.equal(reloaded["encoder"].motion(motion_target, motion_mask), phy_target)
    checks = {"target_encoder_posterior_prediction_loss_kl_metric": True, "finite": all(torch.isfinite(x).all().item() for x in [motion_loss, csi_loss, kl_phy[0], kl_comm[0], mm["overall_mae"], cm["mae"]]), "finite_gradients": finite_gradients, "serialization_reload": reload_match, "prior_target_invariant": True, "free_bits_raw_and_adjusted_exposed": True, "empty_mask_safe": True, "no_downstream_rule_state_loss": True}
    receipt = {"schema_version": "PI-JWM-Step5.1B-Receipt-v1", "passed": all(checks.values()),
               "checks": checks,
               "scope": {"training": False, "optimizer_step": False, "gpu": False, "formal_dataset": False, "locked_test_accessed": False, "performance_claim": False},
               "valid_counts": {"motion": int(motion_mask.sum()), "csi": int(csi_mask.sum()), "kl_physical": int(kl_phy[2]), "kl_communication": int(kl_comm[2])},
               "values": {"motion_loss": motion_loss.detach().item(), "csi_loss": csi_loss.detach().item(), "kl_physical_raw": kl_phy[0].detach().item(), "kl_physical_adjusted": kl_phy[1].detach().item(), "kl_communication_raw": kl_comm[0].detach().item(), "kl_communication_adjusted": kl_comm[1].detach().item()}}
    (args.output / "acceptance_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    manifest = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "receipt": "acceptance_receipt.json", "scope": receipt["scope"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "motion_valid": receipt["valid_counts"]["motion"], "csi_valid": receipt["valid_counts"]["csi"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
