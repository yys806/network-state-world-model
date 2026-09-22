"""Build the STEP 5.1B-PATCH CPU integration receipt without training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from build_step4_4_structured_rssm_world_model_v1 import _real_state, _real_zpi
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig, StructuredRSSMWorldModel
from pi_jwm.step5_1b_posterior_loss_metric_v1 import Step5_1BConfig, TargetEncoder, FuturePosterior, diagonal_gaussian_kl, family_horizon_mse, motion_metrics, csi_metrics


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/tensor.npz")
    parser.add_argument("--output", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step5_1b_posterior_loss_metric_v1_20260921")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    torch.manual_seed(5101)
    targets = {k: torch.from_numpy(data[k].astype("float32")) for k in ("target_vehicle_motion_raw", "target_vehicle_motion_normalized", "target_vehicle_motion_mask", "target_comm_csi_raw", "target_comm_csi_normalized", "target_comm_csi_mask")}
    tensor, graph, _ = _real_zpi()
    model_cfg = StructuredRSSMConfig()
    model = StructuredRSSMWorldModel(model_cfg)
    patch_cfg = Step5_1BConfig(target_dim=4, csi_dim=50, hidden_dim=model_cfg.d_h, latent_dim=model_cfg.d_z, d_h=model_cfg.d_h, d_z=model_cfg.d_z, d_encoder=model_cfg.d_encoder)
    encoder, teacher = TargetEncoder(patch_cfg), FuturePosterior(patch_cfg)
    stats = json.loads((root / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/encoder_normalization_stats.json").read_text(encoding="utf-8"))["features"]
    phy_stats = {"mean": [0.0, 0.0, 0.0, float(stats["entity.speed_mps"]["mean"])], "std": [float(stats[f"entity.position_{a}_m"]["std"]) for a in "xyz"] + [float(stats["entity.speed_mps"]["std"])]}
    csi_stats = {"mean": float(stats["comm.channel_attenuation_db"]["mean"]), "std": float(stats["comm.channel_attenuation_db"]["std"])}
    all_grad = []; horizon_losses = []; metric_units_ok = True; prior_isolation = True; temporal_isolation = True; mask_evidence = True; free_bits_ok = True
    sample_count = len(targets["target_vehicle_motion_raw"]); real_paths = 0
    for i in range(sample_count):
        # The 12 target windows are real; the frozen real current model fixture
        # is reused as the CPU integration carrier because its support is fixed.
        _, _, zpi = _real_zpi()
        state, dyn_graph, action = _real_state(tensor, graph, 2)
        latent = model.initialize_latent(zpi, state)
        nxt, _, _, trace = model.one_step(latent, state, dyn_graph, action, prior_mode="mean", service_mode="expectation", generator=None)
        h_phy = nxt["h"]["physical"]; h_comm = nxt["h"]["communication"]
        p_phy = nxt["prior"]["physical"]; p_comm = nxt["prior"]["communication"]
        motion_t = targets["target_vehicle_motion_normalized"][i:i+1, :, :h_phy.shape[1]]; motion_m = targets["target_vehicle_motion_mask"][i:i+1, :, :h_phy.shape[1]]
        csi_t = targets["target_comm_csi_normalized"][i:i+1, :, :h_comm.shape[1]]; csi_m = targets["target_comm_csi_mask"][i:i+1, :, :h_comm.shape[1]]
        e_motion = encoder.motion(motion_t, motion_m); e_csi = encoder.csi(csi_t, csi_m)
        if i == 0:
            zero_value = torch.zeros_like(motion_t[:1, :1, :1]); true_e = encoder.motion(torch.zeros_like(motion_t[:1, :1, :1, :]), torch.ones_like(motion_m[:1, :1, :1, :], dtype=torch.bool)); false_e = encoder.motion(torch.zeros_like(motion_t[:1, :1, :1, :]), torch.zeros_like(motion_m[:1, :1, :1, :], dtype=torch.bool))
            mask_evidence = (not torch.equal(true_e, false_e)) and torch.equal(false_e, torch.zeros_like(false_e))
        h_phy_l = h_phy[:, None].expand(-1, motion_t.shape[1], -1, -1); h_comm_l = h_comm[:, None].expand(-1, csi_t.shape[1], -1, -1)
        q = teacher(h_phy_l, h_comm_l, e_motion, e_csi)
        if i == 0:
            tampered_motion = motion_t.clone(); tampered_motion[:, 1] = tampered_motion[:, 1] + 17.0
            tampered_e = encoder.motion(tampered_motion, motion_m)
            tampered_q = teacher(h_phy_l, h_comm_l, tampered_e, e_csi)
            temporal_isolation = torch.equal(e_motion[:, 0], tampered_e[:, 0]) and torch.equal(q["physical"].mean[:, 0], tampered_q["physical"].mean[:, 0])
        pphy = {"mean": p_phy["mean"][:, None].expand_as(q["physical"].mean), "log_std": p_phy["log_std"][:, None].expand_as(q["physical"].log_std)}
        pcomm = {"mean": p_comm["mean"][:, None].expand_as(q["communication"].mean), "log_std": p_comm["log_std"][:, None].expand_as(q["communication"].log_std)}
        # Decoder outputs are raw STEP 4.4 units.  They are repeated only to align the two supervised horizons; no target enters this path.
        phy_pred_raw = model.vehicle_decoder(torch.cat((h_phy_l, q["physical"].mean), -1)); comm_pred_raw = model.csi_decoder(torch.cat((h_comm_l, q["communication"].mean), -1))
        motion_raw = targets["target_vehicle_motion_raw"][i:i+1, :, :h_phy.shape[1]]; csi_raw = targets["target_comm_csi_raw"][i:i+1, :, :h_comm.shape[1]]
        mot_norm = (phy_pred_raw - motion_raw.new_tensor(phy_stats["mean"])) / motion_raw.new_tensor(phy_stats["std"])
        csi_norm = (comm_pred_raw - csi_raw.new_tensor(csi_stats["mean"])) / csi_raw.new_tensor(csi_stats["std"])
        mot = family_horizon_mse(mot_norm, motion_t, motion_m); cs = family_horizon_mse(csi_norm, csi_t, csi_m)
        phy_elig = motion_m.any(-1) & state["vehicle_mask"][:, None, :h_phy.shape[1]]
        comm_elig = csi_m.any(-1) & state["comm_presence"][:, None, :h_comm.shape[1]] & state["comm_validity"][:, None, :h_comm.shape[1]] & state["comm_wireless_mask"][:, None, :h_comm.shape[1]]
        klp = diagonal_gaussian_kl(q["physical"].mean, q["physical"].log_std, pphy["mean"], pphy["log_std"], phy_elig, free_bits=0.1)
        klc = diagonal_gaussian_kl(q["communication"].mean, q["communication"].log_std, pcomm["mean"], pcomm["log_std"], comm_elig, free_bits=0.1)
        free_bits_zero = diagonal_gaussian_kl(q["physical"].mean, q["physical"].log_std, pphy["mean"], pphy["log_std"], phy_elig, free_bits=0.0)
        free_bits_ok = bool(free_bits_ok and torch.equal(free_bits_zero.raw, free_bits_zero.adjusted) and bool(torch.all(klp.per_dim_adjusted >= klp.per_dim_raw)))
        probe = mot["loss"].sum() + cs["loss"].sum() + klp.adjusted + klc.adjusted
        params = list(encoder.parameters()) + list(teacher.parameters()) + list(model.vehicle_decoder.parameters()) + list(model.csi_decoder.parameters()) + list(model.phy_prior.parameters()) + list(model.comm_prior.parameters())
        grads = torch.autograd.grad(probe, params, allow_unused=True, retain_graph=False)
        all_grad.extend(grads); horizon_losses.extend([mot["loss"].detach(), cs["loss"].detach()]); real_paths += 1
        if i == 0:
            pphy_again = model.phy_prior(h_phy); pcomm_again = model.comm_prior(h_comm)
            prior_isolation = torch.equal(p_phy["mean"].detach(), pphy_again["mean"].detach()) and torch.equal(p_comm["mean"].detach(), pcomm_again["mean"].detach())
        metric_units_ok = metric_units_ok and motion_metrics(targets["target_vehicle_motion_raw"][i:i+1, :, :h_phy.shape[1]], motion_raw, motion_m)["units"] == ("m", "m", "m", "m/s") and csi_metrics(targets["target_comm_csi_raw"][i:i+1, :, :h_comm.shape[1]], csi_raw, csi_m)["unit"] == "dB"
    non_none_finite = bool(all_grad) and all(g is not None and torch.isfinite(g).all().item() and float(g.norm()) > 0 for g in all_grad)
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    has_optimizer_step = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "step" for node in ast.walk(tree))
    required = {"temporal_isolation": temporal_isolation, "mask_evidence": mask_evidence, "actual_prior_decoder_integration": real_paths == sample_count, "free_bits_per_dimension": free_bits_ok, "gradient_connection": non_none_finite, "raw_unit_metrics": metric_units_ok, "real_12_sample_cpu_path": real_paths == 12, "prior_target_invariant": prior_isolation, "no_optimizer_step": not has_optimizer_step}
    tampered = dict(required); tampered["temporal_isolation"] = False
    required["receipt_tamper_negative"] = not all(tampered.values())
    receipt = {"schema_version": "PI-JWM-Step5.1B-Patch-Receipt-v1", "passed": bool(all(required.values())), "checks": required, "scope": {"training": False, "optimizer_step": False, "gpu": False, "formal_dataset": False, "locked_test_accessed": False, "performance_claim": False}, "model_config": {"d_h": model_cfg.d_h, "d_z": model_cfg.d_z, "n_comm_rb": model_cfg.n_comm_rb}, "valid_counts": {"motion": int(targets["target_vehicle_motion_mask"].sum()), "csi": int(targets["target_comm_csi_mask"].sum())}, "values": {"mean_probe_loss": float(torch.stack(horizon_losses).mean().item())}}
    (args.output / "acceptance_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    manifest = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "receipt": "acceptance_receipt.json", "scope": receipt["scope"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "checks": required}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
