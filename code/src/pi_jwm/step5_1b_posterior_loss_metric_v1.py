"""Definition 05 per-horizon posterior, loss, KL and raw-unit metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class Step5_1BConfig:
    target_dim: int = 4
    csi_dim: int = 1
    latent_dim: int = 4
    hidden_dim: int = 32
    d_h: int | None = None
    d_z: int | None = None
    d_encoder: int | None = None

    @property
    def resolved_d_h(self):
        return int(self.d_h if self.d_h is not None else self.hidden_dim)

    @property
    def resolved_d_z(self):
        return int(self.d_z if self.d_z is not None else self.latent_dim)

    @property
    def resolved_d_encoder(self):
        return int(self.d_encoder if self.d_encoder is not None else self.latent_dim)


@dataclass
class GaussianOutput:
    mean: torch.Tensor
    log_std: torch.Tensor
    sample: torch.Tensor


class _MaskedTargetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if value.shape != mask.shape or value.ndim != 4:
            raise ValueError("target and mask must have shape [B,L,S,F] and match")
        valid = mask.bool()
        encoded = self.net(torch.cat((value * valid.to(value.dtype), valid.to(value.dtype)), dim=-1))
        return encoded * valid.any(dim=-1, keepdim=True).to(encoded.dtype)


class TargetEncoder(nn.Module):
    """Independent Motion/CSI encoders preserving [B,L,S,D]."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__()
        self.motion = _MaskedTargetEncoder(config.target_dim, config.hidden_dim, config.resolved_d_encoder)
        self.csi = _MaskedTargetEncoder(config.csi_dim, config.hidden_dim, config.resolved_d_encoder)


class _FuturePosterior(nn.Module):
    def __init__(self, d_h: int, d_encoder: int, d_z: int):
        super().__init__()
        self.mean = nn.Linear(d_h + d_encoder, d_z)
        self.log_std = nn.Linear(d_h + d_encoder, d_z)

    def forward(self, h: torch.Tensor, target_embedding: torch.Tensor, *, generator=None) -> GaussianOutput:
        if h.ndim != 4 or target_embedding.ndim != 4 or h.shape[:-1] != target_embedding.shape[:-1]:
            raise ValueError("future posterior inputs must be [B,L,S,D]")
        x = torch.cat((h, target_embedding), dim=-1)
        mean = self.mean(x); log_std = self.log_std(x).clamp(-5.0, 2.0)
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return GaussianOutput(mean, log_std, mean + eps * log_std.exp())


class FuturePosterior(nn.Module):
    """Parameter-separated training-only physical/communication teachers."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__()
        self.phy_future_posterior = _FuturePosterior(config.resolved_d_h, config.resolved_d_encoder, config.resolved_d_z)
        self.comm_future_posterior = _FuturePosterior(config.resolved_d_h, config.resolved_d_encoder, config.resolved_d_z)

    def forward(self, h_phy, h_comm, e_motion, e_csi, *, generator=None):
        return {"physical": self.phy_future_posterior(h_phy, e_motion, generator=generator), "communication": self.comm_future_posterior(h_comm, e_csi, generator=generator)}


class PosteriorTeacher(nn.Module):
    """Compatibility single-family wrapper; formal path uses FuturePosterior."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__(); self.teacher = _FuturePosterior(config.resolved_d_h, config.resolved_d_encoder, config.resolved_d_z)

    def forward(self, h, target_embedding, *, generator=None):
        if h.ndim == 3:
            out = self.teacher(h[:, None], target_embedding[:, None], generator=generator)
            return GaussianOutput(out.mean[:, 0], out.log_std[:, 0], out.sample[:, 0])
        return self.teacher(h, target_embedding, generator=generator)


class PriorPredictor(nn.Module):
    """Legacy helper retained for import compatibility; never formal prior path."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__(); self.mean = nn.Linear(config.resolved_d_h, config.resolved_d_z); self.log_std = nn.Linear(config.resolved_d_h, config.resolved_d_z)

    def forward(self, h, *, generator=None):
        mean = self.mean(h); log_std = self.log_std(h).clamp(-5.0, 2.0); eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return GaussianOutput(mean, log_std, mean + eps * log_std.exp())


def _masked_mean(value, mask):
    m = mask.to(value.dtype); count = m.sum(); return (value * m).sum() / count.clamp_min(1.0), count


def masked_family_mse(prediction, target, mask):
    if prediction.shape != target.shape or target.shape != mask.shape: raise ValueError("prediction, target and mask must match")
    return _masked_mean((prediction - target).square(), mask.bool())


def family_horizon_mse(prediction, target, mask):
    if prediction.ndim != 4 or prediction.shape != target.shape or target.shape != mask.shape: raise ValueError("family tensors must have shape [B,L,S,F]")
    valid = mask.bool(); error = (prediction - target).square() * valid; count = valid.sum((-2, -1)); total = error.sum((-2, -1))
    return {"loss": total / count.clamp_min(1).to(error.dtype), "sum": total, "count": count, "available": count > 0}


def _normalize(raw, stats):
    return (raw - raw.new_tensor(stats["mean"])) / raw.new_tensor(stats["std"]).clamp_min(1e-12)


def normalized_prediction_loss(prediction_raw, target_raw, mask, stats):
    return masked_family_mse(_normalize(prediction_raw, stats), _normalize(target_raw, stats), mask)


@dataclass
class KLResult:
    raw: torch.Tensor
    adjusted: torch.Tensor
    count: torch.Tensor
    per_dim_raw: torch.Tensor
    per_dim_adjusted: torch.Tensor
    family_raw: torch.Tensor
    family_adjusted: torch.Tensor

    def __iter__(self):
        yield self.raw; yield self.adjusted; yield self.count


def diagonal_gaussian_kl(q_mean, q_log_std, p_mean, p_log_std, eligibility_mask, *, free_bits=0.0):
    if any(x.shape != q_mean.shape for x in (q_log_std, p_mean, p_log_std)) or eligibility_mask.shape != q_mean.shape[:-1] or free_bits < 0:
        raise ValueError("invalid Gaussian shapes, eligibility mask or free bits")
    q_var, p_var = torch.exp(2 * q_log_std), torch.exp(2 * p_log_std)
    per_dim = p_log_std - q_log_std + (q_var + (q_mean - p_mean).square()) / (2 * p_var) - 0.5
    adjusted_dim = per_dim.clamp_min(float(free_bits)); eligible = eligibility_mask.bool()
    raw, count = _masked_mean(per_dim.sum(-1), eligible); adjusted, _ = _masked_mean(adjusted_dim.sum(-1), eligible)
    return KLResult(raw, adjusted, count, per_dim, adjusted_dim, raw, adjusted)


def complete_total_loss(motion_prediction, motion_target, motion_mask, csi_prediction, csi_target, csi_mask, q_phy, p_phy, phy_eligibility, q_comm, p_comm, comm_eligibility, *, beta_kl, free_bits=0.0):
    mot, csi = family_horizon_mse(motion_prediction, motion_target, motion_mask), family_horizon_mse(csi_prediction, csi_target, csi_mask)
    phy = diagonal_gaussian_kl(*q_phy, *p_phy, phy_eligibility, free_bits=free_bits); comm = diagonal_gaussian_kl(*q_comm, *p_comm, comm_eligibility, free_bits=free_bits)
    pred, kl = 0.5 * mot["loss"] + 0.5 * csi["loss"], phy.adjusted + comm.adjusted
    return {"L_Mot": mot["loss"], "L_CSI": csi["loss"], "L_Pred": pred, "L_KL_Phy_raw": phy.raw, "L_KL_Phy_adjusted": phy.adjusted, "L_KL_Comm_raw": comm.raw, "L_KL_Comm_adjusted": comm.adjusted, "L_KL": kl, "L_Total": pred + float(beta_kl) * kl, "motion_count": mot["count"], "csi_count": csi["count"], "motion_available": mot["available"], "csi_available": csi["available"], "kl_phy_count": phy.count, "kl_comm_count": comm.count, "kl_phy_per_dim_raw": phy.per_dim_raw, "kl_phy_per_dim_adjusted": phy.per_dim_adjusted, "kl_comm_per_dim_raw": comm.per_dim_raw, "kl_comm_per_dim_adjusted": comm.per_dim_adjusted}


def prediction_loss(motion_prediction, motion_target, motion_mask, csi_prediction, csi_target, csi_mask, *, lambda_motion=0.5, lambda_csi=0.5):
    motion, motion_count = masked_family_mse(motion_prediction, motion_target, motion_mask); csi, csi_count = masked_family_mse(csi_prediction, csi_target, csi_mask)
    return {"motion": motion, "csi": csi, "prediction": lambda_motion * motion + lambda_csi * csi, "motion_valid_count": motion_count, "csi_valid_count": csi_count}


def validation_loss(per_horizon_prediction_loss):
    return per_horizon_prediction_loss.mean() if per_horizon_prediction_loss.numel() else per_horizon_prediction_loss.new_zeros(())


def motion_metrics(prediction_raw, target_raw, mask, *, xyz_aggregate=False, stats=None):
    if stats is not None: prediction_raw, target_raw = _normalize(prediction_raw, stats), _normalize(target_raw, stats)
    error, valid = prediction_raw - target_raw, mask.bool(); result = {"units": ("m", "m", "m", "m/s"), "valid_count": valid.sum((-2, -1))}
    for i, name in enumerate(("delta_x", "delta_y", "delta_z", "next_speed")):
        m, e = valid[..., i], error[..., i]; result[f"{name}_mae"] = e.abs().masked_fill(~m, 0).sum(-1) / m.sum(-1).clamp_min(1); result[f"{name}_rmse"] = torch.sqrt(e.square().masked_fill(~m, 0).sum(-1) / m.sum(-1).clamp_min(1))
    if xyz_aggregate:
        m, e = valid[..., :3], error[..., :3]; result["xyz_mae"] = e.abs().masked_fill(~m, 0).sum((-2, -1)) / m.sum((-2, -1)).clamp_min(1); result["xyz_rmse"] = torch.sqrt(e.square().masked_fill(~m, 0).sum((-2, -1)) / m.sum((-2, -1)).clamp_min(1))
    return result


def csi_metrics(prediction_raw, target_raw, mask, *, mean=None, std=None):
    if mean is not None and std is not None: prediction_raw, target_raw = prediction_raw * float(std) + float(mean), target_raw * float(std) + float(mean)
    error, valid = prediction_raw - target_raw, mask.bool(); count = valid.sum((-2, -1)); return {"mae": error.abs().masked_fill(~valid, 0).sum((-2, -1)) / count.clamp_min(1), "rmse": torch.sqrt(error.square().masked_fill(~valid, 0).sum((-2, -1)) / count.clamp_min(1)), "valid_count": count, "unit": "dB"}


__all__ = ["Step5_1BConfig", "GaussianOutput", "TargetEncoder", "FuturePosterior", "PosteriorTeacher", "PriorPredictor", "masked_family_mse", "family_horizon_mse", "normalized_prediction_loss", "KLResult", "diagonal_gaussian_kl", "complete_total_loss", "prediction_loss", "validation_loss", "motion_metrics", "csi_metrics"]
