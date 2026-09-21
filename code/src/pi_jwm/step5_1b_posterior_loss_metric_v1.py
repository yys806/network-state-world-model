"""Definition 05 posterior, prediction loss, KL and metric primitives.

This module is deliberately training-loop free.  Future targets are accepted only
by :class:`TargetEncoder` and :class:`PosteriorTeacher`; prior callers need not
and cannot pass them through these primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class Step5_1BConfig:
    target_dim: int = 4
    csi_dim: int = 1
    latent_dim: int = 16
    hidden_dim: int = 32


@dataclass
class GaussianOutput:
    mean: torch.Tensor
    log_std: torch.Tensor
    sample: torch.Tensor


class _MaskedTargetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if value.shape != mask.shape or value.ndim != 4:
            raise ValueError("target and mask must have shape [B,H,S,F] and match")
        valid = mask.bool()
        encoded = self.net(value * valid.to(value.dtype))
        # A component mask is reduced per horizon/slot; empty slots remain zero.
        slot_valid = valid.any(dim=-1)
        encoded = encoded * slot_valid[..., None].to(encoded.dtype)
        denom = slot_valid.sum(dim=1, keepdim=False).clamp_min(1).to(encoded.dtype)
        return encoded.sum(dim=1) / denom[..., None]


class TargetEncoder(nn.Module):
    """Two separate target-only encoders; no graph/rule state is accepted."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__()
        self.motion = _MaskedTargetEncoder(config.target_dim, config.hidden_dim, config.latent_dim)
        self.csi = _MaskedTargetEncoder(config.csi_dim, config.hidden_dim, config.latent_dim)


class PosteriorTeacher(nn.Module):
    """Training-only q(z|h,target) path with mean and reparameterized sample."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__()
        self.mean = nn.Linear(config.latent_dim * 2, config.latent_dim)
        self.log_std = nn.Linear(config.latent_dim * 2, config.latent_dim)

    def forward(self, h: torch.Tensor, target_embedding: torch.Tensor, *, generator=None) -> GaussianOutput:
        if h.shape != target_embedding.shape or h.ndim != 3:
            raise ValueError("posterior h and target embedding must be [B,S,D]")
        x = torch.cat((h, target_embedding), dim=-1)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-5.0, 2.0)
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return GaussianOutput(mean, log_std, mean + eps * log_std.exp())


class PriorPredictor(nn.Module):
    """Prior path intentionally has no target argument or target-dependent state."""
    def __init__(self, config: Step5_1BConfig):
        super().__init__()
        self.mean = nn.Linear(config.latent_dim, config.latent_dim)
        self.log_std = nn.Linear(config.latent_dim, config.latent_dim)

    def forward(self, h: torch.Tensor, *, generator=None) -> GaussianOutput:
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return GaussianOutput(mean, log_std, mean + eps * log_std.exp())


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.to(value.dtype)
    count = m.sum()
    return (value * m).sum() / count.clamp_min(1.0), count


def masked_family_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target and mask must match")
    return _safe_mean(torch.square(prediction - target), mask.bool())


def _normalize(raw: torch.Tensor, stats: Mapping[str, object]) -> torch.Tensor:
    mean = raw.new_tensor(stats["mean"])
    std = raw.new_tensor(stats["std"]).clamp_min(1e-12)
    return (raw - mean) / std


def normalized_prediction_loss(
    prediction_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    stats: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-stat raw decoder output -> normalized MSE bridge."""
    prediction = _normalize(prediction_raw, stats)
    target = _normalize(target_raw, stats)
    return masked_family_mse(prediction, target, mask)


def diagonal_gaussian_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
    eligibility_mask: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw KL, free-bit-adjusted KL, and eligible slot count."""
    if any(x.shape != q_mean.shape for x in (q_log_std, p_mean, p_log_std)):
        raise ValueError("Gaussian parameters must have identical shapes")
    if eligibility_mask.shape != q_mean.shape[:-1]:
        raise ValueError("eligibility mask must omit latent dimension")
    if free_bits < 0:
        raise ValueError("free_bits must be non-negative")
    q_var, p_var = torch.exp(2 * q_log_std), torch.exp(2 * p_log_std)
    per_dim = p_log_std - q_log_std + (q_var + (q_mean - p_mean).square()) / (2 * p_var) - 0.5
    per_slot = per_dim.sum(dim=-1)
    raw, count = _safe_mean(per_slot, eligibility_mask.bool())
    adjusted = _safe_mean(torch.clamp(per_slot, min=float(free_bits)), eligibility_mask.bool())[0]
    return raw, adjusted, count


def prediction_loss(
    motion_prediction: torch.Tensor,
    motion_target: torch.Tensor,
    motion_mask: torch.Tensor,
    csi_prediction: torch.Tensor,
    csi_target: torch.Tensor,
    csi_mask: torch.Tensor,
    *,
    lambda_motion: float = 0.5,
    lambda_csi: float = 0.5,
) -> dict[str, torch.Tensor]:
    motion, motion_count = masked_family_mse(motion_prediction, motion_target, motion_mask)
    csi, csi_count = masked_family_mse(csi_prediction, csi_target, csi_mask)
    return {"motion": motion, "csi": csi, "prediction": lambda_motion * motion + lambda_csi * csi,
            "motion_valid_count": motion_count, "csi_valid_count": csi_count}


def validation_loss(per_horizon_prediction_loss: torch.Tensor) -> torch.Tensor:
    return per_horizon_prediction_loss.mean() if per_horizon_prediction_loss.numel() else per_horizon_prediction_loss.new_zeros(())


def motion_metrics(prediction_normalized: torch.Tensor, target_normalized: torch.Tensor, mask: torch.Tensor, *, xyz_aggregate: bool = False) -> dict[str, torch.Tensor]:
    error = prediction_normalized - target_normalized
    valid = mask.bool()
    mae, _ = _safe_mean(error.abs(), valid)
    rmse, _ = _safe_mean(error.square(), valid)
    result = {"mae": error.abs().masked_fill(~valid, 0).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1),
              "rmse": torch.sqrt(error.square().masked_fill(~valid, 0).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)),
              "overall_mae": mae, "overall_rmse": torch.sqrt(rmse)}
    if xyz_aggregate:
        xyz = valid[..., :3]
        result["xyz_mae"] = error[..., :3].abs().masked_fill(~xyz, 0).sum(dim=(-2, -1)) / xyz.sum(dim=(-2, -1)).clamp_min(1)
        result["xyz_rmse"] = torch.sqrt(error[..., :3].square().masked_fill(~xyz, 0).sum(dim=(-2, -1)) / xyz.sum(dim=(-2, -1)).clamp_min(1))
    return result


def csi_metrics(prediction_normalized: torch.Tensor, target_normalized: torch.Tensor, mask: torch.Tensor, *, mean: float, std: float) -> dict[str, torch.Tensor]:
    error = (prediction_normalized - target_normalized) * float(std)
    valid = mask.bool()
    denom = valid.sum(dim=(-2, -1)).clamp_min(1)
    return {"mae": error.abs().masked_fill(~valid, 0).sum(dim=(-2, -1)) / denom,
            "rmse": torch.sqrt(error.square().masked_fill(~valid, 0).sum(dim=(-2, -1)) / denom),
            "unit": torch.tensor(1.0, device=error.device)}


__all__ = ["Step5_1BConfig", "GaussianOutput", "TargetEncoder", "PosteriorTeacher", "PriorPredictor", "masked_family_mse", "normalized_prediction_loss", "diagonal_gaussian_kl", "prediction_loss", "validation_loss", "motion_metrics", "csi_metrics"]
