"""R4 objective envelope preserving the frozen R3/R2 target semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .r3_objective import ObjectiveTerm, _continuous_mask, compute_r3_objective
from .r3_preflight_data import CONTINUOUS_STATE_KEYS, ExplicitStateBatch
from .r3_world_model import R3RolloutOutput


R4_OBJECTIVE_SCHEMA = "PIJWM-R4-Controlled-Objective-v1"


@dataclass
class R4ObjectiveReport:
    total: torch.Tensor
    terms: dict[str, ObjectiveTerm]
    auxiliary_terms: dict[str, ObjectiveTerm] = field(default_factory=dict)
    schema_version: str = R4_OBJECTIVE_SCHEMA


def compute_r4_objective(
    output: R3RolloutOutput,
    batch: ExplicitStateBatch,
) -> R4ObjectiveReport:
    """Compute frozen target terms; candidate-only terms are explicit additions."""

    reference = compute_r3_objective(output, batch)
    auxiliary_terms: dict[str, ObjectiveTerm] = {}
    total = reference.total
    probabilistic = getattr(output, "probabilistic_parameters", {})
    required = {
        "context_prior_mean",
        "context_prior_log_std",
        "context_posterior_mean",
        "context_posterior_log_std",
    }
    if probabilistic:
        has_rssm = bool(required & set(probabilistic))
        if has_rssm and not required.issubset(probabilistic):
            raise ValueError("R4 probabilistic output is missing RSSM context parameters")
        if has_rssm:
            prior_mean = probabilistic["context_prior_mean"]
            prior_log_std = probabilistic["context_prior_log_std"]
            posterior_mean = probabilistic["context_posterior_mean"]
            posterior_log_std = probabilistic["context_posterior_log_std"]
            for value in (prior_mean, prior_log_std, posterior_mean, posterior_log_std):
                if not torch.isfinite(value).all():
                    raise ValueError("R4 RSSM parameters contain NaN or Inf")
            posterior_variance = torch.exp(2.0 * posterior_log_std)
            prior_variance = torch.exp(2.0 * prior_log_std)
            elementwise = 0.5 * (
                2.0 * (prior_log_std - posterior_log_std)
                + (
                    posterior_variance
                    + torch.square(posterior_mean - prior_mean)
                )
                / prior_variance
                - 1.0
            )
            kl = elementwise.mean()
            count = int(elementwise.numel())
            auxiliary_terms["rssm_kl"] = ObjectiveTerm(
                status="computed",
                value=float(kl.detach().cpu().item()),
                numerator=float(elementwise.detach().sum().cpu().item()),
                denominator=float(count),
                count=count,
                reason=None,
            )
            total = total + 1.0e-3 * kl

        log_variance_keys = {
            f"{name}_log_variance" for name in CONTINUOUS_STATE_KEYS
        }
        has_heteroscedastic = bool(log_variance_keys & set(probabilistic))
        if has_heteroscedastic:
            if not log_variance_keys.issubset(probabilistic):
                raise ValueError("R4 heteroscedastic output is incomplete")
            selected_losses: list[torch.Tensor] = []
            for name in CONTINUOUS_STATE_KEYS:
                prediction = output.predicted_explicit[name]
                target = batch.target[name][:, : prediction.shape[1]].to(prediction.dtype)
                log_variance = probabilistic[f"{name}_log_variance"]
                if log_variance.shape != prediction.shape or target.shape != prediction.shape:
                    raise ValueError(f"R4 heteroscedastic shape mismatch for {name}")
                mask = _continuous_mask(batch, name, target)
                nll = 0.5 * (
                    torch.exp(-log_variance) * torch.square(target - prediction)
                    + log_variance
                )
                if mask.any():
                    selected_losses.append(nll[mask])
            if not selected_losses:
                raise ValueError("R4 heteroscedastic objective has no observed targets")
            selected = torch.cat(selected_losses)
            if not torch.isfinite(selected).all():
                raise ValueError("R4 heteroscedastic NLL contains NaN or Inf")
            nll_mean = selected.mean()
            count = int(selected.numel())
            auxiliary_terms["heteroscedastic_nll"] = ObjectiveTerm(
                status="computed",
                value=float(nll_mean.detach().cpu().item()),
                numerator=float(selected.detach().sum().cpu().item()),
                denominator=float(count),
                count=count,
                reason=None,
            )
            total = total + 0.1 * nll_mean

        hurdle_keys = {
            "active_rate_log_location",
            "active_rate_log_std",
            "active_rate_raw_mean",
            "active_rate_normalization_mean",
            "active_rate_normalization_scale",
        }
        has_hurdle = bool(hurdle_keys & set(probabilistic))
        if has_hurdle:
            if not hurdle_keys.issubset(probabilistic):
                raise ValueError("R4 hurdle output is incomplete")
            location = probabilistic["active_rate_log_location"]
            log_std = probabilistic["active_rate_log_std"]
            rate_mean = probabilistic["active_rate_normalization_mean"]
            rate_scale = probabilistic["active_rate_normalization_scale"]
            target_normalized = batch.target["information_edge_state"][
                :, : location.shape[1], :, 12
            ].to(location.dtype)
            raw_target = target_normalized * rate_scale + rate_mean
            active = batch.target["information_link_activity"][
                :, : location.shape[1]
            ].bool()
            active_mask = batch.target["information_link_activity_mask"][
                :, : location.shape[1]
            ].bool()
            feature_mask = batch.target["information_edge_feature_mask"][
                :, : location.shape[1], :, 12
            ].bool()
            selected_mask = active & active_mask & feature_mask & (raw_target > 0.0)
            if not selected_mask.any():
                auxiliary_terms["hurdle_active_rate_nll"] = ObjectiveTerm(
                    status="not_computable",
                    value=None,
                    numerator=None,
                    denominator=0.0,
                    count=0,
                    reason="no active observed strictly-positive-rate targets",
                )
            else:
                safe_raw_target = raw_target.clamp_min(1.0e-12)
                log_target = torch.log(safe_raw_target)
                variance = torch.exp(2.0 * log_std)
                nll = (
                    0.5 * torch.square(log_target - location) / variance
                    + log_std
                    + log_target
                    + 0.5 * torch.log(location.new_tensor(2.0 * torch.pi))
                )
                selected = nll[selected_mask]
                if not torch.isfinite(selected).all():
                    raise ValueError("R4 hurdle NLL contains NaN or Inf")
                nll_mean = selected.mean()
                count = int(selected.numel())
                auxiliary_terms["hurdle_active_rate_nll"] = ObjectiveTerm(
                    status="computed",
                    value=float(nll_mean.detach().cpu().item()),
                    numerator=float(selected.detach().sum().cpu().item()),
                    denominator=float(count),
                    count=count,
                    reason=None,
                )
                total = total + 0.1 * nll_mean
    return R4ObjectiveReport(
        total=total,
        terms=reference.terms,
        auxiliary_terms=auxiliary_terms,
    )


__all__ = [
    "ObjectiveTerm",
    "R4_OBJECTIVE_SCHEMA",
    "R4ObjectiveReport",
    "compute_r4_objective",
]
