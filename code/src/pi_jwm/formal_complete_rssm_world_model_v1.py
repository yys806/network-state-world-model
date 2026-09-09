"""Complete RSSM dynamics on the formal PI-JWM dual-graph rollout path.

The deterministic dual-graph model remains the graph decoder.  This module adds
an action-conditioned stochastic prior for deployment and an observation-
conditioned posterior teacher used only while training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .formal_dual_graph_world_model_v1 import (
    COMPONENT_FEATURES,
    FormalDualGraphWorldModel,
    FormalWorldModelConfig,
)


FORMAL_COMPLETE_RSSM_SCHEMA = "PIJWM-Formal-Complete-RSSM-v1"


@dataclass(frozen=True)
class FormalCompleteRSSMConfig:
    hidden_dim: int = 16
    stochastic_dim: int = 8
    history_steps: int = 8
    horizon_steps: int = 20
    overshooting_distance: int = 2
    residual_state_prediction: bool = True
    zero_init_residual_state_heads: bool = False
    residual_state_scale: float = 1.0
    deterministic_rule_layer: bool = False
    rule_layer_stats: Mapping[str, object] | None = None
    n_rb: int = 1
    use_system_energy_head: bool = False
    link_activity_method: str = "absolute_v1"
    link_activity_pos_weight: float = 1.0
    link_activity_missing_history_prior: float | None = None
    latent_dynamics: str = "complete_rssm_prior_posterior_v1"
    training_posterior_teacher: bool = True
    deployment_prior_only: bool = True
    rssm_residual_head_initialization: str = "zero_when_requested_v1"
    node_x_residual_loss_contract: str = "none"


def _masked_pool(value: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    numeric = present.to(value.dtype).unsqueeze(-1)
    return (value * numeric).sum(dim=-2) / numeric.sum(dim=-2).clamp_min(1.0)


def _normal_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = value.chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)


def _sample(mean: torch.Tensor, log_std: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mean
    return mean + torch.randn_like(mean) * torch.exp(log_std)


class FormalCompleteRSSMWorldModel(nn.Module):
    """Formal dual-graph decoder with explicit RSSM prior/posterior semantics."""

    schema_version = FORMAL_COMPLETE_RSSM_SCHEMA
    model_version = "formal_complete_rssm_v1_1"
    latent_dynamics = "complete_rssm_prior_posterior_v1"

    def __init__(self, config: FormalCompleteRSSMConfig) -> None:
        super().__init__()
        if min(
            config.hidden_dim,
            config.stochastic_dim,
            config.history_steps,
            config.horizon_steps,
            config.overshooting_distance,
        ) <= 0:
            raise ValueError("RSSM dimensions and rollout lengths must be positive")
        if config.latent_dynamics != self.latent_dynamics:
            raise ValueError("latent_dynamics does not identify the complete RSSM")
        if not config.training_posterior_teacher or not config.deployment_prior_only:
            raise ValueError("complete RSSM requires posterior teacher and prior-only deployment")
        if config.rssm_residual_head_initialization != "zero_when_requested_v1":
            raise ValueError("unsupported RSSM residual-head initialization contract")
        self.config = config
        hidden = config.hidden_dim
        stochastic = config.stochastic_dim
        self.base = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=hidden,
                history_steps=config.history_steps,
                horizon_steps=config.horizon_steps,
                residual_state_prediction=config.residual_state_prediction,
                zero_init_residual_state_heads=config.zero_init_residual_state_heads,
                residual_state_scale=config.residual_state_scale,
                deterministic_rule_layer=config.deterministic_rule_layer,
                rule_layer_stats=config.rule_layer_stats,
                n_rb=config.n_rb,
                use_system_energy_head=config.use_system_energy_head,
                link_activity_method=config.link_activity_method,
                link_activity_pos_weight=config.link_activity_pos_weight,
                link_activity_missing_history_prior=config.link_activity_missing_history_prior,
            )
        )
        state_dim = sum(COMPONENT_FEATURES.values())
        observation_dim = state_dim + 3
        self.rssm_context = nn.Linear(state_dim, hidden)
        self.rssm_initial = nn.Linear(hidden, stochastic)
        self.rssm_action = nn.Linear(8, hidden)
        self.rssm_transition = nn.GRUCell(stochastic + hidden, hidden)
        self.rssm_prior = nn.Linear(hidden, stochastic * 2)
        self.rssm_observation = nn.Linear(observation_dim, hidden)
        self.rssm_posterior = nn.Linear(hidden * 2, stochastic * 2)
        decoder_dim = hidden + stochastic
        self.rssm_prior_state_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, size) for name, size in COMPONENT_FEATURES.items()}
        )
        self.rssm_teacher_state_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, size) for name, size in COMPONENT_FEATURES.items()}
        )
        self.rssm_prior_dag_head = nn.Linear(decoder_dim, 3)
        self.rssm_teacher_dag_head = nn.Linear(decoder_dim, 3)
        self.rssm_prior_presence_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, 1) for name in COMPONENT_FEATURES}
        )
        self.rssm_teacher_presence_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, 1) for name in COMPONENT_FEATURES}
        )
        self.rssm_prior_link_head = nn.Linear(decoder_dim, 1)
        self.rssm_teacher_link_head = nn.Linear(decoder_dim, 1)
        self.rssm_prior_lifecycle_head = nn.Linear(decoder_dim, 5)
        self.rssm_teacher_lifecycle_head = nn.Linear(decoder_dim, 5)
        self.rssm_prior_dag_release_head = nn.Linear(decoder_dim, 1)
        self.rssm_teacher_dag_release_head = nn.Linear(decoder_dim, 1)
        self.rssm_prior_dag_edge_head = nn.Linear(decoder_dim, 1)
        self.rssm_teacher_dag_edge_head = nn.Linear(decoder_dim, 1)
        self.rssm_prior_energy_head = nn.Linear(decoder_dim, 1)
        self.rssm_teacher_energy_head = nn.Linear(decoder_dim, 1)
        if config.zero_init_residual_state_heads:
            continuous_residual_heads = [
                *self.rssm_prior_state_heads.values(),
                *self.rssm_teacher_state_heads.values(),
                self.rssm_prior_dag_head,
                self.rssm_teacher_dag_head,
                self.rssm_prior_energy_head,
                self.rssm_teacher_energy_head,
            ]
            for head in continuous_residual_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    @staticmethod
    def _state_summary(
        values: Mapping[str, torch.Tensor],
        *,
        suffix: str,
        time_index: int | None = None,
    ) -> torch.Tensor:
        pooled = []
        for name in COMPONENT_FEATURES:
            state = values[f"{name}_state"]
            present = values[f"{name}_present"].bool()
            if time_index is not None:
                state = state[:, time_index]
                present = present[:, time_index]
            pooled.append(_masked_pool(state, present))
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def _target_observation(target: Mapping[str, torch.Tensor], step: int) -> torch.Tensor:
        pooled = []
        for name in COMPONENT_FEATURES:
            pooled.append(
                _masked_pool(
                    target[f"{name}_state"][:, step],
                    target[f"{name}_present"][:, step].bool(),
                )
            )
        pooled.append(
            _masked_pool(
                target["task_dag_state"][:, step],
                target["task_dag_state_present"][:, step].bool(),
            )
        )
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def _action_summary(future_action: Mapping[str, torch.Tensor], step: int) -> torch.Tensor:
        return _masked_pool(
            future_action["task_action"][:, step],
            future_action["task_action_present"][:, step].bool(),
        )

    @staticmethod
    def _apply_state_correction(
        output: dict[str, torch.Tensor],
        decoder: torch.Tensor,
        heads: nn.ModuleDict,
        dag_head: nn.Linear,
        presence_heads: nn.ModuleDict,
        link_head: nn.Linear,
        lifecycle_head: nn.Linear,
        dag_release_head: nn.Linear,
        dag_edge_head: nn.Linear,
        energy_head: nn.Linear,
    ) -> dict[str, torch.Tensor]:
        corrected = dict(output)
        for name, head in heads.items():
            delta = head(decoder).unsqueeze(-2)
            corrected[f"{name}_state_mean"] = output[f"{name}_state_mean"] + delta
            corrected[f"{name}_state"] = corrected[f"{name}_state_mean"]
            if name == "node":
                corrected["rssm_node_state_correction"] = delta.expand_as(
                    output[f"{name}_state_mean"]
                )
            corrected[f"{name}_presence_logits"] = (
                output[f"{name}_presence_logits"]
                + presence_heads[name](decoder).squeeze(-1).unsqueeze(-1)
            )
        corrected["task_dag_state_mean"] = (
            output["task_dag_state_mean"] + dag_head(decoder).unsqueeze(-2)
        )
        corrected["link_activity_logits"] = (
            output["link_activity_logits"]
            + link_head(decoder).squeeze(-1).unsqueeze(-1)
        )
        corrected["task_lifecycle_logits"] = (
            output["task_lifecycle_logits"] + lifecycle_head(decoder).unsqueeze(-2)
        )
        corrected["dag_release_logits"] = (
            output["dag_release_logits"]
            + dag_release_head(decoder).squeeze(-1).unsqueeze(-1)
        )
        corrected["dag_edge_presence_logits"] = (
            output["dag_edge_presence_logits"]
            + dag_edge_head(decoder).squeeze(-1).unsqueeze(-1)
        )
        if "uav_energy_delta_mean" in output:
            corrected["uav_energy_delta_mean"] = (
                output["uav_energy_delta_mean"]
                + energy_head(decoder).squeeze(-1).unsqueeze(-1)
            )
        return corrected

    def forward(self, batch: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor | bool]:
        history = batch["history"]
        future_action = batch["future_action"]
        base_batch = {
            "history": history,
            "future_action": future_action,
            "static": batch["static"],
        }
        base_output = self.base(base_batch)
        context_input = self._state_summary(history, suffix="", time_index=-1)
        context_hidden = torch.tanh(self.rssm_context(context_input))
        prior_hidden = context_hidden
        prior_z = self.rssm_initial(context_hidden)
        prior_means, prior_log_stds, prior_decoders = [], [], []
        for step in range(self.config.horizon_steps):
            action = torch.tanh(self.rssm_action(self._action_summary(future_action, step)))
            prior_hidden = self.rssm_transition(torch.cat((prior_z, action), dim=-1), prior_hidden)
            prior_mean, prior_log_std = _normal_parameters(self.rssm_prior(prior_hidden))
            prior_z = _sample(prior_mean, prior_log_std, self.training)
            prior_means.append(prior_mean)
            prior_log_stds.append(prior_log_std)
            prior_decoders.append(torch.cat((prior_hidden, prior_z), dim=-1))
        prior_decoder = torch.stack(prior_decoders, dim=1)
        output: dict[str, torch.Tensor | bool] = self._apply_state_correction(
            dict(base_output),
            prior_decoder,
            self.rssm_prior_state_heads,
            self.rssm_prior_dag_head,
            self.rssm_prior_presence_heads,
            self.rssm_prior_link_head,
            self.rssm_prior_lifecycle_head,
            self.rssm_prior_dag_release_head,
            self.rssm_prior_dag_edge_head,
            self.rssm_prior_energy_head,
        )
        rollout_prior_mean = torch.stack(prior_means, dim=1)
        rollout_prior_log_std = torch.stack(prior_log_stds, dim=1)
        output.update(
            rssm_rollout_prior_mean=rollout_prior_mean,
            rssm_rollout_prior_log_std=rollout_prior_log_std,
            rssm_deployment_prior_only=True,
        )
        if not self.training:
            return output
        if "target" not in batch:
            raise ValueError("training the complete RSSM requires target for posterior teacher")

        target = batch["target"]
        posterior_hidden = context_hidden
        posterior_z = self.rssm_initial(context_hidden)
        teacher_decoders, posterior_means, posterior_log_stds = [], [], []
        posterior_path_prior_means, posterior_path_prior_log_stds = [], []
        for step in range(self.config.horizon_steps):
            action = torch.tanh(self.rssm_action(self._action_summary(future_action, step)))
            posterior_hidden = self.rssm_transition(
                torch.cat((posterior_z, action), dim=-1), posterior_hidden
            )
            path_prior_mean, path_prior_log_std = _normal_parameters(
                self.rssm_prior(posterior_hidden)
            )
            observation = torch.tanh(
                self.rssm_observation(self._target_observation(target, step))
            )
            posterior_mean, posterior_log_std = _normal_parameters(
                self.rssm_posterior(torch.cat((posterior_hidden, observation), dim=-1))
            )
            posterior_z = _sample(posterior_mean, posterior_log_std, True)
            posterior_path_prior_means.append(path_prior_mean)
            posterior_path_prior_log_stds.append(path_prior_log_std)
            posterior_means.append(posterior_mean)
            posterior_log_stds.append(posterior_log_std)
            teacher_decoders.append(torch.cat((posterior_hidden, posterior_z), dim=-1))
        teacher_output = self._apply_state_correction(
            dict(base_output),
            torch.stack(teacher_decoders, dim=1),
            self.rssm_teacher_state_heads,
            self.rssm_teacher_dag_head,
            self.rssm_teacher_presence_heads,
            self.rssm_teacher_link_head,
            self.rssm_teacher_lifecycle_head,
            self.rssm_teacher_dag_release_head,
            self.rssm_teacher_dag_edge_head,
            self.rssm_teacher_energy_head,
        )
        for key, value in teacher_output.items():
            output[f"training_{key}"] = value
        posterior_mean = torch.stack(posterior_means, dim=1)
        posterior_log_std = torch.stack(posterior_log_stds, dim=1)
        output.update(
            rssm_posterior_path_prior_mean=torch.stack(posterior_path_prior_means, dim=1),
            rssm_posterior_path_prior_log_std=torch.stack(posterior_path_prior_log_stds, dim=1),
            rssm_posterior_mean=posterior_mean,
            rssm_posterior_log_std=posterior_log_std,
            rssm_training_posterior_teacher=True,
        )
        distance = min(self.config.overshooting_distance, self.config.horizon_steps)
        if distance > 1:
            output["rssm_overshooting_prior_mean"] = rollout_prior_mean[:, distance - 1 :]
            output["rssm_overshooting_prior_log_std"] = rollout_prior_log_std[:, distance - 1 :]
            output["rssm_overshooting_posterior_mean"] = posterior_mean[:, distance - 1 :]
            output["rssm_overshooting_posterior_log_std"] = posterior_log_std[:, distance - 1 :]
        else:
            output["rssm_overshooting_prior_mean"] = rollout_prior_mean
            output["rssm_overshooting_prior_log_std"] = rollout_prior_log_std
            output["rssm_overshooting_posterior_mean"] = posterior_mean
            output["rssm_overshooting_posterior_log_std"] = posterior_log_std
        return output


__all__ = [
    "FORMAL_COMPLETE_RSSM_SCHEMA",
    "FormalCompleteRSSMConfig",
    "FormalCompleteRSSMWorldModel",
]
