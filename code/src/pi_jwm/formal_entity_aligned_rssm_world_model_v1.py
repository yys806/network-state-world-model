"""Entity-aligned complete RSSM on the formal PI-JWM dual-graph rollout."""

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


SCHEMA_VERSION = "PIJWM-Formal-Entity-Aligned-RSSM-v1"


@dataclass(frozen=True)
class FormalEntityAlignedRSSMConfig:
    hidden_dim: int = 32
    stochastic_dim: int = 16
    history_steps: int = 8
    horizon_steps: int = 20
    overshooting_distance: int = 5
    residual_state_prediction: bool = True
    zero_init_residual_state_heads: bool = True
    residual_state_scale: float = 1.0
    deterministic_rule_layer: bool = False
    rule_layer_stats: Mapping[str, object] | None = None
    n_rb: int = 1
    use_system_energy_head: bool = False
    link_activity_method: str = "absolute_v1"
    link_activity_pos_weight: float = 1.0
    link_activity_missing_history_prior: float | None = None
    slot_seconds: float = 0.1
    node_position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    node_motion_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    node_motion_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    latent_dynamics: str = "entity_aligned_complete_rssm_prior_posterior_v1"
    training_posterior_teacher: bool = True
    deployment_prior_only: bool = True
    rssm_residual_head_initialization: str = "zero_when_requested_v1"
    entity_latent_layout: str = "node_physical_edge_flow_task_v1"
    node_motion_contract: str = "causal_backward_difference_v1"


def _normal_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = value.chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)


def _sample(mean: torch.Tensor, log_std: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mean
    return mean + torch.randn_like(mean) * torch.exp(log_std)


def _apply_cell(cell: nn.GRUCell, value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    shape = hidden.shape
    return cell(value.reshape(-1, value.shape[-1]), hidden.reshape(-1, shape[-1])).reshape(shape)


def _gather_task_pair(task: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 3 or edge_index.shape[1] != 2:
        raise ValueError("dag_edge_index must have canonical [batch, 2, edge] layout")
    source_index = edge_index[:, 0].clamp(0, max(task.shape[2] - 1, 0))
    target_index = edge_index[:, 1].clamp(0, max(task.shape[2] - 1, 0))
    source = torch.gather(
        task,
        2,
        source_index.unsqueeze(1).unsqueeze(-1).expand(
            task.shape[0], task.shape[1], source_index.shape[1], task.shape[-1]
        ),
    )
    target = torch.gather(
        task,
        2,
        target_index.unsqueeze(1).unsqueeze(-1).expand(
            task.shape[0], task.shape[1], target_index.shape[1], task.shape[-1]
        ),
    )
    return torch.cat((source, target), dim=-1)


class FormalEntityAlignedRSSMWorldModel(nn.Module):
    """Per-entity stochastic dynamics over the existing deterministic dual graph."""

    schema_version = SCHEMA_VERSION
    model_version = "formal_entity_aligned_rssm_v1"
    latent_dynamics = "entity_aligned_complete_rssm_prior_posterior_v1"

    def __init__(self, config: FormalEntityAlignedRSSMConfig) -> None:
        super().__init__()
        if min(
            config.hidden_dim,
            config.stochastic_dim,
            config.history_steps,
            config.horizon_steps,
            config.overshooting_distance,
        ) <= 0:
            raise ValueError("entity RSSM dimensions and rollout lengths must be positive")
        if config.latent_dynamics != self.latent_dynamics:
            raise ValueError("latent_dynamics does not identify the entity-aligned RSSM")
        if not config.training_posterior_teacher or not config.deployment_prior_only:
            raise ValueError("entity-aligned RSSM requires posterior teacher and prior-only deployment")
        if config.entity_latent_layout != "node_physical_edge_flow_task_v1":
            raise ValueError("unsupported entity latent layout")
        if config.node_motion_contract != "causal_backward_difference_v1":
            raise ValueError("unsupported node motion contract")
        if len(config.node_motion_mean) != 6 or len(config.node_motion_scale) != 6:
            raise ValueError("node motion statistics must have six components")
        if len(config.node_position_scale) != 3:
            raise ValueError("node position scale must have three components")
        self.config = config
        hidden, stochastic = config.hidden_dim, config.stochastic_dim
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
        self.initial_hidden = nn.ModuleDict(
            {name: nn.Linear(hidden, hidden) for name in COMPONENT_FEATURES}
        )
        self.initial_z = nn.ModuleDict(
            {name: nn.Linear(hidden, stochastic) for name in COMPONENT_FEATURES}
        )
        self.transitions = nn.ModuleDict(
            {
                name: nn.GRUCell(stochastic + hidden, hidden)
                for name in COMPONENT_FEATURES
            }
        )
        self.priors = nn.ModuleDict(
            {name: nn.Linear(hidden, stochastic * 2) for name in COMPONENT_FEATURES}
        )
        self.observation_encoders = nn.ModuleDict(
            {
                name: nn.Linear(width, hidden)
                for name, width in COMPONENT_FEATURES.items()
            }
        )
        self.posteriors = nn.ModuleDict(
            {name: nn.Linear(hidden * 2, stochastic * 2) for name in COMPONENT_FEATURES}
        )
        decoder_dim = hidden + stochastic
        self.prior_state_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, width) for name, width in COMPONENT_FEATURES.items()}
        )
        self.teacher_state_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, width) for name, width in COMPONENT_FEATURES.items()}
        )
        self.prior_presence_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, 1) for name in COMPONENT_FEATURES}
        )
        self.teacher_presence_heads = nn.ModuleDict(
            {name: nn.Linear(decoder_dim, 1) for name in COMPONENT_FEATURES}
        )
        self.prior_dag_state_head = nn.Linear(decoder_dim, 3)
        self.teacher_dag_state_head = nn.Linear(decoder_dim, 3)
        self.prior_link_head = nn.Linear(decoder_dim, 1)
        self.teacher_link_head = nn.Linear(decoder_dim, 1)
        self.prior_lifecycle_head = nn.Linear(decoder_dim, 5)
        self.teacher_lifecycle_head = nn.Linear(decoder_dim, 5)
        self.prior_dag_release_head = nn.Linear(decoder_dim, 1)
        self.teacher_dag_release_head = nn.Linear(decoder_dim, 1)
        self.prior_dag_edge_head = nn.Linear(decoder_dim * 2, 1)
        self.teacher_dag_edge_head = nn.Linear(decoder_dim * 2, 1)
        self.prior_energy_head = nn.Linear(decoder_dim, 1)
        self.teacher_energy_head = nn.Linear(decoder_dim, 1)
        self.motion_encoder = nn.Linear(6, hidden)
        if config.zero_init_residual_state_heads:
            continuous_heads = [
                *self.prior_state_heads.values(),
                *self.teacher_state_heads.values(),
                self.prior_dag_state_head,
                self.teacher_dag_state_head,
                self.prior_energy_head,
                self.teacher_energy_head,
            ]
            for head in continuous_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            for head in (
                self.prior_link_head,
                self.teacher_link_head,
                self.prior_lifecycle_head,
                self.teacher_lifecycle_head,
                self.prior_dag_release_head,
                self.teacher_dag_release_head,
                self.prior_dag_edge_head,
                self.teacher_dag_edge_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def _entity_masks(
        self, batch: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        static = batch["static"]
        return {
            "node": static["node_kind_index"] >= 0,
            "physical_edge": torch.all(
                static["physical_edge_endpoint_index"] >= 0, dim=-1
            ),
            "flow": static["flow_valid"].bool(),
            "task": static["task_valid"].bool(),
        }

    def _motion_context(
        self, history: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "node_motion_state" not in history or "node_motion_mask" not in history:
            raise ValueError("entity-aligned RSSM requires causal node_motion state and mask")
        motion = history["node_motion_state"][:, -1]
        mask = history["node_motion_mask"][:, -1].bool()
        if motion.shape[-1] != 6 or mask.shape != motion.shape:
            raise ValueError("node_motion state or mask has an invalid shape")
        mean = motion.new_tensor(self.config.node_motion_mean)
        scale = motion.new_tensor(self.config.node_motion_scale).clamp_min(1e-6)
        normalized = torch.where(mask, (motion - mean) / scale, torch.zeros_like(motion))
        return torch.tanh(self.motion_encoder(normalized)), motion, mask

    def _rollout_path(
        self,
        trace: Mapping[str, torch.Tensor],
        contexts: Mapping[str, torch.Tensor],
        *,
        target: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}
        for name in COMPONENT_FEATURES:
            deterministic = trace[name] + contexts[name]
            hidden = torch.tanh(self.initial_hidden[name](deterministic[:, 0]))
            z = self.initial_z[name](deterministic[:, 0])
            path_prior_means, path_prior_log_stds = [], []
            posterior_means, posterior_log_stds, decoders = [], [], []
            for step in range(self.config.horizon_steps):
                hidden = _apply_cell(
                    self.transitions[name],
                    torch.cat((z, deterministic[:, step]), dim=-1),
                    hidden,
                )
                prior_mean, prior_log_std = _normal_parameters(self.priors[name](hidden))
                path_prior_means.append(prior_mean)
                path_prior_log_stds.append(prior_log_std)
                if target is None:
                    mean, log_std = prior_mean, prior_log_std
                    z = _sample(mean, log_std, self.training)
                else:
                    observed = target[f"{name}_state"][:, step]
                    present = target[f"{name}_present"][:, step].bool()
                    observation = torch.tanh(self.observation_encoders[name](observed))
                    observation = observation * present.unsqueeze(-1).to(observation.dtype)
                    mean, log_std = _normal_parameters(
                        self.posteriors[name](torch.cat((hidden, observation), dim=-1))
                    )
                    z = _sample(mean, log_std, True)
                posterior_means.append(mean)
                posterior_log_stds.append(log_std)
                decoders.append(torch.cat((hidden, z), dim=-1))
            result[name] = {
                "prior_mean": torch.stack(path_prior_means, dim=1),
                "prior_log_std": torch.stack(path_prior_log_stds, dim=1),
                "mean": torch.stack(posterior_means, dim=1),
                "log_std": torch.stack(posterior_log_stds, dim=1),
                "decoder": torch.stack(decoders, dim=1),
            }
        return result

    def _kinematic_proposal(
        self,
        base_output: Mapping[str, torch.Tensor],
        history: Mapping[str, torch.Tensor],
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = dict(base_output)
        node_mean = output["node_state_mean"].clone()
        last_position = history["node_state"][:, -1, :, :3]
        velocity = motion[..., :3]
        valid = motion_mask[..., :3].all(dim=-1)
        position_scale = velocity.new_tensor(self.config.node_position_scale).clamp_min(1e-6)
        steps = torch.arange(
            1,
            self.config.horizon_steps + 1,
            device=velocity.device,
            dtype=velocity.dtype,
        ).view(1, -1, 1, 1)
        proposal = last_position.unsqueeze(1) + (
            velocity.unsqueeze(1)
            * velocity.new_tensor(float(self.config.slot_seconds))
            * steps
            / position_scale.view(1, 1, 1, 3)
        )
        node_mean[..., :3] = torch.where(
            valid[:, None, :, None], proposal, node_mean[..., :3]
        )
        output["node_state_mean"] = node_mean
        output["node_state"] = node_mean
        return output

    def _decode(
        self,
        base_output: Mapping[str, torch.Tensor],
        path: Mapping[str, Mapping[str, torch.Tensor]],
        masks: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
        *,
        teacher: bool,
    ) -> dict[str, torch.Tensor]:
        output = dict(base_output)
        state_heads = self.teacher_state_heads if teacher else self.prior_state_heads
        presence_heads = self.teacher_presence_heads if teacher else self.prior_presence_heads
        for name in COMPONENT_FEATURES:
            decoder = path[name]["decoder"]
            entity_mask = masks[name][:, None, :, None]
            correction = state_heads[name](decoder) * entity_mask.to(decoder.dtype)
            output[f"{name}_state_mean"] = output[f"{name}_state_mean"] + correction
            output[f"{name}_state"] = output[f"{name}_state_mean"]
            output[f"{name}_presence_logits"] = output[f"{name}_presence_logits"] + (
                presence_heads[name](decoder).squeeze(-1)
                * entity_mask.squeeze(-1).to(decoder.dtype)
            )
            output[f"rssm_{name}_state_correction"] = correction
        task_decoder = path["task"]["decoder"]
        edge_decoder = path["physical_edge"]["decoder"]
        dag_state_head = self.teacher_dag_state_head if teacher else self.prior_dag_state_head
        link_head = self.teacher_link_head if teacher else self.prior_link_head
        lifecycle_head = self.teacher_lifecycle_head if teacher else self.prior_lifecycle_head
        release_head = self.teacher_dag_release_head if teacher else self.prior_dag_release_head
        dag_edge_head = self.teacher_dag_edge_head if teacher else self.prior_dag_edge_head
        energy_head = self.teacher_energy_head if teacher else self.prior_energy_head
        task_mask = masks["task"][:, None, :]
        edge_mask = masks["physical_edge"][:, None, :]
        dag_state_correction = dag_state_head(task_decoder) * task_mask.unsqueeze(-1).to(
            task_decoder.dtype
        )
        output["task_dag_state_mean"] = output["task_dag_state_mean"] + dag_state_correction
        link_correction = link_head(edge_decoder).squeeze(-1) * edge_mask.to(
            edge_decoder.dtype
        )
        output["link_activity_logits"] = output["link_activity_logits"] + link_correction
        output["rssm_link_activity_correction"] = link_correction
        output["task_lifecycle_logits"] = output["task_lifecycle_logits"] + (
            lifecycle_head(task_decoder) * task_mask.unsqueeze(-1).to(task_decoder.dtype)
        )
        output["dag_release_logits"] = output["dag_release_logits"] + (
            release_head(task_decoder).squeeze(-1) * task_mask.to(task_decoder.dtype)
        )
        dag_pair = _gather_task_pair(task_decoder, static["dag_edge_index"])
        dag_edge_mask = static["dag_edge_valid"].bool()[:, None, :]
        output["dag_edge_presence_logits"] = output["dag_edge_presence_logits"] + (
            dag_edge_head(dag_pair).squeeze(-1) * dag_edge_mask.to(dag_pair.dtype)
        )
        if "uav_energy_delta_mean" in output:
            output["uav_energy_delta_mean"] = output["uav_energy_delta_mean"] + energy_head(
                path["node"]["decoder"]
            ).squeeze(-1) * masks["node"][:, None, :].to(task_decoder.dtype)
        return output

    def _pack_latents(
        self,
        path: Mapping[str, Mapping[str, torch.Tensor]],
        masks: Mapping[str, torch.Tensor],
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([path[name][key] for name in COMPONENT_FEATURES], dim=2)
        mask = torch.cat(
            [
                masks[name][:, None, :].expand(
                    -1, self.config.horizon_steps, -1
                )
                for name in COMPONENT_FEATURES
            ],
            dim=2,
        )
        return values, mask

    def forward(
        self, batch: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | bool]:
        history = batch["history"]
        base_batch = {
            "history": history,
            "future_action": batch["future_action"],
            "static": batch["static"],
        }
        base_output, trace = self.base.forward_with_latent_trace(base_batch)
        motion_context, raw_motion, motion_mask = self._motion_context(history)
        contexts = {
            name: torch.zeros_like(trace[name]) for name in COMPONENT_FEATURES
        }
        contexts["node"] = motion_context.unsqueeze(1).expand_as(trace["node"])
        masks = self._entity_masks(batch)
        proposal = self._kinematic_proposal(
            base_output, history, raw_motion, motion_mask
        )
        prior_path = self._rollout_path(trace, contexts, target=None)
        output: dict[str, torch.Tensor | bool] = self._decode(
            proposal, prior_path, masks, batch["static"], teacher=False
        )
        rollout_mean, latent_mask = self._pack_latents(prior_path, masks, "mean")
        rollout_log_std, _ = self._pack_latents(prior_path, masks, "log_std")
        output.update(
            rssm_rollout_prior_mean=rollout_mean,
            rssm_rollout_prior_log_std=rollout_log_std,
            rssm_latent_mask=latent_mask,
            rssm_deployment_prior_only=True,
        )
        for name in COMPONENT_FEATURES:
            output[f"rssm_{name}_rollout_prior_mean"] = prior_path[name]["mean"]
            output[f"rssm_{name}_rollout_prior_log_std"] = prior_path[name]["log_std"]
        if not self.training:
            return output
        if "target" not in batch:
            raise ValueError("training entity-aligned RSSM requires target")
        posterior_path = self._rollout_path(trace, contexts, target=batch["target"])
        teacher_output = self._decode(
            proposal, posterior_path, masks, batch["static"], teacher=True
        )
        for key, value in teacher_output.items():
            output[f"training_{key}"] = value
        posterior_mean, posterior_mask = self._pack_latents(
            posterior_path, masks, "mean"
        )
        posterior_log_std, _ = self._pack_latents(
            posterior_path, masks, "log_std"
        )
        path_prior_mean, _ = self._pack_latents(
            posterior_path, masks, "prior_mean"
        )
        path_prior_log_std, _ = self._pack_latents(
            posterior_path, masks, "prior_log_std"
        )
        output.update(
            rssm_posterior_path_prior_mean=path_prior_mean,
            rssm_posterior_path_prior_log_std=path_prior_log_std,
            rssm_posterior_mean=posterior_mean,
            rssm_posterior_log_std=posterior_log_std,
            rssm_latent_mask=posterior_mask,
            rssm_training_posterior_teacher=True,
        )
        for name in COMPONENT_FEATURES:
            output[f"rssm_{name}_posterior_mean"] = posterior_path[name]["mean"]
            output[f"rssm_{name}_posterior_log_std"] = posterior_path[name]["log_std"]
        distance = min(self.config.overshooting_distance, self.config.horizon_steps)
        start = distance - 1 if distance > 1 else 0
        output["rssm_overshooting_prior_mean"] = rollout_mean[:, start:]
        output["rssm_overshooting_prior_log_std"] = rollout_log_std[:, start:]
        output["rssm_overshooting_posterior_mean"] = posterior_mean[:, start:]
        output["rssm_overshooting_posterior_log_std"] = posterior_log_std[:, start:]
        output["rssm_overshooting_latent_mask"] = posterior_mask[:, start:]
        return output


__all__ = [
    "FormalEntityAlignedRSSMConfig",
    "FormalEntityAlignedRSSMWorldModel",
    "SCHEMA_VERSION",
]
