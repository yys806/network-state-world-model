"""Complete action-conditioned dual-graph RSSM candidate for PI-JWM.

This module keeps the frozen R3 graph/state heads as the explicit decoder while
providing a separate posterior path for training and a prior-only path for
deployment.  Targets are read only by the posterior teacher path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn

from .r3_preflight_data import CONTINUOUS_STATE_KEYS, ExplicitStateBatch
from .r3_world_model import BeliefSequence, R3ReferenceConfig, R3ReferenceWorldModel


COMPLETE_RSSM_SCHEMA = "PIJWM-Complete-Graph-RSSM-v1"


@dataclass
class CompleteRSSMRolloutOutput:
    predicted_explicit: dict[str, torch.Tensor]
    predicted_logits: dict[str, torch.Tensor]
    predicted_belief: BeliefSequence
    # Training-only teacher outputs.  Formal evaluation uses the prior outputs
    # above so that metrics match deployment semantics.
    training_predicted_explicit: dict[str, torch.Tensor] | None = None
    training_predicted_logits: dict[str, torch.Tensor] | None = None
    probabilistic_parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    execution_metadata: dict[str, bool | str] = field(default_factory=dict)

_FIELD_DIMS = {
    "physical_node": 9,
    "physical_edge": 7,
    "information_node": 7,
    "information_edge": 18,
    "data_flow": 5,
    "task": 8,
    "task_dag": 3,
}

_PRESENCE_KEYS = {
    "physical_node": "physical_node_present",
    "physical_edge": "physical_edge_present",
    "information_node": "information_node_present",
    "information_edge": "information_edge_present",
    "data_flow": "data_flow_present",
    "task": "task_present",
    "task_dag": "task_dag_state_present",
}

_STATE_KEYS = {
    "task_dag": "task_dag_state",
}


@dataclass
class CompleteRSSMBelief:
    base_belief: object
    context_hidden: torch.Tensor
    context_prior_mean: torch.Tensor
    context_prior_log_std: torch.Tensor
    context_posterior_mean: torch.Tensor
    context_posterior_log_std: torch.Tensor


def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = parameters.chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)


def _sample(mean: torch.Tensor, log_std: torch.Tensor, training: bool) -> torch.Tensor:
    if training:
        return mean + torch.randn_like(mean) * torch.exp(log_std)
    return mean


def _masked_pool(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(value.dtype)
    return (value * numeric.unsqueeze(-1)).sum(dim=-2) / numeric.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


class CompleteGraphRSSMBackend(nn.Module):
    """Dual-graph decoder plus a full prior/posterior RSSM state path."""

    schema_version = COMPLETE_RSSM_SCHEMA
    latent_dynamics = "complete_graph_rssm_prior_posterior_v1"

    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.base = R3ReferenceWorldModel(config)
        hidden = config.hidden_dim

        self.context_prior = nn.Linear(hidden, hidden * 2)
        self.context_observation = nn.ModuleDict(
            {name: nn.Linear(dim, hidden) for name, dim in _FIELD_DIMS.items()}
        )
        self.observation_fusion = nn.Linear(hidden * len(_FIELD_DIMS), hidden)
        self.context_posterior = nn.Linear(hidden * 2, hidden * 2)
        self.action_encoder = nn.Linear(8, hidden)
        self.transition = nn.GRUCell(hidden * 2, hidden)
        self.prior = nn.Linear(hidden, hidden * 2)
        self.posterior = nn.Linear(hidden * 2, hidden * 2)
        self.joint_fusion = nn.Linear(hidden * 3, hidden)

        self.continuous_correction = nn.ModuleDict(
            {
                "physical_node_state": nn.Linear(hidden, 9),
                "physical_edge_state": nn.Linear(hidden, 7),
                "information_node_state": nn.Linear(hidden, 7),
                "information_edge_state": nn.Linear(hidden, 18),
                "data_flow_state": nn.Linear(hidden, 5),
                "task_state": nn.Linear(hidden, 8),
                "task_dag_state": nn.Linear(hidden, 3),
            }
        )
        self.logit_correction = nn.ModuleDict(
            {
                "physical_node_present": nn.Linear(hidden, 1),
                "physical_edge_present": nn.Linear(hidden, 1),
                "information_node_present": nn.Linear(hidden, 1),
                "information_edge_present": nn.Linear(hidden, 1),
                "data_flow_present": nn.Linear(hidden, 1),
                "task_present": nn.Linear(hidden, 1),
                "task_dag_state_present": nn.Linear(hidden, 1),
                "information_link_activity": nn.Linear(hidden, 1),
                "task_lifecycle": nn.Linear(hidden, 5),
            }
        )

    def _observation(self, namespace: Mapping[str, torch.Tensor], index: int) -> torch.Tensor:
        encoded = []
        for name, feature_dim in _FIELD_DIMS.items():
            state_key = _STATE_KEYS.get(name, f"{name}_state")
            value = namespace[state_key][:, index]
            present = namespace[_PRESENCE_KEYS[name]][:, index].bool()
            mask_key = f"{name}_feature_mask"
            if mask_key in namespace:
                feature_mask = namespace[mask_key][:, index].bool()
            else:
                feature_mask = present.unsqueeze(-1).expand_as(value)
            observed = value * feature_mask.to(value.dtype)
            pooled = _masked_pool(observed, present)
            encoded.append(torch.tanh(self.context_observation[name](pooled)))
        return torch.tanh(self.observation_fusion(torch.cat(encoded, dim=-1)))

    def _action(self, action: Mapping[str, torch.Tensor], step: int) -> torch.Tensor:
        values = action["task_action"][:, step]
        present = action["task_action_present"][:, step].to(values.dtype)
        encoded = self.action_encoder(values) * present.unsqueeze(-1)
        return _masked_pool(encoded, present.bool())

    def infer_belief(self, batch: ExplicitStateBatch) -> CompleteRSSMBelief:
        base_belief = self.base.infer_belief(batch)
        context_hidden = base_belief.joint
        prior_mean, prior_log_std = _distribution(self.context_prior(context_hidden))
        context_observation = self._observation(batch.history, batch.history["physical_node_state"].shape[1] - 1)
        posterior_input = torch.cat((context_hidden, context_observation), dim=-1)
        posterior_mean, posterior_log_std = _distribution(
            self.context_posterior(posterior_input)
        )
        return CompleteRSSMBelief(
            base_belief=base_belief,
            context_hidden=context_hidden,
            context_prior_mean=prior_mean,
            context_prior_log_std=prior_log_std,
            context_posterior_mean=posterior_mean,
            context_posterior_log_std=posterior_log_std,
        )

    def _apply_decoder(
        self,
        base_output: object,
        joint_sequence: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        explicit = dict(base_output.predicted_explicit)
        for name, prediction in explicit.items():
            if name in self.continuous_correction:
                explicit[name] = prediction + self.continuous_correction[name](joint_sequence).unsqueeze(2)
        logits = dict(base_output.predicted_logits)
        for name, prediction in logits.items():
            if name not in self.logit_correction:
                continue
            correction = self.logit_correction[name](joint_sequence)
            if prediction.ndim == 3:
                correction = correction.squeeze(-1).unsqueeze(2)
            else:
                correction = correction.unsqueeze(2)
            logits[name] = prediction + correction
        return explicit, logits

    def rollout(
        self,
        belief: CompleteRSSMBelief,
        future_action: Mapping[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> CompleteRSSMRolloutOutput:
        required = ("task_action", "task_action_present", "task_action_information_node_index")
        if any(name not in future_action for name in required):
            raise ValueError("future action tensors are incomplete")
        if any(future_action[name].shape[1] < rollout_steps for name in required):
            raise ValueError("future action horizon is shorter than rollout_steps")

        base_output = self.base.rollout(
            belief.base_belief, future_action, batch, rollout_steps=rollout_steps
        )

        # Deployment path: after context inference, only actions and prior are used.
        prior_hidden = belief.context_hidden
        prior_z = _sample(
            belief.context_posterior_mean,
            belief.context_posterior_log_std,
            self.training,
        )
        rollout_prior_mean = []
        rollout_prior_log_std = []
        deployment_joint = []
        for step in range(rollout_steps):
            action = self._action(future_action, step)
            prior_hidden = self.transition(torch.cat((prior_z, action), dim=-1), prior_hidden)
            prior_mean, prior_log_std = _distribution(self.prior(prior_hidden))
            prior_z = _sample(prior_mean, prior_log_std, self.training)
            joint = torch.tanh(
                self.joint_fusion(
                    torch.cat((base_output.predicted_belief.joint_latent[:, step], prior_hidden, prior_z), dim=-1)
                )
            )
            rollout_prior_mean.append(prior_mean)
            rollout_prior_log_std.append(prior_log_std)
            deployment_joint.append(joint)
        joint_sequence = torch.stack(deployment_joint, dim=1)
        explicit, logits = self._apply_decoder(base_output, joint_sequence)

        # Training teacher path: every posterior is conditioned on the matching target observation.
        posterior_hidden = belief.context_hidden
        posterior_z = _sample(
            belief.context_posterior_mean,
            belief.context_posterior_log_std,
            self.training,
        )
        step_prior_mean = []
        step_prior_log_std = []
        step_posterior_mean = []
        step_posterior_log_std = []
        posterior_joint = []
        for step in range(rollout_steps):
            action = self._action(future_action, step)
            posterior_hidden = self.transition(
                torch.cat((posterior_z, action), dim=-1), posterior_hidden
            )
            p_mean, p_log_std = _distribution(self.prior(posterior_hidden))
            observation = self._observation(batch.target, step)
            q_mean, q_log_std = _distribution(
                self.posterior(torch.cat((posterior_hidden, observation), dim=-1))
            )
            posterior_z = _sample(q_mean, q_log_std, self.training)
            posterior_joint.append(
                torch.tanh(
                    self.joint_fusion(
                        torch.cat(
                            (
                                base_output.predicted_belief.joint_latent[:, step],
                                posterior_hidden,
                                posterior_z,
                            ),
                            dim=-1,
                        )
                    )
                )
            )
            step_prior_mean.append(p_mean)
            step_prior_log_std.append(p_log_std)
            step_posterior_mean.append(q_mean)
            step_posterior_log_std.append(q_log_std)

        training_explicit = None
        training_logits = None
        if self.training:
            # The variational teacher grounds z_t with the observed target.  It
            # is used only by the training objective; future deployment never
            # reads these tensors.
            teacher_joint = torch.stack(posterior_joint, dim=1)
            training_explicit, training_logits = self._apply_decoder(
                base_output, teacher_joint
            )

        return CompleteRSSMRolloutOutput(
            predicted_explicit=explicit,
            predicted_logits=logits,
            predicted_belief=BeliefSequence(
                physical_latent=base_output.predicted_belief.physical_latent,
                information_latent=base_output.predicted_belief.information_latent,
                business_latent=base_output.predicted_belief.business_latent,
                joint_latent=joint_sequence,
            ),
            probabilistic_parameters={
                "context_prior_mean": belief.context_prior_mean,
                "context_prior_log_std": belief.context_prior_log_std,
                "context_posterior_mean": belief.context_posterior_mean,
                "context_posterior_log_std": belief.context_posterior_log_std,
                "rollout_prior_mean": torch.stack(rollout_prior_mean, dim=1),
                "rollout_prior_log_std": torch.stack(rollout_prior_log_std, dim=1),
                "posterior_path_prior_mean": torch.stack(step_prior_mean, dim=1),
                "posterior_path_prior_log_std": torch.stack(step_prior_log_std, dim=1),
                "posterior_path_posterior_mean": torch.stack(step_posterior_mean, dim=1),
                "posterior_path_posterior_log_std": torch.stack(step_posterior_log_std, dim=1),
            },
            execution_metadata={
                "deployment_prior_only": True,
                "training_posterior_teacher": True,
                "target_in_prediction_path": False,
                "rssm_kl_balance_alpha": 0.8,
                "training_teacher_reconstruction": bool(self.training),
            },
            training_predicted_explicit=training_explicit,
            training_predicted_logits=training_logits,
        )

    def forward(self, batch: ExplicitStateBatch, *, rollout_steps: int) -> R4RolloutOutput:
        return self.rollout(
            self.infer_belief(batch), batch.future_action, batch, rollout_steps=rollout_steps
        )


__all__ = [
    "COMPLETE_RSSM_SCHEMA",
    "CompleteGraphRSSMBackend",
    "CompleteRSSMBelief",
    "CompleteRSSMRolloutOutput",
]
