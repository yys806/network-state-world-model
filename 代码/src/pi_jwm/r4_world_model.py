"""Strict R4 model factory preserving the frozen R3 public rollout contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from .formal_directed_graph_ops_v2 import directed_relation_messages, weighted_index_mean
from .r3_preflight_data import ExplicitStateBatch
from .r3_world_model import (
    BeliefSequence,
    R3ReferenceConfig,
    R3ReferenceWorldModel,
    R3RolloutOutput,
)
from .r4_module_registry import (
    KNOWN_STATUSES,
    R4ModuleConfig,
    assert_executable_config,
    validate_controlled_config,
)


R4_MODEL_SCHEMA = "PIJWM-R4-Controlled-World-Model-v1"


@dataclass
class R4RolloutOutput:
    predicted_explicit: dict[str, torch.Tensor]
    predicted_logits: dict[str, torch.Tensor]
    predicted_belief: BeliefSequence
    probabilistic_parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    execution_metadata: dict[str, bool | str] = field(default_factory=dict)


def symlog(value: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm used by the frozen R4 field candidate."""

    return torch.sign(value) * torch.log1p(torch.abs(value))


def simnorm(value: torch.Tensor, *, group_size: int = 4) -> torch.Tensor:
    """Normalize contiguous representation groups with a softmax."""

    if group_size <= 0 or value.shape[-1] % group_size != 0:
        raise ValueError("SimNorm group_size must divide the final dimension")
    grouped = value.reshape(*value.shape[:-1], value.shape[-1] // group_size, group_size)
    return F.softmax(grouped, dim=-1).reshape_as(value)


class _CandidateTemporalEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"symlog", "simnorm"}:
            raise ValueError(f"unknown R4 field-encoder mode: {mode}")
        if mode == "simnorm" and hidden_dim % 4 != 0:
            raise ValueError("SimNorm field encoder requires hidden_dim divisible by 4")
        self.mode = mode
        self.input = nn.Linear(feature_dim * 2, hidden_dim)
        self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        value: torch.Tensor,
        feature_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 4 or feature_mask.shape != value.shape:
            raise ValueError("temporal value and feature mask must be [batch,time,entity,feature]")
        if present.shape != value.shape[:-1]:
            raise ValueError("temporal presence does not match values")
        observed = torch.where(feature_mask.bool(), value, torch.zeros_like(value))
        if self.mode == "symlog":
            observed = symlog(observed)
        encoded = self.input(
            torch.cat((observed, feature_mask.to(value.dtype)), dim=-1)
        )
        if self.mode == "simnorm":
            encoded = simnorm(encoded, group_size=4)
        encoded = encoded * present.unsqueeze(-1).to(value.dtype)
        batch, steps, entities, _ = encoded.shape
        sequence = encoded.permute(0, 2, 1, 3).reshape(
            batch * entities, steps, encoded.shape[-1]
        )
        output, _ = self.history(sequence)
        return output[:, -1].reshape(batch, entities, encoded.shape[-1])


def _replace_field_encoders(
    backend: R3ReferenceWorldModel,
    *,
    mode: str,
    hidden_dim: int,
) -> None:
    dimensions = {
        "physical_node_encoder": 9,
        "physical_edge_encoder": 7,
        "information_node_encoder": 7,
        "information_edge_encoder": 18,
        "flow_encoder": 5,
        "task_encoder": 8,
        "dag_encoder": 3,
        "history_action_encoder": 8,
    }
    for name, feature_dim in dimensions.items():
        setattr(
            backend,
            name,
            _CandidateTemporalEncoder(feature_dim, hidden_dim, mode=mode),
        )


def _apply_gru(
    cell: nn.GRUCell,
    update: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    shape = current.shape
    return cell(
        update.reshape(-1, shape[-1]),
        current.reshape(-1, shape[-1]),
    ).reshape(shape)


def _batched_endpoints(
    endpoints: torch.Tensor,
    *,
    batch_size: int,
    relation_count: int,
    entity_count: int,
    active: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if endpoints.ndim == 2 and endpoints.shape == (relation_count, 2):
        endpoints = endpoints.unsqueeze(0).expand(batch_size, -1, -1)
    if endpoints.shape != (batch_size, relation_count, 2):
        raise ValueError(f"{name} shape does not match graph relations")
    if active.shape != (batch_size, relation_count):
        raise ValueError(f"{name} activity shape does not match graph relations")
    valid = (
        (endpoints[..., 0] >= 0)
        & (endpoints[..., 0] < entity_count)
        & (endpoints[..., 1] >= 0)
        & (endpoints[..., 1] < entity_count)
    )
    if ((active > 0) & ~valid).any():
        raise ValueError(f"{name} contains an invalid endpoint")
    return endpoints.clamp(0, max(entity_count - 1, 0)).long()


def _batched_relation_kind(
    kind: torch.Tensor,
    *,
    batch_size: int,
    relation_count: int,
    relation_types: int,
    active: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if kind.ndim == 1 and kind.shape[0] == relation_count:
        kind = kind.unsqueeze(0).expand(batch_size, -1)
    if kind.shape != (batch_size, relation_count):
        raise ValueError(f"{name} shape does not match graph relations")
    if active.shape != (batch_size, relation_count):
        raise ValueError(f"{name} activity shape does not match graph relations")
    valid = (kind >= 0) & (kind < relation_types)
    if ((active > 0) & ~valid).any():
        raise ValueError(f"{name} contains an unknown relation type")
    return kind.clamp(0, relation_types - 1).long()


def _gather_by_index(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        value,
        1,
        index.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
    )


def _relation_weight(
    node_weight: torch.Tensor,
    edge_weight: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    src_weight = torch.gather(node_weight, 1, src)
    dst_weight = torch.gather(node_weight, 1, dst)
    return (
        edge_weight.clamp_min(0.0)
        * src_weight.clamp_min(0.0)
        * dst_weight.clamp_min(0.0)
    )


class _CandidateGraphBackend(R3ReferenceWorldModel):
    def __init__(self, config: R3ReferenceConfig, *, mode: str) -> None:
        super().__init__(config)
        if mode not in {"rgcn", "ecc"}:
            raise ValueError(f"unknown R4 graph mode: {mode}")
        self.r4_graph_mode = mode
        hidden = config.hidden_dim
        if mode == "rgcn":
            self.physical_relation_in = nn.Parameter(torch.empty(1, hidden, hidden))
            self.physical_relation_out = nn.Parameter(torch.empty(1, hidden, hidden))
            self.information_relation_in = nn.Parameter(torch.empty(10, hidden, hidden))
            self.information_relation_out = nn.Parameter(torch.empty(10, hidden, hidden))
            self.physical_relation_edge = nn.Linear(hidden, hidden)
            self.information_relation_edge = nn.Linear(hidden, hidden)
            for parameter in (
                self.physical_relation_in,
                self.physical_relation_out,
                self.information_relation_in,
                self.information_relation_out,
            ):
                nn.init.xavier_uniform_(parameter)
        else:
            self.physical_kernel_in = nn.Linear(hidden, hidden * hidden)
            self.physical_kernel_out = nn.Linear(hidden, hidden * hidden)
            self.information_kernel_in = nn.Linear(hidden, hidden * hidden)
            self.information_kernel_out = nn.Linear(hidden, hidden * hidden)

    def _candidate_messages(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        endpoints: torch.Tensor,
        node_weight: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        graph_name: str,
        relation_kind: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, entity_count, hidden = node_latent.shape
        relation_count = edge_latent.shape[1]
        numeric_edge_weight = edge_weight.to(node_latent.dtype)
        endpoints = _batched_endpoints(
            endpoints,
            batch_size=batch_size,
            relation_count=relation_count,
            entity_count=entity_count,
            active=numeric_edge_weight,
            name=f"{graph_name}_edge_endpoint_index",
        )
        src, dst = endpoints[..., 0], endpoints[..., 1]
        src_latent = _gather_by_index(node_latent, src)
        dst_latent = _gather_by_index(node_latent, dst)
        weight = _relation_weight(
            node_weight.to(node_latent.dtype),
            numeric_edge_weight,
            src,
            dst,
        )

        if self.r4_graph_mode == "rgcn":
            relation_types = 1 if graph_name == "physical" else 10
            kind_name = f"{graph_name}_edge_kind_index"
            if relation_kind is None:
                if graph_name != "physical":
                    raise ValueError("information_edge_kind_index is required by R-GCN")
                relation_kind = torch.zeros(
                    (batch_size, relation_count),
                    dtype=torch.long,
                    device=node_latent.device,
                )
            kinds = _batched_relation_kind(
                relation_kind.to(node_latent.device),
                batch_size=batch_size,
                relation_count=relation_count,
                relation_types=relation_types,
                active=numeric_edge_weight,
                name=kind_name,
            )
            matrices_in = getattr(self, f"{graph_name}_relation_in")[kinds]
            matrices_out = getattr(self, f"{graph_name}_relation_out")[kinds]
            edge_message = getattr(self, f"{graph_name}_relation_edge")(edge_latent)
            incoming_message = torch.einsum(
                "beh,behd->bed", src_latent, matrices_in
            ) + edge_message
            outgoing_message = torch.einsum(
                "beh,behd->bed", dst_latent, matrices_out
            ) + edge_message
        else:
            kernel_in = getattr(self, f"{graph_name}_kernel_in")(edge_latent).reshape(
                batch_size, relation_count, hidden, hidden
            )
            kernel_out = getattr(self, f"{graph_name}_kernel_out")(edge_latent).reshape(
                batch_size, relation_count, hidden, hidden
            )
            incoming_message = torch.einsum(
                "beh,behd->bed", src_latent, kernel_in
            )
            outgoing_message = torch.einsum(
                "beh,behd->bed", dst_latent, kernel_out
            )

        incoming = weighted_index_mean(
            incoming_message,
            dst,
            entity_count,
            weight,
        )
        outgoing = weighted_index_mean(
            outgoing_message,
            src,
            entity_count,
            weight,
        )
        edge_context = torch.cat((src_latent, dst_latent), dim=-1)
        edge_context = edge_context * (weight > 0).unsqueeze(-1).to(node_latent.dtype)
        return incoming, outgoing, edge_context

    def _graph_update(self, state, batch: ExplicitStateBatch):
        history, static = batch.history, batch.static
        physical_messages = self._candidate_messages(
            state.physical_node,
            state.physical_edge,
            static["physical_edge_endpoint_index"],
            history["physical_node_present"][:, -1],
            history["physical_edge_present"][:, -1],
            graph_name="physical",
            relation_kind=None,
        )
        p_in, p_out, p_context = physical_messages
        state.physical_node = _apply_gru(
            self.physical_node_transition,
            self.physical_node_graph(torch.cat((p_in, p_out), dim=-1)),
            state.physical_node,
        )
        state.physical_edge = _apply_gru(
            self.physical_edge_transition,
            self.physical_edge_graph(p_context),
            state.physical_edge,
        )

        information_kind = static.get("information_edge_kind_index")
        i_in, i_out, i_context = self._candidate_messages(
            state.information_node,
            state.information_edge,
            static["information_edge_endpoint_index"],
            history["information_node_present"][:, -1],
            history["information_edge_present"][:, -1],
            graph_name="information",
            relation_kind=information_kind,
        )
        state.information_node = _apply_gru(
            self.information_node_transition,
            self.information_node_graph(torch.cat((i_in, i_out), dim=-1)),
            state.information_node,
        )
        state.information_edge = _apply_gru(
            self.information_edge_transition,
            self.information_edge_graph(i_context),
            state.information_edge,
        )
        return state


def relation_constrained_attention(
    query: torch.Tensor,
    source: torch.Tensor,
    mapping: torch.Tensor,
    query_projection: nn.Module,
    key_projection: nn.Module,
    value_projection: nn.Module,
    *,
    query_weight: torch.Tensor | None = None,
    source_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend only to the one source explicitly named by each PI relation."""

    if query.ndim != 3 or source.ndim != 3 or query.shape[0] != source.shape[0]:
        raise ValueError("relation-constrained attention expects batched rank-3 tensors")
    batch_size, query_count, hidden = query.shape
    if source.shape[-1] != hidden:
        raise ValueError("query and source dimensions do not match")
    if mapping.ndim == 1:
        mapping = mapping.unsqueeze(0).expand(batch_size, -1)
    if mapping.shape != (batch_size, query_count):
        raise ValueError("relation mapping shape does not match query entities")
    if query_weight is None:
        query_weight = query.new_ones((batch_size, query_count))
    if source_weight is None:
        source_weight = source.new_ones((batch_size, source.shape[1]))
    if query_weight.shape != (batch_size, query_count):
        raise ValueError("query_weight shape does not match query entities")
    if source_weight.shape != (batch_size, source.shape[1]):
        raise ValueError("source_weight shape does not match source entities")

    active_query = query_weight > 0
    valid_index = (mapping >= 0) & (mapping < source.shape[1])
    if (active_query & ~valid_index).any():
        raise ValueError("active relation mapping contains an invalid source index")
    safe_mapping = mapping.clamp(0, max(source.shape[1] - 1, 0)).long()
    selected_source = _gather_by_index(source, safe_mapping)
    selected_source_weight = torch.gather(source_weight, 1, safe_mapping)
    valid = active_query & valid_index & (selected_source_weight > 0)
    score = (
        query_projection(query) * key_projection(selected_source)
    ).sum(dim=-1) / float(hidden) ** 0.5
    gate = torch.sigmoid(score) * valid.to(query.dtype)
    message = value_projection(selected_source) * gate.unsqueeze(-1)
    return message, gate


class _CrossAttentionBackend(R3ReferenceWorldModel):
    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim
        self.relation_attention = nn.ModuleDict()
        self.relation_reverse_value = nn.ModuleDict()
        for relation in ("cip", "cep", "cfl"):
            self.relation_attention[f"{relation}_query"] = nn.Linear(hidden, hidden)
            self.relation_attention[f"{relation}_key"] = nn.Linear(hidden, hidden)
            self.relation_attention[f"{relation}_value"] = nn.Linear(hidden, hidden)
            self.relation_reverse_value[relation] = nn.Linear(hidden, hidden)

    def _pair(
        self,
        relation: str,
        target: torch.Tensor,
        source: torch.Tensor,
        mapping: torch.Tensor,
        target_weight: torch.Tensor,
        source_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return relation_constrained_attention(
            target,
            source,
            mapping,
            self.relation_attention[f"{relation}_query"],
            self.relation_attention[f"{relation}_key"],
            self.relation_attention[f"{relation}_value"],
            query_weight=target_weight,
            source_weight=source_weight,
        )

    def _cross_couple(self, state, batch: ExplicitStateBatch):
        history, static = batch.history, batch.static
        physical_node_weight = history["physical_node_present"][:, -1].to(
            state.physical_node.dtype
        )
        information_node_weight = history["information_node_present"][:, -1].to(
            state.information_node.dtype
        )
        cip = static["cip_agent_node_index"].long()
        information_update, cip_gate = self._pair(
            "cip",
            state.information_node,
            state.physical_node,
            cip,
            information_node_weight,
            physical_node_weight,
        )
        physical_update = weighted_index_mean(
            self.relation_reverse_value["cip"](state.information_node),
            cip,
            state.physical_node.shape[1],
            cip_gate,
        )
        state.information_node = _apply_gru(
            self.coupler["information_node"],
            information_update,
            state.information_node,
        )
        state.physical_node = _apply_gru(
            self.coupler["physical_node"],
            physical_update,
            state.physical_node,
        )

        physical_edge_weight = history["physical_edge_present"][:, -1].to(
            state.physical_edge.dtype
        )
        information_edge_weight = history["information_edge_present"][:, -1].to(
            state.information_edge.dtype
        )
        cep = static["cep_information_to_physical_edge_index"].long()
        information_edge_update, cep_gate = self._pair(
            "cep",
            state.information_edge,
            state.physical_edge,
            cep,
            information_edge_weight,
            physical_edge_weight,
        )
        physical_edge_update = weighted_index_mean(
            self.relation_reverse_value["cep"](state.information_edge),
            cep,
            state.physical_edge.shape[1],
            cep_gate,
        )
        state.information_edge = _apply_gru(
            self.coupler["information_edge"],
            information_edge_update,
            state.information_edge,
        )
        state.physical_edge = _apply_gru(
            self.coupler["physical_edge"],
            physical_edge_update,
            state.physical_edge,
        )

        flow_weight = history["data_flow_present"][:, -1].to(state.flow.dtype)
        cfl = static["cfl_information_edge_index"].long()
        flow_update, cfl_gate = self._pair(
            "cfl",
            state.flow,
            state.information_edge,
            cfl,
            flow_weight,
            information_edge_weight,
        )
        information_from_flow = weighted_index_mean(
            self.relation_reverse_value["cfl"](state.flow),
            cfl,
            state.information_edge.shape[1],
            cfl_gate,
        )
        state.flow = _apply_gru(
            self.coupler["flow"],
            flow_update,
            state.flow,
        )
        state.information_edge = _apply_gru(
            self.coupler["information_edge"],
            information_from_flow,
            state.information_edge,
        )
        return state


@dataclass
class _RSSMBelief:
    base_belief: object
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    context_prior_mean: torch.Tensor
    context_prior_log_std: torch.Tensor
    context_posterior_mean: torch.Tensor
    context_posterior_log_std: torch.Tensor


class _GraphRSSMBackend(nn.Module):
    """Graph-context RSSM whose deployment rollout uses its prior only."""

    def __init__(
        self,
        config: R3ReferenceConfig,
        *,
        base: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.base = R3ReferenceWorldModel(config) if base is None else base
        hidden = config.hidden_dim
        self.rssm_context_prior = nn.Linear(hidden, hidden * 2)
        self.rssm_context_posterior = nn.Linear(hidden, hidden * 2)
        self.rssm_action_encoder = nn.Linear(8, hidden)
        self.rssm_deterministic_transition = nn.GRUCell(hidden * 2, hidden)
        self.rssm_prior_head = nn.Linear(hidden, hidden * 2)
        self.rssm_joint_fusion = nn.Linear(hidden * 3, hidden)
        self.rssm_continuous_correction = nn.ModuleDict(
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
        self.rssm_logit_correction = nn.ModuleDict(
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

    @staticmethod
    def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = parameters.chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def _state_value(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mean + torch.randn_like(mean) * torch.exp(log_std)
        return mean

    def infer_belief(self, batch: ExplicitStateBatch) -> _RSSMBelief:
        base_belief = self.base.infer_belief(batch)
        prior_mean, prior_log_std = self._distribution(
            self.rssm_context_prior(base_belief.joint)
        )
        posterior_mean, posterior_log_std = self._distribution(
            self.rssm_context_posterior(base_belief.joint)
        )
        stochastic = self._state_value(posterior_mean, posterior_log_std)
        return _RSSMBelief(
            base_belief=base_belief,
            deterministic=base_belief.joint,
            stochastic=stochastic,
            context_prior_mean=prior_mean,
            context_prior_log_std=prior_log_std,
            context_posterior_mean=posterior_mean,
            context_posterior_log_std=posterior_log_std,
        )

    def rollout(
        self,
        belief: _RSSMBelief,
        future_action: dict[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        base_output = self.base.rollout(
            belief.base_belief,
            future_action,
            batch,
            rollout_steps=rollout_steps,
        )
        deterministic = belief.deterministic
        stochastic = belief.stochastic
        prior_means: list[torch.Tensor] = []
        prior_log_stds: list[torch.Tensor] = []
        joint_steps: list[torch.Tensor] = []
        for step in range(rollout_steps):
            action = future_action["task_action"][:, step]
            present = future_action["task_action_present"][:, step].to(action.dtype)
            encoded_action = self.rssm_action_encoder(action)
            pooled_action = (
                (encoded_action * present.unsqueeze(-1)).sum(dim=1)
                / present.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            deterministic = self.rssm_deterministic_transition(
                torch.cat((stochastic, pooled_action), dim=-1),
                deterministic,
            )
            prior_mean, prior_log_std = self._distribution(
                self.rssm_prior_head(deterministic)
            )
            stochastic = self._state_value(prior_mean, prior_log_std)
            joint = torch.tanh(
                self.rssm_joint_fusion(
                    torch.cat(
                        (
                            base_output.predicted_belief.joint_latent[:, step],
                            deterministic,
                            stochastic,
                        ),
                        dim=-1,
                    )
                )
            )
            prior_means.append(prior_mean)
            prior_log_stds.append(prior_log_std)
            joint_steps.append(joint)
        joint_sequence = torch.stack(joint_steps, dim=1)

        explicit = dict(base_output.predicted_explicit)
        for name, prediction in explicit.items():
            correction = self.rssm_continuous_correction[name](joint_sequence)
            explicit[name] = prediction + correction.unsqueeze(2)
        logits = dict(base_output.predicted_logits)
        for name, prediction in logits.items():
            correction = self.rssm_logit_correction[name](joint_sequence)
            if prediction.ndim == 3:
                correction = correction.squeeze(-1).unsqueeze(2)
            else:
                correction = correction.unsqueeze(2)
            logits[name] = prediction + correction

        return R4RolloutOutput(
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
                "rollout_prior_mean": torch.stack(prior_means, dim=1),
                "rollout_prior_log_std": torch.stack(prior_log_stds, dim=1),
            },
            execution_metadata={"deployment_prior_only": True},
        )

    def forward(
        self,
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        return self.rollout(
            self.infer_belief(batch),
            batch.future_action,
            batch,
            rollout_steps=rollout_steps,
        )


class _HeteroscedasticBackend(nn.Module):
    def __init__(
        self,
        config: R3ReferenceConfig,
        *,
        base: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.base = R3ReferenceWorldModel(config) if base is None else base
        dimensions = {
            "physical_node_state": 9,
            "physical_edge_state": 7,
            "information_node_state": 7,
            "information_edge_state": 18,
            "data_flow_state": 5,
            "task_state": 8,
            "task_dag_state": 3,
        }
        self.heteroscedastic_log_variance = nn.ModuleDict(
            {
                name: nn.Linear(feature_dim, feature_dim)
                for name, feature_dim in dimensions.items()
            }
        )

    def infer_belief(self, batch: ExplicitStateBatch):
        return self.base.infer_belief(batch)

    def _decorate(self, output: R3RolloutOutput | R4RolloutOutput) -> R4RolloutOutput:
        parameters = dict(getattr(output, "probabilistic_parameters", {}))
        parameters.update({
            f"{name}_log_variance": self.heteroscedastic_log_variance[name](prediction).clamp(
                -8.0, 5.0
            )
            for name, prediction in output.predicted_explicit.items()
        })
        metadata = dict(getattr(output, "execution_metadata", {}))
        metadata["continuous_distribution"] = "diagonal_gaussian"
        return R4RolloutOutput(
            predicted_explicit=output.predicted_explicit,
            predicted_logits=output.predicted_logits,
            predicted_belief=output.predicted_belief,
            probabilistic_parameters=parameters,
            execution_metadata=metadata,
        )

    def rollout(
        self,
        belief,
        future_action: dict[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        return self._decorate(
            self.base.rollout(
                belief,
                future_action,
                batch,
                rollout_steps=rollout_steps,
            )
        )

    def forward(
        self,
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        return self._decorate(self.base(batch, rollout_steps=rollout_steps))


class _HurdleRateBackend(nn.Module):
    def __init__(
        self,
        config: R3ReferenceConfig,
        *,
        rate_mean: float,
        rate_scale: float,
    ) -> None:
        super().__init__()
        if rate_scale <= 0.0:
            raise ValueError("hurdle rate normalization scale must be positive")
        self.config = config
        self.base = R3ReferenceWorldModel(config)
        self.rate_mean = float(rate_mean)
        self.rate_scale = float(rate_scale)
        self.hurdle_log_location = nn.Linear(18, 1)
        self.hurdle_log_std = nn.Linear(18, 1)

    def infer_belief(self, batch: ExplicitStateBatch):
        return self.base.infer_belief(batch)

    def _decorate(self, output: R3RolloutOutput) -> R4RolloutOutput:
        information_state = output.predicted_explicit["information_edge_state"]
        log_location = self.hurdle_log_location(information_state).squeeze(-1).clamp(
            -10.0, 10.0
        )
        log_std = self.hurdle_log_std(information_state).squeeze(-1).clamp(-5.0, 2.0)
        raw_mean = torch.exp(log_location + 0.5 * torch.exp(2.0 * log_std))
        activity_probability = torch.sigmoid(
            output.predicted_logits["information_link_activity"]
        )
        expected_normalized_rate = (
            activity_probability * raw_mean - self.rate_mean
        ) / self.rate_scale
        updated_information_state = information_state.clone()
        updated_information_state[..., 12] = expected_normalized_rate
        explicit = dict(output.predicted_explicit)
        explicit["information_edge_state"] = updated_information_state
        return R4RolloutOutput(
            predicted_explicit=explicit,
            predicted_logits=output.predicted_logits,
            predicted_belief=output.predicted_belief,
            probabilistic_parameters={
                "active_rate_log_location": log_location,
                "active_rate_log_std": log_std,
                "active_rate_raw_mean": raw_mean,
                "active_rate_normalization_mean": information_state.new_tensor(
                    self.rate_mean
                ),
                "active_rate_normalization_scale": information_state.new_tensor(
                    self.rate_scale
                ),
            },
            execution_metadata={
                "active_rate_distribution": "log_normal",
                "active_rate_unit": "Mbps",
            },
        )

    def rollout(
        self,
        belief,
        future_action: dict[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        return self._decorate(
            self.base.rollout(
                belief,
                future_action,
                batch,
                rollout_steps=rollout_steps,
            )
        )

    def forward(
        self,
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R4RolloutOutput:
        return self._decorate(self.base(batch, rollout_steps=rollout_steps))


class _ExplicitDAGBackend(R3ReferenceWorldModel):
    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim
        self.dag_relation_embedding = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.dag_relation_embedding, std=hidden ** -0.5)
        self.dag_task_message = nn.Linear(hidden * 2, hidden)
        self.dag_task_transition = nn.GRUCell(hidden, hidden)

    @staticmethod
    def _dag_structure(
        batch: ExplicitStateBatch,
        *,
        task_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "dag_edge_index" not in batch.static:
            raise ValueError("dag_edge_index is required by explicit DAG message passing")
        if "dag_edge_present" not in batch.history:
            raise ValueError("dag_edge_present is required by explicit DAG message passing")
        endpoints = batch.static["dag_edge_index"]
        batch_size = batch.history["task_present"].shape[0]
        edge_count = batch.history["dag_edge_present"].shape[-1]
        if endpoints.ndim == 2 and endpoints.shape == (2, edge_count):
            endpoints = endpoints.transpose(0, 1)
        elif endpoints.ndim == 3 and endpoints.shape == (batch_size, 2, edge_count):
            endpoints = endpoints.transpose(1, 2)
        if endpoints.ndim == 2:
            endpoints = endpoints.unsqueeze(0).expand(batch_size, -1, -1)
        if endpoints.shape != (batch_size, edge_count, 2):
            raise ValueError("dag_edge_index shape does not match DAG presence")
        present = batch.history["dag_edge_present"][:, -1].bool()
        valid_slots = batch.static.get(
            "dag_edge_valid",
            torch.ones_like(present),
        ).bool()
        active = present & valid_slots
        valid_endpoint = (
            (endpoints[..., 0] >= 0)
            & (endpoints[..., 0] < task_count)
            & (endpoints[..., 1] >= 0)
            & (endpoints[..., 1] < task_count)
        )
        if (active & ~valid_endpoint).any():
            raise ValueError("dag_edge_index contains an active invalid task index")

        for batch_index in range(batch_size):
            adjacency = [[] for _ in range(task_count)]
            indegree = [0 for _ in range(task_count)]
            for edge_index in torch.nonzero(active[batch_index], as_tuple=False).flatten():
                source, destination = endpoints[batch_index, edge_index].tolist()
                adjacency[int(source)].append(int(destination))
                indegree[int(destination)] += 1
            frontier = [index for index, degree in enumerate(indegree) if degree == 0]
            visited = 0
            while frontier:
                node = frontier.pop()
                visited += 1
                for destination in adjacency[node]:
                    indegree[destination] -= 1
                    if indegree[destination] == 0:
                        frontier.append(destination)
            if visited != task_count:
                raise ValueError("active task dependency graph must be acyclic")
        return endpoints.long(), active

    def _apply_dag_messages(self, state, batch: ExplicitStateBatch):
        endpoints, active = self._dag_structure(
            batch,
            task_count=state.task.shape[1],
        )
        relation = self.dag_relation_embedding.view(1, 1, -1).expand(
            state.task.shape[0],
            endpoints.shape[1],
            -1,
        )
        task_weight = batch.history["task_present"][:, -1].to(state.task.dtype)
        incoming, outgoing, _ = directed_relation_messages(
            state.task,
            relation,
            endpoints,
            task_weight,
            active.to(state.task.dtype),
        )
        state.task = _apply_gru(
            self.dag_task_transition,
            self.dag_task_message(torch.cat((incoming, outgoing), dim=-1)),
            state.task,
        )
        return state

    def _graph_update(self, state, batch: ExplicitStateBatch):
        state = super()._graph_update(state, batch)
        return self._apply_dag_messages(state, batch)


class _SoftPredictedPresenceBackend(R3ReferenceWorldModel):
    _GRAPH_PRESENCE_KEYS = (
        "physical_node_present",
        "physical_edge_present",
        "information_node_present",
        "information_edge_present",
    )

    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__(config)
        self._soft_presence: dict[str, torch.Tensor] | None = None

    def infer_belief(self, batch: ExplicitStateBatch):
        self._soft_presence = None
        return super().infer_belief(batch)

    def _batch_with_soft_presence(
        self,
        batch: ExplicitStateBatch,
    ) -> ExplicitStateBatch:
        if self._soft_presence is None:
            return batch
        history = dict(batch.history)
        for key, soft in self._soft_presence.items():
            original = history[key]
            history[key] = torch.cat(
                (original[:, :-1].to(soft.dtype), soft.unsqueeze(1)),
                dim=1,
            )
        return ExplicitStateBatch(
            history=history,
            history_action=batch.history_action,
            future_action=batch.future_action,
            target=batch.target,
            static=batch.static,
            metadata=batch.metadata,
        )

    def _graph_update(self, state, batch: ExplicitStateBatch):
        return super()._graph_update(state, self._batch_with_soft_presence(batch))

    def _refresh_global(self, state, batch: ExplicitStateBatch):
        return super()._refresh_global(state, self._batch_with_soft_presence(batch))

    def _predict(self, state):
        explicit, logits = super()._predict(state)
        self._soft_presence = {
            key: torch.sigmoid(logits[key])
            for key in self._GRAPH_PRESENCE_KEYS
        }
        return explicit, logits


class R4WorldModel(nn.Module):
    """Adapter exposing one auditable R4 component configuration."""

    schema_version = R4_MODEL_SCHEMA

    def __init__(
        self,
        config: R4ModuleConfig,
        backend: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend

    def component_registry(self) -> dict[str, str]:
        return self.config.component_names()

    def infer_belief(self, batch: ExplicitStateBatch):
        return self.backend.infer_belief(batch)

    def rollout(
        self,
        belief,
        future_action: dict[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R3RolloutOutput | R4RolloutOutput:
        return self.backend.rollout(
            belief,
            future_action,
            batch,
            rollout_steps=rollout_steps,
        )

    def forward(
        self,
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R3RolloutOutput | R4RolloutOutput:
        return self.backend(batch, rollout_steps=rollout_steps)


def build_r4_world_model(config: R4ModuleConfig) -> R4WorldModel:
    """Build only configurations whose named behavior is actually implemented."""

    validate_controlled_config(config, allow_statuses=KNOWN_STATUSES)
    assert_executable_config(config)
    use_cross_graph_coupling = config.coupling != "no_cross_graph_coupling_v1"
    backend_config = R3ReferenceConfig(
        hidden_dim=config.hidden_dim,
        history_steps=config.history_steps,
        use_cross_graph_coupling=use_cross_graph_coupling,
    )
    if config.head == "hurdle_active_rate_v1":
        backend = _HurdleRateBackend(
            backend_config,
            rate_mean=float(config.information_rate_mean),
            rate_scale=float(config.information_rate_scale),
        )
    elif config.presence == "soft_predicted_presence_v1":
        backend = _SoftPredictedPresenceBackend(backend_config)
    elif config.dag == "explicit_dag_message_passing_v1":
        backend = _ExplicitDAGBackend(backend_config)
    elif config.head == "heteroscedastic_typed_v1":
        backend = _HeteroscedasticBackend(backend_config)
    elif config.dynamics == "graph_rssm_v1":
        backend = _GraphRSSMBackend(backend_config)
    elif config.coupling == "relation_constrained_cross_attention_v1":
        backend = _CrossAttentionBackend(backend_config)
    elif config.graph_encoder == "rgcn_v1":
        backend = _CandidateGraphBackend(backend_config, mode="rgcn")
    elif config.graph_encoder == "edge_conditioned_relation_mpnn_v1":
        backend = _CandidateGraphBackend(backend_config, mode="ecc")
    else:
        backend = R3ReferenceWorldModel(backend_config)
    if config.field_encoder == "symlog_masked_mlp_v1":
        _replace_field_encoders(
            backend,
            mode="symlog",
            hidden_dim=config.hidden_dim,
        )
    elif config.field_encoder == "simnorm_masked_mlp_v1":
        _replace_field_encoders(
            backend,
            mode="simnorm",
            hidden_dim=config.hidden_dim,
        )
    return R4WorldModel(config, backend)


__all__ = [
    "BeliefSequence",
    "R4_MODEL_SCHEMA",
    "R4RolloutOutput",
    "R4WorldModel",
    "R3RolloutOutput",
    "build_r4_world_model",
    "relation_constrained_attention",
    "simnorm",
    "symlog",
]
