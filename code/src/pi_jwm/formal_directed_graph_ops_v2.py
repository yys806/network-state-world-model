"""Directed, soft-weighted graph operators for PI-JWM v2."""

from __future__ import annotations

import torch


def _batched_rows(value: torch.Tensor, batch_size: int, width: int, name: str) -> torch.Tensor:
    if value.ndim == 2 and value.shape[-1] == width:
        return value.unsqueeze(0).expand(batch_size, -1, -1)
    if value.ndim == 3 and value.shape[0] == batch_size and value.shape[-1] == width:
        return value
    raise ValueError(f"{name} must have shape [items, {width}] or [batch, items, {width}]")


def _batched_index(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0).expand(batch_size, -1)
    if value.ndim == 2 and value.shape[0] == batch_size:
        return value
    raise ValueError("index must have shape [items] or [batch, items]")


def weighted_index_mean(
    messages: torch.Tensor,
    index: torch.Tensor,
    output_size: int,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Aggregate gated messages, normalized by structural relation count.

    The denominator deliberately counts active structural slots instead of
    summing their soft weights.  Consequently a presence probability remains
    a message-strength gate even when an entity has only one incident edge.
    """

    if messages.ndim != 3:
        raise ValueError("messages must have shape [batch, items, features]")
    batch_size, item_count, feature_count = messages.shape
    indices = _batched_index(index.to(messages.device), batch_size)
    if indices.shape[1] != item_count or weight.shape != (batch_size, item_count):
        raise ValueError("message, index, and weight shapes do not align")
    if output_size < 0:
        raise ValueError("output_size must be nonnegative")

    output = messages.new_zeros((batch_size, output_size, feature_count))
    denominator = messages.new_zeros((batch_size, output_size, 1))
    numeric_weight = weight.to(device=messages.device, dtype=messages.dtype).clamp_min(0.0)
    for batch_index in range(batch_size):
        valid = (
            torch.isfinite(numeric_weight[batch_index])
            & (numeric_weight[batch_index] > 0)
            & (indices[batch_index] >= 0)
            & (indices[batch_index] < output_size)
        )
        if not torch.any(valid):
            continue
        destinations = indices[batch_index, valid].long()
        selected_weight = numeric_weight[batch_index, valid].unsqueeze(-1)
        output[batch_index].index_add_(
            0,
            destinations,
            messages[batch_index, valid] * selected_weight,
        )
        denominator[batch_index].index_add_(
            0,
            destinations,
            torch.ones_like(selected_weight),
        )
    return output / denominator.clamp_min(1e-12)


def directed_relation_messages(
    entity_latent: torch.Tensor,
    relation_latent: torch.Tensor,
    endpoints: torch.Tensor,
    entity_weight: torch.Tensor,
    relation_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return incoming, outgoing, and ordered source/destination relation context."""

    if entity_latent.ndim != 3 or relation_latent.ndim != 3:
        raise ValueError("entity_latent and relation_latent must be batched rank-3 tensors")
    batch_size, entity_count, feature_count = entity_latent.shape
    if relation_latent.shape[0] != batch_size or relation_latent.shape[-1] != feature_count:
        raise ValueError("entity and relation latent dimensions do not align")
    relation_count = relation_latent.shape[1]
    endpoint_index = _batched_rows(endpoints.to(entity_latent.device), batch_size, 2, "endpoints")
    if endpoint_index.shape[1] != relation_count:
        raise ValueError("endpoint count does not match relation count")
    if entity_weight.shape != (batch_size, entity_count):
        raise ValueError("entity_weight shape does not match entities")
    if relation_weight.shape != (batch_size, relation_count):
        raise ValueError("relation_weight shape does not match relations")

    src = endpoint_index[..., 0]
    dst = endpoint_index[..., 1]
    endpoint_valid = (
        (src >= 0)
        & (src < entity_count)
        & (dst >= 0)
        & (dst < entity_count)
    )
    safe_src = src.clamp(0, max(entity_count - 1, 0)).long()
    safe_dst = dst.clamp(0, max(entity_count - 1, 0)).long()
    numeric_entity_weight = entity_weight.to(entity_latent.dtype).clamp_min(0.0)
    src_weight = torch.gather(numeric_entity_weight, 1, safe_src)
    dst_weight = torch.gather(numeric_entity_weight, 1, safe_dst)
    effective_weight = (
        relation_weight.to(entity_latent.dtype).clamp_min(0.0)
        * src_weight
        * dst_weight
        * endpoint_valid.to(entity_latent.dtype)
    )

    incoming = weighted_index_mean(relation_latent, dst, entity_count, effective_weight)
    outgoing = weighted_index_mean(relation_latent, src, entity_count, effective_weight)
    src_latent = torch.gather(
        entity_latent,
        1,
        safe_src.unsqueeze(-1).expand(-1, -1, feature_count),
    )
    dst_latent = torch.gather(
        entity_latent,
        1,
        safe_dst.unsqueeze(-1).expand(-1, -1, feature_count),
    )
    relation_context = torch.cat((src_latent, dst_latent), dim=-1)
    relation_context = relation_context * (effective_weight > 0).unsqueeze(-1).to(entity_latent.dtype)
    return incoming, outgoing, relation_context


def direct_bearer_candidates(
    flow_endpoints: torch.Tensor,
    physical_edge_endpoints: torch.Tensor,
    flow_valid: torch.Tensor,
    edge_valid: torch.Tensor,
) -> torch.Tensor:
    """Build direct single-hop CFE candidates with exact directed endpoint matching."""

    if flow_valid.ndim != 2 or edge_valid.ndim != 2 or flow_valid.shape[0] != edge_valid.shape[0]:
        raise ValueError("flow_valid and edge_valid must be batch-aligned rank-2 tensors")
    batch_size, flow_count = flow_valid.shape
    edge_count = edge_valid.shape[1]
    flows = _batched_rows(flow_endpoints.to(flow_valid.device), batch_size, 2, "flow_endpoints")
    edges = _batched_rows(
        physical_edge_endpoints.to(flow_valid.device),
        batch_size,
        2,
        "physical_edge_endpoints",
    )
    if flows.shape[1] != flow_count or edges.shape[1] != edge_count:
        raise ValueError("endpoint counts do not match validity weights")

    valid_flow_endpoint = (flows[..., 0] >= 0) & (flows[..., 1] >= 0)
    valid_edge_endpoint = (edges[..., 0] >= 0) & (edges[..., 1] >= 0)
    same_source = flows[:, :, None, 0] == edges[:, None, :, 0]
    same_destination = flows[:, :, None, 1] == edges[:, None, :, 1]
    candidate = (
        same_source
        & same_destination
        & valid_flow_endpoint.unsqueeze(-1)
        & valid_edge_endpoint.unsqueeze(1)
        & (flow_valid > 0).unsqueeze(-1)
        & (edge_valid > 0).unsqueeze(1)
    )
    dtype = flow_valid.dtype if flow_valid.is_floating_point() else torch.float32
    return candidate.to(dtype=dtype)


__all__ = [
    "direct_bearer_candidates",
    "directed_relation_messages",
    "weighted_index_mean",
]
