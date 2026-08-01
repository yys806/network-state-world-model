"""Masked sparse graph operations for the strict PI-JWM dual graph."""

from __future__ import annotations

import torch


def _batched_index(index: torch.Tensor, batch_size: int) -> torch.Tensor:
    if index.ndim == 1:
        return index.unsqueeze(0).expand(batch_size, -1)
    if index.ndim == 2 and index.shape[0] == batch_size:
        return index
    raise ValueError("index must have shape [items] or [batch, items]")


def masked_index_mean(
    messages: torch.Tensor,
    index: torch.Tensor,
    output_size: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool batched messages by a possibly padded destination index."""

    if messages.ndim != 3:
        raise ValueError("messages must have shape [batch, items, features]")
    batch_size, item_count, feature_count = messages.shape
    indices = _batched_index(index.to(messages.device), batch_size)
    if indices.shape[1] != item_count or valid_mask.shape != (batch_size, item_count):
        raise ValueError("message, index, and mask shapes do not align")
    output = messages.new_zeros((batch_size, output_size, feature_count))
    counts = messages.new_zeros((batch_size, output_size, 1))
    for batch_index in range(batch_size):
        valid = (
            valid_mask[batch_index].bool()
            & (indices[batch_index] >= 0)
            & (indices[batch_index] < output_size)
        )
        if not torch.any(valid):
            continue
        destinations = indices[batch_index, valid].long()
        output[batch_index].index_add_(0, destinations, messages[batch_index, valid])
        counts[batch_index].index_add_(
            0,
            destinations,
            torch.ones((int(valid.sum()), 1), dtype=messages.dtype, device=messages.device),
        )
    return output / counts.clamp_min(1.0)


def _endpoint_message_pass(
    entity_latent: torch.Tensor,
    relation_latent: torch.Tensor,
    endpoints: torch.Tensor,
    entity_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, entity_count, _ = entity_latent.shape
    relation_count = relation_latent.shape[1]
    if endpoints.ndim == 2:
        endpoint_index = endpoints.unsqueeze(0).expand(batch_size, -1, -1)
    elif endpoints.ndim == 3 and endpoints.shape[0] == batch_size:
        endpoint_index = endpoints
    else:
        raise ValueError("endpoints must have shape [relations, 2] or [batch, relations, 2]")
    if endpoint_index.shape[1:] != (relation_count, 2):
        raise ValueError("endpoint count does not match relation count")

    src = endpoint_index[..., 0].to(entity_latent.device)
    dst = endpoint_index[..., 1].to(entity_latent.device)
    valid_src = (src >= 0) & (src < entity_count)
    valid_dst = (dst >= 0) & (dst < entity_count)
    safe_src = src.clamp(0, max(entity_count - 1, 0)).long()
    safe_dst = dst.clamp(0, max(entity_count - 1, 0)).long()
    src_exists = torch.gather(entity_mask.bool(), 1, safe_src) & valid_src
    dst_exists = torch.gather(entity_mask.bool(), 1, safe_dst) & valid_dst
    relation_valid = relation_mask.bool() & src_exists & dst_exists

    duplicated_messages = torch.cat((relation_latent, relation_latent), dim=1)
    duplicated_index = torch.cat((src, dst), dim=1)
    duplicated_mask = torch.cat((relation_valid, relation_valid), dim=1)
    entity_messages = masked_index_mean(
        duplicated_messages,
        duplicated_index,
        entity_count,
        duplicated_mask,
    )

    feature_count = entity_latent.shape[-1]
    src_latent = torch.gather(
        entity_latent, 1, safe_src.unsqueeze(-1).expand(-1, -1, feature_count)
    )
    dst_latent = torch.gather(
        entity_latent, 1, safe_dst.unsqueeze(-1).expand(-1, -1, feature_count)
    )
    relation_messages = 0.5 * (src_latent + dst_latent)
    relation_messages = relation_messages * relation_valid.unsqueeze(-1).to(relation_messages.dtype)
    return entity_messages, relation_messages


def physical_message_pass(
    node_latent: torch.Tensor,
    edge_latent: torch.Tensor,
    endpoints: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _endpoint_message_pass(node_latent, edge_latent, endpoints, node_mask, edge_mask)


def information_message_pass(
    agent_latent: torch.Tensor,
    flow_latent: torch.Tensor,
    endpoints: torch.Tensor,
    agent_mask: torch.Tensor,
    flow_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _endpoint_message_pass(agent_latent, flow_latent, endpoints, agent_mask, flow_mask)


def dag_message_pass(
    task_latent: torch.Tensor,
    dag_edges: torch.Tensor,
    dag_edge_mask: torch.Tensor,
    task_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate parent-task latent messages at child tasks."""

    batch_size, task_count, feature_count = task_latent.shape
    if dag_edges.ndim == 2:
        if dag_edges.shape[0] != 2:
            raise ValueError("static dag_edges must have shape [2, edges]")
        edge_index = dag_edges.unsqueeze(0).expand(batch_size, -1, -1)
    elif dag_edges.ndim == 3 and dag_edges.shape[:2] == (batch_size, 2):
        edge_index = dag_edges
    else:
        raise ValueError("dag_edges must have shape [2, edges] or [batch, 2, edges]")
    parents = edge_index[:, 0].to(task_latent.device)
    children = edge_index[:, 1].to(task_latent.device)
    safe_parents = parents.clamp(0, max(task_count - 1, 0)).long()
    safe_children = children.clamp(0, max(task_count - 1, 0)).long()
    parent_exists = torch.gather(task_mask.bool(), 1, safe_parents)
    child_exists = torch.gather(task_mask.bool(), 1, safe_children)
    valid = (
        dag_edge_mask.bool()
        & (parents >= 0)
        & (parents < task_count)
        & (children >= 0)
        & (children < task_count)
        & parent_exists
        & child_exists
    )
    parent_latent = torch.gather(
        task_latent,
        1,
        safe_parents.unsqueeze(-1).expand(-1, -1, feature_count),
    )
    return masked_index_mean(parent_latent, children, task_count, valid)


def couple_agent_physical(
    agent_latent: torch.Tensor,
    node_latent: torch.Tensor,
    agent_node_index: torch.Tensor,
    agent_mask: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange CIP messages between agents and their attached physical nodes."""

    batch_size, agent_count, feature_count = agent_latent.shape
    node_count = node_latent.shape[1]
    attachment = _batched_index(agent_node_index.to(agent_latent.device), batch_size)
    if attachment.shape[1] != agent_count:
        raise ValueError("agent attachment count does not match agent count")
    safe = attachment.clamp(0, max(node_count - 1, 0)).long()
    attached_node_exists = torch.gather(node_mask.bool(), 1, safe)
    valid = agent_mask.bool() & (attachment >= 0) & (attachment < node_count) & attached_node_exists
    gathered_nodes = torch.gather(
        node_latent,
        1,
        safe.unsqueeze(-1).expand(-1, -1, feature_count),
    )
    agent_from_node = gathered_nodes * valid.unsqueeze(-1).to(gathered_nodes.dtype)
    node_from_agent = masked_index_mean(agent_latent, attachment, node_count, valid)
    return agent_from_node, node_from_agent


def couple_flow_bearer(
    flow_latent: torch.Tensor,
    edge_latent: torch.Tensor,
    flow_bearer_mask: torch.Tensor,
    flow_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange CFE messages over the observed flow-to-physical-edge relation."""

    if flow_bearer_mask.shape != (
        flow_latent.shape[0],
        flow_latent.shape[1],
        edge_latent.shape[1],
    ):
        raise ValueError("flow bearer mask does not match flow and edge counts")
    relation = (
        flow_bearer_mask.bool()
        & flow_mask.bool().unsqueeze(-1)
        & edge_mask.bool().unsqueeze(1)
    ).to(flow_latent.dtype)
    flow_counts = relation.sum(dim=2, keepdim=True).clamp_min(1.0)
    edge_counts = relation.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1.0)
    flow_from_edge = torch.bmm(relation, edge_latent) / flow_counts
    edge_from_flow = torch.bmm(relation.transpose(1, 2), flow_latent) / edge_counts
    return flow_from_edge, edge_from_flow


__all__ = [
    "couple_agent_physical",
    "couple_flow_bearer",
    "dag_message_pass",
    "information_message_pass",
    "masked_index_mean",
    "physical_message_pass",
]
