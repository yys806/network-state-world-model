"""Flat baselines for PI-JWM v6 comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class V6FlatBaselineConfig:
    input_dim: int
    node_dim: int
    task_dim: int
    num_nodes: int
    num_edges: int
    horizon: int
    hidden_dim: int = 256


class V6FlatBaseline(nn.Module):
    """A non-graph MLP baseline using flattened PI-JWM inputs.

    The baseline receives the same broad information families as PI-JWM v6,
    but it discards node/edge graph structure by flattening every tensor into
    one vector before prediction.
    """

    def __init__(self, config: V6FlatBaselineConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(config.input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.node_head = nn.Linear(hidden, config.horizon * config.num_nodes * config.node_dim)
        self.link_activity_head = nn.Linear(hidden, config.horizon * config.num_edges)
        self.link_rate_head = nn.Linear(hidden, config.horizon * config.num_edges)
        self.task_head = nn.Linear(hidden, config.horizon * config.task_dim)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        encoded = self.backbone(inputs)
        cfg = self.config
        node = self.node_head(encoded).reshape(-1, cfg.horizon, cfg.num_nodes, cfg.node_dim)
        activity = self.link_activity_head(encoded).reshape(-1, cfg.horizon, cfg.num_edges, 1)
        rate = self.link_rate_head(encoded).reshape(-1, cfg.horizon, cfg.num_edges, 1)
        task = self.task_head(encoded).reshape(-1, cfg.horizon, cfg.task_dim)
        return {
            "node": node,
            "link_activity_logit": activity,
            "link_rate": rate,
            "task": task,
        }
