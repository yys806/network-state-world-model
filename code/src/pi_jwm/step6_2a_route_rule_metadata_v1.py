"""Planner/trainer Route control side-state, separate from learned tensors."""
from __future__ import annotations

from typing import Any, Mapping

import torch


def build_route_rule_metadata(sample: Mapping[str, Any], state: Mapping[str, torch.Tensor],
                              action: Mapping[str, torch.Tensor], horizon: int,
                              *, batch_index: int = 0) -> tuple[dict[str, Any], ...]:
    """Use only the supplied action at this horizon and the current causal state."""
    entries = sample["future_action"][horizon]["route"]["entries"]
    if len(entries) > action["route_flow_index"].shape[1]:
        raise ValueError("ROUTE_METADATA_ACTION_ROW_MISMATCH")
    out = []
    for row, entry in enumerate(entries):
        fi = int(action["route_flow_index"][0, row])
        if fi < 0:
            continue  # pending Flow: fixed-support model cannot create it
        if "current_holder_index" not in state or "flow_route_revision" not in state:
            raise ValueError("ROUTE_RULE_METADATA_CURRENT_STATE_REQUIRED")
        out.append({"batch_index": batch_index, "action_row": row,
                    "flow_index": fi, "task_index": int(action["route_task_index"][0, row]),
                    "current_holder_index": int(state["current_holder_index"][0, fi]),
                    "route_revision_before": int(state["flow_route_revision"][0, fi]),
                    "route_node_indices": tuple(int(x) for x in entry.get("route_node_indices", ()))})
    return tuple(out)
