"""Read-only, component-level audit of frozen future Return support."""
from __future__ import annotations

from typing import Any, Mapping


def detect_future_return_birth(sample: Mapping[str, Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    """Classify real future Return rows against the causal current Flow index.

    This returns side metadata only. It never adds a current slot or masks an
    unrelated Motion/CSI component.
    """
    current = sample.get("static", {}).get("input_entity_index", {}).get("logical_flow", {})
    current_ids = set(current)
    unsupported: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for row in frame.get("logical_flows", []):
        if row.get("flow_type") != "Return":
            continue
        # A Return that was born and completed between Decisions remains a
        # known COMPLETED target row even though its current presence is false.
        if not (row.get("presence", False) or row.get("status") == "COMPLETED"):
            continue
        flow_id = str(row.get("flow_id", ""))
        if not row.get("known", False) or not flow_id:
            unresolved.append({"task_id": str(row.get("task_id", "")), "reason": "future_return_identity_unresolved"})
            continue
        if flow_id not in current_ids:
            unsupported.append({"flow_id": flow_id, "task_id": str(row.get("task_id", "")), "reason": "future_return_birth_outside_current_fixed_support"})
    return {
        "unsupported": unsupported,
        "unresolved": unresolved,
        "fixed_support_blocked": list(unsupported),
        "unsupported_count": len(unsupported),
        "unresolved_count": len(unresolved),
        "fixed_support_blocked_count": len(unsupported),
        "current_input_index_unchanged": True,
        "window_retained": True,
    }
