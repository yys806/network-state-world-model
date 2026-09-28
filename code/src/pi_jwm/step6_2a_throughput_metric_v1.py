"""Shared Flow/Outcome throughput definition for future PI-JWM and baselines.

Only real observed hop events are accepted. Predicted rollout diagnostics must
use separate labels and cannot be reported as closed-loop measurements.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def extract_real_throughput(events: Sequence[Mapping[str, Any]], *, elapsed_s: float,
                            flow_definitions: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("positive real elapsed simulation time required")
    service = useful = 0.0
    remaining = {flow_id: float(row.get("initial_e2e_remaining", row["total_data"]))
                 for flow_id, row in flow_definitions.items()}
    for event in events:
        flow_id = str(event["flow_id"])
        if flow_id not in flow_definitions:
            raise ValueError("unknown logical Flow identity")
        amount = float(event["delivered_data"])
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid real hop bytes")
        service += amount
        after = float(event["e2e_remaining_after"])
        if not math.isfinite(after) or after < -1e-9 or after > remaining[flow_id] + 1e-9:
            raise ValueError("nonconserving E2E Flow event")
        delta = remaining[flow_id] - after
        if delta > amount + 1e-9:
            raise ValueError("E2E delivery exceeds observed hop service")
        if delta > 1e-9 and str(event["target_id"]) != str(flow_definitions[flow_id]["logical_destination"]):
            raise ValueError("nonterminal hop reduced E2E remaining")
        useful += max(delta, 0.0)
        remaining[flow_id] = after
    return {"END_TO_END_USEFUL_THROUGHPUT": useful / elapsed_s,
            "NETWORK_SERVICE_THROUGHPUT": service / elapsed_s,
            "e2e_useful_bytes": useful, "all_hop_service_bytes": service}
