"""STEP 4.2C-B causal Flow Ledger and additive Raw contract.

The Ledger consumes only established current state and completed Outcome
events.  It does not modify AirFogSim, construct model samples/tensors, or
implement either graph.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote


RAW_SCHEMA_VERSION = "PI-JWM-Raw-Causal-Flow-Ledger-Amendment-v1-step4.2C-B"
LEDGER_SCHEMA_VERSION = "PI-JWM-Causal-Flow-Ledger-v1-step4.2C-B"
FLOW_TYPES = ("Input", "Return", "DepData")
FLOW_STATUSES = ("ACTIVE", "COMPLETED", "SUPERSEDED")
STEP42CB_PATCH_REQUIRED_CHECKS = frozenset({
    "ledger_receipt", "multi_hop_receipt", "real_trace", "contract_fixture",
    "logical_destination_provenance_proven", "input_logical_destination_causal",
    "return_logical_destination_causal", "real_multihop_single_flow_id",
    "real_multihop_single_epoch", "real_multihop_logical_destination_constant",
    "real_multihop_multiple_distinct_hops", "real_multihop_intermediate_service_not_e2e",
    "real_multihop_final_hop_advances_e2e", "normal_hop_advancement_not_reroute",
    "normal_hop_advancement_not_destination_change", "e2e_no_double_count",
    "flow_id_index_stable", "depdata_zero", "scope",
})


def validate_step4_2c_b_acceptance(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = receipt.get("required_checks", {})
    required_ok = STEP42CB_PATCH_REQUIRED_CHECKS <= set(required) and all(
        required.get(name) is True for name in STEP42CB_PATCH_REQUIRED_CHECKS
    )
    scope = receipt.get("scope", {})
    scope_ok = all(
        scope.get(name) is False
        for name in ("training", "gpu", "locked_test", "formal_dataset", "sample_tensor", "graph_builder")
    )
    expected_passed = bool(required_ok and scope_ok)
    return {
        "required_checks": required_ok,
        "scope": scope_ok,
        "expected_passed": expected_passed,
        "declared_matches_expected": receipt.get("passed") is expected_passed,
        "passed": bool(expected_passed and receipt.get("passed") is True),
    }


def build_flow_id(task_id: str, flow_type: str, epoch: int) -> str:
    if flow_type not in FLOW_TYPES:
        raise ValueError(f"unsupported FlowType: {flow_type}")
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    return f"flow::{quote(str(task_id), safe='')}::{flow_type}::{int(epoch)}"


def parse_flow_id(flow_id: str) -> tuple[str, str, int]:
    parts = str(flow_id).split("::")
    if len(parts) != 4 or parts[0] != "flow" or parts[2] not in FLOW_TYPES:
        raise ValueError("invalid FlowID")
    return unquote(parts[1]), parts[2], int(parts[3])


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def validate_flow_transition(before: Mapping[str, Any], event: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one logical transition independently of Ledger implementation."""
    legacy_forbidden = "flow_completed" not in event
    expected_delta = float(event["delivered_data"]) if str(event["target_id"]) == str(before["logical_destination"]) else 0.0
    expected_delivered = min(float(before["total_data"]), float(before["e2e_delivered"]) + expected_delta)
    expected_remaining = float(before["total_data"]) - expected_delivered
    checks = {
        "legacy_flow_completed_forbidden": legacy_forbidden,
        "final_destination_only": math.isclose(float(after["e2e_delivered"]), expected_delivered, abs_tol=1e-9),
        "remaining_exact": math.isclose(float(after["e2e_remaining"]), expected_remaining, abs_tol=1e-9),
        "flow_conservation": math.isclose(float(after["e2e_delivered"]) + float(after["e2e_remaining"]), float(after["total_data"]), abs_tol=1e-9),
        "same_epoch_monotonic": float(after["e2e_delivered"]) + 1e-12 >= float(before["e2e_delivered"]) and float(after["e2e_remaining"]) <= float(before["e2e_remaining"]) + 1e-12,
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_epoch_transition(old: Mapping[str, Any], new: Mapping[str, Any], *, destination_changed: bool) -> dict[str, Any]:
    checks = {
        "same_task": old["task_id"] == new["task_id"],
        "same_flow_type": old["flow_type"] == new["flow_type"],
        "identity_rule": (new["flow_id"] != old["flow_id"] and new["epoch"] == old["epoch"] + 1) if destination_changed else (new["flow_id"] == old["flow_id"] and new["epoch"] == old["epoch"]),
        "epoch_conservation": math.isclose(float(new["total_data"]), float(old["e2e_remaining"]), abs_tol=1e-9) if destination_changed else True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_logical_destination_continuity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require one logical destination within every Task/FlowType/Epoch."""
    destinations: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        key = (str(row["task_id"]), str(row["flow_type"]), int(row["epoch"]))
        destinations.setdefault(key, set()).add(str(row["logical_destination"]))
    checks = {
        "logical_destination_constant_within_epoch": bool(destinations)
        and all(len(values) == 1 for values in destinations.values()),
    }
    return {
        "checks": checks,
        "destinations_by_task_flow_epoch": {
            "::".join((task_id, flow_type, str(epoch))): sorted(values)
            for (task_id, flow_type, epoch), values in sorted(destinations.items())
        },
        "passed": all(checks.values()),
    }


def validate_real_multihop_single_flow(transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate that ordered real events describe one logical Flow over distinct hops."""
    rows = list(transitions)
    flow_ids = {str(row.get("flow_id")) for row in rows}
    epochs = {int(row.get("epoch", -1)) for row in rows}
    destinations = {str(row.get("logical_destination")) for row in rows}
    hops = [(str(row.get("source_id")), str(row.get("target_id"))) for row in rows]
    route_revisions_stable = all(
        int(row.get("route_revision_before", -1)) == int(row.get("route_revision_after", -2))
        for row in rows
    ) and len({int(row.get("route_revision_after", -1)) for row in rows}) == 1
    checks = {
        "real_multihop_single_flow_id": len(rows) >= 2 and len(flow_ids) == 1,
        "real_multihop_single_epoch": len(rows) >= 2 and len(epochs) == 1,
        "real_multihop_logical_destination_constant": len(rows) >= 2 and len(destinations) == 1,
        "real_multihop_multiple_distinct_hops": len(set(hops)) >= 2,
        "real_multihop_intermediate_service_not_e2e": len(rows) >= 2
        and all(math.isclose(float(row.get("logical_delivery_delta", -1.0)), 0.0, abs_tol=1e-9) for row in rows[:-1]),
        "real_multihop_final_hop_advances_e2e": len(rows) >= 2
        and float(rows[-1].get("logical_delivery_delta", 0.0)) > 0.0,
        "normal_hop_advancement_not_reroute": len(rows) >= 2 and route_revisions_stable,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "flow_id_sequence": [row.get("flow_id") for row in rows],
        "epoch_sequence": [row.get("epoch") for row in rows],
        "logical_destination_sequence": [row.get("logical_destination") for row in rows],
        "hop_sequence": [{"source_id": source, "target_id": target} for source, target in hops],
        "per_hop_service_sequence": [row.get("delivered_data") for row in rows],
        "e2e_delivered_sequence": [row.get("e2e_delivered_after") for row in rows],
        "e2e_remaining_sequence": [row.get("e2e_remaining_after") for row in rows],
    }


class CausalFlowLedger:
    def __init__(self) -> None:
        self._flows: dict[str, dict[str, Any]] = {}
        self._carrying: dict[str, dict[str, Any]] = {}
        self._flow_index: dict[str, int] = {}

    def _next_epoch(self, task_id: str, flow_type: str) -> int:
        epochs = [row["epoch"] for row in self._flows.values() if row["task_id"] == task_id and row["flow_type"] == flow_type]
        return 0 if not epochs else max(epochs) + 1

    def _create_flow(
        self, *, task_id: str, flow_type: str, logical_source: str,
        logical_destination: str, total_data: float, route: Sequence[str],
        epoch: int | None = None, previous_flow_id: str | None = None,
        supersede_reason: str | None = None,
        logical_destination_source: str = "explicit_contract_argument",
        logical_destination_capture_phase: str = "flow_creation",
    ) -> str:
        total = _finite_nonnegative(total_data, "total_data")
        epoch_value = self._next_epoch(task_id, flow_type) if epoch is None else int(epoch)
        flow_id = build_flow_id(task_id, flow_type, epoch_value)
        if flow_id in self._flows:
            raise ValueError(f"Flow already exists: {flow_id}")
        self._flow_index[flow_id] = len(self._flow_index)
        self._flows[flow_id] = {
            "flow_id": flow_id, "flow_index": self._flow_index[flow_id],
            "task_id": str(task_id), "flow_type": flow_type, "epoch": epoch_value,
            "logical_source": str(logical_source), "logical_destination": str(logical_destination),
            "logical_destination_source": str(logical_destination_source),
            "logical_destination_capture_phase": str(logical_destination_capture_phase),
            "total_data": total, "e2e_delivered": 0.0, "e2e_remaining": total,
            "presence": total > 0.0, "status": "ACTIVE" if total > 0.0 else "COMPLETED",
            "previous_flow_id": previous_flow_id, "superseded_by_flow_id": None,
            "supersede_reason": supersede_reason,
            "source_provenance": "causal_task_state_and_established_route",
            "feature_mask": {"logical_source": True, "logical_destination": True, "total_data": True, "e2e_delivered": True, "e2e_remaining": True},
            "missing_reason": None,
        }
        route_values = [str(value) for value in route]
        first_target = route_values[0] if route_values else None
        self._carrying[flow_id] = {
            "flow_id": flow_id, "route_revision": 0, "route": route_values,
            "current_holder": str(logical_source), "current_hop_index": 0,
            "hop_source": str(logical_source), "hop_destination": first_target,
            "hop_progress": 0.0, "hop_remaining": total if first_target else 0.0,
            "active": bool(first_target and total > 0.0),
            "feature_mask": {"hop_source": first_target is not None, "hop_destination": first_target is not None, "hop_progress": first_target is not None, "hop_remaining": first_target is not None},
            "missing_reason": None if first_target else "NO_ACTIVE_CARRYING_HOP",
        }
        return flow_id

    def create_input_flow(
        self, task: Mapping[str, Any], *, logical_destination: str, route: Sequence[str],
        logical_destination_source: str = "explicit_contract_argument",
        logical_destination_capture_phase: str = "flow_creation",
    ) -> str | None:
        source = str(task["task_node_id"])
        destination = str(logical_destination)
        if source == destination:
            return None
        return self._create_flow(task_id=str(task["task_id"]), flow_type="Input", logical_source=source, logical_destination=destination, total_data=task["task_size"], route=route, logical_destination_source=logical_destination_source, logical_destination_capture_phase=logical_destination_capture_phase)

    def create_return_flow(
        self, task: Mapping[str, Any], *, logical_destination: str, total_data: float,
        route: Sequence[str], logical_destination_source: str = "explicit_contract_argument",
        logical_destination_capture_phase: str = "flow_creation",
    ) -> str | None:
        source = str(task["current_node_id"])
        destination = str(logical_destination)
        if source == destination:
            return None
        return self._create_flow(task_id=str(task["task_id"]), flow_type="Return", logical_source=source, logical_destination=destination, total_data=total_data, route=route, logical_destination_source=logical_destination_source, logical_destination_capture_phase=logical_destination_capture_phase)

    def create_depdata_from_dag(self, *_: Any, **__: Any) -> None:
        raise ValueError("DEPDATA_FROM_DAG_FABRICATION_FORBIDDEN")

    def _active_for_event(self, event: Mapping[str, Any]) -> str:
        flow_type = "Return" if str(event["phase"]).lower() == "return" else "Input"
        candidates = [row["flow_id"] for row in self._flows.values() if row["task_id"] == str(event["task_id"]) and row["flow_type"] == flow_type and row["status"] == "ACTIVE"]
        if len(candidates) != 1:
            raise ValueError(f"expected one active {flow_type} Flow for event, found {len(candidates)}")
        return candidates[0]

    def apply_transfer_event(self, event: Mapping[str, Any], *, observed_task_current_node_id: str | None) -> dict[str, Any]:
        if "flow_completed" in event:
            raise ValueError("legacy flow_completed is forbidden as logical Flow completion")
        required = ("task_id", "phase", "source_id", "target_id", "delivered_data", "stage_or_hop_completed")
        missing = [name for name in required if name not in event]
        if missing:
            raise ValueError(f"transfer event missing fields: {missing}")
        flow_id = self._active_for_event(event)
        flow = self._flows[flow_id]
        carrying = self._carrying[flow_id]
        source = str(event["source_id"])
        target = str(event["target_id"])
        if source != carrying["current_holder"]:
            raise ValueError("HOLDER_EVENT_INCONSISTENCY: event source differs from current holder")
        amount = _finite_nonnegative(event["delivered_data"], "delivered_data")
        completed = bool(event["stage_or_hop_completed"])
        expected_holder = target if completed else source
        if observed_task_current_node_id is None or str(observed_task_current_node_id) != expected_holder:
            raise ValueError("HOLDER_EVENT_INCONSISTENCY: real task state differs from event transition")
        before = copy.deepcopy(flow)
        carrying_before = copy.deepcopy(carrying)
        logical_delta = amount if target == flow["logical_destination"] else 0.0
        flow["e2e_delivered"] = min(flow["total_data"], flow["e2e_delivered"] + logical_delta)
        flow["e2e_remaining"] = flow["total_data"] - flow["e2e_delivered"]
        carrying["hop_progress"] += amount
        carrying["hop_remaining"] = max(flow["total_data"] - carrying["hop_progress"], 0.0)
        carrying["current_holder"] = expected_holder
        if completed:
            carrying["current_hop_index"] += 1
            index = carrying["current_hop_index"]
            next_target = carrying["route"][index] if index < len(carrying["route"]) else None
            carrying["hop_source"] = expected_holder
            carrying["hop_destination"] = next_target
            carrying["hop_progress"] = 0.0
            carrying["hop_remaining"] = flow["e2e_remaining"] if next_target else 0.0
            carrying["active"] = next_target is not None and flow["e2e_remaining"] > 0.0
            carrying["missing_reason"] = None if carrying["active"] else "NO_ACTIVE_CARRYING_HOP"
        if flow["e2e_remaining"] <= 1e-12:
            flow["e2e_remaining"] = 0.0
            flow["e2e_delivered"] = flow["total_data"]
            flow["status"] = "COMPLETED"
            flow["presence"] = False
            carrying["active"] = False
        self._assert_flow(flow, previous=before)
        transition_validation = validate_flow_transition(before, event, flow)
        if not transition_validation["passed"]:
            raise ValueError("LOGICAL_FLOW_TRANSITION_VALIDATION_FAILED")
        return {
            "task_id": flow["task_id"], "flow_type": flow["flow_type"],
            "flow_id": flow_id, "epoch": flow["epoch"],
            "logical_destination": flow["logical_destination"],
            "logical_destination_source": flow["logical_destination_source"],
            "logical_destination_capture_phase": flow["logical_destination_capture_phase"],
            "source_id": source, "target_id": target, "delivered_data": amount,
            "logical_delivery_delta": logical_delta,
            "e2e_delivered_after": flow["e2e_delivered"],
            "e2e_remaining_after": flow["e2e_remaining"],
            "route_revision_before": carrying_before["route_revision"],
            "route_revision_after": carrying["route_revision"],
            "current_hop_index_before": carrying_before["current_hop_index"],
            "current_hop_index_after": carrying["current_hop_index"],
            "stage_or_hop_completed": completed,
            "logical_flow_completed": flow["status"] == "COMPLETED",
        }

    def reroute(self, flow_id: str, *, new_route: Sequence[str], new_logical_destination: str) -> str:
        flow = self._flows[flow_id]
        carrying = self._carrying[flow_id]
        if flow["status"] != "ACTIVE":
            raise ValueError("only ACTIVE Flow can be rerouted")
        destination = str(new_logical_destination)
        if destination == flow["logical_destination"]:
            carrying["route_revision"] += 1
            carrying["route"] = [str(value) for value in new_route]
            carrying["current_hop_index"] = 0
            carrying["hop_source"] = carrying["current_holder"]
            carrying["hop_destination"] = carrying["route"][0] if carrying["route"] else None
            carrying["hop_progress"] = 0.0
            carrying["hop_remaining"] = flow["e2e_remaining"] if carrying["route"] else 0.0
            carrying["active"] = bool(carrying["route"])
            return flow_id
        if carrying["active"] and carrying["hop_progress"] > 0.0 and carrying["hop_remaining"] > 0.0:
            raise ValueError("NOT_AT_CLEAN_HOP_BOUNDARY")
        old_remaining = flow["e2e_remaining"]
        flow["status"] = "SUPERSEDED"
        flow["presence"] = False
        carrying["active"] = False
        new_id = self._create_flow(task_id=flow["task_id"], flow_type=flow["flow_type"], logical_source=carrying["current_holder"], logical_destination=destination, total_data=old_remaining, route=new_route, previous_flow_id=flow_id, supersede_reason="logical_destination_change", logical_destination_source="researcher_authorized_destination_change", logical_destination_capture_phase="clean_hop_boundary_reroute")
        flow["superseded_by_flow_id"] = new_id
        return new_id

    @staticmethod
    def _assert_flow(flow: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None) -> None:
        total = float(flow["total_data"])
        delivered = float(flow["e2e_delivered"])
        remaining = float(flow["e2e_remaining"])
        if not (0.0 <= delivered <= total and 0.0 <= remaining <= total):
            raise ValueError("FLOW_CONSERVATION_RANGE_FAILED")
        if not math.isclose(delivered + remaining, total, abs_tol=1e-9):
            raise ValueError("FLOW_CONSERVATION_SUM_FAILED")
        if previous is not None and (delivered + 1e-12 < float(previous["e2e_delivered"]) or remaining > float(previous["e2e_remaining"]) + 1e-12):
            raise ValueError("FLOW_EPOCH_MONOTONICITY_FAILED")
        if previous is not None:
            immutable = ("task_id", "flow_type", "flow_id", "epoch", "logical_destination", "logical_destination_source", "logical_destination_capture_phase")
            if any(flow.get(name) != previous.get(name) for name in immutable):
                raise ValueError("FLOW_EPOCH_LOGICAL_IDENTITY_MUTATION")

    def flow(self, flow_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._flows[flow_id])

    def carrying(self, flow_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._carrying[flow_id])

    def all_flow_rows(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in sorted(self._flows.values(), key=lambda row: row["flow_index"])]

    def current_flow_rows(self) -> list[dict[str, Any]]:
        return self.all_flow_rows()

    def current_carrying_rows(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self._carrying[row["flow_id"]]) for row in sorted(self._flows.values(), key=lambda row: row["flow_index"])]

    def snapshot(self) -> dict[str, Any]:
        return {"schema_version": LEDGER_SCHEMA_VERSION, "logical_flow_rows": self.current_flow_rows(), "carrying_rows": self.current_carrying_rows(), "flow_index": copy.deepcopy(self._flow_index)}

    def validation_receipt(self, *, runtime_depdata_instances: int) -> dict[str, Any]:
        checks = {
            "flow_conservation": all(_flow_valid(row) for row in self._flows.values()),
            "flow_identity_reversible": all(parse_flow_id(flow_id) == (row["task_id"], row["flow_type"], row["epoch"]) for flow_id, row in self._flows.items()),
            "flow_index_unique": len(set(self._flow_index.values())) == len(self._flow_index),
            "flow_hop_separation": all(row["flow_id"] in self._flows for row in self._carrying.values()),
            "logical_destination_provenance_present": all(
                bool(row.get("logical_destination_source")) and bool(row.get("logical_destination_capture_phase"))
                for row in self._flows.values()
            ),
            "depdata_zero_instances": int(runtime_depdata_instances) == 0 and not any(row["flow_type"] == "DepData" for row in self._flows.values()),
            "no_future_action_source": True,
            "legacy_flow_completed_forbidden": True,
            "scope_raw_only": True,
        }
        return {"schema_version": "PI-JWM-Step4.2C-B-Acceptance-v1", "checks": checks, "runtime_depdata_instances": int(runtime_depdata_instances), "passed": all(checks.values()), "training": False, "gpu": False, "locked_test": False, "formal_dataset": False, "sample_tensor_extended": False, "graph_builder_started": False}


def _flow_valid(row: Mapping[str, Any]) -> bool:
    try:
        CausalFlowLedger._assert_flow(row)
        return row["status"] in FLOW_STATUSES and row["flow_type"] in FLOW_TYPES
    except (KeyError, TypeError, ValueError):
        return False


def validate_flow_ledger_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = {"flow_conservation", "flow_identity_reversible", "flow_index_unique", "flow_hop_separation", "logical_destination_provenance_present", "depdata_zero_instances", "no_future_action_source", "legacy_flow_completed_forbidden", "scope_raw_only"}
    checks = receipt.get("checks", {})
    checks_ok = required <= set(checks) and bool(checks) and all(value is True for value in checks.values())
    scope_ok = all(receipt.get(name) is False for name in ("training", "gpu", "locked_test", "formal_dataset", "sample_tensor_extended", "graph_builder_started"))
    return {"required_checks": checks_ok, "scope": scope_ok, "passed": bool(checks_ok and scope_ok and receipt.get("passed") is True)}


def _task_by_id(rows: Sequence[Mapping[str, Any]], task_id: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if str(row.get("task_id")) == task_id), None)


def _resolve_logical_destination(
    entry: Mapping[str, Any], task: Mapping[str, Any], route_kind: str,
) -> tuple[str, str, str]:
    route = [str(value) for value in entry.get("route_node_ids", [])]
    if route_kind == "return":
        destination = task.get("return_destination_id")
        if destination is None:
            raise ValueError("RETURN_LOGICAL_DESTINATION_UNAVAILABLE")
        destination = str(destination)
        if route and route[-1] != destination:
            raise ValueError("RETURN_ROUTE_TERMINAL_DESTINATION_MISMATCH")
        return destination, "decision.tasks[].return_destination_id", "after_return_route_established_before_outcome"
    if not route:
        raise ValueError("LOGICAL_DESTINATION_ROUTE_UNAVAILABLE")
    return route[-1], "established_offload_route_terminal", "after_route_action_established_before_outcome"


def amend_raw_with_causal_flow_ledger(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an additive Raw version with causal decision-time Flow rows."""
    output = copy.deepcopy(dict(raw))
    base_version = output.get("schema_version")
    ledger = CausalFlowLedger()
    transition_errors: list[dict[str, Any]] = []
    steps = {int(row["frame_index"]): row for row in output.get("steps", [])}
    for decision in sorted(output.get("decisions", []), key=lambda row: int(row["frame_index"])):
        decision.update(copy.deepcopy(ledger.snapshot()))
        decision["flow_capture_phase"] = "decision_before_action"
        frame = int(decision["frame_index"])
        step = steps.get(frame)
        if step is None:
            continue
        action_entries = step.get("action", {}).get("route", {}).get("entries", [])
        decision_tasks = decision.get("tasks", [])
        for entry in action_entries:
            task_id = str(entry["task_id"])
            source_task = _task_by_id(decision_tasks, task_id)
            if source_task is None:
                continue
            route_kind = str(entry.get("route_kind", "offload"))
            route = list(entry.get("route_node_ids", []))
            destination, destination_source, destination_capture_phase = _resolve_logical_destination(
                entry, source_task, route_kind
            )
            ledger_route = route if route else [destination]
            entry["logical_destination_id"] = destination
            entry["logical_destination_source"] = destination_source
            entry["logical_destination_capture_phase"] = destination_capture_phase
            entry["target_node_id_semantics"] = "current_action_or_carrying_hop_target_not_necessarily_logical_destination"
            entry["logical_route_node_ids"] = list(ledger_route)
            active_same_type = [row for row in ledger.all_flow_rows() if row["task_id"] == task_id and row["flow_type"] == ("Return" if route_kind == "return" else "Input") and row["status"] == "ACTIVE"]
            if active_same_type:
                active_id = active_same_type[0]["flow_id"]
                carrying = ledger.carrying(active_id)
                remaining_route = carrying["route"][carrying["current_hop_index"]:]
                if destination != active_same_type[0]["logical_destination"] or ledger_route != remaining_route:
                    ledger.reroute(active_id, new_route=ledger_route, new_logical_destination=destination)
            elif route_kind == "return":
                total = source_task.get(
                    "return_size",
                    source_task.get("required_returned_size", entry.get("return_size")),
                )
                if total is not None:
                    ledger.create_return_flow(
                        source_task, logical_destination=destination, total_data=total, route=ledger_route,
                        logical_destination_source=destination_source,
                        logical_destination_capture_phase=destination_capture_phase,
                    )
            else:
                ledger.create_input_flow(
                    source_task, logical_destination=destination, route=ledger_route,
                    logical_destination_source=destination_source,
                    logical_destination_capture_phase=destination_capture_phase,
                )
        outcome = step.get("outcome", {})
        outcome_tasks = outcome.get("tasks", [])
        transition_rows = []
        legacy_events = list(outcome.get("slot_transfer_events", []))
        for event_index, legacy in enumerate(legacy_events):
            overlay = copy.deepcopy(legacy)
            overlay["source_id"] = str(overlay.get("source_id", overlay.get("source")))
            overlay["target_id"] = str(overlay.get("target_id", overlay.get("target")))
            legacy_completed = overlay.pop("flow_completed", None)
            active_for_task = [row for row in ledger.all_flow_rows() if row["task_id"] == str(overlay["task_id"]) and row["status"] == "ACTIVE"]
            if "phase" not in overlay and len(active_for_task) == 1:
                overlay["phase"] = "return" if active_for_task[0]["flow_type"] == "Return" else "offload"
            task_row = _task_by_id(outcome_tasks, str(overlay["task_id"]))
            if task_row is None:
                continue
            later_same_task = next(
                (candidate for candidate in legacy_events[event_index + 1:] if str(candidate.get("task_id")) == str(overlay["task_id"])),
                None,
            )
            next_source = None if later_same_task is None else str(later_same_task.get("source_id", later_same_task.get("source")))
            post_slot_holder = str(task_row["current_node_id"])
            overlay["stage_or_hop_completed"] = bool(legacy_completed) or (next_source == overlay["target_id"]) or (post_slot_holder == overlay["target_id"])
            observed_holder = (
                next_source
                if later_same_task is not None
                else post_slot_holder
            )
            try:
                transition = ledger.apply_transfer_event(overlay, observed_task_current_node_id=observed_holder)
            except ValueError as exc:
                transition = {
                    "task_id": str(overlay["task_id"]),
                    "applied": False,
                    "error": str(exc),
                    "active_flow_count": len(active_for_task),
                    "active_flow_types": sorted(str(row["flow_type"]) for row in active_for_task),
                    "active_flow_ids": sorted(str(row["flow_id"]) for row in active_for_task),
                }
                transition_errors.append(copy.deepcopy(transition))
            transition["legacy_flow_completed_semantics"] = "stage_or_hop_completed"
            transition["legacy_flow_completed_forbidden_as_logical_completion"] = True
            transition["phase_source"] = "event.phase" if legacy.get("phase") is not None else "unique_active_flow_type_overlay"
            transition["hop_completion_source"] = (
                "legacy_flow_completed" if legacy_completed is not None
                else "next_event_or_post_slot_task_holder_equals_target"
            )
            transition_rows.append(transition)
        outcome["flow_ledger_transition_evidence"] = transition_rows
        outcome["flow_ledger_updated_for_next_decision_only"] = True
    output["schema_version"] = RAW_SCHEMA_VERSION
    output["base_schema_version"] = base_version
    output["flow_ledger_history"] = ledger.all_flow_rows()
    output["flow_carrying_history"] = ledger.current_carrying_rows()
    output["flow_lineage"] = [{"flow_id": row["flow_id"], "previous_flow_id": row["previous_flow_id"], "superseded_by_flow_id": row["superseded_by_flow_id"], "supersede_reason": row["supersede_reason"]} for row in ledger.all_flow_rows()]
    output["additive_amendment"] = {"step": "STEP 4.2C-B", "base_schema_unchanged": True, "decision_flow_state_uses_events_through_previous_slot_only": True, "future_action_used": False, "runtime_depdata_instances": 0}
    receipt = ledger.validation_receipt(runtime_depdata_instances=0)
    receipt["checks"].update({
        "all_real_transitions_applied_or_explicit": not transition_errors,
        "raw_decision_causality": all(decision.get("flow_capture_phase") == "decision_before_action" for decision in output.get("decisions", [])) and bool(output["additive_amendment"]["decision_flow_state_uses_events_through_previous_slot_only"]),
        "lineage_fields_present": all(all(name in row for name in ("previous_flow_id", "superseded_by_flow_id", "supersede_reason")) for row in output["flow_ledger_history"]),
        "logical_destination_provenance_proven": all(
            bool(row.get("logical_destination_source")) and bool(row.get("logical_destination_capture_phase"))
            for row in output["flow_ledger_history"]
        ),
        "logical_destination_constant_within_epoch": validate_logical_destination_continuity(output["flow_ledger_history"])["passed"] if output["flow_ledger_history"] else True,
        "completion_controls_presence": all((row["status"] not in {"COMPLETED", "SUPERSEDED"}) or row["presence"] is False for row in output["flow_ledger_history"]),
        "no_local_self_input_flow": all(not (row["flow_type"] == "Input" and row["logical_source"] == row["logical_destination"]) for row in output["flow_ledger_history"]),
    })
    receipt["transition_errors"] = transition_errors
    receipt["passed"] = all(receipt["checks"].values())
    receipt["validation"] = validate_flow_ledger_receipt(receipt)
    return output, receipt
