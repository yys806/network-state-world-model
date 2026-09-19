"""Leak-safe model-ready sample contract for PI-JWM Step 3.1.

This module consumes the frozen Step 2 raw JSON shape.  It intentionally keeps
the representation JSON-native so the contract can be audited without a model
runtime.  It does not infer a DAG when the raw observer did not provide one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "PI-JWM-Model-Ready-Sample-Contract-v1"
ACTION_FAMILIES = ("route", "comm", "comp", "mobility")


@dataclass(frozen=True)
class TensorContract:
    schema_version: str = SCHEMA_VERSION
    history_steps: int = 2
    horizon_steps: int = 2
    action_families: tuple[str, ...] = ACTION_FAMILIES
    input_index_policy: str = "decision_visible_objects_only"
    target_new_object_policy: str = "target_only_index_never_reused_as_input_index"
    dag_source_policy: str = "missing_is_explicit_until_raw_dependency_observer_exists"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "action_families": list(self.action_families),
            "input_index_policy": self.input_index_policy,
            "target_new_object_policy": self.target_new_object_policy,
            "dag_source_policy": self.dag_source_policy,
        }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("raw artifact must be a JSON object")
    return value


def _ids(rows: list[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) is not None})


def _presence_mask(value: Any, present: bool) -> dict[str, Any]:
    if not present:
        return {"value": None, "presence": False, "feature_mask": False}
    if value is None:
        return {"value": None, "presence": True, "feature_mask": False}
    return {"value": value, "presence": True, "feature_mask": True}


def _action_tensor(action: Mapping[str, Any], family: str, index: Mapping[str, int], node_index: Mapping[str, int]) -> dict[str, Any]:
    record = dict(action[family])
    entries = []
    for row in record.get("entries", []):
        row = dict(row)
        if family in ("route", "comm", "comp"):
            task_id = str(row.get("task_id"))
            row["task_index"] = index.get(task_id, -1)
        if family == "route":
            row["target_node_index"] = node_index.get(str(row.get("target_node_id")), -1)
        if family == "mobility":
            row["uav_index"] = node_index.get(str(row.get("uav_id")), -1)
        entries.append(row)
    missing = not bool(record.get("field_present", False))
    empty = bool(record.get("empty", False)) and not missing
    return {
        "field_present": not missing,
        "empty": empty,
        "missing": missing,
        "missing_reason": "RAW_ACTION_FIELD_MISSING" if missing else None,
        "no_op_reason": record.get("no_op_reason") if empty else None,
        "entries": entries,
    }


def _flow_rows(outcome: Mapping[str, Any], task_index: Mapping[str, int], node_index: Mapping[str, int]) -> list[dict[str, Any]]:
    rows = []
    for event in outcome.get("slot_transfer_events", []):
        event = dict(event)
        task_id = str(event["task_id"])
        source = str(event.get("source"))
        target = str(event.get("target"))
        transport = str(event.get("transport", "wireless"))
        rows.append({
            "flow_id": f"flow::{task_id}::{transport}::{source}->{target}",
            "task_id": task_id,
            "task_index": task_index.get(task_id, -1),
            "source_node_id": source,
            "source_node_index": node_index.get(source, -1),
            "target_node_id": target,
            "target_node_index": node_index.get(target, -1),
            "transport": transport,
            "service_volume": event.get("delivered_data"),
            "progress_semantics": "transport_hop_service_not_end_to_end_task_progress",
        })
    return rows


def build_sample(raw: Mapping[str, Any], *, anchor_step: int = 2, contract: TensorContract = TensorContract()) -> dict[str, Any]:
    decisions = list(raw.get("decisions", []))
    steps = list(raw.get("steps", []))
    if contract.history_steps != 2 or contract.horizon_steps != 2:
        raise ValueError("Step 3.1 minimum artifact is fixed at H=2, L=2")
    if anchor_step < contract.history_steps or anchor_step + contract.horizon_steps > len(steps):
        raise ValueError("anchor does not fit one continuous trajectory")
    if len(decisions) <= anchor_step:
        raise ValueError("raw artifact lacks the anchor decision")

    history_decisions = decisions[anchor_step - contract.history_steps : anchor_step]
    future_steps = steps[anchor_step : anchor_step + contract.horizon_steps]
    trajectory_ids = {str(row["trajectory_id"]) for row in decisions}
    if len(trajectory_ids) != 1:
        raise ValueError("trajectory id changed inside sample")
    times = [float(row["simulation_time_s"]) for row in decisions]
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("decision times are not strictly increasing")
    if any(str(step["action"].get("vehicle_motion")) != "SUMO external" for step in future_steps):
        raise ValueError("vehicle motion must remain external to planner action")

    input_entities = list(history_decisions[-1].get("entities", []))
    input_nodes = _ids(input_entities, "entity_id")
    input_tasks = _ids(list(history_decisions[-1].get("tasks", [])), "task_id")
    node_index = {value: i for i, value in enumerate(input_nodes)}
    task_index = {value: i for i, value in enumerate(input_tasks)}

    target_objects: dict[str, dict[str, Any]] = {}
    for step in future_steps:
        for row in step["outcome"].get("entities", []):
            target_objects.setdefault(str(row["entity_id"]), dict(row))
        for row in step["outcome"].get("tasks", []):
            target_objects.setdefault(str(row["task_id"]), dict(row))
    target_only = sorted(set(target_objects) - set(input_nodes) - set(input_tasks))
    target_index = {value: i for i, value in enumerate(sorted(target_objects))}

    history = []
    for decision in history_decisions:
        history.append({
            "frame_index": decision["frame_index"],
            "simulation_time_s": decision["simulation_time_s"],
            "entities": [
                {"entity_index": node_index[str(row["entity_id"])], "entity_id": str(row["entity_id"]),
                 "presence": True, "feature_mask": {"speed_mps": row.get("speed_mps") is not None,
                 "canonical_acceleration_mps2": bool(row.get("canonical_acceleration_observed_mask", False))},
                 "speed_mps": _presence_mask(row.get("speed_mps"), True),
                 "raw_simulator_acceleration_mps2": {**_presence_mask(row.get("raw_simulator_acceleration_mps2"), row.get("raw_simulator_acceleration_mps2") is not None), "role": "metadata_audit"},
                 "canonical_acceleration_mps2": _presence_mask(row.get("canonical_acceleration_mps2"), bool(row.get("canonical_acceleration_observed_mask", False)))}
                for row in decision.get("entities", []) if str(row["entity_id"]) in node_index
            ],
            "tasks": [{"task_index": task_index[str(row["task_id"])], "task_id": str(row["task_id"]), "presence": True,
                       "lifecycle": row.get("lifecycle"), "feature_mask": {"task_size": row.get("task_size") is not None}}
                      for row in decision.get("tasks", []) if str(row["task_id"]) in task_index],
        })

    future_actions = []
    targets = []
    for step in future_steps:
        future_actions.append({family: _action_tensor(step["action"], family, task_index, node_index) for family in ACTION_FAMILIES})
        outcome = step["outcome"]
        targets.append({
            "frame_index": step["frame_index"],
            "simulation_time_s": outcome["simulation_time_s"],
            "entities": [{"target_index": target_index[str(row["entity_id"])], "entity_id": str(row["entity_id"]), "presence": True,
                           "speed_mps": _presence_mask(row.get("speed_mps"), True)} for row in outcome.get("entities", [])],
            "tasks": [{"target_index": target_index[str(row["task_id"])], "task_id": str(row["task_id"]), "presence": True,
                       "lifecycle": row.get("lifecycle"), "transmitted_size": _presence_mask(row.get("transmitted_size"), True)}
                      for row in outcome.get("tasks", [])],
            "flows": _flow_rows(outcome, task_index, node_index),
            "communication_service": {
                "wireless_delivered_data_by_task": outcome.get("wireless_delivered_data_by_task"),
                "wired_delivered_data_by_task": outcome.get("wired_delivered_data_by_task"),
                "delivered_data_by_task": outcome.get("delivered_data_by_task"),
                "observed_mask": outcome.get("delivered_data_observed_mask"),
                "missing_reason": outcome.get("delivered_data_missing_reason"),
            },
            "communication_service_semantics": "wireless_and_wired_hop_service_are_cumulative_transport_volume; not_end_to_end_payload_progress",
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": contract.to_dict(),
        "history": history,
        "static": {
            "input_entity_index": {"physical": node_index, "task": task_index},
            "target_entity_index": target_index,
            "target_only_objects": target_only,
            "relation_endpoints": [],
            "dag_relations": {"rows": [], "observed_mask": False, "missing_reason": "RAW_DAG_DEPENDENCY_SOURCE_NOT_CAPTURED"},
            "flow_contract": "flow_id is per task/transport/hop; source and destination come from real transfer event",
        },
        "future_action": future_actions,
        "target": targets,
        "metadata": {
            "trajectory_id": next(iter(trajectory_ids)),
            "anchor_decision_frame": anchor_step,
            "history_frame_indices": [row["frame_index"] for row in history_decisions],
            "future_action_frame_indices": [row["frame_index"] for row in future_steps],
            "decision_times_s": [row["simulation_time_s"] for row in history_decisions],
            "split": "non_locked_smoke",
            "seed": raw.get("environment", {}).get("seed"),
            "metadata_is_model_input": False,
            "raw_simulator_acceleration_role": "audit_only",
            "canonical_acceleration_role": "candidate_model_condition",
            "cpu_missing_policy": "null_plus_mask_reason; never zero_fill",
            "future_task_schedule_role": "internal_raw_metadata_only",
            "normalization_policy": "fit_train_trajectory_only_on_valid_masked_values",
        },
    }


def validate_sample(sample: Mapping[str, Any]) -> dict[str, bool]:
    contract = sample["contract"]
    history = list(sample["history"])
    actions = list(sample["future_action"])
    targets = list(sample["target"])
    input_ids = set(sample["static"]["input_entity_index"]["physical"]) | set(sample["static"]["input_entity_index"]["task"])
    target_only = set(sample["static"]["target_only_objects"])
    checks = {
        "schema_present": sample.get("schema_version") == SCHEMA_VERSION,
        "history_length": len(history) == int(contract["history_steps"]),
        "action_target_alignment": len(actions) == len(targets) == int(contract["horizon_steps"]),
        "four_action_families": all(set(row) == set(ACTION_FAMILIES) for row in actions),
        "no_future_action_in_history": all("action" not in row for row in history),
        "target_only_not_in_input_index": not (target_only & input_ids),
        "missing_distinct_from_empty": all(row[family]["missing"] != row[family]["empty"] or not row[family]["missing"] for row in actions for family in ACTION_FAMILIES),
        "flow_hop_semantics_explicit": all(flow["progress_semantics"].startswith("transport_hop") for target in targets for flow in target["flows"]),
        "dag_gap_explicit": sample["static"]["dag_relations"]["observed_mask"] is False,
        "metadata_not_model_input": sample["metadata"]["metadata_is_model_input"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"model-ready sample validation failed: {checks}")
    return checks


def write_sample(sample: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_sample(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["SCHEMA_VERSION", "TensorContract", "build_sample", "load_sample", "validate_sample", "write_sample"]
