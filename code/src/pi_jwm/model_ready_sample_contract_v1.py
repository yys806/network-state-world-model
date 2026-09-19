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


SCHEMA_VERSION = "PI-JWM-Model-Ready-Sample-Contract-v2-step3.1R"
ACTION_FAMILIES = ("route", "comm", "comp", "mobility")


@dataclass(frozen=True)
class TensorContract:
    schema_version: str = SCHEMA_VERSION
    history_steps: int = 2
    horizon_steps: int = 2
    action_families: tuple[str, ...] = ACTION_FAMILIES
    input_index_policy: str = "decision_visible_objects_only"
    target_new_object_policy: str = "target_only_index_never_reused_as_input_index"
    dag_source_policy: str = "decision_time_raw_observer_dag_edges_only"

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


def _required_index(index: Mapping[str, int], object_id: str, *, family: str, field: str) -> int:
    if object_id not in index:
        prefix = "RESEARCHER_DECISION_REQUIRED: unresolved future action reference" if family in ACTION_FAMILIES else "unresolved contract reference"
        raise ValueError(
            f"{prefix} family={family} field={field} id={object_id}"
        )
    return int(index[object_id])


def _action_tensor(action: Mapping[str, Any], family: str, index: Mapping[str, int], node_index: Mapping[str, int]) -> dict[str, Any]:
    record = dict(action[family])
    entries = []
    for row in record.get("entries", []):
        row = dict(row)
        if family in ("route", "comm", "comp"):
            task_id = str(row.get("task_id"))
            row["task_index"] = _required_index(index, task_id, family=family, field="task_id")
        if family == "route":
            target_node_id = str(row.get("target_node_id"))
            row["target_node_index"] = _required_index(node_index, target_node_id, family=family, field="target_node_id")
            if row.get("task_node_id") is not None:
                task_node_id = str(row["task_node_id"])
                row["task_node_index"] = _required_index(node_index, task_node_id, family=family, field="task_node_id")
            row["route_node_indices"] = [
                _required_index(node_index, str(node_id), family=family, field="route_node_ids")
                for node_id in row.get("route_node_ids", [])
            ]
        if family == "comp" and row.get("node_id") is not None:
            node_id = str(row["node_id"])
            row["node_index"] = _required_index(node_index, node_id, family=family, field="node_id")
        if family == "mobility":
            uav_id = str(row.get("uav_id"))
            row["uav_index"] = _required_index(node_index, uav_id, family=family, field="uav_id")
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


def _flow_rows(outcome: Mapping[str, Any], task_index: Mapping[str, int], node_index: Mapping[str, int], flow_index: Mapping[str, int]) -> list[dict[str, Any]]:
    rows = []
    for event in outcome.get("slot_transfer_events", []):
        event = dict(event)
        task_id = str(event["task_id"])
        source = str(event.get("source"))
        target = str(event.get("target"))
        transport = str(event.get("transport", "wireless"))
        flow_id = f"flow::{task_id}::{transport}::{source}->{target}"
        rows.append({
            "flow_id": flow_id,
            "flow_index": _required_index(flow_index, flow_id, family="target_flow", field="flow_id"),
            "task_id": task_id,
            "task_index": _required_index(task_index, task_id, family="target_flow", field="task_id"),
            "source_node_id": source,
            "source_node_index": _required_index(node_index, source, family="target_flow", field="source"),
            "target_node_id": target,
            "target_node_index": _required_index(node_index, target, family="target_flow", field="target"),
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
    if anchor_step < contract.history_steps - 1 or anchor_step + contract.horizon_steps > len(steps):
        raise ValueError("anchor does not fit one continuous trajectory")
    if len(decisions) <= anchor_step:
        raise ValueError("raw artifact lacks the anchor decision")

    history_decisions = decisions[anchor_step - contract.history_steps + 1 : anchor_step + 1]
    future_steps = steps[anchor_step : anchor_step + contract.horizon_steps]
    trajectory_ids = {str(row["trajectory_id"]) for row in decisions}
    if len(trajectory_ids) != 1:
        raise ValueError("trajectory id changed inside sample")
    times = [float(row["simulation_time_s"]) for row in decisions]
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("decision times are not strictly increasing")
    if any(str(step["action"].get("vehicle_motion")) != "SUMO external" for step in future_steps):
        raise ValueError("vehicle motion must remain external to planner action")

    anchor_decision = decisions[anchor_step]
    input_entities = list(anchor_decision.get("entities", []))
    input_nodes = _ids(input_entities, "entity_id")
    input_tasks = _ids(list(anchor_decision.get("tasks", [])), "task_id")
    node_index = {value: i for i, value in enumerate(input_nodes)}
    task_index = {value: i for i, value in enumerate(input_tasks)}

    target_physical: dict[str, dict[str, Any]] = {}
    target_tasks: dict[str, dict[str, Any]] = {}
    target_flows: dict[str, dict[str, Any]] = {}
    for step in future_steps:
        for row in step["outcome"].get("entities", []):
            target_physical.setdefault(str(row["entity_id"]), dict(row))
        for row in step["outcome"].get("tasks", []):
            target_tasks.setdefault(str(row["task_id"]), dict(row))
        for event in step["outcome"].get("slot_transfer_events", []):
            event = dict(event)
            task_id = str(event["task_id"])
            source = str(event.get("source"))
            target = str(event.get("target"))
            transport = str(event.get("transport", "wireless"))
            flow_id = f"flow::{task_id}::{transport}::{source}->{target}"
            target_flows.setdefault(flow_id, {"flow_id": flow_id})
    target_only = {
        "physical": sorted(set(target_physical) - set(input_nodes)),
        "task": sorted(set(target_tasks) - set(input_tasks)),
        "flow": sorted(target_flows),
    }
    target_index = {
        "physical": {value: i for i, value in enumerate(sorted(target_physical))},
        "task": {value: i for i, value in enumerate(sorted(target_tasks))},
        "flow": {value: i for i, value in enumerate(sorted(target_flows))},
    }

    history = []
    for decision in history_decisions:
        entity_rows = {str(row["entity_id"]): row for row in decision.get("entities", [])}
        task_rows = {str(row["task_id"]): row for row in decision.get("tasks", [])}
        history_entities = []
        for entity_id in input_nodes:
            row = entity_rows.get(entity_id)
            present = row is not None
            row = row or {}
            history_entities.append({
                "entity_index": node_index[entity_id], "entity_id": entity_id, "presence": present,
                "feature_mask": {"speed_mps": present and row.get("speed_mps") is not None,
                                  "canonical_acceleration_mps2": present and bool(row.get("canonical_acceleration_observed_mask", False))},
                "speed_mps": _presence_mask(row.get("speed_mps"), present),
                "raw_simulator_acceleration_mps2": {**_presence_mask(row.get("raw_simulator_acceleration_mps2"), present), "role": "metadata_audit"},
                "canonical_acceleration_mps2": _presence_mask(row.get("canonical_acceleration_mps2"), present),
            })
        history_tasks = []
        for task_id in input_tasks:
            row = task_rows.get(task_id)
            present = row is not None
            row = row or {}
            history_tasks.append({"task_index": task_index[task_id], "task_id": task_id, "presence": present,
                                  "lifecycle": row.get("lifecycle") if present else None,
                                  "feature_mask": {"task_size": present and row.get("task_size") is not None},
                                  "task_size": _presence_mask(row.get("task_size"), present)})
        history.append({"frame_index": decision["frame_index"], "simulation_time_s": decision["simulation_time_s"],
                        "entities": history_entities, "tasks": history_tasks})

    future_actions = []
    targets = []
    for step in future_steps:
        future_actions.append({family: _action_tensor(step["action"], family, task_index, node_index) for family in ACTION_FAMILIES})
        outcome = step["outcome"]
        targets.append({
            "frame_index": step["frame_index"],
            "simulation_time_s": outcome["simulation_time_s"],
            "entities": [{"target_index": target_index["physical"][str(row["entity_id"])], "entity_id": str(row["entity_id"]), "presence": True,
                           "speed_mps": _presence_mask(row.get("speed_mps"), True)} for row in outcome.get("entities", [])],
            "tasks": [{"target_index": target_index["task"][str(row["task_id"])], "task_id": str(row["task_id"]), "presence": True,
                       "lifecycle": row.get("lifecycle"), "transmitted_size": _presence_mask(row.get("transmitted_size"), True)}
                      for row in outcome.get("tasks", [])],
            "flows": _flow_rows(outcome, target_index["task"], target_index["physical"], target_index["flow"]),
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
            "input_entity_index": {"physical": node_index, "task": task_index, "flow": {}},
            "target_index": target_index,
            "target_only_objects": target_only,
            "relation_endpoints": [
                {"relation_id": str(row.get("physical_edge_id")),
                 "source_entity_index": _required_index(node_index, str(row.get("source_id")), family="relation", field="source_id"),
                 "target_entity_index": _required_index(node_index, str(row.get("target_id")), family="relation", field="target_id"),
                 "validity_mask": True}
                for row in anchor_decision.get("channel_rows", [])
            ],
            "dag_relations": {"rows": [
                {"dag_edge_id": str(row["dag_edge_id"]),
                 "source_task_index": _required_index(task_index, str(row["source_task_id"]), family="dag", field="source_task_id"),
                 "target_task_index": _required_index(task_index, str(row["target_task_id"]), family="dag", field="target_task_id"),
                 "validity_mask": True}
                for row in anchor_decision.get("dag_edges", [])
                if str(row["source_task_id"]) in task_index and str(row["target_task_id"]) in task_index
            ], "observed_mask": True, "missing_reason": None,
            "source": "airfogsim_full_dual_graph_observer_v1._extract_dag_edges",
            "raw_edge_count": len(anchor_decision.get("dag_edges", [])),
            "input_visible_edge_count": sum(1 for row in anchor_decision.get("dag_edges", []) if str(row["source_task_id"]) in task_index and str(row["target_task_id"]) in task_index),
            "future_filtered_edge_count": sum(1 for row in anchor_decision.get("dag_edges", []) if str(row["source_task_id"]) not in task_index or str(row["target_task_id"]) not in task_index),
            "internal_future_edge_count": len(anchor_decision.get("internal_metadata", {}).get("future_dag_edges", []))},
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
    anchor = int(sample["metadata"]["anchor_decision_frame"])
    history_frames = [int(row["frame_index"]) for row in history]
    action_frames = [int(row) for row in sample["metadata"]["future_action_frame_indices"]]
    target_only = sample["static"]["target_only_objects"]
    input_index = sample["static"]["input_entity_index"]
    target_index = sample["static"]["target_index"]
    all_action_indices = [value for action in actions for family in ACTION_FAMILIES
                          for entry in action[family]["entries"]
                          for key, value in entry.items() if key.endswith("_index")]
    all_action_index_lists = [value for action in actions for family in ACTION_FAMILIES
                              for entry in action[family]["entries"]
                              for key, values in entry.items() if key.endswith("_indices") and key != "rb_indices"
                              for value in values]
    checks = {
        "schema_present": sample.get("schema_version") == SCHEMA_VERSION,
        "history_length": len(history) == int(contract["history_steps"]),
        "action_target_alignment": len(actions) == len(targets) == int(contract["horizon_steps"]),
        "four_action_families": all(set(row) == set(ACTION_FAMILIES) for row in actions),
        "history_ends_at_anchor": history_frames[-1] == anchor,
        "future_action_starts_at_anchor": action_frames[0] == anchor,
        "history_and_action_windows_contiguous": history_frames == list(range(anchor - len(history) + 1, anchor + 1)) and action_frames == list(range(anchor, anchor + len(actions))),
        "no_future_action_in_history": all("action" not in row for row in history),
        "target_only_not_in_input_index": all(not (set(values) & set(input_index.get(namespace, {}))) for namespace, values in target_only.items()),
        "target_namespaces_separate": set(target_index) == {"physical", "task", "flow"} and set(input_index) == {"physical", "task", "flow"},
        "history_fixed_index_rows": all(len(row["entities"]) == len(input_index["physical"]) and len(row["tasks"]) == len(input_index["task"]) for row in history),
        "no_unresolved_action_index": all(value is not None and int(value) >= 0 for value in [*all_action_indices, *all_action_index_lists]),
        "relation_endpoints_valid": bool(sample["static"]["relation_endpoints"]) and all(row["validity_mask"] for row in sample["static"]["relation_endpoints"]),
        "missing_distinct_from_empty": all(row[family]["missing"] != row[family]["empty"] or not row[family]["missing"] for row in actions for family in ACTION_FAMILIES),
        "flow_hop_semantics_explicit": all(flow["progress_semantics"].startswith("transport_hop") for target in targets for flow in target["flows"]),
        "dag_source_observed": sample["static"]["dag_relations"]["observed_mask"] is True and sample["static"]["dag_relations"]["source"].endswith("_extract_dag_edges"),
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
