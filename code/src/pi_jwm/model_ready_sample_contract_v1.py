"""Leak-safe model-ready sample contract for PI-JWM Step 3.1F.

This module consumes the frozen Step 2 raw JSON shape.  It intentionally keeps
the representation JSON-native so the contract can be audited without a model
runtime.  It does not infer a DAG when the raw observer did not provide one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "PI-JWM-Model-Ready-Sample-Contract-v4-step3.3F"
ACTION_FAMILIES = ("route", "comm", "comp", "mobility")


@dataclass(frozen=True)
class TensorContract:
    schema_version: str = SCHEMA_VERSION
    history_steps: int = 2
    horizon_steps: int = 2
    action_families: tuple[str, ...] = ACTION_FAMILIES
    input_index_policy: str = "history_causal_observable_object_union"
    future_action_index_policy: str = "anchor_visibility_then_history_union_input_index"
    target_new_object_policy: str = "target_only_index_never_reused_as_input_index"
    dag_source_policy: str = "decision_time_raw_observer_dag_edges_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "action_families": list(self.action_families),
            "input_index_policy": self.input_index_policy,
            "future_action_index_policy": self.future_action_index_policy,
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


def _validated_index(index: Mapping[str, int], object_id: str, *, family: str, field: str,
                    visible_index: Mapping[str, int] | None = None) -> int:
    if visible_index is not None:
        _required_index(visible_index, object_id, family=family, field=field)
    return _required_index(index, object_id, family=family, field=field)


def _action_tensor(action: Mapping[str, Any], family: str, index: Mapping[str, int], node_index: Mapping[str, int],
                   *, visible_index: Mapping[str, int] | None = None,
                   visible_node_index: Mapping[str, int] | None = None) -> dict[str, Any]:
    record = dict(action[family])
    entries = []
    for row in record.get("entries", []):
        row = dict(row)
        if family in ("route", "comm", "comp"):
            task_id = str(row.get("task_id"))
            row["task_index"] = _validated_index(index, task_id, family=family, field="task_id", visible_index=visible_index)
        if family == "route":
            target_node_id = str(row.get("target_node_id"))
            row["target_node_index"] = _validated_index(node_index, target_node_id, family=family, field="target_node_id", visible_index=visible_node_index)
            if row.get("task_node_id") is not None:
                task_node_id = str(row["task_node_id"])
                row["task_node_index"] = _validated_index(node_index, task_node_id, family=family, field="task_node_id", visible_index=visible_node_index)
            row["route_node_indices"] = [
                _validated_index(node_index, str(node_id), family=family, field="route_node_ids", visible_index=visible_node_index)
                for node_id in row.get("route_node_ids", [])
            ]
        if family == "comp" and row.get("node_id") is not None:
            node_id = str(row["node_id"])
            row["node_index"] = _validated_index(node_index, node_id, family=family, field="node_id", visible_index=visible_node_index)
        if family == "mobility":
            uav_id = str(row.get("uav_id"))
            row["uav_index"] = _validated_index(node_index, uav_id, family=family, field="uav_id", visible_index=visible_node_index)
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


def _flow_id(event: Mapping[str, Any]) -> str:
    return f"flow::{event['task_id']}::{event.get('transport', 'wireless')}::{event.get('source')}->{event.get('target')}"


def _observation_entities(decision: Mapping[str, Any], input_nodes: list[str], node_index: Mapping[str, int]) -> list[dict[str, Any]]:
    rows = {str(row["entity_id"]): row for row in decision.get("entities", [])}
    result = []
    for entity_id in input_nodes:
        row = rows.get(entity_id)
        present = row is not None
        row = row or {}
        result.append({
            "entity_index": node_index[entity_id], "entity_id": entity_id, "entity_type": row.get("entity_type") if present else None, "presence": present,
            "feature_mask": {"speed_mps": present and row.get("speed_mps") is not None,
                              "canonical_acceleration_mps2": present and bool(row.get("canonical_acceleration_observed_mask", False))},
            "speed_mps": _presence_mask(row.get("speed_mps"), present),
            "raw_simulator_acceleration_mps2": {**_presence_mask(row.get("raw_simulator_acceleration_mps2"), present), "role": "metadata_audit"},
            "canonical_acceleration_mps2": _presence_mask(row.get("canonical_acceleration_mps2"), present),
        })
    return result


def _observation_tasks(decision: Mapping[str, Any], input_tasks: list[str], task_index: Mapping[str, int]) -> list[dict[str, Any]]:
    rows = {str(row["task_id"]): row for row in decision.get("tasks", [])}
    result = []
    for task_id in input_tasks:
        row = rows.get(task_id)
        present = row is not None
        row = row or {}
        result.append({
            "task_index": task_index[task_id], "task_id": task_id, "presence": present,
            "lifecycle": row.get("lifecycle") if present else None,
            "feature_mask": {"task_size": present and row.get("task_size") is not None},
            "task_size": _presence_mask(row.get("task_size"), present),
        })
    return result


def _outcome_rows(outcome: Mapping[str, Any], input_nodes: list[str], input_tasks: list[str],
                  node_index: Mapping[str, int], task_index: Mapping[str, int], flow_index: Mapping[str, int]) -> dict[str, Any]:
    entity_rows = {str(row["entity_id"]): row for row in outcome.get("entities", [])}
    task_rows = {str(row["task_id"]): row for row in outcome.get("tasks", [])}
    entities = []
    for entity_id in input_nodes:
        row = entity_rows.get(entity_id)
        present = row is not None
        row = row or {}
        entities.append({
            "entity_index": node_index[entity_id], "entity_id": entity_id, "entity_type": row.get("entity_type") if present else None, "presence": present,
            "feature_mask": {"speed_mps": present and row.get("speed_mps") is not None},
            "speed_mps": _presence_mask(row.get("speed_mps"), present),
        })
    tasks = []
    for task_id in input_tasks:
        row = task_rows.get(task_id)
        present = row is not None
        row = row or {}
        tasks.append({
            "task_index": task_index[task_id], "task_id": task_id, "presence": present,
            "lifecycle": row.get("lifecycle") if present else None,
            "feature_mask": {"transmitted_size": present and row.get("transmitted_size") is not None},
            "transmitted_size": _presence_mask(row.get("transmitted_size"), present),
        })
    return {
        "entities": entities,
        "tasks": tasks,
        "flows": _flow_rows(outcome, task_index, node_index, flow_index),
        "communication_service": {
            "wireless_delivered_data_by_task": outcome.get("wireless_delivered_data_by_task"),
            "wired_delivered_data_by_task": outcome.get("wired_delivered_data_by_task"),
            "delivered_data_by_task": outcome.get("delivered_data_by_task"),
            "observed_mask": outcome.get("delivered_data_observed_mask"),
            "missing_reason": outcome.get("delivered_data_missing_reason"),
        },
        "served_cpu_work_by_task": outcome.get("served_cpu_work_by_task"),
        "relation_endpoints": [
            {"relation_id": str(row.get("physical_edge_id")),
             "source_entity_index": _required_index(node_index, str(row.get("source_id")), family="history_relation", field="source_id"),
             "target_entity_index": _required_index(node_index, str(row.get("target_id")), family="history_relation", field="target_id"),
             "validity_mask": True}
            for row in outcome.get("channel_rows", [])
            if str(row.get("source_id")) in node_index and str(row.get("target_id")) in node_index
        ],
        "dag_relations": {
            "rows": [
                {"dag_edge_id": str(row["dag_edge_id"]),
                 "source_task_index": _required_index(task_index, str(row["source_task_id"]), family="history_dag", field="source_task_id"),
                 "target_task_index": _required_index(task_index, str(row["target_task_id"]), family="history_dag", field="target_task_id"),
                 "validity_mask": True}
                for row in outcome.get("dag_edges", [])
                if str(row["source_task_id"]) in task_index and str(row["target_task_id"]) in task_index
            ],
            "observed_mask": True,
            "source": "airfogsim_full_dual_graph_observer_v1._extract_dag_edges",
        },
        "communication_service_semantics": "wireless_and_wired_hop_service_are_cumulative_transport_volume; not_end_to_end_payload_progress",
    }


def build_sample(raw: Mapping[str, Any], *, anchor_step: int = 2, contract: TensorContract = TensorContract()) -> dict[str, Any]:
    decisions = list(raw.get("decisions", []))
    steps = list(raw.get("steps", []))
    if contract.history_steps <= 0 or contract.horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be positive")
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
    # Input identity is the causal union visible anywhere in O_{t-H+1:t}.
    history_decision_ids = {str(row["entity_id"]) for decision in history_decisions for row in decision.get("entities", [])}
    history_task_ids = {str(row["task_id"]) for decision in history_decisions for row in decision.get("tasks", [])}
    past_steps = steps[anchor_step - contract.history_steps + 1:anchor_step]
    past_outcomes = [step["outcome"] for step in past_steps]
    history_flow_ids = {_flow_id(event) for outcome in past_outcomes for event in outcome.get("slot_transfer_events", [])}
    input_nodes = sorted(history_decision_ids)
    input_tasks = sorted(history_task_ids)
    input_flows = sorted(history_flow_ids)
    node_index = {value: i for i, value in enumerate(input_nodes)}
    task_index = {value: i for i, value in enumerate(input_tasks)}
    flow_index = {value: i for i, value in enumerate(input_flows)}
    anchor_node_index = {value: i for i, value in enumerate(_ids(list(anchor_decision.get("entities", [])), "entity_id"))}
    anchor_task_index = {value: i for i, value in enumerate(_ids(list(anchor_decision.get("tasks", [])), "task_id"))}

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
        "flow": sorted(set(target_flows) - set(input_flows)),
    }
    target_index = {
        "physical": {value: i for i, value in enumerate(sorted(target_physical))},
        "task": {value: i for i, value in enumerate(sorted(target_tasks))},
        "flow": {value: i for i, value in enumerate(sorted(target_flows))},
    }

    history = []
    for history_offset, decision in enumerate(history_decisions):
        frame = int(decision["frame_index"])
        past = frame < anchor_step
        aligned_step = steps[frame] if past and frame < len(steps) else None
        observation = {
            "entities": _observation_entities(decision, input_nodes, node_index),
            "tasks": _observation_tasks(decision, input_tasks, task_index),
            "relation_endpoints": [
                {"relation_id": str(row.get("physical_edge_id")),
                 "source_entity_index": _required_index(node_index, str(row.get("source_id")), family="history_relation", field="source_id"),
                 "target_entity_index": _required_index(node_index, str(row.get("target_id")), family="history_relation", field="target_id"),
                 "validity_mask": True}
                for row in decision.get("channel_rows", [])
                if str(row.get("source_id")) in node_index and str(row.get("target_id")) in node_index
            ],
            "dag_relations": {
                "rows": [
                    {"dag_edge_id": str(row["dag_edge_id"]),
                     "source_task_index": _required_index(task_index, str(row["source_task_id"]), family="history_dag", field="source_task_id"),
                     "target_task_index": _required_index(task_index, str(row["target_task_id"]), family="history_dag", field="target_task_id"),
                     "validity_mask": True}
                    for row in decision.get("dag_edges", [])
                    if str(row["source_task_id"]) in task_index and str(row["target_task_id"]) in task_index
                ],
                "observed_mask": True,
                "source": "airfogsim_full_dual_graph_observer_v1._extract_dag_edges",
            },
        }
        row = {"frame_index": frame, "simulation_time_s": decision["simulation_time_s"], **observation,
               "observation": observation}
        if past:
            if aligned_step is None or int(aligned_step["frame_index"]) != frame:
                raise ValueError(f"history action/outcome is not aligned to observation frame {frame}")
            row["action"] = {family: _action_tensor(aligned_step["action"], family, task_index, node_index) for family in ACTION_FAMILIES}
            row["outcome"] = _outcome_rows(aligned_step["outcome"], input_nodes, input_tasks, node_index, task_index, flow_index)
            row["outcome"]["frame_index"] = int(aligned_step["outcome"]["frame_index"])
            row["outcome"]["simulation_time_s"] = aligned_step["outcome"]["simulation_time_s"]
        history.append(row)

    future_actions = []
    targets = []
    for step in future_steps:
        # Future action references are resolved against objects visible at O_t.
        # The History union index is broader only for preserving past identity.
        future_actions.append({family: _action_tensor(
            step["action"], family, task_index, node_index,
            visible_index=anchor_task_index, visible_node_index=anchor_node_index,
        ) for family in ACTION_FAMILIES})
        outcome = step["outcome"]
        targets.append({
            "frame_index": step["frame_index"],
            "simulation_time_s": outcome["simulation_time_s"],
            "entities": [{"target_index": target_index["physical"][str(row["entity_id"])], "entity_id": str(row["entity_id"]), "entity_type": row.get("entity_type"), "presence": True,
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
            "input_entity_index": {"physical": node_index, "task": task_index, "flow": flow_index},
            "input_entity_type_by_index": {str(index): next((row.get("entity_type") for decision in history_decisions for row in decision.get("entities", []) if str(row.get("entity_id")) == entity_id and row.get("entity_type") is not None), None) for entity_id, index in node_index.items()},
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
            "history_contract": "H_{t-H+1:t}=(O_{t-H+1:t},A_{t-H+1:t-1},Y_{t-H+1:t-1}); current A_t/Y_t excluded",
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
    entity_type_by_index = sample["static"].get("input_entity_type_by_index", {})
    all_action_indices = [value for action in actions for family in ACTION_FAMILIES
                          for entry in action[family]["entries"]
                          for key, value in entry.items() if key.endswith("_index")]
    all_action_index_lists = [value for action in actions for family in ACTION_FAMILIES
                              for entry in action[family]["entries"]
                              for key, values in entry.items() if key.endswith("_indices") and key != "rb_indices"
                              for value in values]
    history_has_current_action_or_outcome = bool(history[-1].get("action")) or bool(history[-1].get("outcome"))
    history_past_rows = history[:-1]
    history_action_frames = [int(row["frame_index"]) for row in history_past_rows if "action" in row]
    history_outcome_frames = [int(row["outcome"]["frame_index"]) for row in history_past_rows if "outcome" in row]
    history_reference_values = []
    future_action_indices_match_input_index = True
    for action in actions:
        for family in ACTION_FAMILIES:
            for entry in action[family]["entries"]:
                if family in ("route", "comm", "comp"):
                    task_id = str(entry.get("task_id"))
                    future_action_indices_match_input_index &= (
                        task_id in input_index["task"]
                        and entry.get("task_index") == input_index["task"][task_id]
                    )
                if family == "route":
                    target_node_id = str(entry.get("target_node_id"))
                    future_action_indices_match_input_index &= (
                        target_node_id in input_index["physical"]
                        and entry.get("target_node_index") == input_index["physical"][target_node_id]
                    )
                    if entry.get("task_node_id") is not None:
                        task_node_id = str(entry["task_node_id"])
                        future_action_indices_match_input_index &= (
                            task_node_id in input_index["physical"]
                            and entry.get("task_node_index") == input_index["physical"][task_node_id]
                        )
                    route_node_ids = [str(value) for value in entry.get("route_node_ids", [])]
                    route_node_indices = list(entry.get("route_node_indices", []))
                    future_action_indices_match_input_index &= len(route_node_ids) == len(route_node_indices)
                    future_action_indices_match_input_index &= all(
                        node_id in input_index["physical"] and numeric_index == input_index["physical"][node_id]
                        for node_id, numeric_index in zip(route_node_ids, route_node_indices)
                    )
                if family == "comp" and entry.get("node_id") is not None:
                    node_id = str(entry["node_id"])
                    future_action_indices_match_input_index &= (
                        node_id in input_index["physical"]
                        and entry.get("node_index") == input_index["physical"][node_id]
                    )
                if family == "mobility":
                    uav_id = str(entry.get("uav_id"))
                    future_action_indices_match_input_index &= (
                        uav_id in input_index["physical"]
                        and entry.get("uav_index") == input_index["physical"][uav_id]
                    )
    for row in history_past_rows:
        for family in ACTION_FAMILIES:
            for entry in row.get("action", {}).get(family, {}).get("entries", []):
                for key, value in entry.items():
                    if key.endswith("_index"):
                        history_reference_values.append(value)
                    if key.endswith("_indices") and key != "rb_indices":
                        history_reference_values.extend(value)
        for endpoint in row.get("outcome", {}).get("relation_endpoints", []):
            history_reference_values.extend([endpoint["source_entity_index"], endpoint["target_entity_index"]])
        for edge in row.get("outcome", {}).get("dag_relations", {}).get("rows", []):
            history_reference_values.extend([edge["source_task_index"], edge["target_task_index"]])
        for flow in row.get("outcome", {}).get("flows", []):
            history_reference_values.extend([flow["flow_index"], flow["task_index"], flow["source_node_index"], flow["target_node_index"]])
    checks = {
        "schema_present": sample.get("schema_version") == SCHEMA_VERSION,
        "history_length": len(history) == int(contract["history_steps"]),
        "action_target_alignment": len(actions) == len(targets) == int(contract["horizon_steps"]),
        "four_action_families": all(set(row) == set(ACTION_FAMILIES) for row in actions),
        "history_ends_at_anchor": history_frames[-1] == anchor,
        "future_action_starts_at_anchor": action_frames[0] == anchor,
        "history_and_action_windows_contiguous": history_frames == list(range(anchor - len(history) + 1, anchor + 1)) and action_frames == list(range(anchor, anchor + len(actions))),
        "history_past_action_outcome_present": all("action" in row and "outcome" in row for row in history_past_rows),
        "history_current_excludes_action_outcome": not history_has_current_action_or_outcome,
        "history_action_outcome_aligned": history_action_frames == history_outcome_frames == history_frames[:-1],
        "no_future_action_in_history": "action" not in history[-1] and "outcome" not in history[-1],
        "target_only_not_in_input_index": all(not (set(values) & set(input_index.get(namespace, {}))) for namespace, values in target_only.items()),
        "target_namespaces_separate": set(target_index) == {"physical", "task", "flow"} and set(input_index) == {"physical", "task", "flow"},
        "history_fixed_index_rows": all(len(row["entities"]) == len(input_index["physical"]) and len(row["tasks"]) == len(input_index["task"]) for row in history),
        "causal_entity_type_static_complete": set(entity_type_by_index) == {str(value) for value in input_index["physical"].values()} and all(value is not None for value in entity_type_by_index.values()),
        "history_entity_type_matches_static": all(row.get("entity_type") is None or row.get("entity_type") == entity_type_by_index.get(str(row["entity_index"])) for frame in history for row in frame["entities"]),
        "no_unresolved_action_index": all(value is not None and int(value) >= 0 for value in [*all_action_indices, *all_action_index_lists]),
        "future_action_indices_match_input_index": future_action_indices_match_input_index,
        "no_unresolved_history_reference_index": all(value is not None and int(value) >= 0 for value in history_reference_values),
        "history_relation_dag_flow_aligned": all(
            endpoint["source_entity_index"] in input_index["physical"].values()
            and endpoint["target_entity_index"] in input_index["physical"].values()
            for row in history_past_rows for endpoint in row.get("outcome", {}).get("relation_endpoints", [])
        ) and all(
            edge["source_task_index"] in input_index["task"].values()
            and edge["target_task_index"] in input_index["task"].values()
            for row in history_past_rows for edge in row.get("outcome", {}).get("dag_relations", {}).get("rows", [])
        ) and all(
            flow["flow_index"] in input_index["flow"].values()
            and flow["task_index"] in input_index["task"].values()
            and flow["source_node_index"] in input_index["physical"].values()
            and flow["target_node_index"] in input_index["physical"].values()
            for row in history_past_rows for flow in row.get("outcome", {}).get("flows", [])
        ),
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


def audit_future_action_references(raw: Mapping[str, Any], *, history_steps: int = 2, horizon_steps: int = 2) -> dict[str, Any]:
    """Count future actions that reference objects absent from anchor O_t.

    This is an observation-only audit. It does not drop windows or choose a
    representation for future arrivals.
    """
    decisions = list(raw.get("decisions", []))
    steps = list(raw.get("steps", []))
    max_anchor = min(len(steps), len(decisions)) - horizon_steps
    windows = 0
    affected = 0
    unresolved_references = 0
    by_family = {family: {"windows": 0, "references": 0} for family in ACTION_FAMILIES}
    by_object_kind = {
        "task": {"windows": 0, "references": 0},
        "physical_entity": {"windows": 0, "references": 0},
    }
    for anchor in range(history_steps - 1, max_anchor + 1):
        windows += 1
        visible_nodes = {str(row.get("entity_id")) for row in decisions[anchor].get("entities", [])}
        visible_tasks = {str(row.get("task_id")) for row in decisions[anchor].get("tasks", [])}
        window_families: set[str] = set()
        for step in steps[anchor + 1:anchor + horizon_steps]:
            for family in ACTION_FAMILIES:
                for row in step.get("action", {}).get(family, {}).get("entries", []):
                    refs: list[tuple[str, str]] = []
                    if family in ("route", "comm", "comp") and row.get("task_id") is not None:
                        refs.append(("task", str(row["task_id"])))
                    if family == "route":
                        refs.extend(("entity", str(row[key])) for key in ("target_node_id", "task_node_id") if row.get(key) is not None)
                        refs.extend(("entity", str(value)) for value in row.get("route_node_ids", []))
                    if family == "comp" and row.get("node_id") is not None:
                        refs.append(("entity", str(row["node_id"])))
                    if family == "mobility" and row.get("uav_id") is not None:
                        refs.append(("entity", str(row["uav_id"])))
                    for kind, object_id in refs:
                        visible = object_id in (visible_tasks if kind == "task" else visible_nodes)
                        if not visible:
                            unresolved_references += 1
                            by_family[family]["references"] += 1
                            window_families.add(family)
                            by_object_kind["task" if kind == "task" else "physical_entity"]["references"] += 1
                            window_families.add(f"object_kind:{'task' if kind == 'task' else 'physical_entity'}")
        if window_families:
            affected += 1
            for family in window_families:
                if family.startswith("object_kind:"):
                    by_object_kind[family.split(":", 1)[1]]["windows"] += 1
                else:
                    by_family[family]["windows"] += 1
    return {
        "window_policy": {"history_steps": history_steps, "horizon_steps": horizon_steps,
                           "future_offsets_audited": list(range(1, horizon_steps))},
        "constructible_window_count": windows,
        "affected_window_count": affected,
        "affected_window_rate": (affected / windows) if windows else None,
        "unresolved_future_reference_count": unresolved_references,
        "by_action_family": by_family,
        "by_object_kind": by_object_kind,
        "observation_only": True,
        "decision": "RESEARCHER_DECISION_REQUIRED_IF_NONZERO",
    }


__all__ = ["SCHEMA_VERSION", "TensorContract", "audit_future_action_references", "build_sample", "load_sample", "validate_sample", "write_sample"]
