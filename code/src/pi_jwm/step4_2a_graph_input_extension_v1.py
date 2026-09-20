"""Step 4.2A existing-source graph-input additive extension.

This module extends the frozen Step 3 contracts without replacing their
artifacts.  It materializes only facts already present in the frozen Raw data
or causally derivable at the decision boundary.  It deliberately does not
construct either graph or a stateful Flow representation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .model_ready_sample_contract_v1 import (
    SCHEMA_VERSION as BASE_SAMPLE_SCHEMA_VERSION,
    build_sample as build_base_sample,
)
from .step3_2_batch_preprocessing_v1 import RawSource, _check_raw
from .step3_3_model_input_tensor_v1 import (
    build_tensor_batch as build_base_tensor_batch,
    validate_tensor_batch_checks as validate_base_tensor_checks,
)


RAW_SCHEMA_VERSION = "PI-JWM-Raw-Graph-Input-Additive-Amendment-v1-step4.2A"
SAMPLE_SCHEMA_VERSION = "PI-JWM-Model-Ready-Sample-Contract-v5-step4.2A"
DATASET_SCHEMA_VERSION = "PI-JWM-Step-4.2A-Dataset-Preprocessing-v1"
NORMALIZATION_SCHEMA_VERSION = "PI-JWM-Step-4.2A-Train-Only-Normalization-v1"
TENSOR_SCHEMA_VERSION = "PI-JWM-Model-Input-Tensor-Collation-v3-step4.2A"

COMM_RELATION_TYPE_VOCAB = ("<PAD>", "unknown", "wireless", "wired")
TASK_AGENT_RELATION_TYPE_VOCAB = ("<PAD>", "unknown", "src", "host", "exec", "ret")
TASK_HISTORY_FEATURE_ORDER = (
    "task_cpu_work",
    "computed_cpu_work",
    "transmitted_size",
    "elapsed_time_s",
)

NORMALIZATION_SPECS = {
    "entity.position_x_m": ("m", "entities[].position_m[0]"),
    "entity.position_y_m": ("m", "entities[].position_m[1]"),
    "entity.position_z_m": ("m", "entities[].position_m[2]"),
    "comm.channel_attenuation_db": ("dB", "channel_rows[].channel_attenuation_db"),
    "agent.cpu_capacity_per_s": ("AirFogSim CPU-work-unit/s", "node_cpu_capacity_observation_rows[].capacity_per_s"),
    "task.task_cpu_work": ("AirFogSim CPU-work-unit", "tasks[].task_cpu_work"),
    "task.computed_cpu_work": ("AirFogSim CPU-work-unit", "tasks[].computed_cpu_work"),
    "task.transmitted_size": ("AirFogSim data-unit", "tasks[].transmitted_size"),
    "task.elapsed_time_s": ("s", "decision_time_s - tasks[].arrival_time_s"),
}

STILL_BLOCKED_FIELDS = (
    "flow.current_state.total_remaining_type_endpoints",
    "task.return_size",
    "task.priority",
    "task.deadline",
    "agent.dynamic_available_cpu",
    "agent.storage",
    "comm.wired_queue_load_utilization",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapped(value: Any, present: bool, *, unit: str) -> dict[str, Any]:
    valid = present and value is not None
    return {
        "value": copy.deepcopy(value) if valid else None,
        "presence": bool(present),
        "feature_mask": bool(valid),
        "unit": unit,
    }


def _vector_wrapped(value: Any, present: bool, *, unit: str, width: int) -> dict[str, Any]:
    values = list(value) if present and value is not None else [None] * width
    if len(values) != width:
        raise ValueError(f"expected vector width {width}, got {len(values)}")
    mask = [present and item is not None for item in values]
    return {
        "value": [float(item) if valid else None for item, valid in zip(values, mask)],
        "presence": bool(present),
        "feature_mask": mask,
        "unit": unit,
    }


def _wired_pairs(environment: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    pairs: set[tuple[str, str, int]] = set()
    for edge_index, edge in enumerate(environment.get("wired_edges", [])):
        source = str(edge["u"])
        target = str(edge["v"])
        pairs.add((source, target, edge_index))
        if bool(edge.get("bidirectional", False)):
            pairs.add((target, source, edge_index))
    return sorted(pairs)


def amend_raw_graph_inputs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new Raw version with decision-side wired relation rows.

    The recorded environment topology is pre-action state.  AirFogSim's
    WiredNetworkManager stores directed ``(source, target)`` keys and inserts
    the reverse key for a bidirectional configured edge; ``hasLink`` queries
    that directed-key set.  This amendment materializes the same semantics and
    records that provenance without reading execution outcomes.
    """

    output = copy.deepcopy(dict(raw))
    previous_version = output.get("schema_version")
    environment = dict(output.get("environment", {}))
    directed_pairs = _wired_pairs(environment)
    for decision in output.get("decisions", []):
        present = {str(row.get("entity_id")) for row in decision.get("entities", [])}
        rows = []
        for source, target, edge_index in directed_pairs:
            endpoint_visible = source in present and target in present
            rows.append({
                "communication_relation_id": f"comm::wired::{source}->{target}",
                "source_id": source,
                "target_id": target,
                "relation_type": "wired",
                "directed": True,
                "direction": f"{source}->{target}",
                "presence": endpoint_visible,
                "validity": endpoint_visible,
                "observed_mask": True,
                "missing_reason": None,
                "csi_value": None,
                "csi_feature_mask": False,
                "capture_phase": "decision_before_action",
                "source_provenance": {
                    "raw_source": f"environment.wired_edges[{edge_index}]",
                    "simulator_semantics": "WiredNetworkManager directed _links / hasLink(source,target)",
                    "outcome_used": False,
                },
            })
        decision["wired_relation_rows"] = rows
    output["schema_version"] = RAW_SCHEMA_VERSION
    output["base_schema_version"] = previous_version
    output["additive_amendment"] = {
        "step": "STEP 4.2A",
        "fields_added": ["decisions[].wired_relation_rows"],
        "old_fields_unchanged": True,
        "decision_time_only": True,
        "execution_outcome_used": False,
    }
    return output


def _communication_rows(decision: Mapping[str, Any], node_index: Mapping[str, int]) -> list[dict[str, Any]]:
    present = {str(row.get("entity_id")) for row in decision.get("entities", [])}
    rows: list[dict[str, Any]] = []
    for row in decision.get("channel_rows", []):
        source = str(row.get("source_id"))
        target = str(row.get("target_id"))
        if source not in node_index or target not in node_index:
            continue
        observed = bool(row.get("observed_mask", False))
        csi = list(row.get("channel_attenuation_db") or [])
        rb_indices = [int(value) for value in row.get("rb_indices", [])]
        if len(csi) != len(rb_indices):
            raise ValueError("wireless CSI and RB identity lengths differ")
        valid = source in present and target in present and observed
        rows.append({
            "communication_relation_id": f"comm::wireless::{row.get('physical_edge_id')}",
            "source_id": source,
            "source_agent_index": int(node_index[source]),
            "target_id": target,
            "target_agent_index": int(node_index[target]),
            "relation_type": "wireless",
            "wireless_channel_type": row.get("channel_type"),
            "presence": source in present and target in present,
            "validity": valid,
            "rb_indices": rb_indices,
            "rb_mask": [valid] * len(rb_indices),
            "csi_values": [float(value) for value in csi],
            "csi_mask": [valid] * len(csi),
            "csi_unit": "dB",
            "source_provenance": row.get("source_method", "channel_manager.getCSI"),
        })
    for row in decision.get("wired_relation_rows", []):
        source = str(row.get("source_id"))
        target = str(row.get("target_id"))
        if source not in node_index or target not in node_index:
            continue
        rows.append({
            "communication_relation_id": str(row.get("communication_relation_id")),
            "source_id": source,
            "source_agent_index": int(node_index[source]),
            "target_id": target,
            "target_agent_index": int(node_index[target]),
            "relation_type": "wired",
            "wireless_channel_type": None,
            "presence": bool(row.get("presence", False)),
            "validity": bool(row.get("validity", False)),
            "rb_indices": [],
            "rb_mask": [],
            "csi_values": None,
            "csi_mask": False,
            "csi_unit": "dB",
            "source_provenance": copy.deepcopy(row.get("source_provenance")),
        })
    return sorted(rows, key=lambda item: item["communication_relation_id"])


def _task_agent_rows(decision: Mapping[str, Any], node_index: Mapping[str, int], task_index: Mapping[str, int]) -> list[dict[str, Any]]:
    visible_nodes = {str(row.get("entity_id")) for row in decision.get("entities", [])}
    result: list[dict[str, Any]] = []
    for task in decision.get("tasks", []):
        task_id = str(task.get("task_id"))
        if task_id not in task_index:
            continue
        candidates = [
            ("src", task.get("task_node_id"), True),
            ("host", task.get("current_node_id"), True),
            ("exec", task.get("current_node_id"), str(task.get("lifecycle")) == "computing"),
            ("ret", task.get("return_destination_id"), task.get("return_destination_id") is not None),
        ]
        for relation_type, agent_id_raw, enabled in candidates:
            agent_id = str(agent_id_raw) if agent_id_raw is not None else ""
            if not enabled or agent_id not in node_index or agent_id not in visible_nodes:
                continue
            result.append({
                "task_agent_relation_id": f"task-agent::{relation_type}::{task_id}->{agent_id}",
                "task_id": task_id,
                "task_index": int(task_index[task_id]),
                "agent_id": agent_id,
                "agent_index": int(node_index[agent_id]),
                "relation_type": relation_type,
                "validity": True,
                "source_field": {
                    "src": "tasks[].task_node_id",
                    "host": "tasks[].current_node_id",
                    "exec": "tasks[].current_node_id conditioned on lifecycle=computing",
                    "ret": "tasks[].return_destination_id",
                }[relation_type],
                "future_route_action_used": False,
            })
    return sorted(result, key=lambda item: item["task_agent_relation_id"])


def _extend_target(sample: dict[str, Any], raw: Mapping[str, Any]) -> None:
    step_by_frame = {int(step["frame_index"]): step for step in raw.get("steps", [])}
    for target in sample.get("target", []):
        outcome = step_by_frame[int(target["frame_index"])]["outcome"]
        entities = {str(row["entity_id"]): row for row in outcome.get("entities", [])}
        tasks = {str(row["task_id"]): row for row in outcome.get("tasks", [])}
        for row in target.get("entities", []):
            source = entities.get(str(row["entity_id"]), {})
            row["position_m"] = _vector_wrapped(source.get("position_m"), True, unit="m", width=3)
        for row in target.get("tasks", []):
            source = tasks.get(str(row["task_id"]), {})
            for field, unit in (
                ("task_cpu_work", "AirFogSim CPU-work-unit"),
                ("computed_cpu_work", "AirFogSim CPU-work-unit"),
                ("transmitted_size", "AirFogSim data-unit"),
            ):
                row[field] = _wrapped(source.get(field), True, unit=unit)


def build_extended_sample(raw: Mapping[str, Any], *, anchor_step: int = 2) -> dict[str, Any]:
    if raw.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("Step 4.2A Raw amendment version required")
    sample = build_base_sample(raw, anchor_step=anchor_step)
    node_index = sample["static"]["input_entity_index"]["physical"]
    task_index = sample["static"]["input_entity_index"]["task"]
    decision_by_frame = {int(row["frame_index"]): row for row in raw.get("decisions", [])}
    history_decisions = [decision_by_frame[int(frame["frame_index"])] for frame in sample["history"]]

    for frame, decision in zip(sample["history"], history_decisions):
        entity_source = {str(row["entity_id"]): row for row in decision.get("entities", [])}
        task_source = {str(row["task_id"]): row for row in decision.get("tasks", [])}
        for row in frame.get("entities", []):
            source = entity_source.get(str(row["entity_id"]))
            row["position_m"] = _vector_wrapped(source.get("position_m") if source else None, source is not None, unit="m", width=3)
            row.setdefault("feature_mask", {})["position_m"] = list(row["position_m"]["feature_mask"])
        for row in frame.get("tasks", []):
            source = task_source.get(str(row["task_id"]))
            present = source is not None
            row["arrival_time_s"] = {
                **_wrapped(source.get("arrival_time_s") if source else None, present, unit="s"),
                "role": "causal_derivation_source_not_separate_tensor_feature",
            }
            for field, unit in (
                ("task_cpu_work", "AirFogSim CPU-work-unit"),
                ("computed_cpu_work", "AirFogSim CPU-work-unit"),
                ("transmitted_size", "AirFogSim data-unit"),
            ):
                row[field] = _wrapped(source.get(field) if source else None, present, unit=unit)
                row.setdefault("feature_mask", {})[field] = row[field]["feature_mask"]
            elapsed = float(decision["simulation_time_s"]) - float(source["arrival_time_s"]) if source else None
            if elapsed is not None and elapsed < -1e-9:
                raise ValueError("negative causal task elapsed time")
            row["elapsed_time_s"] = _wrapped(max(elapsed, 0.0) if elapsed is not None else None, present, unit="s")
            row["feature_mask"]["elapsed_time_s"] = row["elapsed_time_s"]["feature_mask"]
        frame["communication_relations"] = _communication_rows(decision, node_index)
        frame["task_agent_relations"] = _task_agent_rows(decision, node_index, task_index)
        frame["observation"]["communication_relations"] = frame["communication_relations"]
        frame["observation"]["task_agent_relations"] = frame["task_agent_relations"]

    cpu_by_node: dict[str, list[Mapping[str, Any]]] = {}
    for decision in history_decisions:
        for row in decision.get("node_cpu_capacity_observation_rows", []):
            cpu_by_node.setdefault(str(row.get("node_id")), []).append(row)
    static_capability = []
    for entity_id, entity_index in sorted(node_index.items(), key=lambda item: item[1]):
        candidates = cpu_by_node.get(entity_id, [])
        observed_values = {float(row["capacity_per_s"]) for row in candidates if row.get("observed_mask") and row.get("capacity_per_s") is not None}
        if len(observed_values) > 1:
            raise ValueError(f"static CPU capacity changed inside History: {entity_id}")
        value = next(iter(observed_values)) if observed_values else None
        static_capability.append({
            "entity_id": entity_id,
            "agent_index": int(entity_index),
            "cpu_capacity_per_s": {
                **_wrapped(value, True, unit="AirFogSim CPU-work-unit/s"),
                "observed_mask": value is not None,
                "missing_reason": None if value is not None else (candidates[-1].get("missing_reason") if candidates else "CPU_OBSERVATION_ROW_MISSING"),
                "source": "entity.getFogProfile()['cpu']",
                "semantic_role": "information_agent.static_capability",
            },
        })
    sample["static"]["agent_static_capability"] = static_capability
    _extend_target(sample, raw)
    sample["schema_version"] = SAMPLE_SCHEMA_VERSION
    sample["contract"]["schema_version"] = SAMPLE_SCHEMA_VERSION
    sample["contract"]["base_sample_schema_version"] = BASE_SAMPLE_SCHEMA_VERSION
    sample["contract"]["additive_graph_input_extension"] = True
    sample["metadata"]["raw_contract_version"] = RAW_SCHEMA_VERSION
    sample["metadata"]["normalization_policy"] = "trajectory split then train-only valid-value fit; validation reuse"
    checks = validate_extended_sample_checks(sample)
    if not checks["passed"]:
        raise ValueError(f"Step 4.2A sample validation failed: {checks}")
    return sample


def validate_extended_sample_checks(sample: Mapping[str, Any]) -> dict[str, bool]:
    history = list(sample.get("history", []))
    index = sample.get("static", {}).get("input_entity_index", {})
    node_index = index.get("physical", {})
    task_index = index.get("task", {})
    comm_rows = [row for frame in history for row in frame.get("communication_relations", [])]
    wired = [row for row in comm_rows if row.get("relation_type") == "wired"]
    task_agent = [row for frame in history for row in frame.get("task_agent_relations", [])]
    checks = {
        "schema_versions_upgraded": sample.get("schema_version") == SAMPLE_SCHEMA_VERSION and sample.get("metadata", {}).get("raw_contract_version") == RAW_SCHEMA_VERSION,
        "history_position_complete": bool(history) and all("position_m" in row and row["position_m"].get("unit") == "m" for frame in history for row in frame.get("entities", [])),
        "typed_comm_separate": bool(comm_rows) and all(row.get("relation_type") in {"wireless", "wired"} for row in comm_rows),
        "wired_valid_without_csi": bool(wired) and all(row.get("csi_values") is None and row.get("csi_mask") is False for row in wired),
        "comm_endpoint_namespace": all(row.get("source_id") in node_index and row.get("source_agent_index") == node_index[row["source_id"]] and row.get("target_id") in node_index and row.get("target_agent_index") == node_index[row["target_id"]] for row in comm_rows),
        "cpu_static_capability_only": len(sample.get("static", {}).get("agent_static_capability", [])) == len(node_index) and all(row["cpu_capacity_per_s"].get("semantic_role") == "information_agent.static_capability" for row in sample.get("static", {}).get("agent_static_capability", [])),
        "task_current_fields_complete": all("arrival_time_s" in row and all(field in row for field in TASK_HISTORY_FEATURE_ORDER) for frame in history for row in frame.get("tasks", [])),
        "elapsed_time_causal": all(row["elapsed_time_s"].get("value") is None or float(row["elapsed_time_s"]["value"]) >= 0 for frame in history for row in frame.get("tasks", [])),
        "task_agent_typed_and_indexed": bool(task_agent) and all(row.get("relation_type") in {"src", "host", "exec", "ret"} and row.get("task_index") == task_index[row["task_id"]] and row.get("agent_index") == node_index[row["agent_id"]] and row.get("future_route_action_used") is False for row in task_agent),
        "exec_lifecycle_conditioned": all(row.get("relation_type") != "exec" or row.get("source_field", "").endswith("lifecycle=computing") for row in task_agent),
        "stateful_flow_not_added": "stateful_flow" not in sample.get("static", {}) and all("current_stateful_flows" not in frame for frame in history),
        "physical_topology_not_built": all(key not in sample for key in ("physical_edges", "physical_graph", "graph_builder")),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def _iter_normalization_values(sample: Mapping[str, Any]):
    for frame in sample.get("history", []):
        for row in frame.get("entities", []):
            wrapped = row.get("position_m", {})
            values = wrapped.get("value") or [None, None, None]
            masks = wrapped.get("feature_mask") or [False, False, False]
            if row.get("presence", False):
                for axis, value, valid in zip(("x", "y", "z"), values, masks):
                    yield f"entity.position_{axis}_m", value, bool(valid) and value is not None
        for relation in frame.get("communication_relations", []):
            if relation.get("relation_type") != "wireless":
                continue
            values = relation.get("csi_values") or []
            masks = relation.get("csi_mask") if isinstance(relation.get("csi_mask"), list) else []
            for value, valid in zip(values, masks):
                yield "comm.channel_attenuation_db", value, bool(relation.get("presence")) and bool(valid) and value is not None
        for row in frame.get("tasks", []):
            if not row.get("presence", False):
                continue
            for field in TASK_HISTORY_FEATURE_ORDER:
                wrapped = row.get(field, {})
                yield f"task.{field}", wrapped.get("value"), bool(wrapped.get("feature_mask")) and wrapped.get("value") is not None
    for row in sample.get("static", {}).get("agent_static_capability", []):
        wrapped = row.get("cpu_capacity_per_s", {})
        yield "agent.cpu_capacity_per_s", wrapped.get("value"), bool(wrapped.get("observed_mask")) and bool(wrapped.get("feature_mask")) and wrapped.get("value") is not None


def fit_extension_normalization_stats(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {key: [] for key in NORMALIZATION_SPECS}
    for sample in samples:
        if sample.get("metadata", {}).get("split") != "dev_train":
            continue
        for key, value, valid in _iter_normalization_values(sample):
            if valid:
                values[key].append(float(value))
    features: dict[str, dict[str, Any]] = {}
    for key, (unit, source) in NORMALIZATION_SPECS.items():
        observed = values[key]
        count = len(observed)
        mean = sum(observed) / count if count else 0.0
        variance = sum((value - mean) ** 2 for value in observed) / count if count else 0.0
        features[key] = {
            "count": count,
            "mean": mean,
            "std": variance ** 0.5 if variance > 1e-12 else 1.0,
            "zero_variance_handling": "scale=1.0" if variance <= 1e-12 else "population_std",
            "source_field": source,
            "mask_policy": "dev_train only AND presence/validity=true AND feature_mask=true AND value!=null; validation excluded",
            "unit": unit,
            "preprocessing_policy": "z_score_on_valid_train_values",
        }
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "source_split": "dev_train",
        "features": features,
    }


def _normalize(value: Any, feature: Mapping[str, Any], valid: bool) -> float | None:
    if not valid or value is None:
        return None
    return (float(value) - float(feature["mean"])) / float(feature["std"])


def apply_extension_normalization(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(samples))
    features = stats.get("features", {})
    for sample in output:
        for frame in sample.get("history", []):
            for row in frame.get("entities", []):
                wrapped = row["position_m"]
                values = wrapped.get("value") or [None, None, None]
                masks = wrapped.get("feature_mask") or [False, False, False]
                wrapped["normalized_value"] = [
                    _normalize(value, features[f"entity.position_{axis}_m"], bool(row.get("presence")) and bool(valid))
                    for axis, value, valid in zip(("x", "y", "z"), values, masks)
                ]
                wrapped["normalization_valid"] = [value is not None for value in wrapped["normalized_value"]]
            for relation in frame.get("communication_relations", []):
                if relation.get("relation_type") == "wireless":
                    masks = relation.get("csi_mask") if isinstance(relation.get("csi_mask"), list) else []
                    relation["csi_normalized_values"] = [
                        _normalize(value, features["comm.channel_attenuation_db"], bool(relation.get("presence")) and bool(valid))
                        for value, valid in zip(relation.get("csi_values") or [], masks)
                    ]
                else:
                    relation["csi_normalized_values"] = None
            for row in frame.get("tasks", []):
                for field in TASK_HISTORY_FEATURE_ORDER:
                    wrapped = row[field]
                    wrapped["normalized_value"] = _normalize(
                        wrapped.get("value"),
                        features[f"task.{field}"],
                        bool(row.get("presence")) and bool(wrapped.get("feature_mask")),
                    )
                    wrapped["normalization_valid"] = wrapped["normalized_value"] is not None
        for row in sample.get("static", {}).get("agent_static_capability", []):
            wrapped = row["cpu_capacity_per_s"]
            wrapped["normalized_value"] = _normalize(
                wrapped.get("value"),
                features["agent.cpu_capacity_per_s"],
                bool(wrapped.get("observed_mask")) and bool(wrapped.get("feature_mask")),
            )
            wrapped["normalization_valid"] = wrapped["normalized_value"] is not None
    return output


def _relation_capacity(samples: Sequence[Mapping[str, Any]], field: str) -> int:
    return max((len(frame.get(field, [])) for sample in samples for frame in sample.get("history", [])), default=0)


def _n_rb(samples: Sequence[Mapping[str, Any]]) -> int:
    indices = [
        int(rb)
        for sample in samples
        for frame in sample.get("history", [])
        for relation in frame.get("communication_relations", [])
        for rb in relation.get("rb_indices", [])
    ]
    return max(indices, default=-1) + 1


def _require_index(index: Mapping[str, int], object_id: Any, numeric: Any, label: str) -> int:
    key = str(object_id)
    if key not in index or int(index[key]) != int(numeric):
        raise ValueError(f"{label} ID/index mismatch: {key} != {numeric}")
    return int(numeric)


def build_extended_tensor_batch(samples: Sequence[Mapping[str, Any]], *, stats: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(samples)
    if not samples:
        raise ValueError("cannot tensorize empty Step 4.2A samples")
    if any(sample.get("schema_version") != SAMPLE_SCHEMA_VERSION for sample in samples):
        raise ValueError("Step 4.2A sample schema mismatch")

    base_samples = copy.deepcopy(samples)
    for sample in base_samples:
        sample["schema_version"] = BASE_SAMPLE_SCHEMA_VERSION
        sample["contract"]["schema_version"] = BASE_SAMPLE_SCHEMA_VERSION
    base = build_base_tensor_batch(base_samples)
    base_checks = validate_base_tensor_checks(base)
    if not base_checks["passed"]:
        raise ValueError(f"base Step 3.3 regression failed during extension: {base_checks}")

    output = {key: value for key, value in base.items() if key != "contract"}
    base_contract = base["contract"].to_dict()
    batch = len(samples)
    history_steps = len(samples[0]["history"])
    max_entity = int(base_contract["max_entity"])
    max_task = int(base_contract["max_task"])
    max_comm = _relation_capacity(samples, "communication_relations")
    max_task_agent = _relation_capacity(samples, "task_agent_relations")
    n_rb = _n_rb(samples)
    contract = {
        **base_contract,
        "schema_version": TENSOR_SCHEMA_VERSION,
        "sample_contract_version": SAMPLE_SCHEMA_VERSION,
        "dataset_contract_version": DATASET_SCHEMA_VERSION,
        "raw_contract_version": RAW_SCHEMA_VERSION,
        "max_comm_relation": max_comm,
        "max_task_agent_relation": max_task_agent,
        "n_comm_rb": n_rb,
        "entity_position_feature_order": ["x", "y", "z"],
        "task_history_feature_order": list(TASK_HISTORY_FEATURE_ORDER),
        "comm_relation_type_vocab": list(COMM_RELATION_TYPE_VOCAB),
        "task_agent_relation_type_vocab": list(TASK_AGENT_RELATION_TYPE_VOCAB),
        "cpu_capacity_semantic_role": "information_agent.static_capability",
        "physical_topology_policy": "NOT_BUILT_RESEARCHER_DECISION_REQUIRED",
        "stateful_flow_policy": "RAW_INSUFFICIENT_NOT_ADDED",
    }
    output["schema_version"] = TENSOR_SCHEMA_VERSION
    output["contract"] = contract
    output["base_step3_3_validation_checks"] = base_checks

    zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    indices = lambda shape: np.full(shape, -1, dtype=np.int64)
    output.update({
        "entity_position_raw": zeros((batch, history_steps, max_entity, 3)),
        "entity_position": zeros((batch, history_steps, max_entity, 3)),
        "entity_position_mask": zeros((batch, history_steps, max_entity, 3), bool),
        "comm_source_index": indices((batch, history_steps, max_comm)),
        "comm_target_index": indices((batch, history_steps, max_comm)),
        "comm_relation_type_index": zeros((batch, history_steps, max_comm), np.int64),
        "comm_relation_presence": zeros((batch, history_steps, max_comm), bool),
        "comm_relation_validity": zeros((batch, history_steps, max_comm), bool),
        "comm_csi_raw": zeros((batch, history_steps, max_comm, n_rb)),
        "comm_csi": zeros((batch, history_steps, max_comm, n_rb)),
        "comm_csi_mask": zeros((batch, history_steps, max_comm, n_rb), bool),
        "comm_rb_indices": indices((batch, history_steps, max_comm, n_rb)),
        "comm_rb_mask": zeros((batch, history_steps, max_comm, n_rb), bool),
        "agent_cpu_capacity_raw": zeros((batch, max_entity)),
        "agent_cpu_capacity": zeros((batch, max_entity)),
        "agent_cpu_capacity_mask": zeros((batch, max_entity), bool),
        "task_history_extended_raw_features": zeros((batch, history_steps, max_task, len(TASK_HISTORY_FEATURE_ORDER))),
        "task_history_extended_features": zeros((batch, history_steps, max_task, len(TASK_HISTORY_FEATURE_ORDER))),
        "task_history_extended_feature_mask": zeros((batch, history_steps, max_task, len(TASK_HISTORY_FEATURE_ORDER)), bool),
        "task_agent_task_index": indices((batch, history_steps, max_task_agent)),
        "task_agent_agent_index": indices((batch, history_steps, max_task_agent)),
        "task_agent_relation_type_index": zeros((batch, history_steps, max_task_agent), np.int64),
        "task_agent_validity_mask": zeros((batch, history_steps, max_task_agent), bool),
    })

    for batch_index, sample in enumerate(samples):
        namespace = sample["static"]["input_entity_index"]
        node_index = {str(key): int(value) for key, value in namespace["physical"].items()}
        task_index = {str(key): int(value) for key, value in namespace["task"].items()}
        for row in sample["static"]["agent_static_capability"]:
            slot = _require_index(node_index, row["entity_id"], row["agent_index"], "static CPU")
            wrapped = row["cpu_capacity_per_s"]
            valid = bool(wrapped.get("observed_mask")) and wrapped.get("value") is not None
            if valid:
                output["agent_cpu_capacity_raw"][batch_index, slot] = float(wrapped["value"])
                output["agent_cpu_capacity"][batch_index, slot] = float(wrapped.get("normalized_value", wrapped["value"]))
                output["agent_cpu_capacity_mask"][batch_index, slot] = True
        for history_index, frame in enumerate(sample["history"]):
            for row in frame.get("entities", []):
                slot = _require_index(node_index, row["entity_id"], row["entity_index"], "position")
                wrapped = row["position_m"]
                raw_values = wrapped.get("value") or [None, None, None]
                norm_values = wrapped.get("normalized_value", raw_values)
                masks = wrapped.get("feature_mask") or [False, False, False]
                for axis, (raw_value, norm_value, valid) in enumerate(zip(raw_values, norm_values, masks)):
                    if valid and raw_value is not None and norm_value is not None:
                        output["entity_position_raw"][batch_index, history_index, slot, axis] = float(raw_value)
                        output["entity_position"][batch_index, history_index, slot, axis] = float(norm_value)
                        output["entity_position_mask"][batch_index, history_index, slot, axis] = True
            for row in frame.get("tasks", []):
                slot = _require_index(task_index, row["task_id"], row["task_index"], "task history")
                for feature_index, field in enumerate(TASK_HISTORY_FEATURE_ORDER):
                    wrapped = row[field]
                    if row.get("presence") and wrapped.get("feature_mask") and wrapped.get("value") is not None:
                        output["task_history_extended_raw_features"][batch_index, history_index, slot, feature_index] = float(wrapped["value"])
                        output["task_history_extended_features"][batch_index, history_index, slot, feature_index] = float(wrapped.get("normalized_value", wrapped["value"]))
                        output["task_history_extended_feature_mask"][batch_index, history_index, slot, feature_index] = True
            for relation_index, row in enumerate(frame.get("communication_relations", [])):
                source = _require_index(node_index, row["source_id"], row["source_agent_index"], "comm source")
                target = _require_index(node_index, row["target_id"], row["target_agent_index"], "comm target")
                output["comm_source_index"][batch_index, history_index, relation_index] = source
                output["comm_target_index"][batch_index, history_index, relation_index] = target
                output["comm_relation_type_index"][batch_index, history_index, relation_index] = COMM_RELATION_TYPE_VOCAB.index(row["relation_type"])
                output["comm_relation_presence"][batch_index, history_index, relation_index] = bool(row.get("presence"))
                output["comm_relation_validity"][batch_index, history_index, relation_index] = bool(row.get("validity"))
                if row["relation_type"] == "wireless":
                    normalized = row.get("csi_normalized_values", row.get("csi_values", []))
                    for rb, raw_value, norm_value, valid in zip(row.get("rb_indices", []), row.get("csi_values", []), normalized, row.get("csi_mask", [])):
                        rb = int(rb)
                        if rb < 0 or rb >= n_rb:
                            raise ValueError("communication RB index exceeds tensor capacity")
                        output["comm_rb_indices"][batch_index, history_index, relation_index, rb] = rb
                        output["comm_rb_mask"][batch_index, history_index, relation_index, rb] = bool(valid)
                        if valid and norm_value is not None:
                            output["comm_csi_raw"][batch_index, history_index, relation_index, rb] = float(raw_value)
                            output["comm_csi"][batch_index, history_index, relation_index, rb] = float(norm_value)
                            output["comm_csi_mask"][batch_index, history_index, relation_index, rb] = True
            for relation_index, row in enumerate(frame.get("task_agent_relations", [])):
                task_slot = _require_index(task_index, row["task_id"], row["task_index"], "task-agent task")
                agent_slot = _require_index(node_index, row["agent_id"], row["agent_index"], "task-agent agent")
                output["task_agent_task_index"][batch_index, history_index, relation_index] = task_slot
                output["task_agent_agent_index"][batch_index, history_index, relation_index] = agent_slot
                output["task_agent_relation_type_index"][batch_index, history_index, relation_index] = TASK_AGENT_RELATION_TYPE_VOCAB.index(row["relation_type"])
                output["task_agent_validity_mask"][batch_index, history_index, relation_index] = bool(row.get("validity"))
    checks = validate_extended_tensor_checks(output)
    if not checks["passed"]:
        raise ValueError(f"Step 4.2A tensor validation failed: {checks}")
    return output


def validate_extended_tensor_checks(tensor: Mapping[str, Any]) -> dict[str, bool]:
    contract = dict(tensor.get("contract", {}))
    entity_position = tensor.get("entity_position")
    comm_valid = tensor.get("comm_relation_validity")
    comm_types = tensor.get("comm_relation_type_index")
    comm_csi_mask = tensor.get("comm_csi_mask")
    max_entity = int(contract.get("max_entity", -1))
    max_task = int(contract.get("max_task", -1))
    wired_code = COMM_RELATION_TYPE_VOCAB.index("wired")
    active_comm = np.asarray(comm_valid, dtype=bool)
    wired = np.asarray(comm_types) == wired_code
    active_task_agent = np.asarray(tensor.get("task_agent_validity_mask"), dtype=bool)
    comm_source = np.asarray(tensor.get("comm_source_index"))
    comm_target = np.asarray(tensor.get("comm_target_index"))
    task_agent_task = np.asarray(tensor.get("task_agent_task_index"))
    task_agent_agent = np.asarray(tensor.get("task_agent_agent_index"))
    checks = {
        "schema_versions_upgraded": tensor.get("schema_version") == TENSOR_SCHEMA_VERSION and contract.get("sample_contract_version") == SAMPLE_SCHEMA_VERSION and contract.get("raw_contract_version") == RAW_SCHEMA_VERSION,
        "entity_position_shape": isinstance(entity_position, np.ndarray) and entity_position.shape[-1] == 3 and tensor["entity_position_mask"].shape == entity_position.shape,
        "typed_comm_tensor_separate": all(name in tensor for name in ("comm_source_index", "comm_target_index", "comm_relation_type_index", "comm_relation_presence", "comm_relation_validity", "comm_csi", "comm_csi_mask")),
        "wired_valid_without_csi": bool(np.any(wired & active_comm)) and not bool(np.any(np.asarray(comm_csi_mask)[wired])),
        "comm_endpoints_in_entity_namespace": bool(np.all((comm_source[active_comm] >= 0) & (comm_source[active_comm] < max_entity)) and np.all((comm_target[active_comm] >= 0) & (comm_target[active_comm] < max_entity))),
        "cpu_static_has_no_history_axis": tensor["agent_cpu_capacity"].ndim == 2 and tensor["agent_cpu_capacity"].shape[1] == max_entity,
        "task_extended_shape": tensor["task_history_extended_features"].shape[-1] == len(TASK_HISTORY_FEATURE_ORDER),
        "task_agent_typed_relation": bool(np.any(active_task_agent)) and bool(np.all((task_agent_task[active_task_agent] >= 0) & (task_agent_task[active_task_agent] < max_task)) and np.all((task_agent_agent[active_task_agent] >= 0) & (task_agent_agent[active_task_agent] < max_entity))),
        "masked_placeholders_zero": bool(np.all(tensor["entity_position"][~tensor["entity_position_mask"]] == 0) and np.all(tensor["comm_csi"][~tensor["comm_csi_mask"]] == 0) and np.all(tensor["agent_cpu_capacity"][~tensor["agent_cpu_capacity_mask"]] == 0) and np.all(tensor["task_history_extended_features"][~tensor["task_history_extended_feature_mask"]] == 0)),
        "base_step3_3_semantics_preserved": bool(tensor.get("base_step3_3_validation_checks", {}).get("passed")),
        "no_physical_topology": contract.get("physical_topology_policy") == "NOT_BUILT_RESEARCHER_DECISION_REQUIRED" and not any(name.startswith("physical_edge") for name in tensor),
        "no_stateful_flow_added": contract.get("stateful_flow_policy") == "RAW_INSUFFICIENT_NOT_ADDED" and "current_stateful_flow" not in tensor,
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def save_extended_tensor_batch(tensor: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in tensor.items() if isinstance(value, np.ndarray)}
    json_values = {
        "contract_json": tensor["contract"],
        "sample_ids_json": tensor["sample_ids"],
        "sample_static_json": tensor["sample_static"],
        "sample_metadata_json": tensor["sample_metadata"],
        "base_step3_3_validation_checks_json": tensor["base_step3_3_validation_checks"],
    }
    for key, value in json_values.items():
        arrays[key] = np.asarray(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(path, **arrays)


def load_extended_tensor_batch(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        json_names = {
            "contract_json": "contract",
            "sample_ids_json": "sample_ids",
            "sample_static_json": "sample_static",
            "sample_metadata_json": "sample_metadata",
            "base_step3_3_validation_checks_json": "base_step3_3_validation_checks",
        }
        result = {
            target: json.loads(str(data[source]))
            for source, target in json_names.items()
        }
        result.update({key: data[key] for key in data.files if key not in json_names})
    result["schema_version"] = TENSOR_SCHEMA_VERSION
    return result


def build_extension_batch(sources: Sequence[RawSource], *, history_steps: int = 2, horizon_steps: int = 2) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one Raw source is required")
    samples: list[dict[str, Any]] = []
    raw_amendments: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen_trajectories: set[str] = set()
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    for source in sources:
        path = Path(source.path)
        original = json.loads(path.read_text(encoding="utf-8"))
        trajectory_id = _check_raw(original, source)
        if trajectory_id in seen_trajectories:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen_trajectories.add(trajectory_id)
        (train_ids if source.split == "dev_train" else validation_ids).add(trajectory_id)
        amended = amend_raw_graph_inputs(original)
        raw_amendments.append(amended)
        decisions = amended["decisions"]
        steps = amended["steps"]
        provenance.append({
            "trajectory_id": trajectory_id,
            "split": source.split,
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "source_raw_schema_version": original.get("schema_version"),
            "amended_raw_schema_version": RAW_SCHEMA_VERSION,
            "sample_schema_version": SAMPLE_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "tensor_schema_version": TENSOR_SCHEMA_VERSION,
            "seed": amended.get("environment", {}).get("seed"),
            "config_hash": amended.get("environment", {}).get("config_hash"),
            "decision_frame_range": [int(decisions[0]["frame_index"]), int(decisions[-1]["frame_index"])],
            "step_frame_range": [int(steps[0]["frame_index"]), int(steps[-1]["frame_index"])],
            "slot_duration_s": float(amended.get("environment", {})["slot_duration_s"]),
        })
        for anchor in range(history_steps - 1, len(steps) - horizon_steps + 1):
            sample = build_extended_sample(amended, anchor_step=anchor)
            sample["metadata"]["split"] = source.split
            sample["metadata"]["source_split"] = source.split
            sample["metadata"]["sample_id"] = f"{trajectory_id}::anchor-{anchor:04d}"
            samples.append(sample)
    if train_ids & validation_ids:
        raise ValueError("trajectory_id crosses train and validation")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "contract": {
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "sample_schema_version": SAMPLE_SCHEMA_VERSION,
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "split_policy": "trajectory_level_before_window_construction",
            "normalization_policy": "train_valid_values_only_validation_reuse",
        },
        "samples": samples,
        "raw_amendments": raw_amendments,
        "provenance": provenance,
        "fields_resolved": [
            "physical.position_m",
            "comm.wireless_csi",
            "comm.wired_relation",
            "agent.cpu_capacity_per_s",
            "task.demand_progress_time_existing_fields",
            "task_agent.typed_relations",
        ],
        "fields_still_blocked": list(STILL_BLOCKED_FIELDS),
        "step4_1_gap_resolution": {
            "historical_mapping_preserved": True,
            "resolved": [
                {"item": "physical.position_m", "step4_1_availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "current_availability": "AVAILABLE_IN_STEP_4.2A_SAMPLE_TENSOR"},
                {"item": "comm.wireless_csi", "step4_1_availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "current_availability": "AVAILABLE_IN_STEP_4.2A_SAMPLE_TENSOR"},
                {"item": "comm.wired_relation", "step4_1_availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "current_availability": "AVAILABLE_IN_STEP_4.2A_RAW_SAMPLE_TENSOR"},
                {"item": "agent.cpu_capacity_per_s", "step4_1_availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED", "current_availability": "AVAILABLE_IN_STEP_4.2A_STATIC_SAMPLE_TENSOR"},
                {"item": "task.existing_demand_progress_elapsed", "step4_1_availability": "RAW_AVAILABLE_BUT_NOT_EXPOSED_OR_DERIVABLE", "current_availability": "AVAILABLE_IN_STEP_4.2A_SAMPLE_TENSOR"},
                {"item": "task_agent.typed_relations", "step4_1_availability": "DERIVABLE_CAUSALLY", "current_availability": "AVAILABLE_IN_STEP_4.2A_SAMPLE_TENSOR"},
            ],
            "still_blocked": list(STILL_BLOCKED_FIELDS),
            "graph_ready": False,
        },
        "scope": {
            "graph_builder": False,
            "physical_topology": False,
            "training": False,
            "gpu": False,
            "locked_test": False,
            "formal_dataset": False,
        },
    }


def validate_extension_acceptance(
    bundle: Mapping[str, Any],
    stats: Mapping[str, Any],
    tensor: Mapping[str, Any],
    *,
    deterministic_rebuild: bool,
) -> dict[str, bool]:
    provenance = list(bundle.get("provenance", []))
    train = {row["trajectory_id"] for row in provenance if row["split"] == "dev_train"}
    validation = {row["trajectory_id"] for row in provenance if row["split"] == "dev_validation"}
    scope = bundle.get("scope", {})
    checks = {
        "versioned_additive_contracts": bundle.get("schema_version") == DATASET_SCHEMA_VERSION and stats.get("schema_version") == NORMALIZATION_SCHEMA_VERSION and tensor.get("schema_version") == TENSOR_SCHEMA_VERSION,
        "trajectory_level_split": bool(provenance) and not bool(train & validation),
        "raw_sample_tensor_provenance": all(len(row.get("source_sha256", "")) == 64 and row.get("amended_raw_schema_version") == RAW_SCHEMA_VERSION and row.get("sample_schema_version") == SAMPLE_SCHEMA_VERSION and row.get("tensor_schema_version") == TENSOR_SCHEMA_VERSION for row in provenance),
        "all_samples_valid": bool(bundle.get("samples")) and all(validate_extended_sample_checks(sample)["passed"] for sample in bundle.get("samples", [])),
        "train_only_normalization": stats.get("source_split") == "dev_train" and all("validation excluded" in row.get("mask_policy", "") for row in stats.get("features", {}).values()),
        "tensor_valid": validate_extended_tensor_checks(tensor)["passed"],
        "resolved_fields_explicit": set(bundle.get("fields_resolved", [])) == {"physical.position_m", "comm.wireless_csi", "comm.wired_relation", "agent.cpu_capacity_per_s", "task.demand_progress_time_existing_fields", "task_agent.typed_relations"},
        "raw_insufficient_fields_still_blocked": set(bundle.get("fields_still_blocked", [])) == set(STILL_BLOCKED_FIELDS),
        "deterministic_rebuild": bool(deterministic_rebuild),
        "scope_false": all(scope.get(key) is False for key in ("graph_builder", "physical_topology", "training", "gpu", "locked_test", "formal_dataset")),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


__all__ = [
    "COMM_RELATION_TYPE_VOCAB",
    "DATASET_SCHEMA_VERSION",
    "NORMALIZATION_SCHEMA_VERSION",
    "RAW_SCHEMA_VERSION",
    "SAMPLE_SCHEMA_VERSION",
    "STILL_BLOCKED_FIELDS",
    "TASK_AGENT_RELATION_TYPE_VOCAB",
    "TASK_HISTORY_FEATURE_ORDER",
    "TENSOR_SCHEMA_VERSION",
    "amend_raw_graph_inputs",
    "apply_extension_normalization",
    "build_extended_sample",
    "build_extended_tensor_batch",
    "build_extension_batch",
    "fit_extension_normalization_stats",
    "load_extended_tensor_batch",
    "save_extended_tensor_batch",
    "validate_extended_sample_checks",
    "validate_extended_tensor_checks",
    "validate_extension_acceptance",
]
