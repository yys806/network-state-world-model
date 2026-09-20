"""STEP 4.2C-C causal logical-Flow Sample/Tensor additive extension.

This layer only indexes, aligns, masks, normalizes approved continuous fields,
and collates the frozen STEP 4.2C-B Raw Flow rows.  It never recomputes Flow
identity, destination, progress, holder, route revision, or completion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .step4_2a_graph_input_extension_v1 import (
    SAMPLE_SCHEMA_VERSION as STEP42A_SAMPLE_SCHEMA_VERSION,
    build_extended_tensor_batch,
)
from .step4_2c_b_causal_flow_ledger_raw_v1 import (
    RAW_SCHEMA_VERSION as STEP42CB_RAW_SCHEMA_VERSION,
    parse_flow_id,
)


SAMPLE_SCHEMA_VERSION = "PI-JWM-Model-Ready-Sample-Contract-v6-step4.2C-C"
NORMALIZATION_SCHEMA_VERSION = "PI-JWM-Step-4.2C-C-Train-Only-Flow-Normalization-v2-presence-aware"
TENSOR_SCHEMA_VERSION = "PI-JWM-Model-Input-Tensor-Collation-v5-step4.2C-C-PATCH"
STEP32_NORMALIZATION_MASK_POLICY = "presence=true AND feature_mask=true AND value!=null; train split only"
FLOW_TYPE_VOCAB = ("<PAD>", "unknown", "Input", "Return", "DepData")
FLOW_STATUS_VOCAB = ("<PAD>", "unknown", "ACTIVE", "COMPLETED", "SUPERSEDED")
FLOW_NUMERIC_FEATURE_ORDER = (
    "logical.total_data",
    "logical.e2e_delivered",
    "logical.e2e_remaining",
    "carrying.hop_progress",
    "carrying.hop_remaining",
)
FLOW_NUMERIC_UNITS = {name: "AirFogSim data-unit" for name in FLOW_NUMERIC_FEATURE_ORDER}
REQUIRED_ACCEPTANCE_CHECKS = frozenset({
    "raw_sample_history_logical_equality", "raw_sample_history_carrying_equality",
    "raw_sample_target_logical_equality", "raw_sample_target_carrying_equality",
    "raw_sample_semantic_equality", "sample_tensor_history_logical_equality",
    "sample_tensor_history_carrying_equality", "sample_tensor_target_logical_equality",
    "sample_tensor_target_carrying_equality", "sample_tensor_semantic_equality",
    "history_causal_flow_union", "stable_flow_identity",
    "target_only_future_flow_isolation", "epoch_isolation",
    "multi_hop_single_flow_tensor_identity", "presence_mask_correctness",
    "train_only_preprocessing", "presence_aware_normalization_policy",
    "no_silent_truncation", "input_return_categories", "depdata_runtime_zero",
    "deterministic_rebuild", "serialize_load", "scope",
    "future_epoch_target_identity", "target_known_presence_relation",
    "masked_placeholder_zero", "target_masked_placeholder_zero",
    "endpoint_indices_bounded", "task_indices_bounded", "target_endpoint_indices_bounded",
    "target_carrying_alignment", "target_route_mask", "target_namespace",
    "target_carrying_ground_truth", "route_node_mask",
})


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_payload(logical_rows: Sequence[Mapping[str, Any]], carrying_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    carrying = {str(row["flow_id"]): row for row in carrying_rows if row.get("known", True)}
    result = []
    for row in sorted((r for r in logical_rows if r.get("known", True)), key=lambda item: int(item["flow_index"])):
        flow_id = str(row["flow_id"])
        carry = carrying.get(flow_id, {})
        flow_mask = row.get("feature_mask", {})
        carry_mask = carry.get("feature_mask", {})
        result.append({
            "flow_id": flow_id, "flow_index": int(row["flow_index"]),
            "task_id": str(row["task_id"]), "flow_type": str(row["flow_type"]),
            "epoch": int(row["epoch"]), "logical_source": str(row["logical_source"]),
            "logical_destination": str(row["logical_destination"]),
            "total_data": float(row["total_data"]), "e2e_delivered": float(row["e2e_delivered"]),
            "e2e_remaining": float(row["e2e_remaining"]), "presence": bool(row["presence"]),
            "status": str(row["status"]),
            "logical_destination_source": str(row["logical_destination_source"]),
            "logical_destination_capture_phase": str(row["logical_destination_capture_phase"]),
            "logical_feature_mask": {field: bool(flow_mask.get(field)) for field in ("total_data", "e2e_delivered", "e2e_remaining")},
            "carrying_missing": not bool(carry),
            "route_revision": None if carry.get("route_revision") is None else int(carry["route_revision"]),
            "route": list(carry.get("route", [])),
            "current_holder": carry.get("current_holder"),
            "current_hop_index": None if carry.get("current_hop_index") is None else int(carry["current_hop_index"]),
            "hop_source": carry.get("hop_source"), "hop_destination": carry.get("hop_destination"),
            "hop_progress": None if carry.get("hop_progress") is None else float(carry["hop_progress"]),
            "hop_remaining": None if carry.get("hop_remaining") is None else float(carry["hop_remaining"]),
            "carrying_feature_mask": {field: bool(carry_mask.get(field)) for field in ("hop_progress", "hop_remaining")},
            "active": bool(carry.get("active", False)),
        })
    return result


def _logical_semantic_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in sorted((r for r in rows if r.get("known", True)), key=lambda item: int(item["flow_index"])):
        mask = row.get("feature_mask", {})
        result.append({
            "flow_id": str(row["flow_id"]), "flow_index": int(row["flow_index"]),
            "task_id": str(row["task_id"]), "flow_type": str(row["flow_type"]),
            "epoch": int(row["epoch"]), "logical_source": str(row["logical_source"]),
            "logical_destination": str(row["logical_destination"]),
            "total_data": None if row.get("total_data") is None else float(row["total_data"]),
            "e2e_delivered": None if row.get("e2e_delivered") is None else float(row["e2e_delivered"]),
            "e2e_remaining": None if row.get("e2e_remaining") is None else float(row["e2e_remaining"]),
            "presence": bool(row.get("presence", False)), "status": str(row["status"]),
            "logical_destination_source": str(row["logical_destination_source"]),
            "logical_destination_capture_phase": str(row["logical_destination_capture_phase"]),
            "feature_mask": {field: bool(mask.get(field)) for field in ("total_data", "e2e_delivered", "e2e_remaining")},
        })
    return result


def _carrying_semantic_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in sorted((r for r in rows if r.get("known", True)), key=lambda item: str(item["flow_id"])):
        mask = row.get("feature_mask", {})
        result.append({
            "flow_id": str(row["flow_id"]),
            "route_revision": None if row.get("route_revision") is None else int(row["route_revision"]),
            "route": [str(node) for node in row.get("route", [])],
            "current_holder": row.get("current_holder"),
            "current_hop_index": None if row.get("current_hop_index") is None else int(row["current_hop_index"]),
            "hop_source": row.get("hop_source"), "hop_destination": row.get("hop_destination"),
            "hop_progress": None if row.get("hop_progress") is None else float(row["hop_progress"]),
            "hop_remaining": None if row.get("hop_remaining") is None else float(row["hop_remaining"]),
            "active": bool(row.get("active", False)),
            "feature_mask": {field: bool(mask.get(field)) for field in ("hop_progress", "hop_remaining")},
        })
    return result


def _tensor_semantic_payload(
    frame: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a sample frame into the semantic fields represented by the tensor.

    ``apply_flow_normalization`` returns a copy, so callers may legitimately
    compare a pre-normalization sample with the tensor built from its
    normalized copy.  When a row has no ``normalized_numeric`` values yet,
    derive them from the tensor's recorded train-only statistics instead of
    silently treating the raw value as the normalized value.
    """

    def normalized_value(row: Mapping[str, Any], field: str, feature_name: str) -> float:
        numeric = row.get("normalized_numeric", {})
        if field in numeric and numeric.get(field) is not None:
            return float(numeric[field])
        value = row.get(field)
        mask = bool(row.get("feature_mask", {}).get(field))
        if value is None or not mask:
            return 0.0
        feature = (normalization_stats or {}).get("features", {}).get(feature_name)
        if feature is None:
            return float(value)
        return (float(value) - float(feature["mean"])) / float(feature["std"])

    logical = []
    for row in sorted(frame.get("logical_flows", []), key=lambda item: int(item["flow_index"])):
        known = bool(row.get("known"))
        mask = row.get("feature_mask", {})
        logical.append({
            "flow_id": None if not known else str(row["flow_id"]),
            "flow_index": int(row["flow_index"]), "known": known, "presence": bool(row.get("presence", False)),
            "task_id": None if not known else row.get("task_id"),
            "task_index": -1 if not known or row.get("task_index") is None else int(row["task_index"]),
            "flow_type": None if not known else str(row["flow_type"]),
            "status": None if not known else str(row["status"]),
            "epoch": -1 if not known or row.get("epoch") is None else int(row["epoch"]),
            "logical_source": None if not known else row.get("logical_source"),
            "logical_destination": None if not known else row.get("logical_destination"),
            "logical_destination_source": None if not known else row.get("logical_destination_source"),
            "logical_destination_capture_phase": None if not known else row.get("logical_destination_capture_phase"),
            "source_index": -1 if not known or row.get("logical_source_index") is None else int(row["logical_source_index"]),
            "destination_index": -1 if not known or row.get("logical_destination_index") is None else int(row["logical_destination_index"]),
            "raw_features": [0.0 if row.get(field) is None else float(row[field]) for field in ("total_data", "e2e_delivered", "e2e_remaining")],
            "normalized_features": [normalized_value(row, field, f"logical.{field}") for field in ("total_data", "e2e_delivered", "e2e_remaining")],
            "feature_mask": [bool(mask.get(field)) for field in ("total_data", "e2e_delivered", "e2e_remaining")],
        })
    carrying = []
    for row in sorted(frame.get("carrying_states", []), key=lambda item: int(item["flow_index"])):
        known = bool(row.get("known"))
        mask = row.get("feature_mask", {})
        carrying.append({
            "flow_id": None if not known else str(row.get("flow_id")),
            "flow_index": int(row["flow_index"]), "known": known, "active": bool(row.get("active", False)),
            "route_revision": -1 if not known or row.get("route_revision") is None else int(row["route_revision"]),
            "current_holder_index": -1 if not known or row.get("current_holder_index") is None else int(row["current_holder_index"]),
            "current_hop_index": -1 if not known or row.get("current_hop_index") is None else int(row["current_hop_index"]),
            "hop_source_index": -1 if not known or row.get("hop_source_index") is None else int(row["hop_source_index"]),
            "hop_destination_index": -1 if not known or row.get("hop_destination_index") is None else int(row["hop_destination_index"]),
            "route": [str(value) for value in row.get("route", [])] if known else [],
            "current_holder": None if not known else row.get("current_holder"),
            "hop_source": None if not known else row.get("hop_source"),
            "hop_destination": None if not known else row.get("hop_destination"),
            "raw_features": [0.0 if row.get(field) is None else float(row[field]) for field in ("hop_progress", "hop_remaining")],
            "normalized_features": [normalized_value(row, field, f"carrying.{field}") for field in ("hop_progress", "hop_remaining")],
            "feature_mask": [bool(mask.get(field)) for field in ("hop_progress", "hop_remaining")],
            "route_node_indices": [int(value) for value in row.get("route_node_indices", [])] if known else [],
            "active": bool(row.get("active", False)),
        })
    return {"logical": logical, "carrying": carrying}


def _index_by_raw_flow(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    pairs = {str(row["flow_id"]): int(row["flow_index"]) for row in rows}
    if len(set(pairs.values())) != len(pairs):
        raise ValueError("duplicate Raw Flow index")
    return dict(sorted(pairs.items(), key=lambda item: item[1]))


def _padding_flow(flow_id: str, flow_index: int) -> dict[str, Any]:
    return {
        "flow_id": flow_id, "flow_index": int(flow_index), "known": False,
        "task_id": None, "task_index": None, "flow_type": None, "epoch": None,
        "logical_source": None, "logical_source_index": None,
        "logical_destination": None, "logical_destination_index": None,
        "total_data": None, "e2e_delivered": None, "e2e_remaining": None,
        "presence": False, "status": None,
        "feature_mask": {"total_data": False, "e2e_delivered": False, "e2e_remaining": False},
        "logical_destination_source": None, "logical_destination_capture_phase": None,
    }


def _padding_carrying(flow_id: str, flow_index: int) -> dict[str, Any]:
    return {
        "flow_id": flow_id, "flow_index": int(flow_index), "known": False,
        "route_revision": None, "route": [], "route_node_indices": [],
        "current_holder": None, "current_holder_index": None, "current_hop_index": None,
        "hop_source": None, "hop_source_index": None, "hop_destination": None,
        "hop_destination_index": None, "hop_progress": None, "hop_remaining": None,
        "active": False, "feature_mask": {"hop_progress": False, "hop_remaining": False},
    }


def _sample_rows(
    logical_rows: Sequence[Mapping[str, Any]], carrying_rows: Sequence[Mapping[str, Any]],
    flow_index: Mapping[str, int], node_index: Mapping[str, int], task_index: Mapping[str, int],
    *, target: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logical_by_id = {str(row["flow_id"]): row for row in logical_rows}
    carrying_by_id = {str(row["flow_id"]): row for row in carrying_rows}
    result_flows: list[dict[str, Any]] = []
    result_carrying: list[dict[str, Any]] = []
    for flow_id, slot in sorted(flow_index.items(), key=lambda item: item[1]):
        raw = logical_by_id.get(flow_id)
        carry = carrying_by_id.get(flow_id)
        if raw is None:
            if target:
                continue
            result_flows.append(_padding_flow(flow_id, slot))
            result_carrying.append(_padding_carrying(flow_id, slot))
            continue
        if carry is None:
            raise ValueError(f"Raw carrying row missing for {flow_id}")
        raw_slot = int(raw["flow_index"])
        if raw_slot != slot:
            raise ValueError("Raw Flow ID/index mismatch")
        task_id = str(raw["task_id"])
        source = str(raw["logical_source"])
        destination = str(raw["logical_destination"])
        if task_id not in task_index or source not in node_index or destination not in node_index:
            raise ValueError(f"Flow reference absent from {'target' if target else 'History'} namespace: {flow_id}")
        feature_mask = raw.get("feature_mask", {})
        flow = {
            **copy.deepcopy(dict(raw)), "known": True,
            "flow_index": slot, "task_index": int(task_index[task_id]),
            "logical_source_index": int(node_index[source]),
            "logical_destination_index": int(node_index[destination]),
            "feature_mask": {
                "total_data": bool(feature_mask.get("total_data", True)),
                "e2e_delivered": bool(feature_mask.get("e2e_delivered", True)),
                "e2e_remaining": bool(feature_mask.get("e2e_remaining", True)),
            },
        }
        route = [str(value) for value in carry.get("route", [])]
        if any(node not in node_index for node in route):
            raise ValueError(f"carrying route reference absent from namespace: {flow_id}")
        def optional_index(value: Any) -> int | None:
            return None if value is None else int(node_index[str(value)])
        carrying = {
            **copy.deepcopy(dict(carry)), "known": True, "flow_index": slot,
            "route_node_indices": [int(node_index[node]) for node in route],
            "current_holder_index": optional_index(carry.get("current_holder")),
            "hop_source_index": optional_index(carry.get("hop_source")),
            "hop_destination_index": optional_index(carry.get("hop_destination")),
            "feature_mask": {
                "hop_progress": carry.get("hop_progress") is not None,
                "hop_remaining": carry.get("hop_remaining") is not None,
            },
        }
        if target:
            flow["target_index"] = slot
            carrying["target_index"] = slot
        result_flows.append(flow)
        result_carrying.append(carrying)
    return result_flows, result_carrying


def build_flow_extended_sample(base_sample: Mapping[str, Any], ledger_raw: Mapping[str, Any]) -> dict[str, Any]:
    if base_sample.get("schema_version") != STEP42A_SAMPLE_SCHEMA_VERSION:
        raise ValueError("STEP 4.2A sample required")
    if ledger_raw.get("schema_version") != STEP42CB_RAW_SCHEMA_VERSION:
        raise ValueError("STEP 4.2C-B Raw amendment required")
    sample = copy.deepcopy(dict(base_sample))
    decisions = {int(row["frame_index"]): row for row in ledger_raw.get("decisions", [])}
    history_frames = [int(row["frame_index"]) for row in sample["history"]]
    target_source_frames = [int(row["frame_index"]) + 1 for row in sample["target"]]
    history_sources = [decisions[frame] for frame in history_frames]
    target_sources = [decisions[frame] for frame in target_source_frames]

    history_all = [row for decision in history_sources for row in decision.get("logical_flow_rows", [])]
    target_all = [row for decision in target_sources for row in decision.get("logical_flow_rows", [])]
    input_flow_index = _index_by_raw_flow(history_all)
    target_flow_index = _index_by_raw_flow(target_all)
    sample["static"]["input_entity_index"]["logical_flow"] = input_flow_index
    sample["static"]["target_index"]["logical_flow"] = target_flow_index
    sample["static"]["target_only_objects"]["logical_flow"] = sorted(set(target_flow_index) - set(input_flow_index))

    input_nodes = {str(key): int(value) for key, value in sample["static"]["input_entity_index"]["physical"].items()}
    input_tasks = {str(key): int(value) for key, value in sample["static"]["input_entity_index"]["task"].items()}
    target_nodes = {str(key): int(value) for key, value in sample["static"]["target_index"]["physical"].items()}
    target_tasks = {str(key): int(value) for key, value in sample["static"]["target_index"]["task"].items()}

    history_digests: dict[str, str] = {}
    history_logical_digests: dict[str, str] = {}
    history_carrying_digests: dict[str, str] = {}
    for frame, decision in zip(sample["history"], history_sources):
        flows, carrying = _sample_rows(decision.get("logical_flow_rows", []), decision.get("carrying_rows", []), input_flow_index, input_nodes, input_tasks, target=False)
        frame["logical_flows"] = flows
        frame["carrying_states"] = carrying
        frame["logical_flow_source"] = "O_t.logical_flow_rows_and_carrying_rows"
        history_digests[str(frame["frame_index"])] = _digest(_semantic_payload(decision.get("logical_flow_rows", []), decision.get("carrying_rows", [])))
        history_logical_digests[str(frame["frame_index"])] = _digest(_logical_semantic_payload(decision.get("logical_flow_rows", [])))
        history_carrying_digests[str(frame["frame_index"])] = _digest(_carrying_semantic_payload(decision.get("carrying_rows", [])))

    target_digests: dict[str, str] = {}
    target_logical_digests: dict[str, str] = {}
    target_carrying_digests: dict[str, str] = {}
    for target_row, decision, source_frame in zip(sample["target"], target_sources, target_source_frames):
        flows, carrying = _sample_rows(decision.get("logical_flow_rows", []), decision.get("carrying_rows", []), target_flow_index, target_nodes, target_tasks, target=True)
        target_row["logical_flows"] = flows
        target_row["carrying_states"] = carrying
        target_row["logical_flow_source_decision_frame"] = source_frame
        target_digests[str(target_row["frame_index"])] = _digest(_semantic_payload(decision.get("logical_flow_rows", []), decision.get("carrying_rows", [])))
        target_logical_digests[str(target_row["frame_index"])] = _digest(_logical_semantic_payload(decision.get("logical_flow_rows", [])))
        target_carrying_digests[str(target_row["frame_index"])] = _digest(_carrying_semantic_payload(decision.get("carrying_rows", [])))

    sample["schema_version"] = SAMPLE_SCHEMA_VERSION
    sample["contract"]["schema_version"] = SAMPLE_SCHEMA_VERSION
    sample["contract"]["base_step4_2a_sample_schema_version"] = STEP42A_SAMPLE_SCHEMA_VERSION
    sample["contract"]["logical_flow_index_policy"] = "history_causal_logical_flow_union"
    sample["contract"]["target_logical_flow_index_policy"] = "future_ground_truth_separate_namespace"
    sample["contract"]["flow_source_of_truth"] = "STEP_4.2C_B_RAW_ONLY_NO_RECOMPUTATION"
    sample["metadata"]["raw_flow_contract_version"] = STEP42CB_RAW_SCHEMA_VERSION
    sample["metadata"].setdefault(
        "sample_id",
        f"{sample['metadata']['trajectory_id']}::anchor-{int(sample['metadata']['anchor_decision_frame']):04d}",
    )
    sample["metadata"]["history_flow_semantic_digests"] = history_digests
    sample["metadata"]["target_flow_semantic_digests"] = target_digests
    sample["metadata"]["history_logical_flow_semantic_digests"] = history_logical_digests
    sample["metadata"]["history_carrying_semantic_digests"] = history_carrying_digests
    sample["metadata"]["target_logical_flow_semantic_digests"] = target_logical_digests
    sample["metadata"]["target_carrying_semantic_digests"] = target_carrying_digests
    sample["metadata"]["history_flow_tensor_semantics"] = [_tensor_semantic_payload(frame) for frame in sample["history"]]
    sample["metadata"]["target_flow_tensor_semantics"] = [_tensor_semantic_payload(frame) for frame in sample["target"]]
    checks = validate_flow_sample_checks(sample)
    if not checks["passed"]:
        raise ValueError(f"Flow sample validation failed: {checks}")
    return sample


def _frame_digest(frame: Mapping[str, Any]) -> str:
    return _digest(_semantic_payload(frame.get("logical_flows", []), frame.get("carrying_states", [])))


def _sample_layer_digest_checks(sample: Mapping[str, Any]) -> dict[str, bool]:
    metadata = sample.get("metadata", {})
    history_logical = metadata.get("history_logical_flow_semantic_digests", {})
    history_carrying = metadata.get("history_carrying_semantic_digests", {})
    target_logical = metadata.get("target_logical_flow_semantic_digests", {})
    target_carrying = metadata.get("target_carrying_semantic_digests", {})
    return {
        "raw_sample_history_logical_equality": all(
            _digest(_logical_semantic_payload(frame.get("logical_flows", []))) == history_logical.get(str(frame.get("frame_index")))
            for frame in sample.get("history", [])
        ),
        "raw_sample_history_carrying_equality": all(
            _digest(_carrying_semantic_payload(frame.get("carrying_states", []))) == history_carrying.get(str(frame.get("frame_index")))
            for frame in sample.get("history", [])
        ),
        "raw_sample_target_logical_equality": all(
            _digest(_logical_semantic_payload(frame.get("logical_flows", []))) == target_logical.get(str(frame.get("frame_index")))
            for frame in sample.get("target", [])
        ),
        "raw_sample_target_carrying_equality": all(
            _digest(_carrying_semantic_payload(frame.get("carrying_states", []))) == target_carrying.get(str(frame.get("frame_index")))
            for frame in sample.get("target", [])
        ),
    }


def validate_flow_sample_checks(sample: Mapping[str, Any]) -> dict[str, bool]:
    input_index = sample.get("static", {}).get("input_entity_index", {}).get("logical_flow", {})
    target_index = sample.get("static", {}).get("target_index", {}).get("logical_flow", {})
    target_only = sample.get("static", {}).get("target_only_objects", {}).get("logical_flow", [])
    history = list(sample.get("history", []))
    targets = list(sample.get("target", []))
    history_known = [row for frame in history for row in frame.get("logical_flows", []) if row.get("known")]
    target_known = [row for frame in targets for row in frame.get("logical_flows", []) if row.get("known")]
    all_known = history_known + target_known
    digest_history = sample.get("metadata", {}).get("history_flow_semantic_digests", {})
    digest_target = sample.get("metadata", {}).get("target_flow_semantic_digests", {})
    digest_equal = all(_frame_digest(frame) == digest_history.get(str(frame.get("frame_index"))) for frame in history)
    digest_equal = digest_equal and all(_frame_digest(frame) == digest_target.get(str(frame.get("frame_index"))) for frame in targets)
    layer_digests = _sample_layer_digest_checks(sample)
    identity_ok = True
    for row in all_known:
        try:
            identity_ok = identity_ok and parse_flow_id(str(row["flow_id"])) == (str(row["task_id"]), str(row["flow_type"]), int(row["epoch"]))
        except (TypeError, ValueError):
            identity_ok = False
    conservation = all(math.isclose(float(row["e2e_delivered"]) + float(row["e2e_remaining"]), float(row["total_data"]), abs_tol=1e-9) for row in all_known)
    fixed_rows = all(len(frame.get("logical_flows", [])) == len(input_index) and len(frame.get("carrying_states", [])) == len(input_index) for frame in history)
    stable = all(int(row["flow_index"]) == int(input_index[row["flow_id"]]) for row in history_known)
    target_stable = all(int(row["flow_index"]) == int(target_index[row["flow_id"]]) for row in target_known)
    known_vs_padding = all(row.get("status") in FLOW_STATUS_VOCAB[2:] for row in all_known) and all(
        (not row.get("known")) <= (row.get("status") is None and row.get("presence") is False)
        for frame in history for row in frame.get("logical_flows", [])
    )
    checks = {
        "schema_version": sample.get("schema_version") == SAMPLE_SCHEMA_VERSION,
        "raw_sample_semantic_digest_equality": bool(digest_equal),
        **layer_digests,
        "history_causal_flow_union": bool(fixed_rows and sample.get("contract", {}).get("logical_flow_index_policy") == "history_causal_logical_flow_union"),
        "stable_flow_id_index": bool(stable and target_stable),
        "flow_id_epoch_identity": bool(identity_ok),
        "flow_conservation": bool(conservation),
        "target_only_future_flow_isolation": not bool(set(target_only) & set(input_index)) and set(target_only) == set(target_index) - set(input_index),
        "known_inactive_distinct_from_padding": bool(known_vs_padding),
        "logical_destination_provenance": all(bool(row.get("logical_destination_source")) and bool(row.get("logical_destination_capture_phase")) for row in all_known),
        "depdata_runtime_zero": not any(row.get("flow_type") == "DepData" for row in all_known),
        "input_return_explicit_categories": all(row.get("flow_type") in {"Input", "Return"} for row in all_known),
        "raw_only_no_recomputation": sample.get("contract", {}).get("flow_source_of_truth") == "STEP_4.2C_B_RAW_ONLY_NO_RECOMPUTATION" and all(frame.get("logical_flow_source") == "O_t.logical_flow_rows_and_carrying_rows" for frame in history),
        "no_duplicate_flow_per_frame": all(len({row["flow_id"] for row in frame.get("logical_flows", []) if row.get("known")}) == sum(bool(row.get("known")) for row in frame.get("logical_flows", [])) for frame in history + targets),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def _iter_numeric(samples: Sequence[Mapping[str, Any]]):
    for sample in samples:
        if sample.get("metadata", {}).get("split") != "dev_train":
            continue
        for frame in sample.get("history", []):
            carrying = {row["flow_id"]: row for row in frame.get("carrying_states", []) if row.get("known")}
            for row in frame.get("logical_flows", []):
                if not row.get("known") or not row.get("presence"):
                    continue
                for name, field in (("logical.total_data", "total_data"), ("logical.e2e_delivered", "e2e_delivered"), ("logical.e2e_remaining", "e2e_remaining")):
                    if row.get("feature_mask", {}).get(field) and row.get(field) is not None:
                        yield name, float(row[field])
                carry = carrying.get(row["flow_id"])
                if carry is None or not carry.get("known"):
                    continue
                for name, field in (("carrying.hop_progress", "hop_progress"), ("carrying.hop_remaining", "hop_remaining")):
                    if carry.get("feature_mask", {}).get(field) and carry.get(field) is not None:
                        yield name, float(carry[field])


def fit_flow_normalization_stats(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {name: [] for name in FLOW_NUMERIC_FEATURE_ORDER}
    for name, value in _iter_numeric(samples):
        values[name].append(value)
    features = {}
    for name in FLOW_NUMERIC_FEATURE_ORDER:
        data = np.asarray(values[name], dtype=np.float64)
        if data.size == 0:
            mean, std, zero = 0.0, 1.0, True
        else:
            mean = float(data.mean())
            observed_std = float(data.std())
            zero = observed_std <= 1e-12
            std = 1.0 if zero else observed_std
        features[name] = {
            "count": int(data.size), "mean": mean, "std": std,
            "zero_variance_handling": "std_replaced_with_1" if zero else "not_required",
            "source_field": name, "unit": FLOW_NUMERIC_UNITS[name],
            "mask_policy": STEP32_NORMALIZATION_MASK_POLICY,
            "presence_source": "logical_flow.presence for logical fields; corresponding logical_flow.presence plus carrying.known for carrying fields",
        }
    return {"schema_version": NORMALIZATION_SCHEMA_VERSION, "source_split": "dev_train", "features": features}


def _normalized(value: Any, feature: Mapping[str, Any], valid: bool) -> float | None:
    if not valid or value is None:
        return None
    return (float(value) - float(feature["mean"])) / float(feature["std"])


def apply_flow_normalization(samples: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    if stats.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("Flow normalization schema mismatch")
    output = copy.deepcopy(list(samples))
    features = stats["features"]
    for sample in output:
        for frame in [*sample.get("history", []), *sample.get("target", [])]:
            carrying = {row["flow_id"]: row for row in frame.get("carrying_states", []) if row.get("known")}
            for row in frame.get("logical_flows", []):
                if not row.get("known"):
                    continue
                row["normalized_numeric"] = {
                    field: _normalized(row.get(field), features[name], bool(row.get("feature_mask", {}).get(field)))
                    for name, field in (("logical.total_data", "total_data"), ("logical.e2e_delivered", "e2e_delivered"), ("logical.e2e_remaining", "e2e_remaining"))
                }
                carry = carrying[row["flow_id"]]
                carry["normalized_numeric"] = {
                    field: _normalized(carry.get(field), features[name], bool(carry.get("feature_mask", {}).get(field)))
                    for name, field in (("carrying.hop_progress", "hop_progress"), ("carrying.hop_remaining", "hop_remaining"))
                }
        sample["metadata"]["history_flow_tensor_semantics"] = [_tensor_semantic_payload(frame) for frame in sample.get("history", [])]
        sample["metadata"]["target_flow_tensor_semantics"] = [_tensor_semantic_payload(frame) for frame in sample.get("target", [])]
    return output


@dataclass(frozen=True)
class FlowTensorContract:
    max_logical_flow: int = 0
    max_target_logical_flow: int = 0
    max_route_nodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capacity(samples: Sequence[Mapping[str, Any]]) -> FlowTensorContract:
    return FlowTensorContract(
        max_logical_flow=max((len(s["static"]["input_entity_index"]["logical_flow"]) for s in samples), default=0),
        max_target_logical_flow=max((len(s["static"]["target_index"]["logical_flow"]) for s in samples), default=0),
        max_route_nodes=max((len(row.get("route", [])) for s in samples for frame in [*s.get("history", []), *s.get("target", [])] for row in frame.get("carrying_states", []) if row.get("known")), default=0),
    )


def build_flow_tensor_batch(
    samples: Sequence[Mapping[str, Any]], *, stats: Mapping[str, Any],
    contract: FlowTensorContract | None = None,
) -> dict[str, Any]:
    samples = list(samples)
    if not samples:
        raise ValueError("cannot tensorize empty Flow sample batch")
    observed = _capacity(samples)
    c = observed if contract is None else contract
    for field in ("max_logical_flow", "max_target_logical_flow", "max_route_nodes"):
        if getattr(observed, field) > getattr(c, field):
            raise ValueError(f"capacity overflow: {field} observed={getattr(observed, field)} capacity={getattr(c, field)}")

    base_samples = copy.deepcopy(samples)
    for sample in base_samples:
        sample["schema_version"] = STEP42A_SAMPLE_SCHEMA_VERSION
        sample["contract"]["schema_version"] = STEP42A_SAMPLE_SCHEMA_VERSION
    base = build_extended_tensor_batch(base_samples, stats={})
    out = {key: value for key, value in base.items() if key != "contract"}
    base_contract = dict(base["contract"])
    B, H, L = len(samples), len(samples[0]["history"]), len(samples[0]["target"])
    F, TF, Q = c.max_logical_flow, c.max_target_logical_flow, c.max_route_nodes
    E = int(base_contract["max_entity"])
    T = int(base_contract["max_task"])
    out["schema_version"] = TENSOR_SCHEMA_VERSION
    out["contract"] = {
        **base_contract, **c.to_dict(), "schema_version": TENSOR_SCHEMA_VERSION,
        "sample_contract_version": SAMPLE_SCHEMA_VERSION,
        "raw_flow_contract_version": STEP42CB_RAW_SCHEMA_VERSION,
        "flow_type_vocab": list(FLOW_TYPE_VOCAB), "flow_status_vocab": list(FLOW_STATUS_VOCAB),
        "logical_flow_numeric_feature_order": list(FLOW_NUMERIC_FEATURE_ORDER[:3]),
        "carrying_numeric_feature_order": list(FLOW_NUMERIC_FEATURE_ORDER[3:]),
        "logical_destination_provenance_model_feature": False,
        "capacity_scope": "explicit_development_bound_not_formal_capacity",
        "stateful_flow_policy": "STEP_4.2C_C_LOGICAL_FLOW_SEPARATE_FROM_LEGACY_HOP_SERVICE",
        "target_carrying_semantics": "future_ground_truth_deterministic_transition_state; not_a_learned_prediction_head",
        "normalization_mask_policy": STEP32_NORMALIZATION_MASK_POLICY,
        "graph_builder": False,
    }
    z = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    ix = lambda shape: np.full(shape, -1, dtype=np.int64)
    out.update({
        "logical_flow_known_mask": z((B, H, F), bool), "logical_flow_presence": z((B, H, F), bool),
        "logical_flow_index": ix((B, H, F)), "logical_flow_task_index": ix((B, H, F)),
        "logical_flow_type_index": z((B, H, F), np.int64), "logical_flow_status_index": z((B, H, F), np.int64),
        "logical_flow_epoch": ix((B, H, F)), "logical_flow_source_index": ix((B, H, F)),
        "logical_flow_destination_index": ix((B, H, F)),
        "logical_flow_raw_features": z((B, H, F, 3)), "logical_flow_features": z((B, H, F, 3)),
        "logical_flow_feature_mask": z((B, H, F, 3), bool),
        "carrying_known_mask": z((B, H, F), bool), "carrying_active": z((B, H, F), bool),
        "carrying_route_revision": ix((B, H, F)), "carrying_current_hop_index": ix((B, H, F)),
        "carrying_holder_index": ix((B, H, F)), "carrying_hop_source_index": ix((B, H, F)),
        "carrying_hop_destination_index": ix((B, H, F)),
        "carrying_raw_features": z((B, H, F, 2)), "carrying_features": z((B, H, F, 2)),
        "carrying_feature_mask": z((B, H, F, 2), bool),
        "route_node_indices": ix((B, H, F, Q)), "route_node_mask": z((B, H, F, Q), bool),
        "target_logical_flow_known_mask": z((B, L, TF), bool), "target_logical_flow_presence": z((B, L, TF), bool),
        "target_logical_flow_index": ix((B, L, TF)), "target_logical_flow_task_index": ix((B, L, TF)),
        "target_logical_flow_type_index": z((B, L, TF), np.int64), "target_logical_flow_status_index": z((B, L, TF), np.int64),
        "target_logical_flow_epoch": ix((B, L, TF)), "target_logical_flow_source_index": ix((B, L, TF)),
        "target_logical_flow_destination_index": ix((B, L, TF)),
        "target_logical_flow_raw_features": z((B, L, TF, 3)), "target_logical_flow_features": z((B, L, TF, 3)),
        "target_logical_flow_feature_mask": z((B, L, TF, 3), bool),
        "target_carrying_known_mask": z((B, L, TF), bool), "target_carrying_active": z((B, L, TF), bool),
        "target_carrying_route_revision": ix((B, L, TF)), "target_carrying_current_hop_index": ix((B, L, TF)),
        "target_carrying_holder_index": ix((B, L, TF)), "target_carrying_hop_source_index": ix((B, L, TF)),
        "target_carrying_hop_destination_index": ix((B, L, TF)),
        "target_carrying_raw_features": z((B, L, TF, 2)), "target_carrying_features": z((B, L, TF, 2)),
        "target_carrying_feature_mask": z((B, L, TF, 2), bool),
        "target_route_node_indices": ix((B, L, TF, Q)), "target_route_node_mask": z((B, L, TF, Q), bool),
    })

    def fill(frame: Mapping[str, Any], bi: int, ti: int, prefix: str, max_flow: int) -> None:
        known_key = f"{prefix}logical_flow_known_mask"
        presence_key = f"{prefix}logical_flow_presence"
        for row in frame.get("logical_flows", []):
            if not row.get("known"):
                continue
            slot = int(row["flow_index"])
            if slot >= max_flow:
                raise ValueError("capacity overflow: logical Flow slot")
            out[known_key][bi, ti, slot] = True
            out[presence_key][bi, ti, slot] = bool(row["presence"])
            out[f"{prefix}logical_flow_index"][bi, ti, slot] = slot
            out[f"{prefix}logical_flow_task_index"][bi, ti, slot] = int(row["task_index"])
            out[f"{prefix}logical_flow_type_index"][bi, ti, slot] = FLOW_TYPE_VOCAB.index(row["flow_type"])
            out[f"{prefix}logical_flow_status_index"][bi, ti, slot] = FLOW_STATUS_VOCAB.index(row["status"])
            out[f"{prefix}logical_flow_epoch"][bi, ti, slot] = int(row["epoch"])
            out[f"{prefix}logical_flow_source_index"][bi, ti, slot] = int(row["logical_source_index"])
            out[f"{prefix}logical_flow_destination_index"][bi, ti, slot] = int(row["logical_destination_index"])
            for fi, field in enumerate(("total_data", "e2e_delivered", "e2e_remaining")):
                valid = bool(row["feature_mask"].get(field)) and row.get(field) is not None
                if valid:
                    out[f"{prefix}logical_flow_raw_features"][bi, ti, slot, fi] = float(row[field])
                    out[f"{prefix}logical_flow_features"][bi, ti, slot, fi] = float(row.get("normalized_numeric", {}).get(field, row[field]))
                    out[f"{prefix}logical_flow_feature_mask"][bi, ti, slot, fi] = True

    def fill_carrying(frame: Mapping[str, Any], bi: int, ti: int, prefix: str, max_flow: int) -> None:
        for carry in frame.get("carrying_states", []):
            if not carry.get("known"):
                continue
            slot = int(carry["flow_index"])
            if slot >= max_flow:
                raise ValueError("capacity overflow: carrying Flow slot")
            out[f"{prefix}carrying_known_mask"][bi, ti, slot] = True
            out[f"{prefix}carrying_active"][bi, ti, slot] = bool(carry["active"])
            out[f"{prefix}carrying_route_revision"][bi, ti, slot] = int(carry["route_revision"])
            out[f"{prefix}carrying_current_hop_index"][bi, ti, slot] = int(carry["current_hop_index"])
            for key, field in (("carrying_holder_index", "current_holder_index"), ("carrying_hop_source_index", "hop_source_index"), ("carrying_hop_destination_index", "hop_destination_index")):
                if carry.get(field) is not None:
                    out[f"{prefix}{key}"][bi, ti, slot] = int(carry[field])
            for fi, field in enumerate(("hop_progress", "hop_remaining")):
                if carry["feature_mask"].get(field) and carry.get(field) is not None:
                    out[f"{prefix}carrying_raw_features"][bi, ti, slot, fi] = float(carry[field])
                    out[f"{prefix}carrying_features"][bi, ti, slot, fi] = float(carry.get("normalized_numeric", {}).get(field, carry[field]))
                    out[f"{prefix}carrying_feature_mask"][bi, ti, slot, fi] = True
            for qi, node_slot in enumerate(carry.get("route_node_indices", [])):
                if qi >= Q:
                    raise ValueError("capacity overflow: route node")
                out[f"{prefix}route_node_indices"][bi, ti, slot, qi] = int(node_slot)
                out[f"{prefix}route_node_mask"][bi, ti, slot, qi] = True

    for bi, sample in enumerate(samples):
        for hi, frame in enumerate(sample["history"]):
            fill(frame, bi, hi, "", F)
            fill_carrying(frame, bi, hi, "", F)
        for li, frame in enumerate(sample["target"]):
            fill(frame, bi, li, "target_", TF)
            fill_carrying(frame, bi, li, "target_", TF)

    out["sample_static"] = copy.deepcopy([sample["static"] for sample in samples])
    out["sample_metadata"] = copy.deepcopy([sample["metadata"] for sample in samples])
    out["flow_normalization_stats"] = copy.deepcopy(dict(stats))
    checks = validate_flow_tensor_checks(out)
    if not checks["passed"]:
        raise ValueError(f"Flow tensor validation failed: {checks}")
    return out


# These helpers keep the public receipt names stable while the patched
# contract checks every retained Flow/Carrying field.
def _same_array_v2(left: Any, right: Any, *, dtype: Any | None = None) -> bool:
    try:
        a = np.asarray(left, dtype=dtype)
        b = np.asarray(right, dtype=dtype)
        return bool(np.array_equal(a, b) if a.dtype.kind in "biu" else np.allclose(a, b, atol=1e-6, rtol=1e-6))
    except (TypeError, ValueError):
        return False


def _semantic_identity_projection(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the non-numeric identities that the tensor stores in metadata.

    Numeric index arrays are checked separately by
    ``_tensor_semantic_layer_checks_v2``.  These fields preserve the source
    IDs/provenance needed to reject an ID tamper that leaves a numeric index
    unchanged (for example, changing a logical destination name only).
    """

    logical_keys = (
        "flow_id", "flow_index", "known", "presence", "task_id", "task_index",
        "flow_type", "status", "epoch", "logical_source", "logical_destination",
        "logical_destination_source", "logical_destination_capture_phase",
        "source_index", "destination_index",
    )
    carrying_keys = (
        "flow_id", "flow_index", "known", "active", "route_revision",
        "current_hop_index", "current_holder_index", "hop_source_index",
        "hop_destination_index", "route", "current_holder", "hop_source",
        "hop_destination", "route_node_indices",
    )
    projection = []
    for frame in frames:
        projection.append({
            "logical": [{key: row.get(key) for key in logical_keys} for row in frame.get("logical", [])],
            "carrying": [{key: row.get(key) for key in carrying_keys} for row in frame.get("carrying", [])],
        })
    return projection


def _tensor_semantic_layer_checks_v2(tensor: Mapping[str, Any], batch_index: int, prefix: str, frames: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    logical_names = (
        "known_mask", "presence", "index", "task_index", "type_index", "status_index", "epoch",
        "source_index", "destination_index", "raw_features", "features", "feature_mask",
    )
    carrying_names = (
        "known_mask", "active", "route_revision", "current_hop_index", "holder_index",
        "hop_source_index", "hop_destination_index", "raw_features", "features", "feature_mask",
    )
    logical_ok = all(f"{prefix}logical_flow_{name}" in tensor for name in logical_names)
    carrying_ok = all(f"{prefix}carrying_{name}" in tensor for name in carrying_names) and all(
        f"{prefix}{name}" in tensor for name in ("route_node_indices", "route_node_mask")
    )
    if not logical_ok or not carrying_ok:
        return logical_ok, carrying_ok
    try:
        for frame_index, frame in enumerate(frames):
            logical_rows = list(frame.get("logical", []))
            carrying_rows = list(frame.get("carrying", []))
            logical_known = tensor[f"{prefix}logical_flow_known_mask"]
            carrying_known = tensor[f"{prefix}carrying_known_mask"]
            if frame_index >= logical_known.shape[1] or len(logical_rows) > logical_known.shape[2] or len(carrying_rows) > carrying_known.shape[2]:
                return False, False
            for expected in logical_rows:
                slot = int(expected["flow_index"])
                known = bool(expected["known"])
                if slot < 0 or slot >= logical_known.shape[2] or bool(logical_known[batch_index, frame_index, slot]) != known:
                    return False, carrying_ok
                if bool(tensor[f"{prefix}logical_flow_presence"][batch_index, frame_index, slot]) != bool(expected["presence"]):
                    return False, carrying_ok
                scalars = {
                    "index": slot if known else -1,
                    "task_index": int(expected["task_index"]),
                    "type_index": 0 if not known else FLOW_TYPE_VOCAB.index(expected["flow_type"]),
                    "status_index": 0 if not known else FLOW_STATUS_VOCAB.index(expected["status"]),
                    "epoch": int(expected["epoch"]), "source_index": int(expected["source_index"]),
                    "destination_index": int(expected["destination_index"]),
                }
                for name, value in scalars.items():
                    if int(tensor[f"{prefix}logical_flow_{name}"][batch_index, frame_index, slot]) != value:
                        return False, carrying_ok
                if not _same_array_v2(tensor[f"{prefix}logical_flow_feature_mask"][batch_index, frame_index, slot], expected["feature_mask"], dtype=bool):
                    return False, carrying_ok
                if not _same_array_v2(tensor[f"{prefix}logical_flow_raw_features"][batch_index, frame_index, slot], expected["raw_features"]):
                    return False, carrying_ok
                if not _same_array_v2(tensor[f"{prefix}logical_flow_features"][batch_index, frame_index, slot], expected.get("normalized_features", expected["raw_features"])):
                    return False, carrying_ok
            for expected in carrying_rows:
                slot = int(expected["flow_index"])
                known = bool(expected["known"])
                if slot < 0 or slot >= carrying_known.shape[2] or bool(carrying_known[batch_index, frame_index, slot]) != known:
                    return logical_ok, False
                if bool(tensor[f"{prefix}carrying_active"][batch_index, frame_index, slot]) != bool(expected["active"]):
                    return logical_ok, False
                for tensor_name, expected_name in (("route_revision", "route_revision"), ("current_hop_index", "current_hop_index"), ("holder_index", "current_holder_index"), ("hop_source_index", "hop_source_index"), ("hop_destination_index", "hop_destination_index")):
                    if int(tensor[f"{prefix}carrying_{tensor_name}"][batch_index, frame_index, slot]) != int(expected[expected_name]):
                        return logical_ok, False
                if not _same_array_v2(tensor[f"{prefix}carrying_feature_mask"][batch_index, frame_index, slot], expected["feature_mask"], dtype=bool):
                    return logical_ok, False
                if not _same_array_v2(tensor[f"{prefix}carrying_raw_features"][batch_index, frame_index, slot], expected["raw_features"]):
                    return logical_ok, False
                if not _same_array_v2(tensor[f"{prefix}carrying_features"][batch_index, frame_index, slot], expected.get("normalized_features", expected["raw_features"])):
                    return logical_ok, False
                route = tensor[f"{prefix}route_node_indices"][batch_index, frame_index, slot]
                route_mask = tensor[f"{prefix}route_node_mask"][batch_index, frame_index, slot]
                expected_route = list(expected["route_node_indices"])
                if len(expected_route) > len(route):
                    return logical_ok, False
                expected_mask = [True] * len(expected_route) + [False] * (len(route) - len(expected_route))
                expected_values = expected_route + [-1] * (len(route) - len(expected_route))
                if not _same_array_v2(route_mask, expected_mask, dtype=bool) or not _same_array_v2(route, expected_values):
                    return logical_ok, False
    except (KeyError, IndexError, TypeError, ValueError):
        return False, False
    return True, True


def _masked_zero_v2(tensor: Mapping[str, Any], feature_key: str, mask_key: str) -> bool:
    if feature_key not in tensor or mask_key not in tensor:
        return False
    features = np.asarray(tensor[feature_key])
    mask = np.asarray(tensor[mask_key], dtype=bool)
    return bool(np.all(features[~mask] == 0))


def validate_flow_tensor_checks(tensor: Mapping[str, Any]) -> dict[str, bool]:
    contract = dict(tensor.get("contract", {}))
    arrays = (
        "logical_flow_known_mask", "logical_flow_presence", "logical_flow_index", "logical_flow_task_index",
        "logical_flow_type_index", "logical_flow_status_index", "logical_flow_epoch", "logical_flow_source_index",
        "logical_flow_destination_index", "logical_flow_raw_features", "logical_flow_features", "logical_flow_feature_mask",
        "carrying_known_mask", "carrying_active", "carrying_route_revision", "carrying_current_hop_index",
        "carrying_holder_index", "carrying_hop_source_index", "carrying_hop_destination_index", "carrying_raw_features",
        "carrying_features", "carrying_feature_mask", "route_node_indices", "route_node_mask",
        "target_logical_flow_known_mask", "target_logical_flow_presence", "target_logical_flow_index",
        "target_logical_flow_task_index", "target_logical_flow_type_index", "target_logical_flow_status_index",
        "target_logical_flow_epoch", "target_logical_flow_source_index", "target_logical_flow_destination_index",
        "target_logical_flow_raw_features", "target_logical_flow_features", "target_logical_flow_feature_mask",
        "target_carrying_known_mask", "target_carrying_active", "target_carrying_route_revision",
        "target_carrying_current_hop_index", "target_carrying_holder_index", "target_carrying_hop_source_index",
        "target_carrying_hop_destination_index", "target_carrying_raw_features", "target_carrying_features",
        "target_carrying_feature_mask", "target_route_node_indices", "target_route_node_mask",
    )
    arrays_present = all(name in tensor for name in arrays)
    sample_static = list(tensor.get("sample_static", []))
    sample_metadata = list(tensor.get("sample_metadata", []))
    history_semantics = [metadata.get("history_flow_tensor_semantics", []) for metadata in sample_metadata]
    target_semantics = [metadata.get("target_flow_tensor_semantics", []) for metadata in sample_metadata]
    history_layers = [
        _tensor_semantic_layer_checks_v2(tensor, batch_index, "", history_semantics[batch_index])
        if arrays_present and batch_index < len(history_semantics) else (False, False)
        for batch_index in range(len(sample_static))
    ]
    target_layers = [
        _tensor_semantic_layer_checks_v2(tensor, batch_index, "target_", target_semantics[batch_index])
        if arrays_present and batch_index < len(target_semantics) else (False, False)
        for batch_index in range(len(sample_static))
    ]
    history_logical = bool(arrays_present and all(item[0] for item in history_layers))
    history_carrying = bool(arrays_present and all(item[1] for item in history_layers))
    target_logical = bool(arrays_present and all(item[0] for item in target_layers))
    target_carrying = bool(arrays_present and all(item[1] for item in target_layers))
    known = np.asarray(tensor.get("logical_flow_known_mask", np.zeros((0, 0, 0))), dtype=bool)
    presence = np.asarray(tensor.get("logical_flow_presence", np.zeros_like(known)), dtype=bool)
    target_known = np.asarray(tensor.get("target_logical_flow_known_mask", np.zeros((0, 0, 0))), dtype=bool)
    target_presence = np.asarray(tensor.get("target_logical_flow_presence", np.zeros_like(target_known)), dtype=bool)
    identity = True
    target_identity = True
    future_epoch_identity = True
    target_only_isolation = True
    for batch_index, static in enumerate(sample_static):
        input_mapping = static.get("input_entity_index", {}).get("logical_flow", {})
        target_mapping = static.get("target_index", {}).get("logical_flow", {})
        target_only = set(static.get("target_only_objects", {}).get("logical_flow", []))
        if known.ndim != 3 or target_known.ndim != 3:
            identity = target_identity = future_epoch_identity = False
            continue
        for _, slot in input_mapping.items():
            if int(slot) < 0 or int(slot) >= known.shape[2]:
                identity = False
                continue
            active_frames = known[batch_index, :, int(slot)]
            identity = identity and bool(np.all(np.asarray(tensor["logical_flow_index"])[batch_index, :, int(slot)][active_frames] == int(slot)))
        target_only_isolation = target_only_isolation and not bool(target_only & set(input_mapping)) and target_only == (set(target_mapping) - set(input_mapping))
        for flow_id, slot in target_mapping.items():
            if int(slot) < 0 or int(slot) >= target_known.shape[2]:
                target_identity = future_epoch_identity = False
                continue
            active_frames = target_known[batch_index, :, int(slot)]
            target_identity = target_identity and bool(np.all(np.asarray(tensor["target_logical_flow_index"])[batch_index, :, int(slot)][active_frames] == int(slot)))
            try:
                expected_epoch = int(parse_flow_id(str(flow_id))[2])
            except (TypeError, ValueError):
                future_epoch_identity = False
            else:
                future_epoch_identity = future_epoch_identity and bool(np.all(np.asarray(tensor["target_logical_flow_epoch"])[batch_index, :, int(slot)][active_frames] == expected_epoch))
                for frame in target_semantics[batch_index] if batch_index < len(target_semantics) else []:
                    for row in frame.get("logical", []):
                        if row.get("flow_id") == flow_id and row.get("known") and int(row["epoch"]) != expected_epoch:
                            future_epoch_identity = False
    feature_stats = tensor.get("flow_normalization_stats", {}).get("features", {})
    normalization_policy = bool(
        tensor.get("flow_normalization_stats", {}).get("source_split") == "dev_train"
        and set(feature_stats) == set(FLOW_NUMERIC_FEATURE_ORDER)
        and all(feature.get("mask_policy") == STEP32_NORMALIZATION_MASK_POLICY for feature in feature_stats.values())
        and all("presence" in feature.get("presence_source", "") for feature in feature_stats.values())
    )
    max_entity = int(contract.get("max_entity", -1))
    max_task = int(contract.get("max_task", -1))
    endpoint_bounded = bool(arrays_present and np.all((np.asarray(tensor["logical_flow_source_index"])[known] >= 0) & (np.asarray(tensor["logical_flow_source_index"])[known] < max_entity)) and np.all((np.asarray(tensor["logical_flow_destination_index"])[known] >= 0) & (np.asarray(tensor["logical_flow_destination_index"])[known] < max_entity)))
    task_bounded = bool(arrays_present and np.all((np.asarray(tensor["logical_flow_task_index"])[known] >= 0) & (np.asarray(tensor["logical_flow_task_index"])[known] < max_task)))
    target_endpoint_bounded = bool(arrays_present and np.all((np.asarray(tensor["target_logical_flow_source_index"])[target_known] >= 0) & (np.asarray(tensor["target_logical_flow_source_index"])[target_known] < max_entity)) and np.all((np.asarray(tensor["target_logical_flow_destination_index"])[target_known] >= 0) & (np.asarray(tensor["target_logical_flow_destination_index"])[target_known] < max_entity)))
    target_task_bounded = bool(arrays_present and np.all((np.asarray(tensor["target_logical_flow_task_index"])[target_known] >= 0) & (np.asarray(tensor["target_logical_flow_task_index"])[target_known] < max_task)))
    route_mask = np.asarray(tensor.get("route_node_mask", np.zeros((0,), dtype=bool)), dtype=bool)
    route_values = np.asarray(tensor.get("route_node_indices", np.zeros((0,), dtype=np.int64)))
    target_route_mask = np.asarray(tensor.get("target_route_node_mask", np.zeros((0,), dtype=bool)), dtype=bool)
    target_route_values = np.asarray(tensor.get("target_route_node_indices", np.zeros((0,), dtype=np.int64)))
    route_bounds = bool(arrays_present and np.all((route_values[route_mask] >= 0) & (route_values[route_mask] < max_entity)) and np.all(route_values[~route_mask] == -1))
    target_route_bounds = bool(arrays_present and np.all((target_route_values[target_route_mask] >= 0) & (target_route_values[target_route_mask] < max_entity)) and np.all(target_route_values[~target_route_mask] == -1))
    masked_zero = bool(arrays_present and all(_masked_zero_v2(tensor, feature, mask) for feature, mask in (
        ("logical_flow_features", "logical_flow_feature_mask"), ("logical_flow_raw_features", "logical_flow_feature_mask"),
        ("carrying_features", "carrying_feature_mask"), ("carrying_raw_features", "carrying_feature_mask"),
        ("target_logical_flow_features", "target_logical_flow_feature_mask"), ("target_logical_flow_raw_features", "target_logical_flow_feature_mask"),
        ("target_carrying_features", "target_carrying_feature_mask"), ("target_carrying_raw_features", "target_carrying_feature_mask"),
    )))
    no_truncation = bool(arrays_present and known.ndim == 3 and target_known.ndim == 3 and known.shape[2] == int(contract.get("max_logical_flow", -1)) and target_known.shape[2] == int(contract.get("max_target_logical_flow", -1)) and route_mask.ndim == 4 and target_route_mask.ndim == 4 and route_mask.shape[3] == int(contract.get("max_route_nodes", -1)) and target_route_mask.shape[3] == int(contract.get("max_route_nodes", -1)))
    known_presence = bool(np.all(~presence | known) and np.all(~target_presence | target_known))
    carrying_alignment = bool(arrays_present and np.array_equal(np.asarray(tensor["carrying_known_mask"], dtype=bool), known) and np.array_equal(np.asarray(tensor["target_carrying_known_mask"], dtype=bool), target_known))
    target_namespace = bool(
        arrays_present
        and all(
            "target_index" in static
            and "logical_flow" in static.get("target_index", {})
            and "target_only_objects" in static
            for static in sample_static
        )
        and target_identity
        and target_only_isolation
    )
    checks = {
        "schema_version": tensor.get("schema_version") == TENSOR_SCHEMA_VERSION and contract.get("sample_contract_version") == SAMPLE_SCHEMA_VERSION,
        "separate_logical_flow_namespace": contract.get("stateful_flow_policy") == "STEP_4.2C_C_LOGICAL_FLOW_SEPARATE_FROM_LEGACY_HOP_SERVICE" and "past_outcome_flow_service" in tensor,
        "flow_id_tensor_index_round_trip": bool(identity), "target_logical_slot_index_round_trip": bool(target_identity),
        "future_epoch_target_identity": bool(future_epoch_identity), "known_presence_independent": known_presence,
        "target_known_presence_relation": bool(np.all(~target_presence | target_known)),
        "masked_placeholder_zero": masked_zero, "target_masked_placeholder_zero": masked_zero,
        "categorical_not_normalized": all(name not in feature_stats for name in ("epoch", "flow_index", "flow_type", "status", "presence", "route_revision")),
        "presence_aware_normalization_policy": normalization_policy,
        "endpoint_indices_bounded": endpoint_bounded, "task_indices_bounded": task_bounded,
        "target_endpoint_indices_bounded": bool(target_endpoint_bounded and target_task_bounded),
        "target_carrying_alignment": carrying_alignment, "target_route_mask": target_route_bounds,
        "target_only_future_flow_isolation": target_only_isolation, "target_namespace": target_namespace,
        "history_logical_sample_tensor_equality": history_logical, "history_carrying_sample_tensor_equality": history_carrying,
        "target_logical_sample_tensor_equality": target_logical, "target_carrying_sample_tensor_equality": target_carrying,
        "sample_history_logical_equality": history_logical, "sample_history_carrying_equality": history_carrying,
        "sample_target_logical_equality": target_logical, "sample_target_carrying_equality": target_carrying,
        "target_carrying_ground_truth": contract.get("target_carrying_semantics") == "future_ground_truth_deterministic_transition_state; not_a_learned_prediction_head",
        "route_node_mask": route_bounds, "no_silent_truncation": no_truncation,
        "graph_builder_false": contract.get("graph_builder") is False,
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def sample_tensor_semantic_equality_checks(samples: Sequence[Mapping[str, Any]], tensor: Mapping[str, Any]) -> dict[str, bool]:
    results = {
        "sample_tensor_history_logical_equality": True,
        "sample_tensor_history_carrying_equality": True,
        "sample_tensor_target_logical_equality": True,
        "sample_tensor_target_carrying_equality": True,
    }
    normalization_stats = tensor.get("flow_normalization_stats", {})
    tensor_metadata = tensor.get("sample_metadata", [])
    for batch_index, sample in enumerate(samples):
        history = [_tensor_semantic_payload(frame, normalization_stats) for frame in sample.get("history", [])]
        target = [_tensor_semantic_payload(frame, normalization_stats) for frame in sample.get("target", [])]
        if batch_index >= len(tensor_metadata):
            history_identity = target_identity = False
        else:
            metadata = tensor_metadata[batch_index]
            canonical_history = metadata.get("history_flow_tensor_semantics", [])
            canonical_target = metadata.get("target_flow_tensor_semantics", [])
            history_identity = _semantic_identity_projection(history) == _semantic_identity_projection(canonical_history)
            target_identity = _semantic_identity_projection(target) == _semantic_identity_projection(canonical_target)
        logical, carrying = _tensor_semantic_layer_checks_v2(tensor, batch_index, "", history)
        results["sample_tensor_history_logical_equality"] &= logical and history_identity
        results["sample_tensor_history_carrying_equality"] &= carrying and history_identity
        logical, carrying = _tensor_semantic_layer_checks_v2(tensor, batch_index, "target_", target)
        results["sample_tensor_target_logical_equality"] &= logical and target_identity
        results["sample_tensor_target_carrying_equality"] &= carrying and target_identity
    results["sample_tensor_semantic_equality"] = bool(all(results.values()))
    return results


def sample_tensor_semantic_equality(samples: Sequence[Mapping[str, Any]], tensor: Mapping[str, Any]) -> bool:
    return bool(sample_tensor_semantic_equality_checks(samples, tensor)["sample_tensor_semantic_equality"])


def save_flow_tensor_batch(tensor: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in tensor.items() if isinstance(value, np.ndarray)}
    for key in ("contract", "sample_ids", "sample_static", "sample_metadata", "base_step3_3_validation_checks", "flow_normalization_stats"):
        if key in tensor:
            arrays[f"{key}_json"] = np.asarray(json.dumps(tensor[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(path, **arrays)


def load_flow_tensor_batch(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        result = {key: data[key] for key in data.files if not key.endswith("_json")}
        for key in data.files:
            if key.endswith("_json"):
                result[key[:-5]] = json.loads(str(data[key]))
    result["schema_version"] = TENSOR_SCHEMA_VERSION
    return result


def validate_flow_acceptance(receipt: Mapping[str, Any]) -> dict[str, bool]:
    required = receipt.get("required_checks", {})
    required_ok = REQUIRED_ACCEPTANCE_CHECKS <= set(required) and all(required.get(name) is True for name in REQUIRED_ACCEPTANCE_CHECKS)
    scope = receipt.get("scope", {})
    scope_ok = all(scope.get(name) is False for name in ("graph_builder", "information_graph", "physical_topology", "training", "gpu", "locked_test", "formal_dataset"))
    expected = bool(required_ok and scope_ok)
    return {
        "required_checks": bool(required_ok), "scope": bool(scope_ok),
        "expected_passed": expected,
        "declared_matches_expected": receipt.get("passed") is expected,
        "passed": bool(expected and receipt.get("passed") is True),
    }


__all__ = [
    "FLOW_NUMERIC_FEATURE_ORDER", "FLOW_STATUS_VOCAB", "FLOW_TYPE_VOCAB",
    "FlowTensorContract", "NORMALIZATION_SCHEMA_VERSION", "SAMPLE_SCHEMA_VERSION",
    "TENSOR_SCHEMA_VERSION", "apply_flow_normalization", "build_flow_extended_sample",
    "build_flow_tensor_batch", "fit_flow_normalization_stats", "load_flow_tensor_batch",
    "sample_tensor_semantic_equality", "save_flow_tensor_batch", "validate_flow_acceptance",
    "validate_flow_sample_checks", "validate_flow_tensor_checks",
]
