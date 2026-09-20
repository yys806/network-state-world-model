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
NORMALIZATION_SCHEMA_VERSION = "PI-JWM-Step-4.2C-C-Train-Only-Flow-Normalization-v1"
TENSOR_SCHEMA_VERSION = "PI-JWM-Model-Input-Tensor-Collation-v4-step4.2C-C"
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
    "raw_sample_semantic_equality", "sample_tensor_semantic_equality",
    "history_causal_flow_union", "stable_flow_identity",
    "target_only_future_flow_isolation", "epoch_isolation",
    "multi_hop_single_flow_tensor_identity",
    "presence_mask_correctness", "train_only_preprocessing",
    "no_silent_truncation", "input_return_categories",
    "depdata_runtime_zero", "deterministic_rebuild", "serialize_load", "scope",
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
            "carrying_missing": not bool(carry),
            "route_revision": None if carry.get("route_revision") is None else int(carry["route_revision"]),
            "route": list(carry.get("route", [])),
            "current_holder": carry.get("current_holder"),
            "current_hop_index": None if carry.get("current_hop_index") is None else int(carry["current_hop_index"]),
            "hop_source": carry.get("hop_source"), "hop_destination": carry.get("hop_destination"),
            "hop_progress": None if carry.get("hop_progress") is None else float(carry["hop_progress"]),
            "hop_remaining": None if carry.get("hop_remaining") is None else float(carry["hop_remaining"]),
            "active": bool(carry.get("active", False)),
        })
    return result


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
    for frame, decision in zip(sample["history"], history_sources):
        flows, carrying = _sample_rows(decision.get("logical_flow_rows", []), decision.get("carrying_rows", []), input_flow_index, input_nodes, input_tasks, target=False)
        frame["logical_flows"] = flows
        frame["carrying_states"] = carrying
        frame["logical_flow_source"] = "O_t.logical_flow_rows_and_carrying_rows"
        history_digests[str(frame["frame_index"])] = _digest(_semantic_payload(decision.get("logical_flow_rows", []), decision.get("carrying_rows", [])))

    target_digests: dict[str, str] = {}
    for target_row, decision, source_frame in zip(sample["target"], target_sources, target_source_frames):
        flows, carrying = _sample_rows(decision.get("logical_flow_rows", []), decision.get("carrying_rows", []), target_flow_index, target_nodes, target_tasks, target=True)
        target_row["logical_flows"] = flows
        target_row["carrying_states"] = carrying
        target_row["logical_flow_source_decision_frame"] = source_frame
        target_digests[str(target_row["frame_index"])] = _digest(_semantic_payload(decision.get("logical_flow_rows", []), decision.get("carrying_rows", [])))

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
    checks = validate_flow_sample_checks(sample)
    if not checks["passed"]:
        raise ValueError(f"Flow sample validation failed: {checks}")
    return sample


def _frame_digest(frame: Mapping[str, Any]) -> str:
    return _digest(_semantic_payload(frame.get("logical_flows", []), frame.get("carrying_states", [])))


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
                if not row.get("known"):
                    continue
                for name, field in (("logical.total_data", "total_data"), ("logical.e2e_delivered", "e2e_delivered"), ("logical.e2e_remaining", "e2e_remaining")):
                    if row.get("feature_mask", {}).get(field) and row.get(field) is not None:
                        yield name, float(row[field])
                carry = carrying.get(row["flow_id"])
                if carry is None:
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
            "mask_policy": "known=true and feature_mask=true; dev_train only; validation/test/target/padding excluded",
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

    for bi, sample in enumerate(samples):
        for hi, frame in enumerate(sample["history"]):
            fill(frame, bi, hi, "", F)
            carrying_by_id = {row["flow_id"]: row for row in frame.get("carrying_states", []) if row.get("known")}
            for row in frame.get("logical_flows", []):
                if not row.get("known"):
                    continue
                slot = int(row["flow_index"])
                carry = carrying_by_id[row["flow_id"]]
                out["carrying_known_mask"][bi, hi, slot] = True
                out["carrying_active"][bi, hi, slot] = bool(carry["active"])
                out["carrying_route_revision"][bi, hi, slot] = int(carry["route_revision"])
                out["carrying_current_hop_index"][bi, hi, slot] = int(carry["current_hop_index"])
                for key, field in (("carrying_holder_index", "current_holder_index"), ("carrying_hop_source_index", "hop_source_index"), ("carrying_hop_destination_index", "hop_destination_index")):
                    if carry.get(field) is not None:
                        out[key][bi, hi, slot] = int(carry[field])
                for fi, field in enumerate(("hop_progress", "hop_remaining")):
                    if carry["feature_mask"].get(field) and carry.get(field) is not None:
                        out["carrying_raw_features"][bi, hi, slot, fi] = float(carry[field])
                        out["carrying_features"][bi, hi, slot, fi] = float(carry.get("normalized_numeric", {}).get(field, carry[field]))
                        out["carrying_feature_mask"][bi, hi, slot, fi] = True
                for qi, node_slot in enumerate(carry.get("route_node_indices", [])):
                    out["route_node_indices"][bi, hi, slot, qi] = int(node_slot)
                    out["route_node_mask"][bi, hi, slot, qi] = True
        for li, frame in enumerate(sample["target"]):
            fill(frame, bi, li, "target_", TF)

    out["sample_static"] = copy.deepcopy([sample["static"] for sample in samples])
    out["sample_metadata"] = copy.deepcopy([sample["metadata"] for sample in samples])
    out["flow_normalization_stats"] = copy.deepcopy(dict(stats))
    checks = validate_flow_tensor_checks(out)
    if not checks["passed"]:
        raise ValueError(f"Flow tensor validation failed: {checks}")
    return out


def validate_flow_tensor_checks(tensor: Mapping[str, Any]) -> dict[str, bool]:
    contract = dict(tensor.get("contract", {}))
    known = np.asarray(tensor.get("logical_flow_known_mask"), dtype=bool)
    presence = np.asarray(tensor.get("logical_flow_presence"), dtype=bool)
    indices = np.asarray(tensor.get("logical_flow_index"))
    feature_mask = np.asarray(tensor.get("logical_flow_feature_mask"), dtype=bool)
    features = np.asarray(tensor.get("logical_flow_features"))
    identity = True
    for bi, static in enumerate(tensor.get("sample_static", [])):
        mapping = static.get("input_entity_index", {}).get("logical_flow", {})
        for _, slot in mapping.items():
            active_frames = known[bi, :, int(slot)]
            identity = identity and bool(np.all(indices[bi, :, int(slot)][active_frames] == int(slot)))
    checks = {
        "schema_version": tensor.get("schema_version") == TENSOR_SCHEMA_VERSION and contract.get("sample_contract_version") == SAMPLE_SCHEMA_VERSION,
        "separate_logical_flow_namespace": contract.get("stateful_flow_policy") == "STEP_4.2C_C_LOGICAL_FLOW_SEPARATE_FROM_LEGACY_HOP_SERVICE" and "past_outcome_flow_service" in tensor,
        "flow_id_tensor_index_round_trip": bool(identity),
        "known_presence_independent": bool(np.all(~presence | known)),
        "masked_placeholder_zero": bool(np.all(features[~feature_mask] == 0) and np.all(np.asarray(tensor["carrying_features"])[~np.asarray(tensor["carrying_feature_mask"], dtype=bool)] == 0)),
        "categorical_not_normalized": all(name not in tensor.get("flow_normalization_stats", {}).get("features", {}) for name in ("epoch", "flow_index", "flow_type", "status", "presence", "route_revision")),
        "endpoint_indices_bounded": bool(
            np.all(
                (np.asarray(tensor["logical_flow_source_index"])[known] >= 0)
                & (np.asarray(tensor["logical_flow_source_index"])[known] < int(contract["max_entity"]))
            )
            and np.all(
                (np.asarray(tensor["logical_flow_destination_index"])[known] >= 0)
                & (np.asarray(tensor["logical_flow_destination_index"])[known] < int(contract["max_entity"]))
            )
        ),
        "task_indices_bounded": bool(np.all((np.asarray(tensor["logical_flow_task_index"])[known] >= 0) & (np.asarray(tensor["logical_flow_task_index"])[known] < int(contract["max_task"])))),
        "no_silent_truncation": known.shape[2] == int(contract["max_logical_flow"]) and np.asarray(tensor["target_logical_flow_known_mask"]).shape[2] == int(contract["max_target_logical_flow"]) and np.asarray(tensor["route_node_mask"]).shape[3] == int(contract["max_route_nodes"]),
        "graph_builder_false": contract.get("graph_builder") is False,
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def sample_tensor_semantic_equality(samples: Sequence[Mapping[str, Any]], tensor: Mapping[str, Any]) -> bool:
    for bi, sample in enumerate(samples):
        for hi, frame in enumerate(sample["history"]):
            for row in frame.get("logical_flows", []):
                if not row.get("known"):
                    continue
                slot = int(row["flow_index"])
                if not tensor["logical_flow_known_mask"][bi, hi, slot]:
                    return False
                expected = [float(row[name]) for name in ("total_data", "e2e_delivered", "e2e_remaining")]
                if not np.allclose(tensor["logical_flow_raw_features"][bi, hi, slot], expected):
                    return False
                if int(tensor["logical_flow_epoch"][bi, hi, slot]) != int(row["epoch"]):
                    return False
    return True


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
