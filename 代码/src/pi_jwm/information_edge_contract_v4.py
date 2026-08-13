"""Truthful field and validation contract for PI-JWM information edges v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from typing import Any

import numpy as np


CONTRACT_VERSION = "PIJWM-DG-Contract-v4"


class MissingReason(IntEnum):
    NONE = 0
    NOT_APPLICABLE = 1
    NOT_COLLECTED = 2
    NO_HISTORY = 3
    SOURCE_ABSENT = 4
    SOURCE_ERROR = 5
    INVALID_VALUE = 6
    TIME_MISMATCH = 7
    SHAPE_MISMATCH = 8


@dataclass(frozen=True)
class FieldSpec:
    name: str
    namespace: str
    tier: str
    dtype: str
    shape: tuple[str, ...]
    unit: str
    temporal_role: str
    provenance_level: str
    applicable_edge_classes: tuple[str, ...]
    source_requirement: str
    valid_min: float | None = None
    valid_max: float | None = None
    dependency_formula: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["applicable_edge_classes"] = list(self.applicable_edge_classes)
        return value


def _field_specs() -> tuple[FieldSpec, ...]:
    both = ("wireless", "wired")
    wireless = ("wireless",)
    specs = (
        FieldSpec("structure.edge_present", "structure", "E0", "bool", ("time", "edge"), "boolean", "action_pre", "derived", both, "stable endpoints and edge type"),
        FieldSpec("structure.edge_type", "structure", "E0", "int16", ("edge",), "category", "static_or_action_pre", "derived", both, "audited endpoint-type mapping"),
        FieldSpec("structure.endpoint_index", "structure", "E0", "int64", ("edge", "two"), "index", "static", "derived", both, "stable directed endpoint vocabulary"),
        FieldSpec("structure.cep_physical_edge_index", "structure", "E0", "int64", ("edge",), "index", "static", "derived", both, "same-endpoint directed physical edge"),
        FieldSpec("pre_link.channel_attenuation_mean_db", "pre_link", "E1", "float32", ("time", "edge"), "dB", "action_pre", "derived", wireless, "direct per-RB attenuation snapshot before action"),
        FieldSpec("pre_link.channel_attenuation_std_db", "pre_link", "E1", "float32", ("time", "edge"), "dB", "action_pre", "derived", wireless, "same snapshot and RB identity as attenuation mean"),
        FieldSpec("pre_link.prev_active_flow_count", "pre_link", "E1", "float32", ("time", "edge"), "flow", "prior_outcome", "derived", both, "prior-slot direct transfer events"),
        FieldSpec("pre_link.prev_effective_rate_per_s", "pre_link", "E1", "float32", ("time", "edge"), "AirFogSim data-unit/s", "prior_outcome", "derived", both, "prior-slot explicitly assigned resource rates with calibrated semantics"),
        FieldSpec("pre_link.prev_served_data", "pre_link", "E1", "float32", ("time", "edge"), "AirFogSim data-unit", "prior_outcome", "derived", both, "prior-slot delivered_data events"),
        FieldSpec("pre_rb_optional.prev_sinr_db", "pre_rb_optional", "E2", "float32", ("time", "edge", "resource"), "dB", "prior_outcome", "derived", wireless, "prior activated-RB simulator outcome"),
        FieldSpec("pre_rb_optional.prev_interference_plus_noise_mw", "pre_rb_optional", "E2", "float32", ("time", "edge", "resource"), "mW", "prior_outcome", "derived", wireless, "prior activated-RB interference including noise"),
        FieldSpec("pre_rb_optional.prev_rate_per_s", "pre_rb_optional", "E2", "float32", ("time", "edge", "resource"), "AirFogSim data-unit/s", "prior_outcome", "derived", wireless, "prior assigned-RB rate with calibrated semantics"),
        FieldSpec("pre_rb_optional.prev_outage", "pre_rb_optional", "E2", "bool", ("time", "edge", "resource"), "boolean", "prior_outcome", "derived", wireless, "prior activated-RB outage"),
        FieldSpec("pre_rb_optional.channel_attenuation_db", "pre_rb_optional", "E3", "float32", ("time", "edge", "resource"), "dB", "action_pre", "direct", wireless, "current action-pre per-RB attenuation"),
        FieldSpec("action.assignment_coo", "action", "E0", "int64", ("record", "four"), "index", "current_action", "direct", both, "unique (time, flow, edge, resource) assignments"),
        FieldSpec("action.offload_target_agent_index", "action", "E0", "int64", ("time", "task"), "index", "current_action", "direct", both, "task-level offload action"),
        FieldSpec("action.offload_action_present", "action", "E0", "bool", ("time", "task"), "boolean", "current_action", "direct", both, "offload action record presence"),
        FieldSpec("outcome_link.active_flow_count", "outcome_link", "supervision", "float32", ("time", "edge"), "flow", "current_outcome", "derived", both, "current direct transfer events"),
        FieldSpec("outcome_link.effective_rate_per_s", "outcome_link", "supervision", "float32", ("time", "edge"), "AirFogSim data-unit/s", "current_outcome", "derived", both, "current explicitly assigned resource rates with calibrated semantics"),
        FieldSpec("outcome_link.served_data", "outcome_link", "supervision", "float32", ("time", "edge"), "AirFogSim data-unit", "current_outcome", "derived", both, "current delivered_data events"),
        FieldSpec("outcome_rb_optional.sinr_db", "outcome_rb_optional", "optional_supervision", "float32", ("time", "edge", "resource"), "dB", "current_outcome", "direct", wireless, "current activated-RB simulator outcome"),
        FieldSpec("outcome_rb_optional.interference_plus_noise_mw", "outcome_rb_optional", "optional_supervision", "float32", ("time", "edge", "resource"), "mW", "current_outcome", "direct", wireless, "current activated-RB interference including noise"),
        FieldSpec("outcome_rb_optional.rate_per_s", "outcome_rb_optional", "optional_supervision", "float32", ("time", "edge", "resource"), "AirFogSim data-unit/s", "current_outcome", "direct", wireless, "current assigned-RB rate with calibrated semantics"),
        FieldSpec("outcome_rb_optional.outage", "outcome_rb_optional", "optional_supervision", "bool", ("time", "edge", "resource"), "boolean", "current_outcome", "direct", wireless, "current activated-RB outage"),
        FieldSpec("config.noise_power_dbm", "config", "metadata", "float64", (), "dBm", "static", "fixed_config", both, "simulator noise configuration"),
        FieldSpec("config.rb_bandwidth_mhz", "config", "metadata", "float64", (), "MHz", "static", "fixed_config", wireless, "simulator resource-block bandwidth configuration"),
        FieldSpec("config.n_rb", "config", "metadata", "int64", (), "resource", "static", "fixed_config", wireless, "simulator resource-block count"),
        FieldSpec("config.tx_power_dbm", "config", "metadata", "float64", ("edge_type",), "dBm", "static", "fixed_config", wireless, "link-type transmit-power configuration"),
        FieldSpec("excluded.mcs", "excluded", "unavailable", "float32", (), "unavailable", "unavailable", "unavailable", wireless, "no direct source or executable mechanism"),
    )
    nonnegative = {
        "pre_link.prev_active_flow_count",
        "pre_link.prev_effective_rate_per_s",
        "pre_link.prev_served_data",
        "pre_rb_optional.prev_interference_plus_noise_mw",
        "pre_rb_optional.prev_rate_per_s",
        "outcome_link.active_flow_count",
        "outcome_link.effective_rate_per_s",
        "outcome_link.served_data",
        "outcome_rb_optional.interference_plus_noise_mw",
        "outcome_rb_optional.rate_per_s",
        "config.rb_bandwidth_mhz",
        "config.n_rb",
    }
    booleans = {
        "structure.edge_present",
        "action.offload_action_present",
        "pre_rb_optional.prev_outage",
        "outcome_rb_optional.outage",
    }
    formulas = {
        "action.assignment_coo": "unique (time, flow, edge, resource) records; dense assignment is binary",
        "outcome_link.active_flow_count": "count unique active assigned flows per (time, edge)",
        "outcome_link.effective_rate_per_s": "sum outcome_rb_optional.rate_per_s over assigned RBs for each (time, edge)",
        "outcome_link.served_data": "min(remaining_before, effective_rate_per_s * slot_seconds) under calibrated transfer semantics",
        "outcome_rb_optional.interference_plus_noise_mw": "must be >= config noise power converted to mW",
        "outcome_rb_optional.outage": "outage implies outcome_rb_optional.rate_per_s == 0",
    }
    result: list[FieldSpec] = []
    for spec in specs:
        minimum = 1.0 if spec.name == "config.n_rb" else 0.0 if spec.name in nonnegative or spec.name in booleans else None
        maximum = 1.0 if spec.name in booleans else None
        result.append(
            replace(
                spec,
                valid_min=minimum,
                valid_max=maximum,
                dependency_formula=formulas.get(spec.name),
            )
        )
    return tuple(result)


def build_field_registry() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in _field_specs()]


_LEGACY_MAPPING = (
    ("pre.interface_available", "delete_continuous_feature", "structure.edge_present", "hardcoded_constant", "express through structure and feasibility"),
    ("pre.csi_mean", "rename_after_semantic_check", "pre_link.channel_attenuation_mean_db", "legacy_observed", "legacy value is attenuation mean, not SINR"),
    ("pre.channel_gain", "do_not_fill", "", "unobserved", "would duplicate or contradict attenuation semantics"),
    ("pre.path_loss", "do_not_fill", "", "unobserved", "current source includes fast fading and is not pure path loss"),
    ("pre.noise", "move_to_config", "config.noise_power_dbm", "fixed_config", "not a dynamic edge observation"),
    ("pre.historical_interference", "new_optional_collection", "pre_rb_optional.prev_interference_plus_noise_mw", "unobserved", "requires prior-slot per-RB collection"),
    ("pre.historical_sinr", "new_optional_collection", "pre_rb_optional.prev_sinr_db", "unobserved", "requires prior-slot per-RB collection"),
    ("pre.historical_rate", "new_collection", "pre_link.prev_effective_rate_per_s", "unobserved", "requires assigned-RB rate and unit calibration"),
    ("action.allocated_rb_count", "derive_from_coo", "action.assignment_coo", "legacy_observed_aggregate", "count loses flow, edge, and RB identity"),
    ("action.tx_power", "move_to_config", "config.tx_power_dbm", "fixed_config", "not a current decision variable"),
    ("action.mcs", "unavailable", "", "unavailable", "no source or executable mechanism"),
    ("outcome.active_task_count", "rename_and_recompute", "outcome_link.active_flow_count", "legacy_observed", "must count flows from direct events"),
    ("outcome.rate_sum", "regenerate_from_assigned_rb", "outcome_link.effective_rate_per_s", "legacy_semantics_conflicted", "full-RB snapshots and assigned-RB events disagree"),
    ("outcome.actual_interference", "new_optional_collection", "outcome_rb_optional.interference_plus_noise_mw", "unobserved", "simulator quantity includes noise"),
    ("outcome.actual_sinr", "new_optional_collection", "outcome_rb_optional.sinr_db", "unobserved", "requires activated-RB collection"),
    ("outcome.outage", "new_optional_collection", "outcome_rb_optional.outage", "unobserved", "requires activated-RB collection"),
    ("outcome.throughput", "delete_ambiguous_field", "", "unobserved", "overlaps capacity and delivered data"),
    ("outcome.served_data", "new_collection", "outcome_link.served_data", "unobserved_in_v3_tensor", "derive from direct delivered_data events"),
)


def build_legacy_slot_mapping() -> list[dict[str, Any]]:
    return [
        {
            "legacy_index": index,
            "legacy_slot": slot,
            "decision": decision,
            "v4_target": target,
            "source_status": source_status,
            "reason": reason,
        }
        for index, (slot, decision, target, source_status, reason) in enumerate(
            _LEGACY_MAPPING
        )
    ]


def validate_masked_field(
    values: np.ndarray,
    valid_mask: np.ndarray,
    missing_reason: np.ndarray,
) -> None:
    values = np.asarray(values)
    valid_mask = np.asarray(valid_mask)
    missing_reason = np.asarray(missing_reason)
    if valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must have bool dtype")
    if missing_reason.dtype.kind not in "iu":
        raise ValueError("missing_reason must have integer dtype")
    if values.shape != valid_mask.shape or values.shape != missing_reason.shape:
        raise ValueError("value, valid mask, and missing reason shapes differ")
    if not np.isfinite(values).all():
        raise ValueError("field contains NaN or Inf")
    valid_reason_values = {reason.value for reason in MissingReason}
    if any(int(value) not in valid_reason_values for value in missing_reason.flat):
        raise ValueError("unknown missing reason")
    if np.any(valid_mask & (missing_reason != MissingReason.NONE.value)):
        raise ValueError("valid element must use missing_reason=none")
    if np.any((~valid_mask) & (missing_reason == MissingReason.NONE.value)):
        raise ValueError("invalid element requires a non-none missing reason")
    if np.any((~valid_mask) & (values != 0)):
        raise ValueError("invalid element must use zero fill")


def validate_edge_applicability(
    field_name: str,
    edge_class: str,
    valid_mask: np.ndarray,
    missing_reason: np.ndarray,
) -> None:
    registry = {row["name"]: row for row in build_field_registry()}
    if field_name not in registry:
        raise ValueError(f"unknown field: {field_name}")
    if edge_class not in {"wireless", "wired"}:
        raise ValueError(f"unknown edge class: {edge_class}")
    valid_mask = np.asarray(valid_mask)
    missing_reason = np.asarray(missing_reason)
    if valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must have bool dtype")
    if missing_reason.dtype.kind not in "iu":
        raise ValueError("missing_reason must have integer dtype")
    if valid_mask.shape != missing_reason.shape:
        raise ValueError("valid mask and missing reason shapes differ")
    applicable = edge_class in registry[field_name]["applicable_edge_classes"]
    if not applicable:
        if np.any(valid_mask) or np.any(
            missing_reason != MissingReason.NOT_APPLICABLE.value
        ):
            raise ValueError(
                "non-applicable field must be invalid with not_applicable reason"
            )
        return
    if np.any(
        (~valid_mask) & (missing_reason == MissingReason.NOT_APPLICABLE.value)
    ):
        raise ValueError("applicable field cannot hide missing data as not_applicable")


def validate_assignment_coo(
    assignment_coo: np.ndarray,
    capacities: tuple[int, int, int, int],
) -> None:
    coo = np.asarray(assignment_coo)
    if coo.dtype.kind not in "iu" or coo.ndim != 2 or coo.shape[1] != 4:
        raise ValueError("assignment_coo must be integer [record,4]")
    if len(capacities) != 4:
        raise ValueError("capacities must contain time, flow, edge, and resource")
    if len({tuple(int(value) for value in row) for row in coo.tolist()}) != len(
        coo
    ):
        raise ValueError("duplicate assignment")
    labels = ("time", "flow", "edge", "resource")
    for column, (label, capacity) in enumerate(
        zip(labels, capacities, strict=True)
    ):
        if capacity < 0:
            raise ValueError(f"{label} capacity must be non-negative")
        if np.any(coo[:, column] < 0) or np.any(coo[:, column] >= capacity):
            raise ValueError(f"{label} index out of range")


def assignment_coo_to_dense(
    assignment_coo: np.ndarray,
    capacities: tuple[int, int, int, int],
) -> np.ndarray:
    validate_assignment_coo(assignment_coo, capacities)
    dense = np.zeros(capacities, dtype=bool)
    coo = np.asarray(assignment_coo)
    if len(coo):
        dense[tuple(coo[:, index] for index in range(4))] = True
    return dense


def validate_field_values(
    field_name: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    missing_reason: np.ndarray,
) -> None:
    registry = {row["name"]: row for row in build_field_registry()}
    if field_name not in registry:
        raise ValueError(f"unknown field: {field_name}")
    validate_masked_field(values, valid_mask, missing_reason)
    values = np.asarray(values)
    valid_mask = np.asarray(valid_mask)
    valid_values = values[valid_mask]
    spec = registry[field_name]
    if spec["valid_min"] is not None and np.any(valid_values < spec["valid_min"]):
        raise ValueError(f"{field_name} contains a value below valid_min")
    if spec["valid_max"] is not None and np.any(valid_values > spec["valid_max"]):
        raise ValueError(f"{field_name} contains a value above valid_max")


def validate_prev_field_timing(
    valid_mask: np.ndarray,
    missing_reason: np.ndarray,
) -> None:
    valid_mask = np.asarray(valid_mask)
    missing_reason = np.asarray(missing_reason)
    if valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must have bool dtype")
    if missing_reason.dtype.kind not in "iu":
        raise ValueError("missing_reason must have integer dtype")
    if valid_mask.shape != missing_reason.shape or valid_mask.ndim < 1:
        raise ValueError("previous-outcome mask and reason shapes differ")
    if np.any(valid_mask[0]) or np.any(
        missing_reason[0] != MissingReason.NO_HISTORY.value
    ):
        raise ValueError("previous-outcome first frame must be invalid with no_history")


def validate_link_outcome(
    *,
    effective_rate_per_s: np.ndarray,
    served_data: np.ndarray,
    slot_seconds: float,
    remaining_before: np.ndarray,
    assigned_rate_by_rb: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    rate = np.asarray(effective_rate_per_s)
    served = np.asarray(served_data)
    remaining = np.asarray(remaining_before)
    by_rb = np.asarray(assigned_rate_by_rb)
    if rate.shape != served.shape or rate.shape != remaining.shape:
        raise ValueError("link outcome shapes differ")
    if by_rb.ndim != rate.ndim + 1 or by_rb.shape[:-1] != rate.shape:
        raise ValueError("assigned-RB rate shape differs from link outcome")
    if not np.isfinite(slot_seconds) or slot_seconds <= 0:
        raise ValueError("slot_seconds must be finite and positive")
    if not all(np.isfinite(value).all() for value in (rate, served, remaining, by_rb)):
        raise ValueError("link outcome contains NaN or Inf")
    if any(np.any(value < 0) for value in (rate, served, remaining, by_rb)):
        raise ValueError("link outcome values must be non-negative")
    assigned_sum = by_rb.sum(axis=-1)
    if not np.allclose(rate, assigned_sum, rtol=rtol, atol=atol):
        raise ValueError("effective rate differs from assigned-RB sum")
    tolerance = atol + rtol * np.maximum(np.abs(rate * slot_seconds), 1.0)
    if np.any(served > rate * slot_seconds + tolerance):
        raise ValueError("served data exceeds rate times slot")
    if np.any(served > remaining + tolerance):
        raise ValueError("served data exceeds remaining-before data")


def validate_rb_outcome(
    *,
    rate_per_s: np.ndarray,
    outage: np.ndarray,
    interference_plus_noise_mw: np.ndarray,
    noise_power_mw: float,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    rate = np.asarray(rate_per_s)
    outage = np.asarray(outage)
    interference = np.asarray(interference_plus_noise_mw)
    if outage.dtype != np.bool_:
        raise ValueError("outage must have bool dtype")
    if rate.shape != outage.shape or rate.shape != interference.shape:
        raise ValueError("RB outcome shapes differ")
    if not np.isfinite(noise_power_mw) or noise_power_mw < 0:
        raise ValueError("noise power must be finite and non-negative")
    if not np.isfinite(rate).all() or not np.isfinite(interference).all():
        raise ValueError("RB outcome contains NaN or Inf")
    if np.any(rate < 0) or np.any(interference < 0):
        raise ValueError("RB outcome values must be non-negative")
    if np.any(outage & (~np.isclose(rate, 0.0, rtol=rtol, atol=atol))):
        raise ValueError("outage RB must have zero rate")
    if np.any(interference + atol < noise_power_mw):
        raise ValueError("interference plus noise is below noise power")
