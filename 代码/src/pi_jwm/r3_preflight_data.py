"""Leak-safe dynamic views and explicit batches for the PI-JWM R3 CPU preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .airfogsim_teacher_tensor_v3 import (
    INFORMATION_EDGE_FEATURES,
    INFORMATION_NODE_FEATURES,
    PHYSICAL_EDGE_FEATURES,
    PHYSICAL_NODE_FEATURES,
)
from .formal_airfogsim_graph_v1 import FORMAL_DAG_STATE_FEATURES


R3_VIEW_SCHEMA = "PIJWM-R3-Smoke-View-v1"

ACTION_KEYS = (
    "task_action",
    "task_action_present",
    "task_action_information_node_index",
)

TEMPORAL_STATE_KEYS = (
    "physical_node_state",
    "physical_node_feature_mask",
    "physical_node_present",
    "physical_edge_state",
    "physical_edge_feature_mask",
    "physical_edge_present",
    "information_node_state",
    "information_node_feature_mask",
    "information_node_present",
    "information_edge_state",
    "information_edge_feature_mask",
    "information_edge_present",
    "data_flow_state",
    "data_flow_present",
    "data_flow_completed",
    "task_state",
    "task_present",
    "task_lifecycle_index",
    "task_information_node_index",
    "task_dag_state",
    "task_dag_state_present",
    "dag_edge_present",
    "cfl_mask",
)

CONTINUOUS_STATE_KEYS = (
    "physical_node_state",
    "physical_edge_state",
    "information_node_state",
    "information_edge_state",
    "data_flow_state",
    "task_state",
    "task_dag_state",
)

FEATURE_MASK_BY_STATE = {
    "physical_node_state": "physical_node_feature_mask",
    "physical_edge_state": "physical_edge_feature_mask",
    "information_node_state": "information_node_feature_mask",
    "information_edge_state": "information_edge_feature_mask",
}

PRESENCE_BY_STATE = {
    "physical_node_state": "physical_node_present",
    "physical_edge_state": "physical_edge_present",
    "information_node_state": "information_node_present",
    "information_edge_state": "information_edge_present",
    "data_flow_state": "data_flow_present",
    "task_state": "task_present",
    "task_dag_state": "task_dag_state_present",
}

STATIC_VALID_BY_STATE = {
    "data_flow_state": "data_flow_valid",
    "task_state": "task_valid",
    "task_dag_state": "task_valid",
}


@dataclass(frozen=True)
class R3Window:
    trajectory_id: str
    environment_seed: int
    split: str
    tensor_path: Path
    history_start: int
    history_end: int
    target_start: int
    target_end: int
    horizon_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": R3_VIEW_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "environment_seed": self.environment_seed,
            "split": self.split,
            "tensor_path": str(self.tensor_path),
            "history_start": self.history_start,
            "history_end": self.history_end,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "horizon_steps": self.horizon_steps,
        }


@dataclass
class ExplicitStateBatch:
    history: dict[str, torch.Tensor]
    history_action: dict[str, torch.Tensor]
    future_action: dict[str, torch.Tensor]
    target: dict[str, torch.Tensor]
    static: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_entry(root: Path, manifest: Mapping[str, Any], name: str) -> str:
    entry = manifest.get("files", {}).get(name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest does not bind required input: {name}")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"bound R3 input does not exist: {path}")
    actual = _sha256(path)
    if actual != entry.get("sha256") or path.stat().st_size != int(
        entry.get("size_bytes", -1)
    ):
        raise ValueError(f"R3 input hash or size mismatch: {name}")
    return actual


def verify_r3_inputs(
    dataset_root: str | Path,
    evaluation_root: str | Path,
) -> dict[str, Any]:
    """Bind R3 to the frozen R1 contract and ready, nonlocked R2 protocol."""

    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    dataset_manifest = _read_json(dataset_root / "manifest.json")
    evaluation_manifest = _read_json(evaluation_root / "manifest.json")
    if not dataset_manifest.get("teacher_aligned_graph_tensor_ready"):
        raise ValueError("R1 teacher-aligned graph tensor is not ready")
    if not evaluation_manifest.get("evaluation_protocol_ready"):
        raise ValueError("R2 evaluation protocol is not ready")

    dataset_hashes = {
        name: _verify_manifest_entry(dataset_root, dataset_manifest, name)
        for name in (
            "tensor_contract.json",
            "protocol.json",
            "trajectory_index.csv",
            "normalization_stats.json",
            "validation_report.json",
            "locked_test_integrity.json",
        )
    }
    evaluation_hashes = {
        name: _verify_manifest_entry(evaluation_root, evaluation_manifest, name)
        for name in (
            "evaluation_normalization_stats.json",
            "metric_registry.json",
            "fair_experiment_protocol.json",
            "validation_report.json",
        )
    }
    validation = _read_json(evaluation_root / "validation_report.json")
    if not validation.get("evaluation_protocol_ready"):
        raise ValueError("R2 validation report is not ready")

    contract = _read_json(dataset_root / "tensor_contract.json")
    if contract.get("schema_version") != "PIJWM-DG-Contract-v3-tensor":
        raise ValueError("R1 tensor contract schema is incompatible")
    expected_features = {
        "physical_node_features": list(PHYSICAL_NODE_FEATURES),
        "physical_edge_features": list(PHYSICAL_EDGE_FEATURES),
        "information_node_features": list(INFORMATION_NODE_FEATURES),
        "information_edge_features": list(INFORMATION_EDGE_FEATURES),
    }
    for key, expected in expected_features.items():
        if contract.get(key) != expected:
            raise ValueError(f"R1 tensor contract feature order mismatch: {key}")
    if contract.get("cross_graph_relations") != ["CIP", "CEP"]:
        raise ValueError("R1 cross-graph relations must be CIP and CEP")
    if contract.get("business_relations") != ["CFL"]:
        raise ValueError("R1 business relation must be CFL")

    rows = read_trajectory_index(dataset_root)
    locked_rows = [row for row in rows if row.get("split") == "locked_test"]
    if not locked_rows or any(
        row.get("v3_status") != "locked_integrity_only"
        or bool(str(row.get("v3_seed_dir", "")).strip())
        or (dataset_root / f"seed_{int(row['seed']):03d}").exists()
        for row in locked_rows
    ):
        raise ValueError("locked-test trajectories are not sealed by the R1 index")

    normalization = _read_json(
        evaluation_root / "evaluation_normalization_stats.json"
    )
    if normalization.get("source_split") != "train":
        raise ValueError("R2 normalization is not train-only")
    dag_stats = normalization.get("features", {}).get("task_dag_state", {})
    if dag_stats.get("feature_names") != list(FORMAL_DAG_STATE_FEATURES):
        raise ValueError("R2 DAG normalization order disagrees with the R1 contract")

    source_files = (
        Path(__file__),
        Path(__file__).with_name("r3_world_model.py"),
        Path(__file__).with_name("r3_objective.py"),
        Path(__file__).with_name("r3_checkpoint.py"),
    )
    source_digest = hashlib.sha256()
    source_hashes = {}
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"R3 source binding is missing: {path.name}")
        source_hashes[path.name] = _sha256(path)
        source_digest.update(path.name.encode("utf-8"))
        source_digest.update(source_hashes[path.name].encode("ascii"))
    bindings = {
        "tensor_contract_sha256": dataset_hashes["tensor_contract.json"],
        "dataset_protocol_sha256": dataset_hashes["protocol.json"],
        "normalization_sha256": evaluation_hashes[
            "evaluation_normalization_stats.json"
        ],
        "metric_registry_sha256": evaluation_hashes["metric_registry.json"],
        "source_code_sha256": source_digest.hexdigest(),
    }
    return {
        "ready": True,
        "locked_test_accessed": False,
        "dataset_root": str(dataset_root),
        "evaluation_root": str(evaluation_root),
        "verified_dataset_inputs": dataset_hashes,
        "verified_evaluation_inputs": evaluation_hashes,
        "source_code_files": source_hashes,
        "bindings": bindings,
        "dag_feature_count": len(FORMAL_DAG_STATE_FEATURES),
        "information_edge_feature_count": len(INFORMATION_EDGE_FEATURES),
    }


def _stable_start(
    trajectory_id: str, horizon_steps: int, maximum_start: int, seed: int
) -> int:
    if maximum_start < 0:
        raise ValueError("trajectory is too short for the requested R3 window")
    digest = hashlib.sha256(
        f"{int(seed)}::{trajectory_id}::{int(horizon_steps)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (maximum_start + 1)


def select_r3_windows(
    dataset_root: str | Path,
    index_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    horizons: Iterable[int] = (1, 5, 20),
    history_steps: int = 8,
    per_horizon: int = 1,
    seed: int = 20260804,
) -> list[R3Window]:
    """Select deterministic windows without changing the frozen R1 window contract."""

    split = str(split)
    if split == "locked_test":
        raise ValueError("locked_test cannot be used by the R3 CPU preflight")
    if split not in {"train", "validation", "calibration"}:
        raise ValueError(f"unsupported R3 split: {split}")
    history_steps = int(history_steps)
    per_horizon = int(per_horizon)
    requested_horizons = tuple(int(value) for value in horizons)
    if history_steps <= 0 or per_horizon <= 0 or not requested_horizons:
        raise ValueError("history_steps, per_horizon and horizons must be positive")
    if any(value <= 0 for value in requested_horizons):
        raise ValueError("rollout horizons must be positive")

    dataset_root = Path(dataset_root).resolve()
    eligible = [
        dict(row)
        for row in index_rows
        if str(row.get("split")) == split
        and str(row.get("v3_status")) == "materialized"
    ]
    if not eligible:
        raise ValueError(f"no materialized R3 trajectories for split {split}")

    selected: list[R3Window] = []
    for horizon in requested_horizons:
        candidates = [
            row
            for row in eligible
            if int(row.get("observed_steps", 0)) >= history_steps + horizon
        ]
        if len(candidates) < per_horizon:
            raise ValueError(
                f"split {split} has fewer than {per_horizon} trajectories for horizon {horizon}"
            )
        candidates.sort(key=lambda row: (int(row["seed"]), str(row["trajectory_id"])))
        generator = random.Random(int(seed) + horizon)
        generator.shuffle(candidates)
        for row in candidates[:per_horizon]:
            tensor_path = dataset_root / str(row["v3_seed_dir"]) / "trajectory_tensors.npz"
            if not tensor_path.is_file():
                raise FileNotFoundError(f"R3 source tensor does not exist: {tensor_path}")
            observed_steps = int(row["observed_steps"])
            maximum_start = observed_steps - history_steps - horizon
            history_start = _stable_start(
                str(row["trajectory_id"]), horizon, maximum_start, seed
            )
            history_end = history_start + history_steps
            selected.append(
                R3Window(
                    trajectory_id=str(row["trajectory_id"]),
                    environment_seed=int(row["seed"]),
                    split=split,
                    tensor_path=tensor_path,
                    history_start=history_start,
                    history_end=history_end,
                    target_start=history_end,
                    target_end=history_end + horizon,
                    horizon_steps=horizon,
                )
            )
    return selected


def load_r3_window(window: R3Window) -> dict[str, Any]:
    """Load one verified nonlocked trajectory view with separate input namespaces."""

    if window.split == "locked_test":
        raise ValueError("locked_test cannot be loaded by R3")
    if not (
        0 <= window.history_start < window.history_end == window.target_start
        < window.target_end
        and window.target_end - window.target_start == window.horizon_steps
    ):
        raise ValueError("R3 window bounds are invalid")
    with np.load(window.tensor_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if "time" not in arrays:
        raise ValueError("R3 source tensor is missing time")
    time_count = int(len(arrays["time"]))
    if window.target_end > time_count:
        raise ValueError("R3 target window exceeds its trajectory")

    history: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    history_action: dict[str, np.ndarray] = {}
    future_action: dict[str, np.ndarray] = {}
    for key in TEMPORAL_STATE_KEYS:
        if key not in arrays:
            continue
        if arrays[key].shape[0] != time_count:
            raise ValueError(f"temporal R3 tensor {key} is not time aligned")
        history[key] = arrays[key][window.history_start : window.history_end].copy()
        target[key] = arrays[key][window.target_start : window.target_end].copy()
    for key in ACTION_KEYS:
        if key not in arrays:
            continue
        if arrays[key].shape[0] != time_count:
            raise ValueError(f"action tensor {key} is not time aligned")
        history_action[key] = arrays[key][
            window.history_start : window.history_end
        ].copy()
        future_action[key] = arrays[key][window.target_start : window.target_end].copy()

    consumed = set(TEMPORAL_STATE_KEYS) | set(ACTION_KEYS) | {"time"}
    static = {key: value.copy() for key, value in arrays.items() if key not in consumed}
    return {
        "schema_version": R3_VIEW_SCHEMA,
        "window": window.to_dict(),
        "history": history,
        "history_action": history_action,
        "future_action": future_action,
        "target": target,
        "static": static,
        "history_time": arrays["time"][window.history_start : window.history_end].copy(),
        "target_time": arrays["time"][window.target_start : window.target_end].copy(),
    }


def _observed_mask(
    namespace: Mapping[str, np.ndarray],
    static: Mapping[str, np.ndarray],
    state_key: str,
) -> np.ndarray:
    value = np.asarray(namespace[state_key])
    feature_key = FEATURE_MASK_BY_STATE.get(state_key)
    presence_key = PRESENCE_BY_STATE[state_key]
    if presence_key not in namespace:
        raise ValueError(f"{state_key} is missing {presence_key}")
    presence = np.asarray(namespace[presence_key], dtype=bool)
    if presence.shape != value.shape[:-1]:
        raise ValueError(f"{presence_key} shape does not match {state_key}")
    observed = np.broadcast_to(presence[..., None], value.shape).copy()
    if feature_key is not None:
        if feature_key not in namespace:
            raise ValueError(f"{state_key} is missing {feature_key}")
        feature_mask = np.asarray(namespace[feature_key], dtype=bool)
        if feature_mask.shape != value.shape:
            raise ValueError(f"{feature_key} shape does not match {state_key}")
        observed &= feature_mask
    valid_key = STATIC_VALID_BY_STATE.get(state_key)
    if valid_key is not None:
        if valid_key not in static:
            raise ValueError(f"{state_key} is missing static {valid_key}")
        valid = np.asarray(static[valid_key], dtype=bool)
        if valid.shape != value.shape[1:-1]:
            raise ValueError(f"{valid_key} shape does not match {state_key}")
        observed &= np.broadcast_to(valid[None, ..., None], value.shape)
    return observed


def _normalize_namespace(
    namespace: Mapping[str, np.ndarray],
    static: Mapping[str, np.ndarray],
    normalization_stats: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in namespace.items()}
    features = normalization_stats.get("features", {})
    for state_key in CONTINUOUS_STATE_KEYS:
        if state_key not in result:
            continue
        value = np.asarray(result[state_key], dtype=np.float32)
        observed = _observed_mask(result, static, state_key)
        if np.any(value[~observed] != 0.0):
            raise ValueError(f"{state_key} contains nonzero masked missing values")
        stats = features.get(state_key)
        if not isinstance(stats, Mapping):
            raise ValueError(f"train-only normalization is missing {state_key}")
        mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        scale = np.asarray(stats.get("scale", []), dtype=np.float32)
        if mean.shape != (value.shape[-1],) or scale.shape != (value.shape[-1],):
            raise ValueError(f"normalization dimension mismatch for {state_key}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError(f"normalization contains non-finite values for {state_key}")
        if np.any(scale <= 0.0):
            raise ValueError(f"normalization scale must be positive for {state_key}")
        normalized = (value - mean) / scale
        normalized[~observed] = 0.0
        result[state_key] = normalized.astype(np.float32, copy=False)
    return result


def _add_information_link_activity(
    namespace: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze the activity label before continuous-state normalization."""

    result = {key: np.asarray(value).copy() for key, value in namespace.items()}
    required = (
        "information_edge_state",
        "information_edge_feature_mask",
        "information_edge_present",
    )
    if any(key not in result for key in required):
        return result
    active_index = INFORMATION_EDGE_FEATURES.index("outcome.active_task_count")
    state = np.asarray(result["information_edge_state"])
    feature_mask = np.asarray(result["information_edge_feature_mask"], dtype=bool)
    present = np.asarray(result["information_edge_present"], dtype=bool)
    if state.shape != feature_mask.shape or present.shape != state.shape[:-1]:
        raise ValueError("information-edge activity source shapes are inconsistent")
    result["information_link_activity"] = state[..., active_index] > 0.0
    result["information_link_activity_mask"] = (
        present & feature_mask[..., active_index]
    )
    return result


def _torch_namespace(
    namespace: Mapping[str, np.ndarray], device: str | torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(np.asarray(value).copy(), device=device).unsqueeze(0)
        for key, value in namespace.items()
    }


def _validate_index_range(name: str, value: np.ndarray, upper: int) -> None:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if upper <= 0 or np.any((array < -1) | (array >= upper)):
        raise ValueError(f"{name} contains an out-of-range index")


def _validate_r3_relations(
    history: Mapping[str, np.ndarray],
    history_action: Mapping[str, np.ndarray],
    future_action: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    static: Mapping[str, np.ndarray],
) -> None:
    """Reject malformed v3 relations instead of silently masking them."""

    counts = {
        "physical_node": int(np.asarray(history["physical_node_state"]).shape[1]),
        "physical_edge": int(np.asarray(history["physical_edge_state"]).shape[1]),
        "information_node": int(np.asarray(history["information_node_state"]).shape[1]),
        "information_edge": int(np.asarray(history["information_edge_state"]).shape[1]),
        "data_flow": int(np.asarray(history["data_flow_state"]).shape[1]),
        "task": int(np.asarray(history["task_state"]).shape[1]),
    }
    expected_shapes = {
        "physical_edge_endpoint_index": (counts["physical_edge"], 2),
        "information_edge_endpoint_index": (counts["information_edge"], 2),
        "cip_agent_node_index": (counts["information_node"],),
        "cep_information_to_physical_edge_index": (counts["information_edge"],),
        "cfl_information_edge_index": (counts["data_flow"],),
    }
    for name, shape in expected_shapes.items():
        if name not in static or np.asarray(static[name]).shape != shape:
            raise ValueError(f"{name} shape does not match the v3 entity slots")

    _validate_index_range(
        "physical_edge_endpoint_index",
        static["physical_edge_endpoint_index"],
        counts["physical_node"],
    )
    _validate_index_range(
        "information_edge_endpoint_index",
        static["information_edge_endpoint_index"],
        counts["information_node"],
    )
    _validate_index_range(
        "cip_agent_node_index",
        static["cip_agent_node_index"],
        counts["physical_node"],
    )
    _validate_index_range(
        "cep_information_to_physical_edge_index",
        static["cep_information_to_physical_edge_index"],
        counts["physical_edge"],
    )
    _validate_index_range(
        "cfl_information_edge_index",
        static["cfl_information_edge_index"],
        counts["information_edge"],
    )

    for name, endpoint_name in (
        ("physical_edge_endpoint_index", "physical"),
        ("information_edge_endpoint_index", "information"),
    ):
        endpoints = np.asarray(static[name])
        partial_padding = (endpoints[:, 0] < 0) != (endpoints[:, 1] < 0)
        if np.any(partial_padding):
            raise ValueError(f"{name} contains a partially padded endpoint pair")

    active_relations = (
        (
            "cip_agent_node_index",
            np.any(np.asarray(history["information_node_present"], dtype=bool), axis=0)
            | np.any(np.asarray(target["information_node_present"], dtype=bool), axis=0),
        ),
        (
            "cep_information_to_physical_edge_index",
            np.any(np.asarray(history["information_edge_present"], dtype=bool), axis=0)
            | np.any(np.asarray(target["information_edge_present"], dtype=bool), axis=0),
        ),
        (
            "cfl_information_edge_index",
            np.any(np.asarray(history["data_flow_present"], dtype=bool), axis=0)
            | np.any(np.asarray(target["data_flow_present"], dtype=bool), axis=0),
        ),
    )
    for name, active in active_relations:
        if np.any(active & (np.asarray(static[name]) < 0)):
            raise ValueError(f"{name} is missing for an active entity")

    for namespace_name, namespace in (
        ("history_action", history_action),
        ("future_action", future_action),
    ):
        for required in ACTION_KEYS:
            if required not in namespace:
                raise ValueError(f"{namespace_name} is missing {required}")
        action = np.asarray(namespace["task_action"])
        action_present = np.asarray(namespace["task_action_present"])
        role_index = np.asarray(namespace["task_action_information_node_index"])
        if action.shape[-2:] != (counts["task"], 8):
            raise ValueError(f"{namespace_name}.task_action shape is invalid")
        if action_present.shape != action.shape[:-1]:
            raise ValueError(f"{namespace_name}.task_action_present shape is invalid")
        if role_index.shape != (*action_present.shape, 4):
            raise ValueError(
                f"{namespace_name}.task_action_information_node_index shape is invalid"
            )
        _validate_index_range(
            f"{namespace_name}.task_action_information_node_index",
            role_index,
            counts["information_node"],
        )


def make_explicit_batch(
    payload: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> ExplicitStateBatch:
    """Convert one R3 view into a batch while preserving missing-value semantics."""

    if payload.get("schema_version") != R3_VIEW_SCHEMA:
        raise ValueError("R3 payload schema_version is invalid")
    if normalization_stats.get("source_split") != "train":
        raise ValueError("R3 normalization must use train-only statistics")
    static_np = {
        key: np.asarray(value).copy()
        for key, value in dict(payload.get("static", {})).items()
    }
    history_action_np = {
        key: np.asarray(value).copy()
        for key, value in dict(payload.get("history_action", {})).items()
    }
    history_np = _normalize_namespace(
        _add_information_link_activity(dict(payload.get("history", {}))),
        static_np,
        normalization_stats,
    )
    target_np = _normalize_namespace(
        _add_information_link_activity(dict(payload.get("target", {}))),
        static_np,
        normalization_stats,
    )
    future_action_np = {
        key: np.asarray(value).copy()
        for key, value in dict(payload.get("future_action", {})).items()
    }
    _validate_r3_relations(
        history_np,
        history_action_np,
        future_action_np,
        target_np,
        static_np,
    )
    window = dict(payload.get("window", {}))
    return ExplicitStateBatch(
        history=_torch_namespace(history_np, device),
        history_action=_torch_namespace(history_action_np, device),
        future_action=_torch_namespace(future_action_np, device),
        target=_torch_namespace(target_np, device),
        static=_torch_namespace(static_np, device),
        metadata={
            "schema_version": R3_VIEW_SCHEMA,
            "trajectory_id": str(window.get("trajectory_id", "")),
            "environment_seed": window.get("environment_seed"),
            "split": str(window.get("split", "")),
            "history_start": window.get("history_start"),
            "target_start": window.get("target_start"),
            "horizon_steps": window.get("horizon_steps"),
            "state_source": str(window.get("state_source", "frozen_trajectory_window")),
            "future_placeholder_semantics": window.get(
                "future_placeholder_semantics"
            ),
        },
    )


def read_trajectory_index(dataset_root: str | Path) -> list[dict[str, str]]:
    path = Path(dataset_root) / "trajectory_index.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


__all__ = [
    "ACTION_KEYS",
    "CONTINUOUS_STATE_KEYS",
    "ExplicitStateBatch",
    "R3_VIEW_SCHEMA",
    "R3Window",
    "TEMPORAL_STATE_KEYS",
    "load_r3_window",
    "make_explicit_batch",
    "read_trajectory_index",
    "select_r3_windows",
    "verify_r3_inputs",
]
