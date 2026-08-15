"""Online strict-dual-graph observations for factual R6 closed-loop control."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from .airfogsim_dual_graph_v2 import build_dual_graph_v2_bundle
from .airfogsim_teacher_graph_v3 import (
    remap_teacher_aligned_graph,
    validate_teacher_aligned_graph,
)
from .airfogsim_teacher_tensor_v3 import (
    TeacherAlignedTensorContract,
    tensorize_teacher_aligned_graph,
)
from .r3_preflight_data import (
    ACTION_KEYS,
    R3_VIEW_SCHEMA,
    TEMPORAL_STATE_KEYS,
)


R6_ONLINE_OBSERVATION_SCHEMA = "PIJWM-R6-online-strict-dual-graph-v1"


class OnlineDualGraphHistory:
    """Accumulate only observations and executed actions from one live episode."""

    def __init__(self, trajectory_id: str, *, max_frames: int | None = None) -> None:
        self.trajectory_id = str(trajectory_id)
        if not self.trajectory_id:
            raise ValueError("online trajectory_id must be non-empty")
        if max_frames is not None and int(max_frames) <= 0:
            raise ValueError("max_frames must be positive when provided")
        self.max_frames = None if max_frames is None else int(max_frames)
        self._physical_nodes: dict[str, dict[str, Any]] = {}
        self._physical_edges: dict[str, dict[str, Any]] = {}
        self._task_records: dict[str, dict[str, Any]] = {}
        self._dag_edges: dict[str, dict[str, Any]] = {}
        self._transfer_events: dict[str, dict[str, Any]] = {}
        self._node_snapshots: list[dict[str, Any]] = []
        self._edge_snapshots: list[dict[str, Any]] = []
        self._task_snapshots: list[dict[str, Any]] = []
        self._offload_actions: list[dict[str, Any]] = []
        self._return_actions: list[dict[str, Any]] = []
        self._rb_actions: list[dict[str, Any]] = []
        self._cpu_actions: list[dict[str, Any]] = []
        self._frame_times: list[float] = []

    @property
    def frame_count(self) -> int:
        return len(self._frame_times)

    @property
    def last_observed_time(self) -> float:
        if not self._frame_times:
            raise ValueError("online history has no frames")
        return self._frame_times[-1]

    @staticmethod
    def _rows(values: Any) -> list[dict[str, Any]]:
        return [copy.deepcopy(dict(row)) for row in values]

    def append_frame(
        self,
        *,
        physical_nodes: Any,
        physical_edges: Any,
        task_records: Any,
        dag_edges: Any,
        transfer_events: Any,
        offload_actions: Any,
        return_actions: Any,
        rb_actions: Any,
        cpu_actions: Any,
    ) -> None:
        nodes = self._rows(physical_nodes)
        edges = self._rows(physical_edges)
        tasks = self._rows(task_records)
        dags = self._rows(dag_edges)
        events = self._rows(transfer_events)
        if not nodes:
            raise ValueError("an online frame must contain physical nodes")
        times = {round(float(row["observed_time"]), 6) for row in nodes}
        if len(times) != 1:
            raise ValueError("online physical-node frame must use one observation time")
        observed_time = next(iter(times))
        if self._frame_times and observed_time <= self._frame_times[-1]:
            raise ValueError("online frame times must be strictly increasing")
        self._frame_times.append(observed_time)
        for row in nodes:
            self._physical_nodes[str(row["id"])] = row
            self._node_snapshots.append(copy.deepcopy(row))
        for row in edges:
            self._physical_edges[str(row["id"])] = row
            self._edge_snapshots.append(copy.deepcopy(row))
        for row in tasks:
            self._task_records[str(row["id"])] = row
            self._task_snapshots.append(copy.deepcopy(row))
        for row in dags:
            self._dag_edges.setdefault(str(row["id"]), row)
        for row in events:
            self._transfer_events[str(row["event_id"])] = row
        self._offload_actions.extend(self._rows(offload_actions))
        self._return_actions.extend(self._rows(return_actions))
        self._rb_actions.extend(self._rows(rb_actions))
        self._cpu_actions.extend(self._rows(cpu_actions))
        self._prune_bounded_history()

    def _prune_bounded_history(self) -> None:
        if self.max_frames is None or len(self._frame_times) <= self.max_frames:
            return
        self._frame_times = self._frame_times[-self.max_frames :]
        first_time = self._frame_times[0]
        self._node_snapshots = [
            row for row in self._node_snapshots if float(row["observed_time"]) >= first_time
        ]
        self._edge_snapshots = [
            row for row in self._edge_snapshots if float(row["observed_time"]) >= first_time
        ]
        self._task_snapshots = [
            row for row in self._task_snapshots if float(row["observed_time"]) >= first_time
        ]
        self._physical_nodes = {
            str(row["id"]): row for row in self._node_snapshots
        }
        self._physical_edges = {
            str(row["id"]): row for row in self._edge_snapshots
        }
        self._task_records = {
            str(row["id"]): row for row in self._task_snapshots
        }
        self._transfer_events = {
            key: row
            for key, row in self._transfer_events.items()
            if float(row.get("time", first_time)) >= first_time
        }
        self._offload_actions = [
            row for row in self._offload_actions if float(row.get("time", first_time)) >= first_time
        ]
        self._return_actions = [
            row for row in self._return_actions if float(row.get("time", first_time)) >= first_time
        ]
        self._rb_actions = [
            row for row in self._rb_actions if float(row.get("time", first_time)) >= first_time
        ]
        self._cpu_actions = [
            row for row in self._cpu_actions if float(row.get("time", first_time)) >= first_time
        ]

    def build_source_graph(self) -> dict[str, Any]:
        if not self._frame_times:
            raise ValueError("cannot build an online graph without frames")
        graph = build_dual_graph_v2_bundle(
            trajectory_id=self.trajectory_id,
            physical_nodes=self._physical_nodes.values(),
            physical_edges=self._physical_edges.values(),
            task_records=self._task_records.values(),
            dag_edges=self._dag_edges.values(),
            transfer_events=self._transfer_events.values(),
            task_snapshots=self._task_snapshots,
            offload_actions=self._offload_actions,
            return_actions=self._return_actions,
            rb_actions=self._rb_actions,
        )
        kept_node_ids = {str(row["id"]) for row in graph["physical_nodes"]}
        kept_edge_ids = {str(row["id"]) for row in graph["physical_edges"]}
        graph["source_physical_node_snapshots"] = [
            copy.deepcopy(row)
            for row in self._node_snapshots
            if str(row.get("id")) in kept_node_ids
        ]
        graph["source_physical_edge_snapshots"] = [
            copy.deepcopy(row)
            for row in self._edge_snapshots
            if str(row.get("id")) in kept_edge_ids
        ]
        graph["source_cpu_actions"] = copy.deepcopy(self._cpu_actions)
        graph["online_observation_schema"] = R6_ONLINE_OBSERVATION_SCHEMA
        graph["online_frame_count"] = self.frame_count
        return graph


def build_online_teacher_arrays(
    source_graph: Mapping[str, Any],
    *,
    contract: TeacherAlignedTensorContract,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the frozen R1 ontology and capacity to current AirFogSim history."""

    node_ids = {str(row.get("id")) for row in source_graph.get("physical_nodes", [])}
    edge_endpoints = {
        str(row.get(field))
        for row in source_graph.get("source_physical_edge_snapshots", [])
        for field in ("src", "dst")
    }
    missing_endpoints = sorted(edge_endpoints - node_ids)
    if missing_endpoints:
        raise ValueError(
            "online channel history references physical nodes outside the recent vocabulary: "
            f"{missing_endpoints[:10]}"
        )
    teacher_graph = remap_teacher_aligned_graph(source_graph)
    graph_validation = validate_teacher_aligned_graph(teacher_graph)
    if not graph_validation.get("teacher_aligned_graph_valid"):
        raise ValueError("online teacher-aligned graph failed validation")
    arrays, report = tensorize_teacher_aligned_graph(teacher_graph, contract)
    if not report.get("validation", {}).get("teacher_aligned_tensor_valid"):
        raise ValueError("online strict dual-graph tensor failed validation")
    report = dict(report)
    report["online_observation_schema"] = R6_ONLINE_OBSERVATION_SCHEMA
    report["graph_validation"] = graph_validation
    return arrays, report


def make_online_inference_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    trajectory_id: str,
    environment_seed: int,
    split: str,
    history_steps: int = 8,
) -> dict[str, Any]:
    """Create an inference-only R3 view without reading a factual future state."""

    split_value = str(split)
    if split_value == "locked_test":
        raise ValueError("locked_test is sealed until R9")
    if split_value not in {"train", "validation", "calibration"}:
        raise ValueError(f"unsupported online observation split: {split_value}")
    times = np.asarray(arrays.get("time"), dtype=np.float32)
    steps = int(history_steps)
    if times.ndim != 1 or len(times) < steps:
        raise ValueError("online history is shorter than the frozen history length")
    start = len(times) - steps
    history: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    for key in TEMPORAL_STATE_KEYS:
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.shape[0] != len(times):
            raise ValueError(f"online temporal state is not time aligned: {key}")
        history[key] = value[start:].copy()
        # Required only by the shared batch validator. It is the current state,
        # not a future label, and infer_belief never consumes this namespace.
        target[key] = value[-1:].copy()
    history_action: dict[str, np.ndarray] = {}
    future_action: dict[str, np.ndarray] = {}
    for key in ACTION_KEYS:
        if key not in arrays:
            raise ValueError(f"online action history is missing {key}")
        value = np.asarray(arrays[key])
        if value.shape[0] != len(times):
            raise ValueError(f"online action tensor is not time aligned: {key}")
        history_action[key] = value[start:].copy()
        shape = (1, *value.shape[1:])
        if key.endswith("node_index"):
            future_action[key] = np.full(shape, -1, dtype=value.dtype)
        else:
            future_action[key] = np.zeros(shape, dtype=value.dtype)
    consumed = set(TEMPORAL_STATE_KEYS) | set(ACTION_KEYS) | {"time"}
    static = {
        key: np.asarray(value).copy()
        for key, value in arrays.items()
        if key not in consumed
    }
    return {
        "schema_version": R3_VIEW_SCHEMA,
        "online_observation_schema": R6_ONLINE_OBSERVATION_SCHEMA,
        "window": {
            "trajectory_id": str(trajectory_id),
            "environment_seed": int(environment_seed),
            "split": split_value,
            "history_start": start,
            "history_end": len(times),
            "target_start": len(times),
            "target_end": len(times) + 1,
            "horizon_steps": 1,
            "state_source": "online_airfogsim_strict_dual_graph",
            "future_placeholder_semantics": "duplicate_current_for_shape_validation_only",
        },
        "history": history,
        "history_action": history_action,
        "future_action": future_action,
        "target": target,
        "static": static,
        "history_time": times[start:].copy(),
        "target_time": times[-1:].copy(),
    }


__all__ = [
    "OnlineDualGraphHistory",
    "R6_ONLINE_OBSERVATION_SCHEMA",
    "build_online_teacher_arrays",
    "make_online_inference_payload",
]
