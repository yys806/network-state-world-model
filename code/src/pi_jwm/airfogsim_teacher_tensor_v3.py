"""Fixed tensor contract for teacher-aligned PI-JWM AirFogSim graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .airfogsim_tensor_v2 import TensorContract, infer_tensor_contract, natural_id_key
from .formal_airfogsim_graph_v1 import tensorize_formal_graph


SCHEMA_VERSION = "PIJWM-DG-Contract-v3-tensor"
PHYSICAL_NODE_FEATURES = (
    "x",
    "y",
    "z",
    "speed",
    "acceleration",
    "cpu",
    "storage",
    "energy",
    "heading",
)
PHYSICAL_EDGE_FEATURES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "distance",
    "relative_speed",
    "los",
    "blocked",
)
INFORMATION_NODE_FEATURES = (
    "unassigned_queue_count",
    "tx_queue_count",
    "cpu_backlog",
    "running_count",
    "return_queue_count",
    "deadline_min",
    "priority_mean",
)
INFORMATION_EDGE_FEATURES = (
    "pre.interface_available",
    "pre.csi_mean",
    "pre.channel_gain",
    "pre.path_loss",
    "pre.noise",
    "pre.historical_interference",
    "pre.historical_sinr",
    "pre.historical_rate",
    "action.allocated_rb_count",
    "action.tx_power",
    "action.mcs",
    "outcome.active_task_count",
    "outcome.rate_sum",
    "outcome.actual_interference",
    "outcome.actual_sinr",
    "outcome.outage",
    "outcome.throughput",
    "outcome.served_data",
)
INFORMATION_EDGE_TYPES = (
    "V2V",
    "V2U",
    "V2I",
    "U2V",
    "U2U",
    "U2I",
    "I2V",
    "I2U",
    "I2I",
    "wired",
)


@dataclass(frozen=True)
class TeacherAlignedTensorContract:
    max_nodes: int
    max_physical_edges: int
    max_information_edges: int
    max_flows: int
    max_tasks: int
    max_dag_edges: int
    history_steps: int = 8
    horizon_steps: int = 3
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_nodes": self.max_nodes,
            "max_physical_edges": self.max_physical_edges,
            "max_information_edges": self.max_information_edges,
            "max_flows": self.max_flows,
            "max_tasks": self.max_tasks,
            "max_dag_edges": self.max_dag_edges,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "physical_node_features": list(PHYSICAL_NODE_FEATURES),
            "physical_edge_features": list(PHYSICAL_EDGE_FEATURES),
            "information_node_features": list(INFORMATION_NODE_FEATURES),
            "information_edge_features": list(INFORMATION_EDGE_FEATURES),
            "information_edge_feature_roles": {
                "pre": [name for name in INFORMATION_EDGE_FEATURES if name.startswith("pre.")],
                "action": [name for name in INFORMATION_EDGE_FEATURES if name.startswith("action.")],
                "outcome": [name for name in INFORMATION_EDGE_FEATURES if name.startswith("outcome.")],
            },
            "information_edge_types": list(INFORMATION_EDGE_TYPES),
            "cross_graph_relations": ["CIP", "CEP"],
            "business_relations": ["CFL"],
            "deprecated_relations": ["CFE"],
        }


def _compatibility_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build an internal view used only to reuse audited task/action tensor code."""

    channel_edges = []
    for edge in graph.get("information_edges", []):
        source = str(edge["src"]).removeprefix("agent::")
        target = str(edge["dst"]).removeprefix("agent::")
        channel_edges.append(
            {
                "id": f"pe::{source}::{target}",
                "src": source,
                "dst": target,
                "kind": str(edge.get("kind", "")),
            }
        )
    channel_snapshots = []
    for row in graph.get("source_information_edge_snapshots", []):
        source = str(row["src"]).removeprefix("agent::")
        target = str(row["dst"]).removeprefix("agent::")
        channel_snapshots.append(
            {
                "id": f"pe::{source}::{target}",
                "src": source,
                "dst": target,
                "kind": str(row.get("kind", "")),
                "observed_time": float(row["observed_time"]),
                "distance": 0.0,
                "csi_mean": float(row["pre"].get("csi_mean", 0.0)),
                "rate_sum": float(row["outcome"].get("rate_sum", 0.0)),
                "active_task_count": float(
                    row["outcome"].get("active_task_count", 0.0)
                ),
                "allocated_rb_count": float(
                    row["action"].get("allocated_rb_count", 0.0)
                ),
            }
        )
    return {
        "physical_nodes": list(graph.get("physical_nodes", [])),
        "physical_edges": channel_edges,
        "information_nodes": list(graph.get("information_nodes", [])),
        "information_edges": list(graph.get("data_flows", [])),
        "agent_attachments": [
            {
                "agent_id": row["information_node_id"],
                "physical_node_id": row["physical_node_id"],
            }
            for row in graph.get("cip_relations", [])
        ],
        "flow_bearers": [],
        "task_nodes": list(graph.get("task_nodes", [])),
        "task_dag_edges": list(graph.get("task_dag_edges", [])),
        "source_physical_node_snapshots": list(
            graph.get("source_physical_node_snapshots", [])
        ),
        "source_physical_edge_snapshots": channel_snapshots,
        "source_task_snapshots": list(graph.get("source_task_snapshots", [])),
        "source_transfer_events": list(graph.get("source_transfer_events", [])),
        "source_offload_actions": list(graph.get("source_offload_actions", [])),
        "source_return_actions": list(graph.get("source_return_actions", [])),
        "source_rb_actions": list(graph.get("source_rb_actions", [])),
        "source_cpu_actions": list(graph.get("source_cpu_actions", [])),
    }


def infer_teacher_tensor_contract(
    graphs: Sequence[Mapping[str, Any]],
    *,
    history_steps: int = 8,
    horizon_steps: int = 3,
) -> TeacherAlignedTensorContract:
    if not graphs:
        raise ValueError("at least one teacher-aligned graph is required")
    legacy_contracts = [
        infer_tensor_contract(
            [_compatibility_graph(graph)],
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
        for graph in graphs
    ]
    return TeacherAlignedTensorContract(
        max_nodes=max(len(graph.get("physical_nodes", [])) for graph in graphs),
        max_physical_edges=max(len(graph.get("physical_edges", [])) for graph in graphs),
        max_information_edges=max(
            len(graph.get("information_edges", [])) for graph in graphs
        ),
        max_flows=max(row.max_flows for row in legacy_contracts),
        max_tasks=max(row.max_tasks for row in legacy_contracts),
        max_dag_edges=max(row.max_dag_edges for row in legacy_contracts),
        history_steps=int(history_steps),
        horizon_steps=int(horizon_steps),
    )


def contract_from_dict(value: Mapping[str, Any]) -> TeacherAlignedTensorContract:
    fields = (
        "max_nodes",
        "max_physical_edges",
        "max_information_edges",
        "max_flows",
        "max_tasks",
        "max_dag_edges",
        "history_steps",
        "horizon_steps",
    )
    return TeacherAlignedTensorContract(**{field: int(value[field]) for field in fields})


def _time_index(times: np.ndarray, value: Any) -> int:
    matches = np.flatnonzero(
        np.isclose(times.astype(np.float64), float(value), rtol=0.0, atol=1e-5)
    )
    if matches.size != 1:
        raise ValueError(f"observed time {value} is not on the tensor grid")
    return int(matches[0])


def _feature_value(row: Mapping[str, Any], name: str) -> tuple[float, bool]:
    group, field = name.split(".", 1)
    return (
        float(row[group].get(field, 0.0) or 0.0),
        bool(row.get("feature_mask", {}).get(group, {}).get(field, False)),
    )


def _build_structural_arrays(
    graph: Mapping[str, Any],
    contract: TeacherAlignedTensorContract,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    node_vocab = sorted(
        (str(row["id"]) for row in graph.get("physical_nodes", [])),
        key=natural_id_key,
    )
    agent_vocab = sorted(
        (str(row["id"]) for row in graph.get("information_nodes", [])),
        key=natural_id_key,
    )
    physical_edge_vocab = sorted(
        (dict(row) for row in graph.get("physical_edges", [])),
        key=lambda row: (natural_id_key(row["src"]), natural_id_key(row["dst"])),
    )
    information_edge_vocab = sorted(
        (dict(row) for row in graph.get("information_edges", [])),
        key=lambda row: (
            str(row.get("kind", "")),
            natural_id_key(row["src"]),
            natural_id_key(row["dst"]),
        ),
    )
    node_index = {value: index for index, value in enumerate(node_vocab)}
    agent_index = {value: index for index, value in enumerate(agent_vocab)}
    physical_edge_index = {
        str(row["id"]): index for index, row in enumerate(physical_edge_vocab)
    }
    information_edge_index = {
        str(row["id"]): index for index, row in enumerate(information_edge_vocab)
    }
    step_count = len(times)
    arrays: dict[str, np.ndarray] = {
        "physical_node_state": np.zeros(
            (step_count, contract.max_nodes, len(PHYSICAL_NODE_FEATURES)),
            dtype=np.float32,
        ),
        "physical_node_feature_mask": np.zeros(
            (step_count, contract.max_nodes, len(PHYSICAL_NODE_FEATURES)),
            dtype=bool,
        ),
        "physical_node_present": np.zeros(
            (step_count, contract.max_nodes), dtype=bool
        ),
        "physical_edge_state": np.zeros(
            (step_count, contract.max_physical_edges, len(PHYSICAL_EDGE_FEATURES)),
            dtype=np.float32,
        ),
        "physical_edge_feature_mask": np.zeros(
            (step_count, contract.max_physical_edges, len(PHYSICAL_EDGE_FEATURES)),
            dtype=bool,
        ),
        "physical_edge_present": np.zeros(
            (step_count, contract.max_physical_edges), dtype=bool
        ),
        "physical_edge_endpoint_index": np.full(
            (contract.max_physical_edges, 2), -1, dtype=np.int32
        ),
        "information_node_state": np.zeros(
            (step_count, contract.max_nodes, len(INFORMATION_NODE_FEATURES)),
            dtype=np.float32,
        ),
        "information_node_feature_mask": np.zeros(
            (step_count, contract.max_nodes, len(INFORMATION_NODE_FEATURES)),
            dtype=bool,
        ),
        "information_node_present": np.zeros(
            (step_count, contract.max_nodes), dtype=bool
        ),
        "information_edge_state": np.zeros(
            (
                step_count,
                contract.max_information_edges,
                len(INFORMATION_EDGE_FEATURES),
            ),
            dtype=np.float32,
        ),
        "information_edge_feature_mask": np.zeros(
            (
                step_count,
                contract.max_information_edges,
                len(INFORMATION_EDGE_FEATURES),
            ),
            dtype=bool,
        ),
        "information_edge_present": np.zeros(
            (step_count, contract.max_information_edges), dtype=bool
        ),
        "information_edge_endpoint_index": np.full(
            (contract.max_information_edges, 2), -1, dtype=np.int32
        ),
        "information_edge_kind_index": np.full(
            (contract.max_information_edges,), -1, dtype=np.int16
        ),
        "cip_agent_node_index": np.full((contract.max_nodes,), -1, dtype=np.int32),
        "cep_information_to_physical_edge_index": np.full(
            (contract.max_information_edges,), -1, dtype=np.int32
        ),
    }
    for edge_id, index in physical_edge_index.items():
        row = physical_edge_vocab[index]
        arrays["physical_edge_endpoint_index"][index] = [
            node_index[str(row["src"])],
            node_index[str(row["dst"])],
        ]
    for edge_id, index in information_edge_index.items():
        row = information_edge_vocab[index]
        arrays["information_edge_endpoint_index"][index] = [
            agent_index[str(row["src"])],
            agent_index[str(row["dst"])],
        ]
        kind = str(row.get("kind", ""))
        if kind in INFORMATION_EDGE_TYPES:
            arrays["information_edge_kind_index"][index] = INFORMATION_EDGE_TYPES.index(
                kind
            )
    for row in graph.get("cip_relations", []):
        ai = agent_index[str(row["information_node_id"])]
        arrays["cip_agent_node_index"][ai] = node_index[str(row["physical_node_id"])]
    for row in graph.get("cep_relations", []):
        ii = information_edge_index[str(row["information_edge_id"])]
        arrays["cep_information_to_physical_edge_index"][ii] = physical_edge_index[
            str(row["physical_edge_id"])
        ]
    node_kind_by_id = {
        str(row["id"]): str(row.get("kind", ""))
        for row in graph.get("physical_nodes", [])
    }
    arrays["physical_node_kind_index"] = np.asarray(
        [
            ("vehicle", "uav", "rsu", "edge_server", "cloud").index(
                node_kind_by_id[node]
            )
            if node_kind_by_id.get(node) in ("vehicle", "uav", "rsu", "edge_server", "cloud")
            else -1
            for node in node_vocab
        ]
        + [-1] * (contract.max_nodes - len(node_vocab)),
        dtype=np.int16,
    )
    for row in graph.get("source_physical_node_snapshots", []):
        ti = _time_index(times, row["observed_time"])
        ni = node_index[str(row["id"])]
        position = list(row.get("position", []))
        position.extend([0.0] * (3 - len(position)))
        values = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(row.get("speed", 0.0) or 0.0),
            float(row.get("acceleration", 0.0) or 0.0),
            float(row.get("cpu", 0.0) or 0.0),
            float(row.get("storage", 0.0) or 0.0),
            float(row.get("energy", 0.0) or 0.0),
            float(row.get("heading", 0.0) or 0.0),
        ]
        mask = [
            len(row.get("position", [])) > index for index in range(3)
        ] + [
            row.get("speed") is not None,
            row.get("acceleration") is not None,
            row.get("cpu") is not None,
            row.get("storage") is not None,
            row.get("energy") is not None,
            row.get("heading") is not None,
        ]
        arrays["physical_node_state"][ti, ni] = values
        arrays["physical_node_feature_mask"][ti, ni] = mask
        arrays["physical_node_present"][ti, ni] = True
    for row in graph.get("source_physical_edge_snapshots", []):
        ti = _time_index(times, row["observed_time"])
        ei = physical_edge_index[str(row["id"])]
        delta = list(row.get("delta_position", []))
        delta.extend([0.0] * (3 - len(delta)))
        values = [
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            float(row.get("distance", 0.0) or 0.0),
            float(row.get("relative_speed", 0.0) or 0.0),
            float(row.get("los", 0.0) or 0.0),
            float(row.get("blocked", 0.0) or 0.0),
        ]
        mask = row.get("feature_mask", {})
        masks = [
            bool(mask.get("delta_position", False)),
            bool(mask.get("delta_position", False)),
            bool(mask.get("delta_position", False)),
            bool(mask.get("distance", False)),
            bool(mask.get("relative_speed", False)),
            bool(mask.get("los", False)),
            bool(mask.get("blocked", False)),
        ]
        arrays["physical_edge_state"][ti, ei] = values
        arrays["physical_edge_feature_mask"][ti, ei] = masks
        arrays["physical_edge_present"][ti, ei] = True
    for row in graph.get("source_information_node_snapshots", []):
        ti = _time_index(times, row["observed_time"])
        ai = agent_index[str(row["id"])]
        arrays["information_node_state"][ti, ai] = [
            float(row.get(name, 0.0) or 0.0) for name in INFORMATION_NODE_FEATURES
        ]
        arrays["information_node_feature_mask"][ti, ai] = [
            bool(row.get("feature_mask", {}).get(name, False))
            for name in INFORMATION_NODE_FEATURES
        ]
        arrays["information_node_present"][ti, ai] = True
    for row in graph.get("source_information_edge_snapshots", []):
        ti = _time_index(times, row["observed_time"])
        ii = information_edge_index[str(row["id"])]
        values, masks = zip(*(_feature_value(row, name) for name in INFORMATION_EDGE_FEATURES))
        arrays["information_edge_state"][ti, ii] = values
        arrays["information_edge_feature_mask"][ti, ii] = masks
        arrays["information_edge_present"][ti, ii] = True
    report = {
        "node_vocab": node_vocab,
        "agent_vocab": agent_vocab,
        "physical_edge_vocab": [str(row["id"]) for row in physical_edge_vocab],
        "information_edge_vocab": [str(row["id"]) for row in information_edge_vocab],
        "index_maps": {
            "node": node_index,
            "agent": agent_index,
            "physical_edge": physical_edge_index,
            "information_edge": information_edge_index,
        },
    }
    return arrays, report


def _map_physical_to_agent_indices(
    value: np.ndarray, node_to_agent: Mapping[int, int]
) -> np.ndarray:
    result = np.full(value.shape, -1, dtype=np.int32)
    for old_index, agent_index in node_to_agent.items():
        result[value == old_index] = agent_index
    return result


def _event_matches_flow(event: Mapping[str, Any], flow: Mapping[str, Any]) -> bool:
    phase_type = (
        "task_input"
        if event.get("phase") == "offload"
        else "result_return"
        if event.get("phase") == "return"
        else None
    )
    return (
        phase_type == flow.get("flow_type")
        and str(event.get("task_id")) == str(flow.get("task_id"))
        and str(event.get("source"))
        == str(flow.get("src", "")).removeprefix("agent::")
        and str(event.get("target"))
        == str(flow.get("dst", "")).removeprefix("agent::")
    )


def _materialize_waiting_data_flows(
    graph: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    flow_vocab: Sequence[str],
) -> None:
    """Keep flows present from creation while they wait for link service."""

    by_id = {str(row["id"]): row for row in graph.get("data_flows", [])}
    events = list(graph.get("source_transfer_events", []))
    arrays["data_flow_state"].fill(0.0)
    arrays["data_flow_present"].fill(False)
    arrays["data_flow_completed"].fill(False)
    for fi, flow_id in enumerate(flow_vocab):
        flow = by_id[flow_id]
        matching = [event for event in events if _event_matches_flow(event, flow)]
        candidate_times = []
        if flow.get("first_time") is not None:
            candidate_times.append(float(flow["first_time"]))
        candidate_times.extend(float(event["time"]) for event in matching)
        creation_time = min(candidate_times) if candidate_times else float(arrays["time"][0])
        total = max(float(flow.get("total_data", 0.0) or 0.0), 0.0)
        for ti, time_value in enumerate(arrays["time"]):
            time = float(time_value)
            delivered_here = sum(
                float(event.get("delivered_data", 0.0) or 0.0)
                for event in matching
                if np.isclose(float(event["time"]), time, rtol=0.0, atol=1e-5)
            )
            delivered_cumulative = sum(
                float(event.get("delivered_data", 0.0) or 0.0)
                for event in matching
                if float(event["time"]) <= time + 1e-5
            )
            completed = any(
                bool(event.get("flow_completed"))
                and float(event["time"]) <= time + 1e-5
                for event in matching
            )
            completed_before = any(
                bool(event.get("flow_completed"))
                and float(event["time"]) < time - 1e-5
                for event in matching
            )
            present = time + 1e-5 >= creation_time and not completed_before
            arrays["data_flow_present"][ti, fi] = present
            arrays["data_flow_completed"][ti, fi] = completed
            if present:
                arrays["data_flow_state"][ti, fi] = [
                    total,
                    max(total - delivered_cumulative, 0.0),
                    delivered_cumulative,
                    delivered_here,
                    max(time - creation_time, 0.0),
                ]


def tensorize_teacher_aligned_graph(
    graph: Mapping[str, Any],
    contract: TeacherAlignedTensorContract,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Tensorize one v3 trajectory while reusing audited task/action logic."""

    compatibility = _compatibility_graph(graph)
    legacy_contract = TensorContract(
        max_nodes=contract.max_nodes,
        max_physical_edges=contract.max_information_edges,
        max_flows=contract.max_flows,
        max_tasks=contract.max_tasks,
        max_dag_edges=contract.max_dag_edges,
        history_steps=contract.history_steps,
        horizon_steps=contract.horizon_steps,
    )
    legacy, legacy_report = tensorize_formal_graph(compatibility, legacy_contract)
    arrays, report = _build_structural_arrays(graph, contract, legacy["time"])
    arrays["time"] = legacy["time"]
    physical_node_to_agent: dict[int, int] = {}
    for agent_index, node_index in enumerate(arrays["cip_agent_node_index"]):
        if node_index >= 0:
            physical_node_to_agent[int(node_index)] = int(agent_index)
    copied = (
        "task_state",
        "task_present",
        "task_valid",
        "task_lifecycle_index",
        "task_action",
        "task_action_present",
        "dag_edge_index",
        "dag_edge_valid",
        "dag_edge_present",
        "task_dag_state",
        "task_dag_state_present",
    )
    for name in copied:
        arrays[name] = legacy[name]
    arrays["task_information_node_index"] = _map_physical_to_agent_indices(
        legacy["task_node_index"], physical_node_to_agent
    )
    arrays["task_action_information_node_index"] = _map_physical_to_agent_indices(
        legacy["task_action_node_index"], physical_node_to_agent
    )
    arrays["data_flow_state"] = legacy["flow_state"]
    arrays["data_flow_present"] = legacy["flow_present"]
    arrays["data_flow_completed"] = legacy["flow_completed"]
    arrays["data_flow_valid"] = legacy["flow_valid"]
    arrays["data_flow_type_index"] = legacy["flow_type_index"]
    arrays["data_flow_endpoint_index"] = _map_physical_to_agent_indices(
        legacy["flow_endpoint_index"], physical_node_to_agent
    )
    flow_index = {
        value: index for index, value in enumerate(legacy_report["flow_vocab"])
    }
    info_edge_index = report["index_maps"]["information_edge"]
    _materialize_waiting_data_flows(
        graph, arrays, legacy_report["flow_vocab"]
    )
    arrays["cfl_information_edge_index"] = np.full(
        (contract.max_flows,), -1, dtype=np.int32
    )
    arrays["cfl_mask"] = np.zeros(
        (len(arrays["time"]), contract.max_flows, contract.max_information_edges),
        dtype=bool,
    )
    for row in graph.get("cfl_relations", []):
        fi = flow_index[str(row["flow_id"])]
        ii = info_edge_index[str(row["information_edge_id"])]
        arrays["cfl_information_edge_index"][fi] = ii
        arrays["cfl_mask"][:, fi, ii] = (
            arrays["data_flow_present"][:, fi]
            & arrays["information_edge_present"][:, ii]
        )
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "flow_vocab": legacy_report["flow_vocab"],
            "task_vocab": legacy_report["task_vocab"],
            "dag_vocab": legacy_report["dag_vocab"],
            "time_count": int(len(arrays["time"])),
            "counts": {
                "physical_nodes": len(report["node_vocab"]),
                "physical_edges": len(report["physical_edge_vocab"]),
                "information_nodes": len(report["agent_vocab"]),
                "information_edges": len(report["information_edge_vocab"]),
                "data_flows": len(legacy_report["flow_vocab"]),
                "cfl_relations": int(
                    np.count_nonzero(arrays["cfl_information_edge_index"] >= 0)
                ),
                "tasks": len(legacy_report["task_vocab"]),
                "dag_edges": len(legacy_report["dag_vocab"]),
            },
        }
    )
    report.pop("index_maps", None)
    validation = validate_teacher_tensors(arrays, contract)
    report["validation"] = validation
    return arrays, report


def validate_teacher_tensors(
    arrays: Mapping[str, np.ndarray],
    contract: TeacherAlignedTensorContract,
) -> dict[str, Any]:
    required = (
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
        "cip_agent_node_index",
        "cep_information_to_physical_edge_index",
        "cfl_information_edge_index",
        "cfl_mask",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing v3 tensor arrays: {missing}")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    masked_pairs = (
        ("physical_node_state", "physical_node_feature_mask", "masked physical-node values"),
        ("physical_edge_state", "physical_edge_feature_mask", "masked physical-edge values"),
        ("information_node_state", "information_node_feature_mask", "masked information-node values"),
        ("information_edge_state", "information_edge_feature_mask", "masked information-edge values"),
    )
    for value_name, mask_name, message in masked_pairs:
        if np.any(np.asarray(arrays[value_name])[~np.asarray(arrays[mask_name], dtype=bool)] != 0.0):
            raise ValueError(message)
    padding_pairs = (
        ("physical_node_state", "physical_node_present"),
        ("physical_edge_state", "physical_edge_present"),
        ("information_node_state", "information_node_present"),
        ("information_edge_state", "information_edge_present"),
        ("data_flow_state", "data_flow_present"),
        ("task_state", "task_present"),
    )
    for value_name, present_name in padding_pairs:
        if value_name in arrays and np.any(
            np.asarray(arrays[value_name])[~np.asarray(arrays[present_name], dtype=bool)]
            != 0.0
        ):
            raise ValueError(f"{value_name} padding must be zero")
    for name in (
        "physical_edge_endpoint_index",
        "information_edge_endpoint_index",
        "cip_agent_node_index",
        "cep_information_to_physical_edge_index",
        "cfl_information_edge_index",
        "data_flow_endpoint_index",
        "task_information_node_index",
        "task_action_information_node_index",
        "dag_edge_index",
    ):
        if name in arrays and np.any(np.asarray(arrays[name]) < -1):
            raise ValueError(f"{name} contains an invalid negative index")
    checks = {
        "finite_values": True,
        "missing_values_are_masked": True,
        "padding_is_zero": True,
        "indices_are_valid_or_padding": True,
        "physical_information_features_are_separate": True,
    }
    return {
        "schema_version": "PIJWM-DG-Contract-v3-tensor-validation",
        "teacher_aligned_tensor_valid": all(checks.values()),
        "checks": checks,
        "time_count": int(arrays["time"].shape[0]),
        "capacities": contract.to_dict(),
    }


__all__ = [
    "INFORMATION_EDGE_FEATURES",
    "INFORMATION_NODE_FEATURES",
    "PHYSICAL_EDGE_FEATURES",
    "PHYSICAL_NODE_FEATURES",
    "SCHEMA_VERSION",
    "TeacherAlignedTensorContract",
    "contract_from_dict",
    "infer_teacher_tensor_contract",
    "tensorize_teacher_aligned_graph",
    "validate_teacher_tensors",
]
