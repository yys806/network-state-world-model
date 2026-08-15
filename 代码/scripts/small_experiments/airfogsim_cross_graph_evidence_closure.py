from __future__ import annotations

"""Evidence closure helpers for PI-JWM experiment 03.

The module keeps AirFogSim-native observations separate from the explicitly
declared PI-JWM ``shared_parent_output`` DAG-flow semantics.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import types
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_contract_adapter import (
    ADAPTER_VERSION,
    apply_transmission_totals,
    capacity_safe_cpu_allocations,
    direct_transmission_totals,
)


PIJWM_EP_SOURCE = "pijwm_declared_semantics_airfogsim_observed_flow"


def nonmutating_transmission_tasks(task_manager: Any) -> tuple[dict[str, list[Any]], int]:
    """Return offload and return tasks without aliasing AirFogSim's internal lists."""

    combined = {
        str(node_id): list(tasks)
        for node_id, tasks in task_manager._offloading_tasks.items()
    }
    for node_id, tasks in task_manager._returning_tasks.items():
        target = combined.setdefault(str(node_id), [])
        existing_objects = {id(task) for task in target}
        target.extend(task for task in tasks if id(task) not in existing_objects)
    return combined, sum(len(tasks) for tasks in combined.values())


def install_capacity_safe_cpu_callback(env: Any, computation_scheduler: Any) -> None:
    """Install the PI-JWM capacity-safe callback without patching AirFogSim."""

    def callback(computing_tasks: dict[str, list[Any]], **_: Any) -> dict[str, float]:
        return capacity_safe_cpu_allocations(env, computing_tasks, max_tasks_per_node=3)

    computation_scheduler.setComputingCallBack(env, callback)


def repair_channel_energy_inputs(
    channel_manager: Any,
    direct_events: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Replace the faulty AirFogSim tx/rx accounting from direct events."""

    sending, receiving = direct_transmission_totals(direct_events)
    apply_transmission_totals(channel_manager, sending, receiving)
    return sending, receiving


def _plain_float(value: Any) -> float:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class WirelessTransferEventRecorder:
    """Create a stable record from a directly activated AirFogSim link."""

    def make_event(
        self,
        env: Any,
        profile: dict[str, Any],
        before: dict[str, Any],
    ) -> dict[str, Any]:
        rb_indices = [int(value) for value in profile.get("RB_Nos", [])]
        rate_values = env.channel_manager.getRateByChannelType(
            profile["tx_idx"],
            profile["rx_idx"],
            profile["channel_type"],
            rb_indices,
        )
        rate_by_rb = [_plain_float(value) for value in rate_values]
        planned_capacity = sum(rate_by_rb) * float(env.simulation_interval)
        transmitted_before = float(before.get("transmitted_before", 0.0))
        required_size = float(before.get("required_size", 0.0))
        remaining_before = max(required_size - transmitted_before, 0.0)
        delivered_data = min(max(planned_capacity, 0.0), remaining_before)
        source = str(before["source"])
        target = str(before["target"])
        task_id = str(profile.get("task_id") or before.get("task_id"))
        phase = str(before["phase"])
        sequence = int(before.get("sequence", 0))
        time_value = float(env.simulation_time)
        return {
            "event_id": f"event::{task_id}::{phase}::{sequence}::{time_value:.6f}",
            "task_id": task_id,
            "phase": phase,
            "source": source,
            "target": target,
            "path": [] if source == target else [f"pe::{source}::{target}"],
            "channel_type": str(profile["channel_type"]),
            "rb_indices": rb_indices,
            "rate_by_rb": rate_by_rb,
            "planned_capacity": planned_capacity,
            "remaining_before": remaining_before,
            "delivered_data": delivered_data,
            "flow_completed": delivered_data >= remaining_before - 1e-12,
            "sequence": sequence,
            "time": time_value,
            "evidence": "direct_runtime_channel_event",
        }


def _topological_path_from_events(events: list[dict[str, Any]]) -> list[str]:
    """Collapse consecutive time slices on one edge into one topology hop."""

    path: list[str] = []
    for event in events:
        for edge_id in event.get("path", []):
            edge_id = str(edge_id)
            if not path or path[-1] != edge_id:
                path.append(edge_id)
    return path


def build_observed_me_relations(
    transfer_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble lifecycle paths using only directly executed channel events."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in transfer_events:
        if event.get("evidence") != "direct_runtime_channel_event":
            continue
        phase = str(event.get("phase"))
        if phase not in {"offload", "return"}:
            continue
        grouped[(str(event["task_id"]), phase)].append(event)

    relations: list[dict[str, Any]] = []
    for (task_id, phase), events in sorted(grouped.items()):
        ordered = sorted(
            events,
            key=lambda row: (
                float(row.get("time", 0.0)),
                int(row.get("sequence", 0)),
                str(row.get("event_id", "")),
            ),
        )
        relations.append(
            {
                "task": task_id,
                "relation": "in" if phase == "offload" else "out",
                "source": str(ordered[0]["source"]),
                "target": str(ordered[-1]["target"]),
                "path": _topological_path_from_events(ordered),
                "event_ids": [str(event["event_id"]) for event in ordered],
                "physical_delivered_data": sum(
                    float(event.get("delivered_data", 0.0)) for event in ordered
                ),
                "evidence": "direct_runtime_channel_event",
            }
        )
    return relations


def _task_index(task_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in task_records}


def _completed_return_events(
    transfer_events: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in transfer_events:
        if event.get("phase") != "return":
            continue
        if event.get("evidence") != "direct_runtime_channel_event":
            continue
        grouped[str(event["task_id"])].append(event)
    return {
        task_id: sorted(
            events,
            key=lambda row: (
                float(row.get("time", 0.0)),
                int(row.get("sequence", 0)),
                str(row.get("event_id", "")),
            ),
        )
        for task_id, events in grouped.items()
        if any(bool(row.get("flow_completed")) for row in events)
    }


def build_shared_parent_output_relations(
    information_edges: list[dict[str, Any]],
    task_records: list[dict[str, Any]],
    transfer_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map DAG edges to one observed (or zero-hop) parent-output flow.

    One physical flow is emitted per parent task even when several children
    consume the same returned result.  Relations reference that shared flow.
    """

    tasks = _task_index(task_records)
    completed_returns = _completed_return_events(transfer_events)
    flows_by_parent: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []

    for edge in sorted(information_edges, key=lambda row: str(row.get("id", ""))):
        edge_id = str(edge["id"])
        parent_id = str(edge["src"])
        child_id = str(edge["dst"])
        parent = tasks.get(parent_id, {})
        child = tasks.get(child_id, {})
        source = parent.get("source")
        parent_exec = parent.get("exec")
        child_source = child.get("source")
        payload = float(parent.get("returned_size", parent.get("return_size", 0.0)) or 0.0)
        flow_id = f"flow::{parent_id}::return"
        evidence = "not_observed"

        if source not in (None, "") and source == child_source and parent_exec == source:
            flows_by_parent.setdefault(
                parent_id,
                {
                    "dependency_flow_id": flow_id,
                    "parent_task": parent_id,
                    "source": str(source),
                    "target": str(source),
                    "path": [],
                    "event_ids": [],
                    "dependency_payload": payload,
                    "physical_delivered_data": 0.0,
                    "flow_completed": True,
                    "evidence": "direct_zero_hop_state",
                },
            )
            evidence = "direct_zero_hop_state"
        elif parent_id in completed_returns and source == child_source:
            events = completed_returns[parent_id]
            if events and str(events[-1].get("target")) == str(source):
                path = _topological_path_from_events(events)
                flows_by_parent.setdefault(
                    parent_id,
                    {
                        "dependency_flow_id": flow_id,
                        "parent_task": parent_id,
                        "source": str(parent_exec),
                        "target": str(source),
                        "path": path,
                        "event_ids": [str(event["event_id"]) for event in events],
                        "dependency_payload": payload,
                        "physical_delivered_data": sum(
                            float(event.get("delivered_data", 0.0)) for event in events
                        ),
                        "flow_completed": True,
                        "evidence": "direct_runtime_channel_event",
                    },
                )
                evidence = "direct_runtime_channel_event"

        relations.append(
            {
                "info_edge": edge_id,
                "parent_task": parent_id,
                "child_task": child_id,
                "dependency_flow_id": flow_id,
                "dependency_payload": payload,
                "payload_semantics": "shared_parent_output",
                "dependency_status": "arrived" if evidence != "not_observed" else "pending",
                "evidence": evidence,
                "evidence_source": PIJWM_EP_SOURCE,
            }
        )

    return relations, [flows_by_parent[key] for key in sorted(flows_by_parent)]


def validate_shared_flow_accounting(
    ep_relations: list[dict[str, Any]],
    dependency_flows: list[dict[str, Any]],
) -> list[str]:
    """Return stable error codes for invalid shared-flow accounting."""

    errors: list[str] = []
    flow_ids = [str(row.get("dependency_flow_id")) for row in dependency_flows]
    if len(flow_ids) != len(set(flow_ids)):
        errors.append("duplicate_dependency_flow")

    flows_by_id = {
        str(row.get("dependency_flow_id")): row
        for row in dependency_flows
        if str(row.get("dependency_flow_id")) not in ("", "None")
    }
    for relation in ep_relations:
        if relation.get("evidence") == "not_observed":
            continue
        flow_id = str(relation.get("dependency_flow_id"))
        flow = flows_by_id.get(flow_id)
        if flow is None:
            errors.append("missing_dependency_flow")
            continue
        relation_payload = float(relation.get("dependency_payload", 0.0) or 0.0)
        flow_payload = float(flow.get("dependency_payload", 0.0) or 0.0)
        if abs(relation_payload - flow_payload) > 1e-9:
            errors.append("dependency_payload_mismatch")

    return sorted(set(errors))


def _is_dag(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("src"))
        target = str(edge.get("dst"))
        if source not in node_ids or target not in node_ids:
            return False
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(node_ids)


def _path_valid(
    path: list[str],
    source: Any,
    target: Any,
    physical_edges: dict[str, dict[str, Any]],
) -> bool:
    source = str(source)
    target = str(target)
    if source == target:
        return path == []
    if not path or len(path) != len(set(path)):
        return False
    rows = [physical_edges.get(str(edge_id)) for edge_id in path]
    if any(row is None for row in rows):
        return False
    typed_rows = [row for row in rows if row is not None]
    if str(typed_rows[0].get("src")) != source or str(typed_rows[-1].get("dst")) != target:
        return False
    return all(
        str(left.get("dst")) == str(right.get("src"))
        for left, right in zip(typed_rows, typed_rows[1:])
    )


def validate_exp03_bundle(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Validate experiment-03 evidence without erasing source semantics."""

    physical_nodes = bundle.get("physical_nodes", [])
    physical_edges = bundle.get("physical_edges", [])
    information_nodes = bundle.get("information_nodes", [])
    information_edges = bundle.get("information_edges", [])
    mn_relations = bundle.get("mn_relations", [])
    transfer_events = bundle.get("transfer_events", [])
    me_relations = bundle.get("me_relations", [])
    dependency_flows = bundle.get("dependency_flows", [])
    ep_relations = bundle.get("ep_relations", [])

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    physical_ids = {str(row.get("id")) for row in physical_nodes}
    physical_edge_by_id = {str(row.get("id")): row for row in physical_edges}
    info_ids = {str(row.get("id")) for row in information_nodes}
    info_edge_ids = {str(row.get("id")) for row in information_edges}
    physical_valid = (
        bool(physical_ids)
        and len(physical_ids) == len(physical_nodes)
        and len(physical_edge_by_id) == len(physical_edges)
        and all(
            str(row.get("src")) in physical_ids and str(row.get("dst")) in physical_ids
            for row in physical_edges
        )
    )
    check("physical_graph_valid", physical_valid, "Physical nodes and edge endpoints are unique and valid.")

    info_endpoints = all(
        str(row.get("src")) in info_ids and str(row.get("dst")) in info_ids
        for row in information_edges
    )
    check("information_graph_nonempty", bool(info_ids) and bool(information_edges) and info_endpoints, "A non-empty DAG is required.")
    check("information_graph_is_dag", info_endpoints and _is_dag(info_ids, information_edges), "The information graph must be acyclic.")

    mn_counts: Counter[tuple[str, str]] = Counter()
    mn_refs_valid = True
    for row in mn_relations:
        task = str(row.get("task"))
        relation = str(row.get("relation"))
        node = str(row.get("physical_node"))
        mn_counts[(task, relation)] += 1
        mn_refs_valid &= (
            task in info_ids
            and relation in {"source", "host", "exec", "ret"}
            and node in physical_ids
            and row.get("evidence") in {"direct", "deterministic"}
        )
    mn_valid = mn_refs_valid and all(
        mn_counts[(task, "source")] == 1
        and mn_counts[(task, "host")] == 1
        and mn_counts[(task, "exec")] <= 1
        and mn_counts[(task, "ret")] <= 1
        for task in info_ids
    )
    check("mn_cardinality", mn_valid, "MN source/host are unique and exec/ret are at most one.")

    me_valid = bool(me_relations) and all(
        str(row.get("task")) in info_ids
        and row.get("relation") in {"in", "out"}
        and row.get("evidence") == "direct_runtime_channel_event"
        and _path_valid(list(row.get("path", [])), row.get("source"), row.get("target"), physical_edge_by_id)
        for row in me_relations
    )
    check("me_paths_valid", me_valid, "ME paths must be continuous paths assembled from runtime events.")

    direct_events = {
        str(row.get("event_id")): row
        for row in transfer_events
        if row.get("evidence") == "direct_runtime_channel_event"
    }
    me_channel_observed = bool(me_relations) and all(
        all(str(event_id) in direct_events for event_id in row.get("event_ids", []))
        and all(
            physical_edge_by_id.get(str(edge_id), {}).get("source_interface")
            == "AirFogSim channel manager"
            for edge_id in row.get("path", [])
        )
        for row in me_relations
    )
    check("me_channel_observed", me_channel_observed, "Every ME edge and event must be directly observed by the channel runtime.")

    ep_counts = Counter(str(row.get("info_edge")) for row in ep_relations)
    ep_contract = (
        bool(info_edge_ids)
        and set(ep_counts) == info_edge_ids
        and all(ep_counts[edge_id] == 1 for edge_id in info_edge_ids)
        and all(
            row.get("payload_semantics") == "shared_parent_output"
            and row.get("evidence_source") == PIJWM_EP_SOURCE
            and float(row.get("dependency_payload", 0.0) or 0.0) > 0
            and str(row.get("dependency_flow_id", "")) != ""
            and row.get("dependency_status") in {"pending", "arrived"}
            for row in ep_relations
        )
    )
    check("ep_contract_complete", ep_contract, "Every DAG edge must carry the frozen PI-JWM dependency contract.")

    flows_by_id = {str(row.get("dependency_flow_id")): row for row in dependency_flows}
    arrived = [row for row in ep_relations if row.get("dependency_status") == "arrived"]
    arrived_nonempty = any(float(row.get("dependency_payload", 0.0) or 0.0) > 0 for row in arrived)
    check("arrived_dependency_nonempty", arrived_nonempty, "At least one non-zero dependency must arrive in the frozen run.")

    ep_direct = bool(arrived) and all(
        str(row.get("dependency_flow_id")) in flows_by_id
        and flows_by_id[str(row.get("dependency_flow_id"))].get("flow_completed") is True
        and flows_by_id[str(row.get("dependency_flow_id"))].get("evidence")
        in {"direct_runtime_channel_event", "direct_zero_hop_state"}
        and _path_valid(
            list(flows_by_id[str(row.get("dependency_flow_id"))].get("path", [])),
            flows_by_id[str(row.get("dependency_flow_id"))].get("source"),
            flows_by_id[str(row.get("dependency_flow_id"))].get("target"),
            physical_edge_by_id,
        )
        and all(
            str(event_id) in direct_events
            and direct_events[str(event_id)].get("phase") == "return"
            for event_id in flows_by_id[str(row.get("dependency_flow_id"))].get("event_ids", [])
        )
        for row in arrived
    )
    check("ep_directly_observed", ep_direct, "Arrived dependencies must reference observed return events or a zero-hop state.")

    accounting_errors = validate_shared_flow_accounting(ep_relations, dependency_flows)
    check("shared_flow_accounting", not accounting_errors, "Shared parent-output flows must be counted once.")

    failed_checks = [row["name"] for row in checks if not row["passed"]]
    return {
        "experiment_completed": True,
        "strict_dual_graph_ready": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "counts": {
            "physical_nodes": len(physical_nodes),
            "physical_edges": len(physical_edges),
            "information_nodes": len(information_nodes),
            "information_edges": len(information_edges),
            "transfer_events": len(transfer_events),
            "me_relations": len(me_relations),
            "dependency_flows": len(dependency_flows),
            "ep_relations": len(ep_relations),
            "arrived_ep_relations": len(arrived),
        },
    }


def build_exp03_corruption_report(
    bundle: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prove that four evidence corruptions are rejected."""

    cases: list[dict[str, Any]] = []

    missing_event = copy.deepcopy(bundle)
    first_me_event = str(missing_event["me_relations"][0]["event_ids"][0])
    missing_event["transfer_events"] = [
        row for row in missing_event["transfer_events"] if str(row.get("event_id")) != first_me_event
    ]

    wrong_phase = copy.deepcopy(bundle)
    parent_event_ids = set(wrong_phase["dependency_flows"][0].get("event_ids", []))
    for event in wrong_phase["transfer_events"]:
        if event.get("event_id") in parent_event_ids:
            event["phase"] = "offload"

    disconnected = copy.deepcopy(bundle)
    disconnected["me_relations"][0]["path"] = ["pe::p0::p2"]

    duplicated = copy.deepcopy(bundle)
    duplicated["dependency_flows"].append(copy.deepcopy(duplicated["dependency_flows"][0]))

    for name, corrupted in (
        ("missing_channel_event", missing_event),
        ("offload_as_parent_output", wrong_phase),
        ("disconnected_me_path", disconnected),
        ("duplicate_shared_flow", duplicated),
    ):
        report = validate_exp03_bundle(corrupted)
        cases.append(
            {
                "case": name,
                "detected": not report["strict_dual_graph_ready"],
                "failed_checks": report["failed_checks"],
            }
        )
    return {
        "all_corruptions_detected": all(row["detected"] for row in cases),
        "cases": cases,
    }


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_code_metadata() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    code_root = script_path.parents[2]
    repository_root = code_root.parent
    preflight_path = script_path.parent / "airfogsim_strict_dual_graph_preflight.py"
    airfogsim_root = code_root / "reference" / "AirFogSim"
    source_paths = [
        airfogsim_root / "airfogsim" / "airfogsim_env.py",
        airfogsim_root / "airfogsim" / "entities" / "task.py",
        airfogsim_root / "airfogsim" / "manager" / "task_manager.py",
        airfogsim_root / "airfogsim" / "manager" / "channel_manager_cp.py",
        airfogsim_root / "airfogsim" / "manager" / "energy_manager.py",
        airfogsim_root / "examples" / "config.yaml",
        airfogsim_root / "examples" / "export_strict_actions_v0.py",
    ]
    contract_adapter_path = code_root / "src" / "pi_jwm" / "airfogsim_contract_adapter.py"
    aggregate = hashlib.sha256()
    relative_sources: list[str] = []
    for path in sorted(source_paths):
        relative = path.relative_to(repository_root).as_posix()
        relative_sources.append(relative)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(path.read_bytes())
        aggregate.update(b"\0")
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        repository_head = None
    return {
        "repository_head": repository_head,
        "exp03_script": script_path.relative_to(repository_root).as_posix(),
        "exp03_script_sha256": _sha256_file(script_path),
        "exp02_preflight_script": preflight_path.relative_to(repository_root).as_posix(),
        "exp02_preflight_script_sha256": _sha256_file(preflight_path),
        "contract_adapter": contract_adapter_path.relative_to(repository_root).as_posix(),
        "contract_adapter_sha256": _sha256_file(contract_adapter_path),
        "contract_adapter_version": ADAPTER_VERSION,
        "airfogsim_source_files": relative_sources,
        "airfogsim_source_sha256": aggregate.hexdigest(),
    }


def _environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "networkx", "pandas", "PyYAML", "traci"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        completed = subprocess.run(
            ["sumo", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        sumo_version = completed.stdout.splitlines()[0] if completed.returncode == 0 else None
    except OSError:
        sumo_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "sumo": sumo_version,
        "scope": "local_environment_snapshot_not_a_complete_transitive_lockfile",
    }


def _report_markdown(validation: dict[str, Any], runtime_summary: dict[str, Any]) -> str:
    lines = [
        "# 小实验03：AirFogSim跨图证据闭合",
        "",
        "## 结论",
        "",
        f"- 实验流程完成：{'是' if validation['experiment_completed'] else '否'}",
        f"- 严格双图数据就绪：{'是' if validation['strict_dual_graph_ready'] else '否'}",
        "- DAG载荷语义来源：PI-JWM `shared_parent_output`扩展，不是AirFogSim原生DAG载荷",
        "",
        "## 运行计数",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in sorted(runtime_summary.items()):
        if isinstance(value, (int, float, bool)):
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 严格检查", "", "| 检查 | 状态 | 说明 |", "| --- | --- | --- |"])
    for row in validation["checks"]:
        lines.append(f"| {row['name']} | {'通过' if row['passed'] else '失败'} | {row['details']} |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "ME路径只来自实际信道执行事件；EP使用运行前冻结的PI-JWM共享父输出语义并引用实际父任务返回事件或合法零跳状态。该实验不训练模型，也不证明真实网络泛化。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_exp03(
    output_dir: Path,
    seed: int,
    max_time: float,
    runtime_runner: Any,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = runtime_runner(int(seed), float(max_time))
    repeat = runtime_runner(int(seed), float(max_time))
    bundle = primary["bundle"]
    validation = validate_exp03_bundle(bundle)
    corruption = build_exp03_corruption_report(bundle)
    bundle_hash = canonical_json_hash(bundle)
    repeat_bundle_hash = canonical_json_hash(repeat["bundle"])
    config_hash = canonical_json_hash(primary["config"])
    repeat_config_hash = canonical_json_hash(repeat["config"])
    reproducible = bundle_hash == repeat_bundle_hash and config_hash == repeat_config_hash

    validation["checks"].extend(
        [
            {
                "name": "same_seed_reproducible",
                "passed": reproducible,
                "details": "Same-seed config and normalized bundle hashes must match in the current environment.",
            },
            {
                "name": "corruption_detection",
                "passed": corruption["all_corruptions_detected"],
                "details": "All four evidence corruptions must be rejected.",
            },
        ]
    )
    validation["failed_checks"] = [row["name"] for row in validation["checks"] if not row["passed"]]
    validation["strict_dual_graph_ready"] = not validation["failed_checks"]
    validation["reproducibility_passed"] = reproducible
    validation["corruption_detection_passed"] = corruption["all_corruptions_detected"]
    validation["bundle_hash"] = bundle_hash

    _write_json(output_dir / "bundle.json", bundle)
    _write_json(output_dir / "config_snapshot.json", primary["config"])
    _write_json(output_dir / "corruption_report.json", corruption)
    _write_json(output_dir / "runtime_summary.json", primary["runtime_summary"])
    _write_json(output_dir / "validation_report.json", validation)
    for key in (
        "physical_nodes",
        "physical_edges",
        "physical_node_snapshots",
        "physical_edge_snapshots",
        "task_snapshots",
        "information_nodes",
        "information_edges",
        "mn_relations",
        "transfer_events",
        "offload_actions",
        "return_actions",
        "rb_actions",
        "me_relations",
        "dependency_flows",
        "ep_relations",
    ):
        _write_csv(output_dir / f"{key}.csv", bundle.get(key, []))
    (output_dir / "REPORT.md").write_text(
        _report_markdown(validation, primary["runtime_summary"]),
        encoding="utf-8",
    )

    frozen_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": "AirFogSim-PIJWM-exp03-v1",
        "seed": int(seed),
        "max_time": float(max_time),
        "bundle_hash": bundle_hash,
        "config_hash": config_hash,
        "reproducibility": {
            "repeat_bundle_hash": repeat_bundle_hash,
            "repeat_config_hash": repeat_config_hash,
            "same_seed_bundle_hash_equal": bundle_hash == repeat_bundle_hash,
            "same_seed_config_hash_equal": config_hash == repeat_config_hash,
        },
        "source_code": _source_code_metadata(),
        "environment": _environment_metadata(),
        "files": {path.name: _sha256_file(path) for path in frozen_files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        **validation,
        "output_dir": str(output_dir),
    }


def all_directed_link_rows(
    env: Any,
    active: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    rate_reader: Any,
    csi_reader: Any,
) -> list[dict[str, Any]]:
    """Export every directed V/U/I channel family exposed by AirFogSim."""

    groups = {
        "V": env.vehicles,
        "U": env.UAVs,
        "I": env.RSUs,
    }
    rows: list[dict[str, Any]] = []
    for source_type in ("V", "U", "I"):
        for target_type in ("V", "U", "I"):
            link_type = f"{source_type}2{target_type}"
            for source in groups[source_type]:
                for target in groups[target_type]:
                    if source == target:
                        continue
                    active_info = active.get((source, target, link_type), {})
                    rows.append(
                        {
                            "time": round(float(env.simulation_time), 3),
                            "tx_id": str(source),
                            "rx_id": str(target),
                            "link_type": link_type,
                            "distance": float(env.getDistanceBetweenNodesById(source, target)),
                            "rate_sum": float(rate_reader(env, source, target, link_type)),
                            "csi_mean": float(csi_reader(env, source, target)),
                            "active_task_count": int(active_info.get("task_count", 0)),
                            "allocated_rb_count": int(active_info.get("rb_count", 0)),
                        }
                    )
    return rows


def run_airfogsim_evidence_seed(seed: int, max_time: float) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    reference_root = code_root / "reference" / "AirFogSim"
    example_dir = reference_root / "examples"
    script_dir = Path(__file__).resolve().parent
    for path in (script_dir, reference_root, example_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np
    import yaml
    from airfogsim import AirFogSimEnv
    from airfogsim.scheduler import RewardScheduler, TaskScheduler
    from export_dataset_demo import (
        active_link_counts,
        channel_csi_mean,
        channel_rate_sum,
        node_rows,
    )
    from export_strict_actions_v0 import LoggingAlgorithmModule, action_effect_time, as_float, node_type_from_id
    import airfogsim_strict_dual_graph_preflight as preflight

    recorder = WirelessTransferEventRecorder()

    class ObservedAirFogSimEnv(AirFogSimEnv):
        def __init__(self, *args: Any, **kwargs: Any):
            self.pi_jwm_transfer_events: list[dict[str, Any]] = []
            self.pi_jwm_event_sequence: dict[tuple[str, str], int] = defaultdict(int)
            super().__init__(*args, **kwargs)

        def _updateWirelessCommunication(self) -> None:
            activated = self._allocate_communication_RBs(self.activated_offloading_tasks_with_RB_Nos)
            self._compute_communication_rate(activated)
            pending_events: list[dict[str, Any]] = []
            for profile in activated.values():
                task = profile["task"]
                route = list(task.getToOffloadRoute())
                if not route:
                    continue
                phase = "return" if task.isReturning() else "offload"
                task_id = str(task.getTaskId())
                sequence_key = (task_id, phase)
                event_profile = dict(profile, task_id=task_id)
                pending_events.append(
                    recorder.make_event(
                        self,
                        event_profile,
                        {
                            "task_id": task_id,
                            "phase": phase,
                            "source": str(task.getCurrentNodeId()),
                            "target": str(route[0]),
                            "transmitted_before": float(task.getTransmittedSize()),
                            "required_size": float(
                                task.getReturnedSize() if phase == "return" else task.getTaskSize()
                            ),
                            "sequence": self.pi_jwm_event_sequence[sequence_key],
                        },
                    )
                )
                self.pi_jwm_event_sequence[sequence_key] += 1
            self._execute_communication(activated)
            repair_channel_energy_inputs(self.channel_manager, pending_events)
            self.pi_jwm_transfer_events.extend(pending_events)

    class EvidenceLoggingAlgorithm(LoggingAlgorithmModule):
        def scheduleComputing(self, env: Any) -> None:
            install_capacity_safe_cpu_callback(env, self.compScheduler)

        def scheduleOffloading(self, env: Any) -> None:
            decisions = preflight.select_ready_offload_decisions(
                env,
                self.taskScheduler,
                self.entityScheduler,
            )
            for decision in decisions:
                flag = self.taskScheduler.setTaskOffloading(
                    env,
                    decision["task_node_id"],
                    decision["task_id"],
                    decision["target_node_id"],
                    route=[decision["target_node_id"]],
                )
                if not flag:
                    continue
                self.offload_rows.append(
                    {
                        "seed": self.seed,
                        "time": action_effect_time(env),
                        "task_id": decision["task_id"],
                        "task_node_id": decision["task_node_id"],
                        "source_node_id": decision["source_node_id"],
                        "target_node_id": decision["target_node_id"],
                        "target_node_type": node_type_from_id(decision["target_node_id"]),
                        "nearest_distance": as_float(decision["distance"]),
                        "route_nodes": decision["route_nodes"],
                        "evidence": "direct_scheduler_decision",
                    }
                )

        def scheduleReturning(self, env: Any) -> None:
            waiting = self.taskScheduler.getWaitingToReturnTaskInfos(env)
            for task_node_id, tasks in waiting.items():
                for task in tasks:
                    current_node_id = str(task.getCurrentNodeId())
                    source_node_id = str(task.getTaskNodeId())
                    return_route = [source_node_id]
                    self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)
                    self.return_rows.append(
                        {
                            "seed": self.seed,
                            "time": action_effect_time(env),
                            "task_id": str(task.getTaskId()),
                            "task_node_id": str(task_node_id),
                            "current_node_id": current_node_id,
                            "return_target_id": source_node_id,
                            "route_nodes": [current_node_id, source_node_id]
                            if current_node_id != source_node_id
                            else [source_node_id],
                            "evidence": "direct_scheduler_decision",
                        }
                    )

    config_path = example_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = preflight.build_preflight_config(config, int(seed), float(max_time))
    config["pi_jwm_exp03"] = {
        "schema_version": "AirFogSim-PIJWM-exp03-v1",
        "contract_adapter_version": ADAPTER_VERSION,
        "cpu_semantics": "per_node_capacity_safe_equal_share_max_three",
        "channel_energy_semantics": "direct_event_source_target_accounting",
        "dependency_semantics": "shared_parent_output",
        "evidence_source": PIJWM_EP_SOURCE,
        "trajectory_id": f"exp03_seed_{int(seed)}_time_{float(max_time):g}",
    }
    trajectory_id = config["pi_jwm_exp03"]["trajectory_id"]
    np.random.seed(int(seed))
    random.seed(int(seed))

    physical_node_records: dict[str, dict[str, Any]] = {}
    physical_edge_records: dict[str, dict[str, Any]] = {}
    physical_node_snapshots: list[dict[str, Any]] = []
    physical_edge_snapshots: list[dict[str, Any]] = []
    task_records: dict[str, dict[str, Any]] = {}
    task_snapshots: list[dict[str, Any]] = []
    dag_edge_records: dict[str, dict[str, Any]] = {}
    step = 0
    old_cwd = Path.cwd()
    env = None
    try:
        os.chdir(example_dir)
        env = ObservedAirFogSimEnv(config, interactive_mode=None)
        env.task_manager.getOffloadingTasksWithNumber = types.MethodType(
            lambda manager: nonmutating_transmission_tasks(manager),
            env.task_manager,
        )
        algorithm = EvidenceLoggingAlgorithm(int(seed))
        algorithm.initialize(env)
        RewardScheduler.setModel(env, "REWARD", "1/task_delay")
        while not env.isDone():
            algorithm.scheduleStep(env)
            active = active_link_counts(env)
            env.step()
            step += 1
            current_time = float(env.simulation_time)
            for row in node_rows(env):
                normalized = preflight._node_record(row, trajectory_id)
                physical_node_snapshots.append(normalized)
                physical_node_records[normalized["id"]] = normalized
            for row in all_directed_link_rows(
                env,
                active,
                rate_reader=channel_rate_sum,
                csi_reader=channel_csi_mean,
            ):
                normalized = preflight._physical_edge_record(row, trajectory_id)
                physical_edge_snapshots.append(normalized)
                physical_edge_records[normalized["id"]] = normalized
            for task in preflight.iter_airfogsim_tasks(env.task_manager):
                normalized = preflight._task_record(task, trajectory_id, current_time)
                task_snapshots.append(copy.deepcopy(normalized))
                task_records[normalized["id"]] = normalized
            for row in preflight.normalize_airfogsim_dags(
                TaskScheduler.getAllTaskDAGs(env),
                trajectory_id=trajectory_id,
                step=step,
                time_value=current_time,
            ):
                dag_edge_records.setdefault(row["id"], row)
        transfer_events = copy.deepcopy(env.pi_jwm_transfer_events)
        offload_actions = copy.deepcopy(algorithm.offload_rows)
        return_actions = copy.deepcopy(algorithm.return_rows)
        rb_actions = copy.deepcopy(algorithm.rb_rows)
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)

    for event in transfer_events:
        for edge_id in event.get("path", []):
            source = str(event["source"])
            target = str(event["target"])
            physical_edge_records[str(edge_id)] = {
                "trajectory_id": trajectory_id,
                "id": str(edge_id),
                "src": source,
                "dst": target,
                "kind": str(event.get("channel_type", "unknown")),
                "rate_sum": sum(float(value) for value in event.get("rate_by_rb", [])),
                "active_task_count": 1,
                "allocated_rb_count": len(event.get("rb_indices", [])),
                "observed_time": float(event.get("time", 0.0)),
                "evidence": "direct",
                "source_interface": "AirFogSim channel manager",
            }

    base_bundle = preflight.assemble_airfogsim_bundle(
        trajectory_id=trajectory_id,
        physical_nodes=list(physical_node_records.values()),
        physical_edges=list(physical_edge_records.values()),
        task_records=list(task_records.values()),
        dag_edges=list(dag_edge_records.values()),
        offload_actions=offload_actions,
        return_actions=return_actions,
    )
    ep_relations, dependency_flows = build_shared_parent_output_relations(
        base_bundle["information_edges"],
        base_bundle["information_nodes"],
        transfer_events,
    )
    bundle = {
        **base_bundle,
        "physical_node_snapshots": physical_node_snapshots,
        "physical_edge_snapshots": physical_edge_snapshots,
        "task_snapshots": task_snapshots,
        "transfer_events": transfer_events,
        "offload_actions": offload_actions,
        "return_actions": return_actions,
        "rb_actions": rb_actions,
        "me_relations": build_observed_me_relations(transfer_events),
        "dependency_flows": dependency_flows,
        "ep_relations": ep_relations,
    }
    runtime_summary = {
        "seed": int(seed),
        "steps": int(step),
        "max_time": float(max_time),
        "physical_nodes": len(bundle["physical_nodes"]),
        "physical_edges": len(bundle["physical_edges"]),
        "physical_node_snapshots": len(bundle["physical_node_snapshots"]),
        "physical_edge_snapshots": len(bundle["physical_edge_snapshots"]),
        "task_snapshots": len(bundle["task_snapshots"]),
        "information_nodes": len(bundle["information_nodes"]),
        "information_edges": len(bundle["information_edges"]),
        "transfer_events": len(transfer_events),
        "offload_actions": len(offload_actions),
        "return_actions": len(return_actions),
        "rb_actions": len(rb_actions),
        "return_transfer_events": sum(row.get("phase") == "return" for row in transfer_events),
        "me_relations": len(bundle["me_relations"]),
        "dependency_flows": len(dependency_flows),
        "ep_relations": len(ep_relations),
        "arrived_ep_relations": sum(row.get("dependency_status") == "arrived" for row in ep_relations),
        "pending_ep_relations": sum(row.get("dependency_status") == "pending" for row in ep_relations),
    }
    return {"config": config, "bundle": bundle, "runtime_summary": runtime_summary}


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "small_experiments"
        / "exp03_airfogsim_cross_graph_evidence"
        / "evidence_v1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PI-JWM experiment 03 cross-graph evidence closure.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    result = run_exp03(
        output_dir=args.output_dir,
        seed=args.seed,
        max_time=args.max_time,
        runtime_runner=run_airfogsim_evidence_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
