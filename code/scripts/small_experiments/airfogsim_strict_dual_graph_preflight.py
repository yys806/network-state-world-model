from __future__ import annotations

"""AirFogSim evidence-first preflight for the PI-JWM strict dual graph."""

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
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "AirFogSim-PIJWM-preflight-v1"


def _without_keys(value: Any, exclude_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_keys(item, exclude_keys)
            for key, item in value.items()
            if key not in exclude_keys
        }
    if isinstance(value, list):
        return [_without_keys(item, exclude_keys) for item in value]
    if isinstance(value, tuple):
        return [_without_keys(item, exclude_keys) for item in value]
    return value


def canonical_json_hash(payload: Any, exclude_keys: set[str] | None = None) -> str:
    normalized = _without_keys(payload, set(exclude_keys or ()))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preflight_config(config: dict[str, Any], seed: int, max_time: float) -> dict[str, Any]:
    configured = copy.deepcopy(config)
    configured["simulation"]["max_simulation_time"] = float(max_time)
    configured["traffic"]["max_n_vehicles"] = 20
    configured["traffic"]["max_n_UAVs"] = 2
    configured["traffic"]["RSU_positions"] = [
        [100, 100, 0],
        [700, 100, 0],
        [100, 700, 0],
        [700, 700, 0],
    ]

    task = configured["task"]
    task["task_generation_kwargs"]["lambda"] = 1.5
    task["task_min_required_returned_size"] = 0.05
    task["task_max_required_returned_size"] = 0.10
    task["required_returned_size_kwargs"].update({"low": 0.05, "high": 0.10})
    task["task_min_deadline"] = 8.0
    task["task_max_deadline"] = 12.0
    task["deadline_kwargs"].update({"low": 8.0, "high": 12.0})

    profile = configured["task_profile"]
    profile["task_node_gen_poss"] = 1.0
    profile["task_node_profiles"] = [
        {"type": "UAV", "max_node_num": 2},
        {"type": "vehicle", "max_node_num": 6},
    ]
    for node_kind in ("vehicle", "uav"):
        profile[node_kind]["lambda"] = 2.0
        profile[node_kind]["dag_edge_prob"] = 0.6

    configured["pi_jwm_preflight"] = {
        "schema_version": SCHEMA_VERSION,
        "simulator": "AirFogSim",
        "seed": int(seed),
        "evidence_policy": "direct_or_deterministic_only",
    }
    digest = canonical_json_hash(configured, exclude_keys={"trajectory_id", "generated_at"})
    configured["pi_jwm_preflight"]["trajectory_id"] = (
        f"afs-pijwm-preflight-v1-s{int(seed):03d}-{digest[:12]}"
    )
    return configured


def physical_edge_id(source: str, target: str) -> str:
    return f"pe::{source}::{target}"


def information_edge_id(source: str, target: str) -> str:
    return f"ie::{source}::{target}"


def route_nodes_to_edges(route_nodes: list[str]) -> list[str]:
    return [
        physical_edge_id(str(source), str(target))
        for source, target in zip(route_nodes, route_nodes[1:])
    ]


def normalize_airfogsim_dags(
    dag_by_task_node: dict[str, Any],
    trajectory_id: str,
    step: int,
    time_value: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for task_node_id, dag in sorted(dag_by_task_node.items()):
        if dag is None:
            continue
        for source, target in sorted(dag.edges()):
            key = (str(source), str(target))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "step": int(step),
                    "time": round(float(time_value), 6),
                    "task_node_id": str(task_node_id),
                    "id": information_edge_id(*key),
                    "src": key[0],
                    "dst": key[1],
                    "data_mb": None,
                    "semantic": "precedence_only",
                    "evidence": "direct",
                    "source_interface": "TaskScheduler.getAllTaskDAGs",
                }
            )
    return rows


def classify_ep_evidence(
    information_edge: dict[str, Any],
    source: str | None,
    target: str | None,
    transfer_event: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        transfer_event is not None
        and transfer_event.get("evidence") == "direct"
        and transfer_event.get("data_mb") is not None
        and transfer_event.get("path") is not None
    ):
        return {
            "info_edge": information_edge["id"],
            "source": source,
            "target": target,
            "path": list(transfer_event["path"]),
            "data_mb": float(transfer_event["data_mb"]),
            "evidence": "direct",
            "source_interface": transfer_event.get("source_interface", "runtime_transfer_event"),
        }
    return {
        "info_edge": information_edge["id"],
        "source": source,
        "target": target,
        "path": [],
        "data_mb": None,
        "evidence": "not_modeled",
        "source_interface": "AirFogSim_DAG_precedence_without_dependency_transfer",
    }


def _is_dag(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    indegree = {node_id: 0 for node_id in node_ids}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("src")
        target = edge.get("dst")
        if source not in indegree or target not in indegree:
            return False
        successors[source].append(target)
        indegree[target] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for target in successors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(node_ids)


def validate_directed_path(
    path: list[str],
    source: str | None,
    target: str | None,
    physical_edge_by_id: dict[str, dict[str, Any]],
) -> bool:
    if source is None or target is None:
        return False
    if source == target:
        return path == []
    if not path or len(path) != len(set(path)):
        return False
    try:
        edges = [physical_edge_by_id[edge_id] for edge_id in path]
    except KeyError:
        return False
    if edges[0].get("src") != source or edges[-1].get("dst") != target:
        return False
    if any(left.get("dst") != right.get("src") for left, right in zip(edges, edges[1:])):
        return False
    visited_nodes = [edges[0]["src"]] + [edge["dst"] for edge in edges]
    if len(visited_nodes) != len(set(visited_nodes)):
        return False
    balance: Counter[str] = Counter()
    for edge in edges:
        balance[edge["src"]] += 1
        balance[edge["dst"]] -= 1
    if balance[source] != 1 or balance[target] != -1:
        return False
    return all(value == 0 for node, value in balance.items() if node not in {source, target})


def validate_export_bundle(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    physical_nodes = bundle.get("physical_nodes", [])
    physical_edges = bundle.get("physical_edges", [])
    information_nodes = bundle.get("information_nodes", [])
    information_edges = bundle.get("information_edges", [])
    mn_relations = bundle.get("mn_relations", [])
    me_relations = bundle.get("me_relations", [])
    ep_relations = bundle.get("ep_relations", [])

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    physical_node_ids = {str(row.get("id")) for row in physical_nodes}
    physical_edge_by_id = {str(row.get("id")): row for row in physical_edges}
    information_node_ids = {str(row.get("id")) for row in information_nodes}
    information_edge_by_id = {str(row.get("id")): row for row in information_edges}

    physical_valid = (
        bool(physical_node_ids)
        and len(physical_node_ids) == len(physical_nodes)
        and len(physical_edge_by_id) == len(physical_edges)
        and all(
            row.get("src") in physical_node_ids and row.get("dst") in physical_node_ids
            for row in physical_edges
        )
    )
    check("physical_graph_valid", physical_valid, "Physical IDs are unique and edge endpoints exist.")

    info_endpoints_valid = all(
        row.get("src") in information_node_ids and row.get("dst") in information_node_ids
        for row in information_edges
    )
    check(
        "information_graph_nonempty",
        bool(information_node_ids) and bool(information_edges) and info_endpoints_valid,
        "At least one directly exported DAG edge with valid endpoints is required.",
    )
    check(
        "information_graph_is_dag",
        info_endpoints_valid and _is_dag(information_node_ids, information_edges),
        "The exported information graph must be acyclic.",
    )

    mn_counts: Counter[tuple[str, str]] = Counter()
    mn_refs_valid = True
    exec_mapping: dict[str, str] = {}
    for row in mn_relations:
        task = row.get("task")
        relation = row.get("relation")
        node = row.get("physical_node")
        mn_counts[(task, relation)] += 1
        mn_refs_valid &= (
            task in information_node_ids
            and relation in {"source", "host", "exec", "ret"}
            and node in physical_node_ids
            and row.get("evidence") in {"direct", "deterministic"}
        )
        if relation == "exec" and mn_counts[(task, relation)] == 1:
            exec_mapping[str(task)] = str(node)
    mn_cardinality = mn_refs_valid and all(
        mn_counts[(task, "source")] == 1
        and mn_counts[(task, "host")] == 1
        and mn_counts[(task, "exec")] <= 1
        and mn_counts[(task, "ret")] <= 1
        for task in information_node_ids
    )
    check(
        "mn_cardinality",
        mn_cardinality,
        "In each task's last-observed record, source/host are exactly one and exec/ret are at most one.",
    )

    me_valid = bool(me_relations) and all(
        row.get("task") in information_node_ids
        and row.get("relation") in {"in", "out"}
        and row.get("evidence") == "direct"
        and validate_directed_path(
            list(row.get("path", [])),
            row.get("source"),
            row.get("target"),
            physical_edge_by_id,
        )
        for row in me_relations
    )
    check(
        "me_paths_valid",
        me_valid,
        "Every exported scheduler-selected lifecycle route must be structurally continuous and conserved.",
    )
    me_channel_observed = bool(me_relations) and all(
        all(
            physical_edge_by_id.get(edge_id, {}).get("source_interface")
            == "AirFogSim channel manager"
            for edge_id in row.get("path", [])
        )
        for row in me_relations
    )
    check(
        "me_channel_observed",
        me_channel_observed,
        "Every non-zero-hop ME edge must also be present in the exported channel graph; scheduler selection alone is insufficient.",
    )

    ep_by_edge = {row.get("info_edge"): row for row in ep_relations}
    ep_direct = bool(information_edges) and all(
        edge_id in ep_by_edge
        and ep_by_edge[edge_id].get("evidence") == "direct"
        and info_edge.get("data_mb") is not None
        and ep_by_edge[edge_id].get("data_mb") is not None
        and ep_by_edge[edge_id].get("source") == exec_mapping.get(str(info_edge.get("src")))
        and ep_by_edge[edge_id].get("target") == exec_mapping.get(str(info_edge.get("dst")))
        and validate_directed_path(
            list(ep_by_edge[edge_id].get("path", [])),
            ep_by_edge[edge_id].get("source"),
            ep_by_edge[edge_id].get("target"),
            physical_edge_by_id,
        )
        for edge_id, info_edge in information_edge_by_id.items()
    )
    check(
        "ep_directly_observed",
        ep_direct,
        "Each DAG payload and its physical carrying path must be directly observed; precedence-only is insufficient.",
    )

    failed_checks = [row["name"] for row in checks if not row["passed"]]
    counts = {
        "physical_nodes": len(physical_nodes),
        "physical_edges": len(physical_edges),
        "information_nodes": len(information_nodes),
        "information_edges": len(information_edges),
        "mn_relations": len(mn_relations),
        "me_relations": len(me_relations),
        "ep_relations": len(ep_relations),
        "ep_not_modeled": sum(row.get("evidence") == "not_modeled" for row in ep_relations),
    }
    return {
        "experiment_completed": True,
        "strict_dual_graph_ready": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "counts": counts,
    }


def select_ready_offload_decisions(env: Any, task_scheduler: Any, entity_scheduler: Any) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    ready_tasks = task_scheduler.getAllToOffloadTaskInfos(env, check_dependency=True)
    for task in ready_tasks:
        task_node_id = str(task["task_node_id"])
        neighbors = entity_scheduler.getNeighborNodeInfosById(
            env,
            task_node_id,
            sorted_by="distance",
            max_num=5,
        )
        if not neighbors:
            continue
        target = neighbors[0]
        target_id = str(target["id"])
        decisions.append(
            {
                "task_node_id": task_node_id,
                "task_id": str(task["task_id"]),
                "source_node_id": task_node_id,
                "target_node_id": target_id,
                "route_nodes": [task_node_id, target_id],
                "distance": float(target.get("distance", 0.0)),
            }
        )
    return decisions


def _deduplicate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        selected[tuple(row.get(key) for key in keys)] = copy.deepcopy(row)
    return [selected[key] for key in sorted(selected, key=lambda item: tuple(str(value) for value in item))]


def assemble_airfogsim_bundle(
    trajectory_id: str,
    physical_nodes: list[dict[str, Any]],
    physical_edges: list[dict[str, Any]],
    task_records: list[dict[str, Any]],
    dag_edges: list[dict[str, Any]],
    offload_actions: list[dict[str, Any]],
    return_actions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nodes = _deduplicate_rows(physical_nodes, ("id",))
    edges = _deduplicate_rows(physical_edges, ("id",))
    tasks = _deduplicate_rows(task_records, ("id",))
    info_edges = _deduplicate_rows(dag_edges, ("id",))

    mn_relations: list[dict[str, Any]] = []
    exec_mapping: dict[str, str] = {}
    for task in tasks:
        task_id = str(task["id"])
        for relation, field in (
            ("source", "source"),
            ("host", "host"),
            ("exec", "exec"),
            ("ret", "ret"),
        ):
            node = task.get(field)
            if node in (None, ""):
                continue
            mn_relations.append(
                {
                    "trajectory_id": trajectory_id,
                    "task": task_id,
                    "relation": relation,
                    "physical_node": str(node),
                    "evidence": task.get("evidence", "direct"),
                    "observed_time": task.get("observed_time"),
                }
            )
            if relation == "exec":
                exec_mapping[task_id] = str(node)

    me_relations: list[dict[str, Any]] = []
    for relation_name, actions, source_field, target_field in (
        ("in", offload_actions, "source_node_id", "target_node_id"),
        ("out", return_actions, "current_node_id", "return_target_id"),
    ):
        for action in actions:
            source = str(action[source_field])
            target = str(action[target_field])
            route_nodes = [str(item) for item in action.get("route_nodes", [source, target])]
            path = [] if source == target else route_nodes_to_edges(route_nodes)
            me_relations.append(
                {
                    "trajectory_id": trajectory_id,
                    "task": str(action["task_id"]),
                    "relation": relation_name,
                    "source": source,
                    "target": target,
                    "path": path,
                    "evidence": action.get("evidence", "direct"),
                }
            )

    ep_relations = [
        classify_ep_evidence(
            edge,
            source=exec_mapping.get(str(edge.get("src"))),
            target=exec_mapping.get(str(edge.get("dst"))),
            transfer_event=None,
        )
        for edge in info_edges
    ]

    return {
        "physical_nodes": nodes,
        "physical_edges": edges,
        "information_nodes": tasks,
        "information_edges": info_edges,
        "mn_relations": _deduplicate_rows(mn_relations, ("task", "relation")),
        "me_relations": _deduplicate_rows(me_relations, ("task", "relation")),
        "ep_relations": _deduplicate_rows(ep_relations, ("info_edge",)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key, "")) for key in fieldnames}
            for row in rows
        )


def build_corruption_report(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    cycle = copy.deepcopy(bundle)
    if cycle.get("information_edges"):
        first = cycle["information_edges"][0]
        cycle["information_edges"].append(
            {
                "id": information_edge_id(str(first["dst"]), str(first["src"])),
                "src": first["dst"],
                "dst": first["src"],
                "data_mb": None,
                "semantic": "precedence_only",
                "evidence": "direct",
            }
        )
    cycle_report = validate_export_bundle(cycle)
    cases.append(
        {
            "case": "dag_cycle",
            "detected": "information_graph_is_dag" in cycle_report["failed_checks"],
            "failed_checks": cycle_report["failed_checks"],
        }
    )

    duplicate_mn = copy.deepcopy(bundle)
    if duplicate_mn.get("mn_relations"):
        duplicate_mn["mn_relations"].append(copy.deepcopy(duplicate_mn["mn_relations"][0]))
    duplicate_report = validate_export_bundle(duplicate_mn)
    cases.append(
        {
            "case": "duplicate_mn",
            "detected": "mn_cardinality" in duplicate_report["failed_checks"],
            "failed_checks": duplicate_report["failed_checks"],
        }
    )

    broken_me = copy.deepcopy(bundle)
    if broken_me.get("me_relations"):
        broken_me["me_relations"][0]["path"] = ["pe::missing::edge"]
    broken_report = validate_export_bundle(broken_me)
    cases.append(
        {
            "case": "broken_me_path",
            "detected": "me_paths_valid" in broken_report["failed_checks"],
            "failed_checks": broken_report["failed_checks"],
        }
    )
    return {
        "all_corruptions_detected": all(case["detected"] for case in cases),
        "cases": cases,
    }


def _report_markdown(
    validation: dict[str, Any],
    runtime_summary: dict[str, Any],
    reproducible: bool,
    corruption_report: dict[str, Any],
) -> str:
    lines = [
        "# 小实验02：AirFogSim严格双图构造可行性",
        "",
        "## 结论",
        "",
        f"- 实验流程完成：{'是' if validation['experiment_completed'] else '否'}",
        f"- 严格双图数据就绪：{'是' if validation['strict_dual_graph_ready'] else '否'}",
        f"- 同seed规范化输出可重复：{'是' if reproducible else '否'}",
        f"- 三类破坏性输入均被拒绝：{'是' if corruption_report['all_corruptions_detected'] else '否'}",
        "- 当前已核验的AirFogSim任务、DAG和路由接口只直接提供DAG先后约束；本导出器未取得独立的DAG依赖载荷与跨执行节点传输事件，因此不能把最短路或返回路径补造成EP真值，也不把该结果表述为对模拟器全部潜在接口的不存在证明。",
        "",
        "## 运行计数",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(runtime_summary.items()) if isinstance(value, (int, float, bool)))
    lines.extend(
        [
            "",
            "## 严格检查",
            "",
            "| 检查 | 状态 | 说明 |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {row['name']} | {'通过' if row['passed'] else '失败'} | {row['details']} |"
        for row in validation["checks"]
    )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "本实验使用实际执行的AirFogSim轨迹验证导出能力，不训练模型、不证明双图优于单图，也不证明真实网络泛化。`strict_dual_graph_ready=false`是数据/模拟器语义结论，不是程序执行失败。",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_code_metadata() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    code_root = script_path.parents[2]
    repository_root = code_root.parent
    airfogsim_root = code_root / "reference" / "AirFogSim"
    source_paths = [
        airfogsim_root / "airfogsim" / "airfogsim_env.py",
        airfogsim_root / "airfogsim" / "entities" / "task.py",
        airfogsim_root / "airfogsim" / "manager" / "task_manager.py",
        airfogsim_root / "airfogsim" / "scheduler" / "task_sched.py",
        airfogsim_root / "examples" / "config.yaml",
        airfogsim_root / "examples" / "export_dataset_demo.py",
        airfogsim_root / "examples" / "export_strict_actions_v0.py",
    ]
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
        "preflight_script": script_path.relative_to(repository_root).as_posix(),
        "preflight_script_sha256": _sha256_file(script_path),
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
        sumo_version = subprocess.check_output(
            ["sumo", "--version"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        sumo_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
        "sumo": sumo_version,
        "scope": "local_environment_snapshot_not_a_complete_transitive_lockfile",
    }


def run_preflight_experiment(
    output_dir: Path,
    seed: int,
    max_time: float,
    runtime_runner: Any,
    corruption_builder: Any = build_corruption_report,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = runtime_runner(int(seed), float(max_time))
    repeat = runtime_runner(int(seed), float(max_time))
    bundle = primary["bundle"]
    bundle_hash = canonical_json_hash(bundle)
    repeat_hash = canonical_json_hash(repeat["bundle"])
    config_hash = canonical_json_hash(primary["config"])
    repeat_config_hash = canonical_json_hash(repeat["config"])
    bundle_reproducible = bundle_hash == repeat_hash
    config_reproducible = config_hash == repeat_config_hash
    reproducible = bundle_reproducible and config_reproducible
    validation = validate_export_bundle(bundle)
    validation["bundle_hash"] = bundle_hash
    validation["reproducibility_passed"] = reproducible
    corruption_report = corruption_builder(bundle)
    validation["checks"].extend(
        [
            {
                "name": "same_seed_reproducible",
                "passed": reproducible,
                "details": "Repeated config and normalized bundle hashes must match in the current local environment.",
            },
            {
                "name": "corruption_detection",
                "passed": bool(corruption_report.get("all_corruptions_detected", False)),
                "details": "Cycle, duplicate-MN, and broken-ME corruptions must all be rejected.",
            },
        ]
    )
    validation["failed_checks"] = [
        row["name"] for row in validation["checks"] if not row["passed"]
    ]
    validation["strict_dual_graph_ready"] = not validation["failed_checks"]

    _write_json(output_dir / "config_snapshot.json", primary["config"])
    _write_json(output_dir / "bundle.json", bundle)
    _write_json(output_dir / "runtime_summary.json", primary["runtime_summary"])
    _write_json(output_dir / "validation_report.json", validation)
    _write_json(output_dir / "corruption_report.json", corruption_report)
    for key in (
        "physical_nodes",
        "physical_edges",
        "information_nodes",
        "information_edges",
        "mn_relations",
        "me_relations",
        "ep_relations",
    ):
        _write_csv(output_dir / f"{key}.csv", bundle.get(key, []))
    (output_dir / "REPORT.md").write_text(
        _report_markdown(validation, primary["runtime_summary"], reproducible, corruption_report),
        encoding="utf-8",
    )

    artifact_paths = sorted(path for path in output_dir.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "max_time": float(max_time),
        "bundle_hash": bundle_hash,
        "config_hash": config_hash,
        "reproducibility": {
            "repeat_bundle_hash": repeat_hash,
            "repeat_config_hash": repeat_config_hash,
            "same_seed_bundle_hash_equal": bundle_reproducible,
            "same_seed_config_hash_equal": config_reproducible,
        },
        "source_code": _source_code_metadata(),
        "environment": _environment_metadata(),
        "files": {path.name: _sha256_file(path) for path in artifact_paths},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "experiment_completed": validation["experiment_completed"],
        "strict_dual_graph_ready": validation["strict_dual_graph_ready"],
        "failed_checks": validation["failed_checks"],
        "bundle_hash": bundle_hash,
        "output_dir": str(output_dir),
    }


def _node_record(row: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "id": str(row["node_id"]),
        "kind": str(row["node_type"]),
        "position": [float(row["x"]), float(row["y"]), float(row["z"])],
        "speed": float(row["speed"]),
        "acceleration": float(row["acceleration"]),
        "cpu": row.get("cpu"),
        "storage": row.get("storage"),
        "observed_time": float(row["time"]),
        "evidence": "direct",
        "source_interface": "AirFogSim entity objects",
    }


def _physical_edge_record(row: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
    source = str(row["tx_id"])
    target = str(row["rx_id"])
    return {
        "trajectory_id": trajectory_id,
        "id": physical_edge_id(source, target),
        "src": source,
        "dst": target,
        "kind": str(row.get("link_type", "unknown")),
        "distance": float(row.get("distance", 0.0)),
        "rate_sum": float(row.get("rate_sum", 0.0)),
        "csi_mean": float(row.get("csi_mean", 0.0)),
        "active_task_count": int(row.get("active_task_count", 0)),
        "allocated_rb_count": int(row.get("allocated_rb_count", 0)),
        "observed_time": float(row["time"]),
        "evidence": "direct",
        "source_interface": "AirFogSim channel manager",
    }


def _task_record(task: Any, trajectory_id: str, time_value: float) -> dict[str, Any]:
    failure_reason_value = str(task.getTaskFailureReason())
    failure_reason = None if failure_reason_value == "Unknown code." else failure_reason_value
    completed = bool(task.isFinished()) and failure_reason is None
    terminal_status = "failed" if failure_reason is not None else "completed" if completed else "pending"
    last_operation_time = float(task.getLastOperationTime())
    arrival_time = float(task.getTaskArrivalTime())
    deadline = float(task.getTaskDeadline())
    return {
        "trajectory_id": trajectory_id,
        "id": str(task.getTaskId()),
        "source": str(task.getTaskNodeId()),
        "host": str(task.getCurrentNodeId()),
        "exec": task.getAssignedTo(),
        "ret": task.getToReturnNodeId(),
        "task_size": float(task.getTaskSize()),
        "task_cpu": float(task.getTaskCPU()),
        "return_size": float(task.getReturnedSize()),
        "arrival_time": arrival_time,
        "deadline": deadline,
        "deadline_time": arrival_time + deadline,
        "priority": float(task.getTaskPriority()),
        "in_stage_transmitted_size": float(task.getTransmittedSize()),
        "computed_size": float(task.getComputedSize()),
        "last_transmission_time": float(task.getLastTransmissionTime()),
        "last_compute_time": float(task.getLastComputeTime()),
        "last_return_time": float(task.getLastReturnTime()),
        "last_operation_time": last_operation_time,
        "terminal_status": terminal_status,
        "completion_time": last_operation_time if completed else None,
        "task_delay": last_operation_time - arrival_time if completed else None,
        "failure_reason": failure_reason,
        "lifecycle_state": "failed" if failure_reason is not None else str(task.task_lifecycle_state),
        "observed_time": float(time_value),
        "evidence": "direct",
        "source_interface": "AirFogSim Task object",
    }


def iter_airfogsim_tasks(task_manager: Any) -> list[Any]:
    selected: dict[str, Any] = {}
    for task in task_manager.getAllTasks():
        selected[str(task.getTaskId())] = task
    for tasks in getattr(task_manager, "_to_generate_task_infos", {}).values():
        for task in tasks:
            selected.setdefault(str(task.getTaskId()), task)
    return [selected[task_id] for task_id in sorted(selected)]


def _add_route_edges(
    physical_edge_records: dict[str, dict[str, Any]],
    physical_node_records: dict[str, dict[str, Any]],
    trajectory_id: str,
    actions: list[dict[str, Any]],
    source_field: str,
    target_field: str,
) -> None:
    for action in actions:
        source = str(action[source_field])
        target = str(action[target_field])
        if source == target:
            action["route_nodes"] = [source]
            continue
        action["route_nodes"] = [source, target]
        edge_id = physical_edge_id(source, target)
        if source in physical_node_records and target in physical_node_records and edge_id not in physical_edge_records:
            physical_edge_records[edge_id] = {
                "trajectory_id": trajectory_id,
                "id": edge_id,
                "src": source,
                "dst": target,
                "kind": "scheduled_direct_route",
                "distance": float(action.get("nearest_distance", action.get("return_distance", 0.0))),
                "rate_sum": None,
                "csi_mean": None,
                "active_task_count": 1,
                "allocated_rb_count": None,
                "observed_time": float(action.get("time", 0.0)),
                "evidence": "direct_scheduler_decision",
                "source_interface": "AirFogSim scheduler route",
            }


def run_airfogsim_seed(seed: int, max_time: float) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    reference_root = code_root / "reference" / "AirFogSim"
    example_dir = reference_root / "examples"
    for path in (reference_root, example_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np
    import yaml
    from airfogsim import AirFogSimEnv
    from airfogsim.scheduler import RewardScheduler, TaskScheduler
    from export_dataset_demo import active_link_counts, link_rows, node_rows
    from export_strict_actions_v0 import LoggingAlgorithmModule, as_float, node_type_from_id

    class PreflightLoggingAlgorithm(LoggingAlgorithmModule):
        def scheduleOffloading(self, env: Any) -> None:
            decisions = select_ready_offload_decisions(env, self.taskScheduler, self.entityScheduler)
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
                        "time": round(float(env.simulation_time), 6),
                        "task_id": decision["task_id"],
                        "task_node_id": decision["task_node_id"],
                        "source_node_id": decision["source_node_id"],
                        "target_node_id": decision["target_node_id"],
                        "target_node_type": node_type_from_id(decision["target_node_id"]),
                        "candidate_count": 1,
                        "nearest_distance": as_float(decision["distance"]),
                        "route_nodes": decision["route_nodes"],
                        "evidence": "direct",
                    }
                )

    config_path = example_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = build_preflight_config(config, seed=int(seed), max_time=float(max_time))
    trajectory_id = config["pi_jwm_preflight"]["trajectory_id"]
    np.random.seed(int(seed))
    random.seed(int(seed))

    physical_node_records: dict[str, dict[str, Any]] = {}
    physical_edge_records: dict[str, dict[str, Any]] = {}
    task_records: dict[str, dict[str, Any]] = {}
    dag_edge_records: dict[str, dict[str, Any]] = {}
    step = 0
    old_cwd = Path.cwd()
    env = None
    try:
        os.chdir(example_dir)
        env = AirFogSimEnv(config, interactive_mode=None)
        algorithm = PreflightLoggingAlgorithm(int(seed))
        algorithm.initialize(env)
        RewardScheduler.setModel(env, "REWARD", "1/task_delay")
        while not env.isDone():
            algorithm.scheduleStep(env)
            active = active_link_counts(env)
            env.step()
            step += 1
            current_time = float(env.simulation_time)
            for row in node_rows(env):
                normalized = _node_record(row, trajectory_id)
                physical_node_records[normalized["id"]] = normalized
            for row in link_rows(env, active):
                normalized = _physical_edge_record(row, trajectory_id)
                physical_edge_records[normalized["id"]] = normalized
            for task in iter_airfogsim_tasks(env.task_manager):
                normalized = _task_record(task, trajectory_id, current_time)
                task_records[normalized["id"]] = normalized
            dag_rows = normalize_airfogsim_dags(
                TaskScheduler.getAllTaskDAGs(env),
                trajectory_id=trajectory_id,
                step=step,
                time_value=current_time,
            )
            for row in dag_rows:
                dag_edge_records.setdefault(row["id"], row)
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)

    offload_actions = [dict(row, evidence="direct") for row in algorithm.offload_rows]
    return_actions = [dict(row, evidence="direct") for row in algorithm.return_rows]
    _add_route_edges(
        physical_edge_records,
        physical_node_records,
        trajectory_id,
        offload_actions,
        "source_node_id",
        "target_node_id",
    )
    _add_route_edges(
        physical_edge_records,
        physical_node_records,
        trajectory_id,
        return_actions,
        "current_node_id",
        "return_target_id",
    )
    bundle = assemble_airfogsim_bundle(
        trajectory_id=trajectory_id,
        physical_nodes=list(physical_node_records.values()),
        physical_edges=list(physical_edge_records.values()),
        task_records=list(task_records.values()),
        dag_edges=list(dag_edge_records.values()),
        offload_actions=offload_actions,
        return_actions=return_actions,
    )
    runtime_summary = {
        "seed": int(seed),
        "steps": int(step),
        "max_time": float(max_time),
        "physical_nodes": len(bundle["physical_nodes"]),
        "physical_edges": len(bundle["physical_edges"]),
        "information_nodes": len(bundle["information_nodes"]),
        "information_edges": len(bundle["information_edges"]),
        "offload_actions": len(offload_actions),
        "return_actions": len(return_actions),
        "mn_relations": len(bundle["mn_relations"]),
        "me_relations": len(bundle["me_relations"]),
        "ep_relations": len(bundle["ep_relations"]),
        "ep_direct_events": sum(row.get("evidence") == "direct" for row in bundle["ep_relations"]),
        "ep_not_modeled": sum(row.get("evidence") == "not_modeled" for row in bundle["ep_relations"]),
        "scheduler_only_physical_edges": sum(
            row.get("source_interface") == "AirFogSim scheduler route"
            for row in bundle["physical_edges"]
        ),
    }
    return {"config": config, "bundle": bundle, "runtime_summary": runtime_summary}


def _default_output_dir() -> Path:
    code_root = Path(__file__).resolve().parents[2]
    return (
        code_root
        / "artifacts"
        / "small_experiments"
        / "exp02_airfogsim_strict_dual_graph"
        / "preflight_v1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an evidence-first AirFogSim strict dual-graph preflight."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    result = run_preflight_experiment(
        args.output_dir,
        seed=args.seed,
        max_time=args.max_time,
        runtime_runner=run_airfogsim_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["experiment_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
