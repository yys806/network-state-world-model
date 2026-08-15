from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def build_controlled_fixture() -> dict[str, Any]:
    """Return a minimal non-empty strict dual-graph scenario.

    The fixture is intentionally synthetic.  It exercises all relation
    families without claiming that the current AirFogSim export contains the
    same fields.
    """

    physical_nodes = [
        {"id": "veh0", "kind": "vehicle", "position": [0.0, 0.0]},
        {"id": "rsu0", "kind": "rsu", "position": [50.0, 0.0]},
        {"id": "edge0", "kind": "edge_server", "position": [50.0, 0.0]},
        {"id": "uav0", "kind": "uav", "position": [80.0, 40.0]},
    ]
    physical_edges = [
        {"id": "e_veh_rsu", "src": "veh0", "dst": "rsu0", "kind": "wireless"},
        {"id": "e_rsu_veh", "src": "rsu0", "dst": "veh0", "kind": "wireless"},
        {"id": "e_rsu_edge", "src": "rsu0", "dst": "edge0", "kind": "wired"},
        {"id": "e_edge_rsu", "src": "edge0", "dst": "rsu0", "kind": "wired"},
        {"id": "e_rsu_uav", "src": "rsu0", "dst": "uav0", "kind": "wireless"},
        {"id": "e_uav_rsu", "src": "uav0", "dst": "rsu0", "kind": "wireless"},
    ]
    information_nodes = [
        {"id": "m0", "stage": "input", "remaining_input_mb": 10.0},
        {"id": "m1", "stage": "waiting", "remaining_input_mb": 4.0},
        {"id": "m2", "stage": "waiting", "remaining_input_mb": 3.0},
        {"id": "m3", "stage": "waiting", "remaining_input_mb": 2.0},
    ]
    information_edges = [
        {"id": "ie_m0_m1", "src": "m0", "dst": "m1", "data_mb": 4.0},
        {"id": "ie_m0_m2", "src": "m0", "dst": "m2", "data_mb": 3.0},
        {"id": "ie_m1_m3", "src": "m1", "dst": "m3", "data_mb": 2.0},
        {"id": "ie_m2_m3", "src": "m2", "dst": "m3", "data_mb": 2.0},
    ]

    assignments = {
        "m0": {"source": "veh0", "host": "veh0", "exec": "edge0", "ret": "veh0"},
        "m1": {"source": "veh0", "host": "edge0", "exec": "uav0", "ret": "veh0"},
        "m2": {"source": "veh0", "host": "edge0", "exec": "edge0", "ret": "veh0"},
        "m3": {"source": "veh0", "host": "uav0", "exec": "uav0", "ret": "veh0"},
    }
    mn_relations = [
        {"task": task, "relation": relation, "physical_node": node}
        for task, relations in assignments.items()
        for relation, node in relations.items()
    ]
    me_relations = [
        {
            "task": "m0",
            "relation": "in",
            "source": "veh0",
            "target": "edge0",
            "path": ["e_veh_rsu", "e_rsu_edge"],
        },
        {
            "task": "m3",
            "relation": "out",
            "source": "uav0",
            "target": "veh0",
            "path": ["e_uav_rsu", "e_rsu_veh"],
        },
    ]
    ep_relations = [
        {
            "info_edge": "ie_m0_m1",
            "source": "edge0",
            "target": "uav0",
            "path": ["e_edge_rsu", "e_rsu_uav"],
        },
        {
            "info_edge": "ie_m0_m2",
            "source": "edge0",
            "target": "edge0",
            "path": [],
        },
        {
            "info_edge": "ie_m1_m3",
            "source": "uav0",
            "target": "uav0",
            "path": [],
        },
        {
            "info_edge": "ie_m2_m3",
            "source": "edge0",
            "target": "uav0",
            "path": ["e_edge_rsu", "e_rsu_uav"],
        },
    ]
    return {
        "scenario_id": "controlled_strict_dual_graph_v0",
        "evidence_scope": "controlled_fixture_only",
        "physical_nodes": physical_nodes,
        "physical_edges": physical_edges,
        "information_nodes": information_nodes,
        "information_edges": information_edges,
        "mn_relations": mn_relations,
        "me_relations": me_relations,
        "ep_relations": ep_relations,
    }


def validate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    physical_nodes = scenario.get("physical_nodes", [])
    physical_edges = scenario.get("physical_edges", [])
    information_nodes = scenario.get("information_nodes", [])
    information_edges = scenario.get("information_edges", [])
    mn_relations = scenario.get("mn_relations", [])
    me_relations = scenario.get("me_relations", [])
    ep_relations = scenario.get("ep_relations", [])

    physical_node_ids = {node["id"] for node in physical_nodes}
    information_node_ids = {node["id"] for node in information_nodes}
    physical_edge_by_id = {edge["id"]: edge for edge in physical_edges}
    information_edge_by_id = {edge["id"]: edge for edge in information_edges}

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    disjoint = physical_node_ids.isdisjoint(information_node_ids)
    add_check("node_sets_disjoint", disjoint, "Physical and information node IDs must be disjoint.")

    edge_ids_unique = len(physical_edge_by_id) == len(physical_edges)
    physical_endpoints_valid = all(
        edge.get("src") in physical_node_ids and edge.get("dst") in physical_node_ids
        for edge in physical_edges
    )
    add_check(
        "physical_edges_valid",
        edge_ids_unique and physical_endpoints_valid,
        "Physical edge IDs must be unique and all endpoints must exist.",
    )

    information_nonempty = bool(information_nodes) and bool(information_edges)
    information_endpoints_valid = all(
        edge.get("src") in information_node_ids and edge.get("dst") in information_node_ids
        for edge in information_edges
    )
    add_check(
        "information_graph_nonempty",
        information_nonempty and information_endpoints_valid,
        "The information graph must contain at least one valid dependency edge.",
    )
    dag = information_endpoints_valid and _is_dag(information_node_ids, information_edges)
    add_check("information_graph_is_dag", dag, "A topological ordering must exist.")

    mn_valid, mn_details, exec_mapping = _validate_mn_relations(
        information_node_ids, physical_node_ids, mn_relations
    )
    add_check("mn_relation_cardinality", mn_valid, mn_details)

    families_nonempty = bool(mn_relations) and bool(me_relations) and bool(ep_relations)
    add_check(
        "relation_families_nonempty",
        families_nonempty,
        "MN, ME and EP relation families must all contain records.",
    )

    me_valid, me_details = _validate_me_paths(
        me_relations, information_node_ids, physical_edge_by_id
    )
    add_check("me_paths_valid", me_valid, me_details)

    ep_valid, ep_details = _validate_ep_paths(
        ep_relations,
        information_edge_by_id,
        exec_mapping,
        physical_edge_by_id,
    )
    add_check("ep_paths_valid", ep_valid, ep_details)

    counts = {
        "physical_nodes": len(physical_nodes),
        "physical_edges": len(physical_edges),
        "information_nodes": len(information_nodes),
        "information_edges": len(information_edges),
        "mn_relations": len(mn_relations),
        "me_relations": len(me_relations),
        "me_edge_incidents": sum(len(relation.get("path", [])) for relation in me_relations),
        "ep_relations": len(ep_relations),
        "ep_edge_incidents": sum(len(relation.get("path", [])) for relation in ep_relations),
    }
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "counts": counts,
        "evidence_scope": scenario.get("evidence_scope", "unspecified"),
    }


def compare_candidate_actions(scenario: dict[str, Any]) -> dict[str, Any]:
    """Apply two legal input-stage actions with one shared deterministic rule."""

    physical_edge_by_id = {edge["id"]: edge for edge in scenario["physical_edges"]}
    task = next(node for node in scenario["information_nodes"] if node["id"] == "m0")
    candidates = [
        {
            "action": "offload_edge_rb3",
            "task": "m0",
            "execution_target": "edge0",
            "rb_count": 3,
            "input_path": ["e_veh_rsu", "e_rsu_edge"],
        },
        {
            "action": "offload_uav_rb1",
            "task": "m0",
            "execution_target": "uav0",
            "rb_count": 1,
            "input_path": ["e_veh_rsu", "e_rsu_uav"],
        },
    ]
    rows = []
    for candidate in candidates:
        legal = (
            task["stage"] == "input"
            and candidate["rb_count"] > 0
            and _validate_path(
                candidate["input_path"],
                "veh0",
                candidate["execution_target"],
                physical_edge_by_id,
            )
        )
        service_mb = min(task["remaining_input_mb"], candidate["rb_count"] * 2.0) if legal else 0.0
        rows.append(
            {
                **candidate,
                "legal": legal,
                "service_mb": service_mb,
                "remaining_input_mb_next": task["remaining_input_mb"] - service_mb,
            }
        )

    signatures = {
        (
            row["execution_target"],
            row["rb_count"],
            tuple(row["input_path"]),
            row["remaining_input_mb_next"],
        )
        for row in rows
    }
    return {
        "passed": all(row["legal"] for row in rows) and len(signatures) == len(rows),
        "rows": rows,
        "evidence_scope": "deterministic_fixture_rule_only",
    }


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_controlled_fixture()
    validation = validate_scenario(scenario)
    action_sensitivity = compare_candidate_actions(scenario)
    failed_checks = [check["name"] for check in validation["checks"] if not check["passed"]]
    passed = validation["passed"] and action_sensitivity["passed"]

    _write_json(output_dir / "scenario.json", scenario)
    _write_json(
        output_dir / "validation_report.json",
        {
            **validation,
            "passed": passed,
            "failed_checks": failed_checks,
            "action_sensitivity_passed": action_sensitivity["passed"],
            "airfogsim_real_nonempty_dag_validated": False,
        },
    )
    _write_cross_relations_csv(output_dir / "cross_relations.csv", scenario)
    _write_action_sensitivity_csv(output_dir / "action_sensitivity.csv", action_sensitivity["rows"])
    _draw_physical_graph(output_dir / "physical_graph.png", scenario)
    _draw_information_graph(output_dir / "information_graph.png", scenario)
    _draw_joint_graph(output_dir / "joint_graph.png", scenario)
    (output_dir / "REPORT.md").write_text(
        _build_markdown_report(validation, action_sensitivity, failed_checks),
        encoding="utf-8",
    )
    return {
        "passed": passed,
        "output_dir": str(output_dir),
        "failed_checks": failed_checks,
        "counts": validation["counts"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_cross_relations_csv(path: Path, scenario: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for relation in scenario["mn_relations"]:
        rows.append(
            {
                "family": "MN",
                "relation": relation["relation"],
                "information_object": relation["task"],
                "physical_source": relation["physical_node"],
                "physical_target": relation["physical_node"],
                "physical_path": "",
                "edge_incident_count": 0,
                "zero_hop": True,
            }
        )
    for family, key in (("ME", "task"), ("EP", "info_edge")):
        for relation in scenario[f"{family.lower()}_relations"]:
            rows.append(
                {
                    "family": family,
                    "relation": relation.get("relation", "dependency"),
                    "information_object": relation[key],
                    "physical_source": relation["source"],
                    "physical_target": relation["target"],
                    "physical_path": ";".join(relation["path"]),
                    "edge_incident_count": len(relation["path"]),
                    "zero_hop": len(relation["path"]) == 0,
                }
            )
    fieldnames = [
        "family",
        "relation",
        "information_object",
        "physical_source",
        "physical_target",
        "physical_path",
        "edge_incident_count",
        "zero_hop",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_action_sensitivity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    output_rows = [{**row, "input_path": ";".join(row["input_path"])} for row in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def _physical_graph(scenario: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in scenario["physical_nodes"]:
        graph.add_node(node["id"], **node)
    for edge in scenario["physical_edges"]:
        graph.add_edge(edge["src"], edge["dst"], edge_id=edge["id"], kind=edge["kind"])
    return graph


def _information_graph(scenario: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in scenario["information_nodes"]:
        graph.add_node(node["id"], **node)
    for edge in scenario["information_edges"]:
        graph.add_edge(edge["src"], edge["dst"], edge_id=edge["id"], data_mb=edge["data_mb"])
    return graph


def _physical_plot_positions(scenario: dict[str, Any]) -> dict[str, tuple[float, float]]:
    grouped: dict[tuple[float, float], list[str]] = defaultdict(list)
    for node in scenario["physical_nodes"]:
        grouped[tuple(node["position"])].append(node["id"])
    positions: dict[str, tuple[float, float]] = {}
    for (x, y), node_ids in grouped.items():
        ordered = sorted(node_ids)
        for index, node_id in enumerate(ordered):
            vertical_offset = (index - (len(ordered) - 1) / 2.0) * 12.0
            positions[node_id] = (x, y + vertical_offset)
    return positions


def _draw_physical_graph(path: Path, scenario: dict[str, Any]) -> None:
    graph = _physical_graph(scenario)
    positions = _physical_plot_positions(scenario)
    colors = {"vehicle": "#4C78A8", "rsu": "#F58518", "edge_server": "#54A24B", "uav": "#E45756"}
    node_colors = [colors[graph.nodes[node]["kind"]] for node in graph.nodes]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=axis,
        node_color=node_colors,
        node_size=1800,
        font_color="white",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
    )
    edge_legend = "\n".join(
        f"{edge['id']}: {edge['src']} -> {edge['dst']}" for edge in scenario["physical_edges"]
    )
    axis.text(1.02, 0.5, edge_legend, transform=axis.transAxes, va="center", fontsize=8)
    axis.set_title("Physical graph: directed feasible links")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_information_graph(path: Path, scenario: dict[str, Any]) -> None:
    graph = _information_graph(scenario)
    positions = {"m0": (0.0, 0.5), "m1": (1.5, 1.0), "m2": (1.5, 0.0), "m3": (3.0, 0.5)}
    fig, axis = plt.subplots(figsize=(8, 4.5))
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=axis,
        node_color="#72B7B2",
        node_size=1800,
        arrows=True,
        arrowsize=20,
    )
    edge_labels = {
        (edge["src"], edge["dst"]): f"{edge['data_mb']} MB"
        for edge in scenario["information_edges"]
    }
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, ax=axis)
    axis.set_title("Information graph: non-empty diamond DAG")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_joint_graph(path: Path, scenario: dict[str, Any]) -> None:
    graph = nx.DiGraph()
    physical_positions = {"veh0": (0.0, 2.2), "rsu0": (1.5, 2.2), "edge0": (3.0, 2.8), "uav0": (3.0, 1.6)}
    information_positions = {"m0": (0.0, 0.0), "m1": (1.4, 0.55), "m2": (1.4, -0.55), "m3": (2.8, 0.0)}
    graph.add_nodes_from(physical_positions)
    graph.add_nodes_from(information_positions)
    physical_edges = [(edge["src"], edge["dst"]) for edge in scenario["physical_edges"]]
    information_edges = [(edge["src"], edge["dst"]) for edge in scenario["information_edges"]]
    exec_relations = [
        (relation["task"], relation["physical_node"])
        for relation in scenario["mn_relations"]
        if relation["relation"] == "exec"
    ]
    graph.add_edges_from(physical_edges + information_edges + exec_relations)
    positions = {**physical_positions, **information_positions}
    fig, axis = plt.subplots(figsize=(10, 6.5))
    nx.draw_networkx_nodes(graph, positions, nodelist=list(physical_positions), node_shape="s", node_color="#4C78A8", node_size=1500, ax=axis)
    nx.draw_networkx_nodes(graph, positions, nodelist=list(information_positions), node_shape="o", node_color="#72B7B2", node_size=1400, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_color="white", ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=physical_edges, edge_color="#888888", arrows=True, alpha=0.55, connectionstyle="arc3,rad=0.07", ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=information_edges, edge_color="#2E86AB", arrows=True, width=2.0, ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=exec_relations, edge_color="#D62728", style="dashed", arrows=True, width=1.8, ax=axis)
    axis.text(0.0, 3.35, "Physical layer", fontsize=12, weight="bold")
    axis.text(0.0, 0.9, "Information layer", fontsize=12, weight="bold")
    axis.text(3.65, 2.75, "ME paths:\ninput veh0-rsu0-edge0\noutput uav0-rsu0-veh0", fontsize=9, color="#F58518", va="top")
    axis.text(3.65, 1.15, "EP paths:\nm0->m1: edge0-rsu0-uav0\nm2->m3: edge0-rsu0-uav0\nother dependencies: zero-hop", fontsize=9, color="#7A5195", va="top")
    axis.text(3.65, -0.45, "Red dashed: MN(exec)", fontsize=9, color="#D62728", va="top")
    axis.set_title("Strict dual graph and explicit cross-graph relations")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_markdown_report(
    validation: dict[str, Any], action_sensitivity: dict[str, Any], failed_checks: list[str]
) -> str:
    lines = [
        "# 小实验00：严格双图数据与关系合法性验证",
        "",
        "## 结论状态",
        "",
        f"- 受控夹具验证：{'已完成' if validation['passed'] and action_sensitivity['passed'] else '未通过'}",
        "- AirFogSim真实非空DAG验证：待完成",
        "- 神经网络双图耦合有效性：待完成",
        "- 证据边界：本实验仅证明受控样本和验证程序能够表达并检查严格双图，不证明当前AirFogSim数据已经包含真实DAG，也不证明双图优于单图。",
        "",
        "## 合法性检查",
        "",
        "| 检查 | 状态 | 说明 |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {check['name']} | {'通过' if check['passed'] else '失败'} | {check['details']} |"
        for check in validation["checks"]
    )
    lines.extend(
        [
            "",
            "## 结构计数",
            "",
            "| 对象 | 数量 |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {name} | {value} |" for name, value in validation["counts"].items())
    lines.extend(
        [
            "",
            "## 一步动作敏感性",
            "",
            "两个动作从完全相同的初始状态出发，并使用同一条确定性服务规则：每个RB在该受控样本中服务2 MB。该规则只用于检查动作是否被写入转移，不是AirFogSim链路模型。",
            "",
            "| 动作 | 执行节点 | RB | 服务量/MB | 下一步剩余输入/MB | 输入路径 |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {row['action']} | {row['execution_target']} | {row['rb_count']} | {row['service_mb']:.1f} | {row['remaining_input_mb_next']:.1f} | {' -> '.join(row['input_path'])} |"
        for row in action_sensitivity["rows"]
    )
    lines.extend(
        [
            "",
            f"动作敏感性检查：{'通过' if action_sensitivity['passed'] else '失败'}。",
            f"失败检查：{', '.join(failed_checks) if failed_checks else '无'}。",
            "",
            "## 下一步",
            "",
            "将同一验证器接到真实AirFogSim导出或新的主仿真数据上；只有真实样本中的信息边、MN/ME/EP关系和动作后继全部通过，才能开始严格双图与单图的训练消融。",
            "",
        ]
    )
    return "\n".join(lines)


def _default_output_dir() -> Path:
    code_root = Path(__file__).resolve().parents[2]
    return code_root / "artifacts" / "small_experiments" / "exp00_strict_dual_graph_validity"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a controlled strict PI-JWM dual graph.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    result = run_experiment(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def _is_dag(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    indegree = {node_id: 0 for node_id in node_ids}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        successors[src].append(dst)
        indegree[dst] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    return visited == len(node_ids)


def _validate_mn_relations(
    task_ids: set[str], physical_node_ids: set[str], relations: list[dict[str, Any]]
) -> tuple[bool, str, dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    exec_mapping: dict[str, str] = {}
    references_valid = True
    allowed = {"source", "host", "exec", "ret"}
    for relation in relations:
        task = relation.get("task")
        relation_type = relation.get("relation")
        physical_node = relation.get("physical_node")
        references_valid &= (
            task in task_ids and relation_type in allowed and physical_node in physical_node_ids
        )
        counts[(task, relation_type)] += 1
        if relation_type == "exec" and counts[(task, relation_type)] == 1:
            exec_mapping[task] = physical_node

    cardinality_valid = all(
        counts[(task, "source")] == 1
        and counts[(task, "host")] == 1
        and counts[(task, "exec")] <= 1
        and counts[(task, "ret")] <= 1
        for task in task_ids
    )
    passed = references_valid and cardinality_valid
    return passed, "source/host require exactly one mapping; exec/ret allow at most one.", exec_mapping


def _validate_me_paths(
    relations: list[dict[str, Any]],
    task_ids: set[str],
    physical_edge_by_id: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    allowed = {"in", "out"}
    failures = []
    for relation in relations:
        valid = (
            relation.get("task") in task_ids
            and relation.get("relation") in allowed
            and _validate_path(
                relation.get("path", []),
                relation.get("source"),
                relation.get("target"),
                physical_edge_by_id,
            )
        )
        if not valid:
            failures.append(f"{relation.get('task')}:{relation.get('relation')}")
    return not failures, "Invalid lifecycle paths: " + ", ".join(failures) if failures else "All lifecycle paths are valid."


def _validate_ep_paths(
    relations: list[dict[str, Any]],
    information_edge_by_id: dict[str, dict[str, Any]],
    exec_mapping: dict[str, str],
    physical_edge_by_id: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    relation_by_edge = {relation.get("info_edge"): relation for relation in relations}
    failures = []
    for edge_id, info_edge in information_edge_by_id.items():
        relation = relation_by_edge.get(edge_id)
        if relation is None:
            failures.append(f"{edge_id}:missing")
            continue
        expected_source = exec_mapping.get(info_edge["src"])
        expected_target = exec_mapping.get(info_edge["dst"])
        valid = (
            relation.get("source") == expected_source
            and relation.get("target") == expected_target
            and _validate_path(
                relation.get("path", []),
                expected_source,
                expected_target,
                physical_edge_by_id,
            )
        )
        if not valid:
            failures.append(edge_id)
    extra = set(relation_by_edge) - set(information_edge_by_id)
    failures.extend(f"{edge_id}:extra" for edge_id in sorted(extra))
    return not failures, "Invalid dependency paths: " + ", ".join(failures) if failures else "All dependency paths are valid."


def _validate_path(
    path: list[str],
    source: str | None,
    target: str | None,
    physical_edge_by_id: dict[str, dict[str, Any]],
) -> bool:
    if source is None or target is None:
        return False
    if source == target:
        return len(path) == 0
    if not path or len(path) != len(set(path)):
        return False
    try:
        edges = [physical_edge_by_id[edge_id] for edge_id in path]
    except KeyError:
        return False
    if edges[0]["src"] != source or edges[-1]["dst"] != target:
        return False
    if any(left["dst"] != right["src"] for left, right in zip(edges, edges[1:])):
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


if __name__ == "__main__":
    raise SystemExit(main())
