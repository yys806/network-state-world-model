"""Build deterministic STEP 4.3A typed dual-graph evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import (
    REQUIRED_ACCEPTANCE_CHECKS,
    PhysicalTopologyConfig,
    build_typed_dual_graph_batch,
    graph_semantic_digest,
    load_typed_dual_graph_batch,
    save_typed_dual_graph_batch,
    validate_acceptance_receipt,
    validate_typed_dual_graph_checks,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920"
SOURCE_TENSOR = SOURCE_DIR / "flow_tensor.npz"
SOURCE_ACCEPTANCE = SOURCE_DIR / "acceptance.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _arrays_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return graph_semantic_digest(left) == graph_semantic_digest(right)


def _tamper(graph: dict[str, Any], block: str, field: str, delta: float = 1) -> dict[str, Any]:
    value = copy.deepcopy(graph)
    value["blocks"][block][field].flat[0] += delta
    return value


def fixture_checks(tensor: dict[str, Any], graph: dict[str, Any], config: PhysicalTopologyConfig) -> dict[str, bool]:
    def rejected(value: dict[str, Any]) -> bool:
        return not validate_typed_dual_graph_checks(value, tensor)["passed"]

    physical_comm = copy.deepcopy(graph)
    physical_comm["blocks"]["physical_relations"]["csi"] = np.zeros_like(physical_comm["blocks"]["physical_relations"]["features"])
    physical_rb = copy.deepcopy(graph)
    physical_rb["blocks"]["physical_relations"]["rb_allocation"] = np.zeros(physical_rb["blocks"]["physical_relations"]["presence"].shape, dtype=np.int64)

    no_information = copy.deepcopy(tensor)
    no_information["comm_relation_presence"][:] = False
    no_information["task_agent_validity_mask"][:] = False
    no_information["logical_flow_known_mask"][:] = False
    no_information["dag_mask"][:] = False
    no_information_graph = build_typed_dual_graph_batch(no_information, config)

    current_hop = copy.deepcopy(graph)
    active = np.argwhere(current_hop["blocks"]["flow_carrying_state"]["active"])
    hop_rejected = True
    if len(active):
        bi, fi = map(int, active[0])
        # Force the relation endpoint to a carrying-hop endpoint.  In the real
        # multi-hop current frame the hop destination can coincide with the
        # logical destination, while the hop source remains stage-local.
        hop = int(current_hop["blocks"]["flow_carrying_state"]["hop_source_index"][bi, fi])
        current_hop["blocks"]["flow_relations"]["destination_index"][bi, fi] = hop
        hop_rejected = rejected(current_hop)

    split_hops = copy.deepcopy(graph)
    split_hops["blocks"]["flow_relations"]["hop_split_relation"] = np.zeros_like(split_hops["blocks"]["flow_relations"]["known"])
    deduplicated = copy.deepcopy(graph)
    deduplicated["blocks"]["flow_relations"]["known"][:, 1:] = False
    reverse_task = _tamper(graph, "task_agent_relations", "task_index")
    reverse_dag = copy.deepcopy(graph)
    reverse_dag["blocks"]["dag_relations"]["source_task_index"], reverse_dag["blocks"]["dag_relations"]["target_task_index"] = (
        reverse_dag["blocks"]["dag_relations"]["target_task_index"].copy(), reverse_dag["blocks"]["dag_relations"]["source_task_index"].copy()
    )
    fake_dep = copy.deepcopy(graph)
    fake_dep["blocks"]["flow_relations"]["dag_generated_depdata"] = np.ones_like(fake_dep["blocks"]["flow_relations"]["known"])

    no_comm = copy.deepcopy(tensor)
    no_comm["comm_relation_presence"][:] = False
    no_comm["comm_relation_validity"][:] = False
    flow_before = graph["blocks"]["flow_relations"]
    flow_after = build_typed_dual_graph_batch(no_comm, config)["blocks"]["flow_relations"]
    flow_independent = all(np.array_equal(flow_before[key], flow_after[key]) for key in flow_before)

    no_csi = copy.deepcopy(tensor)
    no_csi["comm_csi_mask"][:] = False
    no_csi["comm_csi_raw"][:] = 0.0
    no_csi_graph = build_typed_dual_graph_batch(no_csi, config)
    comm_retained = np.array_equal(graph["blocks"]["comm_relations"]["presence"], no_csi_graph["blocks"]["comm_relations"]["presence"])
    validity_retained = np.array_equal(graph["blocks"]["comm_relations"]["validity"], no_csi_graph["blocks"]["comm_relations"]["validity"])

    task_shortcut = copy.deepcopy(graph); task_shortcut["blocks"]["task_physical_relations"] = {}
    flow_shortcut = copy.deepcopy(graph); flow_shortcut["blocks"]["flow_physical_relations"] = {}
    geo_match = copy.deepcopy(graph); geo_match["contract"]["geo_comm_semantics"] = "requires_matching_physical_edge"

    future = copy.deepcopy(tensor)
    for key in list(future):
        if key.startswith("target_") and isinstance(future[key], np.ndarray):
            if future[key].dtype == bool:
                future[key] = ~future[key]
            elif np.issubdtype(future[key].dtype, np.number):
                future[key] = future[key] + 123
    future_unchanged = _arrays_equal(graph, build_typed_dual_graph_batch(future, config))

    presence_zero = copy.deepcopy(graph)
    candidate = np.argwhere(~presence_zero["blocks"]["task_nodes"]["presence"])
    if len(candidate):
        bi, ti = map(int, candidate[0]); presence_zero["blocks"]["task_nodes"]["feature_mask"][bi, ti, 0] = True
        presence_zero["blocks"]["task_nodes"]["features"][bi, ti, 0] = 0.0

    overflow = False
    try:
        build_typed_dual_graph_batch(tensor, PhysicalTopologyConfig(mode=config.mode, radius_m=config.radius_m, k=config.k, max_physical_relations=0))
    except ValueError as exc:
        overflow = "capacity overflow" in str(exc)
    relation_truncated = copy.deepcopy(graph)
    for field in relation_truncated["blocks"]["comm_relations"]:
        relation_truncated["blocks"]["comm_relations"][field] = relation_truncated["blocks"]["comm_relations"][field][:, :-1]

    receipt = {"required_checks": {name: True for name in REQUIRED_ACCEPTANCE_CHECKS}, "passed": True,
               "scope": {name: False for name in ("encoder", "gnn", "message_passing", "world_model", "loss", "planner", "training", "gpu", "locked_test", "formal_dataset")}}
    receipt["required_checks"]["current_graph_causal"] = False

    return {
        "01_csi_in_physical_edge_rejected": rejected(physical_comm),
        "02_rb_in_physical_edge_rejected": rejected(physical_rb),
        "03_comm_task_flow_cannot_control_physical_topology": all(np.array_equal(graph["blocks"]["physical_relations"][key], no_information_graph["blocks"]["physical_relations"][key]) for key in graph["blocks"]["physical_relations"]),
        "04_current_hop_cannot_replace_logical_endpoint": hop_rejected,
        "05_multihop_cannot_create_extra_flow_relations": rejected(split_hops),
        "06_parallel_flows_cannot_be_deduplicated": rejected(deduplicated),
        "07_task_agent_direction_tamper_rejected": rejected(reverse_task),
        "08_dag_direction_tamper_rejected": rejected(reverse_dag),
        "09_dag_cannot_create_fake_depdata": rejected(fake_dep),
        "10_flow_does_not_require_comm": flow_independent,
        "11_wired_or_no_csi_comm_retained": comm_retained,
        "12_missing_csi_does_not_invalidate_comm": validity_retained,
        "13_task_physical_shortcut_rejected": rejected(task_shortcut),
        "14_flow_physical_shortcut_rejected": rejected(flow_shortcut),
        "15_geocomm_matching_physical_edge_requirement_rejected": rejected(geo_match),
        "16_future_target_counterfactual_unchanged": future_unchanged,
        "17_presence_false_not_padding_or_real_zero": rejected(presence_zero) if len(candidate) else True,
        "18_physical_topology_capacity_overflow_rejected": overflow,
        "19_relation_capacity_truncation_rejected": rejected(relation_truncated),
        "20_acceptance_receipt_tamper_rejected": not validate_acceptance_receipt(receipt)["passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.refresh_existing)

    tensor = load_flow_tensor_batch(SOURCE_TENSOR)
    config = PhysicalTopologyConfig(mode="radius_knn", radius_m=1000.0, k=2, self_loops=False)
    graph = build_typed_dual_graph_batch(tensor, config)
    graph_2 = build_typed_dual_graph_batch(tensor, config)
    deterministic = graph_semantic_digest(graph) == graph_semantic_digest(graph_2)
    graph_checks = validate_typed_dual_graph_checks(graph, tensor)
    fixtures = fixture_checks(tensor, graph, config)
    graph_path = args.output_dir / "typed_dual_graph_batch.npz"
    save_typed_dual_graph_batch(graph, graph_path)
    loaded = load_typed_dual_graph_batch(graph_path)
    round_trip = _arrays_equal(graph, loaded) and validate_typed_dual_graph_checks(loaded, tensor)["passed"]

    required = {name: bool(graph_checks[name]) for name in REQUIRED_ACCEPTANCE_CHECKS}
    required["physical_topology_independent_of_comm_task_flow"] &= fixtures["03_comm_task_flow_cannot_control_physical_topology"]
    required["flow_comm_independence"] &= fixtures["10_flow_does_not_require_comm"]
    required["future_target_isolation"] &= fixtures["16_future_target_counterfactual_unchanged"]
    required["deterministic_rebuild"] &= deterministic
    required["no_silent_truncation"] &= fixtures["18_physical_topology_capacity_overflow_rejected"] and fixtures["19_relation_capacity_truncation_rejected"]
    required["serialize_load"] &= round_trip
    scope = {name: False for name in ("encoder", "gnn", "message_passing", "world_model", "loss", "planner", "training", "gpu", "locked_test", "formal_dataset")}
    required["scope"] &= all(value is False for value in scope.values())
    passed = bool(all(required.values()) and all(fixtures.values()))
    acceptance = {
        "schema_version": "PI-JWM-Step-4.3A-Acceptance-v1", "required_checks": required,
        "passed": passed, "scope": scope, "graph_checks": graph_checks,
        "negative_fixture_checks": fixtures, "deterministic_semantic_digest": graph_semantic_digest(graph),
        "evidence_classes": {
            "REAL_TRACE": ["STEP 4.2A Comm/Task-Agent Tensor", "STEP 4.2C-C stateful Flow Tensor", "real two-hop Input", "real cross-slot Flow", "real direct Input/Return"],
            "DERIVED_GRAPH_STRUCTURE": ["typed blocks", "Physical topology", "Align", "GeoComm"],
            "DEVELOPMENT_TOPOLOGY_CONFIG": config.__dict__,
            "CONTRACT_FIXTURE": list(fixtures),
        },
    }
    write_json(args.output_dir / "graph_schema_config.json", graph["contract"])
    write_json(args.output_dir / "acceptance_report.json", acceptance)
    manifest = {
        "schema_version": "PI-JWM-Step-4.3A-Artifact-Manifest-v1",
        "source": {"path": str(SOURCE_TENSOR.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SOURCE_TENSOR), "acceptance_path": str(SOURCE_ACCEPTANCE.relative_to(ROOT)).replace("\\", "/"), "acceptance_sha256": sha256(SOURCE_ACCEPTANCE)},
        "files": {}, "development_only": True, "research_frozen_topology": False,
        "training": False, "gpu": False, "locked_test": False, "formal_dataset": False,
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"passed": passed, "required_checks": required, "negative_fixture_checks": fixtures, "artifact": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
