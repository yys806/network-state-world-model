"""Build STEP 4.3B deterministic untrained encoder evidence."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import PhysicalTopologyConfig, build_typed_dual_graph_batch, graph_semantic_digest, load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import (
    REQUIRED_ACCEPTANCE_CHECKS, DualGraphEncoderConfig, PIJointGraphEncoder,
    build_encoder_input_audit, fit_encoder_normalization_stats, load_encoder_package,
    output_semantic_digest, save_encoder_package, validate_encoder_acceptance,
    validate_encoder_architecture_checks, validate_encoder_contract_checks,
)


ROOT = Path(__file__).resolve().parents[2]
TENSOR_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz"
GRAPH_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/typed_dual_graph_batch.npz"
GRAPH_MANIFEST = GRAPH_PATH.parent / "manifest.json"
UPSTREAM_STATS = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/train_normalization_stats.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _array_mapping_digest(value: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(value):
        item = value[key]
        if isinstance(item, np.ndarray):
            digest.update(key.encode("utf-8")); digest.update(str(item.dtype).encode("ascii")); digest.update(str(item.shape).encode("ascii")); digest.update(item.tobytes())
    return digest.hexdigest()


def _consistent_slot_permutation(tensor: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    changed = copy.deepcopy(tensor)
    n_entity = changed["entity_presence"].shape[2]
    permutation = np.arange(n_entity - 1, -1, -1)
    old_to_new = np.empty(n_entity, dtype=np.int64); old_to_new[permutation] = np.arange(n_entity)
    for key in ("entity_raw_features", "entity_features", "entity_feature_mask", "entity_presence", "entity_type_index", "entity_position_raw", "entity_position", "entity_position_mask"):
        changed[key] = changed[key][:, :, permutation].copy()
    for key in ("agent_cpu_capacity_raw", "agent_cpu_capacity", "agent_cpu_capacity_mask"):
        changed[key] = changed[key][:, permutation].copy()
    for key in ("comm_source_index", "comm_target_index", "task_agent_agent_index", "logical_flow_source_index", "logical_flow_destination_index", "carrying_holder_index", "carrying_hop_source_index", "carrying_hop_destination_index", "route_node_indices"):
        values = changed[key]; valid = values >= 0; values[valid] = old_to_new[values[valid]]
    n_task = changed["task_presence"].shape[2]
    task_permutation = np.arange(n_task - 1, -1, -1)
    task_old_to_new = np.empty(n_task, dtype=np.int64); task_old_to_new[task_permutation] = np.arange(n_task)
    for key in ("task_raw_features", "task_features", "task_feature_mask", "task_presence", "task_lifecycle_index", "task_history_extended_raw_features", "task_history_extended_features", "task_history_extended_feature_mask"):
        changed[key] = changed[key][:, :, task_permutation].copy()
    for key in ("task_agent_task_index", "logical_flow_task_index"):
        values = changed[key]; valid = values >= 0; values[valid] = task_old_to_new[values[valid]]
    dag_values = changed["dag_edges"]; dag_valid = dag_values >= 0; dag_values[dag_valid] = task_old_to_new[dag_values[dag_valid]]
    n_flow = changed["logical_flow_presence"].shape[2]
    flow_permutation = np.arange(n_flow - 1, -1, -1)
    for key in ("logical_flow_known_mask", "logical_flow_presence", "logical_flow_index", "logical_flow_task_index", "logical_flow_type_index", "logical_flow_status_index", "logical_flow_epoch", "logical_flow_source_index", "logical_flow_destination_index", "logical_flow_raw_features", "logical_flow_features", "logical_flow_feature_mask", "carrying_known_mask", "carrying_active", "carrying_route_revision", "carrying_current_hop_index", "carrying_holder_index", "carrying_hop_source_index", "carrying_hop_destination_index", "carrying_raw_features", "carrying_features", "carrying_feature_mask", "route_node_indices", "route_node_mask"):
        changed[key] = changed[key][:, :, flow_permutation].copy()
    flow_old_to_new = np.empty(n_flow, dtype=np.int64); flow_old_to_new[flow_permutation] = np.arange(n_flow)
    for static in changed.get("sample_static", []):
        indices = static.get("input_entity_index", {})
        for object_id, old_index in list(indices.get("physical", {}).items()): indices["physical"][object_id] = int(old_to_new[old_index])
        for object_id, old_index in list(indices.get("task", {}).items()): indices["task"][object_id] = int(task_old_to_new[old_index])
        for object_id, old_index in list(indices.get("logical_flow", {}).items()): indices["logical_flow"][object_id] = int(flow_old_to_new[old_index])
        for row in static.get("agent_static_capability", []): row["agent_index"] = int(old_to_new[row["agent_index"]])
        for row in static.get("relation_endpoints", []):
            row["source_entity_index"] = int(old_to_new[row["source_entity_index"]]); row["target_entity_index"] = int(old_to_new[row["target_entity_index"]])
        for row in static.get("dag_relations", {}).get("rows", []):
            row["source_task_index"] = int(task_old_to_new[row["source_task_index"]]); row["target_task_index"] = int(task_old_to_new[row["target_task_index"]])
        old_types = static.get("input_entity_type_by_index", {}); static["input_entity_type_by_index"] = {str(int(old_to_new[int(old)])): value for old, value in old_types.items()}
    return changed, permutation, task_permutation, flow_permutation


def fixture_checks(model: PIJointGraphEncoder, tensor: dict[str, Any], graph: dict[str, Any]) -> dict[str, bool]:
    model.eval()
    tensor_before = _array_mapping_digest(tensor)
    graph_before = graph_semantic_digest(graph)
    baseline = model(tensor, graph)
    baseline_digest = output_semantic_digest(baseline)

    missing = copy.deepcopy(tensor)
    missing_candidates = np.argwhere(~missing["entity_feature_mask"] & missing["entity_presence"][..., None])
    missing_ignored = False; real_zero_distinct = False
    if len(missing_candidates):
        idx = tuple(map(int, missing_candidates[0])); missing["entity_raw_features"][idx] = 999999.0
        missing_ignored = torch.equal(baseline["physical"]["node_latent"], model(missing, graph, validate_inputs=False)["physical"]["node_latent"])
        missing["entity_feature_mask"][idx] = True; missing["entity_raw_features"][idx] = 0.0
        real_zero_distinct = not torch.equal(baseline["physical"]["node_latent"], model(missing, graph, validate_inputs=False)["physical"]["node_latent"])

    early = copy.deepcopy(tensor)
    bi, ei = map(int, np.argwhere(early["entity_presence"][:, 0] & early["entity_position_mask"][:, 0].all(axis=-1))[0])
    early["entity_position_raw"][bi, 0, ei, 0] += 100.0
    early_changes = not torch.equal(baseline["physical"]["node_latent"][bi, ei], model(early, graph, validate_inputs=False)["physical"]["node_latent"][bi, ei])
    absent_base = copy.deepcopy(tensor); absent_base["entity_presence"][bi, 0, ei] = False
    absent_extreme = copy.deepcopy(absent_base); absent_extreme["entity_position_raw"][bi, 0, ei] = 888888.0; absent_extreme["entity_raw_features"][bi, 0, ei] = 888888.0
    absent_ignored = torch.equal(model(absent_base, graph, validate_inputs=False)["physical"]["node_latent"], model(absent_extreme, graph, validate_inputs=False)["physical"]["node_latent"])

    future = copy.deepcopy(tensor)
    for key, value in list(future.items()):
        if key.startswith("target_") and isinstance(value, np.ndarray):
            future[key] = (~value) if value.dtype == bool else value + 991
    future_ignored = baseline_digest == output_semantic_digest(model(future, graph, validate_inputs=False))
    structural = copy.deepcopy(tensor)
    structural["logical_flow_raw_features"][..., 1] += 77
    structural["logical_flow_features"][..., 1] -= 88
    structural["logical_flow_epoch"] += 9
    structural["logical_flow_index"] += 19
    flow_structural_ignored = torch.equal(baseline["information"]["flow_relation_latent"], model(structural, graph, validate_inputs=False)["information"]["flow_relation_latent"])
    hop_endpoint = copy.deepcopy(graph); active_flow = np.argwhere(hop_endpoint["blocks"]["flow_carrying_state"]["active"])
    logical_endpoint_not_hop = False
    if len(active_flow):
        fbi, ffi = map(int, active_flow[0])
        hop_endpoint["blocks"]["flow_carrying_state"]["hop_destination_index"][fbi, ffi] = 0
        hop_unchanged = torch.equal(baseline["information"]["flow_relation_latent"], model(tensor, hop_endpoint, validate_inputs=False)["information"]["flow_relation_latent"])
        logical_endpoint = copy.deepcopy(graph); destinations = logical_endpoint["blocks"]["flow_relations"]["destination_index"]
        destinations[fbi, ffi] = (int(destinations[fbi, ffi]) + 1) % tensor["entity_presence"].shape[2]
        logical_changes = not torch.equal(baseline["information"]["flow_relation_latent"][fbi, ffi], model(tensor, logical_endpoint, validate_inputs=False)["information"]["flow_relation_latent"][fbi, ffi])
        logical_endpoint_not_hop = bool(hop_unchanged and logical_changes)

    no_phy_edges = copy.deepcopy(graph); no_phy_edges["blocks"]["physical_relations"]["presence"][:] = False; no_phy_edges["blocks"]["physical_relations"]["validity"][:] = False
    p2c_without_edge = torch.equal(baseline["diagnostics"]["p2c_initial_message"], model(tensor, no_phy_edges, validate_inputs=False)["diagnostics"]["p2c_initial_message"])
    comm_type = torch.as_tensor(graph["blocks"]["comm_relations"]["relation_type_index"])
    geo_valid = torch.as_tensor(graph["blocks"]["geo_comm_relations"]["validity"])
    p2c_masks = bool(torch.all(baseline["diagnostics"]["p2c_message"][comm_type != 2] == 0) and torch.all(baseline["diagnostics"]["p2c_message"][~geo_valid] == 0))

    padded = copy.deepcopy(tensor); padding = np.argwhere(~padded["task_presence"])
    padding_ignored = False
    if len(padding):
        pbi, phi, pti = map(int, padding[0]); padded["task_raw_features"][pbi, phi, pti] = 123456; padded["task_history_extended_raw_features"][pbi, phi, pti] = 123456
        changed = model(padded, graph, validate_inputs=False); active = torch.as_tensor(graph["blocks"]["task_nodes"]["presence"])
        padding_ignored = torch.equal(baseline["information"]["task_latent"][active], changed["information"]["task_latent"][active])

    permuted_tensor, permutation, task_permutation, flow_permutation = _consistent_slot_permutation(tensor)
    permuted_graph = build_typed_dual_graph_batch(permuted_tensor, PhysicalTopologyConfig())
    permuted = model(permuted_tensor, permuted_graph)
    equivariant = bool(
        torch.allclose(baseline["physical"]["node_latent"][:, permutation], permuted["physical"]["node_latent"], atol=1e-6)
        and torch.allclose(baseline["information"]["agent_latent"][:, permutation], permuted["information"]["agent_latent"], atol=1e-6)
        and torch.allclose(baseline["information"]["task_latent"][:, task_permutation], permuted["information"]["task_latent"], atol=1e-6)
        and torch.allclose(baseline["information"]["flow_relation_latent"][:, flow_permutation], permuted["information"]["flow_relation_latent"], atol=1e-6)
    )

    deterministic_model = PIJointGraphEncoder(model.config, tensor["contract"], graph["contract"], model.normalization_stats).eval()
    deterministic = baseline_digest == output_semantic_digest(deterministic_model(tensor, graph))
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "encoder.pt"; save_encoder_package(model, package)
        loaded = load_encoder_package(package, tensor["contract"], graph["contract"]).eval()
        round_trip = baseline_digest == output_semantic_digest(loaded(tensor, graph))
    train_model = PIJointGraphEncoder(model.config, tensor["contract"], graph["contract"], model.normalization_stats)
    grad_output = train_model(tensor, graph); sum(value.sum() for section in (grad_output["physical"], grad_output["information"]) for key, value in section.items() if key.endswith("latent")).backward()
    autograd = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in train_model.parameters()) and next(train_model.parameters()).device.type == "cpu"

    physical = graph["blocks"]["physical_relations"]; bidirectional = True
    for batch_index in range(physical["presence"].shape[0]):
        edges = [(int(s), int(t)) for s, t, valid in zip(physical["source_index"][batch_index], physical["target_index"][batch_index], physical["presence"][batch_index] & physical["validity"][batch_index]) if valid]
        bidirectional &= len(edges) == len(set(edges)) and all((t, s) in edges for s, t in edges)

    tampered = {"required_checks": {name: True for name in REQUIRED_ACCEPTANCE_CHECKS}, "passed": True, "scope": {name: False for name in ("rssm", "world_model", "future_action", "prediction", "loss", "planner", "training", "optimizer", "gpu", "locked_test", "formal_dataset")}}
    tampered["required_checks"]["future_target_isolation"] = False
    return {
        "missing_placeholder_ignored": bool(missing_ignored), "real_zero_distinct_from_missing": bool(real_zero_distinct),
        "early_valid_history_changes_latent": bool(early_changes), "absent_history_placeholder_ignored": bool(absent_ignored),
        "future_target_counterfactual_ignored": bool(future_ignored), "flow_redundant_and_id_fields_ignored": bool(flow_structural_ignored),
        "flow_logical_endpoint_not_carrying_hop": logical_endpoint_not_hop,
        "p2c_without_matching_physical_edge": bool(p2c_without_edge), "p2c_wireless_and_validity_masks": p2c_masks,
        "inactive_padding_isolated": bool(padding_ignored), "entity_slot_permutation_equivariant": equivariant,
        "deterministic_initialization_and_forward": bool(deterministic), "serialize_load_round_trip": bool(round_trip),
        "cpu_autograd_without_optimizer": bool(autograd), "physical_bidirectional_no_double_count": bool(bidirectional),
        "input_objects_unchanged": tensor_before == _array_mapping_digest(tensor) and graph_before == graph_semantic_digest(graph),
        "acceptance_receipt_tamper_rejected": not validate_encoder_acceptance(tampered)["passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.refresh_existing)
    tensor = load_flow_tensor_batch(TENSOR_PATH); graph = load_typed_dual_graph_batch(GRAPH_PATH)
    audit = build_encoder_input_audit(tensor, graph, UPSTREAM_STATS, GRAPH_MANIFEST, TENSOR_PATH)
    audit["source_hash_evidence"]["graph_manifest"] = str(GRAPH_MANIFEST.relative_to(ROOT)).replace("\\", "/")
    audit["source_hash_evidence"]["tensor_path"] = str(TENSOR_PATH.relative_to(ROOT)).replace("\\", "/")
    stats = fit_encoder_normalization_stats(tensor, graph, UPSTREAM_STATS)
    config = DualGraphEncoderConfig()
    model = PIJointGraphEncoder(config, tensor["contract"], graph["contract"], stats).eval()
    output = model(tensor, graph)
    contract_checks = validate_encoder_contract_checks(model, output, tensor, graph)
    fixtures = fixture_checks(model, tensor, graph)
    architecture = validate_encoder_architecture_checks(model)["checks"]
    architecture["fixed_normalization"] = stats["source_split"] == "dev_train" and audit["passed"]
    required = {
        "typed_feature_encoding": architecture["type_specific_encoders"],
        "explicit_feature_mask_input": fixtures["missing_placeholder_ignored"] and fixtures["real_zero_distinct_from_missing"],
        "objectwise_temporal_encoding": architecture["independent_grus"],
        "presence_gated_gru": fixtures["absent_history_placeholder_ignored"],
        "current_relation_encoding": contract_checks["passed"],
        "no_phy_edge_temporal_memory": architecture["no_relation_temporal_grus"],
        "no_comm_temporal_memory": architecture["no_relation_temporal_grus"],
        "logical_carrying_separate_then_fuse": architecture["logical_carrying_separate_then_fuse"],
        "one_flow_relation_latent": contract_checks["checks"]["single_flow_relation"],
        "relation_family_specific_processing": architecture["family_specific_processors"],
        "semantic_computational_direction_separated": architecture["direction_embedding"] and fixtures["physical_bidirectional_no_double_count"] and fixtures["flow_logical_endpoint_not_carrying_hop"],
        "reverse_comm": config.enable_reverse_comm, "reverse_flow": config.enable_reverse_flow,
        "reverse_task_agent": config.enable_reverse_task_agent, "dag_forward_only": not config.enable_reverse_dag,
        "direction_embedding": architecture["direction_embedding"],
        "relation_wise_aggregation": hasattr(model, "agent_family_fuse") and hasattr(model, "task_family_fuse"),
        "masked_mean": config.aggregator == "masked_mean",
        "residual_node_update": architecture["independent_node_updates"] and config.update_rule == "residual_layernorm",
        "residual_relation_update": len(model.relation_norms) == 5 and config.update_rule == "residual_layernorm",
        "p2a_gated_align_only": bool(torch.all((output["diagnostics"]["p2a_gate"] >= 0) & (output["diagnostics"]["p2a_gate"] <= 1))) and bool(torch.all(output["diagnostics"]["p2a_message"][~output["diagnostics"]["p2a_valid"]] == 0)),
        "p2c_gated_geocomm_only": fixtures["p2c_wireless_and_validity_masks"] and bool(torch.all((output["diagnostics"]["p2c_gate"] >= 0) & (output["diagnostics"]["p2c_gate"] <= 1))),
        "wired_p2c_disabled": fixtures["p2c_wireless_and_validity_masks"],
        "no_matching_physical_edge_requirement": fixtures["p2c_without_matching_physical_edge"],
        "no_info_to_physical": architecture["no_forbidden_shortcut_modules"],
        "no_shortcut_relations": architecture["no_forbidden_shortcut_modules"],
        "history_causal": fixtures["early_valid_history_changes_latent"],
        "future_target_isolation": fixtures["future_target_counterfactual_ignored"],
        "index_not_numeric_feature": fixtures["flow_redundant_and_id_fields_ignored"] and fixtures["entity_slot_permutation_equivariant"],
        "permutation_equivariance": fixtures["entity_slot_permutation_equivariant"],
        "inactive_padding_isolation": fixtures["inactive_padding_isolated"],
        "output_alignment": all(contract_checks["checks"][name] for name in ("physical_shape", "agent_shape", "task_shape", "flow_shape")),
        "z_pi_not_world_model_latent": contract_checks["checks"]["z_pi_only"],
        "deterministic_forward": fixtures["deterministic_initialization_and_forward"],
        "serialize_load": fixtures["serialize_load_round_trip"],
        "autograd_smoke": fixtures["cpu_autograd_without_optimizer"],
        "scope": True,
    }
    scope = {name: False for name in ("rssm", "world_model", "future_action", "prediction", "loss", "planner", "training", "optimizer", "gpu", "locked_test", "formal_dataset")}
    required["scope"] = all(value is False for value in scope.values()) and config.development_only and not config.research_frozen
    passed = bool(audit["passed"] and contract_checks["passed"] and all(architecture.values()) and all(fixtures.values()) and all(required.values()))
    acceptance = {
        "schema_version": "PI-JWM-Step-4.3B-Acceptance-v1", "evidence_class": "UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE",
        "required_checks": required, "architecture_checks": architecture, "contract_checks": contract_checks,
        "negative_counterfactual_checks": fixtures, "scope": scope, "passed": passed,
        "semantic_output_digest": output_semantic_digest(output),
    }
    if validate_encoder_acceptance(acceptance)["passed"] is not passed:
        raise RuntimeError("acceptance validator disagrees with builder verdict")
    write_json(args.output_dir / "encoder_config.json", asdict(config))
    write_json(args.output_dir / "encoder_input_audit.json", audit)
    write_json(args.output_dir / "encoder_normalization_stats.json", stats)
    write_json(args.output_dir / "acceptance_report.json", acceptance)
    save_encoder_package(model, args.output_dir / "untrained_encoder_package.pt")
    torch.save(output, args.output_dir / "deterministic_encoder_output.pt")
    manifest = {
        "schema_version": "PI-JWM-Step-4.3B-Artifact-Manifest-v1", "evidence_class": "UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE",
        "sources": {
            "history_tensor": {"path": str(TENSOR_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TENSOR_PATH)},
            "typed_graph": {"path": str(GRAPH_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(GRAPH_PATH)},
            "upstream_normalization": {"path": str(UPSTREAM_STATS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(UPSTREAM_STATS)},
        },
        "files": {}, "training": False, "optimizer": False, "gpu": False, "locked_test": False, "formal_dataset": False,
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "manifest.json": manifest["files"][path.name] = sha256(path)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"passed": passed, "required_checks": required, "fixtures": fixtures, "artifact": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
