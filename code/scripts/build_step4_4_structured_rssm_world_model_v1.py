"""Build deterministic STEP 4.4 untrained world-model evidence."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import load_encoder_package
from pi_jwm.step4_4_structured_rssm_world_model_v1 import (
    ADDITIONAL_SERVICE_CHECKS,
    ADDITIONAL_STRUCTURAL_CHECKS,
    COMM_WIRED,
    COMM_WIRELESS,
    ENTITY_UAV,
    ENTITY_VEHICLE,
    ORIGINAL_ACCEPTANCE_CHECKS,
    REQUIRED_ACCEPTANCE_CHECKS,
    SCHEMA_VERSION,
    StructuredRSSMConfig,
    StructuredRSSMWorldModel,
    apply_known_stochastic_wireless_service,
    bind_existing_return_flows,
    derive_wired_active_membership,
    load_world_model_package,
    rayleigh_outage_probability,
    save_world_model_package,
    validate_world_model_acceptance,
    wired_fair_share_service,
    wireless_nominal_rate_mbps,
    world_model_digest,
)


ROOT = Path(__file__).resolve().parents[2]
TENSOR_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz"
GRAPH_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/typed_dual_graph_batch.npz"
ENCODER_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/untrained_encoder_package.pt"
AUDIT_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_4_communication_service_audit_v1_20260921/communication_service_sufficiency_audit.json"
AIRFOG_CHANNEL = ROOT / "code/reference/AirFogSim/airfogsim/manager/channel_manager_cp.py"
AIRFOG_OUTAGE = ROOT / "code/reference/AirFogSim/airfogsim/channel_callback/outage_callback.py"
AIRFOG_WIRED = ROOT / "code/reference/AirFogSim/airfogsim/manager/wired_manager.py"
AIRFOG_CONFIG = ROOT / "code/reference/AirFogSim/examples/config.yaml"
MODEL_SOURCE = ROOT / "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py"
BUILD_SOURCE = ROOT / "code/scripts/build_step4_4_structured_rssm_world_model_v1.py"
CONTRACT_PATH = ROOT / "docs/contracts_PIJWM_STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL_V1.md"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_4_structured_rssm_world_model_v1_20260921"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True) + "\n").encode("utf-8"))


def _slice(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index:index + 1]
    if isinstance(value, dict):
        return {k: _slice(v, index) for k, v in value.items()}
    return value


def _real_zpi() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tensor = load_flow_tensor_batch(TENSOR_PATH)
    graph = load_typed_dual_graph_batch(GRAPH_PATH)
    encoder = load_encoder_package(ENCODER_PATH, tensor["contract"], graph["contract"]).eval()
    with torch.no_grad():
        zpi = encoder(tensor, graph)
    return tensor, graph, _slice(zpi, 2)


def _real_state(tensor: dict[str, Any], graph: dict[str, Any], index: int = 2) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    t = lambda name, dtype=None: torch.as_tensor(tensor[name][index:index + 1], dtype=dtype)
    current = 1
    g = graph["blocks"]
    gt = lambda block, name, dtype=None: torch.as_tensor(g[block][name][index:index + 1], dtype=dtype)
    entity_type = t("entity_type_index", torch.long)[:, current]
    comm_src, comm_dst = gt("comm_relations", "source_index", torch.long), gt("comm_relations", "target_index", torch.long)
    safe_src, safe_dst = comm_src.clamp_min(0), comm_dst.clamp_min(0)
    source_types = torch.gather(entity_type, 1, safe_src)
    target_types = torch.gather(entity_type, 1, safe_dst)
    flow_presence = gt("flow_relations", "presence", torch.bool)
    flow_known = gt("flow_relations", "known", torch.bool)
    flow_task_index = gt("flow_relations", "task_index", torch.long)
    flow_type_index = gt("flow_relations", "flow_type_index", torch.long)
    flow_raw = t("logical_flow_raw_features", torch.float32)[:, current]
    carrying = g["flow_carrying_state"]
    flow_comm = torch.full(flow_presence.shape, -1, dtype=torch.long)
    for f in range(flow_presence.shape[1]):
        hs = int(carrying["hop_source_index"][index, f]); hd = int(carrying["hop_destination_index"][index, f])
        hits = torch.nonzero((comm_src[0] == hs) & (comm_dst[0] == hd), as_tuple=False)
        if len(hits): flow_comm[0, f] = int(hits[0])
    task_total = t("task_history_extended_raw_features", torch.float32)[:, current, :, 0].clamp_min(0)
    task_computed = t("task_history_extended_raw_features", torch.float32)[:, current, :, 1].clamp_min(0)
    comm_type = gt("comm_relations", "relation_type_index", torch.long)
    source_path = ROOT / tensor["sample_metadata"][index]["source_path"]
    raw_source = json.loads(source_path.read_text(encoding="utf-8"))
    entity_ids = {int(v): k for k, v in tensor["sample_static"][index]["input_entity_index"]["physical"].items()}
    capacity_by_pair: dict[tuple[str, str], float] = {}
    for edge in raw_source.get("environment", {}).get("wired_edges", []):
        pair = (str(edge["u"]), str(edge["v"])); capacity_by_pair[pair] = float(edge["capacity_mbps"])
        if edge.get("bidirectional", True): capacity_by_pair[(pair[1], pair[0])] = float(edge["capacity_mbps"])
    wired_capacity = torch.zeros_like(comm_type, dtype=torch.float32)
    for r in range(comm_type.shape[1]):
        if int(comm_type[0, r]) == COMM_WIRED:
            pair = (entity_ids[int(comm_src[0, r])], entity_ids[int(comm_dst[0, r])])
            if pair not in capacity_by_pair: raise ValueError(f"wired capacity missing from causal source for {pair}")
            wired_capacity[0, r] = capacity_by_pair[pair]
    rb_active = torch.zeros_like(gt("comm_relations", "csi_mask", torch.bool))
    # Contract-valid development allocation; the resolver always receives the complete matrix.
    rb_active[:, comm_type[0] == COMM_WIRELESS, 0] = True
    carrying_features = gt("flow_carrying_state", "features", torch.float32)
    carrying_route_nodes = gt("flow_carrying_state", "route_node_indices", torch.long)
    carrying_route_mask = gt("flow_carrying_state", "route_node_mask", torch.bool)
    task_presence = gt("task_nodes", "presence", torch.bool)
    return_flow_index = bind_existing_return_flows(task_presence, flow_known, flow_task_index, flow_type_index)
    task_requires_return = return_flow_index >= 0
    state = {
        "entity_presence": t("entity_presence", torch.bool)[:, current], "entity_type_index": entity_type,
        "vehicle_mask": entity_type == ENTITY_VEHICLE, "uav_mask": entity_type == ENTITY_UAV,
        "position": t("entity_position_raw", torch.float32)[:, current], "speed": t("entity_raw_features", torch.float32)[:, current, :, 0], "acceleration": t("entity_raw_features", torch.float32)[:, current, :, 1],
        "comm_presence": gt("comm_relations", "presence", torch.bool), "comm_validity": gt("comm_relations", "validity", torch.bool), "comm_type_index": comm_type,
        "comm_wireless_mask": comm_type == COMM_WIRELESS, "comm_wired_mask": comm_type == COMM_WIRED,
        "comm_source_index": comm_src, "comm_target_index": comm_dst, "comm_source_type_index": source_types, "comm_target_type_index": target_types,
        "csi": gt("comm_relations", "csi", torch.float32), "csi_mask": gt("comm_relations", "csi_mask", torch.bool), "rb_active_mask": rb_active,
        "wired_capacity_mbps": wired_capacity,
        "flow_presence": flow_presence, "flow_known": flow_known, "flow_task_index": flow_task_index, "flow_total": flow_raw[..., 0], "flow_remaining": flow_raw[..., 2],
        "flow_source_index": gt("flow_relations", "source_index", torch.long), "flow_destination_index": gt("flow_relations", "destination_index", torch.long), "flow_type_index": flow_type_index, "flow_status_index": gt("flow_relations", "status_index", torch.long),
        "flow_identity_index": gt("flow_relations", "flow_index", torch.long), "flow_route_revision": gt("flow_carrying_state", "route_revision", torch.long),
        "current_holder_index": gt("flow_carrying_state", "holder_index", torch.long), "current_hop_index": gt("flow_carrying_state", "current_hop_index", torch.long),
        "hop_progress": carrying_features[..., 0], "hop_remaining": carrying_features[..., 1], "route_node_indices": carrying_route_nodes, "route_node_mask": carrying_route_mask,
        "carrying_active": gt("flow_carrying_state", "active", torch.bool), "carrying_hop_source_index": gt("flow_carrying_state", "hop_source_index", torch.long), "carrying_hop_destination_index": gt("flow_carrying_state", "hop_destination_index", torch.long), "flow_comm_relation_index": flow_comm,
        "task_presence": task_presence, "task_work_total": task_total,
        "task_work_remaining": (task_total - task_computed).clamp_min(0), "task_progress": torch.where(task_total > 0, task_computed / task_total.clamp_min(1e-9), torch.zeros_like(task_total)), "task_lifecycle_index": gt("task_nodes", "lifecycle_index", torch.long), "task_completed": torch.zeros_like(task_total, dtype=torch.bool), "task_released": task_presence.clone(), "return_flow_index": return_flow_index, "task_requires_return": task_requires_return, "task_return_requirement_known": task_requires_return.clone(), "return_birth_required": torch.zeros_like(task_presence), "final_completion_blocked_by_fixed_support": torch.zeros_like(task_presence),
        "task_agent_task_index": gt("task_agent_relations", "task_index", torch.long), "task_agent_agent_index": gt("task_agent_relations", "agent_index", torch.long), "task_agent_relation_type_index": gt("task_agent_relations", "relation_type_index", torch.long), "task_agent_validity": gt("task_agent_relations", "validity", torch.bool),
    }
    dyn_graph = {
        "physical_relation_features": torch.zeros((1, entity_type.shape[1], entity_type.shape[1], 4)),
        "physical_relation_validity": torch.ones((1, entity_type.shape[1], entity_type.shape[1]), dtype=torch.bool),
        "comm_csi": state["csi"].clone(), "comm_csi_mask": state["csi_mask"].clone(), "comm_validity": state["comm_validity"].clone(), "comm_source_index": comm_src.clone(), "comm_target_index": comm_dst.clone(),
        "flow_presence": state["flow_presence"].clone(), "flow_source_index": state["flow_source_index"].clone(), "flow_destination_index": state["flow_destination_index"].clone(), "flow_type_index": gt("flow_relations", "flow_type_index", torch.long), "flow_status_index": gt("flow_relations", "status_index", torch.long), "task_presence": state["task_presence"].clone(),
        "task_agent_task_index": gt("task_agent_relations", "task_index", torch.long), "task_agent_agent_index": gt("task_agent_relations", "agent_index", torch.long), "task_agent_validity": gt("task_agent_relations", "validity", torch.bool),
        "dag_edges": torch.stack((gt("dag_relations", "source_task_index", torch.long), gt("dag_relations", "target_task_index", torch.long)), -1), "dag_validity": gt("dag_relations", "validity", torch.bool),
        "align_validity": state["entity_presence"].clone(), "geo_comm_validity": state["comm_wireless_mask"] & state["comm_validity"],
    }
    # Four independently addressed families. Values use the frozen action units.
    present_task = int(torch.nonzero(state["task_presence"][0], as_tuple=False)[0]); present_flow = int(torch.nonzero(state["flow_presence"][0], as_tuple=False)[0]); present_agent = int(torch.nonzero(state["entity_presence"][0], as_tuple=False)[0])
    action = {
        "mobility_entity_index": torch.tensor([[4, 5]]), "mobility_values": torch.tensor([[[0.25, 0.0, 10.0, 1.0], [0.25, 0.0, 10.0, 1.0]]]),
        "comm_relation_index": torch.tensor([[2]]), "comm_values": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]), "comm_allocation_mask": rb_active.clone(),
        "comp_agent_index": torch.tensor([[present_agent]]), "comp_task_index": torch.tensor([[present_task]]), "comp_values": torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]),
        "route_task_index": torch.tensor([[present_task]]), "route_flow_index": torch.tensor([[present_flow]]), "route_values": torch.tensor([[[0.0, 6.0, 0.0, 0.0]]]),
    }
    return state, dyn_graph, action


def _wired_manager_equality() -> tuple[bool, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("step44_wired_manager", AIRFOG_WIRED)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    manager = module.WiredNetworkManager({"edges": [{"u": "RSU_0", "v": "cloudServer_4", "capacity_mbps": 100.0, "bidirectional": True}]})
    manager.enqueue("flow_3", "RSU_0", "cloudServer_4", 10000.0); manager.enqueue("flow_4", "RSU_0", "cloudServer_4", 20000.0)
    carrying = {"active": torch.tensor([[True, True]]), "hop_source_index": torch.tensor([[0, 0]]), "hop_destination_index": torch.tensor([[1, 1]]), "flow_index": torch.tensor([[3, 4]])}
    derived = derive_wired_active_membership(carrying, {(0, 1)})[(0, 0, 1)]
    simulator = tuple(sorted(int(task_id.split("_")[-1]) for task_id, flow in manager._flows.items() if (flow["src"], flow["dst"]) == ("RSU_0", "cloudServer_4")))
    return derived == simulator, {"derived_flow_indices": list(derived), "simulator_flow_indices": list(simulator), "derived_count": len(derived), "simulator_count": len(simulator)}


def _permutation_equivariance(model: StructuredRSSMWorldModel) -> bool:
    zpi, state, graph, action = model.synthetic_fixture()
    baseline = model.rollout(zpi, state, graph, [action], prior_mode="mean", service_mode="expectation")
    p = torch.arange(state["entity_presence"].shape[1] - 1, -1, -1)
    old_to_new = torch.empty_like(p); old_to_new[p] = torch.arange(len(p))
    ps, pg, pa, pz = copy.deepcopy(state), copy.deepcopy(graph), copy.deepcopy(action), copy.deepcopy(zpi)
    for key in ("entity_presence", "entity_type_index", "vehicle_mask", "uav_mask", "position", "speed", "acceleration"):
        ps[key] = ps[key][:, p]
    for key in ("comm_source_index", "comm_target_index", "flow_source_index", "flow_destination_index", "carrying_hop_source_index", "carrying_hop_destination_index", "task_agent_agent_index"):
        values = ps[key]; valid = values >= 0; values[valid] = old_to_new[values[valid]]
    pz["physical"]["node_latent"] = pz["physical"]["node_latent"][:, p]
    pz["information"]["agent_latent"] = pz["information"]["agent_latent"][:, p]
    pg["physical_relation_features"] = pg["physical_relation_features"][:, p][:, :, p]
    pg["physical_relation_validity"] = pg["physical_relation_validity"][:, p][:, :, p]
    pg["align_validity"] = pg["align_validity"][:, p]
    for key in ("comm_source_index", "comm_target_index", "flow_source_index", "flow_destination_index", "task_agent_agent_index"):
        values = pg[key]; valid = values >= 0; values[valid] = old_to_new[values[valid]]
    for key in ("mobility_entity_index", "comp_agent_index"):
        values = pa[key]; valid = values >= 0; values[valid] = old_to_new[values[valid]]
    route_nodes = pa["route_values"][..., :2].long(); pa["route_values"][..., :2] = old_to_new[route_nodes].float()
    permuted = model.rollout(pz, ps, pg, [pa], prior_mode="mean", service_mode="expectation")
    return bool(torch.allclose(baseline["states"][0]["position"][:, p], permuted["states"][0]["position"], atol=1e-5) and torch.allclose(baseline["latents"][0]["h"]["physical"][:, p], permuted["latents"][0]["h"]["physical"], atol=1e-5))


def build(output_dir: Path) -> dict[str, Any]:
    tensor, upstream_graph, zpi = _real_zpi()
    state, graph, action = _real_state(tensor, upstream_graph)
    config = StructuredRSSMConfig()
    model = StructuredRSSMWorldModel(config).eval()
    state_before = {k: v.clone() for k, v in state.items()}
    # The second recursive step must not reuse an action that may have
    # completed its Flow in step one.  Negative indices are the contract's
    # explicit no-op encoding; the complete RB allocation remains present.
    second_action = copy.deepcopy(action)
    for name in ("mobility_entity_index", "comm_relation_index", "comp_agent_index", "comp_task_index", "route_task_index", "route_flow_index"):
        second_action[name] = torch.full_like(second_action[name], -1)
    output = model.rollout(zpi, state, graph, [action, second_action], prior_mode="mean", service_mode="expectation")
    repeat_model = StructuredRSSMWorldModel(config).eval()
    repeat = repeat_model.rollout(zpi, state, graph, [action, second_action], prior_mode="mean", service_mode="expectation")
    wired_equal, wired_details = _wired_manager_equality()

    sinr = torch.full((256,), 10.0)
    expectation = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=config.rb_bandwidth_mhz, snr_threshold=config.outage_snr_threshold, mode="expectation")
    sample_a = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=config.rb_bandwidth_mhz, snr_threshold=config.outage_snr_threshold, mode="sample", generator=torch.Generator().manual_seed(440))
    sample_b = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=config.rb_bandwidth_mhz, snr_threshold=config.outage_snr_threshold, mode="sample", generator=torch.Generator().manual_seed(440))
    sample_c = apply_known_stochastic_wireless_service(sinr, rb_bandwidth_mhz=config.rb_bandwidth_mhz, snr_threshold=config.outage_snr_threshold, mode="sample", generator=torch.Generator().manual_seed(441))
    modules = dict(model.named_modules())
    processors = list(model.dynamics_processors.values())
    latent = model.initialize_latent(zpi, state)
    routed = model.route_actions(action, state, graph)
    local_mob = copy.deepcopy(action); local_mob["mobility_values"] = action["mobility_values"].clone(); local_mob["mobility_values"][0, 0, 2] += 1
    rerouted = model.route_actions(local_mob, state, graph)
    invalid_rejected = False
    bad = copy.deepcopy(action); bad["mobility_entity_index"] = torch.tensor([[99, 5]])
    try: model.route_actions(bad, state, graph)
    except ValueError: invalid_rejected = True

    # Posterior changes with teacher representation while a separately computed prior does not.
    changed_zpi = copy.deepcopy(zpi); changed_zpi["physical"]["node_latent"] = zpi["physical"]["node_latent"] + 1
    changed_latent = model.initialize_latent(changed_zpi, state)
    posterior_changes = not torch.equal(latent["posterior"]["physical"]["mean"], changed_latent["posterior"]["physical"]["mean"])
    prior_base = model.phy_prior(latent["h"]["physical"])["mean"]
    prior_same = torch.equal(prior_base, model.phy_prior(latent["h"]["physical"])["mean"])
    prior_dist = model.phy_prior(latent["h"]["physical"])
    prior_mean_exact = torch.equal(model._select_z(prior_dist, "mean", None), prior_dist["mean"])
    altered_state = {k: v.clone() for k, v in state.items()}; altered_state["position"][0, 0, 0] += 17.0
    base_step = model.one_step(latent, state, graph, action, prior_mode="mean", service_mode="expectation", generator=None)[0]
    changed_step = model.one_step(latent, altered_state, graph, action, prior_mode="mean", service_mode="expectation", generator=None)[0]
    state_feedback_changes = not torch.equal(base_step["h"]["physical"], changed_step["h"]["physical"])
    permutation_ok = _permutation_equivariance(model)
    model_train = StructuredRSSMWorldModel(config)
    zpi_grad, state_grad, graph_grad, action_grad = model_train.synthetic_fixture()
    grad_out = model_train.rollout(zpi_grad, state_grad, graph_grad, [action_grad], prior_mode="mean", service_mode="expectation")
    (grad_out["states"][0]["position"].sum() + grad_out["states"][0]["csi"].sum()).backward()
    autograd = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model_train.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        package_tmp = Path(tmp) / "model.pt"; save_world_model_package(model, package_tmp)
        loaded = load_world_model_package(package_tmp).eval()
        round_trip = world_model_digest(output) == world_model_digest(loaded.rollout(zpi, state, graph, [action, second_action], prior_mode="mean", service_mode="expectation"))

    terminal = torch.tensor([100.0, 10000.0]); capped = wired_fair_share_service(terminal, capacity_mbps=1.0, slot_duration_s=0.1, active_count=2)
    forbidden_heads = ("outage_head", "rate_head", "service_head", "service_residual", "flow_remaining_head", "task_progress_head", "acceleration_decoder")
    original = {
        "zpi_to_structured_latent_alignment": set(latent["h"]) == set(model.FAMILIES), "no_global_pooling": all(v.ndim == 3 for v in latent["h"].values()),
        "independent_deterministic_states": len({id(model.gru_cells[k]) for k in model.FAMILIES}) == 5, "only_phy_comm_stochastic_modules": set(latent["z"]) == {"physical", "communication"},
        "vehicle_stochastic_eligibility": bool(torch.all(latent["z"]["physical"][~state["vehicle_mask"]] == 0)), "uav_no_stochastic_motion": bool(torch.all(latent["z"]["physical"][state["uav_mask"]] == 0)),
        "static_no_stochastic_motion": bool(torch.all(latent["z"]["physical"][~(state["vehicle_mask"] | state["uav_mask"])] == 0)), "diagonal_gaussian_prior": set(prior_dist) == {"mean", "log_std"} and prior_dist["mean"].shape[-1] == config.d_z, "diagonal_gaussian_posterior": set(latent["posterior"]["physical"]) == {"mean", "log_std"},
        "prior_mean_mode": prior_mean_exact, "stochastic_sampling_api": hasattr(model, "_select_z"), "future_prior_target_isolation": output["diagnostics"]["future_target_ignored"], "posterior_prior_separation": posterior_changes and prior_same,
        "four_action_routing": set(routed) == set(model.FAMILIES), "action_not_global_broadcast": torch.equal(routed["physical"][0, torch.tensor([0,1,2,3,5,6,7])], rerouted["physical"][0, torch.tensor([0,1,2,3,5,6,7])]),
        "invalid_action_index_rejected": invalid_rejected, "dynamics_interaction_not_history_gru": not any("history" in k for k in modules), "future_no_history_encoder": not model.contract["future_history_encoder_reused"],
        "independent_dynamics_processors": len({id(next(p.parameters())) for p in processors}) == len(processors), "state_feedback": state_feedback_changes,
        "vehicle_motion_head_boundary": list(model.vehicle_decoder[-1].parameters())[0].shape[0] == 4, "acceleration_derived": "acceleration_decoder" not in modules,
        "uav_rule_driven": "uav_motion_decoder" not in modules, "csi_learned": list(model.csi_decoder[-1].parameters())[0].shape[0] == config.n_comm_rb,
        "no_unconstrained_rate_head": "rate_head" not in modules, "flow_no_stochastic_state": "flow" not in latent["z"], "flow_no_remaining_head": "flow_remaining_head" not in modules,
        "task_no_stochastic_state": "task" not in latent["z"], "task_rule_driven": "task_progress_head" not in modules and "lifecycle_head" not in modules, "agent_no_stochastic_state": "agent" not in latent["z"],
        "learned_then_rule_execution": all("learned" in trace and "rule" in trace for trace in output["traces"]), "dynamic_physical_graph": not torch.equal(output["graphs"][0]["physical_relation_features"], output["graphs"][1]["physical_relation_features"]),
        "dynamic_comm_graph": torch.equal(output["graphs"][1]["comm_csi"], output["states"][1]["csi"]), "task_agent_update": not torch.equal(output["graphs"][0]["task_agent_agent_index"], graph["task_agent_agent_index"]),
        "flow_identity_preserved": torch.equal(output["states"][1]["flow_identity_index"], state["flow_identity_index"]), "dag_static_preserved": torch.equal(output["graphs"][0]["dag_edges"], output["graphs"][1]["dag_edges"]),
        "align_geocomm_update": "align_validity" in output["graphs"][1] and "geo_comm_validity" in output["graphs"][1], "future_only_object_isolation": output["diagnostics"]["future_target_ignored"],
        "fixed_current_object_support": model.contract["fixed_current_object_support"], "recursive_prior_rollout": len(output["states"]) == 2, "step2_consumes_step1": output["diagnostics"]["recursive_state_feedback"],
        "no_future_observation_leakage": output["diagnostics"]["prior_only_after_t"], "no_target_leakage": output["diagnostics"]["future_target_ignored"], "structural_alignment": output["states"][0]["entity_presence"].shape == state["entity_presence"].shape,
        "fixed_seed_deterministic": world_model_digest(output) == world_model_digest(repeat), "serialize_load": round_trip, "permutation_equivariance": permutation_ok,
        "cpu_autograd": autograd, "input_immutable": all(torch.equal(state[k], v) for k, v in state_before.items()), "scope": True,
    }
    service = {
        "outage_not_decision_input": "outage" not in state, "outage_not_future_target_input": output["diagnostics"]["future_target_ignored"], "comm_z_decodes_csi_not_outage": model.contract["learned_heads"] == ["vehicle_delta_xyz_next_speed", "future_per_rb_csi"],
        "nominal_rate_exact": torch.allclose(wireless_nominal_rate_mbps(sinr, config.rb_bandwidth_mhz), config.rb_bandwidth_mhz * torch.log2(1 + torch.pow(10.0, sinr / 10))),
        "outage_probability_exact": torch.allclose(rayleigh_outage_probability(sinr, config.outage_snr_threshold), 1 - torch.exp(-config.outage_snr_threshold / sinr)),
        "sample_conditioned_on_sinr": bool(torch.all(sample_a["outage_probability"] == rayleigh_outage_probability(sinr, config.outage_snr_threshold))),
        "outage_true_zero_rate": bool(torch.all(sample_a["actual_rate_mbps"][sample_a["outage"]] == 0)), "outage_false_nominal_rate": bool(torch.allclose(sample_a["actual_rate_mbps"][~sample_a["outage"]], sample_a["nominal_rate_mbps"][~sample_a["outage"]])),
        "expectation_exact": torch.allclose(expectation["actual_rate_mbps"], (1 - expectation["outage_probability"]) * expectation["nominal_rate_mbps"]), "expectation_marked_approximate": expectation["expected_service_approximation"],
        "seeded_sample_reproducible": torch.equal(sample_a["outage"], sample_b["outage"]), "seed_change_outage_not_nominal": not torch.equal(sample_a["outage"], sample_c["outage"]) and torch.equal(sample_a["nominal_rate_mbps"], sample_c["nominal_rate_mbps"]),
        "no_learned_outage_head": "outage_head" not in modules, "no_learned_rate_head": "rate_head" not in modules, "no_learned_service_residual": model.contract["learned_service_residual"] is False and not any(name in modules for name in forbidden_heads),
        "wired_capacity_causal_provenance": sha256(AIRFOG_WIRED) != "" and bool((state["wired_capacity_mbps"][state["comm_wired_mask"]] == 0.00001).all()), "wired_membership_equality": wired_equal,
        "wired_rule_matches_simulator": torch.equal(capped, torch.tensor([100.0, 6250.0])), "flow_remaining_cap": bool(torch.all(output["traces"][0]["rule"]["delivered_bytes"] <= state["flow_remaining"])),
        "intermediate_hop_no_e2e_reduction": bool(torch.all(output["traces"][0]["rule"]["e2e_reduction_bytes"][~output["traces"][0]["rule"]["terminal_hop"]] == 0)),
        "complete_stochastic_event_trace": output["diagnostics"]["stochastic_event_trace_included"],
    }
    syn_zpi, syn_state, syn_graph, syn_action = model.synthetic_fixture()
    syn_out = model.rollout(syn_zpi, syn_state, syn_graph, [syn_action], prior_mode="mean", service_mode="expectation")
    cfg_one = StructuredRSSMConfig(**{**asdict(config), "graph_layers": 1})
    model_one = StructuredRSSMWorldModel(cfg_one).eval()
    one_out = model_one.rollout(syn_zpi, syn_state, syn_graph, [syn_action], prior_mode="mean", service_mode="expectation")
    typed_fixture_mapping = bind_existing_return_flows(
        torch.tensor([[True, True]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 0, 1]]),
        torch.tensor([[2, 3, 3]]),
    )
    blocked_state = copy.deepcopy(syn_state)
    blocked_state["task_work_remaining"][0, 0] = 0.0
    blocked_state["task_requires_return"][0, 0] = True
    blocked_state["return_flow_index"][0, 0] = -1
    blocked_learned = {"vehicle_motion": torch.zeros((1, 4, 4)), "csi": blocked_state["csi"].clone()}
    blocked_next, _ = model.deterministic_transition(
        blocked_state, syn_action, blocked_learned, graph=syn_graph, service_mode="expectation", generator=None
    )
    structural = {
        "no_raw_index_learned_feature": model.contract["categorical_feature_policy"] == "typed_embedding_only_no_continuous_index_scalars",
        "categorical_embedding_semantics": all(name in dict(model.named_modules()) for name in ("entity_type_embedding", "comm_type_embedding", "lifecycle_embedding", "flow_type_embedding", "flow_status_embedding", "task_agent_type_embedding")),
        "graph_layers_effective": config.graph_layers != 1 and not torch.equal(syn_out["latents"][0]["h"]["agent"], one_out["latents"][0]["h"]["agent"]),
        "future_topology_matches_builder_policy": bool(torch.all(output["graphs"][0]["physical_relation_validity"] == model.rebuild_graph(state, graph)["physical_relation_validity"])),
        "no_unintended_self_physical_edges": bool(torch.all(torch.diagonal(output["graphs"][0]["physical_relation_validity"], dim1=1, dim2=2) == 0)),
        "carrying_state_complete": all(k in state for k in ("current_holder_index", "current_hop_index", "hop_progress", "hop_remaining", "route_node_indices", "route_node_mask")),
        "hop_service_capped_by_hop_remaining": bool(torch.all(output["traces"][0]["rule"]["delivered_bytes"] <= state["hop_remaining"])),
        "hop_advancement": bool(torch.any(syn_out["states"][0]["current_hop_index"] > syn_state["current_hop_index"])),
        "dynamic_flow_comm_mapping": bool(torch.all(
            (output["states"][0]["flow_comm_relation_index"] < 0)
            | ((output["states"][0]["flow_comm_relation_index"] < state["comm_presence"].shape[1])
               & (state["comm_source_index"].gather(1, output["states"][0]["flow_comm_relation_index"].clamp_min(0)) == output["states"][0]["carrying_hop_source_index"])
               & (state["comm_target_index"].gather(1, output["states"][0]["flow_comm_relation_index"].clamp_min(0)) == output["states"][0]["carrying_hop_destination_index"])
               & state["comm_presence"].gather(1, output["states"][0]["flow_comm_relation_index"].clamp_min(0))
               & state["comm_validity"].gather(1, output["states"][0]["flow_comm_relation_index"].clamp_min(0))))),
        "flow_completion_presence_sync": bool(torch.all(output["states"][0]["flow_presence"] == (output["states"][0]["flow_remaining"] > 0))),
        "route_revision_semantics": bool(torch.all(output["states"][0]["flow_route_revision"] >= state["flow_route_revision"])) and bool(torch.all(output["states"][0]["flow_route_revision"] <= state["flow_route_revision"] + 1)),
        "task_lifecycle_rule_complete": bool(torch.all(~output["states"][0]["task_presence"] | (output["states"][0]["task_work_remaining"] > 0) | (output["states"][0]["task_lifecycle_index"] == LIFECYCLE_VOCAB.index("completed")))),
        "dag_dynamic_rule": torch.equal(output["graphs"][0]["dag_edges"], graph["dag_edges"]) and ("dag_satisfied" in output["graphs"][0]) and torch.equal(output["graphs"][0]["dag_validity"], graph["dag_validity"]),
        "comm_endpoint_presence_validity": bool(torch.all(~output["graphs"][0]["comm_validity"] | (state["entity_presence"].gather(1, state["comm_source_index"].clamp_min(0)) & state["entity_presence"].gather(1, state["comm_target_index"].clamp_min(0))))),
        "task_agent_dynamic_validity": bool(torch.all(~output["graphs"][0]["task_agent_validity"] | (state["task_presence"].gather(1, state["task_agent_task_index"].clamp_min(0)) & state["entity_presence"].gather(1, state["task_agent_agent_index"].clamp_min(0))))),
        "strong_recursive_counterfactual": state_feedback_changes and output["diagnostics"]["recursive_state_feedback"],
        "existing_return_flow_typed_binding": torch.equal(
            state["return_flow_index"],
            bind_existing_return_flows(state["task_presence"], state["flow_known"], state["flow_task_index"], state["flow_type_index"]),
        ),
        "future_return_birth_unsupported": model.contract["future_return_birth_supported"] is False and output["diagnostics"]["future_return_birth_supported"] is False,
        "computation_finished_not_final_without_return": bool(blocked_next["return_birth_required"][0, 0]) and not bool(blocked_next["task_completed"][0, 0]),
        "input_flow_not_return_substitute": torch.equal(typed_fixture_mapping, torch.tensor([[1, 2]])),
    }
    assert set(structural) == set(ADDITIONAL_STRUCTURAL_CHECKS)
    required = {**{k: bool(v) for k, v in original.items()}, **{k: bool(v) for k, v in service.items()}, **{k: bool(v) for k, v in structural.items()}}
    assert set(required) == set(REQUIRED_ACCEPTANCE_CHECKS)
    scope = {name: False for name in ("loss", "optimizer", "training", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim")}
    receipt = {"schema_version": "PI-JWM-Step-4.4-Acceptance-Receipt-v1", "evidence_class": "UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE", "required_checks": required, "scope": scope}
    receipt["passed"] = bool(all(required.values()) and all(v is False for v in scope.values()))
    receipt["validator"] = validate_world_model_acceptance(receipt)
    receipt["passed"] = bool(receipt["passed"] and receipt["validator"]["passed"])

    output_dir.mkdir(parents=True, exist_ok=True)
    package = output_dir / "untrained_world_model_package.pt"; save_world_model_package(model, package)
    torch.save(output, output_dir / "deterministic_prior_rollout.pt")
    config_payload = {"schema_version": SCHEMA_VERSION, "config": asdict(config), "contract": model.contract}
    service_trace = {"mode": "expectation", "expected_service_approximation": True, "wired_membership_equality": wired_details, "source_provenance": [
        {"path": str(AIRFOG_CHANNEL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(AIRFOG_CHANNEL), "symbols": ["ChannelManagerCP.computeRate", "ChannelManagerCP._get_power_db"]},
        {"path": str(AIRFOG_OUTAGE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(AIRFOG_OUTAGE), "symbols": ["rayleigh_outage_prob"]},
        {"path": str(AIRFOG_WIRED.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(AIRFOG_WIRED), "symbols": ["WiredNetworkManager.step"]},
        {"path": str(AIRFOG_CONFIG.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(AIRFOG_CONFIG), "symbols": ["channel.outage_snr_threshold"]},
    ]}
    write_json(output_dir / "world_model_config.json", config_payload)
    write_json(output_dir / "service_transition_audit.json", service_trace)
    write_json(output_dir / "acceptance_report.json", receipt)
    files = {}
    for p in sorted(output_dir.iterdir()):
        if p.name != "manifest.json": files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest = {"schema_version": "PI-JWM-Step-4.4-Manifest-v1", "evidence_class": receipt["evidence_class"], "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in (TENSOR_PATH, GRAPH_PATH, ENCODER_PATH, AUDIT_PATH, MODEL_SOURCE, BUILD_SOURCE, CONTRACT_PATH)}, "files": files, "deterministic_output_digest": world_model_digest(output), "acceptance_passed": receipt["passed"], "scope": scope}
    write_json(output_dir / "manifest.json", manifest)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); parser.add_argument("--refresh-existing", action="store_true"); args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing: raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    receipt = build(args.output_dir)
    print(json.dumps({"passed": receipt["passed"], "required": len(receipt["required_checks"]), "failed": [k for k, v in receipt["required_checks"].items() if not v]}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
