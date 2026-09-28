"""CPU-only Step 6.1 validation-anchor candidate rollout preflight."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import psutil

from build_step5_1d_unified_model_chain_v1 import build_action, build_state
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import PIJointGraphEncoder
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel
from pi_jwm.step5_2_training_loop_v1 import _jsonable, _torch_tree
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES, _one
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1
from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, PlannerCandidateContext,
    compile_candidate_step,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import (
    PROFILES, annotate_domain, context_from_current_raw, rule_fallback_v1,
)
from pi_jwm.step6_1_trained_candidate_rollout_v1 import (
    action_response_summary, audit_recursive_link, fingerprint, max_delta, prepare_anchor,
    rollout_candidate_batch, rollout_candidate_sequential,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CHECKPOINT = ROOT / "code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt"
OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928"
EXPECTED_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"
SOURCE_SHA = "6e15ec2da0e3a6e0561dc821d0aaef90696a2387"
ACTION_KEYS = {"Route": ("route_task_index", "route_flow_index", "route_values"),
               "Comm": ("comm_relation_index", "comm_values", "comm_allocation_mask"),
               "Comp": ("comp_agent_index", "comp_task_index", "comp_values"),
               "Mob": ("mobility_entity_index", "mobility_values")}
ROUTED_KEYS = {"Route": ("task", "flow"), "Comm": ("communication",),
               "Comp": ("agent", "task"), "Mob": ("physical",)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n", encoding="utf-8")


def finite(value: Any, *, expectation: bool = False) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite(child, expectation=expectation) for key, child in value.items()
                   if not (expectation and key == "outage_uniform_draw"))
    if isinstance(value, (list, tuple)):
        return all(finite(child, expectation=expectation) for child in value)
    return True


def load_current(interface: FormalTrainingInterface, shards: FullFormalShardDataset, index: int):
    row = interface.samples[index]
    metadata = row["metadata"]
    if metadata["split"] != "dev_validation" or index not in interface.validation_indices:
        raise ValueError("anchor must be Formal Validation")
    trajectory = metadata["trajectory_id"]
    shard = shards.shards[trajectory]
    for family in ("samples", "tensor", "graph"):
        entry = shard["files"][family]
        if sha(shards.paths[family] / entry["path"]) != entry["sha256"]:
            raise ValueError(f"current {family} shard hash mismatch")
    with gzip.open(shards.paths["samples"] / f"{trajectory}.json.gz", "rt", encoding="utf-8") as stream:
        sample = json.load(stream)[row["shard_index"]]
    if sample["metadata"]["sample_id"] != metadata["sample_id"]:
        raise ValueError("sample index identity mismatch")
    all_tensor = load_flow_tensor_batch(shards.paths["tensor"] / f"{trajectory}.npz")
    all_graph = load_typed_dual_graph_batch(shards.paths["graph"] / f"{trajectory}.npz")
    tensor = _one(all_tensor, row["shard_index"], shard["sample_count"])
    graph_input = _one(all_graph, row["shard_index"], shard["sample_count"])
    state_tensor = dict(tensor)
    state_tensor["sample_metadata"] = [{**tensor["sample_metadata"][0],
        "source_path": str((ROOT / metadata["source_path"]).resolve())}]
    state, graph = build_state(state_tensor, graph_input, 0, wired_edges=FORMAL_V1_WIRED_EDGES)
    raw_path = ROOT / metadata["source_path"]
    if sha(raw_path) != metadata["source_sha256"]:
        raise ValueError("Raw source identity mismatch")
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    matches = [decision for decision in raw["decisions"]
               if int(decision["frame_index"]) == int(metadata["anchor_decision_frame"])]
    if len(matches) != 1 or matches[0]["trajectory_id"] != trajectory:
        raise ValueError("current Raw decision identity mismatch")
    base = PlannerCandidateContext.from_sample(sample, state, metadata["sample_id"])
    domain = context_from_current_raw(base, matches[0])
    return sample, tensor, graph_input, state, graph, base, domain, raw_path, matches[0]


def candidate(base: PlannerCandidateContext, domain: Any, control: CandidateActionSequence,
              family: str, row: dict[str, Any], source: str) -> CandidateActionSequence:
    first = replace(control.steps[0], **{family.lower(): (row,) if family != "Mob" else tuple(
        row if int(old["uav_index"]) == int(row["uav_index"]) else old for old in control.steps[0].mob)})
    tail = control.steps[1:]
    if family == "Mob":
        # HOLD at H2-H4 uses the command-updated control heading, not H0 heading.
        hold = {**row, "speed_mps": 0.0}
        tail = tuple(replace(step, mob=tuple(hold if int(old["uav_index"]) == int(row["uav_index"])
                    else old for old in step.mob)) for step in tail)
    sequence = CandidateActionSequence((first, *tail), f"{family.lower()}_h1", Backend.RULE_FALLBACK,
        6101, base.causal_provenance, frozenset({source}), generation_metadata={"probe_source": source})
    return annotate_domain(sequence, base, domain)


def make_probes(sample: dict[str, Any], state: dict[str, torch.Tensor],
                base: PlannerCandidateContext, domain: Any):
    control = rule_fallback_v1(base, domain, 4, 6101)
    probes: dict[str, tuple[CandidateActionSequence, str, str]] = {}
    physical = {int(slot): node for node, slot in base.static["input_entity_index"]["physical"].items()}
    tasks = {int(slot): task for task, slot in base.static["input_entity_index"]["task"].items()}
    # Current typed Flow and relation state are the only Route/Comm causal sources.
    for flow in range(state["flow_presence"].shape[1]):
        if not bool(state["flow_presence"][0, flow] & state["carrying_active"][0, flow]):
            continue
        task = int(state["flow_task_index"][0, flow]); src = int(state["carrying_hop_source_index"][0, flow])
        old_dst = int(state["carrying_hop_destination_index"][0, flow])
        for relation in range(state["comm_presence"].shape[1]):
            dst = int(state["comm_target_index"][0, relation])
            if (bool(state["comm_presence"][0, relation] & state["comm_validity"][0, relation])
                    and int(state["comm_source_index"][0, relation]) == src and dst != old_dst
                    and dst in physical and src in physical and task in tasks):
                row = {"task_id": tasks[task], "task_index": task, "task_node_index": src,
                       "target_node_index": dst, "route_node_indices": [dst], "route_kind": "offload"}
                try:
                    probes["Route"] = (candidate(base, domain, control, "Route", row, "CAUSAL_DOMAIN_PROBE"),
                                       "CAUSAL_DOMAIN_PROBE", f"active Flow {flow}; current relation {relation}")
                    break
                except ValueError:
                    pass
        if "Route" in probes:
            break
    for flow in range(state["flow_presence"].shape[1]):
        if not bool(state["flow_presence"][0, flow] & state["carrying_active"][0, flow]):
            continue
        relation = int(state["flow_comm_relation_index"][0, flow]); task = int(state["flow_task_index"][0, flow])
        if relation < 0 or task not in tasks or not bool(state["comm_wireless_mask"][0, relation]):
            continue
        free = torch.nonzero(~state["rb_active_mask"][0, relation], as_tuple=False)
        if not len(free):
            continue
        row = {"task_id": tasks[task], "task_index": task, "relation_index": relation,
               "rb_indices": [int(free[0, 0])]}
        try:
            probes["Comm"] = (candidate(base, domain, control, "Comm", row, "CAUSAL_DOMAIN_PROBE"),
                              "CAUSAL_DOMAIN_PROBE", f"active Flow {flow}; free current RB")
            break
        except ValueError:
            pass
    for relation in range(state["task_agent_validity"].shape[1]):
        if not bool(state["task_agent_validity"][0, relation]):
            continue
        task = int(state["task_agent_task_index"][0, relation])
        agent = int(state["task_agent_agent_index"][0, relation])
        if task not in tasks or agent not in physical or float(state["task_work_remaining"][0, task]) <= 0:
            continue
        budget = domain.compute_budgets[physical[agent]]
        if not budget.observed_mask or budget.static_capacity_per_s <= 0:
            continue
        amount = min(budget.static_capacity_per_s * 0.5,
                     float(state["task_work_remaining"][0, task]) / 0.2)
        row = {"task_id": tasks[task], "node_id": physical[agent], "allocated_cpu_per_s": amount}
        try:
            probes["Comp"] = (candidate(base, domain, control, "Comp", row, "CAUSAL_DOMAIN_PROBE"),
                              "CAUSAL_DOMAIN_PROBE", f"current Task-Agent relation {relation}; static CPU budget")
            break
        except ValueError:
            pass
    for slot, control_state in domain.mobility_states.items():
        if control_state.heading_observed and control_state.elevation_observed:
            _, delta, speed = PROFILES[1]
            row = {"uav_index": slot, "azimuth_rad": control_state.heading_rad + delta,
                   "elevation_rad": control_state.elevation_rad, "speed_mps": speed}
            probes["Mob"] = (candidate(base, domain, control, "Mob", row, "CAUSAL_DOMAIN_PROBE"),
                             "CAUSAL_DOMAIN_PROBE", "current observed UAV + frozen PROFILE_1")
            break
    # Recorded action is a conditioning diagnostic, never an online generator claim.
    for family, frame_key in (("Route", "route"), ("Comm", "comm")):
        if family in probes:
            continue
        entries = sample["future_action"][0][frame_key]["entries"]
        if not entries:
            continue
        row = dict(entries[0])
        if family == "Comm":
            try:
                _, mapping = build_action(sample, state, 0)
                row["relation_index"] = int(mapping["comm"][0]["relation_index"])
            except (KeyError, ValueError, IndexError):
                continue
        try:
            probes[family] = (candidate(base, domain, control, family, row,
                                         "RECORDED_ACTION_CONDITIONING_PROBE"),
                              "RECORDED_ACTION_DIAGNOSTIC", "first future action only; no target/outcome")
        except ValueError:
            pass
    return control, probes


def tensor_changed(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], keys: tuple[str, ...]) -> bool:
    return any(not torch.equal(a[key], b[key]) for key in keys)


def tree_numeric_delta(left: Any, right: Any, *, key: str = "") -> float:
    """Compare all rollout tensors; expectation's unused RNG draw is NaN by contract."""
    if key == "outage_uniform_draw":
        if not (isinstance(left, torch.Tensor) and torch.equal(torch.isnan(left), torch.isnan(right))):
            return float("inf")
        return 0.0
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return float("inf")
        if left.is_floating_point():
            return max_delta(left, right) if finite(left) and finite(right) else float("inf")
        return 0.0 if torch.equal(left, right) else float("inf")
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return float("inf")
        return max((tree_numeric_delta(left[k], right[k], key=k) for k in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf")
        return max((tree_numeric_delta(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else float("inf")


def direct_response(family: str, control: Any, probe: Any, anchor: Any, model: Any) -> dict[str, Any]:
    action = probe.actions[0]; ca = control.actions[0]
    state = probe.states[0]; old = anchor.state
    changed = tensor_changed(ca, action, ACTION_KEYS[family])
    routed = {name: max_delta(control.model_traces[0]["routed"][name], probe.model_traces[0]["routed"][name])
              for name in ROUTED_KEYS[family]}
    result: dict[str, Any] = {"formal_action_changed": changed, "routed_delta": routed,
                              "routed_response": any(value > 0 for value in routed.values())}
    if family == "Comp":
        row = probe.actions[0]; ti = int(row["comp_task_index"][0, 0]); amount = float(row["comp_values"][0, 0, 0])
        expected = max(0.0, float(old["task_work_remaining"][0, ti]) - amount * model.config.slot_duration_s)
        actual = float(state["task_work_remaining"][0, ti])
        progress = 1.0 - actual / max(float(old["task_work_total"][0, ti]), 1e-9)
        result.update({"task_index": ti, "expected_remaining": expected, "actual_remaining": actual,
                       "work_rule_ok": math.isclose(actual, expected, abs_tol=1e-5),
                       "progress_rule_ok": math.isclose(float(state["task_progress"][0, ti]), progress, abs_tol=1e-5)})
    elif family == "Mob":
        idx = int(action["mobility_entity_index"][0, 0]); az, el, speed = action["mobility_values"][0, 0, :3]
        expected_tensor = torch.stack((speed * torch.cos(az) * torch.cos(el),
                                       speed * torch.sin(az) * torch.cos(el),
                                       speed * torch.sin(el))) * model.config.slot_duration_s
        expected = expected_tensor.tolist()
        actual = (state["position"][0, idx] - old["position"][0, idx]).tolist()
        expected_position = old["position"][0, idx] + expected_tensor
        result.update({"uav_index": idx, "expected_delta_m": expected, "actual_delta_m": actual,
                       "position_rule_ok": bool(torch.allclose(state["position"][0, idx], expected_position,
                                                                  rtol=0, atol=1e-6)),
                       "position_delta_fp32_cancellation_m": max(abs(a-b) for a,b in zip(actual, expected)),
                       "speed_rule_ok": math.isclose(float(state["speed"][0, idx]), float(speed), abs_tol=1e-6)})
    elif family == "Comm":
        ri = int(action["comm_relation_index"][0, 0])
        result.update({"relation_index": ri,
                       "rb_mask_matches_action": torch.equal(state["rb_active_mask"], action["comm_allocation_mask"]),
                       "wireless_service_trace_finite": all(finite(probe.model_traces[0]["rule"]["wireless"][key])
                            for key in ("nominal_rate_mbps", "outage_probability", "actual_rate_mbps")),
                       "expectation_draw_is_intentional_nan": bool(torch.isnan(probe.model_traces[0]["rule"]["wireless"]["outage_uniform_draw"]).all()),
                       "carrying_uses_relation": bool(((old["flow_comm_relation_index"] == ri) & old["carrying_active"] & old["flow_presence"]).any())})
        result["delivered_service_delta"] = max_delta(control.model_traces[0]["rule"]["delivered_bytes"],
                                                        probe.model_traces[0]["rule"]["delivered_bytes"])
    elif family == "Route":
        fi = int(action["route_flow_index"][0, 0]); ti = int(action["route_task_index"][0, 0])
        result.update({"flow_index": fi, "task_index": ti,
                       "fixed_flow_support_preserved": torch.equal(old["flow_known"], state["flow_known"])
                           and torch.equal(old["flow_identity_index"], state["flow_identity_index"]),
                       "route_mapping": probe.mappings[0]["route"],
                       "structural_response_required": fi >= 0})
        if fi >= 0:
            result["hop_source_matches_action"] = int(state["carrying_hop_source_index"][0, fi]) == int(action["route_values"][0, 0, 0])
            result["hop_destination_matches_action"] = int(state["carrying_hop_destination_index"][0, fi]) == int(action["route_values"][0, 0, 1])
            result["route_revision_changed"] = int(state["flow_route_revision"][0, fi]) > int(old["flow_route_revision"][0, fi])
            host = ((old["task_agent_task_index"] == ti) &
                    (old["task_agent_relation_type_index"] == 3) & old["task_agent_validity"])
            result["host_relation_present"] = bool(host.any())
            result["host_matches_action_if_present"] = (not bool(host.any()) or
                int(state["task_agent_agent_index"][0, torch.nonzero(host[0], as_tuple=False)[0, 0]]) == int(action["route_values"][0, 0, 1]))
    return result


def rule_response_ok(family: str, direct: dict[str, Any]) -> bool:
    if family == "Comp":
        return direct["work_rule_ok"] and direct["progress_rule_ok"]
    if family == "Mob":
        return direct["position_rule_ok"] and direct["speed_rule_ok"]
    if family == "Comm":
        return (direct["rb_mask_matches_action"] and direct["wireless_service_trace_finite"]
                and (not direct["carrying_uses_relation"] or direct["delivered_service_delta"] > 0))
    return (direct["fixed_flow_support_preserved"] and
            (not direct["structural_response_required"] or
             (direct["hop_source_matches_action"] and direct["hop_destination_matches_action"]
              and direct["route_revision_changed"] and direct["host_matches_action_if_present"])))


def main() -> None:
    torch.set_num_threads(1)
    if not CHECKPOINT.is_file() or sha(CHECKPOINT) != EXPECTED_SHA:
        write("checkpoint_identity_receipt.json", {"status": "BLOCKED", "reason": "best.pt missing or SHA mismatch"})
        raise RuntimeError("frozen best checkpoint missing or SHA mismatch")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    protocol = formal_training_config_v1(DATASET)
    shards = FullFormalShardDataset(interface)
    with torch.no_grad():
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    tracked_checkpoint = subprocess.run(["git", "ls-files", "--error-unmatch", CHECKPOINT.relative_to(ROOT).as_posix()],
        cwd=ROOT, capture_output=True, text=True).returncode == 0
    if tracked_checkpoint:
        raise ValueError("frozen checkpoint bytes must remain outside Git")
    stats = shards.encoder_normalization_stats()
    stats_sha = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    identity = {**shards.identity, "encoder_normalization_sha256": stats_sha}
    if payload["data_identity"] != identity or payload["git_commit"] != SOURCE_SHA or payload["config"] != _jsonable(asdict(protocol.training)):
        raise ValueError("checkpoint data/source/frozen configuration identity mismatch")
    write("checkpoint_identity_receipt.json", {"status": "PASS", "run_id": CHECKPOINT.parents[1].name,
        "best_checkpoint_sha256": EXPECTED_SHA, "training_source_git_sha": SOURCE_SHA,
        "dataset_manifest_sha256": interface.dataset_manifest_hash, "device": "cpu", "bytes_in_git": tracked_checkpoint,
        "loaded_components": ["encoder", "rssm"], "optimizer_step": False})
    # Deterministic, bounded search in the first Formal Validation trajectory.
    selected: dict[str, Any] = {}
    anchors: dict[int, Any] = {}
    for index in interface.validation_indices[:24]:
        loaded = load_current(interface, shards, index)
        sample, tensor, graph_input, state, graph, base, domain, raw_path, raw = loaded
        control, probes = make_probes(sample, state, base, domain)
        anchors[index] = (loaded, control, probes)
        for family in ("Route", "Comm", "Comp", "Mob"):
            if family in probes and family not in selected:
                selected[family] = index
        if len(selected) == 4:
            break
    if not selected:
        raise RuntimeError("no legal Formal Validation probe found")
    anchor_rows = []
    for index in sorted(set(selected.values())):
        loaded, _, probes = anchors[index]
        sample, _, _, _, _, _, _, raw_path, raw = loaded
        anchor_rows.append({"sample_index": index, "sample_id": sample["metadata"]["sample_id"],
            "split": "dev_validation", "raw_sha256": sha(raw_path), "raw_frame_index": raw["frame_index"],
            "raw_capture_event_id": raw["capture_event_id"], "families": [f for f, i in selected.items() if i == index]})
    write("anchor_probe_manifest.json", {"status": "READY", "selection": "first 24 Formal Validation windows, first legal probe per family",
        "anchors": anchor_rows, "family_probe_sources": {f: anchors[i][2][f][1] for f, i in selected.items()},
        "missing_families": [f for f in ("Route", "Comm", "Comp", "Mob") if f not in selected],
        "future_target_used_for_probe": False, "locked_test_accessed": False})
    # Build only the frozen modules used at deployment; no training target or optimizer is loaded.
    first = anchors[next(iter(selected.values()))][0]
    encoder = PIJointGraphEncoder(protocol.training.encoder, first[1]["contract"], first[2]["contract"], stats)
    model = StructuredRSSMWorldModel(protocol.training.rssm)
    encoder.load_state_dict(payload["model_state"]["encoder"], strict=True)
    model.load_state_dict(payload["model_state"]["rssm"], strict=True)
    encoder.eval(); model.eval()
    before = fingerprint({"encoder": encoder.state_dict(), "model": model.state_dict()})
    family_rows: dict[str, Any] = {}
    recursion_rows: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    with torch.no_grad():
        for index in sorted(set(selected.values())):
            loaded, control, probes = anchors[index]
            sample, tensor, graph_input, state, graph, base, domain, _, _ = loaded
            prepared = prepare_anchor(model, encoder, _torch_tree(tensor), _torch_tree(graph_input),
                                      state, graph, base, domain, sample["metadata"]["sample_id"])
            control_trace = rollout_candidate_sequential(model, prepared, control)
            for family in (f for f, i in selected.items() if i == index):
                intervention, source, provenance = probes[family]
                trace = rollout_candidate_sequential(model, prepared, intervention)
                traces[family] = (prepared, control, intervention, control_trace, trace)
                sensitivity = action_response_summary(control_trace, trace)
                direct = direct_response(family, control_trace, trace, prepared, model)
                target = ROUTED_KEYS[family]
                path_response = any(any(row["hidden_delta"][key] > 0 for key in target) for row in sensitivity)
                family_rows[family] = {"sample_id": prepared.sample_id, "probe_source": source,
                    "provenance": provenance, "deployment_candidate_evidence": source == "CAUSAL_DOMAIN_PROBE",
                    "planner_generator_evidence": source == "CAUSAL_DOMAIN_PROBE",
                    "recorded_action_probe": source == "RECORDED_ACTION_DIAGNOSTIC",
                    "anchor_fingerprints": prepared.fingerprints,
                    "same_current_belief": control_trace.initial_fingerprints == trace.initial_fingerprints,
                    "direct": direct, "sensitivity": sensitivity,
                    "target_family_latent_response": path_response,
                    "status": ("PASS" if direct["formal_action_changed"] and direct["routed_response"]
                               and path_response and rule_response_ok(family, direct) else
                               "ACTION_CONDITIONING_PATH_NO_RESPONSE" if not direct["routed_response"] or not path_response
                               else "DIRECT_RULE_RESPONSE_FAILURE")}
                pairs = [(control_trace, "CONTROL"), (trace, family)]
                for candidate_trace, label in pairs:
                    for h in range(1, 4):
                        audit_recursive_link(candidate_trace, h)
                    negative_rejected = False
                    try:
                        audit_recursive_link(candidate_trace, 1, substituted_input={
                            "state": prepared.fingerprints["state"], "graph": prepared.fingerprints["graph"],
                            "latent": prepared.fingerprints["latent"]})
                    except ValueError:
                        negative_rejected = True
                    recursion_rows[f"{family}:{label}"] = {"sample_id": prepared.sample_id,
                        "state_fingerprints": [prepared.fingerprints["state"], *[r["state"] for r in candidate_trace.output_fingerprints]],
                        "graph_fingerprints": [prepared.fingerprints["graph"], *[r["graph"] for r in candidate_trace.output_fingerprints]],
                        "latent_fingerprints": [prepared.fingerprints["latent"], *[r["latent"] for r in candidate_trace.output_fingerprints]],
                        "feedback_exact": all(candidate_trace.input_fingerprints[h] == candidate_trace.output_fingerprints[h-1]
                                              for h in range(1, 4)),
                        "h2_h0_substitution_rejected_by_fingerprint_audit": negative_rejected,
                        "finite_h1_h4": all(finite(value, expectation=True) for value in (*candidate_trace.states, *candidate_trace.graphs,
                                                                         *candidate_trace.latents, *candidate_trace.model_traces))}
        # Deterministic batch consistency on each real pair.
        batch_rows = {}
        for family, (prepared, control, intervention, serial_control, serial_probe) in traces.items():
            b_control, b_probe = rollout_candidate_batch(model, prepared, (control, intervention))
            pair_rows = []
            for serial, batched in ((serial_control, b_control), (serial_probe, b_probe)):
                pair_rows.append({name: tree_numeric_delta(getattr(serial, name), getattr(batched, name))
                                  for name in ("actions", "latents", "states", "graphs", "model_traces")})
            all_deltas = [delta for pair in pair_rows for delta in pair.values()]
            batch_rows[family] = {"component_max_abs_delta": pair_rows,
                                  "max_abs_delta": max(all_deltas),
                                  "within_1e_minus_4": max(all_deltas) <= 1e-4}
        deterministic_pass = (len(family_rows) == 4 and all(row["status"] == "PASS" and row["same_current_belief"] for row in family_rows.values())
            and all(row["feedback_exact"] and row["h2_h0_substitution_rejected_by_fingerprint_audit"] and row["finite_h1_h4"] for row in recursion_rows.values())
            and all(row["within_1e_minus_4"] for row in batch_rows.values()))
        stochastic: dict[str, Any] = {"status": "NOT_RUN_DETERMINISTIC_GATE"}
        if deterministic_pass:
            stochastic = {"status": "PASS", "seeds": [6101, 6102, 6103], "families": {},
                          "rng_stream_limitation": "single generator couples latent and wireless outage draws"}
            for family, (prepared, control, intervention, _, _) in traces.items():
                seed_rows = []
                for seed in stochastic["seeds"]:
                    def run(candidate: CandidateActionSequence):
                        generator = torch.Generator(device="cpu").manual_seed(seed)
                        return rollout_candidate_sequential(model, prepared, candidate, prior_mode="sample",
                            service_mode="sample", generator=generator)
                    c1, c2, p1, p2 = run(control), run(control), run(intervention), run(intervention)
                    repeat = c1.output_fingerprints == c2.output_fingerprints and p1.output_fingerprints == p2.output_fingerprints
                    valid = all(finite(item) for trace in (c1, p1) for item in (*trace.latents, *trace.states, *trace.model_traces))
                    same = c1.initial_fingerprints == p1.initial_fingerprints
                    seed_rows.append({"seed": seed, "same_candidate_same_seed_reproducible": repeat,
                                      "finite": valid, "same_current_belief": same,
                                      "paired_common_seed": True})
                stochastic["families"][family] = seed_rows
            stochastic["status"] = "PASS" if all(all(row["same_candidate_same_seed_reproducible"] and row["finite"] and row["same_current_belief"] for row in rows)
                                                   for rows in stochastic["families"].values()) else "FAIL"
        # A bounded implementation diagnostic, using the first real anchor.
        prepared, control, intervention, _, _ = next(iter(traces.values()))
        runtime = {"status": "PASS", "scope": "CPU_IMPLEMENTATION_DIAGNOSTIC", "horizon": 4,
                   "warmup": 1, "repeats": 3, "rows": [],
                   "memory_source": "psutil.Process.memory_info; peak_wset is process lifetime peak on Windows"}
        process = psutil.Process()
        for k in (1, 2, 4, 8):
            candidates = tuple((control, intervention)[j % 2] for j in range(k))
            rss_before = process.memory_info().rss
            for _ in range(1):
                for item in candidates: rollout_candidate_sequential(model, prepared, item)
                rollout_candidate_batch(model, prepared, candidates)
            serial_times, batch_times = [], []
            for _ in range(3):
                start = time.perf_counter()
                for item in candidates: rollout_candidate_sequential(model, prepared, item)
                serial_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                rollout_candidate_batch(model, prepared, candidates)
                batch_times.append(time.perf_counter() - start)
            runtime["rows"].append({"K": k, "serial_wall_median_s": statistics.median(serial_times),
                "serial_per_candidate_median_s": statistics.median(serial_times) / k,
                "batch_wall_median_s": statistics.median(batch_times),
                "batch_per_candidate_median_s": statistics.median(batch_times) / k,
                "rss_before_bytes": rss_before, "rss_after_bytes": process.memory_info().rss,
                "process_peak_wset_bytes": process.memory_info().peak_wset})
    after = fingerprint({"encoder": encoder.state_dict(), "model": model.state_dict()})
    write("action_family_response.json", {"families": family_rows, "serial_batch_consistency": batch_rows})
    write("recursive_feedback_audit.json", {"rows": recursion_rows,
        "h2_negative_test": "replace recorded H2 input fingerprint with H0; equality assertion rejects it"})
    write("stochastic_common_seed_diagnostic.json", stochastic)
    write("cpu_batch_runtime_diagnostic.json", runtime)
    status = "PASS" if deterministic_pass and stochastic["status"] == "PASS" and before == after else "PARTIAL"
    write("candidate_rollout_preflight_receipt.json", {"status": status,
        "trained_world_model_candidate_rollout": status, "current_posterior_mode": "mean",
        "future_prior_mode": "mean", "wireless_service_mode": "expectation", "horizons": [1, 2, 3, 4],
        "model_eval": not model.training and not encoder.training, "torch_no_grad": True,
        "parameter_digest_before": before, "parameter_digest_after": after,
        "parameter_unchanged": before == after, "future_target_consumed_by_rollout": False,
        "future_posterior_teacher_calls": 0, "future_target_encoder_calls": 0,
        "posterior_used_as_future_rollout_state": False, "locked_test_accessed": False,
        "baseline": False, "closed_loop": False, "planner_optimization": False,
        "performance_claim": False, "mpc_objective": "NOT_STARTED",
        "candidate_method_selection": "RESEARCH_PENDING",
        "family_status": {family: row["status"] for family, row in family_rows.items()},
        "serial_batch_consistency": all(row["within_1e_minus_4"] for row in batch_rows.values()),
        "stochastic_status": stochastic["status"]})
    print(json.dumps({"status": status, "family_sources": {f: r["probe_source"] for f, r in family_rows.items()},
                      "parameter_unchanged": before == after, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
