"""One bounded non-locked checkpoint mechanism probe for STEP 6.2B."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from build_step5_1d_unified_model_chain_v1 import build_state
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import PIJointGraphEncoder
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel
from pi_jwm.step5_2_training_loop_v1 import _jsonable, _torch_tree
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES, _one
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1
from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence, PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import annotate_domain, context_from_current_raw, rule_fallback_v1
from pi_jwm.step6_1_trained_candidate_rollout_v1 import fingerprint, prepare_anchor, rollout_candidate_sequential
from pi_jwm.step6_2a_route_rule_metadata_v1 import build_route_rule_metadata
from pi_jwm.step6_2a_planner_objective_side_state_v1 import PlannerRouteCausalSideState, prepare_objective_side_state
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CHECKPOINT = ROOT / "code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt"
SIDECAR = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928/deadline_sidecar_anchor_0001.json"
EXPECTED_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict:
    torch.set_num_threads(1)
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("frozen best.pt SHA mismatch")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    index = next(i for i in interface.validation_indices
                 if interface.samples[i]["metadata"]["sample_id"].endswith("anchor-0001"))
    indexed = interface.samples[index]
    meta = indexed["metadata"]
    trajectory = meta["trajectory_id"]
    shard = shards.shards[trajectory]
    for family in ("samples", "tensor", "graph"):
        entry = shard["files"][family]
        if sha(shards.paths[family] / entry["path"]) != entry["sha256"]:
            raise ValueError(f"{family} frozen shard hash mismatch")
    with gzip.open(shards.paths["samples"] / f"{trajectory}.json.gz", "rt", encoding="utf-8") as handle:
        sample = json.load(handle)[indexed["shard_index"]]
    if sample["metadata"]["sample_id"] != meta["sample_id"]:
        raise ValueError("sample identity mismatch")
    tensor = _one(load_flow_tensor_batch(shards.paths["tensor"] / f"{trajectory}.npz"),
                  indexed["shard_index"], shard["sample_count"])
    graph_input = _one(load_typed_dual_graph_batch(shards.paths["graph"] / f"{trajectory}.npz"),
                       indexed["shard_index"], shard["sample_count"])
    state_tensor = dict(tensor)
    state_tensor["sample_metadata"] = [{**tensor["sample_metadata"][0],
        "source_path": str((ROOT / meta["source_path"]).resolve())}]
    state, graph = build_state(state_tensor, graph_input, 0, wired_edges=FORMAL_V1_WIRED_EDGES)
    raw_path = ROOT / meta["source_path"]
    if sha(raw_path) != meta["source_sha256"]:
        raise ValueError("Raw source identity mismatch")
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    decision = next(row for row in raw["decisions"]
                    if int(row["frame_index"]) == int(meta["anchor_decision_frame"]))
    if decision["trajectory_id"] != trajectory:
        raise ValueError("Raw trajectory mismatch")
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    routes = []
    carrying = {row["flow_id"]: row for row in sample["history"][-1]["carrying_states"]
                if row.get("known") and row.get("active")}
    for row in sample["history"][-1]["logical_flows"]:
        if row.get("known") and row.get("presence") and row["flow_id"] in carrying:
            current = carrying[row["flow_id"]]
            routes.append(PlannerRouteCausalSideState(
                row["flow_id"], int(row["flow_index"]), row["task_id"], row["flow_type"],
                int(row["epoch"]), int(row["logical_destination_index"]),
                tuple(map(int, current["route_node_indices"])), int(current["current_hop_index"]),
                int(current["route_revision"]), float(decision["simulation_time_s"]),
                int(current["current_holder_index"])))
    side = prepare_objective_side_state(decision=decision, deadline_sidecar=sidecar,
        task_slots=sample["static"]["input_entity_index"]["task"],
        physical_slots=sample["static"]["input_entity_index"]["physical"],
        routes=routes, model_state=state)
    base = PlannerCandidateContext.from_sample(sample, state, meta["sample_id"])
    domain = context_from_current_raw(base, decision)
    control = rule_fallback_v1(base, domain, 4, 6202)
    physical_by_index = {int(slot): node for node, slot in base.static["input_entity_index"]["physical"].items()}
    task_by_index = {int(slot): task for task, slot in base.static["input_entity_index"]["task"].items()}
    # This is a bounded causal fixture, not a candidate-generation method.
    probe = None
    for relation in range(state["task_agent_validity"].shape[1]):
        if not bool(state["task_agent_validity"][0, relation]):
            continue
        ti = int(state["task_agent_task_index"][0, relation])
        agent = int(state["task_agent_agent_index"][0, relation])
        if ti not in task_by_index or agent not in physical_by_index:
            continue
        work = float(state["task_work_remaining"][0, ti])
        budget = domain.compute_budgets[physical_by_index[agent]]
        if work <= 0 or not budget.observed_mask or budget.static_capacity_per_s <= 0:
            continue
        amount = min(budget.static_capacity_per_s * 0.25, work / 0.4)
        comp = {"task_id": task_by_index[ti], "node_id": physical_by_index[agent],
                "allocated_cpu_per_s": amount}
        steps = (replace(control.steps[0], comp=(comp,)), *control.steps[1:])
        candidate = CandidateActionSequence(steps, "causal_comp_probe", Backend.RULE_FALLBACK,
            6202, base.causal_provenance, frozenset({"CURRENT_CAUSAL_FIXTURE"}),
            generation_metadata={"probe_only": True})
        try:
            probe = annotate_domain(candidate, base, domain)
            break
        except ValueError:
            continue
    if probe is None:
        raise ValueError("no legal current-time Comp probe")
    protocol = formal_training_config_v1(DATASET)
    with torch.no_grad():
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stats = shards.encoder_normalization_stats()
    encoder = PIJointGraphEncoder(protocol.training.encoder, tensor["contract"], graph_input["contract"], stats)
    model = StructuredRSSMWorldModel(protocol.training.rssm)
    encoder.load_state_dict(payload["model_state"]["encoder"], strict=True)
    model.load_state_dict(payload["model_state"]["rssm"], strict=True)
    encoder.eval(); model.eval()
    before = fingerprint({"encoder": encoder.state_dict(), "rssm": model.state_dict()})
    with torch.no_grad():
        anchor = prepare_anchor(model, encoder, _torch_tree(tensor), _torch_tree(graph_input),
                                state, graph, base, domain, meta["sample_id"])
        traces = tuple(rollout_candidate_sequential(model, anchor, candidate)
                       for candidate in (control, probe))
        scored = score_candidate_set(state, side, tuple(zip((control, probe), traces)),
            slot_duration_s=protocol.training.rssm.slot_duration_s)
        # Bounded effective-action audit: current existing Flow and one
        # pending task, both chosen solely from the anchor's causal support.
        audit_rows = []
        for kind in ("existing_flow", "pending_no_current_flow"):
            target_task = None
            for task in side.tasks:
                hits = [f for f in range(state["flow_presence"].shape[1])
                        if bool(state["flow_presence"][0, f]) and
                        int(state["flow_task_index"][0, f]) == task.task_index]
                if (kind == "existing_flow") == bool(hits):
                    target_task = (task, hits[0] if hits else -1)
                    break
            if target_task is None:
                continue
            task, flow = target_task
            if flow >= 0:
                source = int(state["current_holder_index"][0, flow])
                destination = int(state["flow_destination_index"][0, flow])
                flow_id = next(route.flow_id for route in side.routes if route.flow_index == flow)
            else:
                source = int(state["task_agent_agent_index"][0,
                    next(r for r in range(state["task_agent_validity"].shape[1])
                         if bool(state["task_agent_validity"][0, r]) and
                         int(state["task_agent_task_index"][0, r]) == task.task_index)])
                destination = next(index for index in physical_by_index if index != source)
                flow_id = None
            route_row = {"task_id": task.task_id, "task_index": task.task_index,
                         "task_node_index": source, "target_node_index": destination,
                         "route_node_indices": [destination], "route_kind": "offload"}
            if flow_id is not None:
                route_row["flow_id"] = flow_id
            route_sequence = CandidateActionSequence(
                (replace(control.steps[0], route=(route_row,)), *control.steps[1:]),
                f"route_effective_{kind}", Backend.RULE_FALLBACK, 6202,
                base.causal_provenance, frozenset({"CURRENT_CAUSAL_FIXTURE"}),
                generation_metadata={"probe_only": True})
            route_sequence = annotate_domain(route_sequence, base, domain)
            route_trace = rollout_candidate_sequential(model, anchor, route_sequence)
            route_mapping = route_trace.mappings[0]["route"][0]
            metadata = build_route_rule_metadata(
                {"future_action": [route_sequence.steps[0].frame()]}, state,
                route_trace.actions[0], 0)
            learned = traces[0].model_traces[0]["learned"]
            baseline_rule, _ = model.deterministic_transition(state, traces[0].actions[0],
                learned, graph=graph, service_mode="expectation", generator=None)
            route_rule, _ = model.deterministic_transition(state, route_trace.actions[0],
                learned, graph=graph, service_mode="expectation", generator=None,
                route_rule_metadata=metadata)
            rule_keys = ("flow_presence", "flow_remaining", "flow_route_revision",
                         "route_node_indices", "current_holder_index", "current_hop_index",
                         "carrying_hop_destination_index", "task_agent_agent_index")
            changed_rule = [key for key in rule_keys if not torch.equal(baseline_rule[key], route_rule[key])]
            changed_predicted = [key for key in rule_keys if not torch.equal(
                traces[0].states[0][key], route_trace.states[0][key])]
            audit_rows.append({"kind": kind, "candidate_id": route_sequence.candidate_id,
                               "action_mapping_mode": route_mapping["mode"],
                               "route_flow_index": route_mapping["flow_index"],
                               "deterministic_rule_changed_keys": changed_rule,
                               "predicted_state_changed_keys": changed_predicted,
                               "learned_latent_changed": fingerprint(traces[0].latents[0]) !=
                                                          fingerprint(route_trace.latents[0]),
                               "fixed_support_flow_created": bool((route_rule["flow_known"] &
                                   ~state["flow_known"]).any())})
    after = fingerprint({"encoder": encoder.state_dict(), "rssm": model.state_dict()})
    if before != after or sha(CHECKPOINT) != EXPECTED_SHA:
        raise AssertionError("frozen checkpoint or parameters changed")
    if scored.status != "SCOREABLE":
        raise ValueError(f"real score not finite: {scored.status}")
    if any(not all(torch.isfinite(value).all() for value in trace.states[h].values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point())
           for trace in traces for h in range(4)):
        raise ValueError("nonfinite predicted state")
    return {"status": "PASS", "sample_id": meta["sample_id"], "split": "dev_validation",
            "sample_index": index, "checkpoint_sha256": EXPECTED_SHA,
            "strict_load": True, "parameter_digest_unchanged": before == after,
            "prior_mode": "mean", "service_mode": "expectation", "horizon": 4,
            "H_eff": scored.H_eff, "candidate_count": len(scored.scores),
            "candidate_scores": [{"candidate_id": x.candidate_id,
                                  "candidate_fingerprint": x.candidate_fingerprint,
                                  "anchor_fingerprint": x.anchor_fingerprint,
                                  "H_sup": x.H_sup, "H_eff": x.H_eff,
                                  "N_DDL": x.N_DDL, "A_DDL": x.A_DDL,
                                  "J_Delay": x.J_Delay, "J_Burden": x.J_Burden,
                                  "J_Effort": x.J_Effort,
                                  "objective_tuple": x.objective_tuple,
                                  "support_boundary_reason": x.support_boundary_reason,
                                  "per_horizon_rows": [asdict(row) for row in x.per_horizon_rows],
                                  "per_task_rows": x.per_task_rows, "diagnostics": x.diagnostics}
                                 for x in scored.scores],
            "effective_route_action_audit": audit_rows,
            "future_target_used": False, "candidate_method_claim": False,
            "performance_claim": False, "gpu": False, "locked_test": False,
            "training": False, "optimizer_step": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, allow_nan=False))
