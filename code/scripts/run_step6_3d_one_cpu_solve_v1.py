"""One frozen-checkpoint H4 solve on the selected execution device."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

# The reused CPU preflight module defaults to hiding CUDA at import time.
# Select the visible-device policy here; a CPU solve still uses device="cpu".
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
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
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_1_trained_candidate_rollout_v1 import (
    prepare_anchor, rollout_one_step, rollout_one_step_batch, fingerprint,
)
from pi_jwm.step6_2a_planner_objective_side_state_v1 import PlannerRouteCausalSideState, prepare_objective_side_state
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, score_candidate_set,
)
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from run_step6_1_trained_candidate_rollout_preflight_v1 import (
    CHECKPOINT, DATASET, EXPECTED_SHA, SOURCE_SHA, sha,
)

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"
CATALOG = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
SOURCE_FILES = (
    "code/src/pi_jwm/step6_3b_candidate_grammar_v1.py",
    "code/src/pi_jwm/step6_3b_candidate_support_v1.py",
    "code/src/pi_jwm/step6_3c_candidate_domain_v1.py",
    "code/src/pi_jwm/step6_3c_search_protocol_v1.py",
    "code/src/pi_jwm/step6_3d_structured_proposal_v1.py",
    "code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py",
    "code/src/pi_jwm/step6_3d_anchor_selection_v1.py",
    "code/src/pi_jwm/step6_3d_method_selection_v1.py",
    "code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py",
    "code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py",
    "code/scripts/run_step6_3d_one_cpu_solve_v1.py",
    "code/scripts/run_step6_3d_formal_cpu_matrix_v1.py",
    "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/15_train_anchor_manifest_objective_eligible.json",
    "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/16_validation_anchor_manifest_objective_eligible.json",
    "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/19_formal_selected_deadline_sidecars.json",
    "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json",
)


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def execution_identity(*, device: str, gpu_model: str | None, batch_size: int,
                       checkpoint_sha256: str, source_sha256: dict[str, str],
                       bucket_strategy: str = "none") -> dict:
    if device not in ("cpu", "cuda") or batch_size < 1 or bucket_strategy != "none":
        raise ValueError("unsupported formal execution configuration")
    if (device == "cuda") != (gpu_model is not None):
        raise ValueError("GPU model must match execution device")
    fields = {"execution_device": device, "gpu_model": gpu_model,
              "precision": "FP32", "batch_size": batch_size,
              "state_storage": "cpu_cache_and_prefix" if device == "cuda" else "cpu_native",
              "checkpoint_sha256": checkpoint_sha256,
              "source_sha256": source_sha256,
              "bucket_strategy": bucket_strategy, "locked_test": False}
    fields["execution_config_id"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return fields


def load_selected(sample_id: str, interface: FormalTrainingInterface,
                  shards: FullFormalShardDataset):
    found = [(i, row) for i, row in enumerate(interface.samples)
             if row["metadata"]["sample_id"] == sample_id]
    if len(found) != 1:
        raise ValueError("selected anchor identity not unique")
    _, indexed = found[0]
    meta = indexed["metadata"]
    tid = meta["trajectory_id"]
    shard = shards.shards[tid]
    for family in ("samples", "tensor", "graph"):
        entry = shard["files"][family]
        if sha(shards.paths[family] / entry["path"]) != entry["sha256"]:
            raise ValueError("frozen shard identity mismatch")
    with gzip.open(shards.paths["samples"] / f"{tid}.json.gz", "rt", encoding="utf-8") as stream:
        sample = json.load(stream)[indexed["shard_index"]]
    if sample["metadata"]["sample_id"] != sample_id:
        raise ValueError("sample identity mismatch")
    tensor = _one(load_flow_tensor_batch(shards.paths["tensor"] / f"{tid}.npz"),
                  indexed["shard_index"], shard["sample_count"])
    graph_input = _one(load_typed_dual_graph_batch(shards.paths["graph"] / f"{tid}.npz"),
                       indexed["shard_index"], shard["sample_count"])
    state_tensor = dict(tensor)
    state_tensor["sample_metadata"] = [{**tensor["sample_metadata"][0],
        "source_path": str((ROOT / meta["source_path"]).resolve())}]
    state, graph = build_state(state_tensor, graph_input, 0,
                               wired_edges=FORMAL_V1_WIRED_EDGES)
    raw_path = ROOT / meta["source_path"]
    if sha(raw_path) != meta["source_sha256"]:
        raise ValueError("Formal Raw hash mismatch")
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    decision = raw["decisions"][int(meta["anchor_decision_frame"])]
    if decision["trajectory_id"] != tid or decision["frame_index"] != meta["anchor_decision_frame"]:
        raise ValueError("current decision identity mismatch")
    context = PlannerCandidateContext.from_sample(sample, state, sample_id)
    domain = context_from_current_raw(context, decision)
    return sample, tensor, graph_input, state, graph, context, domain, decision, meta


def objective_side(sample, state, context, decision, sidecar):
    carrying = {row["flow_id"]: row for row in sample["history"][-1]["carrying_states"]
                if row.get("known") and row.get("active")}
    routes = []
    for row in sample["history"][-1]["logical_flows"]:
        if row.get("known") and row.get("presence") and row["flow_id"] in carrying:
            current = carrying[row["flow_id"]]
            routes.append(PlannerRouteCausalSideState(
                row["flow_id"], int(row["flow_index"]), row["task_id"], row["flow_type"],
                int(row["epoch"]), int(row["logical_destination_index"]),
                tuple(map(int, current["route_node_indices"])), int(current["current_hop_index"]),
                int(current["route_revision"]), float(decision["simulation_time_s"]),
                int(current["current_holder_index"])))
    return prepare_objective_side_state(decision=decision, deadline_sidecar=sidecar,
        task_slots=context.static["input_entity_index"]["task"],
        physical_slots=context.static["input_entity_index"]["physical"],
        routes=routes, model_state=state)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def offload_transition_result(result):
    """Keep exact transition values in host RAM while preserving B_WM cache keys."""
    cpu = torch.device("cpu")
    return replace(result,
        latent=_to_device(result.latent, cpu),
        state=_to_device(result.state, cpu),
        graph=_to_device(result.graph, cpu),
        mobility_control=_to_device(result.mobility_control, cpu),
        action_tensor=_to_device(result.action_tensor, cpu),
        action_mapping=_to_device(result.action_mapping, cpu),
        model_trace=_to_device(result.model_trace, cpu))


def offload_prepared_anchor(prepared):
    cpu = torch.device("cpu")
    return replace(prepared, latent=_to_device(prepared.latent, cpu),
        state=_to_device(prepared.state, cpu),
        graph=_to_device(prepared.graph, cpu),
        z_pi=_to_device(prepared.z_pi, cpu))


def _gpu_search_node(node):
    cuda = torch.device("cuda")
    return replace(node, latent=_to_device(node.latent, cuda),
        state=_to_device(node.state, cuda), graph=_to_device(node.graph, cuda))


def gpu_transition_batch_with_cpu_storage(model, context, domain, requests):
    gpu_requests = [(_gpu_search_node(node), bound) for node, bound in requests]
    return tuple(offload_transition_result(result) for result in
                 rollout_one_step_batch(model, context, domain, gpu_requests))


def gpu_transition_one_with_cpu_storage(model, context, domain, node, bound):
    gpu_node = _gpu_search_node(node)
    return offload_transition_result(rollout_one_step(
        model, context, domain, gpu_node.latent, gpu_node.state,
        gpu_node.graph, gpu_node.mobility_control, bound))


def load_frozen_runtime(sample_id: str, *, device: str = "cpu"):
    """Prepare one authenticated anchor on one requested device."""
    torch.set_num_threads(1)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable for GPU rollout")
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("frozen best checkpoint SHA mismatch")
    selected = json.loads((OUT / "19_formal_selected_deadline_sidecars.json").read_text(encoding="utf-8"))
    if sample_id not in selected or not selected[sample_id]["alignment_passed"]:
        raise ValueError("selected anchor lacks aligned deadline sidecar")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    sample, tensor, graph_input, state, graph, context, domain, decision, meta = load_selected(
        sample_id, interface, shards)
    side = objective_side(sample, state, context, decision, selected[sample_id])
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    if catalog.dataset_manifest_sha256 != interface.dataset_manifest_hash:
        raise ValueError("TRAIN catalog and Formal Dataset differ")
    protocol = formal_training_config_v1(DATASET)
    stats = shards.encoder_normalization_stats()
    identity = {**shards.identity, "encoder_normalization_sha256": hashlib.sha256(
        json.dumps(stats, sort_keys=True).encode()).hexdigest()}
    with torch.no_grad():
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload["data_identity"] != identity or payload["git_commit"] != SOURCE_SHA or \
            payload["config"] != _jsonable(asdict(protocol.training)):
        raise ValueError("checkpoint data/source/config identity mismatch")
    encoder = PIJointGraphEncoder(protocol.training.encoder, tensor["contract"],
                                  graph_input["contract"], stats)
    model = StructuredRSSMWorldModel(protocol.training.rssm)
    encoder.load_state_dict(payload["model_state"]["encoder"], strict=True)
    model.load_state_dict(payload["model_state"]["rssm"], strict=True)
    encoder.to(target_device).eval(); model.to(target_device).eval()
    before = fingerprint({"encoder": encoder.state_dict(), "rssm": model.state_dict()})
    with torch.no_grad():
        prepared = prepare_anchor(model, encoder,
            _to_device(_torch_tree(tensor), target_device),
            _to_device(_torch_tree(graph_input), target_device),
            _to_device(state, target_device), _to_device(graph, target_device),
            context, domain, sample_id)
    return model, encoder, prepared, side, catalog, protocol, meta, before


def run(sample_id: str, method: str, seed: int, budget: int,
        iterations: int, elite_ratio: float | None, batch_size: int = 1,
        device: str = "cpu") -> dict:
    model, encoder, prepared, side, catalog, protocol, meta, before = load_frozen_runtime(
        sample_id, device=device)
    gpu_model = torch.cuda.get_device_name(0) if device == "cuda" else None
    identity = execution_identity(device=device, gpu_model=gpu_model,
        batch_size=batch_size, checkpoint_sha256=EXPECTED_SHA,
        source_sha256=source_hashes())
    if device == "cuda":
        prepared = offload_prepared_anchor(prepared)
    state, context, domain = prepared.state, prepared.context, prepared.domain
    score_residuals = {}
    support_horizons = {}
    def score_h4(candidate, trace):
        try:
            scored = score_candidate_set(state, side, ((candidate, trace),),
                slot_duration_s=protocol.training.rssm.slot_duration_s)
        except (ScorerStateInconsistency, BurdenSemanticsBlocked) as exc:
            reason = type(exc).__name__ + ":" + str(exc)
            score_residuals[reason] = score_residuals.get(reason, 0) + 1
            support_horizons["SCORER_EXCEPTION"] = support_horizons.get("SCORER_EXCEPTION", 0) + 1
            return None
        for horizon in scored.support_horizons.values():
            key = str(horizon)
            support_horizons[key] = support_horizons.get(key, 0) + 1
        if scored.status != "SCOREABLE" or scored.H_eff != 4 or not scored.scores or scored.scores[0].H_sup != 4:
            reason = "H4_UNSCOREABLE:" + scored.status
            score_residuals[reason] = score_residuals.get(reason, 0) + 1
            if scored.diagnostics.get("support_boundary_reasons"):
                for boundary in scored.diagnostics["support_boundary_reasons"].values():
                    if boundary is not None:
                        key = "H4_SUPPORT_BOUNDARY:" + str(boundary)
                        score_residuals[key] = score_residuals.get(key, 0) + 1
            return None
        return scored.scores[0]
    with torch.inference_mode():
        outcome = solve_fixed_budget(method=method, seed=seed, b_wm=budget,
            anchor=prepared, catalog=catalog,
            transition=(lambda node, bound: gpu_transition_one_with_cpu_storage(
                model, context, domain, node, bound)) if device == "cuda" else
                (lambda node, bound: rollout_one_step(model, context, domain,
                    node.latent, node.state, node.graph, node.mobility_control, bound)),
            score_h4=score_h4,
            transition_batch=(lambda requests: gpu_transition_batch_with_cpu_storage(
                model, context, domain, requests)) if device == "cuda" and batch_size > 1 else
                (lambda requests: rollout_one_step_batch(
                    model, context, domain, requests)) if batch_size > 1 else None,
            batch_size=batch_size,
            iterations=iterations, elite_ratio=elite_ratio)
    after = fingerprint({"encoder": encoder.state_dict(), "rssm": model.state_dict()})
    if before != after:
        raise AssertionError("frozen World Model parameters changed")
    return {"sample_id": sample_id, "split": meta["split"], "method": method,
            "seed": seed, "budget": budget, "iterations": iterations,
            "elite_ratio": elite_ratio, "outcome": asdict(outcome),
            "score_residuals": score_residuals,
            "support_horizon_counts": support_horizons,
            **identity, "parameter_digest": before,
            "sidecar_alignment_passed": True, "gpu": device == "cuda", "training": False,
            "closed_loop": False, "locked_test": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--method", choices=("HRS", "S-CEM", "MH-CEM"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--elite-ratio", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.sample_id, args.method, args.seed, args.budget,
                 args.iterations, args.elite_ratio, args.batch_size, args.device)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("sample_id", "method", "seed", "budget", "outcome")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
