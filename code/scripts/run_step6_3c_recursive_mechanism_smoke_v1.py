"""Frozen-checkpoint, one-anchor CPU interleaved mechanism smoke.

The anchor is fixed by the accepted STEP 6.1 receipt. Lexicographically first
grammar action is a deterministic fixture, not candidate ranking or policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/scripts"))

import torch

from pi_jwm.step4_3b_dual_graph_encoder_v1 import PIJointGraphEncoder
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel
from pi_jwm.step5_2_training_loop_v1 import _jsonable, _torch_tree
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1
from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from pi_jwm.step6_1_trained_candidate_rollout_v1 import (
    fingerprint, prepare_anchor, rollout_candidate_sequential, rollout_one_step,
)
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import (
    SearchNode, TransitionBudgetAccountant, action_step_fingerprint,
)
from run_step6_1_trained_candidate_rollout_preflight_v1 import (
    CHECKPOINT, DATASET, EXPECTED_SHA, SOURCE_SHA, load_current, sha,
    tree_numeric_delta,
)

CATALOG = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
ANCHOR_RECEIPT = ROOT / "code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/anchor_probe_manifest.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"


def write(name: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes((json.dumps(value, sort_keys=True, indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def main() -> None:
    prior = json.loads(ANCHOR_RECEIPT.read_text(encoding="utf-8"))
    anchor_index = int(prior["anchors"][0]["sample_index"])
    assert anchor_index == 4416 and prior["anchors"][0]["split"] == "dev_validation"
    if not CHECKPOINT.is_file() or sha(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("frozen best checkpoint missing or SHA mismatch")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    if catalog.dataset_manifest_sha256 != interface.dataset_manifest_hash:
        raise ValueError("TRAIN catalog dataset identity mismatch")
    protocol = formal_training_config_v1(DATASET)
    shards = FullFormalShardDataset(interface)
    sample, tensor, graph_input, state, graph, context, domain, _, _ = load_current(
        interface, shards, anchor_index)
    stats = shards.encoder_normalization_stats()
    identity = {**shards.identity,
                "encoder_normalization_sha256": hashlib.sha256(
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
    encoder.eval(); model.eval()
    digest_before = fingerprint({"encoder": encoder.state_dict(), "model": model.state_dict()})
    budget = TransitionBudgetAccountant(4)
    with torch.no_grad():
        prepared = prepare_anchor(model, encoder, _torch_tree(tensor),
                                  _torch_tree(graph_input), state, graph, context,
                                  domain, sample["metadata"]["sample_id"])
        node = SearchNode.from_anchor(prepared.latent, prepared.state, prepared.graph,
                                      domain.mobility_states, context.causal_provenance)
        transitions = []
        step_results = []
        dead_end = None
        for target_depth in range(1, 5):
            candidate_domain = CandidateDomain.from_state(context, domain, node.state,
                node.mobility_control, catalog, node.structural_signature_prefix)
            if candidate_domain.is_empty:
                dead_end = {"first_unreachable_horizon": target_depth,
                            "reason": "NO_FORMAL_CANDIDATE" if target_depth == 1 else "GRAMMAR_DEAD_END",
                            "domain_reason": candidate_domain.empty_reason}
                budget.dead_end()
                break
            bound = next(candidate_domain.iter_bound())
            before_fp = node.fingerprints
            budget.proposed(True)
            result, hit = budget.evaluate(node, bound, lambda: rollout_one_step(
                model, context, domain, node.latent, node.state, node.graph,
                node.mobility_control, bound))
            node = node.advance(bound, result, cache_hit=hit)
            step_results.append(result)
            budget.complete()  # the prefix is separately feasible at this horizon
            transitions.append({"horizon": target_depth,
                                "domain_structural_mode_count": len(candidate_domain.modes),
                                "domain_exact_concrete_count": candidate_domain.exact_unique_single_step_count,
                                "choice_rule": "first canonical grammar iteration; mechanism fixture only",
                                "structural_signature": list(bound.structural_signature),
                                "action_fingerprint": action_step_fingerprint(bound.action),
                                "input_fingerprints": before_fp,
                                "output_fingerprints": node.fingerprints,
                                "support_label": {
                                    "joint": bound.support.joint_structural,
                                    "temporal_prefix": bound.support.temporal_prefix,
                                    "temporal_adjacent": bound.support.temporal_adjacent}})
        equivalence = {"executed": False, "reason": "NO_FORMAL_CANDIDATE_AT_H1"}
        if node.depth:
            sequence = CandidateActionSequence(node.action_prefix, "canonical_mechanism_fixture",
                                               Backend.RULE_FALLBACK, None,
                                               context.causal_provenance)
            reference = rollout_candidate_sequential(model, prepared, sequence)
            component_deltas = {name: tree_numeric_delta(getattr(reference, name),
                                  tuple(getattr(item, attribute) for item in step_results))
                                for name, attribute in (("latents", "latent"),
                                                        ("states", "state"),
                                                        ("graphs", "graph"),
                                                        ("actions", "action_tensor"),
                                                        ("model_traces", "model_trace"))}
            # Fingerprint equality is stronger than the existing numeric
            # tolerance and avoids a second model-specific comparison rule.
            output_equal = all(reference.output_fingerprints[i] == transitions[i]["output_fingerprints"]
                               for i in range(node.depth))
            input_equal = all(reference.input_fingerprints[i] == transitions[i]["input_fingerprints"]
                              for i in range(node.depth))
            action_equal = all(fingerprint(reference.actions[i]) == fingerprint(step_results[i].action_tensor)
                               for i in range(node.depth))
            equivalence = {"executed": True,
                           "equivalent_prebound_horizon": node.depth,
                           "input_fingerprints_exact": input_equal,
                           "output_fingerprints_exact": output_equal,
                           "formal_action_fingerprints_exact": action_equal,
                           "component_max_abs_delta": component_deltas,
                           "within_existing_1e_minus_4_tolerance": all(v <= 1e-4 for v in component_deltas.values()),
                           "pass": input_equal and output_equal and action_equal and
                                   all(v <= 1e-4 for v in component_deltas.values())}
    digest_after = fingerprint({"encoder": encoder.state_dict(), "model": model.state_dict()})
    horizon_status = {f"H{h}": ("COMPLETE" if h <= node.depth else
                      dead_end["reason"] if dead_end else "UNREACHED") for h in range(1, 5)}
    write("03_recursive_mechanism_smoke.json", {
        "anchor_sample_index": anchor_index, "anchor_sample_id": prepared.sample_id,
        "anchor_selected_by": "accepted STEP 6.1 anchor_probe_manifest; no reselection",
        "checkpoint_sha256": EXPECTED_SHA,
        "parameter_digest_before": digest_before,
        "parameter_digest_after": digest_after,
        "parameter_unchanged": digest_before == digest_after,
        "prior_mode": "mean", "service_mode": "expectation",
        "model_eval": True, "torch_no_grad": True,
        "transitions": transitions, "horizon_feasibility": horizon_status,
        "dead_end": dead_end, "sequential_equivalence": equivalence,
        "budget": budget.receipt(),
        "future_target_used": False, "validation_used_for_selection": False,
        "gpu": False, "training": False, "locked_test": False,
        "candidate_ranking": False, "optimizer": False,
        "performance_claim": False})
    print(json.dumps({"depth": node.depth, "dead_end": dead_end,
                      "sequential_equivalence": equivalence["pass"] if equivalence["executed"] else None}))


if __name__ == "__main__":
    main()
