"""Fixed real TRAIN paths: serial/batch H4 support and scorer equivalence."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import torch

from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step, rollout_one_step_batch
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, score_candidate_set,
    objective_sort_key,
)
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode
from pi_jwm.step6_3d_fixed_budget_search_v1 import _trace
from pi_jwm.step6_3d_structured_proposal_v1 import StructuredProposalDistribution, sample_structured_step
from run_step6_1_trained_candidate_rollout_preflight_v1 import tree_numeric_delta
from run_step6_3d_one_cpu_solve_v1 import OUT, load_frozen_runtime

SIZES = (1, 4, 8, 16)
TOL = 1e-4


def _score(root, node, results, anchor, side, slot_duration_s, seed):
    if node.depth != 4:
        return {"status": "GRAMMAR_DEAD_END", "H_sup": None, "objective": None}
    candidate = CandidateActionSequence(node.action_prefix, "step6_3d_batch_gate",
        Backend.SEARCH, seed, anchor.context.causal_provenance,
        generation_metadata={"planner_action_domain": "V1"})
    ident = candidate.fingerprint
    candidate = CandidateActionSequence(node.action_prefix, ident,
        Backend.SEARCH, seed, anchor.context.causal_provenance,
        generation_metadata={"planner_action_domain": "V1"})
    try:
        scored = score_candidate_set(anchor.state, side,
            ((candidate, _trace(ident, anchor.fingerprints, tuple(results))),),
            slot_duration_s=slot_duration_s)
    except (BurdenSemanticsBlocked, ScorerStateInconsistency) as exc:
        return {"status": type(exc).__name__ + ":" + str(exc),
                "H_sup": None, "objective": None}
    horizon = scored.support_horizons.get(ident)
    objective = (tuple(scored.scores[0].objective_tuple)
                 if scored.status == "SCOREABLE" and scored.scores else None)
    return {"status": scored.status, "H_sup": horizon, "objective": objective}


def main() -> None:
    manifest = json.loads((OUT / "15_train_anchor_manifest_objective_eligible.json").read_text(encoding="utf-8"))
    # Fixed manifest position, independent of search outcomes.
    sample_id = manifest["selected"][2]["sample_id"]
    model, _, anchor, side, catalog, protocol, meta, digest = load_frozen_runtime(sample_id)
    if meta["split"] != "dev_train":
        raise ValueError("real TRAIN fixture required")
    root = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
        anchor.domain.mobility_states, anchor.context.causal_provenance)
    rng = random.Random(6393)
    proposal = StructuredProposalDistribution("HRS")
    seed = 6393
    path_rows = []
    with torch.no_grad():
        for size in SIZES:
            serial = [(root, []) for _ in range(size)]
            batched = [(root, []) for _ in range(size)]
            grammar_equal = True
            support_equal = True
            max_delta = 0.0
            for _depth in range(4):
                requests = []
                live = []
                for index, ((sn, sr), (bn, br)) in enumerate(zip(serial, batched)):
                    sd = CandidateDomain.from_state(anchor.context, anchor.domain,
                        sn.state, sn.mobility_control, catalog, sn.structural_signature_prefix)
                    bd = CandidateDomain.from_state(anchor.context, anchor.domain,
                        bn.state, bn.mobility_control, catalog, bn.structural_signature_prefix)
                    grammar_equal &= (sd.is_empty == bd.is_empty and
                        sd.exact_unique_single_step_count == bd.exact_unique_single_step_count and
                        tuple(m.signature for m in sd.modes) == tuple(m.signature for m in bd.modes))
                    if sd.is_empty or bd.is_empty:
                        continue
                    bound, _ = sample_structured_step(sd, proposal, rng)
                    requests.append((bn, bound))
                    live.append((index, sn, sr, bn, br, bound))
                if not requests:
                    break
                batch_results = rollout_one_step_batch(model, anchor.context, anchor.domain, requests)
                for (index, sn, sr, bn, br, bound), batch_result in zip(live, batch_results):
                    one = rollout_one_step(model, anchor.context, anchor.domain,
                        sn.latent, sn.state, sn.graph, sn.mobility_control, bound)
                    max_delta = max(max_delta, *(tree_numeric_delta(getattr(one, field),
                        getattr(batch_result, field)) for field in
                        ("action_tensor", "latent", "state", "graph", "model_trace")))
                    grammar_equal &= one.action_mapping == batch_result.action_mapping
                    serial[index] = (sn.advance(bound, one, cache_hit=False), sr + [one])
                    batched[index] = (bn.advance(bound, batch_result, cache_hit=False), br + [batch_result])
            serial_scores = [_score(root, node, results, anchor, side,
                protocol.training.rssm.slot_duration_s, seed) for node, results in serial]
            batch_scores = [_score(root, node, results, anchor, side,
                protocol.training.rssm.slot_duration_s, seed) for node, results in batched]
            support_equal = all((a["status"], a["H_sup"]) == (b["status"], b["H_sup"])
                for a, b in zip(serial_scores, batch_scores))
            serial_order = sorted((objective_sort_key(row["objective"], str(i)), i)
                for i, row in enumerate(serial_scores) if row["objective"] is not None)
            batch_order = sorted((objective_sort_key(row["objective"], str(i)), i)
                for i, row in enumerate(batch_scores) if row["objective"] is not None)
            ordering_equal = [i for _, i in serial_order] == [i for _, i in batch_order]
            objectives_equal = all(
                (a["objective"] is None and b["objective"] is None) or
                (a["objective"] is not None and b["objective"] is not None and
                 a["objective"][0] == b["objective"][0] and
                 all(abs(x - y) <= TOL for x, y in zip(a["objective"][1:],
                                                       b["objective"][1:])))
                for a, b in zip(serial_scores, batch_scores))
            row = {"batch_size": size, "paths": size,
                "complete_h4": sum(node.depth == 4 for node, _ in serial),
                "h4_scoreable": len(serial_order), "grammar_admission_equal": grammar_equal,
                "H_sup_and_support_equal": support_equal,
                "objective_ordering_equal": ordering_equal,
                "objective_values_equal": objectives_equal,
                "max_abs_numeric_delta": max_delta,
                "pass": grammar_equal and support_equal and ordering_equal and
                        objectives_equal and max_delta <= TOL}
            path_rows.append(row)
            print(f"H4 batch={size}: {row['pass']} complete={row['complete_h4']} scoreable={row['h4_scoreable']}", flush=True)
            if not row["pass"]:
                break
    receipt = {"verdict": "PASS" if len(path_rows) == 4 and all(r["pass"] for r in path_rows)
               else "BLOCKED", "sample_id": sample_id, "split": "dev_train",
               "rows": path_rows, "numeric_tolerance": TOL,
               "parameter_digest": digest, "gpu": False, "locked_test": False,
               "validation_search_executed": False}
    (OUT / "23_h4_batch_equivalence_cpu_gate.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(receipt["verdict"])


if __name__ == "__main__":
    main()
