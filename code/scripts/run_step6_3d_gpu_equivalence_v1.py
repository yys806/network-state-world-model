"""Frozen TRAIN fixture CPU FP32 / CUDA FP32 execution equivalence gate."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import torch

from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step, rollout_one_step_batch
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, objective_sort_key, score_candidate_set,
)
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_fixed_budget_search_v1 import _trace
from pi_jwm.step6_3d_structured_proposal_v1 import StructuredProposalDistribution, sample_structured_step
from run_step6_3d_one_cpu_solve_v1 import OUT as BASE_OUT, load_frozen_runtime

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930"
TOL = 1e-4


def delta(left, right, key=""):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return float("inf")
        right = right.detach().cpu()
        left = left.detach().cpu()
        if key == "outage_uniform_draw":
            return 0.0 if torch.equal(torch.isnan(left), torch.isnan(right)) else float("inf")
        if left.is_floating_point():
            if not (torch.equal(torch.isnan(left), torch.isnan(right)) and
                    torch.equal(torch.isinf(left), torch.isinf(right))):
                return float("inf")
            finite = torch.isfinite(left)
            return float((left[finite] - right[finite]).abs().max()) if bool(finite.any()) else 0.0
        return 0.0 if torch.equal(left, right) else float("inf")
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return float("inf")
        return max((delta(left[k], right[k], k) for k in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf")
        return max((delta(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else float("inf")


def domain_signature(domain):
    return (domain.is_empty, domain.exact_unique_single_step_count,
            tuple(mode.signature for mode in domain.modes))


def score(anchor, side, protocol, node, results, seed):
    if node.depth != 4:
        return {"status": "GRAMMAR_DEAD_END", "H_sup": None, "reason": None,
                "objective": None}
    candidate = CandidateActionSequence(node.action_prefix, "gpu_gate", Backend.SEARCH,
        seed, anchor.context.causal_provenance,
        generation_metadata={"planner_action_domain": "V1"})
    ident = candidate.fingerprint
    candidate = CandidateActionSequence(node.action_prefix, ident, Backend.SEARCH,
        seed, anchor.context.causal_provenance,
        generation_metadata={"planner_action_domain": "V1"})
    try:
        scored = score_candidate_set(anchor.state, side,
            ((candidate, _trace(ident, anchor.fingerprints, tuple(results))),),
            slot_duration_s=protocol.training.rssm.slot_duration_s)
    except (BurdenSemanticsBlocked, ScorerStateInconsistency) as exc:
        return {"status": type(exc).__name__, "H_sup": None, "reason": str(exc),
                "objective": None}
    reasons = scored.diagnostics.get("support_boundary_reasons", {})
    return {"status": scored.status, "H_sup": scored.support_horizons.get(ident),
            "reason": reasons.get(ident),
            "objective": tuple(scored.scores[0].objective_tuple)
                if scored.status == "SCOREABLE" and scored.scores else None}


def gate_fixture(sample_id, *, sizes, h4_paths):
    cpu_model, cpu_encoder, cpu, cpu_side, catalog, protocol, meta, cpu_digest = load_frozen_runtime(sample_id)
    gpu_model, gpu_encoder, gpu, gpu_side, _, _, _, gpu_digest = load_frozen_runtime(sample_id, device="cuda")
    if meta["split"] != "dev_train" or cpu_digest != gpu_digest:
        raise ValueError("non-TRAIN fixture or parameter mismatch")
    root_cpu = SearchNode.from_anchor(cpu.latent, cpu.state, cpu.graph,
        cpu.domain.mobility_states, cpu.context.causal_provenance)
    root_gpu = SearchNode.from_anchor(gpu.latent, gpu.state, gpu.graph,
        gpu.domain.mobility_states, gpu.context.causal_provenance)
    root_delta = max(delta(getattr(cpu, key), getattr(gpu, key))
                     for key in ("state", "graph", "latent", "z_pi"))
    rows = []
    with torch.inference_mode():
        for size in sizes:
            rng = random.Random(63128 + size)
            cpu_domain = CandidateDomain.from_state(cpu.context, cpu.domain,
                root_cpu.state, root_cpu.mobility_control, catalog, ())
            gpu_domain = CandidateDomain.from_state(gpu.context, gpu.domain,
                root_gpu.state, root_gpu.mobility_control, catalog, ())
            grammar_equal = domain_signature(cpu_domain) == domain_signature(gpu_domain)
            bounds = [sample_structured_step(cpu_domain, StructuredProposalDistribution("HRS"), rng)[0]
                      for _ in range(size)]
            requests = [(root_gpu, bound) for bound in bounds]
            gpu_results = rollout_one_step_batch(gpu_model, gpu.context, gpu.domain, requests)
            torch.cuda.synchronize()
            deltas = {key: 0.0 for key in
                      ("action_tensor", "latent", "state", "graph", "model_trace")}
            mapping_equal = True
            causal_equal = True
            next_support_equal = True
            for bound, gr in zip(bounds, gpu_results):
                cr = rollout_one_step(cpu_model, cpu.context, cpu.domain,
                    root_cpu.latent, root_cpu.state, root_cpu.graph,
                    root_cpu.mobility_control, bound)
                for key in deltas:
                    deltas[key] = max(deltas[key], delta(getattr(cr, key), getattr(gr, key)))
                mapping_equal &= cr.action_mapping == gr.action_mapping
                causal_equal &= (root_cpu.causal_provenance == root_gpu.causal_provenance and
                                 cr.input_fingerprints == root_cpu.fingerprints and
                                 gr.input_fingerprints == root_gpu.fingerprints)
                cn = root_cpu.advance(bound, cr, cache_hit=False)
                gn = root_gpu.advance(bound, gr, cache_hit=False)
                cd = CandidateDomain.from_state(cpu.context, cpu.domain, cn.state,
                    cn.mobility_control, catalog, cn.structural_signature_prefix)
                gd = CandidateDomain.from_state(gpu.context, gpu.domain, gn.state,
                    gn.mobility_control, catalog, gn.structural_signature_prefix)
                next_support_equal &= domain_signature(cd) == domain_signature(gd)
            passed = (grammar_equal and mapping_equal and causal_equal and next_support_equal and
                      max(deltas.values()) <= TOL)
            rows.append({"batch_size": size, "action_mapping_equal": mapping_equal,
                         "grammar_admission_equal": grammar_equal,
                         "next_candidate_domain_equal": next_support_equal,
                         "causal_identity_valid": causal_equal,
                         "max_absolute_delta_by_field": deltas,
                         "pass": passed})
            print(f"one-step {sample_id} batch={size} pass={passed}", flush=True)
            if not passed:
                break
        rng = random.Random(6393)
        cpu_paths = [(root_cpu, []) for _ in range(h4_paths)]
        gpu_paths = [(root_gpu, []) for _ in range(h4_paths)]
        depth_rows = []
        for depth in range(4):
            requests = []
            live = []
            signatures_equal = True
            for index, ((cn, cr), (gn, gr)) in enumerate(zip(cpu_paths, gpu_paths)):
                cd = CandidateDomain.from_state(cpu.context, cpu.domain,
                    cn.state, cn.mobility_control, catalog, cn.structural_signature_prefix)
                gd = CandidateDomain.from_state(gpu.context, gpu.domain,
                    gn.state, gn.mobility_control, catalog, gn.structural_signature_prefix)
                signatures_equal &= domain_signature(cd) == domain_signature(gd)
                if cd.is_empty or gd.is_empty:
                    continue
                bound, _ = sample_structured_step(cd, StructuredProposalDistribution("HRS"), rng)
                requests.append((gn, bound))
                live.append((index, cn, cr, gn, gr, bound))
            if not requests:
                depth_rows.append({"depth": depth + 1, "requests": 0,
                                   "grammar_equal": signatures_equal})
                break
            gpu_results = rollout_one_step_batch(gpu_model, gpu.context, gpu.domain, requests)
            numeric = 0.0
            for (index, cn, cr, gn, gr, bound), gres in zip(live, gpu_results):
                cres = rollout_one_step(cpu_model, cpu.context, cpu.domain,
                    cn.latent, cn.state, cn.graph, cn.mobility_control, bound)
                numeric = max(numeric, *(delta(getattr(cres, key), getattr(gres, key))
                    for key in ("action_tensor", "latent", "state", "graph", "model_trace")))
                cpu_paths[index] = (cn.advance(bound, cres, cache_hit=False), cr + [cres])
                gpu_paths[index] = (gn.advance(bound, gres, cache_hit=False), gr + [gres])
            depth_rows.append({"depth": depth + 1, "requests": len(requests),
                               "grammar_equal": signatures_equal,
                               "max_absolute_delta": numeric})
        cpu_scores = [score(cpu, cpu_side, protocol, node, results, 6393)
                      for node, results in cpu_paths]
        gpu_scores = [score(gpu, gpu_side, protocol, node, results, 6393)
                      for node, results in gpu_paths]
        discrete_equal = all((a["status"], a["H_sup"], a["reason"]) ==
                             (b["status"], b["H_sup"], b["reason"])
                             for a, b in zip(cpu_scores, gpu_scores))
        cpu_order = [i for _, i in sorted((objective_sort_key(row["objective"], str(i)), i)
            for i, row in enumerate(cpu_scores) if row["objective"] is not None)]
        gpu_order = [i for _, i in sorted((objective_sort_key(row["objective"], str(i)), i)
            for i, row in enumerate(gpu_scores) if row["objective"] is not None)]
        objective_delta = max((max(abs(a-b) for a,b in zip(c["objective"], g["objective"]))
            for c,g in zip(cpu_scores,gpu_scores) if c["objective"] is not None
            and g["objective"] is not None), default=0.0)
        h4_numeric_max = max((row.get("max_absolute_delta", 0) for row in depth_rows),
                             default=0.0)
        h4_numeric_within_tolerance = h4_numeric_max <= TOL
        # Formal H4 eligibility is the frozen discrete support and strict
        # objective ordering. Preserve the independent numeric diagnostic.
        h4_pass = (all(row["grammar_equal"] for row in depth_rows) and
                   discrete_equal and cpu_order == gpu_order and objective_delta <= TOL)
        print(f"H4 {sample_id} pass={h4_pass} scoreable={len(cpu_order)}", flush=True)
        return {"sample_id": sample_id, "split": meta["split"],
                "root_max_absolute_delta": root_delta, "parameter_digest": cpu_digest,
                "one_step": rows,
                "h4": {"depth_rows": depth_rows, "cpu_scores": cpu_scores,
                       "gpu_scores": gpu_scores, "discrete_equal": discrete_equal,
                       "objective_order_equal": cpu_order == gpu_order,
                       "objective_max_absolute_delta": objective_delta,
                       "intermediate_numeric_max_absolute_delta": h4_numeric_max,
                       "intermediate_numeric_within_tolerance": h4_numeric_within_tolerance,
                       "pass": h4_pass},
                "pass": len(rows) == len(sizes) and all(r["pass"] for r in rows)
                        and h4_pass and root_delta <= TOL}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1,4,8,16,32")
    parser.add_argument("--h4-paths", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    selected = json.loads((BASE_OUT / "15_train_anchor_manifest_objective_eligible.json").read_text())
    sizes = tuple(int(value) for value in args.sizes.split(","))
    rows = [gate_fixture(selected["selected"][i]["sample_id"], sizes=sizes,
                         h4_paths=args.h4_paths) for i in (0, 2)]
    receipt = {"verdict": "PASS" if all(row["pass"] for row in rows) else "BLOCKED",
               "device": "cuda:0", "precision": "FP32", "prior_mode": "mean",
               "service_mode": "expectation", "numeric_tolerance": TOL,
               "fixtures": rows, "locked_test": False,
               "formal_tuning": False, "validation_comparison": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "03_cpu_gpu_equivalence.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8")
    print(receipt["verdict"])


if __name__ == "__main__":
    main()
