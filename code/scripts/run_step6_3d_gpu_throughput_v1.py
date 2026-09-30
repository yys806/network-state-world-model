"""Bounded HRS-only GPU execution throughput probe; no method comparison."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import psutil
import torch

from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step, rollout_one_step_batch
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, score_candidate_set,
)
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3d_structured_proposal_v1 import StructuredProposalDistribution, sample_structured_step
from run_step6_3d_one_cpu_solve_v1 import (
    OUT as BASE_OUT, gpu_transition_batch_with_cpu_storage,
    gpu_transition_one_with_cpu_storage, load_frozen_runtime,
    offload_prepared_anchor,
)
from run_step6_3d_gpu_equivalence_v1 import OUT, delta, domain_signature


def gpu_sample():
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        return [int(item.strip()) for item in raw.split(",")]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8,16,32,64,128,256")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-multiplier", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--cpu-storage", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    selected = json.loads((BASE_OUT / "15_train_anchor_manifest_objective_eligible.json").read_text())
    sample_id = selected["selected"][2]["sample_id"]
    model, encoder, anchor, side, catalog, protocol, meta, digest = load_frozen_runtime(
        sample_id, device="cuda")
    if args.cpu_storage:
        anchor = offload_prepared_anchor(anchor)
    if meta["split"] != "dev_train":
        raise ValueError("TRAIN fixture required")
    encoder_calls = [0]
    original_forward = encoder.forward
    def counted_forward(*values, **kwargs):
        encoder_calls[0] += 1
        return original_forward(*values, **kwargs)
    encoder.forward = counted_forward
    model_calls = [0]
    original_one_step = model.one_step
    def counted_one_step(*values, **kwargs):
        model_calls[0] += 1
        return original_one_step(*values, **kwargs)
    model.one_step = counted_one_step
    process = psutil.Process()
    total_vram = torch.cuda.get_device_properties(0).total_memory

    def solve(size, budget, seed):
        support = {}
        batch_shapes = []
        def score_h4(candidate, trace):
            try:
                scored = score_candidate_set(anchor.state, side, ((candidate, trace),),
                    slot_duration_s=protocol.training.rssm.slot_duration_s)
            except (BurdenSemanticsBlocked, ScorerStateInconsistency):
                return None
            for reason in scored.diagnostics.get("support_boundary_reasons", {}).values():
                if reason:
                    support[str(reason)] = support.get(str(reason), 0) + 1
            return scored.scores[0] if scored.status == "SCOREABLE" and scored.H_eff == 4 and scored.scores else None
        def batch(requests):
            batch_shapes.append({"requests": len(requests),
                "comp_rows": [len(bound.action.comp) for _, bound in requests],
                "comm_rows": [len(bound.action.comm) for _, bound in requests],
                "state_shape": list(requests[0][0].state["position"].shape),
                "graph_shape": list(requests[0][0].graph[next(iter(requests[0][0].graph))].shape)})
            if args.cpu_storage:
                return gpu_transition_batch_with_cpu_storage(
                    model, anchor.context, anchor.domain, requests)
            return rollout_one_step_batch(model, anchor.context, anchor.domain, requests)
        def one(node, bound):
            if args.cpu_storage:
                return gpu_transition_one_with_cpu_storage(
                    model, anchor.context, anchor.domain, node, bound)
            return rollout_one_step(model, anchor.context,
                anchor.domain, node.latent, node.state, node.graph,
                node.mobility_control, bound)
        with torch.inference_mode():
            return solve_fixed_budget(method="HRS", seed=seed, b_wm=budget,
                anchor=anchor, catalog=catalog,
                transition=one,
                transition_batch=batch if size > 1 else None,
                score_h4=score_h4, batch_size=size), batch_shapes, support

    with torch.inference_mode():
        root = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
            anchor.domain.mobility_states, anchor.context.causal_provenance)
        domain = CandidateDomain.from_state(anchor.context, anchor.domain, root.state,
            root.mobility_control, catalog, ())
        import random
        rng = random.Random(6399)
        bound, _ = sample_structured_step(domain, StructuredProposalDistribution("HRS"), rng)
        accountant = TransitionBudgetAccountant(2)
        duplicate = accountant.evaluate_batch([(root, bound), (root, bound)],
            lambda requests: gpu_transition_batch_with_cpu_storage(
                model, anchor.context, anchor.domain, requests) if args.cpu_storage else
                rollout_one_step_batch(model, anchor.context, anchor.domain, requests))
        calls_after_first = model_calls[0]
        repeated = accountant.evaluate_batch([(root, bound)],
            lambda requests: gpu_transition_batch_with_cpu_storage(
                model, anchor.context, anchor.domain, requests) if args.cpu_storage else
                rollout_one_step_batch(model, anchor.context, anchor.domain, requests))
        cache_pass = (accountant.n_unique_transition_evals == 1 and
            accountant.n_cache_hits == 2 and model_calls[0] == calls_after_first and
            [hit for _, hit in duplicate] == [False, True] and repeated[0][1])
        cache_receipt = {"pass": cache_pass, "B_WM": accountant.n_unique_transition_evals,
            "cache_hits": accountant.n_cache_hits,
            "model_forward_calls_after_first": calls_after_first,
            "model_forward_calls_after_repeat": model_calls[0],
            "key_fields": ["causal provenance", "parent latent/state/graph fingerprint",
                           "canonical action fingerprint"]}
        print(f"cache/accounting pass={cache_pass}", flush=True)
        # One warm-up solve covers all four depths before steady-state timing.
        solve(8, 32, 6399)
        torch.cuda.synchronize()

        rows = []
        for size in (int(v) for v in args.sizes.split(",")):
            budget = max(64, size * args.budget_multiplier)
            repeats = []
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                before_calls = model_calls[0]
                before_encoder = encoder_calls[0]
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    outcome, shapes, support = solve(size, budget, 6400 + repeat)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    unique = outcome.budget_receipt["N_unique_transition_evals"]
                    if unique != budget:
                        raise AssertionError("B_WM budget not filled")
                    repeats.append({"repeat": repeat, "status": "PASS",
                        "unique_transitions": unique, "wall_clock_seconds": elapsed,
                        "transitions_per_second": unique / elapsed,
                        "model_forward_calls": model_calls[0] - before_calls,
                        "encoder_forward_calls": encoder_calls[0] - before_encoder,
                        "allocated_vram_bytes": torch.cuda.memory_allocated(),
                        "reserved_vram_bytes": torch.cuda.memory_reserved(),
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                        "host_rss_bytes": process.memory_info().rss,
                        "gpu_utilization_snapshot": gpu_sample(),
                        "cache_hits": outcome.budget_receipt["N_cache_hits"],
                        "proposed_steps": outcome.budget_receipt["N_proposed_steps"],
                        "admitted_steps": outcome.budget_receipt["N_admitted_steps"],
                        "complete_sequences": outcome.complete_sequence_count,
                        "h4_scoreable": outcome.h4_scoreable_count,
                        "support_reasons": support,
                        "batch_shapes": shapes})
                    print(f"batch={size} repeat={repeat} tps={unique/elapsed:.3f} "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    repeats.append({"repeat": repeat, "status": "OOM", "reason": str(exc)})
                    torch.cuda.empty_cache()
                    print(f"batch={size} OOM", flush=True)
                    break
            good = [row for row in repeats if row["status"] == "PASS"]
            row = {"batch_size": size, "budget_per_repeat": budget, "repeats": repeats,
                "stable": len(good) == args.repeats,
                "median_transitions_per_second": statistics.median(
                    row["transitions_per_second"] for row in good) if good else None,
                "min_transitions_per_second": min(
                    (row["transitions_per_second"] for row in good), default=None),
                "max_transitions_per_second": max(
                    (row["transitions_per_second"] for row in good), default=None),
                "max_peak_vram_bytes": max(
                    (row["peak_allocated_vram_bytes"] for row in good), default=None)}
            rows.append(row)
        eligible = [row for row in rows if row["stable"] and
                    row["max_peak_vram_bytes"] <= total_vram * 0.85 and
                    all(r["encoder_forward_calls"] == 0 for r in row["repeats"])]
        best = max(eligible, key=lambda row: row["median_transitions_per_second"]) if eligible else None
        selected_size = best["batch_size"] if eligible else None
        median = best["median_transitions_per_second"] if eligible else None
        receipt = {"verdict": "PASS" if cache_pass and eligible else "BLOCKED",
            "sample_id": sample_id, "split": "dev_train", "method": "HRS_DIAGNOSTIC_ONLY",
            "precision": "FP32", "prior_mode": "mean", "service_mode": "expectation",
            "checkpoint_parameter_digest": digest, "total_vram_bytes": total_vram,
            "safety_margin_minimum_fraction": 0.15,
            "cache_and_budget": cache_receipt,
            "anchor_encoder_reuse": {"passed": encoder_calls[0] == 0,
                "encoder_calls_after_anchor_initialization": encoder_calls[0]},
            "rows": rows, "recommended_batch_size": selected_size,
            "recommended_median_transitions_per_second": median,
            "speedup_vs_cpu_batch8_0_964_tps": median / 0.964 if median else None,
            "estimated_seconds_for_2113536_transitions": 2113536 / median if median else None,
            "bucket_strategy": "none", "locked_test": False,
            "state_storage": "cpu_cache_and_prefix" if args.cpu_storage else "gpu_native",
            "formal_train_tuning": False, "validation_comparison": False}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "04_gpu_batch_throughput.json").write_text(json.dumps(receipt,
            indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        print(f"throughput verdict={receipt['verdict']} recommended={selected_size}")


if __name__ == "__main__":
    main()
