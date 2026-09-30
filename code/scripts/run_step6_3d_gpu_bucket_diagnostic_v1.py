"""Compare execution-only structural buckets on the same TRAIN requests."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import torch

from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step_batch
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_structured_proposal_v1 import StructuredProposalDistribution, sample_structured_step
from run_step6_3d_gpu_equivalence_v1 import OUT, TOL, delta
from run_step6_3d_one_cpu_solve_v1 import OUT as BASE_OUT, load_frozen_runtime


def main():
    selected = json.loads((BASE_OUT / "15_train_anchor_manifest_objective_eligible.json").read_text())
    sample_id = selected["selected"][2]["sample_id"]
    model, _, anchor, _, catalog, _, meta, digest = load_frozen_runtime(sample_id, device="cuda")
    root = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
        anchor.domain.mobility_states, anchor.context.causal_provenance)
    domain = CandidateDomain.from_state(anchor.context, anchor.domain, root.state,
        root.mobility_control, catalog, ())
    rng = random.Random(6432)
    proposal = StructuredProposalDistribution("HRS")
    requests = [(root, sample_structured_step(domain, proposal, rng)[0]) for _ in range(32)]
    key_counts = defaultdict(int)
    for _, bound in requests:
        key_counts[(len(bound.action.comp), len(bound.action.comm))] += 1

    def plain(items):
        return rollout_one_step_batch(model, anchor.context, anchor.domain, items)

    def bucketed(items):
        groups = defaultdict(list)
        for index, (node, bound) in enumerate(items):
            groups[(len(bound.action.comp), len(bound.action.comm))].append((index, node, bound))
        ordered = [None] * len(items)
        for group in groups.values():
            values = plain([(node, bound) for _, node, bound in group])
            for (index, _, _), value in zip(group, values):
                ordered[index] = value
        return tuple(ordered)

    rows = []
    outputs = {}
    with torch.inference_mode():
        # Separate warm-up before timing.
        plain(requests[:4]); bucketed(requests[:4]); torch.cuda.synchronize()
        for name, transition in (("unbucketed", plain), ("bucketed", bucketed)):
            repeats = []
            for _ in range(3):
                accountant = TransitionBudgetAccountant(32)
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = accountant.evaluate_batch(requests, transition)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                unique = accountant.n_unique_transition_evals
                outputs[name] = [value for value, _ in result]
                repeats.append({"seconds": elapsed, "transitions_per_second": unique/elapsed,
                    "unique_transitions": unique,
                    "cache_hits": accountant.n_cache_hits})
            rows.append({"strategy": name, "repeats": repeats})
    field_deltas = {key: max(delta(getattr(a, key), getattr(b, key))
        for a,b in zip(outputs["unbucketed"], outputs["bucketed"]))
        for key in ("action_tensor", "latent", "state", "graph", "model_trace")}
    mapping_equal = all(a.action_mapping == b.action_mapping
        for a,b in zip(outputs["unbucketed"], outputs["bucketed"]))
    import statistics
    unbucketed_tps = statistics.median(r["transitions_per_second"] for r in rows[0]["repeats"])
    bucketed_tps = statistics.median(r["transitions_per_second"] for r in rows[1]["repeats"])
    budget_equal = (rows[0]["repeats"][0]["unique_transitions"] ==
                    rows[1]["repeats"][0]["unique_transitions"])
    equivalence = mapping_equal and budget_equal and max(field_deltas.values()) <= TOL
    receipt = {"sample_id": sample_id, "split": meta["split"],
        "parameter_digest": digest, "request_count": len(requests),
        "structural_row_count_groups": {str(key): value for key,value in key_counts.items()},
        "rows": rows, "max_absolute_delta_by_field": field_deltas,
        "action_mapping_equal": mapping_equal, "numeric_tolerance": TOL,
        "budget_equal": budget_equal,
        "equivalence_pass": equivalence,
        "bucketed_speedup": bucketed_tps / unbucketed_tps,
        "adopt_bucket": bool(equivalence and bucketed_tps > unbucketed_tps),
        "locked_test": False, "formal_method_comparison": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "05_bucket_diagnostic.json").write_text(json.dumps(receipt,
        sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"bucket equivalence={equivalence} speedup={bucketed_tps/unbucketed_tps:.3f} "
          f"adopt={receipt['adopt_bucket']}")


if __name__ == "__main__":
    main()
