"""Real TRAIN one-step serial/batch equivalence and bounded CPU throughput."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import psutil
import torch

from pi_jwm.step6_1_trained_candidate_rollout_v1 import (
    rollout_one_step, rollout_one_step_batch,
)
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_structured_proposal_v1 import (
    StructuredProposalDistribution, sample_structured_step,
)
from run_step6_1_trained_candidate_rollout_preflight_v1 import tree_numeric_delta
from run_step6_3d_one_cpu_solve_v1 import OUT, load_frozen_runtime

SIZES = (1, 4, 8, 16)
TOL = 1e-4  # accepted STEP 6.1 deterministic serial/batch numeric gate


def _requests(anchor, catalog, count: int):
    root = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
                                  anchor.domain.mobility_states,
                                  anchor.context.causal_provenance)
    domain = CandidateDomain.from_state(anchor.context, anchor.domain, root.state,
                                         root.mobility_control, catalog)
    if domain.is_empty:
        raise ValueError("chosen real TRAIN fixture has empty CandidateDomain")
    proposal = StructuredProposalDistribution("HRS")
    rng = random.Random(6392)
    accountant = TransitionBudgetAccountant(count)
    requests = []
    seen = set()
    for _ in range(5000):
        bound, _ = sample_structured_step(domain, proposal, rng)
        key = accountant.cache_key(root, bound)
        if key not in seen:
            seen.add(key)
            requests.append((root, bound))
        if len(requests) == count:
            return requests
    raise RuntimeError("could not sample required distinct real TRAIN actions")


def _peak_rss_during(run):
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def watch():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        start = time.perf_counter()
        value = run()
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        watcher.join()
    return value, elapsed, max(peak[0], process.memory_info().rss)


def _run_batches(model, anchor, requests, size, *, accounted: bool):
    outcomes = []
    budget = TransitionBudgetAccountant(len(requests)) if accounted else None
    for index in range(0, len(requests), size):
        chunk = requests[index:index + size]
        if budget is None:
            outcomes.extend(rollout_one_step_batch(model, anchor.context,
                            anchor.domain, chunk))
        else:
            outcomes.extend(result for result, _ in budget.evaluate_batch(chunk,
                lambda rows: rollout_one_step_batch(model, anchor.context,
                                                    anchor.domain, rows)))
    return outcomes, budget


def main() -> None:
    selected = json.loads((OUT / "15_train_anchor_manifest_objective_eligible.json").read_text(encoding="utf-8"))
    sample_id = selected["selected"][0]["sample_id"]
    model, encoder, anchor, side, catalog, protocol, meta, parameter_digest = load_frozen_runtime(sample_id)
    if meta["split"] != "dev_train":
        raise ValueError("batch fixture must be Formal TRAIN")
    torch.set_num_threads(1)
    with torch.no_grad():
        requests = _requests(anchor, catalog, 64)
        serial = [rollout_one_step(model, anchor.context, anchor.domain,
                                  node.latent, node.state, node.graph,
                                  node.mobility_control, bound)
                  for node, bound in requests[:16]]
        rows = []
        for size in SIZES:
            batched, _, = _run_batches(model, anchor, requests[:16], size, accounted=False)
            deltas = {name: max(tree_numeric_delta(getattr(a, name), getattr(b, name))
                                for a, b in zip(serial, batched))
                      for name in ("action_tensor", "latent", "state", "graph", "model_trace")}
            mappings_equal = all(a.action_mapping == b.action_mapping for a, b in zip(serial, batched))
            inputs_equal = all(a.input_fingerprints == b.input_fingerprints for a, b in zip(serial, batched))
            support_equal = True
            for (node, bound), a, b in zip(requests[:16], serial, batched):
                next_a = node.advance(bound, a, cache_hit=False)
                next_b = node.advance(bound, b, cache_hit=False)
                da = CandidateDomain.from_state(anchor.context, anchor.domain,
                    next_a.state, next_a.mobility_control, catalog,
                    next_a.structural_signature_prefix)
                db = CandidateDomain.from_state(anchor.context, anchor.domain,
                    next_b.state, next_b.mobility_control, catalog,
                    next_b.structural_signature_prefix)
                support_equal &= (da.is_empty == db.is_empty and
                    da.exact_unique_single_step_count == db.exact_unique_single_step_count and
                    tuple(mode.signature for mode in da.modes) ==
                    tuple(mode.signature for mode in db.modes))
            passed = max(deltas.values()) <= TOL and mappings_equal and inputs_equal and support_equal
            rows.append({"batch_size": size, "max_abs_delta_by_component": deltas,
                         "action_mapping_equal": mappings_equal,
                         "parent_fingerprint_equal": inputs_equal,
                         "next_candidate_domain_support_equal": support_equal,
                         "one_step_equivalence_pass": passed})
            print(f"equivalence batch={size}: {passed}", flush=True)
            if not passed:
                break
        if len(rows) != 4 or not all(row["one_step_equivalence_pass"] for row in rows):
            verdict = "BLOCKED_BATCH_SERIAL_DIFFERENCE"
            throughput = []
        else:
            verdict = "ONE_STEP_EQUIVALENCE_PASS_H4_ORDER_PENDING"
            throughput = []
            for size in SIZES:
                (outcomes, budget), elapsed, peak = _peak_rss_during(
                    lambda s=size: _run_batches(model, anchor, requests, s, accounted=True))
                if len(outcomes) != 64 or budget.n_unique_transition_evals != 64:
                    raise AssertionError("batch transition budget mismatch")
                for index in range(0, len(requests), size):
                    chunk = requests[index:index + size]
                    budget.evaluate_batch(chunk, lambda _: (_ for _ in ()).throw(
                        AssertionError("cache hit repeated a transition")))
                if budget.n_unique_transition_evals != 64 or budget.n_cache_hits != 64:
                    raise AssertionError("batch cache accounting mismatch")
                throughput.append({"batch_size": size, "unique_transitions": 64,
                                   "seconds": elapsed, "transitions_per_second": 64 / elapsed,
                                   "peak_process_rss_bytes_diagnostic": peak,
                                   "cache_hits_on_replay": 64,
                                   "B_WM_accounting_pass": True})
                print(f"throughput batch={size}: {64/elapsed:.4f} transitions/s", flush=True)
            base = throughput[0]["transitions_per_second"]
            for row in throughput:
                row["effective_speedup_vs_batch1"] = row["transitions_per_second"] / base
    receipt = {"verdict": verdict, "sample_id": sample_id, "split": meta["split"],
               "batch_sizes": list(SIZES), "numeric_tolerance": TOL,
               "equivalence": rows, "throughput": throughput,
               "H4_support_and_objective_ordering": "PENDING_SEPARATE_GATE",
               "parameter_digest": parameter_digest, "gpu_batch_probe": "NOT_AVAILABLE",
               "locked_test": False, "validation_search_executed": False}
    (OUT / "21_batch_one_step_cpu_gate.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": verdict, "sample_id": sample_id}))


if __name__ == "__main__":
    main()
