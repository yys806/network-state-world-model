"""Repeat the same two R1 diagnostic actions with complete read-only evidence.

This does not implement a new Route action or change any model/scoring rule.
The B anchor remains explicitly pre-injected Shadow, not a paired objective.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v4"


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    import torch
    import run_step6_4l_r1_frozen_wm_scorer_v2 as harness
    import pi_jwm.step6_1_trained_candidate_rollout_v1 as rollout
    import pi_jwm.step6_2b_planner_objective_scorer_v1 as scorer
    from pi_jwm.step6_0c_planner_action_domain_v1 import require_domain_admissible
    from audit_step6_4l_r1_acceptance_v1 import validate_observed_identity

    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("evidence namespace already used; refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic(OUT / "00_scope.json", {"status": "STARTED", "unique_actions": 2,
           "repeat_of": "R1 v2 A/B, no new action or root", "new_transition_cap": 8,
           "timeout_seconds": 300, "GPU": False, "locked_test": False,
           "pre_route_paired_objective": False})

    def timeout():
        atomic(OUT / "timeout.json", {"status": "TIMEOUT", "elapsed_seconds": time.perf_counter() - started})
        os._exit(124)

    timer = threading.Timer(300, timeout)
    timer.daemon = True
    timer.start()
    original_rollout = rollout.rollout_candidate_sequential
    original_score = scorer.score_candidate_set
    observed = {}

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True

    def save_tensors(path, values):
        tmp = path.with_suffix(".pt.tmp")
        torch.save(values, tmp)
        tmp.replace(path)

    def capture_rollout(model, anchor, candidate, **kwargs):
        mode = candidate.generation_metadata["route_experiment"]
        admission = require_domain_admissible(candidate, anchor.context, anchor.domain)
        row = {"candidate": asdict(candidate), "official_admission_passed": True,
               "admission_result": asdict(admission),
               "parameter_before": rollout.fingerprint(model.state_dict()),
               "initial": {"state": rollout.fingerprint(anchor.state),
                           "graph": rollout.fingerprint(anchor.graph),
                           "latent": rollout.fingerprint(anchor.latent)},
               "inputs": [], "outputs": [], "finite": [], "ledger": [],
               "shadow_anchor_modified": mode == "B", "formal_route_action": False}
        observed[mode] = row
        save_tensors(OUT / f"{mode}_anchor.pt", {"state": anchor.state,
                      "graph": anchor.graph, "latent": anchor.latent})
        task_index = int(anchor.context.static["input_entity_index"]["task"]["Task_5"])
        original_one_step = model.one_step

        def one_step(*args, **kw):
            latent, state, graph = args[:3]
            row["inputs"].append({"state": rollout.fingerprint(state),
                                 "graph": rollout.fingerprint(graph),
                                 "latent": rollout.fingerprint(latent)})
            t0 = time.perf_counter()
            result = original_one_step(*args, **kw)
            nxt_latent, nxt_state, nxt_graph, model_trace = result
            h = len(row["outputs"]) + 1
            row["outputs"].append({"state": rollout.fingerprint(nxt_state),
                                  "graph": rollout.fingerprint(nxt_graph),
                                  "latent": rollout.fingerprint(nxt_latent)})
            # The model trace contains an intentionally masked diagnostic
            # outage draw (NaN outside its support).  Finite acceptance is
            # defined on the recursively consumed state/graph/latent only;
            # retain the raw trace separately for audit.
            row["finite"].append(finite((nxt_latent, nxt_state, nxt_graph)))
            task = {k: v[0, task_index].tolist() for k, v in nxt_state.items()
                    if k.startswith("task_") and isinstance(v, torch.Tensor)
                    and v.ndim == 2 and v.shape[1] == nxt_state["task_presence"].shape[1]}
            slot = int(nxt_state["return_flow_index"][0, task_index])
            flow = {k: v[0, slot].tolist() for k, v in nxt_state.items()
                    if slot >= 0 and isinstance(v, torch.Tensor) and v.ndim >= 2
                    and (k.startswith("flow_") or k in ("hop_remaining", "hop_progress",
                      "carrying_active", "carrying_hop_source_index", "carrying_hop_destination_index",
                      "current_holder_index", "current_hop_index", "route_node_indices", "route_node_mask"))}
            row["ledger"].append({"horizon": h, "one_step_seconds": time.perf_counter() - t0,
                                  "task_id": "Task_5", "task_index": task_index,
                                  "task": task, "return_slot": slot, "flow": flow})
            save_tensors(OUT / f"{mode}_H{h}.pt", {"state": nxt_state,
                         "graph": nxt_graph, "latent": nxt_latent, "model_trace": model_trace})
            atomic(OUT / f"{mode}_observed.json", row)
            return result

        model.one_step = one_step
        try:
            trace = original_rollout(model, anchor, candidate, **kwargs)
        finally:
            model.one_step = original_one_step
        row["parameter_after"] = rollout.fingerprint(model.state_dict())
        row["mappings"] = trace.mappings
        validate_observed_identity(row)
        atomic(OUT / f"{mode}_observed.json", row)
        return trace

    def capture_score(anchor_state, side, pairs, **kwargs):
        t0 = time.perf_counter()
        result = original_score(anchor_state, side, pairs, **kwargs)
        mode = pairs[0][0].generation_metadata["route_experiment"]
        atomic(OUT / f"{mode}_raw_scorer.json", {"raw_result": asdict(result),
               "side": asdict(side), "scorer_seconds": time.perf_counter() - t0,
               "anchor_components": [{"task_id": t.task_id, "task_index": t.task_index,
                   "terminal_at_anchor": t.terminal_at_anchor,
                   "task_presence": bool(anchor_state["task_presence"][0, t.task_index]),
                   "task_completed": bool(anchor_state["task_completed"][0, t.task_index]),
                   "work_remaining": float(anchor_state["task_work_remaining"][0, t.task_index]),
                   "anchor_flows": scorer._anchor_flows(anchor_state, t.task_index)} for t in side.tasks],
               "formal_score_promoted": False})
        return result

    rollout.rollout_candidate_sequential = capture_rollout
    scorer.score_candidate_set = capture_score
    harness.OUT = OUT
    try:
        harness.main()
    except BaseException as exc:
        atomic(OUT / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        timer.cancel()
        rollout.rollout_candidate_sequential = original_rollout
        scorer.score_candidate_set = original_score
    atomic(OUT / "10_observation_completion.json", {"status": "COMPLETED",
           "new_wm_transitions": sum(len(r["outputs"]) for r in observed.values()),
           "unique_actions": 2, "elapsed_seconds": time.perf_counter() - started,
           "formal_semantics_modified": False, "GPU": False, "locked_test": False})


if __name__ == "__main__":
    main()
