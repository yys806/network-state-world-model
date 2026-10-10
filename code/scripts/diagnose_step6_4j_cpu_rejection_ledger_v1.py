"""Read-only CPU rejection ledger for two representative STEP 6.4J roots.

This reuses the frozen CPU search/scorer path. It is deliberately labeled a
conditional diagnostic: r39 did not persist candidate prefixes, RNG state,
H1-H4 traces, or objective side-state, so this cannot be an exact replay of
the CUDA candidates.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step, rollout_one_step_batch
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, score_candidate_set,
)
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from run_step6_3d_one_cpu_solve_v1 import (
    load_frozen_runtime, fingerprint, _to_device,
)


def diagnose(sample_id: str, *, seed: int = 6311, budget: int = 512,
             batch_size: int = 1) -> dict:
    model, encoder, prepared, side, catalog, protocol, meta, parameter_digest = load_frozen_runtime(
        sample_id, device="cpu")
    state, context, domain = prepared.state, prepared.context, prepared.domain
    ledger: list[dict] = []
    reason_counts: dict[str, int] = {}
    h_eff_counts: dict[str, int] = {}
    support_horizon_counts: dict[str, int] = {}
    complete_events = 0
    scorer_exceptions = 0

    def add_count(mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    def score_h4(candidate, trace):
        nonlocal complete_events, scorer_exceptions
        complete_events += 1
        row = {
            "candidate_fingerprint": candidate.fingerprint,
            "candidate_id": candidate.candidate_id,
            "root_sample_id": sample_id,
            "horizon": candidate.horizon,
            "planner_action_domain": candidate.generation_metadata.get("planner_action_domain"),
            "trace_candidate_id_matches": candidate.candidate_id == trace.candidate_id,
            "anchor_fingerprint": trace.initial_fingerprints.get("anchor"),
            "trace_h1_h4_state_count": len(trace.states),
        }
        try:
            scored = score_candidate_set(state, side, ((candidate, trace),),
                slot_duration_s=protocol.training.rssm.slot_duration_s)
        except (ScorerStateInconsistency, BurdenSemanticsBlocked, ValueError) as exc:
            scorer_exceptions += 1
            reason = type(exc).__name__ + ":" + str(exc)
            add_count(reason_counts, "SCORER_EXCEPTION")
            row.update({"status": "EXCEPTION", "H_eff": None,
                        "support_horizons": {}, "primary_reason": reason,
                        "future_return_birth": False})
            ledger.append(row)
            return None
        supports = {str(k): int(v) for k, v in scored.support_horizons.items()}
        reasons = scored.diagnostics.get("support_boundary_reasons", {})
        reason_values = [str(v) for v in reasons.values() if v is not None]
        h_eff = int(scored.H_eff)
        for horizon in supports.values():
            add_count(support_horizon_counts, str(horizon))
        add_count(h_eff_counts, str(h_eff))
        future = any("UNSUPPORTED_FUTURE_RETURN_BIRTH" in reason for reason in reason_values)
        if scored.status == "SCOREABLE" and h_eff == 4 and scored.scores:
            primary = "SCOREABLE_H4"
        elif reason_values:
            primary = "UNSUPPORTED_FUTURE_RETURN_BIRTH" if future else "OTHER_SUPPORT_BOUNDARY"
        else:
            primary = str(scored.status)
        add_count(reason_counts, primary)
        row.update({
            "status": scored.status,
            "H_eff": h_eff,
            "support_horizons": supports,
            "support_boundary_reasons": reason_values,
            "primary_reason": primary,
            "future_return_birth": future,
            "cohort_size": scored.diagnostics.get("cohort_size"),
            "scores_present": bool(scored.scores),
            "H_sup_values": [int(s.H_sup) for s in scored.scores],
        })
        ledger.append(row)
        return scored.scores[0] if primary == "SCOREABLE_H4" else None

    with __import__("torch").inference_mode():
        outcome = solve_fixed_budget(
            method="S-CEM", seed=seed, b_wm=budget, anchor=prepared,
            catalog=catalog,
            transition=lambda node, bound: rollout_one_step(
                model, context, domain, node.latent, node.state, node.graph,
                node.mobility_control, bound),
            score_h4=score_h4, iterations=4, elite_ratio=0.2,
            batch_size=batch_size,
            transition_batch=(lambda requests: rollout_one_step_batch(
                model, context, domain, requests)) if batch_size > 1 else None,
        )
    after = fingerprint({"encoder": encoder.state_dict(), "rssm": model.state_dict()})
    if after != parameter_digest:
        raise AssertionError("frozen World Model parameters changed")
    receipt = asdict(outcome)
    receipt.pop("winner_sequence", None)
    return {
        "diagnostic_label": "NON_EQUIVALENT_DIAGNOSTIC",
        "sample_id": sample_id,
        "split": meta["split"],
        "seed": seed,
        "budget": budget,
        "batch_size": batch_size,
        "method": "S-CEM",
        "iterations": 4,
        "elite_ratio": 0.2,
        "outcome": receipt,
        "complete_h4_events": complete_events,
        "dead_end_branches": receipt["budget_receipt"]["N_dead_end_branches"],
        "scoreable_h4_events": reason_counts.get("SCOREABLE_H4", 0),
        "unscoreable_h4_events": complete_events - reason_counts.get("SCOREABLE_H4", 0) - scorer_exceptions,
        "scorer_exception_events": scorer_exceptions,
        "reason_counts_primary": reason_counts,
        "H_eff_counts": h_eff_counts,
        "support_horizon_counts": support_horizon_counts,
        "ledger": ledger,
        "parameter_digest": parameter_digest,
        "checkpoint_and_dataset_verified_by_loader": True,
        "original_r39_candidate_replay": "NOT_POSSIBLE_FROM_ARTIFACTS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if len(args.sample_id) != 2:
        raise SystemExit("exactly two representative sample ids are required")
    results = [diagnose(sample_id, batch_size=args.batch_size) for sample_id in args.sample_id]
    payload = {
        "schema": "PI-JWM-STEP-6.4J-CPU-REJECTION-LEDGER-v1",
        "diagnostic_label": "NON_EQUIVALENT_DIAGNOSTIC",
        "original_r39_replay_status": "NOT_POSSIBLE_FROM_ARTIFACTS",
        "roots": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps({
        "diagnostic_label": payload["diagnostic_label"],
        "roots": [{k: r[k] for k in ("sample_id", "complete_h4_events", "dead_end_branches",
                                     "scoreable_h4_events", "unscoreable_h4_events",
                                     "scorer_exception_events", "reason_counts_primary", "H_eff_counts")}
                  for r in results],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
