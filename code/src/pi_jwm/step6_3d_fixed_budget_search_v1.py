"""Fixed-budget H4 structured search on the frozen 6.3C transition contract."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from pi_jwm.step6_1_trained_candidate_rollout_v1 import CandidateRolloutTrace, OneStepRolloutResult
from pi_jwm.step6_2b_planner_objective_scorer_v1 import objective_sort_key
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_structured_proposal_v1 import (
    StructuredProposalDistribution, sample_structured_step,
)


@dataclass(frozen=True)
class SearchOutcome:
    method: str
    seed: int
    budget: int
    iterations: int
    elite_ratio: float | None
    best_objective: tuple[int, float, float, float, float] | None
    best_fingerprint: str | None
    h4_scoreable_count: int
    h4_unscoreable_count: int
    complete_sequence_count: int
    budget_receipt: Mapping[str, Any]
    iteration_rows: tuple[Mapping[str, Any], ...]
    batch_size: int = 1


def quotas(budget: int, iterations: int) -> tuple[int, ...]:
    if budget <= 0 or iterations <= 0 or iterations > budget:
        raise ValueError("positive budget and valid iteration count required")
    base, remainder = divmod(budget, iterations)
    return (base,) * (iterations - 1) + (base + remainder,)


def _trace(candidate_id: str, initial_fingerprints: Mapping[str, str],
           results: tuple[OneStepRolloutResult, ...]) -> CandidateRolloutTrace:
    return CandidateRolloutTrace(candidate_id, initial_fingerprints,
        tuple(row.input_fingerprints for row in results),
        tuple(row.output_fingerprints for row in results),
        tuple(row.action_tensor for row in results),
        tuple(row.action_mapping for row in results),
        tuple(row.latent for row in results),
        tuple(row.state for row in results),
        tuple(row.graph for row in results),
        tuple(row.model_trace for row in results))


def solve_fixed_budget(
    *, method: str, seed: int, b_wm: int, anchor: Any,
    catalog: Any, transition: Callable[[SearchNode, Any], OneStepRolloutResult],
    score_h4: Callable[[CandidateActionSequence, CandidateRolloutTrace], Any | None],
    iterations: int = 1, elite_ratio: float | None = None,
    max_stalled_proposals: int = 5000,
    transition_batch: Callable[[list[tuple[SearchNode, Any]]],
                               tuple[OneStepRolloutResult, ...]] | None = None,
    batch_size: int = 1,
) -> SearchOutcome:
    """Search H4; score_h4 returns None for any H_sup<4 or scorer failure.

    The shared CandidateDomain is rebound at every predicted state. The
    accountant owns the transition cache and unique one-step budget. A run
    that cannot spend its quota due to repeated proposals is explicitly
    incomplete and must not enter the formal comparison.
    """
    if method == "HRS":
        if iterations != 1 or elite_ratio is not None:
            raise ValueError("HRS uses one immutable uniform proposal")
    elif method not in {"S-CEM", "MH-CEM"} or iterations not in (3, 4) or elite_ratio not in (0.1, 0.2):
        raise ValueError("CEM configuration outside frozen search grid")
    if batch_size < 1 or (batch_size > 1 and transition_batch is None):
        raise ValueError("positive batch size and batch transition callback required")
    rng = random.Random(seed)
    proposal = StructuredProposalDistribution(method)
    accountant = TransitionBudgetAccountant(b_wm)
    allocation = quotas(b_wm, iterations)
    root = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
                                  anchor.domain.mobility_states,
                                  anchor.context.causal_provenance)
    scoreable: dict[str, tuple[Any, dict]] = {}
    retained: list[tuple[Any, dict]] = []
    unscoreable = 0
    iteration_rows = []
    def score_complete(node: SearchNode, results: list[OneStepRolloutResult],
                       draws: list, current: list[tuple[Any, dict]]) -> None:
        nonlocal unscoreable
        if node.depth != 4:
            return
        accountant.complete()
        candidate = CandidateActionSequence(node.action_prefix, "step6_3d_candidate",
            Backend.SEARCH, seed, anchor.context.causal_provenance,
            generation_metadata={"planner_action_domain": "V1"})
        candidate_id = candidate.fingerprint
        candidate = CandidateActionSequence(node.action_prefix, candidate_id,
            Backend.SEARCH, seed, anchor.context.causal_provenance,
            generation_metadata={"planner_action_domain": "V1"})
        if candidate_id in scoreable:
            score = scoreable[candidate_id][0]
        else:
            score = score_h4(candidate, _trace(candidate_id, anchor.fingerprints, tuple(results)))
        if score is None:
            unscoreable += 1
        else:
            pair = (score, {"draws": draws})
            scoreable[candidate_id] = pair
            current.append(pair)

    for iteration, quota in enumerate(allocation):
        limit = accountant.n_unique_transition_evals + quota
        current: list[tuple[Any, dict]] = []
        stalled = 0
        proposed_sequences = 0
        while accountant.n_unique_transition_evals < limit:
            if batch_size > 1:
                branches = [(root, [], []) for _ in range(batch_size)]
                proposed_sequences += len(branches)
                new_this_wave = 0
                for _depth in range(1, 5):
                    requests = []
                    branch_rows = []
                    pending_new = set()
                    remaining = limit - accountant.n_unique_transition_evals
                    for node, results, draws in branches:
                        domain = CandidateDomain.from_state(anchor.context, anchor.domain,
                            node.state, node.mobility_control, catalog,
                            node.structural_signature_prefix)
                        if domain.is_empty:
                            accountant.dead_end()
                            continue
                        bound, path = sample_structured_step(domain, proposal, rng)
                        accountant.proposed(True)
                        key = accountant.cache_key(node, bound)
                        if key not in accountant._cache and key not in pending_new:
                            if len(pending_new) >= remaining:
                                continue
                            pending_new.add(key)
                        requests.append((node, bound))
                        branch_rows.append((node, results, draws, bound, path))
                    if not requests:
                        break
                    outcomes = accountant.evaluate_batch(requests, transition_batch)
                    next_branches = []
                    for (node, results, draws, bound, path), (result, hit) in zip(branch_rows, outcomes):
                        new_this_wave += int(not hit)
                        next_node = node.advance(bound, result, cache_hit=hit)
                        next_results = results + [result]
                        next_draws = draws + path["draws"]
                        if next_node.depth == 4:
                            score_complete(next_node, next_results, next_draws, current)
                        else:
                            next_branches.append((next_node, next_results, next_draws))
                    branches = next_branches
                    if not branches:
                        break
                stalled = stalled + batch_size if new_this_wave == 0 else 0
                if stalled >= max_stalled_proposals:
                    raise RuntimeError("B_WM_QUOTA_UNFILLED_PROPOSAL_STALL")
                continue
            proposed_sequences += 1
            node = root
            results = []
            draws = []
            new_this_proposal = 0
            stopped_for_quota = False
            for depth in range(1, 5):
                domain = CandidateDomain.from_state(anchor.context, anchor.domain,
                    node.state, node.mobility_control, catalog,
                    node.structural_signature_prefix)
                if domain.is_empty:
                    accountant.dead_end()
                    break
                bound, path = sample_structured_step(domain, proposal, rng)
                accountant.proposed(True)
                key = accountant.cache_key(node, bound)
                if key not in accountant._cache and accountant.n_unique_transition_evals >= limit:
                    stopped_for_quota = True
                    break
                result, hit = accountant.evaluate(node, bound,
                    lambda n=node, b=bound: transition(n, b))
                new_this_proposal += int(not hit)
                results.append(result)
                draws.extend(path["draws"])
                node = node.advance(bound, result, cache_hit=hit)
            score_complete(node, results, draws, current)
            stalled = stalled + 1 if new_this_proposal == 0 else 0
            if stalled >= max_stalled_proposals:
                raise RuntimeError("B_WM_QUOTA_UNFILLED_PROPOSAL_STALL")
            if stopped_for_quota and accountant.n_unique_transition_evals >= limit:
                break
        if accountant.n_unique_transition_evals != limit:
            raise RuntimeError("B_WM_ITERATION_QUOTA_UNFILLED")
        current.sort(key=lambda row: objective_sort_key(row[0].objective_tuple,
                                                        row[0].candidate_fingerprint))
        unique_current = {row[0].candidate_fingerprint: row for row in current}
        elite_pool = {row[0].candidate_fingerprint: row for row in retained}
        elite_pool.update(unique_current)
        elite_count = max(1, math.ceil(len(elite_pool) * elite_ratio)) if elite_ratio else 0
        elites = sorted(elite_pool.values(), key=lambda row: objective_sort_key(
            row[0].objective_tuple, row[0].candidate_fingerprint))[:elite_count]
        updated = method != "HRS" and len(unique_current) >= 2
        if updated:
            proposal.update_from_elites([path for _, path in elites])
        # Only the previous iteration's best completed elites enter the next
        # pool. Fewer than two newly completed scoreable candidates freeze q.
        retained = elites[:min(5, len(elites))]
        iteration_rows.append({"iteration": iteration + 1, "quota": quota,
            "proposed_sequences": proposed_sequences, "unique_h4_scoreable": len(unique_current),
            "retained_from_previous": len(elite_pool) - len(unique_current),
            "elite_count": len(elites) if updated else 0, "proposal_updated": updated,
            "retained_count": len(retained), "unique_transition_evals": accountant.n_unique_transition_evals})
    best = min(scoreable.values(), key=lambda row: objective_sort_key(
        row[0].objective_tuple, row[0].candidate_fingerprint))[0] if scoreable else None
    return SearchOutcome(method, seed, b_wm, iterations, elite_ratio,
        None if best is None else best.objective_tuple,
        None if best is None else best.candidate_fingerprint,
        len(scoreable), unscoreable, accountant.n_complete_sequences,
        accountant.receipt(), tuple(iteration_rows), batch_size)
