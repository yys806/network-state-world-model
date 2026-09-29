"""Shared search-prefix and one-step transition budget protocol, no optimizer."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep
from pi_jwm.step6_1_trained_candidate_rollout_v1 import (
    OneStepRolloutResult, fingerprint, rollout_one_step,
)
from pi_jwm.step6_3b_candidate_grammar_v1 import BoundStructuredStep
from pi_jwm.step6_3b_candidate_support_v1 import CandidateSupportLabel, Signature


def action_step_fingerprint(step: CandidateActionStep) -> str:
    encoded = json.dumps(step.frame(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SearchNode:
    action_prefix: tuple[CandidateActionStep, ...]
    structural_signature_prefix: tuple[Signature, ...]
    support_labels: tuple[CandidateSupportLabel, ...]
    latent: Mapping[str, Any]
    state: Mapping[str, torch.Tensor]
    graph: Mapping[str, torch.Tensor]
    mobility_control: Mapping[int, Any]
    causal_provenance: str
    fingerprints: Mapping[str, str]
    depth: int
    evaluated_transition_count: int
    cache_hit_count: int

    @classmethod
    def from_anchor(cls, latent: Mapping[str, Any], state: Mapping[str, torch.Tensor],
                    graph: Mapping[str, torch.Tensor], mobility_control: Mapping[int, Any],
                    causal_provenance: str) -> "SearchNode":
        return cls((), (), (), latent, state, graph, mobility_control,
                   causal_provenance,
                   {"latent": fingerprint(latent), "state": fingerprint(state),
                    "graph": fingerprint(graph)}, 0, 0, 0)

    def advance(self, bound: BoundStructuredStep, result: OneStepRolloutResult, *,
                cache_hit: bool) -> "SearchNode":
        if self.depth >= 4 or result.input_fingerprints != self.fingerprints:
            raise ValueError("search prefix must advance from its current causal state")
        return SearchNode((*self.action_prefix, bound.action),
                          (*self.structural_signature_prefix, bound.structural_signature),
                          (*self.support_labels, bound.support), result.latent, result.state,
                          result.graph, result.mobility_control, self.causal_provenance,
                          result.output_fingerprints, self.depth + 1,
                          self.evaluated_transition_count + int(not cache_hit),
                          self.cache_hit_count + int(cache_hit))


@dataclass
class TransitionBudgetAccountant:
    """B_WM counts candidate-step evaluations, including every batch member."""
    b_wm: int
    n_proposed_steps: int = 0
    n_admitted_steps: int = 0
    n_rejected_steps: int = 0
    n_unique_transition_evals: int = 0
    n_cache_hits: int = 0
    n_complete_sequences: int = 0
    n_dead_end_branches: int = 0
    _cache: dict[tuple[str, str, str], OneStepRolloutResult] = field(default_factory=dict, repr=False)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self) -> None:
        if self.b_wm < 0:
            raise ValueError("B_WM must be nonnegative")

    def proposed(self, admitted: bool) -> None:
        self.n_proposed_steps += 1
        if admitted:
            self.n_admitted_steps += 1
        else:
            self.n_rejected_steps += 1

    def dead_end(self) -> None:
        self.n_dead_end_branches += 1

    def complete(self) -> None:
        self.n_complete_sequences += 1

    def cache_key(self, node: SearchNode, bound: BoundStructuredStep) -> tuple[str, str, str]:
        if node.causal_provenance == "":
            raise ValueError("missing causal provenance")
        if (fingerprint(node.latent) != node.fingerprints["latent"] or
                fingerprint(node.state) != node.fingerprints["state"] or
                fingerprint(node.graph) != node.fingerprints["graph"]):
            raise ValueError("search node mutated after fingerprinting")
        parent = fingerprint({"latent": node.fingerprints["latent"],
                              "state": node.fingerprints["state"],
                              "graph": node.fingerprints["graph"]})
        return node.causal_provenance, parent, action_step_fingerprint(bound.action)

    def evaluate(self, node: SearchNode, bound: BoundStructuredStep,
                 transition: Callable[[], OneStepRolloutResult]) -> tuple[OneStepRolloutResult, bool]:
        if not bound.support.formal_pool_admitted:
            raise ValueError("rejected grammar step cannot use transition budget")
        key = self.cache_key(node, bound)
        if key in self._cache:
            self.n_cache_hits += 1
            return copy.deepcopy(self._cache[key]), True
        if self.n_unique_transition_evals >= self.b_wm:
            raise RuntimeError("B_WM_EXHAUSTED")
        result = transition()
        if result.input_fingerprints != node.fingerprints:
            raise ValueError("transition consumed a different parent state")
        self._cache[key] = copy.deepcopy(result)
        self.n_unique_transition_evals += 1
        return result, False

    def evaluate_batch(self, requests: Sequence[tuple[SearchNode, BoundStructuredStep]],
                       transition_batch: Callable[[list[tuple[SearchNode, BoundStructuredStep]]],
                                                  Sequence[OneStepRolloutResult]]) -> tuple[tuple[OneStepRolloutResult, bool], ...]:
        """One Python batch call charges one B_WM unit per unique candidate-step."""
        keys = []
        missing: dict[tuple[str, str, str], tuple[SearchNode, BoundStructuredStep]] = {}
        for node, bound in requests:
            if not bound.support.formal_pool_admitted:
                raise ValueError("rejected grammar step cannot use transition budget")
            key = self.cache_key(node, bound)
            keys.append(key)
            if key not in self._cache and key not in missing:
                missing[key] = (node, bound)
        if self.n_unique_transition_evals + len(missing) > self.b_wm:
            raise RuntimeError("B_WM_EXHAUSTED")
        if missing:
            outcomes = tuple(transition_batch(list(missing.values())))
            if len(outcomes) != len(missing):
                raise ValueError("batch transition result count mismatch")
            for (_, (node, _)), outcome in zip(missing.items(), outcomes):
                if outcome.input_fingerprints != node.fingerprints:
                    raise ValueError("batch transition consumed a different parent state")
            for (key, _), outcome in zip(missing.items(), outcomes):
                self._cache[key] = copy.deepcopy(outcome)
            self.n_unique_transition_evals += len(missing)
        seen: set[tuple[str, str, str]] = set()
        result = []
        for key in keys:
            cache_hit = key not in missing or key in seen
            self.n_cache_hits += int(cache_hit)
            result.append((copy.deepcopy(self._cache[key]), cache_hit))
            seen.add(key)
        return tuple(result)

    def receipt(self) -> dict[str, int | float | str]:
        return {"budget_unit": "candidate_one_step_transition",
                "B_WM": self.b_wm,
                "N_proposed_steps": self.n_proposed_steps,
                "N_admitted_steps": self.n_admitted_steps,
                "N_rejected_steps": self.n_rejected_steps,
                "N_unique_transition_evals": self.n_unique_transition_evals,
                "N_cache_hits": self.n_cache_hits,
                "N_complete_sequences": self.n_complete_sequences,
                "N_dead_end_branches": self.n_dead_end_branches,
                "wall_clock_seconds_diagnostic_only": time.perf_counter() - self._started_at}


def advance_one_step(model: torch.nn.Module, context: Any, domain: Any,
                     node: SearchNode, bound: BoundStructuredStep,
                     budget: TransitionBudgetAccountant) -> SearchNode:
    """Shared mechanism used by any later optimizer; chooses no action itself."""
    budget.proposed(True)
    result, hit = budget.evaluate(node, bound, lambda: rollout_one_step(
        model, context, domain, node.latent, node.state, node.graph,
        node.mobility_control, bound))
    return node.advance(bound, result, cache_hit=hit)
