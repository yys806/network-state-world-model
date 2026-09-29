"""Shared state-masked hierarchical proposal over frozen CandidateDomain."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement, product
from typing import Any

from pi_jwm.step6_3b_candidate_grammar_v1 import (
    BoundStructuredStep, CommBlockChoice, StructuredStepChoice, bind_structured_step,
)
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain, StructuralMode


@dataclass
class SparseCategorical:
    """Exact q0-uniform plus sparse elite mass on one fixed legal option mask."""
    prior_weight: float = 1.0
    sparse_weights: dict[Any, float] = field(default_factory=dict)

    def snapshot(self) -> tuple[float, tuple[tuple[str, float], ...]]:
        return self.prior_weight, tuple(sorted((repr(key), value)
                                                for key, value in self.sparse_weights.items()))

    def sample(self, rng: random.Random, uniform_sampler):
        if not self.sparse_weights or rng.random() < self.prior_weight:
            return uniform_sampler(rng)
        draw = rng.random() * sum(self.sparse_weights.values())
        for option, weight in sorted(self.sparse_weights.items(), key=lambda row: repr(row[0])):
            draw -= weight
            if draw <= 0:
                return option
        return option

    def update(self, elite_frequencies: dict[Any, float], eta: float = 0.5,
               epsilon: float = 0.05) -> None:
        if not elite_frequencies:
            return
        if abs(sum(elite_frequencies.values()) - 1.0) > 1e-9:
            raise ValueError("elite frequencies must sum to one")
        old = self.sparse_weights
        self.prior_weight = (1 - epsilon) * (1 - eta) * self.prior_weight + epsilon
        self.sparse_weights = {key: (1 - epsilon) * ((1 - eta) * old.get(key, 0.0) +
                               eta * elite_frequencies.get(key, 0.0))
                               for key in old.keys() | elite_frequencies.keys()}
        self.sparse_weights = {key: value for key, value in self.sparse_weights.items() if value > 0}
        if abs(self.prior_weight + sum(self.sparse_weights.values()) - 1.0) > 1e-9:
            raise AssertionError("categorical distribution lost normalization")


@dataclass
class StructuredProposalDistribution:
    method: str
    tables: dict[tuple[Any, ...], SparseCategorical] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in {"HRS", "S-CEM", "MH-CEM"}:
            raise ValueError("unknown structured proposal method")

    def table(self, layer: str, key: tuple[Any, ...]) -> SparseCategorical:
        if key[0] != layer:
            raise ValueError("proposal table key must begin with its layer")
        return self.tables.setdefault(key, SparseCategorical())

    def draw(self, layer: str, key: tuple[Any, ...], rng: random.Random,
             uniform_sampler, path: list) -> Any:
        option = self.table(layer, key).sample(rng, uniform_sampler)
        path.append((key, option))
        return option

    def update_from_elites(self, elites: list[dict]) -> None:
        if self.method == "HRS" or not elites:
            return
        allowed = {"mode", "count"} if self.method == "S-CEM" else {
            "mode", "count", "subset", "assignment", "start"}
        counts: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
        for elite in elites:
            for key, option in elite["draws"]:
                if key[0] in allowed:
                    counts[key][option] += 1
        for key, frequencies in counts.items():
            total = sum(frequencies.values())
            self.table(key[0], key).update({option: n / total
                                            for option, n in frequencies.items()})


def feasible_counts(domain: CandidateDomain, mode: StructuralMode) -> tuple[int, ...]:
    n = len(domain.wireless_task_to_relation)
    return tuple(k for k in mode.selected_task_counts
                 if (k == 0 and not mode.widths) or
                 (1 <= k <= min(n, len(mode.widths))))


def feasible_assignments(widths: tuple[int, ...], selected: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Canonical row-to-task assignments in sorted width-group order."""
    if not widths:
        return ((),) if not selected else ()
    groups = tuple((width, widths.count(width)) for width in sorted(set(widths)))
    choices = (tuple(combinations_with_replacement(selected, amount)) for _, amount in groups)
    return tuple(tuple(task for group in assignment for task in group)
                 for assignment in product(*choices)
                 if {task for group in assignment for task in group} == set(selected))


def _uniform_multiset(starts: tuple[int, ...], amount: int, rng: random.Random) -> tuple[int, ...]:
    if amount == 0:
        return ()
    # A uniform subset of n+m-1 positions maps bijectively to a multiset.
    indices = sorted(rng.sample(range(len(starts) + amount - 1), amount))
    return tuple(starts[index - offset] for offset, index in enumerate(indices))


def sample_structured_step(domain: CandidateDomain,
                           proposal: StructuredProposalDistribution,
                           rng: random.Random) -> tuple[BoundStructuredStep, dict[str, Any]]:
    """One shared five-layer proposal; each legal mask is renormalized."""
    if domain.is_empty:
        raise ValueError("cannot sample an empty CandidateDomain")
    draws = []
    mode_options = tuple(mode.signature for mode in domain.modes)
    mode_signature = proposal.draw("mode", ("mode", mode_options), rng,
                                   lambda r: r.choice(mode_options), draws)
    mode = next(mode for mode in domain.modes if mode.signature == mode_signature)
    counts = feasible_counts(domain, mode)
    if not counts:
        raise AssertionError("CandidateDomain contains an impossible structural mode")
    k = proposal.draw("count", ("count", mode.signature, counts), rng,
                      lambda r: r.choice(counts), draws)
    eligible = tuple(sorted(domain.wireless_task_to_relation))
    subsets = tuple(combinations(eligible, k))
    selected = proposal.draw("subset", ("subset", eligible, k), rng,
                             lambda r: r.choice(subsets), draws)
    assignments = feasible_assignments(mode.widths, selected)
    assignment = proposal.draw("assignment", ("assignment", mode.widths, selected), rng,
                               lambda r: r.choice(assignments), draws)
    rows = []
    for width, task in zip(mode.widths, assignment):
        rows.append((width, task))
    # Each task-width group receives a uniformly selected canonical RB-start
    # multiset. This removes within-group row permutation duplicates.
    blocks = []
    start_groups = {}
    for width, task in sorted(set(rows)):
        amount = rows.count((width, task))
        starts = domain.pair_starts_by_width[width]
        chosen = proposal.draw("start", ("start", task, width, amount, starts), rng,
                               lambda r: _uniform_multiset(starts, amount, r), draws)
        start_groups[f"{task}:w{width}"] = chosen
        blocks.extend(CommBlockChoice(task, start, width) for start in chosen)
    comp = "NOOP" if mode.signature[1] == "NOOP" else "SCALE_" + mode.signature[1].split("alpha:")[1]
    mobility = ("PROFILE_HOLD" if mode.signature[2] == "NOOP" or
                mode.signature[2].split(",")[0] == "HOLD" else mode.signature[2].split(",")[0])
    choice = StructuredStepChoice(tuple(blocks), comp, mobility)
    bound = bind_structured_step(choice, domain.context, domain.operational_domain,
                                 domain.state, domain.mobility_control, domain.catalog,
                                 domain.prior_signatures, slot_seconds=domain.slot_seconds)
    if not bound.support.formal_pool_admitted or bound.structural_signature != mode.signature:
        raise AssertionError("proposal and frozen CandidateDomain admission disagree")
    return bound, {"joint_mode": mode.signature, "selected_task_count": k,
                   "selected_task_subset": selected, "row_assignment": assignment,
                   "rb_start_groups": start_groups, "draws": draws}


def sample_uniform_structured_step(domain: CandidateDomain, rng: random.Random
                                   ) -> tuple[BoundStructuredStep, dict[str, Any]]:
    """HRS q0: each currently legal hierarchy layer is uniform."""
    return sample_structured_step(domain, StructuredProposalDistribution("HRS"), rng)
