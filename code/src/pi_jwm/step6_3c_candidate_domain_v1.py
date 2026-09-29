"""Search-independent symbolic Planner v1 candidate domain.

Counts canonical concrete action steps without materializing the search space.
Binding and support admission are delegated to the frozen STEP 6.3B grammar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterator, Mapping, Sequence

import torch

from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import PlannerActionDomainContext, PlannerMobilityControlState
from pi_jwm.step6_3b_candidate_grammar_v1 import (
    BoundStructuredStep, CandidateGrammarViolation, CommBlockChoice,
    StructuredStepChoice, _compute_base, _wireless_bindings, bind_structured_step,
)
from pi_jwm.step6_3b_candidate_support_v1 import Signature, TrainStructuralSupportCatalog


def _widths(signature: str) -> tuple[int, ...] | None:
    if signature == "NOOP":
        return ()
    if not signature.startswith("rows:"):
        return None
    try:
        widths = tuple(int(value) for value in signature[5:].split(","))
    except ValueError:
        return None
    return widths if widths and len(widths) <= 4 and all(w in (1, 2, 3) for w in widths) else None


def count_comm_multisets(task_count: int, widths: Sequence[int],
                         pair_counts: Mapping[int, int]) -> int:
    """Inclusion-exclusion: unique row multisets covering every eligible Task.

    For each width group of multiplicity m, choose a multiset of (Task,start)
    pairs. Subtract groups omitting at least one Task. Repeated identical rows
    are one canonical multiset; row permutations are never counted again.
    """
    if task_count < 0 or any(w not in (1, 2, 3) for w in widths):
        raise ValueError("invalid Comm support count request")
    if not widths:
        return int(task_count == 0)
    if task_count == 0 or task_count > len(widths):
        return 0
    multiplicities = {w: widths.count(w) for w in set(widths)}
    total = 0
    for omitted in range(task_count + 1):
        available = task_count - omitted
        product = 1
        for width, multiplicity in multiplicities.items():
            alternatives = available * int(pair_counts.get(width, 0))
            product *= math.comb(alternatives + multiplicity - 1, multiplicity) if alternatives else 0
        total += (-1 if omitted % 2 else 1) * math.comb(task_count, omitted) * product
    return total


@dataclass(frozen=True)
class StructuralMode:
    signature: Signature
    widths: tuple[int, ...]
    comm_concrete_count: int


@dataclass(frozen=True)
class CandidateDomain:
    context: PlannerCandidateContext
    operational_domain: PlannerActionDomainContext
    state: Mapping[str, torch.Tensor]
    mobility_control: Mapping[int, PlannerMobilityControlState]
    catalog: TrainStructuralSupportCatalog
    prior_signatures: tuple[Signature, ...]
    wireless_task_to_relation: Mapping[str, int]
    compute_base: Mapping[str, tuple[str, float]]
    present_uav_slots: tuple[int, ...]
    modes: tuple[StructuralMode, ...]
    pair_starts_by_width: Mapping[int, tuple[int, ...]]
    empty_reason: str | None
    slot_seconds: float

    @classmethod
    def from_state(cls, context: PlannerCandidateContext,
                   operational_domain: PlannerActionDomainContext,
                   state: Mapping[str, torch.Tensor],
                   mobility_control: Mapping[int, PlannerMobilityControlState],
                   catalog: TrainStructuralSupportCatalog,
                   prior_signatures: Sequence[Signature] = (), *,
                   slot_seconds: float = 0.1) -> "CandidateDomain":
        if not 0 <= len(prior_signatures) < 4 or slot_seconds <= 0:
            raise ValueError("H1-H4 and positive slot duration required")
        if context.causal_provenance != operational_domain.causal_provenance:
            raise ValueError("causal provenance mismatch")
        indices = context.static["input_entity_index"]
        starts = {w: tuple(sorted(s for s, width in catalog.comm_start_width_pairs if width == w))
                  for w in (1, 2, 3)}
        try:
            if int(state["rb_active_mask"].shape[-1]) != 50:
                raise CandidateGrammarViolation("RB_SUPPORT_NOT_FORMAL_TRAIN_50")
            wireless = _wireless_bindings(state, indices["task"])
            base = _compute_base(state, indices["task"], indices["physical"],
                                 operational_domain, slot_seconds)
            present = tuple(slot for slot in range(state["uav_mask"].shape[1])
                            if bool(state["uav_mask"][0, slot])
                            and bool(state["entity_presence"][0, slot]))
            if any(slot not in mobility_control or
                   not mobility_control[slot].heading_observed or
                   not mobility_control[slot].elevation_observed for slot in present):
                raise CandidateGrammarViolation("CURRENT_UAV_CONTROL_STATE_UNOBSERVED")
            if base and not any(amount > 0 for _, amount in base.values()):
                raise CandidateGrammarViolation("COMP_ALPHA_UNIDENTIFIABLE_FROM_ZERO_BASE")
        except CandidateGrammarViolation as exc:
            return cls(context, operational_domain, state, mobility_control, catalog,
                       tuple(prior_signatures), {}, {}, (), (), starts, str(exc), slot_seconds)
        comp_modes = ("NOOP",) if not base else tuple(
            f"tasks:{len(base)};alpha:{alpha}" for alpha in (0.5, 0.75, 1.0))
        mobility_modes = ("NOOP",) if not present else tuple(
            ",".join(("HOLD" if profile == "PROFILE_HOLD" else profile)
                     for _ in present)
            for profile in ("PROFILE_HOLD", "PROFILE_1", "PROFILE_2",
                            "PROFILE_3", "PROFILE_4", "PROFILE_5"))
        modes = []
        pair_counts = {w: len(values) for w, values in starts.items()}
        for signature in sorted(catalog.joint):
            comm, comp, mobility = signature
            if comp not in comp_modes or mobility not in mobility_modes:
                continue
            widths = _widths(comm)
            if widths is None:
                continue
            count = count_comm_multisets(len(wireless), widths, pair_counts)
            if count:
                modes.append(StructuralMode(signature, widths, count))
        return cls(context, operational_domain, state, mobility_control, catalog,
                   tuple(prior_signatures), wireless, base, present, tuple(modes), starts,
                   None if modes else "NO_TRAIN_OBSERVED_JOINT_STRUCTURE_COMPATIBLE",
                   slot_seconds)

    @property
    def exact_unique_single_step_count(self) -> int:
        return sum(mode.comm_concrete_count for mode in self.modes)

    @property
    def comm_concrete_count(self) -> int:
        return sum({mode.signature[0]: mode.comm_concrete_count for mode in self.modes}.values())

    @property
    def comp_mode_count(self) -> int:
        return len({mode.signature[1] for mode in self.modes})

    @property
    def mobility_profile_count(self) -> int:
        return len({mode.signature[2] for mode in self.modes})

    @property
    def is_empty(self) -> bool:
        return not self.modes

    def iter_choices(self) -> Iterator[StructuredStepChoice]:
        """Lazy deterministic grammar order; never builds the whole space."""
        tasks = tuple(sorted(self.wireless_task_to_relation))
        for mode in self.modes:
            groups = tuple((width, mode.widths.count(width)) for width in sorted(set(mode.widths)))
            comp = "NOOP" if mode.signature[1] == "NOOP" else "SCALE_" + mode.signature[1].split("alpha:")[1]
            mob = ("PROFILE_HOLD" if mode.signature[2] in ("NOOP", ",".join("HOLD" for _ in self.present_uav_slots))
                   else mode.signature[2].split(",")[0])

            def rows_at(index: int, accumulated: tuple[CommBlockChoice, ...]) -> Iterator[tuple[CommBlockChoice, ...]]:
                if index == len(groups):
                    if {row.task_id for row in accumulated} == set(tasks):
                        yield accumulated
                    return
                width, multiplicity = groups[index]
                options = tuple(CommBlockChoice(task, start, width)
                                for task in tasks for start in self.pair_starts_by_width[width])
                for row_indices in combinations_with_replacement(range(len(options)), multiplicity):
                    yield from rows_at(index + 1, accumulated + tuple(options[i] for i in row_indices))

            for rows in rows_at(0, ()):
                yield StructuredStepChoice(rows, comp, mob)

    def iter_bound(self) -> Iterator[BoundStructuredStep]:
        for choice in self.iter_choices():
            bound = bind_structured_step(choice, self.context, self.operational_domain,
                                         self.state, self.mobility_control, self.catalog,
                                         self.prior_signatures, slot_seconds=self.slot_seconds)
            if not bound.support.formal_pool_admitted:
                raise AssertionError("symbolic domain disagrees with STEP 6.3B admission")
            yield bound
