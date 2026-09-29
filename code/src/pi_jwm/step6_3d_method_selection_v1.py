"""Frozen H4 paired comparison and anchor-cluster method selection."""
from __future__ import annotations

import random
from typing import Mapping, Sequence


def paired_outcome(left: tuple[int, float, float, float, float] | None,
                   right: tuple[int, float, float, float, float] | None) -> int:
    """+1 left win, -1 left loss, 0 tie; None means no H4 scoreable result."""
    if left is None or right is None:
        return int(left is not None) - int(right is not None)
    if len(left) != 5 or len(right) != 5:
        raise ValueError("frozen five-part objective required")
    return (left < right) - (left > right)


def cluster_bootstrap_advantage(
        per_anchor_seed_outcomes: Mapping[str, Sequence[int]], *, seed: int = 6316,
        replications: int = 10000) -> dict:
    """Percentile bootstrap resamples anchors, preserving five paired seeds."""
    if not per_anchor_seed_outcomes or replications <= 0:
        raise ValueError("nonempty paired anchors and replications required")
    rows = []
    for anchor, outcomes in sorted(per_anchor_seed_outcomes.items()):
        if len(outcomes) != 5 or any(value not in (-1, 0, 1) for value in outcomes):
            raise ValueError(f"five paired outcomes required for {anchor}")
        rows.append(sum(outcomes) / 5)
    n = len(rows)
    observed = sum(rows) / n
    rng = random.Random(seed)
    boot = sorted(sum(rows[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(replications))
    lower = boot[int(0.025 * replications)]
    upper = boot[min(replications - 1, int(0.975 * replications))]
    return {"anchor_cluster_count": n, "seeds_per_anchor": 5,
            "mean_paired_advantage": observed, "ci95_lower": lower,
            "ci95_upper": upper, "bootstrap_replications": replications,
            "bootstrap_seed": seed, "bootstrap_type": "anchor_cluster_percentile"}


def select_method(pairwise_ci: Mapping[tuple[str, str], Mapping[str, float]]) -> str:
    def better(a: str, b: str) -> bool:
        return pairwise_ci[(a, b)]["ci95_lower"] > 0
    s = better("S-CEM", "HRS")
    mh = better("MH-CEM", "HRS")
    if not s and not mh:
        return "HRS"
    if s and not mh:
        return "S-CEM"
    if mh and not s:
        return "MH-CEM"
    return "MH-CEM" if better("MH-CEM", "S-CEM") else "S-CEM"


def tune_cem_config(results: Mapping[tuple[int, float], Mapping[tuple[str, int],
                                                              tuple[int, float, float, float, float] | None]]) -> dict:
    """TRAIN-only config tournament: success, paired net outcome, smaller K/rho."""
    required = {(k, rho) for k in (3, 4) for rho in (0.1, 0.2)}
    if set(results) != required:
        raise ValueError("CEM grid must contain exactly four frozen configs")
    pair_keys = {key for outcomes in results.values() for key in outcomes}
    if any(set(outcomes) != pair_keys for outcomes in results.values()):
        raise ValueError("CEM configurations must share paired TRAIN anchors/seeds")
    if not pair_keys:
        raise ValueError("empty TRAIN tuning results")
    ranking = []
    for config, outcomes in results.items():
        success = sum(value is not None for value in outcomes.values())
        paired_net = sum(paired_outcome(outcomes[key], results[other][key])
                         for other in required if other != config for key in pair_keys)
        ranking.append({"config": config, "h4_success_count": success,
                        "paired_objective_net_outcome": paired_net})
    ranking.sort(key=lambda row: (-row["h4_success_count"],
                                  -row["paired_objective_net_outcome"],
                                  row["config"][0], row["config"][1]))
    return {"selected_config": ranking[0]["config"], "ranking": ranking,
            "paired_case_count": len(pair_keys),
            "selection_order": "H4 success count; paired objective net outcome; smaller K; smaller rho"}
