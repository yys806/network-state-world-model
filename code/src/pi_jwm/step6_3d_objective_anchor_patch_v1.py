"""Deterministic within-stratum replacement for statically empty Objective cohorts."""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from typing import Callable, Mapping, Sequence


def stratum_of(row: Mapping[str, object], cutoffs: Sequence[float]) -> str:
    if int(row["concrete_count"]) <= 0 or len(cutoffs) != 3:
        raise ValueError("replacement requires nonempty domain and three frozen cutoffs")
    quartile = bisect_left(cutoffs, math.log(int(row["concrete_count"])))
    comp = int(int(row["eligible_compute_tasks"]) > 0)
    return f"q{quartile}:comp{comp}"


def replace_empty_cohorts(
    old_manifest: Mapping[str, object], population: Sequence[Mapping[str, object]],
    old_cohort_counts: Mapping[str, int],
    static_cohort_count: Callable[[str], int],
) -> dict:
    """Replace only old cohort=0 IDs, never consulting rollout or search results."""
    old = list(old_manifest["selected"])
    old_ids = [str(row["sample_id"]) for row in old]
    if len(old_ids) != len(set(old_ids)) or set(old_ids) != set(old_cohort_counts):
        raise ValueError("old selection/cohort identity mismatch")
    cutoffs = tuple(float(x) for x in old_manifest["log_cardinality_quartile_cutoffs"])
    by_stratum: dict[str, list[Mapping[str, object]]] = {}
    for row in population:
        if int(row["concrete_count"]) <= 0:
            continue
        by_stratum.setdefault(stratum_of(row, cutoffs), []).append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: (hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest(),
                                   str(row["sample_id"])))
    forbidden = set(old_ids)
    selected = []
    replacements = []
    for old_row in old:
        old_id = str(old_row["sample_id"])
        if old_cohort_counts[old_id] > 0:
            selected.append(dict(old_row))
            continue
        if old_cohort_counts[old_id] < 0:
            raise ValueError("negative cohort count")
        key = str(old_row["stratum"])
        if key != stratum_of(old_row, cutoffs):
            raise ValueError("old stratum differs from frozen cutoffs")
        skipped_zero = []
        replacement = None
        for row in by_stratum.get(key, ()):
            sample_id = str(row["sample_id"])
            if sample_id in forbidden:
                continue
            cohort = static_cohort_count(sample_id)
            if cohort < 0:
                raise ValueError("negative replacement cohort")
            if cohort == 0:
                skipped_zero.append(sample_id)
                continue
            replacement = {"sample_id": sample_id,
                           "concrete_count": int(row["concrete_count"]),
                           "eligible_compute_tasks": int(row["eligible_compute_tasks"]),
                           "stratum": key,
                           "sample_id_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest()}
            forbidden.add(sample_id)
            replacements.append({"old_sample_id": old_id, "new_sample_id": sample_id,
                                 "stratum": key, "new_cohort_count": cohort,
                                 "earlier_hash_candidates_with_empty_cohort": skipped_zero})
            selected.append(replacement)
            break
        if replacement is None:
            raise ValueError(f"no Objective-eligible replacement in {key}")
    if len(selected) != len(old) or len({row["sample_id"] for row in selected}) != len(old):
        raise AssertionError("replacement lost or duplicated an anchor")
    return {"selected": selected, "replacements": replacements,
            "original_selected_count": len(old),
            "zero_cohort_removed": len(replacements),
            "replacement_rule": "same split and frozen cardinality quartile/Comp-base stratum; exclude original selection and empty domains/cohorts; first SHA256(sample_id) eligible candidate"}
