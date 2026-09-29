"""Deterministic Formal split anchor selection from frozen static domain receipts."""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections import defaultdict
from typing import Mapping, Sequence


def _hash(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def select_stratified_anchors(rows: Sequence[Mapping[str, object]], count: int) -> dict:
    """Balanced hash selection over log-cardinality quartile × Comp-base presence.

    Equal quartile boundaries collapse when the observed count distribution has
    ties. This yields up to eight strata, never an artificial split of equal
    cardinality values. Hash order inside a stratum is independent of input
    enumeration order. No Validation row can influence TRAIN boundaries.
    """
    if count <= 0:
        raise ValueError("positive selection count required")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id")
    nonempty = [row for row in rows if int(row["concrete_count"]) > 0]
    if len(nonempty) < count or any(int(row["concrete_count"]) < 0 for row in rows):
        raise ValueError("insufficient nonempty anchors or negative cardinality")
    values = sorted(math.log(int(row["concrete_count"])) for row in nonempty)
    n = len(values)
    cuts = tuple(values[min(n - 1, math.ceil(n * q / 4) - 1)] for q in (1, 2, 3))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in nonempty:
        q = bisect_left(cuts, math.log(int(row["concrete_count"])))
        comp = int(int(row["eligible_compute_tasks"]) > 0)
        key = f"q{q}:comp{comp}"
        grouped[key].append({"sample_id": str(row["sample_id"]),
                             "concrete_count": int(row["concrete_count"]),
                             "eligible_compute_tasks": int(row["eligible_compute_tasks"]),
                             "stratum": key, "sample_id_sha256": _hash(str(row["sample_id"]))})
    for group in grouped.values():
        group.sort(key=lambda row: (row["sample_id_sha256"], row["sample_id"]))
    strata = sorted(grouped)
    selected = []
    cursor = {key: 0 for key in strata}
    while len(selected) < count:
        changed = False
        for key in strata:
            if len(selected) == count:
                break
            if cursor[key] < len(grouped[key]):
                selected.append(grouped[key][cursor[key]])
                cursor[key] += 1
                changed = True
        if not changed:
            raise AssertionError("selection exhausted despite sufficient anchors")
    return {"requested_count": count, "nonempty_population": n,
            "empty_sample_ids": sorted(str(row["sample_id"]) for row in rows
                                       if int(row["concrete_count"]) == 0),
            "log_cardinality_quartile_cutoffs": list(cuts),
            "strata": {key: {"population": len(grouped[key]), "selected": cursor[key]}
                       for key in strata},
            "selected": selected,
            "selection_rule": "per-split log-cardinality value quartiles × Comp-base presence; balanced strata round-robin; SHA256(sample_id) within stratum"}
