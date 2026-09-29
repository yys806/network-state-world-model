"""Search-independent TRAIN structural support labels for Planner candidates.

The catalog is built from Formal TRAIN action frames. It describes observed
structures and never reads a future target during candidate classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

Signature = tuple[str, str, str]  # Comm, Comp, Mobility


@dataclass(frozen=True)
class CandidateSupportLabel:
    family_marginal: Mapping[str, str]
    joint_structural: str
    temporal_prefix: str
    temporal_adjacent: str
    causal_binding: str
    fixed_support: str
    h_sup: int | None
    formal_pool_admitted: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class TrainStructuralSupportCatalog:
    dataset_manifest_sha256: str
    joint: frozenset[Signature]
    family_marginal: tuple[frozenset[str], frozenset[str], frozenset[str]]
    temporal_prefixes: tuple[frozenset[tuple[Signature, ...]], ...]
    adjacent_pairs: frozenset[tuple[Signature, Signature]]
    comm_start_width_pairs: frozenset[tuple[int, int]]

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainStructuralSupportCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != "PI-JWM-STEP-6.3B-TRAIN-STRUCTURAL-SUPPORT-V1":
            raise ValueError("wrong TRAIN support catalog schema")
        if raw.get("source") != "FORMAL_TRAIN_ONLY" or raw.get("validation_used") is not False:
            raise ValueError("support catalog must be frozen from Formal TRAIN only")
        joint = frozenset(tuple(row["signature"]) for row in raw["joint_structural_signatures"])
        if any(len(signature) != 3 for signature in joint):
            raise ValueError("joint signature must contain Comm, Comp, Mobility")
        temporal = tuple(
            frozenset(tuple(tuple(signature) for signature in row["sequence"])
                      for row in raw["temporal_structural_prefixes"][str(h)])
            for h in range(1, 5)
        )
        if any(any(len(sequence) != h for sequence in temporal[h - 1]) for h in range(1, 5)):
            raise ValueError("temporal prefix length mismatch")
        marginal = tuple(frozenset(signature[i] for signature in joint) for i in range(3))
        pairs = frozenset((int(row["start"]), int(row["width"]))
                          for row in raw["comm_start_width_pairs"])
        adjacent = frozenset((sequence[i], sequence[i + 1])
                             for horizon in temporal[1:] for sequence in horizon
                             for i in range(len(sequence) - 1))
        return cls(str(raw["dataset_manifest_sha256"]), joint, marginal, temporal,
                   adjacent, pairs)

    def label(self, signature: Signature, prior: Sequence[Signature] = (), *,
              causal_binding: str = "VALID", fixed_support: str = "PENDING_ROLLOUT",
              h_sup: int | None = None) -> CandidateSupportLabel:
        """Label a bound step. Temporal history is diagnostic, not an admission gate."""
        if len(signature) != 3 or len(prior) >= 4:
            raise ValueError("support label requires a 3-family H1-H4 signature")
        family = {name: ("TRAIN_OBSERVED" if signature[i] in self.family_marginal[i] else "TRAIN_UNSEEN")
                  for i, name in enumerate(("Comm", "Comp", "Mob"))}
        joint = "TRAIN_OBSERVED_STRUCTURAL" if signature in self.joint else "TRAIN_UNSEEN_STRUCTURAL"
        prefix = tuple(prior) + (signature,)
        temporal_prefix = ("TRAIN_OBSERVED_PREFIX" if prefix in self.temporal_prefixes[len(prior)]
                           else "TRAIN_UNSEEN_PREFIX")
        adjacent = ("NOT_APPLICABLE_H1" if not prior else
                    "TRAIN_OBSERVED_ADJACENT" if (prior[-1], signature) in self.adjacent_pairs
                    else "TRAIN_UNSEEN_ADJACENT")
        reasons = []
        if any(value == "TRAIN_UNSEEN" for value in family.values()):
            reasons.append("OUTSIDE_TRAIN_FAMILY_MARGINAL")
        if joint != "TRAIN_OBSERVED_STRUCTURAL":
            reasons.append("JOINT_STRUCTURE_UNSEEN_IN_TRAIN")
        if causal_binding != "VALID":
            reasons.append("CAUSAL_BINDING_UNRESOLVED")
        if fixed_support not in {"PENDING_ROLLOUT", "SUPPORTED"}:
            reasons.append("FIXED_SUPPORT_BOUNDARY")
        return CandidateSupportLabel(family, joint, temporal_prefix, adjacent,
                                     causal_binding, fixed_support, h_sup,
                                     not reasons, tuple(reasons))
