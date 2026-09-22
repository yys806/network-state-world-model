"""STEP 5.4 formal-training interface and readiness audits.

This module deliberately separates the manifest-driven formal interface from
the legacy 5.2 development adapter.  It validates contracts and provenance;
it does not create a formal dataset or start training.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_counts(samples: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    counts = {name: {"non_empty_count": 0, "unique_count": 0, "sample_count": 0, "trajectory_count": 0, "signatures": set(), "trajectories": set()} for name in ("route", "comm", "comp", "mobility")}
    for sample in samples:
        metadata = sample.get("metadata", {})
        trajectory = metadata.get("trajectory_id", metadata.get("source_trajectory_id", metadata.get("source_path", "unknown")))
        actions = sample.get("future_action", [])
        for frame in actions if isinstance(actions, list) else []:
            for name, value in frame.items() if isinstance(frame, Mapping) else []:
                key = "mobility" if name == "mob" else name
                if key not in counts or not isinstance(value, Mapping):
                    continue
                entries = value.get("entries") or []
                counts[key]["sample_count"] += 1
                if entries:
                    signature = json.dumps(entries, sort_keys=True, separators=(",", ":"))
                    counts[key]["non_empty_count"] += 1
                    counts[key]["signatures"].add(signature)
                    counts[key]["trajectories"].add(str(trajectory))
    result = {}
    for key, value in counts.items():
        result[key] = {"non_empty_count": value["non_empty_count"], "unique_count": len(value["signatures"]), "sample_count": value["sample_count"], "trajectory_count": len(value["trajectories"]), "variation_across_sample_or_trajectory": len(value["signatures"]) > 1 or len(value["trajectories"]) > 1}
    return result


@dataclass(frozen=True)
class FormalTrainingInterface:
    """Manifest-driven package boundary; no development-size assumptions."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    samples: list[Mapping[str, Any]]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    dataset_manifest_hash: str

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "FormalTrainingInterface":
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sample_path = manifest.get("sample_path") or manifest.get("files", {}).get("samples", {}).get("path")
        if not sample_path:
            raise ValueError("manifest must declare sample_path")
        sample_file = Path(sample_path)
        if not sample_file.is_absolute():
            sample_file = path.parent.parent.parent / sample_file
        samples = json.loads(sample_file.read_text(encoding="utf-8"))
        if not isinstance(samples, list) or not samples:
            raise ValueError("manifest samples must be a non-empty list")
        train_names = {"train", "dev_train"}
        validation_names = {"validation", "dev_validation"}
        train = tuple(i for i, s in enumerate(samples) if s.get("metadata", {}).get("split") in train_names)
        validation = tuple(i for i, s in enumerate(samples) if s.get("metadata", {}).get("split") in validation_names)
        if not train or not validation:
            raise ValueError("manifest must provide trajectory-level train and validation splits")
        return cls(path, manifest, samples, train, validation, sha256_file(path))

    def action_coverage(self) -> dict[str, dict[str, Any]]:
        return _action_counts(self.samples)

    def contract_audit(self) -> dict[str, Any]:
        contract = self.manifest.get("contract", {})
        return {
            "real_causal_trajectory": bool(contract.get("real_causal_trajectory", False)),
            "trajectory_level_split": bool(contract.get("trajectory_level_split", False)),
            "train_only_normalization": bool(contract.get("train_only_normalization", False)),
            "future_target_excluded_from_input": bool(contract.get("future_target_excluded_from_input", False)),
            "stable_id_index_presence_mask": bool(contract.get("stable_id_index_presence_mask", False)),
            "motion_future_target": bool(contract.get("motion_future_target", False)),
            "wireless_per_rb_csi_target": bool(contract.get("wireless_per_rb_csi_target", False)),
            "flow_semantics": bool(contract.get("flow_semantics", False)),
            "component_unsupported_mask": bool(contract.get("component_unsupported_mask", False)),
            "deterministic_rebuild": bool(contract.get("deterministic_rebuild", False)),
            "manifest_hash": self.dataset_manifest_hash,
            "provenance_present": bool(self.manifest.get("provenance")),
        }


__all__ = ["FormalTrainingInterface", "sha256_file"]
