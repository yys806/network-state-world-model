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


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
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

    @property
    def package_paths(self) -> dict[str, Path]:
        packages = self.manifest.get("packages", {})
        required = ("samples", "tensor", "graph", "target", "normalization")
        result: dict[str, Path] = {}
        for name in required:
            value = packages.get(name)
            if not value:
                raise ValueError(f"manifest must declare packages.{name}")
            path = Path(value)
            if not path.is_absolute():
                path = (self.manifest_path.parent / path).resolve()
            result[name] = path
        return result

    @property
    def runtime_package_paths(self) -> dict[str, Path]:
        values = self.manifest.get("runtime_packages")
        if not values:
            return self.package_paths
        result: dict[str, Path] = {}
        for name in ("samples", "tensor", "graph", "target", "normalization"):
            value = values.get(name)
            if not value:
                raise ValueError(f"manifest must declare runtime_packages.{name}")
            path = Path(value)
            result[name] = path if path.is_absolute() else (self.manifest_path.parent / path).resolve()
        return result

    def verify_packages(self) -> dict[str, Any]:
        verified: dict[str, Any] = {}
        for name, path in self.package_paths.items():
            if not path.exists():
                verified[name] = {"exists": False, "sha256": None}
                continue
            expected = self.manifest.get("hashes", {}).get(name)
            actual = sha256_path(path)
            verified[name] = {"exists": True, "sha256": actual, "expected_sha256_declared": isinstance(expected, str) and bool(expected), "hash_matches": isinstance(expected, str) and expected == actual}
        verified["all_present_and_matching"] = all(item.get("exists") and item.get("hash_matches", False) for item in verified.values())
        return verified

    def verify_runtime_packages(self) -> dict[str, Any]:
        verified: dict[str, Any] = {}
        expected_hashes = self.manifest.get("runtime_hashes", {})
        for name, path in self.runtime_package_paths.items():
            if not path.exists():
                verified[name] = {"exists": False, "sha256": None}
                continue
            expected = expected_hashes.get(name)
            actual = sha256_path(path)
            verified[name] = {
                "exists": True,
                "sha256": actual,
                "expected_sha256_declared": isinstance(expected, str) and bool(expected),
                "hash_matches": isinstance(expected, str) and expected == actual,
            }
        verified["all_present_and_matching"] = all(
            item.get("exists") and item.get("hash_matches", False)
            for item in verified.values()
        )
        return verified

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "FormalTrainingInterface":
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sample_path = manifest.get("sample_path") or manifest.get("files", {}).get("samples", {}).get("path") or manifest.get("packages", {}).get("samples")
        if not sample_path:
            raise ValueError("manifest must declare sample_path")
        sample_file = Path(sample_path)
        if not sample_file.is_absolute():
            sample_file = path.parent / sample_file
        if sample_file.is_dir():
            sample_file = sample_file / "index.json"
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
            "package_paths_declared": all(name in self.manifest.get("packages", {}) for name in ("samples", "tensor", "graph", "target", "normalization")),
            "package_hashes_verified": self.verify_packages()["all_present_and_matching"] if "packages" in self.manifest else False,
        }


__all__ = ["FormalTrainingInterface", "sha256_file", "sha256_path"]


REQUIRED_CONFIG_FIELDS = ("dataset", "model", "training", "evaluation", "runtime")


def load_training_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_FIELDS if key not in payload]
    if missing:
        raise ValueError(f"training config missing sections: {missing}")
    return payload


def unresolved_training_fields(config: Mapping[str, Any]) -> list[str]:
    unresolved: list[str] = []
    for section in REQUIRED_CONFIG_FIELDS:
        value = config.get(section, {})
        if isinstance(value, Mapping):
            unresolved.extend(f"{section}.{key}" for key, item in value.items() if item is None)
    return sorted(unresolved)


def validate_readiness(checks: Mapping[str, bool], *, formal_dataset: bool, research_decisions_frozen: bool) -> dict[str, Any]:
    training_checks = ("package_load", "trainer_construct", "action_adapter_support", "cpu_train_step", "prior_only_validation", "checkpoint_reload")
    device_checks = ("model_device", "data_device", "checkpoint_map_location", "cpu_generic_dry_run")
    training_stack = all(bool(checks.get(name, False)) for name in training_checks)
    device_ready = all(bool(checks.get(name, False)) for name in device_checks)
    dataset_ready = bool(formal_dataset and all(bool(checks.get(name, False)) for name in ("dataset_contract", "action_coverage", "split_isolation")))
    return {"training_stack_readiness": "PASS" if training_stack else "FAIL", "formal_dataset_readiness": "READY" if dataset_ready else "NOT_READY", "gpu_codepath_readiness": "PREPARED" if device_ready else "NOT_PREPARED", "formal_training_readiness": "READY" if training_stack and dataset_ready and device_ready and research_decisions_frozen else "BLOCKED", "checks": dict(checks)}


__all__ += ["load_training_config", "unresolved_training_fields", "validate_readiness"]
