"""Versioned transition reward and train-only scale protocol for PI-JWM R6."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


R6_REWARD_PROTOCOL_VERSION = "PIJWM-R6-service-first-reward-v1"
R6_REWARD_SCALE_VERSION = "PIJWM-R6-train-only-reward-scale-v1"
NONLOCKED_SPLITS = frozenset({"train", "validation", "calibration"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _positive_finite(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _nonnegative_finite(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be nonnegative and finite")
    return result


def _nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class RewardScale:
    completed_delay_p95: float
    delivered_data_p95: float
    energy_delta_p95: float
    train_trajectory_count: int
    delay_step_count: int
    throughput_step_count: int
    energy_step_count: int
    source_manifest_sha256: str
    schema_version: str = R6_REWARD_SCALE_VERSION
    quantile_method: str = "numpy_linear"

    def __post_init__(self) -> None:
        for field in (
            "completed_delay_p95",
            "delivered_data_p95",
            "energy_delta_p95",
        ):
            object.__setattr__(self, field, _positive_finite(getattr(self, field), field=field))
        for field in (
            "train_trajectory_count",
            "delay_step_count",
            "throughput_step_count",
            "energy_step_count",
        ):
            value = _nonnegative_int(getattr(self, field), field=field)
            if value <= 0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        digest = str(self.source_manifest_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")
        if str(self.schema_version) != R6_REWARD_SCALE_VERSION:
            raise ValueError("unsupported reward scale schema")
        if str(self.quantile_method) != "numpy_linear":
            raise ValueError("unsupported reward scale quantile method")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RewardScale":
        return cls(**dict(value))


@dataclass(frozen=True)
class TransitionFacts:
    on_time_completion_count: int
    failure_count: int
    completed_delay_sum: float
    delivered_data_delta: float
    energy_delta: float
    hard_violation_count: int = 0

    def __post_init__(self) -> None:
        for field in ("on_time_completion_count", "failure_count", "hard_violation_count"):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field=field))
        for field in ("completed_delay_sum", "delivered_data_delta", "energy_delta"):
            object.__setattr__(
                self,
                field,
                _nonnegative_finite(getattr(self, field), field=field),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardBreakdown:
    protocol_version: str
    valid: bool
    invalid_reason: str | None
    raw_facts: Mapping[str, Any]
    normalized_components: Mapping[str, float]
    weighted_components: Mapping[str, float]
    total_reward: float | None

    def __post_init__(self) -> None:
        if bool(self.valid) != (self.total_reward is not None):
            raise ValueError("valid reward must have a total and invalid reward must not")
        if self.total_reward is not None and not math.isfinite(float(self.total_reward)):
            raise ValueError("total_reward must be finite")
        for namespace in (self.normalized_components, self.weighted_components):
            if any(not math.isfinite(float(value)) for value in namespace.values()):
                raise ValueError("reward components must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ServiceFirstRewardProtocol:
    """Primary service events dominate bounded delay/throughput/energy terms."""

    def __init__(self, scale: RewardScale) -> None:
        self.scale = scale

    def score(self, facts: TransitionFacts) -> RewardBreakdown:
        normalized = {
            "on_time_completion": float(facts.on_time_completion_count),
            "failure": float(facts.failure_count),
            "delay": min(float(facts.completed_delay_sum) / self.scale.completed_delay_p95, 1.0),
            "throughput": min(float(facts.delivered_data_delta) / self.scale.delivered_data_p95, 1.0),
            "energy": min(float(facts.energy_delta) / self.scale.energy_delta_p95, 1.0),
        }
        weighted = {
            "on_time_completion": normalized["on_time_completion"],
            "failure": -normalized["failure"],
            "delay": -0.1 * normalized["delay"],
            "throughput": 0.1 * normalized["throughput"],
            "energy": -0.1 * normalized["energy"],
        }
        if facts.hard_violation_count > 0:
            return RewardBreakdown(
                protocol_version=R6_REWARD_PROTOCOL_VERSION,
                valid=False,
                invalid_reason="hard_constraint_violation",
                raw_facts=facts.to_dict(),
                normalized_components=normalized,
                weighted_components=weighted,
                total_reward=None,
            )
        total = float(sum(weighted.values()))
        return RewardBreakdown(
            protocol_version=R6_REWARD_PROTOCOL_VERSION,
            valid=True,
            invalid_reason=None,
            raw_facts=facts.to_dict(),
            normalized_components=normalized,
            weighted_components=weighted,
            total_reward=total,
        )


def _positive(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat) & (flat > 0.0)]


def _step_delay_sum(npz: Mapping[str, np.ndarray]) -> np.ndarray:
    delay = np.asarray(npz["completed_task_delay"], dtype=np.float64)
    valid = np.asarray(npz["completed_task_delay_valid"], dtype=bool)
    if delay.shape != valid.shape or delay.ndim != 2:
        raise ValueError("completed task delay arrays must be aligned [step, task]")
    return np.where(valid, delay, 0.0).sum(axis=1)


def _step_delivered_data(npz: Mapping[str, np.ndarray]) -> np.ndarray:
    """Read the legacy-named field whose verified semantics are per-step delivery."""

    delivered = np.asarray(npz["delivered_data_total"], dtype=np.float64)
    if delivered.ndim != 1 or not np.isfinite(delivered).all():
        raise ValueError("delivered_data_total must be a finite [step] array")
    if (delivered < -1e-9).any():
        raise ValueError("per-step delivered_data_total must be nonnegative")
    return np.maximum(delivered, 0.0)


def _step_energy_sum(npz: Mapping[str, np.ndarray]) -> np.ndarray:
    energy = np.asarray(npz["uav_energy_delta"], dtype=np.float64)
    valid = np.asarray(npz["uav_energy_valid"], dtype=bool)
    if energy.shape != valid.shape or energy.ndim != 2:
        raise ValueError("UAV energy arrays must be aligned [step, node]")
    if (energy[valid] < -1e-9).any():
        raise ValueError("uav_energy_delta must be nonnegative")
    return np.where(valid, np.maximum(energy, 0.0), 0.0).sum(axis=1)


def freeze_train_reward_scale(dataset_root: str | Path) -> RewardScale:
    """Derive P95 reward scales from train trajectories only and audit every split."""

    root = Path(dataset_root).resolve()
    summary_path = root / "dataset_summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("formal system target summary or manifest is missing")
    summary = _read_json(summary_path)
    if summary.get("system_targets_ready") is not True:
        raise ValueError("formal system targets are not ready")
    if summary.get("locked_test_accessed") is not False:
        raise ValueError("formal system target dataset accessed locked_test")

    delays: list[np.ndarray] = []
    throughputs: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    train_count = 0
    directories = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not directories:
        raise ValueError("formal system target dataset has no trajectories")
    for directory in directories:
        report_path = directory / "system_target_report.json"
        arrays_path = directory / "system_targets.npz"
        if not report_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"incomplete system target trajectory: {directory}")
        report = _read_json(report_path)
        split = str(report.get("split", ""))
        if split == "locked_test":
            raise ValueError(f"locked_test trajectory is forbidden in reward scaling: {directory.name}")
        if split not in NONLOCKED_SPLITS:
            raise ValueError(f"unsupported system target split: {split}")
        if split != "train":
            continue
        train_count += 1
        with np.load(arrays_path, allow_pickle=False) as arrays:
            delays.append(_positive(_step_delay_sum(arrays)))
            throughputs.append(_positive(_step_delivered_data(arrays)))
            energies.append(_positive(_step_energy_sum(arrays)))

    samples = {
        "delay": np.concatenate(delays) if delays else np.empty(0, dtype=np.float64),
        "throughput": np.concatenate(throughputs) if throughputs else np.empty(0, dtype=np.float64),
        "energy": np.concatenate(energies) if energies else np.empty(0, dtype=np.float64),
    }
    missing = [name for name, value in samples.items() if value.size == 0]
    if train_count <= 0 or missing:
        raise ValueError(f"positive train reward support is missing: {missing}")
    return RewardScale(
        completed_delay_p95=float(np.quantile(samples["delay"], 0.95, method="linear")),
        delivered_data_p95=float(np.quantile(samples["throughput"], 0.95, method="linear")),
        energy_delta_p95=float(np.quantile(samples["energy"], 0.95, method="linear")),
        train_trajectory_count=train_count,
        delay_step_count=int(samples["delay"].size),
        throughput_step_count=int(samples["throughput"].size),
        energy_step_count=int(samples["energy"].size),
        source_manifest_sha256=_sha256(manifest_path),
    )
