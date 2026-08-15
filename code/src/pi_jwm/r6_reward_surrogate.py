"""Evidence gates for an action-conditioned R6 reward surrogate.

The factual AirFogSim trajectories can supervise rewards for the behavior action.
They cannot, by themselves, supervise the five non-default candidate templates.
This module keeps those two claims separate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r6_joint_action import TEMPLATE_ORDER
from .r6_reward_protocol import RewardScale, ServiceFirstRewardProtocol, TransitionFacts


R6_REWARD_SURROGATE_SCHEMA_VERSION = "PIJWM-R6-factual-reward-surrogate-v1"
NONLOCKED_SPLITS = frozenset({"train", "validation", "calibration"})
ACTION_MAGNITUDE_FIELDS = (
    "offload_count",
    "rb_task_count",
    "rb_total",
    "cpu_task_count",
    "cpu_total",
)


def _aligned_time_axis(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> int:
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError(f"missing time-aligned arrays: {missing}")
    lengths = {int(np.asarray(arrays[name]).shape[0]) for name in names}
    if len(lengths) != 1:
        raise ValueError("time-aligned arrays disagree on step count")
    return lengths.pop()


def derive_failure_events(
    task_lifecycle_index: np.ndarray,
    task_present: np.ndarray,
) -> np.ndarray:
    """Count first transitions into AirFogSim's frozen ``failed`` lifecycle."""

    lifecycle = np.asarray(task_lifecycle_index)
    present = np.asarray(task_present, dtype=bool)
    if lifecycle.ndim != 2 or present.shape != lifecycle.shape:
        raise ValueError("task lifecycle and presence must be aligned [step, task]")
    previous = np.full(lifecycle.shape, -1, dtype=lifecycle.dtype)
    previous[1:] = lifecycle[:-1]
    events = present & (lifecycle == 4) & (previous != 4)
    return events.sum(axis=1, dtype=np.int64)


def summarize_factual_actions(
    task_action: np.ndarray,
    task_action_present: np.ndarray,
) -> np.ndarray:
    """Map factual task actions to the five candidate magnitude dimensions."""

    action = np.asarray(task_action, dtype=np.float64)
    present = np.asarray(task_action_present, dtype=bool)
    if action.ndim != 3 or action.shape[-1] != 8:
        raise ValueError("task_action must have shape [step, task, 8]")
    if present.shape != action.shape[:2]:
        raise ValueError("task_action_present must align with task_action")
    masked = np.where(present[..., None], action, 0.0)
    if not np.isfinite(masked).all():
        raise ValueError("observed task actions must be finite")
    if (masked < -1e-9).any():
        raise ValueError("observed task actions must be nonnegative")
    return np.stack(
        (
            (masked[..., 0] > 0.0).sum(axis=1),
            (masked[..., 1] > 0.0).sum(axis=1),
            masked[..., 3].sum(axis=1),
            (masked[..., 5] > 0.0).sum(axis=1),
            masked[..., 6].sum(axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def build_factual_reward_arrays(
    teacher_arrays: Mapping[str, np.ndarray],
    system_target_arrays: Mapping[str, np.ndarray],
    *,
    scale: RewardScale,
) -> dict[str, np.ndarray]:
    """Build directly grounded per-step rewards for the observed behavior action."""

    teacher_names = (
        "time",
        "task_lifecycle_index",
        "task_present",
        "task_action",
        "task_action_present",
    )
    target_names = (
        "time",
        "task_on_time_completion_event",
        "completed_task_delay",
        "completed_task_delay_valid",
        "delivered_data_total",
        "uav_energy_delta",
        "uav_energy_valid",
    )
    teacher_steps = _aligned_time_axis(teacher_arrays, teacher_names)
    target_steps = _aligned_time_axis(system_target_arrays, target_names)
    if teacher_steps != target_steps:
        raise ValueError("teacher and system-target step counts disagree")
    teacher_time = np.asarray(teacher_arrays["time"], dtype=np.float64)
    target_time = np.asarray(system_target_arrays["time"], dtype=np.float64)
    if teacher_time.shape != target_time.shape or not np.allclose(
        teacher_time, target_time, rtol=0.0, atol=1e-6
    ):
        raise ValueError("teacher and system-target time grids disagree")

    on_time = np.asarray(
        system_target_arrays["task_on_time_completion_event"], dtype=bool
    )
    delay = np.asarray(system_target_arrays["completed_task_delay"], dtype=np.float64)
    delay_valid = np.asarray(
        system_target_arrays["completed_task_delay_valid"], dtype=bool
    )
    energy = np.asarray(system_target_arrays["uav_energy_delta"], dtype=np.float64)
    energy_valid = np.asarray(system_target_arrays["uav_energy_valid"], dtype=bool)
    delivered = np.asarray(
        system_target_arrays["delivered_data_total"], dtype=np.float64
    )
    if on_time.ndim != 2 or delay.shape != on_time.shape or delay_valid.shape != on_time.shape:
        raise ValueError("task outcome arrays must be aligned [step, task]")
    if energy.ndim != 2 or energy_valid.shape != energy.shape:
        raise ValueError("energy outcome arrays must be aligned [step, node]")
    if delivered.shape != (teacher_steps,):
        raise ValueError("delivered_data_total must have shape [step]")

    facts = {
        "on_time_completion_count": on_time.sum(axis=1, dtype=np.int64),
        "failure_count": derive_failure_events(
            teacher_arrays["task_lifecycle_index"], teacher_arrays["task_present"]
        ),
        "completed_delay_sum": np.where(delay_valid, delay, 0.0).sum(axis=1),
        "delivered_data_delta": delivered,
        "energy_delta": np.where(energy_valid, energy, 0.0).sum(axis=1),
    }
    protocol = ServiceFirstRewardProtocol(scale)
    reward = np.empty(teacher_steps, dtype=np.float32)
    for step in range(teacher_steps):
        result = protocol.score(
            TransitionFacts(
                on_time_completion_count=int(facts["on_time_completion_count"][step]),
                failure_count=int(facts["failure_count"][step]),
                completed_delay_sum=float(facts["completed_delay_sum"][step]),
                delivered_data_delta=float(facts["delivered_data_delta"][step]),
                energy_delta=float(facts["energy_delta"][step]),
                hard_violation_count=0,
            )
        )
        if not result.valid or result.total_reward is None:
            raise RuntimeError("factual legal trajectory unexpectedly produced invalid reward")
        reward[step] = float(result.total_reward)

    return {
        "time": teacher_time.astype(np.float32),
        "action_magnitude": summarize_factual_actions(
            teacher_arrays["task_action"], teacher_arrays["task_action_present"]
        ),
        "on_time_completion_count": facts["on_time_completion_count"],
        "failure_count": facts["failure_count"],
        "completed_delay_sum": np.asarray(facts["completed_delay_sum"], dtype=np.float32),
        "delivered_data_delta": np.asarray(facts["delivered_data_delta"], dtype=np.float32),
        "energy_delta": np.asarray(facts["energy_delta"], dtype=np.float32),
        "reward_total": reward,
    }


def audit_template_support(
    *,
    observed_template_ids: Sequence[str],
    minimum_samples_per_template: int,
) -> dict[str, Any]:
    """Gate counterfactual use on observed support for every frozen template."""

    minimum = int(minimum_samples_per_template)
    if minimum <= 0:
        raise ValueError("minimum_samples_per_template must be positive")
    unsupported = sorted(set(observed_template_ids).difference(TEMPLATE_ORDER))
    if unsupported:
        raise ValueError(f"unsupported observed template IDs: {unsupported}")
    counts = Counter(str(value) for value in observed_template_ids)
    missing = [name for name in TEMPLATE_ORDER if counts[name] < minimum]
    covered = len(TEMPLATE_ORDER) - len(missing)
    return {
        "required_templates": list(TEMPLATE_ORDER),
        "required_template_count": len(TEMPLATE_ORDER),
        "minimum_samples_per_template": minimum,
        "observed_counts": {name: int(counts[name]) for name in TEMPLATE_ORDER},
        "covered_template_count": covered,
        "coverage_ratio": covered / len(TEMPLATE_ORDER),
        "missing_templates": missing,
        "candidate_reward_surrogate_ready": not missing,
    }


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_reward_surrogate_preflight(
    *,
    teacher_root: str | Path,
    system_root: str | Path,
    reward_scale_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize factual rewards and issue a strict candidate-support verdict."""

    teacher = Path(teacher_root).resolve()
    system = Path(system_root).resolve()
    scale_path = Path(reward_scale_path).resolve()
    output = Path(output_root).resolve()
    index_path = teacher / "trajectory_index.csv"
    if not index_path.is_file() or not scale_path.is_file():
        raise FileNotFoundError("teacher trajectory index or frozen reward scale is missing")
    scale = RewardScale.from_mapping(_read_json(scale_path))
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    selected = [row for row in index_rows if row.get("v3_status") == "materialized"]
    if not selected:
        raise ValueError("teacher dataset contains no materialized trajectories")
    if any(str(row.get("split")) not in NONLOCKED_SPLITS for row in selected):
        raise ValueError("materialized teacher trajectories must be nonlocked")

    output.mkdir(parents=True, exist_ok=True)
    accumulated: dict[str, list[np.ndarray]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    input_hashes = {
        str(index_path): _sha256(index_path),
        str(scale_path): _sha256(scale_path),
    }
    offset = 0
    cpu_policy_counts: Counter[str] = Counter()
    split_step_counts: Counter[str] = Counter()
    split_trajectory_counts: Counter[str] = Counter()
    for row in sorted(selected, key=lambda item: int(item["seed"])):
        seed = int(row["seed"])
        split = str(row["split"])
        trajectory_id = str(row["trajectory_id"])
        seed_dir_name = str(row.get("v3_seed_dir", ""))
        if not seed_dir_name:
            raise ValueError(f"materialized trajectory lacks v3_seed_dir: {trajectory_id}")
        teacher_path = teacher / seed_dir_name / "trajectory_tensors.npz"
        system_seed = system / f"seed_{seed:03d}"
        system_path = system_seed / "system_targets.npz"
        report_path = system_seed / "system_target_report.json"
        for path in (teacher_path, system_path, report_path):
            if not path.is_file():
                raise FileNotFoundError(f"reward preflight input is missing: {path}")
            input_hashes[str(path)] = _sha256(path)
        report = _read_json(report_path)
        if (
            int(report.get("seed", -1)) != seed
            or str(report.get("split")) != split
            or str(report.get("trajectory_id")) != trajectory_id
        ):
            raise ValueError(f"system target identity mismatch for seed {seed:03d}")
        with np.load(teacher_path, allow_pickle=False) as loaded:
            teacher_arrays = {name: loaded[name] for name in loaded.files}
        with np.load(system_path, allow_pickle=False) as loaded:
            system_arrays = {name: loaded[name] for name in loaded.files}
        factual = build_factual_reward_arrays(teacher_arrays, system_arrays, scale=scale)
        step_count = int(len(factual["time"]))
        for name, value in factual.items():
            accumulated.setdefault(name, []).append(np.asarray(value))
        accumulated.setdefault("seed", []).append(
            np.full(step_count, seed, dtype=np.int32)
        )
        accumulated.setdefault("trajectory_index", []).append(
            np.full(step_count, len(trajectory_rows), dtype=np.int32)
        )
        accumulated.setdefault("step_index", []).append(
            np.arange(step_count, dtype=np.int32)
        )
        accumulated.setdefault("split_index", []).append(
            np.full(
                step_count,
                {"train": 0, "validation": 1, "calibration": 2}[split],
                dtype=np.int8,
            )
        )
        trajectory_rows.append(
            {
                "trajectory_index": len(trajectory_rows),
                "trajectory_id": trajectory_id,
                "seed": seed,
                "split": split,
                "cpu_policy": str(row.get("cpu_policy", "")),
                "array_offset": offset,
                "step_count": step_count,
            }
        )
        offset += step_count
        split_step_counts[split] += step_count
        split_trajectory_counts[split] += 1
        cpu_policy_counts[str(row.get("cpu_policy", ""))] += 1

    concatenated = {
        name: np.concatenate(values, axis=0) for name, values in accumulated.items()
    }
    arrays_path = output / "factual_reward_rows.npz"
    np.savez_compressed(arrays_path, **concatenated)
    index_output_path = output / "trajectory_rows.csv"
    with index_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    # Every logged action is the behavior action preserved by the candidate set's
    # default arm. The five alternative joint templates were never executed here.
    template_support = audit_template_support(
        observed_template_ids=["default"] * offset,
        minimum_samples_per_template=1,
    )
    summary = {
        "schema_version": R6_REWARD_SURROGATE_SCHEMA_VERSION,
        "factual_reward_dataset_ready": True,
        "candidate_reward_surrogate_ready": template_support[
            "candidate_reward_surrogate_ready"
        ],
        "imagined_rollout_training_allowed": False,
        "blocking_reason": "five_nondefault_joint_templates_have_no_counterfactual_reward_labels",
        "locked_test_accessed": False,
        "trajectory_count": len(trajectory_rows),
        "step_count": offset,
        "split_trajectory_counts": dict(sorted(split_trajectory_counts.items())),
        "split_step_counts": dict(sorted(split_step_counts.items())),
        "cpu_policy_trajectory_counts": dict(sorted(cpu_policy_counts.items())),
        "action_magnitude_fields": list(ACTION_MAGNITUDE_FIELDS),
        "template_support": template_support,
        "claim_boundary": (
            "The artifact supports factual behavior-action reward diagnostics only; "
            "it does not support counterfactual ranking of the five unexecuted templates."
        ),
        "input_hashes": input_hashes,
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": R6_REWARD_SURROGATE_SCHEMA_VERSION,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (arrays_path, index_output_path, summary_path)
        },
        "candidate_reward_surrogate_ready": False,
        "locked_test_accessed": False,
    }
    _write_json(output / "manifest.json", manifest)
    return summary


__all__ = [
    "ACTION_MAGNITUDE_FIELDS",
    "R6_REWARD_SURROGATE_SCHEMA_VERSION",
    "audit_template_support",
    "build_reward_surrogate_preflight",
    "build_factual_reward_arrays",
    "derive_failure_events",
    "summarize_factual_actions",
]
