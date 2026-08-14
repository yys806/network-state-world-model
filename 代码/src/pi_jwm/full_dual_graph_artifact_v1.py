"""Trajectory validation, replay comparison, and atomic v4 publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping, Sequence


ARTIFACT_CONTRACT_VERSION = "PIJWM-Full-Dual-Graph-Artifact-v1"
REPLAY_ABS_TOL = 1e-9
REPLAY_REL_TOL = 1e-7
REQUIRED_ARTIFACT_FILES = (
    "collector_config.json",
    "vocabularies.json",
    "frames.jsonl",
    "coverage_report.json",
    "validation_report.json",
    "replay_report.json",
    "status_flags.json",
    "manifest.json",
)
E1_FIELD_NAMES = (
    "channel_attenuation_mean_db",
    "channel_attenuation_std_db",
    "prev_active_flow_count",
    "prev_effective_rate_per_s",
    "prev_served_data",
)


_ACTIONABLE = {
    "waiting_to_offload",
    "offloading",
    "waiting_to_return",
    "returning",
}


def _phase(snapshot: Mapping[str, object] | None) -> object:
    return snapshot.get("phase") if isinstance(snapshot, Mapping) else None


def _physical_width(frame: Mapping[str, object], vocabulary_width: int, errors: list[str]) -> None:
    for snapshot_name in (
        "decision_snapshot",
        "execution_snapshot",
        "outcome_snapshot",
    ):
        snapshot = frame.get(snapshot_name)
        if not isinstance(snapshot, Mapping):
            continue
        presence = snapshot.get("physical_edge_presence")
        if not isinstance(presence, list) or len(presence) != vocabulary_width:
            errors.append(
                f"frame {frame.get('frame_index')}: {snapshot_name} physical-edge presence width mismatch"
            )


def _validate_cep(frame: Mapping[str, object], errors: list[str]) -> None:
    decision_snapshot = frame.get("decision_snapshot", {})
    action = frame.get("action", {})
    if not isinstance(decision_snapshot, Mapping) or not isinstance(action, Mapping):
        return
    edge_rows = decision_snapshot.get("physical_edges", [])
    edge_by_id = {
        str(row.get("edge_id")): row
        for row in edge_rows
        if isinstance(row, Mapping)
    }
    for hop in action.get("hops", []):
        if not isinstance(hop, Mapping):
            continue
        physical = edge_by_id.get(str(hop.get("physical_edge_id")))
        if physical is None:
            continue
        if (
            physical.get("source_id"),
            physical.get("target_id"),
        ) != (hop.get("source_id"), hop.get("target_id")):
            errors.append(
                f"frame {frame.get('frame_index')}: CEP endpoints mismatch"
            )


def _masked(value: float | None, *, valid: bool, reason: str | None) -> dict[str, object]:
    return {"value": value, "valid_mask": valid, "missing_reason": reason}


def build_e1_rows(
    *,
    decision_snapshot: Mapping[str, object],
    previous_transfer_rows: Sequence[Mapping[str, object]] | None,
    physical_edge_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Build the five real E1 fields; never manufacture legacy-width values."""

    channel_by_edge = {
        str(row.get("physical_edge_id")): row
        for row in decision_snapshot.get("channel_rows", [])
        if isinstance(row, Mapping)
    }
    edge_type_by_id = {
        str(row.get("edge_id")): str(row.get("edge_type"))
        for row in decision_snapshot.get("physical_edges", [])
        if isinstance(row, Mapping)
    }
    prior_by_edge: dict[str, list[Mapping[str, object]]] = {}
    if previous_transfer_rows is not None:
        for row in previous_transfer_rows:
            prior_by_edge.setdefault(str(row.get("physical_edge_id")), []).append(row)
    result: list[dict[str, object]] = []
    for edge_id in physical_edge_ids:
        channel = channel_by_edge.get(edge_id)
        values = channel.get("channel_attenuation_db", []) if channel else []
        valid_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        if valid_values and len(valid_values) == len(values):
            mean = sum(valid_values) / len(valid_values)
            variance = sum((value - mean) ** 2 for value in valid_values) / len(valid_values)
            attenuation_mean = _masked(mean, valid=True, reason=None)
            attenuation_std = _masked(math.sqrt(variance), valid=True, reason=None)
        else:
            reason = (
                "NOT_APPLICABLE_WIRED"
                if edge_type_by_id.get(edge_id) == "wired"
                else "CURRENT_CSI_UNAVAILABLE"
            )
            attenuation_mean = _masked(None, valid=False, reason=reason)
            attenuation_std = _masked(None, valid=False, reason=reason)
        if previous_transfer_rows is None:
            active = rate = served = _masked(None, valid=False, reason="NO_HISTORY")
        else:
            observed = [
                row for row in prior_by_edge.get(edge_id, []) if row.get("observed_mask") is True
            ]
            active = _masked(
                float(len({str(row.get("flow_id")) for row in observed})),
                valid=True,
                reason=None,
            )
            rate = _masked(
                sum(float(row.get("rate_per_s", 0.0)) for row in observed),
                valid=True,
                reason=None,
            )
            served = _masked(
                sum(float(row.get("delivered_data", 0.0)) for row in observed),
                valid=True,
                reason=None,
            )
        result.append(
            {
                "physical_edge_id": str(edge_id),
                "fields": {
                    "channel_attenuation_mean_db": attenuation_mean,
                    "channel_attenuation_std_db": attenuation_std,
                    "prev_active_flow_count": active,
                    "prev_effective_rate_per_s": rate,
                    "prev_served_data": served,
                },
            }
        )
    return result


def _validate_history(
    frame: Mapping[str, object], frame_index: int, errors: list[str]
) -> None:
    rows = frame.get("e1_rows")
    if not isinstance(rows, list) or not rows:
        errors.append(f"frame {frame_index}: E1 rows missing")
        return
    for row in rows:
        if not isinstance(row, Mapping) or set(row.get("fields", {})) != set(E1_FIELD_NAMES):
            errors.append(f"frame {frame_index}: E1 must contain exactly five named fields")
            continue
        fields = row["fields"]
        for field_name in E1_FIELD_NAMES:
            field = fields[field_name]
            if not isinstance(field, Mapping):
                errors.append(f"frame {frame_index}: invalid E1 field {field_name}")
                continue
            value, valid, reason = field.get("value"), field.get("valid_mask"), field.get("missing_reason")
            if valid is True:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or reason is not None
                ):
                    errors.append(f"frame {frame_index}: valid E1 field {field_name} is malformed")
            elif value is not None or not isinstance(reason, str) or not reason:
                errors.append(f"frame {frame_index}: masked E1 field {field_name} is malformed")
        if frame_index == 0:
            for field_name in E1_FIELD_NAMES[2:]:
                field = fields[field_name]
                if field.get("valid_mask") is not False or field.get("missing_reason") != "NO_HISTORY":
                    errors.append("first-frame E1 history fields must be masked NO_HISTORY")


def validate_trajectory_frames(
    frames: Sequence[Mapping[str, object]],
    *,
    vocabulary: Mapping[str, object],
    fixture: bool,
) -> list[str]:
    """Return deterministic trajectory errors without modifying the inputs."""

    errors: list[str] = []
    frame_rows = tuple(frames)
    expected_indices = list(range(len(frame_rows)))
    observed_indices = [row.get("frame_index") for row in frame_rows]
    if observed_indices != expected_indices:
        errors.append("frame indices must be contiguous from zero")
    physical_indices = vocabulary.get("physical_edge_indices", {})
    if not isinstance(physical_indices, Mapping):
        return [*errors, "physical_edge_indices vocabulary is invalid"]
    vocabulary_width = len(physical_indices)
    for ordinal, frame in enumerate(frame_rows):
        frame_index = frame.get("frame_index", ordinal)
        if frame.get("fixture") is not fixture:
            errors.append(f"frame {frame_index}: fixture/natural trajectory mismatch")
        if frame.get("quarantined") is True:
            errors.append(f"frame {frame_index}: quarantined frame cannot enter validated trajectory")
        if _phase(frame.get("decision_snapshot")) != "decision":
            errors.append(f"frame {frame_index}: decision snapshot phase mismatch")
        if _phase(frame.get("execution_snapshot")) != "execution":
            errors.append(f"frame {frame_index}: execution snapshot phase mismatch")
        if _phase(frame.get("outcome_snapshot")) != "outcome":
            errors.append(f"frame {frame_index}: outcome snapshot phase mismatch")
        source_phases = frame.get("decision_input_source_phases", {})
        if isinstance(source_phases, Mapping) and any(
            source_phase == "outcome" for source_phase in source_phases.values()
        ):
            errors.append(f"frame {frame_index}: decision input has same-frame outcome source")

        decision_snapshot = frame.get("decision_snapshot", {})
        action = frame.get("action", {})
        tasks = (
            decision_snapshot.get("tasks", [])
            if isinstance(decision_snapshot, Mapping)
            else []
        )
        decisions = action.get("decisions", []) if isinstance(action, Mapping) else []
        actionable = {
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("lifecycle") in _ACTIONABLE
        }
        decision_ids = [
            str(decision.get("task_id"))
            for decision in decisions
            if isinstance(decision, Mapping)
        ]
        missing = sorted(actionable - set(decision_ids))
        if missing:
            errors.append(f"frame {frame_index}: missing decision rows for {missing}")
        if len(decision_ids) != len(set(decision_ids)):
            errors.append(f"frame {frame_index}: duplicate task decision row")
        _physical_width(frame, vocabulary_width, errors)
        _validate_cep(frame, errors)
        _validate_history(frame, ordinal, errors)
    return errors


_EXACT_KEY_TOKENS = (
    "id",
    "index",
    "indices",
    "action",
    "decision",
    "flow",
    "hop",
    "route",
    "rb",
    "vocab",
    "phase",
    "lifecycle",
    "mask",
    "reason",
)


def _requires_exact(path: tuple[object, ...]) -> bool:
    return any(
        any(token in str(component).lower() for token in _EXACT_KEY_TOKENS)
        for component in path
        if isinstance(component, str)
    )


def compare_replays(
    reference: Sequence[Mapping[str, object]],
    replay: Sequence[Mapping[str, object]],
    *,
    abs_tol: float = REPLAY_ABS_TOL,
    rel_tol: float = REPLAY_REL_TOL,
) -> dict[str, object]:
    """Compare identity/action values exactly and report every float delta."""

    exact_mismatches: list[dict[str, object]] = []
    numeric_differences: list[dict[str, object]] = []

    def walk(left: object, right: object, path: tuple[object, ...]) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                exact_mismatches.append(
                    {"path": list(path), "reference_keys": sorted(left), "replay_keys": sorted(right)}
                )
                return
            for key in sorted(left):
                walk(left[key], right[key], (*path, key))
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                exact_mismatches.append(
                    {"path": list(path), "reference_length": len(left), "replay_length": len(right)}
                )
                return
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                walk(left_value, right_value, (*path, index))
            return
        numeric = (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        )
        if numeric and (isinstance(left, float) or isinstance(right, float)):
            left_float, right_float = float(left), float(right)
            if left_float != right_float:
                close = math.isclose(
                    left_float, right_float, rel_tol=rel_tol, abs_tol=abs_tol
                )
                numeric_differences.append(
                    {
                        "path": list(path),
                        "reference": left_float,
                        "replay": right_float,
                        "absolute_difference": abs(left_float - right_float),
                        "within_tolerance": close,
                    }
                )
                if _requires_exact(path) or not close:
                    exact_mismatches.append(
                        {"path": list(path), "reference": left, "replay": right}
                    )
            return
        if type(left) is not type(right) or left != right:
            exact_mismatches.append(
                {"path": list(path), "reference": left, "replay": right}
            )

    walk(reference, replay, ())
    return {
        "passed": not exact_mismatches,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "exact_mismatches": exact_mismatches,
        "numeric_differences": numeric_differences,
    }


def build_full_collector_status_flags(*, passed: bool) -> dict[str, bool]:
    return {
        "v4_collector_implemented": bool(passed),
        "v4_dataset_complete": False,
        "training_eligible": False,
        "model_training_started": False,
        "gpu_started": False,
        "locked_test_accessed": False,
        "candidate_rollout_planner_complete": False,
        "final_method_frozen": False,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: object) -> None:
    if not isinstance(rows, (list, tuple)):
        raise TypeError("frames.jsonl payload must be a sequence")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _portable_source_key(path: Path, common_root: Path) -> str:
    try:
        return path.absolute().relative_to(common_root).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def publish_atomic_bundle(
    output_dir: Path,
    payloads: Mapping[str, object],
    source_paths: Sequence[Path],
) -> None:
    """Publish a complete immutable bundle only after content verification."""

    output_dir = Path(output_dir)
    expected_payloads = set(REQUIRED_ARTIFACT_FILES) - {"manifest.json"}
    if set(payloads) != expected_payloads:
        raise ValueError(
            f"payload names differ: {sorted(set(payloads) ^ expected_payloads)}"
        )
    validation = payloads["validation_report.json"]
    replay = payloads["replay_report.json"]
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise ValueError("validation report did not pass")
    if not isinstance(replay, Mapping) or replay.get("passed") is not True:
        raise ValueError("replay report did not pass")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {output_dir}")
    # Preserve the project-logical path through a junction.  ``resolve()`` would
    # rewrite the AirFogSim junction to the main checkout and make otherwise
    # project-relative manifest keys depend on the temporary worktree name.
    sources = tuple(Path(path).absolute() for path in source_paths)
    if not sources:
        raise ValueError("source_paths must not be empty")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        for name in sorted(expected_payloads):
            destination = temporary / name
            if name == "frames.jsonl":
                _write_jsonl(destination, payloads[name])
            else:
                _write_json(destination, payloads[name])
        artifact_hashes = {
            name: _file_hash(temporary / name) for name in sorted(expected_payloads)
        }
        common_root = Path(os.path.commonpath([str(path) for path in sources]))
        if common_root.is_file():
            common_root = common_root.parent
        source_hashes = {
            _portable_source_key(path, common_root): _file_hash(path)
            for path in sorted(sources, key=lambda item: str(item))
        }
        manifest = {
            "schema_version": ARTIFACT_CONTRACT_VERSION,
            "required_files": list(REQUIRED_ARTIFACT_FILES),
            "artifact_hashes": artifact_hashes,
            "source_hashes": source_hashes,
            "status_flags": payloads["status_flags.json"],
        }
        _write_json(temporary / "manifest.json", manifest)
        for name, expected_hash in artifact_hashes.items():
            if _file_hash(temporary / name) != expected_hash:
                raise RuntimeError(f"artifact hash changed before publication: {name}")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
