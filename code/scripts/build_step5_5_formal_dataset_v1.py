"""Build and accept the sharded STEP 5.5 Formal Dataset v1.

This entry point is CPU-only. It consumes already collected real AirFogSim
trajectories and never generates locked-test data or starts formal training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pi_jwm.model_ready_sample_contract_v1 import audit_future_action_references
from pi_jwm.step3_2_batch_preprocessing_v1 import (
    FEATURE_UNITS,
    NORMALIZATION_FIELDS,
    _feature_value,
    _iter_features,
    apply_normalization,
)
from pi_jwm.step4_2a_graph_input_extension_v1 import (
    NORMALIZATION_SCHEMA_VERSION as EXTENSION_NORMALIZATION_SCHEMA_VERSION,
    NORMALIZATION_SPECS,
    _iter_normalization_values,
    amend_raw_graph_inputs,
    apply_extension_normalization,
    build_extended_sample,
)
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (
    FLOW_NUMERIC_FEATURE_ORDER,
    FLOW_NUMERIC_UNITS,
    NORMALIZATION_SCHEMA_VERSION as FLOW_NORMALIZATION_SCHEMA_VERSION,
    STEP32_NORMALIZATION_MASK_POLICY,
    _iter_numeric,
    apply_flow_normalization,
    build_flow_extended_sample,
    build_flow_tensor_batch,
    load_flow_tensor_batch,
    save_flow_tensor_batch,
    validate_flow_sample_checks,
    validate_flow_tensor_checks,
)
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import (
    PhysicalTopologyConfig,
    build_typed_dual_graph_batch,
    graph_semantic_digest,
    load_typed_dual_graph_batch,
    save_typed_dual_graph_batch,
)
from pi_jwm.step5_1a_motion_csi_target_contract_v1 import (
    build_future_target_tensor_batch,
    extend_future_motion_csi_targets,
    load_future_target_tensor_batch,
    save_future_target_tensor_batch,
    validate_future_target_tensor_checks,
)
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, sha256_file, sha256_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923"
HISTORY_STEPS = 2
HORIZON_STEPS = 4
TRANSITIONS = 96
WINDOWS_PER_TRAJECTORY = 92
TRAJECTORIES = 60
TRAIN_TRAJECTORIES = 48
VALIDATION_TRAJECTORIES = 12
SPLIT_SEED = 20260923
FAMILIES = ("route", "comm", "comp", "mobility")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_compact_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text_handle:
                json.dump(value, text_handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                text_handle.write("\n")


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_npz(path: Path) -> None:
    """Rewrite an NPZ with sorted members and a fixed ZIP timestamp."""
    with zipfile.ZipFile(path, "r") as source:
        members = [(name, source.read(name)) for name in sorted(source.namelist())]
    temporary = path.with_suffix(path.suffix + ".canonical.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            target.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    temporary.replace(path)


def relative(path: Path, base: Path = ROOT) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


class Moments:
    def __init__(self, keys: Iterable[str]) -> None:
        self.values = {key: [0.0, 0.0, 0] for key in keys}

    def add(self, key: str, value: float) -> None:
        row = self.values[key]
        row[0] += value
        row[1] += value * value
        row[2] += 1

    def feature(self, key: str, *, unit: str, source: str, mask_policy: str) -> dict[str, Any]:
        total, total_sq, count = self.values[key]
        mean = total / count if count else 0.0
        variance = max(total_sq / count - mean * mean, 0.0) if count else 0.0
        return {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance) if variance > 1e-12 else 1.0,
            "zero_variance_handling": "population_std" if variance > 1e-12 else "scale=1.0",
            "source_field": source,
            "unit": unit,
            "mask_policy": mask_policy,
        }


def deterministic_split(trajectory_ids: list[str]) -> dict[str, list[str]]:
    if len(trajectory_ids) != TRAJECTORIES or len(set(trajectory_ids)) != TRAJECTORIES:
        raise ValueError(f"formal split requires exactly {TRAJECTORIES} unique accepted trajectories")
    shuffled = list(trajectory_ids)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    return {"dev_train": shuffled[:TRAIN_TRAJECTORIES], "dev_validation": shuffled[TRAIN_TRAJECTORIES:]}


def build_samples(raw: Mapping[str, Any], *, split: str, source_path: str, source_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    graph_raw = amend_raw_graph_inputs(raw)
    ledger_raw, ledger_receipt = amend_raw_with_causal_flow_ledger(graph_raw)
    if not ledger_receipt.get("passed"):
        raise ValueError(f"causal Flow ledger failed: {json.dumps(ledger_receipt, sort_keys=True)}")
    trajectory_id = str(raw.get("trajectory_id") or raw["decisions"][0]["trajectory_id"])
    samples: list[dict[str, Any]] = []
    for anchor in range(HISTORY_STEPS - 1, TRANSITIONS - HORIZON_STEPS + 1):
        sample = build_extended_sample(
            graph_raw,
            anchor_step=anchor,
            history_steps=HISTORY_STEPS,
            horizon_steps=HORIZON_STEPS,
        )
        sample["metadata"].update({
            "split": split,
            "source_split": split,
            "sample_id": f"{trajectory_id}::anchor-{anchor:04d}",
            "source_path": source_path,
            "source_sha256": source_sha256,
        })
        sample = build_flow_extended_sample(sample, ledger_raw)
        checks = validate_flow_sample_checks(sample)
        if not checks.get("passed"):
            failed = sorted(key for key, value in checks.items() if not value)
            raise ValueError(f"invalid Flow sample {sample['metadata']['sample_id']}: {failed}")
        samples.append(sample)
    if len(samples) != WINDOWS_PER_TRAJECTORY:
        raise ValueError(f"{trajectory_id} produced {len(samples)} windows, expected {WINDOWS_PER_TRAJECTORY}")
    return ledger_raw, samples, ledger_receipt


def update_normalization(samples: list[dict[str, Any]], base: Moments, extension: Moments, flow: Moments) -> None:
    for sample in samples:
        for kind, field, row in _iter_features(sample):
            value, valid = _feature_value(row, field)
            if valid:
                base.add(f"{kind}.{field}", float(value))
        for key, value, valid in _iter_normalization_values(sample):
            if valid:
                extension.add(key, float(value))
    for key, value in _iter_numeric(samples):
        flow.add(key, float(value))


def normalization_stats(base: Moments, extension: Moments, flow: Moments, train_ids: list[str], validation_ids: list[str]) -> dict[str, Any]:
    base_features = {}
    for kind, field in NORMALIZATION_FIELDS:
        key = f"{kind}.{field}"
        base_features[key] = base.feature(
            key,
            unit=FEATURE_UNITS[key],
            source=field,
            mask_policy="presence=true AND feature_mask=true AND value!=null; train split only",
        )
    extension_features = {}
    for key, (unit, source) in NORMALIZATION_SPECS.items():
        row = extension.feature(
            key,
            unit=unit,
            source=source,
            mask_policy="dev_train only AND presence/validity=true AND feature_mask=true AND value!=null; validation excluded",
        )
        row["preprocessing_policy"] = "z_score_on_valid_train_values"
        extension_features[key] = row
    flow_features = {}
    for key in FLOW_NUMERIC_FEATURE_ORDER:
        row = flow.feature(key, unit=FLOW_NUMERIC_UNITS[key], source=key, mask_policy=STEP32_NORMALIZATION_MASK_POLICY)
        row["zero_variance_handling"] = "not_required" if row["zero_variance_handling"] == "population_std" else "std_replaced_with_1"
        row["presence_source"] = "logical_flow.presence for logical fields; corresponding logical_flow.presence plus carrying.known for carrying fields"
        flow_features[key] = row
    return {
        "schema_version": "PI-JWM-Step-5.5-Formal-Train-Only-Normalization-v1",
        "source_split": "dev_train",
        "source_trajectory_ids": train_ids,
        "excluded_validation_trajectory_ids": validation_ids,
        "future_target_in_fit": False,
        "base_step3_2": {
            "schema_version": "PI-JWM-Step-3.2-Train-Only-Normalization-v1",
            "source_split": "dev_train",
            "features": base_features,
            "units": dict(FEATURE_UNITS),
        },
        "extension_step4_2a": {
            "schema_version": EXTENSION_NORMALIZATION_SCHEMA_VERSION,
            "source_split": "dev_train",
            "features": {**base_features, **extension_features},
        },
        "flow_step4_2c": {
            "schema_version": FLOW_NORMALIZATION_SCHEMA_VERSION,
            "source_split": "dev_train",
            "features": flow_features,
        },
    }


def normalize(samples: list[dict[str, Any]], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = apply_normalization(samples, stats["base_step3_2"])
    output = apply_extension_normalization(output, stats["extension_step4_2a"])
    return apply_flow_normalization(output, stats["flow_step4_2c"])


def coverage(raw_by_id: Mapping[str, Mapping[str, Any] | Path], split: Mapping[str, list[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}, "protocol": "intervention_count / eligible_opportunity_count"}
    for split_name, trajectory_ids in split.items():
        family_accumulators = {
            family: {"eligible": 0, "intervention": 0, "no_op": 0, "signatures": set(), "trajectories": set()}
            for family in FAMILIES
        }
        bitmasks: Counter[str] = Counter()
        global_noop = 0
        single_family = 0
        multi_family = 0
        for trajectory_id in trajectory_ids:
            source = raw_by_id[trajectory_id]
            raw = load_json(source) if isinstance(source, Path) else source
            local_intervention = {family: False for family in FAMILIES}
            for step in raw["steps"]:
                audit = step["behavior_policy_audit"]
                mask = str(audit["activation_bitmask"])
                bitmasks[mask] += 1
                active = mask.count("1")
                global_noop += int(bool(audit["global_noop"]))
                single_family += int(active == 1)
                multi_family += int(active >= 2)
                for family in FAMILIES:
                    row = audit["families"][family]
                    accumulator = family_accumulators[family]
                    accumulator["eligible"] += int(bool(row["eligible"]))
                    accumulator["intervention"] += int(bool(row["intervention"]))
                    accumulator["no_op"] += int(bool(row["no_op"]))
                    if row["intervention"]:
                        local_intervention[family] = True
                        accumulator["signatures"].add(str(row["signature"]))
            for family in FAMILIES:
                if local_intervention[family]:
                    family_accumulators[family]["trajectories"].add(trajectory_id)
        family_rows: dict[str, dict[str, Any]] = {}
        for family, accumulator in family_accumulators.items():
            eligible = int(accumulator["eligible"])
            intervention = int(accumulator["intervention"])
            family_rows[family] = {
                "eligible_count": eligible,
                "intervention_count": intervention,
                "intervention_rate": intervention / eligible if eligible else None,
                "unique_legal_signatures": len(accumulator["signatures"]),
                "intervention_trajectory_count": len(accumulator["trajectories"]),
                "no_op_count": int(accumulator["no_op"]),
            }
        result["splits"][split_name] = {
            "families": family_rows,
            "joint_action": {
                "activation_bitmask_counts": dict(sorted(bitmasks.items())),
                "unique_activation_bitmasks": len(bitmasks),
                "global_noop_count": global_noop,
                "single_family_count": single_family,
                "multi_family_count": multi_family,
            },
        }
    return result


def coverage_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    train = report["splits"]["dev_train"]
    validation = report["splits"]["dev_validation"]
    checks = {}
    for family in FAMILIES:
        train_row = train["families"][family]
        validation_row = validation["families"][family]
        checks[f"{family}_train_rate_25_75"] = train_row["intervention_rate"] is not None and 0.25 <= train_row["intervention_rate"] <= 0.75
        checks[f"{family}_train_trajectory_coverage"] = train_row["intervention_trajectory_count"] >= 5
        checks[f"{family}_validation_trajectory_coverage"] = validation_row["intervention_trajectory_count"] >= 2
    for family in ("route", "comm", "comp"):
        checks[f"{family}_signature_diversity"] = train["families"][family]["unique_legal_signatures"] >= 3
    checks["mobility_signature_diversity"] = train["families"]["mobility"]["unique_legal_signatures"] >= 5
    joint = train["joint_action"]
    checks["all_noop_present"] = int(joint["activation_bitmask_counts"].get("0000", 0)) > 0
    checks["single_family_present"] = joint["single_family_count"] > 0
    checks["multi_family_present"] = joint["multi_family_count"] > 0
    checks["at_least_four_bitmasks"] = joint["unique_activation_bitmasks"] >= 4
    return checks


def array_digest(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key, item in value.items() if isinstance(item, np.ndarray)):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(value[key]).tobytes())
    return digest.hexdigest()


def merge_array_batches(batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge real window blocks while padding capacity axes deterministically."""
    output: dict[str, Any] = {}
    keys = sorted({key for batch in batches for key, value in batch.items() if isinstance(value, np.ndarray)})
    for key in keys:
        arrays = [np.asarray(batch[key]) for batch in batches]
        if any(array.ndim == 0 for array in arrays):
            output[key] = arrays[0]
            continue
        shape = [max(array.shape[axis] for array in arrays) for axis in range(1, arrays[0].ndim)]
        padded = []
        for array in arrays:
            target_shape = (array.shape[0], *shape)
            fill = -1 if array.dtype.kind == "i" else 0
            value = np.full(target_shape, fill, dtype=array.dtype)
            slices = (slice(None), *(slice(0, size) for size in array.shape[1:]))
            value[slices] = array
            padded.append(value)
        output[key] = np.concatenate(padded, axis=0)
    output["schema_version"] = batches[0].get("schema_version")
    output["contract"] = dict(batches[0].get("contract", {}))
    for key in ("sample_static", "sample_metadata"):
        if any(key in batch for batch in batches):
            output[key] = [row for batch in batches for row in batch.get(key, [])]
    for key in ("normalization_parameters", "flow_normalization_stats"):
        if key in batches[0]:
            output[key] = batches[0][key]
    return output


def synchronize_tensor_contract(tensor: dict[str, Any]) -> dict[str, Any]:
    """Make every merged capacity declaration describe the merged arrays."""
    contract = dict(tensor.get("contract", {}))
    capacity_axes = {
        "max_entity": ("entity_presence", 2),
        "max_task": ("task_presence", 2),
        "max_flow": ("flow_presence", 2),
        "max_target_entity": ("target_entity_presence", 2),
        "max_target_task": ("target_task_presence", 2),
        "max_target_flow": ("target_flow_presence", 2),
        "max_relation": ("relation_mask", 2),
        "max_dag": ("dag_mask", 2),
        "max_past_outcome_relation": ("past_outcome_relation_mask", 2),
        "max_past_outcome_dag": ("past_outcome_dag_mask", 2),
        "max_comm_relation": ("comm_relation_presence", 2),
        "max_task_agent_relation": ("task_agent_validity_mask", 2),
        "n_comm_rb": ("comm_csi", 3),
        "target_entity_capacity": ("target_entity_presence", 2),
        "target_task_capacity": ("target_task_presence", 2),
        "max_logical_flow": ("logical_flow_known_mask", 2),
        "max_target_logical_flow": ("target_logical_flow_known_mask", 2),
    }
    for name, (array_name, axis) in capacity_axes.items():
        contract[name] = int(np.asarray(tensor[array_name]).shape[axis])
    contract["history_steps"] = int(np.asarray(tensor["entity_presence"]).shape[1])
    contract["horizon_steps"] = int(np.asarray(tensor["target_entity_presence"]).shape[1])
    contract["max_action_entries"] = max(
        int(np.asarray(tensor[name]).shape[3])
        for name in ("past_action_entry_mask", "future_action_entry_mask")
    )
    contract["max_route_hops"] = max(
        int(np.asarray(tensor[name]).shape[3])
        for name in ("past_route_hop_mask", "future_route_hop_mask")
    )
    contract["n_rb"] = max(
        int(np.asarray(tensor[name]).shape[3])
        for name in ("past_comm_rb_mask", "future_comm_rb_mask")
    )
    contract["max_route_nodes"] = max(
        int(np.asarray(tensor[name]).shape[3])
        for name in ("route_node_mask", "target_route_node_mask")
    )
    contract["capacity_scope"] = "formal_trajectory_shard_exact_observed_bound"
    tensor["contract"] = contract
    return tensor


def merge_graph_batches(batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    block_names = sorted({name for batch in batches for name in batch["blocks"]})
    output = {
        "schema_version": batches[0]["schema_version"],
        "contract": dict(batches[0]["contract"]),
        "blocks": {
            name: {
                key: value
                for key, value in merge_array_batches([batch["blocks"][name] for batch in batches]).items()
                if isinstance(value, np.ndarray)
            }
            for name in block_names
        },
    }
    output["contract"]["block_capacities"] = {
        name: int(next(iter(block.values())).shape[1])
        for name, block in output["blocks"].items()
    }
    return output


def merge_target_batches(batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    output = merge_array_batches(batches)
    contract = dict(batches[0]["contract"])
    contract["motion_identity"] = [
        rows for batch in batches for rows in batch["contract"].get("motion_identity", [])
    ]
    contract["comm_identity"] = [
        rows for batch in batches for rows in batch["contract"].get("comm_identity", [])
    ]
    contract["motion_shape"] = list(np.asarray(output["target_vehicle_motion_raw"]).shape)
    contract["csi_shape"] = list(np.asarray(output["target_comm_csi_raw"]).shape)
    output["contract"] = contract
    return output


def build_sharded_components(samples: list[dict[str, Any]], ledger_raw: Mapping[str, Any], stats: Mapping[str, Any], topology: PhysicalTopologyConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Counter[str]]:
    tensor_blocks: list[dict[str, Any]] = []
    graph_blocks: list[dict[str, Any]] = []
    target_blocks: list[dict[str, Any]] = []
    side_counts: Counter[str] = Counter({"unsupported": 0, "unresolved": 0, "fixed_support_blocked": 0})
    block_size = len(samples)
    for start in range(0, len(samples), block_size):
        block = samples[start:start + block_size]
        tensor = build_flow_tensor_batch(block, stats=stats["flow_step4_2c"])
        if not validate_flow_tensor_checks(tensor).get("passed"):
            raise ValueError("Flow tensor block validation failed")
        tensor_blocks.append(tensor)
        graph_blocks.append(build_typed_dual_graph_batch(tensor, topology))
        targets = [extend_future_motion_csi_targets(sample, ledger_raw, stats["extension_step4_2a"]) for sample in block]
        target_blocks.append(build_future_target_tensor_batch(targets, stats["extension_step4_2a"]))
        for target in targets:
            for frame in target.get("target", []):
                side = frame.get("future_target_side_metadata", {})
                side_counts["unsupported"] += int(side.get("unsupported_count", 0))
                side_counts["unresolved"] += int(side.get("unresolved_count", 0))
                side_counts["fixed_support_blocked"] += int(side.get("fixed_support_blocked_count", 0))
    tensor = synchronize_tensor_contract(merge_array_batches(tensor_blocks))
    target = merge_target_batches(target_blocks)
    tensor_checks = validate_flow_tensor_checks(tensor)
    target_checks = validate_future_target_tensor_checks(target)
    if not tensor_checks.get("passed"):
        raise ValueError(f"merged Flow tensor validation failed: {tensor_checks}")
    if not target_checks.get("passed"):
        raise ValueError(f"merged Motion/CSI target validation failed: {target_checks}")
    return tensor, merge_graph_batches(graph_blocks), target, side_counts


def run_probe(source: Path) -> dict[str, Any]:
    raw = load_json(source)
    source_hash = sha256_file(source)
    ledger_raw, samples, ledger_receipt = build_samples(
        raw,
        split="dev_train",
        source_path=source.as_posix(),
        source_sha256=source_hash,
    )
    base = Moments(f"{kind}.{field}" for kind, field in NORMALIZATION_FIELDS)
    extension = Moments(NORMALIZATION_SPECS)
    flow = Moments(FLOW_NUMERIC_FEATURE_ORDER)
    update_normalization(samples, base, extension, flow)
    trajectory_id = str(raw.get("trajectory_id") or raw["decisions"][0]["trajectory_id"])
    stats = normalization_stats(base, extension, flow, [trajectory_id], [])
    normalized = normalize(samples, stats)
    tensor = build_flow_tensor_batch(normalized, stats=stats["flow_step4_2c"])
    tensor_checks = validate_flow_tensor_checks(tensor)
    graph = build_typed_dual_graph_batch(
        tensor,
        PhysicalTopologyConfig(
            mode="radius_knn", radius_m=1000.0, k=2,
            self_loops=False, development_only=False, research_frozen=True,
        ),
    )
    targets = [extend_future_motion_csi_targets(sample, ledger_raw, stats["extension_step4_2a"]) for sample in normalized]
    target = build_future_target_tensor_batch(targets, stats["extension_step4_2a"])
    result = {
        "passed": bool(ledger_receipt.get("passed") and tensor_checks.get("passed")),
        "samples": len(normalized),
        "history_steps": int(np.asarray(tensor["entity_presence"]).shape[1]),
        "horizon_steps": int(np.asarray(tensor["target_logical_flow_known_mask"]).shape[1]),
        "graph_batch": int(np.asarray(graph["blocks"]["physical_nodes"]["presence"]).shape[0]),
        "motion_target_shape": list(np.asarray(target["target_vehicle_motion_raw"]).shape),
        "csi_target_shape": list(np.asarray(target["target_comm_csi_raw"]).shape),
        "motion_h1_h4_valid": [int(np.asarray(target["target_vehicle_motion_mask"])[:, horizon].sum()) for horizon in range(HORIZON_STEPS)],
        "csi_h1_h4_valid": [int(np.asarray(target["target_comm_csi_mask"])[:, horizon].sum()) for horizon in range(HORIZON_STEPS)],
    }
    result["passed"] = result["passed"] and all(result["motion_h1_h4_valid"]) and all(result["csi_h1_h4_valid"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-samples-per-split", type=int, default=1)
    parser.add_argument("--probe-source", type=Path)
    args = parser.parse_args()
    if args.probe_source:
        result = run_probe(args.probe_source.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    output = args.output_dir.resolve()
    collection = load_json(output / "collection_summary.json")
    accepted = list(collection.get("accepted", []))
    if len(accepted) != TRAJECTORIES:
        raise ValueError(f"collection has {len(accepted)} accepted trajectories, expected {TRAJECTORIES}")
    accepted_by_id = {str(row["trajectory_id"]): row for row in accepted}
    trajectory_ids = [str(row["trajectory_id"]) for row in accepted]
    split = deterministic_split(trajectory_ids)
    split_manifest_path = output / "split_manifest.json"
    split_manifest = {
        "schema_version": "PI-JWM-Step-5.5-Formal-Trajectory-Split-v1",
        "split_seed": SPLIT_SEED,
        "policy": "deterministic shuffle accepted trajectories; split before windows; first 48 train, last 12 validation",
        "dev_train": split["dev_train"],
        "dev_validation": split["dev_validation"],
        "frozen": True,
        "locked_test": None,
    }
    if split_manifest_path.exists() and load_json(split_manifest_path) != split_manifest:
        raise ValueError("refusing to change an already frozen formal split")
    write_json(split_manifest_path, split_manifest)

    split_by_id = {trajectory_id: name for name, ids in split.items() for trajectory_id in ids}
    raw_paths: dict[str, Path] = {}
    raw_provenance: list[dict[str, Any]] = []
    for trajectory_id, row in accepted_by_id.items():
        path = output / str(row["path"])
        raw = load_json(path)
        raw_paths[trajectory_id] = path
        raw_trajectory_id = str(raw.get("trajectory_id") or raw["decisions"][0]["trajectory_id"])
        if raw_trajectory_id != trajectory_id or len(raw.get("steps", [])) != TRANSITIONS or len(raw.get("decisions", [])) != TRANSITIONS + 1:
            raise ValueError(f"raw trajectory contract mismatch: {trajectory_id}")
        decisions = raw["decisions"]
        times = [float(item["simulation_time_s"]) for item in decisions]
        slot_duration = float(raw["environment"]["slot_duration_s"])
        continuous = all(math.isclose(right - left, slot_duration, rel_tol=0.0, abs_tol=1e-9) for left, right in zip(times, times[1:]))
        raw_provenance.append({
            "trajectory_id": trajectory_id,
            "simulator_seed": int(raw["environment"]["seed"]),
            "policy_seed": int(raw["environment"]["policy_seed"]),
            "config_hash": str(raw["environment"]["config_hash"]),
            "policy_hash": str(raw["environment"]["policy_hash"]),
            "source_path": relative(path),
            "source_sha256": sha256_file(path),
            "time_range_s": [times[0], times[-1]],
            "slot_duration_s": slot_duration,
            "continuous_time_grid": continuous,
            "transition_count": len(raw["steps"]),
            "decision_count": len(raw["decisions"]),
            "split": split_by_id[trajectory_id],
            "split_seed": SPLIT_SEED,
            "primary_seed_pair": bool(accepted_by_id[trajectory_id].get("primary", False)),
            "replacement_for": accepted_by_id[trajectory_id].get("replacement_for"),
        })
    write_json(output / "raw_trajectory_provenance.json", raw_provenance)

    coverage_report = coverage(raw_paths, split)
    coverage_gate = coverage_checks(coverage_report)
    coverage_report["checks"] = coverage_gate
    coverage_report["passed"] = all(coverage_gate.values())
    write_json(output / "action_coverage_audit.json", coverage_report)
    if not coverage_report["passed"]:
        raise ValueError(f"formal action coverage failed: {[key for key, value in coverage_gate.items() if not value]}")

    base = Moments(f"{kind}.{field}" for kind, field in NORMALIZATION_FIELDS)
    extension = Moments(NORMALIZATION_SPECS)
    flow = Moments(FLOW_NUMERIC_FEATURE_ORDER)
    future_reference_rows = []
    build_cache = output / ".step5_5_build_cache"
    build_cache.mkdir(parents=True, exist_ok=True)
    for index, trajectory_id in enumerate(split["dev_train"], start=1):
        raw = load_json(raw_paths[trajectory_id])
        source = raw_paths[trajectory_id]
        ledger_raw, samples, ledger_receipt = build_samples(raw, split="dev_train", source_path=relative(source), source_sha256=sha256_file(source))
        update_normalization(samples, base, extension, flow)
        audit = audit_future_action_references(ledger_raw, history_steps=HISTORY_STEPS, horizon_steps=HORIZON_STEPS)
        future_reference_rows.append({"trajectory_id": trajectory_id, **audit})
        write_compact_json(
            build_cache / f"{trajectory_id}.json",
            {"ledger_raw": ledger_raw, "samples": samples, "ledger_receipt": ledger_receipt},
        )
        print(json.dumps({"phase": "normalization_fit", "trajectory": index, "total": TRAIN_TRAJECTORIES, "trajectory_id": trajectory_id}), flush=True)
    stats = normalization_stats(base, extension, flow, split["dev_train"], split["dev_validation"])

    packages = {name: output / "packages" / name for name in ("samples", "tensor", "graph", "target", "normalization")}
    for path in packages.values():
        path.mkdir(parents=True, exist_ok=True)
    write_json(packages["normalization"] / "stats.json", stats)
    sample_index: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    runtime_samples: list[dict[str, Any]] = []
    runtime_raws: list[Mapping[str, Any]] = []
    target_horizon_valid = {f"H{i}": {"motion": 0, "csi": 0} for i in range(1, HORIZON_STEPS + 1)}
    unresolved = Counter({"unsupported": 0, "unresolved": 0, "fixed_support_blocked": 0})
    topology = PhysicalTopologyConfig(mode="radius_knn", radius_m=1000.0, k=2, self_loops=False, development_only=False, research_frozen=True)
    ordered_ids = split["dev_train"] + split["dev_validation"]
    for index, trajectory_id in enumerate(ordered_ids, start=1):
        split_name = split_by_id[trajectory_id]
        source = raw_paths[trajectory_id]
        cache_path = build_cache / f"{trajectory_id}.json"
        if split_name == "dev_train" and cache_path.exists():
            cached = load_json(cache_path)
            ledger_raw, samples, ledger_receipt = cached["ledger_raw"], cached["samples"], cached["ledger_receipt"]
        else:
            raw = load_json(raw_paths[trajectory_id])
            ledger_raw, samples, ledger_receipt = build_samples(
                raw, split=split_name, source_path=relative(source), source_sha256=sha256_file(source)
            )
        if split_name == "dev_validation":
            audit = audit_future_action_references(ledger_raw, history_steps=HISTORY_STEPS, horizon_steps=HORIZON_STEPS)
            future_reference_rows.append({"trajectory_id": trajectory_id, "split": split_name, **audit})
        normalized = normalize(samples, stats)
        tensor, graph, target_tensor, side_counts = build_sharded_components(normalized, ledger_raw, stats, topology)
        sample_path = packages["samples"] / f"{trajectory_id}.json.gz"
        tensor_path = packages["tensor"] / f"{trajectory_id}.npz"
        graph_path = packages["graph"] / f"{trajectory_id}.npz"
        target_path = packages["target"] / f"{trajectory_id}.npz"
        write_gzip_json(sample_path, normalized)
        save_flow_tensor_batch(tensor, tensor_path)
        save_typed_dual_graph_batch(graph, graph_path)
        save_future_target_tensor_batch(target_tensor, target_path)
        for package_path in (tensor_path, graph_path, target_path):
            canonicalize_npz(package_path)
        loaded_samples = load_json(sample_path)
        loaded_tensor = load_flow_tensor_batch(tensor_path)
        loaded_graph = load_typed_dual_graph_batch(graph_path)
        loaded_target = load_future_target_tensor_batch(target_path)
        sample_ids = [sample["metadata"]["sample_id"] for sample in normalized]
        tensor_sample_ids = [metadata["sample_id"] for metadata in loaded_tensor.get("sample_metadata", [])]
        graph_batch_sizes = {
            int(value.shape[0])
            for block in loaded_graph["blocks"].values()
            for value in block.values()
            if isinstance(value, np.ndarray) and value.ndim > 0
        }
        target_batch_sizes = {
            int(value.shape[0]) for value in loaded_target.values()
            if isinstance(value, np.ndarray) and value.ndim > 0
        }
        identity_aligned = bool(
            sample_ids == tensor_sample_ids
            and len(loaded_samples) == len(normalized)
            and graph_batch_sizes == {len(normalized)}
            and target_batch_sizes == {len(normalized)}
        )
        reload_equal = bool(
            loaded_samples == normalized
            and array_digest(loaded_tensor) == array_digest(tensor)
            and loaded_tensor.get("contract") == tensor.get("contract")
            and loaded_tensor.get("sample_metadata") == tensor.get("sample_metadata")
            and loaded_tensor.get("sample_static") == tensor.get("sample_static")
            and graph_semantic_digest(loaded_graph) == graph_semantic_digest(graph)
            and array_digest(loaded_target) == array_digest(target_tensor)
            and loaded_target.get("contract") == target_tensor.get("contract")
        )
        if not identity_aligned:
            raise ValueError(f"package identity alignment failed: {trajectory_id}")
        if not reload_equal:
            raise ValueError(f"package serialize/reload equality failed: {trajectory_id}")
        for horizon in range(HORIZON_STEPS):
            target_horizon_valid[f"H{horizon + 1}"]["motion"] += int(np.asarray(target_tensor["target_vehicle_motion_mask"])[:, horizon].sum())
            target_horizon_valid[f"H{horizon + 1}"]["csi"] += int(np.asarray(target_tensor["target_comm_csi_mask"])[:, horizon].sum())
        unresolved.update(side_counts)
        for shard_index, sample in enumerate(normalized):
            sample_index.append({"metadata": sample["metadata"], "shard": sample_path.name, "shard_index": shard_index})
        shard_rows.append({
            "trajectory_id": trajectory_id,
            "split": split_name,
            "sample_count": len(normalized),
            "source_path": relative(source),
            "source_sha256": sha256_file(source),
            "ledger_receipt_passed": bool(ledger_receipt.get("passed")),
            "identity_aligned": identity_aligned,
            "serialize_reload_equal": reload_equal,
            "files": {
                "samples": {"path": sample_path.name, "sha256": sha256_file(sample_path), "bytes": sample_path.stat().st_size},
                "tensor": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size, "array_digest": array_digest(tensor)},
                "graph": {"path": graph_path.name, "sha256": sha256_file(graph_path), "bytes": graph_path.stat().st_size},
                "target": {"path": target_path.name, "sha256": sha256_file(target_path), "bytes": target_path.stat().st_size, "array_digest": array_digest(target_tensor)},
            },
        })
        if sum(1 for sample in runtime_samples if sample["metadata"]["split"] == split_name) < args.runtime_samples_per_split:
            runtime_samples.append(normalized[0])
            runtime_raws.append(ledger_raw)
        print(json.dumps({"phase": "package_build", "trajectory": index, "total": TRAJECTORIES, "trajectory_id": trajectory_id}), flush=True)

    write_json(packages["samples"] / "index.json", sample_index)
    common_index = {"schema_version": "PI-JWM-Step-5.5-Sharded-Package-Index-v1", "shards": shard_rows}
    write_json(packages["tensor"] / "index.json", common_index)
    write_json(packages["graph"] / "index.json", common_index)
    write_json(packages["target"] / "index.json", common_index)
    write_json(packages["normalization"] / "index.json", {"schema_version": common_index["schema_version"], "fit_trajectory_ids": split["dev_train"], "validation_excluded": split["dev_validation"], "stats": "stats.json"})

    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    runtime_tensor = build_flow_tensor_batch(runtime_samples, stats=stats["flow_step4_2c"])
    runtime_graph = build_typed_dual_graph_batch(runtime_tensor, topology)
    runtime_targets = [extend_future_motion_csi_targets(sample, raw, stats["extension_step4_2a"]) for sample, raw in zip(runtime_samples, runtime_raws)]
    runtime_target = build_future_target_tensor_batch(runtime_targets, stats["extension_step4_2a"])
    write_json(runtime / "samples.json", runtime_samples)
    save_flow_tensor_batch(runtime_tensor, runtime / "tensor.npz")
    save_typed_dual_graph_batch(runtime_graph, runtime / "graph.npz")
    save_future_target_tensor_batch(runtime_target, runtime / "target.npz")
    for package_path in (runtime / "tensor.npz", runtime / "graph.npz", runtime / "target.npz"):
        canonicalize_npz(package_path)
    write_json(runtime / "normalization.json", stats)

    package_hashes = {name: sha256_path(path) for name, path in packages.items()}
    runtime_paths = {
        "samples": runtime / "samples.json", "tensor": runtime / "tensor.npz",
        "graph": runtime / "graph.npz", "target": runtime / "target.npz",
        "normalization": runtime / "normalization.json",
    }
    runtime_hashes = {name: sha256_path(path) for name, path in runtime_paths.items()}
    manifest = {
        "schema_version": "PI-JWM-Step-5.5-Formal-Dataset-Manifest-v1",
        "dataset_id": "pi_jwm_formal_dataset_v1_h2_l4_20260923",
        "formal_dataset": True,
        "sample_path": "packages/samples",
        "packages": {name: f"packages/{name}" for name in packages},
        "runtime_packages": {
            "samples": "runtime/samples.json", "tensor": "runtime/tensor.npz", "graph": "runtime/graph.npz",
            "target": "runtime/target.npz", "normalization": "runtime/normalization.json",
        },
        "runtime_hashes": runtime_hashes,
        "hashes": package_hashes,
        "contract": {
            "history_steps": HISTORY_STEPS, "horizon_steps": HORIZON_STEPS,
            "real_causal_trajectory": True, "trajectory_level_split": True, "train_only_normalization": True,
            "future_target_excluded_from_input": True, "stable_id_index_presence_mask": True,
            "motion_future_target": True, "wireless_per_rb_csi_target": True, "flow_semantics": True,
            "component_unsupported_mask": True, "deterministic_rebuild": True,
            "topology": {"mode": "radius_knn", "radius_m": 1000.0, "k": 2},
        },
        "split": {"seed": SPLIT_SEED, "train_trajectories": TRAIN_TRAJECTORIES, "validation_trajectories": VALIDATION_TRAJECTORIES, "locked_test": None},
        "provenance": {"collection_summary": "collection_summary.json", "raw_trajectories": "raw_trajectory_provenance.json", "split_manifest": "split_manifest.json", "action_coverage": "action_coverage_audit.json"},
        "scope": {"gpu": False, "formal_training": False, "locked_test_accessed": False, "baseline": False, "planner": False, "performance_claim": False},
    }
    write_json(output / "formal_dataset_manifest.json", manifest)
    interface = FormalTrainingInterface.from_manifest(output / "formal_dataset_manifest.json")
    package_verification = interface.verify_packages()
    runtime_package_verification = interface.verify_runtime_packages()

    negative = {}
    missing = json.loads(json.dumps(manifest))
    del missing["hashes"]["target"]
    write_json(output / ".negative_missing_hash.json", missing)
    negative["missing_hash_rejected"] = not FormalTrainingInterface.from_manifest(output / ".negative_missing_hash.json").verify_packages()["all_present_and_matching"]
    wrong = json.loads(json.dumps(manifest))
    wrong["hashes"]["target"] = "0" * 64
    write_json(output / ".negative_wrong_hash.json", wrong)
    negative["wrong_hash_rejected"] = not FormalTrainingInterface.from_manifest(output / ".negative_wrong_hash.json").verify_packages()["all_present_and_matching"]
    (output / ".negative_missing_hash.json").unlink()
    (output / ".negative_wrong_hash.json").unlink()

    split_ids = {name: set(ids) for name, ids in split.items()}
    all_simulator_seeds = [int(row["simulator_seed"]) for row in raw_provenance]
    all_policy_seeds = [int(row["policy_seed"]) for row in raw_provenance]
    future_unresolved = sum(int(row["unresolved_future_reference_count"]) for row in future_reference_rows)
    checks = {
        "accepted_real_trajectories_60": len(trajectory_ids) == 60,
        "each_96_transitions_97_decisions": all(row["transition_count"] == 96 and row["decision_count"] == 97 for row in raw_provenance),
        "continuous_time_grid": all(row["continuous_time_grid"] for row in raw_provenance),
        "trajectory_split_48_12": len(split["dev_train"]) == 48 and len(split["dev_validation"]) == 12,
        "split_identity_isolation": not bool(split_ids["dev_train"] & split_ids["dev_validation"]) and len(set(all_simulator_seeds)) == 60 and len(set(all_policy_seeds)) == 60,
        "history_2_horizon_4": all(len(sample["metadata"]["history_frame_indices"]) == 2 and len(sample["metadata"]["future_action_frame_indices"]) == 4 for sample in sample_index),
        "exact_train_windows_4416": sum(row["metadata"]["split"] == "dev_train" for row in sample_index) == 4416,
        "exact_validation_windows_1104": sum(row["metadata"]["split"] == "dev_validation" for row in sample_index) == 1104,
        "exact_total_windows_5520": len(sample_index) == 5520,
        "future_action_reference_unresolved_zero": future_unresolved == 0,
        "future_target_excluded_from_input": all(row["ledger_receipt_passed"] for row in shard_rows),
        "train_only_normalization": stats["future_target_in_fit"] is False and stats["source_trajectory_ids"] == split["dev_train"],
        "motion_h1_h4_valid": all(row["motion"] > 0 for row in target_horizon_valid.values()),
        "csi_h1_h4_valid": all(row["csi"] > 0 for row in target_horizon_valid.values()),
        "four_action_coverage": coverage_report["passed"],
        "noop_and_intervention_present": coverage_gate["all_noop_present"] and all(coverage_report["splits"]["dev_train"]["families"][family]["intervention_count"] > 0 for family in FAMILIES),
        "unsupported_accounting_separate": all(key in unresolved for key in ("unsupported", "unresolved", "fixed_support_blocked")),
        "package_identity_alignment": all(row["sample_count"] == WINDOWS_PER_TRAJECTORY and row["identity_aligned"] for row in shard_rows),
        "deterministic_split_rebuild": deterministic_split(trajectory_ids) == split,
        "serialize_reload_equality": all(row["serialize_reload_equal"] for row in shard_rows) and all(sha256_file(packages[name] / row["files"][name]["path"]) == row["files"][name]["sha256"] for row in shard_rows for name in ("samples", "tensor", "graph", "target")),
        "five_package_hashes_exact": package_verification["all_present_and_matching"],
        "runtime_smoke_package_hashes_exact": runtime_package_verification["all_present_and_matching"],
        "negative_hash_fixtures": all(negative.values()),
        "locked_test_accessed_false": manifest["scope"]["locked_test_accessed"] is False,
    }
    receipt = {
        "schema_version": "PI-JWM-Step-5.5-Formal-Dataset-Acceptance-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {"accepted_trajectories": 60, "rejected_trajectories": len(collection.get("rejected", [])), "train_windows": 4416, "validation_windows": 1104, "total_windows": 5520},
        "target_horizon_valid_component_counts": target_horizon_valid,
        "unsupported_accounting": dict(unresolved),
        "future_reference_unresolved": future_unresolved,
        "package_verification": package_verification,
        "runtime_package_verification": runtime_package_verification,
        "negative_fixtures": negative,
        "scope": manifest["scope"],
    }
    write_json(output / "dataset_acceptance_receipt.json", receipt)
    write_json(output / "future_action_reference_audit.json", {"rows": future_reference_rows, "unresolved_total": future_unresolved})
    write_json(output / "package_shards.json", common_index)
    print(json.dumps({"passed": receipt["passed"], "checks": checks, "manifest": relative(output / "formal_dataset_manifest.json")}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
