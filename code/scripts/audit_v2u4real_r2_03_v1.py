"""Read-only R2-03 audit for the pinned V2U4Real sample subset.

The audit deliberately keeps filename keys separate from physical timestamps
and does not execute a coordinate transform.  It reuses the project's strict
PCD/YAML readers and records file-level and decoded-point-cloud identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from pi_jwm.v2u4real_interface_v1 import UniqueKeyLoader, read_pcd_xyz, sha256


DATASET_VERSION = "36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a"
SCENE_ID = "2025-07-18-12-53_2"
SELECTED_KEYS = [f"1752814590.{i}" for i in range(2, 10)] + [
    f"175281459{i}.0" for i in range(1, 2)
] + [f"1752814591.{i}" for i in range(1, 8)]


def _json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(value):
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, (int, float)):
        return {"type": type(value).__name__, "finite": bool(math.isfinite(value))}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "element_types": sorted({type(v).__name__ for v in value})}
    return {"type": type(value).__name__}


def _time_fields(value, path=""):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if re.search(r"(?:^|_)(?:time|timestamp|stamp)(?:$|_)", str(key), re.I):
                found[child] = {
                    "python_type": type(item).__name__,
                    "value": item if isinstance(item, (str, int, float, bool, type(None))) else repr(item),
                }
            found.update(_time_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_time_fields(item, f"{path}[{index}]"))
    return found


def _metadata(path: Path):
    raw = path.read_text(encoding="utf-8-sig")
    value = yaml.load(raw, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("YAML root is not a mapping")
    pose = value.get("lidar_pose")
    if isinstance(pose, list):
        pose_finite = all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in pose)
    else:
        pose_finite = False
    vehicles = value.get("vehicles", {})
    if not isinstance(vehicles, dict):
        vehicles = {}
    semantic = {
        "lidar_pose": pose,
        "ego_speed": value.get("ego_speed"),
        "vehicles": vehicles,
    }
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "root_keys": list(value.keys()),
        "raw_time_fields": _time_fields(value),
        "lidar_pose": {
            "type": type(pose).__name__,
            "length": len(pose) if isinstance(pose, list) else None,
            "finite": pose_finite,
            "values": pose,
        },
        "ego_speed": _summary(value.get("ego_speed")),
        "vehicle_count": len(vehicles),
        "vehicle_ids": sorted(str(k) for k in vehicles)[:32],
        "semantic_sha256": _json_hash(semantic),
    }


def _pcd(path: Path):
    xyz, header = read_pcd_xyz(path)
    xyz = np.ascontiguousarray(xyz)
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "point_count": int(xyz.shape[0]),
        "shape": list(xyz.shape),
        "xyz_finite": bool(np.isfinite(xyz).all()),
        "xyz_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
        "header": {key: value for key, value in header.items() if key in {"FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT", "POINTS", "DATA"}},
        "_xyz": xyz,
    }


def _rel(path: Path, root: Path):
    return path.resolve().relative_to(root.resolve()).as_posix()


def run(data_root: Path):
    scene_root = data_root / "train" / SCENE_ID
    project_root = Path(__file__).resolve().parents[2]
    pairs = []
    pcd_records = []
    yaml_records = []
    errors = []
    for key in SELECTED_KEYS:
        row = {
            "frame_key": key,
            "vehicle_frame_id": key,
            "uav_frame_id": key,
            "pairing_basis": "same published YAML stem; physical acquisition pairing unverified",
            "published_key_numeric_delta_to_previous": None,
        }
        if pairs:
            row["published_key_numeric_delta_to_previous"] = round(float(key) - float(pairs[-1]["frame_key"]), 10)
        for agent in ("1", "2"):
            pcd_path = scene_root / agent / "ouster" / f"{key}.pcd"
            yaml_path = scene_root / agent / "yaml" / "ouster" / f"{key}.yaml"
            side = "vehicle" if agent == "1" else "uav"
            side_row = {"pcd_path": _rel(pcd_path, project_root), "yaml_path": _rel(yaml_path, project_root)}
            try:
                if not pcd_path.is_file() or not yaml_path.is_file():
                    raise FileNotFoundError("required PCD/YAML missing")
                pcd_row = _pcd(pcd_path)
                xyz = pcd_row.pop("_xyz")
                meta_row = _metadata(yaml_path)
                side_row.update({"pcd": pcd_row, "yaml": meta_row, "read_status": "PASS"})
                pcd_records.append({"path": _rel(pcd_path, project_root), "sha256": pcd_row["sha256"], "xyz_sha256": pcd_row["xyz_sha256"], "xyz": xyz})
                yaml_records.append({"path": _rel(yaml_path, project_root), "sha256": meta_row["sha256"], "semantic_sha256": meta_row["semantic_sha256"]})
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                side_row.update({"read_status": "BLOCKED", "error": str(exc)})
                errors.append({"agent": agent, "frame_key": key, "error": str(exc)})
            row[side] = side_row
        pairs.append(row)

    grouped = defaultdict(list)
    for record in pcd_records:
        grouped[record["sha256"]].append(record)
    duplicate_groups = []
    for file_hash, records in grouped.items():
        if len(records) < 2:
            continue
        comparison = []
        first = records[0]["xyz"]
        for record in records[1:]:
            other = record["xyz"]
            comparison.append({
                "path": record["path"],
                "byte_identical": True,
                "decoded_xyz_identical": bool(first.shape == other.shape and np.array_equal(first, other)),
                "shape": list(other.shape),
                "max_abs_xyz_difference": float(np.max(np.abs(first - other))) if first.shape == other.shape else None,
            })
        duplicate_groups.append({
            "sha256": file_hash,
            "paths": [record["path"] for record in records],
            "comparison_to_first": comparison,
        })

    yaml_groups = defaultdict(list)
    semantic_groups = defaultdict(list)
    for record in yaml_records:
        yaml_groups[record["sha256"]].append(record["path"])
        semantic_groups[record["semantic_sha256"]].append(record["path"])
    repeated_yaml = [paths for paths in yaml_groups.values() if len(paths) > 1]
    repeated_semantics = [paths for paths in semantic_groups.values() if len(paths) > 1]
    for record in pcd_records:
        record.pop("xyz", None)

    physical_timestamp_recovered = False
    physical_pairing_verified = False
    return {
        "schema_version": "V2U4Real-R2-03-v1",
        "dataset": "VJiaLi/V2U4Real",
        "dataset_version": DATASET_VERSION,
        "split": "train",
        "scene_id": SCENE_ID,
        "official_code_commit": "e099a3a746c6092a614e18b74d1f5b2f3c8351e8",
        "paper": {"arxiv": "2603.25275v1", "url": "https://arxiv.org/abs/2603.25275"},
        "scope": {
            "selected_pair_count": len(pairs),
            "selected_keys": SELECTED_KEYS,
            "coordinate_transform_executed": False,
            "physical_timestamp_recovered": physical_timestamp_recovered,
            "physical_pairing_verified": physical_pairing_verified,
        },
        "pairs": pairs,
        "duplicate_pcd_groups": duplicate_groups,
        "repeated_yaml_byte_groups": repeated_yaml,
        "repeated_yaml_semantic_groups": repeated_semantics,
        "errors": errors,
        "status": "BLOCKED"
        if errors
        or not pairs
        or duplicate_groups
        or not physical_timestamp_recovered
        or not physical_pairing_verified
        else "PASS",
        "status_reason": "Files and decoded point clouds were audited, but the selected samples have no independently verified physical timestamps or cross-agent pairing; duplicate vehicle payloads also require explanation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("code/artifacts/data/v2u4real_r2_01/selected_samples"))
    parser.add_argument("--output", type=Path, default=Path("code/artifacts/manifests/v2u4real_r2_03_20261026/real_data_contract_audit.json"))
    args = parser.parse_args()
    result = run(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_pair_count": result["scope"]["selected_pair_count"],
        "read_errors": len(result["errors"]),
        "duplicate_pcd_groups": len(result["duplicate_pcd_groups"]),
        "repeated_yaml_byte_groups": len(result["repeated_yaml_byte_groups"]),
        "output": str(args.output),
    }, ensure_ascii=True))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
