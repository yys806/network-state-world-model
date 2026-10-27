"""Freeze and audit the official V2U4Real label merge policy on five real frames.

This is an in-memory, read-only reproduction of the fixed converter's merge
loop and annotation conversion. It never writes training labels or source/data.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, UniqueKeyLoader, sha256
from audit_v2u4real_r2_04a_single_frame_geometry_v1 import (
    matrix_stats, official_functions, strict_vector, verify_source,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/audit/v2u4real_r2_04c_20261027"
R2_04B_RESULT = ROOT / "code/artifacts/audit/v2u4real_r2_04b_20261026/result.json"
SOURCE_MANIFEST = ROOT / "code/reference/V2U4Real.SOURCE.json"
OFFICIAL_PATH = ROOT / "code/reference/V2U4Real/opencood/utils/uav_to_single.py"
OFFICIAL_REPOSITORY = "https://github.com/VjiaLi/V2U4Real"
OFFICIAL_COMMIT = "e099a3a746c6092a614e18b74d1f5b2f3c8351e8"
OFFICIAL_SHA = "cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b"
SCENE = "2025-07-18-12-53_2"
DATASET_VERSION = "36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a"
KEYS = ("1752814590.2", "1752814590.4", "1752814590.6", "1752814591.2", "1752814591.6")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf8", newline="\n")


def verify_official_source():
    snapshot = verify_source()
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf8"))
    if manifest.get("repository") != OFFICIAL_REPOSITORY or manifest.get("commit") != OFFICIAL_COMMIT:
        raise InterfaceBlocked("official repository/commit mismatch")
    if manifest.get("files", {}).get("opencood/utils/uav_to_single.py") != OFFICIAL_SHA:
        raise InterfaceBlocked("uav_to_single manifest SHA mismatch")
    actual = sha256(OFFICIAL_PATH)
    if actual != OFFICIAL_SHA:
        raise InterfaceBlocked("fixed uav_to_single source SHA mismatch")
    lines = OFFICIAL_PATH.read_text(encoding="utf8").splitlines()
    start, end = 190, 199
    excerpt = lines[start - 1:end]
    required = [
        "merged_objs = {}",
        "ego_objs = load_annotation_objects(ego_y)",
        "merged_objs.update(ego_objs)",
        "uav_objs = load_annotation_objects(uav_y, transform=transform)",
        "for oid, obj in uav_objs.items():",
        "merged_objs[oid] = obj  # UAV 优先覆盖",
    ]
    if not all(any(pattern in line for line in excerpt) for pattern in required):
        raise InterfaceBlocked("official merge excerpt does not match the frozen behavior")
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "path": OFFICIAL_PATH.relative_to(ROOT).as_posix(),
        "sha256": actual,
        "snapshot_identity": snapshot,
        "source_lines": {"start": start, "end": end, "text": excerpt},
        "merge_semantics": {
            "vehicle_first": "merged_objs.update(ego_objs)",
            "uav_second": "for oid, obj in uav_objs.items(): merged_objs[oid] = obj",
            "same_id_conflict": "UAV overwrites vehicle",
            "key_normalization": "none",
        },
    }


def load_r2_04b_reference():
    if not R2_04B_RESULT.is_file():
        raise InterfaceBlocked("R2-04B result is missing")
    result = json.loads(R2_04B_RESULT.read_text(encoding="utf8"))
    if (result.get("dataset_version") != DATASET_VERSION or result.get("scene") != SCENE
            or result.get("split") != "train" or tuple(result.get("fixed_keys", [])) != KEYS):
        raise InterfaceBlocked("R2-04B fixed identity mismatch")
    frames = {row["frame_key"]: row for row in result["frames"]}
    if set(frames) != set(KEYS):
        raise InterfaceBlocked("R2-04B does not contain exactly the five fixed keys")
    return result, frames


def _read_yaml(path: Path):
    if not path.is_file():
        raise InterfaceBlocked(f"missing YAML: {path}")
    raw_bytes = path.read_bytes()
    value = yaml.load(raw_bytes.decode("utf-8-sig"), Loader=UniqueKeyLoader)
    if not isinstance(value, dict) or not isinstance(value.get("vehicles"), dict):
        raise InterfaceBlocked(f"YAML root/vehicles mapping required: {path}")
    return value, {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw_bytes),
        "sha256": sha256(path),
        "root_field_types": {str(k): type(v).__name__ for k, v in value.items()},
        "object_count": len(value["vehicles"]),
        "object_ids": list(value["vehicles"].keys()),
        "read_status": "PASS",
    }


def _validate_id_keys(mapping, side, key):
    if len(set(mapping)) != len(mapping):
        raise InterfaceBlocked(f"{key}/{side}: duplicate object ID")
    for oid, obj in mapping.items():
        if not isinstance(obj, dict):
            raise InterfaceBlocked(f"{key}/{side}/{oid}: object must be a mapping")


def merge_official_policy(vehicle_objects, uav_objects):
    """Equivalent to the official update order, with a provenance ledger."""
    if not isinstance(vehicle_objects, dict) or not isinstance(uav_objects, dict):
        raise InterfaceBlocked("both object collections must be mappings")
    _validate_id_keys(vehicle_objects, "vehicle", "synthetic")
    _validate_id_keys(uav_objects, "uav", "synthetic")
    merged = {}
    provenance = {}
    for oid, obj in vehicle_objects.items():
        merged[oid] = copy.deepcopy(obj)
        provenance[oid] = {
            "final_source": "vehicle",
            "vehicle_input_present": True,
            "uav_input_present": False,
            "overwritten_vehicle": None,
        }
    for oid, obj in uav_objects.items():
        was_present = oid in merged
        previous = copy.deepcopy(merged[oid]) if was_present else None
        merged[oid] = copy.deepcopy(obj)
        provenance[oid] = {
            "final_source": "uav",
            "vehicle_input_present": was_present,
            "uav_input_present": True,
            "overwritten_vehicle": previous,
        }
    return merged, provenance


def audit_frame(key, expected_reference, official):
    rows = {}
    raw = {}
    for side, agent in (("vehicle", "1"), ("uav", "2")):
        base = ROOT / "code/artifacts/data/v2u4real_r2_01/selected_samples/train" / SCENE / agent
        path = base / "yaml/ouster" / f"{key}.yaml"
        value, metadata = _read_yaml(path)
        objects = value["vehicles"]
        _validate_id_keys(objects, side, key)
        strict_vector(value["lidar_pose"], 6, f"{key}/{side}/lidar_pose")
        for oid, obj in objects.items():
            # The fixed sample contract expects string IDs. Refuse unexpected
            # types instead of coercing keys or risking JSON key collisions.
            if not isinstance(oid, str):
                raise InterfaceBlocked(f"{key}/{side}: unexpected ID {oid!r} ({type(oid).__name__}); no normalization")
            for field in ("location", "center", "extent", "angle"):
                strict_vector(obj[field], 3, f"{key}/{side}/{oid}/{field}")
            if "obj_type" in obj and not isinstance(obj["obj_type"], str):
                raise InterfaceBlocked(f"{key}/{side}/{oid}: obj_type must be a string")
        expected = expected_reference[side]["yaml"]
        expected_sha = expected.get("sha256", expected.get("sha256_before"))
        if metadata["sha256"] != expected_sha:
            raise InterfaceBlocked(f"{key}/{side}: SHA differs from R2-04B")
        raw[side] = value
        rows[side] = metadata
    vehicle_objects = raw["vehicle"]["vehicles"]
    uav_objects = raw["uav"]["vehicles"]
    vehicle_ids = list(vehicle_objects.keys())
    uav_ids = list(uav_objects.keys())
    vehicle_set, uav_set = set(vehicle_ids), set(uav_ids)
    transform = official.x1_to_x2(raw["uav"]["lidar_pose"], raw["vehicle"]["lidar_pose"])
    matrix_check = matrix_stats(transform)
    converted = {}
    for side, t in (("vehicle", None), ("uav", transform)):
        # Unchanged fixed-source AST function; never invoke converter main or
        # write_annotation_file. Strict YAML validation above runs first.
        objects = official.load_annotation_objects(str(ROOT / rows[side]["path"]), transform=t)
        if list(objects) != list(raw[side]["vehicles"]):
            raise InterfaceBlocked(f"{key}/{side}: official loader changed the ID sequence")
        for oid, obj in objects.items():
            for field in ("loc", "dims"):
                if np.asarray(obj[field]).shape != (3,) or not np.isfinite(obj[field]).all():
                    raise InterfaceBlocked(f"{key}/{side}/{oid}: invalid converted {field}")
                obj[field] = [float(x) for x in obj[field]]
            if not np.isfinite(obj["heading"]):
                raise InterfaceBlocked(f"{key}/{side}/{oid}: invalid converted heading")
            obj["heading"] = float(obj["heading"])
        converted[side] = objects
    merged, provenance = merge_official_policy(converted["vehicle"], converted["uav"])
    out_ids = list(merged.keys())
    expected_out = vehicle_set | uav_set
    if set(out_ids) != expected_out:
        raise InterfaceBlocked(f"{key}: output ID set is not the input union")
    shared = vehicle_set & uav_set
    if any(provenance[oid]["final_source"] != "uav" for oid in shared):
        raise InterfaceBlocked(f"{key}: shared ID did not resolve to UAV")
    if any(provenance[oid]["final_source"] != "vehicle" for oid in vehicle_set - uav_set):
        raise InterfaceBlocked(f"{key}: vehicle-only ID was not retained")
    if any(provenance[oid]["final_source"] != "uav" for oid in uav_set - vehicle_set):
        raise InterfaceBlocked(f"{key}: UAV-only ID was not retained")
    final_objects = {}
    for oid in out_ids:
        p = provenance[oid]
        final_objects[oid] = {
            "final_source": p["final_source"],
            "object": copy.deepcopy(merged[oid]),
            "vehicle_input": copy.deepcopy(vehicle_objects[oid]) if oid in vehicle_objects else None,
            "uav_input": copy.deepcopy(uav_objects[oid]) if oid in uav_objects else None,
            "overwritten_vehicle": copy.deepcopy(vehicle_objects[oid]) if oid in shared else None,
            "overwritten_vehicle_official_object": p["overwritten_vehicle"],
            "vehicle_official_object": copy.deepcopy(converted["vehicle"].get(oid)),
            "uav_official_object": copy.deepcopy(converted["uav"].get(oid)),
            "interpretation": "official_code_interpretation_only",
        }
    return {
        "frame_key": key,
        "vehicle_input": rows["vehicle"],
        "uav_input": rows["uav"],
        "raw_lidar_pose": {side: raw[side]["lidar_pose"] for side in raw},
        "T_vehicle_from_uav": transform.tolist(),
        "matrix_check": matrix_check,
        "annotation_conversion": "unchanged official load_annotation_objects; official_code_interpretation_only",
        "vehicle_ids": vehicle_ids,
        "uav_ids": uav_ids,
        "vehicle_id_types": [type(oid).__name__ for oid in vehicle_ids],
        "uav_id_types": [type(oid).__name__ for oid in uav_ids],
        "shared_ids": sorted(shared),
        "vehicle_only_ids": sorted(vehicle_set - uav_set),
        "uav_only_ids": sorted(uav_set - vehicle_set),
        "merged_ids": out_ids,
        "counts": {
            "vehicle_input": len(vehicle_ids),
            "uav_input": len(uav_ids),
            "shared": len(shared),
            "vehicle_only": len(vehicle_set - uav_set),
            "uav_only": len(uav_set - vehicle_set),
            "merged_output": len(out_ids),
            "vehicle_overwritten_by_uav": len(shared),
        },
        "set_checks": {
            "merged_equals_vehicle_union_uav": set(out_ids) == expected_out,
            "shared_ids_final_source_uav": all(provenance[oid]["final_source"] == "uav" for oid in shared),
            "vehicle_only_retained": all(oid in merged for oid in vehicle_set - uav_set),
            "uav_only_retained": all(oid in merged for oid in uav_set - vehicle_set),
        },
        "final_objects": final_objects,
        "raw_annotations_preserved_in_audit": True,
        "official_policy": "implementation-faithful_policy_B",
    }


def run(output=OUT):
    if (output.resolve().parent != (ROOT / "code/artifacts/audit").resolve()
            or not output.name.startswith("v2u4real_r2_04c_")):
        raise InterfaceBlocked("output must be an R2-04C directory under code/artifacts/audit")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "V2U4Real-R2-04C-v1",
        "policy_decision": "B",
        "policy_status": "FROZEN_BY_RESEARCHER_CHOICE",
        "policy_type": "implementation-faithful_policy",
        "policy_boundary": "Reproduces fixed official converter behavior; does not claim consistency with every paper wording.",
        "dataset_version": DATASET_VERSION,
        "split": "train",
        "scene": SCENE,
        "agent_vehicle": "1",
        "agent_uav": "2",
        "sensor": "ouster",
        "fixed_keys": list(KEYS),
        "official_source": None,
        "runtime": {
            "device": "CPU",
            "python": platform.python_version(),
            "gpu_used": False,
            "locked_test_accessed": False,
            "r2_05_started": False,
            "labels_written": False,
        },
        "source_contract_status": "NOT_RUN",
        "input_integrity_status": "NOT_RUN",
        "label_merge_status": "NOT_RUN",
        "physical_alignment_status": "NOT_ESTABLISHED",
        "task_execution_status": "BLOCKED",
        "frames": [],
        "errors": [],
    }
    input_hashes = []
    try:
        result["official_source"] = verify_official_source()
        official, execution_identity = official_functions()
        result["official_source"]["execution"] = execution_identity["execution"]
        result["source_contract_status"] = "PASS"
        _, frames = load_r2_04b_reference()
        for key in KEYS:
            for agent in ("1", "2"):
                path = ROOT / "code/artifacts/data/v2u4real_r2_01/selected_samples/train" / SCENE / agent / "yaml/ouster" / f"{key}.yaml"
                if not path.is_file():
                    raise InterfaceBlocked(f"missing fixed YAML: {path}")
                input_hashes.append({"path": path.relative_to(ROOT).as_posix(), "sha256_before": sha256(path)})
        expected_frames = {key: frames[key] for key in KEYS}
        for key in KEYS:
            result["frames"].append(audit_frame(key, expected_frames[key]["inputs"], official))
        result["input_integrity_status"] = "PASS"
        result["label_merge_status"] = "PASS"
        result["task_execution_status"] = "COMPLETED"
        result["aggregate"] = {
            "vehicle_overwritten_total": sum(f["counts"]["vehicle_overwritten_by_uav"] for f in result["frames"]),
            "merged_output_total": sum(f["counts"]["merged_output"] for f in result["frames"]),
            "source_counts": {
                "vehicle_final": sum(sum(1 for row in f["final_objects"].values() if row["final_source"] == "vehicle") for f in result["frames"]),
                "uav_final": sum(sum(1 for row in f["final_objects"].values() if row["final_source"] == "uav") for f in result["frames"]),
            },
            "all_set_checks_pass": all(all(f["set_checks"].values()) for f in result["frames"]),
        }
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        result["errors"].append({"error": str(exc)})
        if result["source_contract_status"] != "PASS":
            result["source_contract_status"] = "BLOCKED"
        else:
            result["input_integrity_status"] = "BLOCKED"
        result["label_merge_status"] = "BLOCKED"
    finally:
        result["input_hashes_before_after"] = []
        for row in input_hashes:
            path = ROOT / row["path"]
            after = sha256(path) if path.is_file() else None
            result["input_hashes_before_after"].append({**row,
                "sha256_after": after, "unchanged": row["sha256_before"] == after})
        if result["input_hashes_before_after"] and not all(row["unchanged"] for row in result["input_hashes_before_after"]):
            result["input_integrity_status"] = "BLOCKED"
            result["task_execution_status"] = "BLOCKED"
        if result["errors"]:
            result["task_execution_status"] = "BLOCKED"
    write_json(output / "result.json", result)
    write_summary(result, output / "summary.md")
    return result


def write_summary(result, path):
    lines = ["# STEP R2-04C — V2U4Real 官方标签合并规则冻结与真实样本核验", "",
             "本轮冻结研究者选择的方案 B：固定官方转换器的 implementation-faithful policy。",
             "这表示复现官方实现行为，不声称它与论文所有文字描述完全一致。", ""]
    for key in ["source_contract_status", "input_integrity_status", "label_merge_status", "physical_alignment_status", "task_execution_status"]:
        lines.append(f"- `{key}`：`{result[key]}`")
    lines += ["", "## 官方行为", "", "固定 `uav_to_single.py` 先写入车辆对象，再遍历 UAV 对象执行 `merged_objs[oid] = obj`。因此同 ID 由 UAV 覆盖；车辆独有和 UAV 独有对象都保留；ID 不做整数化、前导零清理或空间近邻匹配。", "", "## 五帧核验", "", "| 文件键 | 车辆 IDs | UAV IDs | 共有 | 车辆独有 | UAV 独有 | 覆盖车辆数 | 合并后数量 |", "|---|---|---|---|---|---|---:|---:|"]
    for frame in result.get("frames", []):
        c = frame["counts"]
        lines.append(f"| {frame['frame_key']} | {frame['vehicle_ids']} | {frame['uav_ids']} | {frame['shared_ids']} | {frame['vehicle_only_ids']} | {frame['uav_only_ids']} | {c['vehicle_overwritten_by_uav']} | {c['merged_output']} |")
    if result.get("aggregate"):
        lines += ["", "累计：车辆被覆盖对象 **%d** 个；最终车辆来源对象 **%d** 个；最终 UAV 来源对象 **%d** 个。" % (result["aggregate"]["vehicle_overwritten_total"], result["aggregate"]["source_counts"]["vehicle_final"], result["aggregate"]["source_counts"]["uav_final"])]
    lines += ["", "## 追溯和边界", "", "复用 R2-04A 的固定源码加载器，实际调用 x1_to_x2(uav_pose, vehicle_pose) 与 load_annotation_objects：车辆不叠加相对变换，UAV 转到车辆参考系，再按官方循环合并。转换框均仅为 official_code_interpretation_only。result.json 对每个最终对象保存最终来源、两端原始输入、两端官方转换对象和被覆盖的车辆对象；输入 YAML SHA 在模拟前后复核不变。模拟只在内存中进行，没有写回标签、点云或 YAML。", "", "未解决：论文与官方实现的标签优先级差异、angle 顺序、location/center/extent 语义及单位、尺寸差异原因、物理轴向/标定/时间同步。方案 B 只冻结来源优先级，不表示 UAV 标签或位姿天然更准确。", "", "本轮未启动 R2-05、GPU、locked_test、训练、仿真或感知服务。", ""]
    path.write_text("\n".join(lines), encoding="utf8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ["policy_status", "source_contract_status", "input_integrity_status", "label_merge_status", "physical_alignment_status", "task_execution_status"]}, ensure_ascii=True))
    return 0 if result["task_execution_status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
