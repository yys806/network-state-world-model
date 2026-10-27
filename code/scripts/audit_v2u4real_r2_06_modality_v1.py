"""R2-06: offline, train-only incremental modality audit; never runs a detector.

Reuse R2-02 complete archive inventories after rechecking archive SHA-256.
Reuse R2-03/04/05 receipts rather than re-decoding PCD or rerunning geometry.
Only two raw YAML anchors are parsed to check the inherited camera records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from pi_jwm.v2u4real_interface_v1 import UniqueKeyLoader

OUT = "code/artifacts/audit/v2u4real_r2_06_20261027"
PRE = "code/artifacts/manifests/v2u4real_r2_02_20261026"
SCENE = "2025-07-18-12-53_2"
REVISION = "36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def head_identity(working: bytes, committed: bytes) -> str:
    if working == committed:
        return "EXACT_BYTES"
    if working.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"):
        return "CRLF_LF_ONLY"
    raise ValueError("inherited evidence content differs from HEAD")


def checked_train_member(name: str, agent: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if (p.is_absolute() or ".." in p.parts or "\\" in name
            or p.parts[:3] != ("train", SCENE, agent)):
        raise ValueError(f"outside frozen train scene: {name}")
    return p


def camera_paths(agent: str, key: str) -> list[str]:
    base = f"train/{SCENE}/{agent}/camera"
    if agent == "1":
        return [f"{base}/{side}/{key}.jpg" for side in ("left", "middle", "right")]
    if agent == "2":
        return [f"{base}/{key}.jpg"]
    raise ValueError("only frozen agents 1/2 are authorized")


def inventory_summary(inventory: dict, agent: str) -> dict:
    names = []
    rgb, camera_members = [], []
    for member in inventory["members"]:
        p = checked_train_member(member["name"], agent)
        names.append(member["name"])
        if "camera" in p.parts:
            camera_members.append(member["name"])
        if member["type"] == "file" and p.suffix.lower() in IMAGE_EXTENSIONS:
            rgb.append(member)
    duplicates = [k for k, n in Counter(names).items() if n > 1]
    if duplicates or len(names) != inventory["member_count"]:
        raise ValueError("duplicate or inconsistent inherited inventory")
    return {"agent": agent, "member_count": len(names),
            "file_counts_by_subdirectory": inventory["file_counts_by_subdirectory"],
            "rgb_file_count": len(rgb), "rgb_files": rgb,
            "camera_members": camera_members, "duplicate_member_names": duplicates,
            "inventory_basis": "R2-02 complete stream inventory, archive SHA reverified"}


def matrix_diagnostics(camera: dict) -> dict:
    k = np.asarray(camera["intrinsic"]["values"], dtype=np.float64)
    e = np.asarray(camera["extrinsic"]["values"], dtype=np.float64)
    if k.shape != (3, 3) or e.shape != (4, 4):
        raise ValueError("invalid camera matrix shape")
    if not np.isfinite(k).all() or not np.isfinite(e).all():
        raise ValueError("nonfinite camera matrix")
    # Diagnostics, not a new physical calibration acceptance threshold.
    return {"intrinsic": k.tolist(), "extrinsic": e.tolist(),
            "intrinsic_shape": [3, 3], "extrinsic_shape": [4, 4], "finite": True,
            "focal_entries": [float(k[0, 0]), float(k[1, 1])],
            "principal_point_entries": [float(k[0, 2]), float(k[1, 2])],
            "rotation_determinant": float(np.linalg.det(e[:3, :3])),
            "orthogonality_max_abs_error": float(np.max(np.abs(e[:3, :3].T @ e[:3, :3] - np.eye(3)))),
            "bottom_row": e[3].tolist(), "source_frame": None, "target_frame": None,
            "direction_status": "BLOCKED", "length_unit": None,
            "image_resolution": None, "distortion_model": None,
            "camera_slot_binding": "loader order only; YAML-to-physical-camera identity unverified"}


def geometry_gate(rgb_count: int, direction_verified: bool, time_verified: bool) -> dict:
    reasons = []
    if rgb_count == 0:
        reasons.append("NO_REAL_RGB_SAMPLE")
    if not direction_verified:
        reasons.append("CAMERA_LIDAR_MATRIX_DIRECTION_UNVERIFIED")
    if not time_verified:
        reasons.append("RGB_LIDAR_PHYSICAL_TIME_ASSOCIATION_UNVERIFIED")
    return {"status": "BLOCKED" if reasons else "SUPPORTED",
            "blocking_reasons": reasons, "projection_executed": False,
            "note": "gate only; passing prerequisites would still require a separately recorded real projection"}


def uses() -> list[dict]:
    return [
        {"id": 1, "use": "单车 LiDAR 3D 感知", "status": "PARTIAL",
         "evidence": ["R2-02 16 vehicle Ouster/YAML files", "R2-05A field and native LiDAR path audit"],
         "allowed": "固定 train 点云/标签读取和候选输入验证",
         "missing": ["标签角度/尺寸合同", "类别、ROI、评价维度/排序合同", "真实检测路径准入"]},
        {"id": 2, "use": "单 UAV LiDAR 3D 感知", "status": "PARTIAL",
         "evidence": ["R2-02 16 UAV Ouster/YAML files", "R2-04B official UAV-to-ego software transform PASS"],
         "allowed": "固定 train UAV 输入与候选坐标路径诊断",
         "missing": ["原生/ego 坐标与标签合同", "独立物理几何依据", "真实检测路径准入"]},
        {"id": 3, "use": "车辆与 UAV 协同 3D 感知", "status": "PARTIAL",
         "evidence": ["same-stem file association", "R2-04A/B software transform PASS", "R2-04C B policy frozen"],
         "allowed": "两来源软件接口和有限几何诊断；不是已验收协同效果",
         "missing": ["逐样本真实同步", "独立跨端物理配准", "B 与原生训练/评价路径适配", "RSU/ROI 合同"]},
        {"id": 4, "use": "RGB 单模态感知", "status": "BLOCKED",
         "evidence": ["complete downloaded train archive inventories: RGB=0", "owner issue #7"],
         "allowed": "仅相机元数据诊断和未来扩展需求整理",
         "missing": ["真实 RGB", "解码/分辨率/畸变元数据", "适用的图像任务标注"]},
        {"id": 5, "use": "RGB 与 LiDAR 多模态融合", "status": "BLOCKED",
         "evidence": ["RGB=0", "64 inherited camera matrix instances", "physical time/direction unverified"],
         "allowed": "未来扩展候选；第一阶段不纳入",
         "missing": ["真实配套图像", "相机-LiDAR 有向标定", "逐模态物理时间关联", "真实投影验收"]},
        {"id": 6, "use": "车辆-UAV 数据与 AirFogSim 场景结合", "status": "PARTIAL",
         "evidence": ["R2-05B real CPU application ingress PASS; wireless bytes=0"],
         "allowed": "软件 payload/应用任务队列接入",
         "missing": ["轨迹与仿真时钟合同", "公共物理坐标到 RSU", "真实无线服务累计/交付"]},
        {"id": 7, "use": "当前任务的多车辆、区域任务和通信调度实验", "status": "NOT_APPLICABLE",
         "evidence": ["frozen scene has one vehicle source and one UAV source", "no measured RSU/CSI/task queues"],
         "allowed": "仅可提供真实双源感知内容；完整系统须仿真补充",
         "missing": ["多传感车辆来源/轨迹", "RSU、区域任务、截止时间", "CSI、资源竞争、逐时隙服务"],
         "boundary": "NOT_APPLICABLE 指数据原生不包含完整系统；不表示禁止后续明确标注的仿真组合。标注目标车辆不等于传感信息源。"},
    ]


def run(output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "code/artifacts/audit").resolve()):
        raise ValueError("output must stay in repository audit directory")
    if (output / "result.json").exists():
        raise FileExistsError("existing result: use a new additive output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = read_json("code/scripts/configs/v2u4real_r2_01.json")
    if (config["split"], config["scene_id"], config["dataset_revision"]) != ("train", SCENE, REVISION):
        raise ValueError("frozen train identity changed")
    inherited_paths = [f"{PRE}/{n}.json" for n in
                       ["members_agent_1", "members_agent_2", "extraction", "selection", "metadata_field_audit"]]
    inherited_paths += [
        "code/artifacts/manifests/v2u4real_r2_03_20261026/real_data_contract_audit.json",
        "code/artifacts/audit/v2u4real_r2_04a_20261026/result.json",
        "code/artifacts/audit/v2u4real_r2_04b_20261026/result.json",
        "code/artifacts/audit/v2u4real_r2_04c_20261027/result.json",
        "code/artifacts/audit/v2u4real_r2_05a_20261027/result.json",
        "code/artifacts/audit/v2u4real_r2_05b_20261027/airfogsim_cpu_smoke_20261027/acceptance.json",
    ]
    provenance = []
    for relative in inherited_paths:
        current = sha256(ROOT / relative)
        probe = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True)
        if probe.returncode:
            if relative != inherited_paths[-1]:
                raise ValueError(f"expected committed evidence absent: {relative}")
            receipt = read_json(relative)
            if (receipt["scene"] != SCENE or receipt["data_ingestion"]["split"] != "train"
                    or receipt["overall_status"] != "PASS_SOFTWARE_CHAIN_WITH_WIRELESS_BLOCK"
                    or receipt["wireless_delivery"]["status"] != "BLOCKED_NOT_ATTEMPTED"):
                raise ValueError("unexpected local R2-05B receipt identity/scope")
            copy_path = output / "inherited_r2_05b_acceptance.json"
            with copy_path.open("xb") as f:
                f.write((ROOT / relative).read_bytes())
            provenance.append({"path": relative, "sha256": current,
                               "git_blob_content_sha256": None, "matches_head_content": None,
                               "head_comparison": "LOCAL_EVIDENCE_NOT_IN_GIT",
                               "reason": "existing ignored R2-05B conda CPU receipt; not inferred from remote",
                               "preserved_copy": copy_path.relative_to(ROOT).as_posix()})
        else:
            committed = probe.stdout
            identity = head_identity((ROOT / relative).read_bytes(), committed)
            provenance.append({"path": relative, "sha256": current,
                               "git_blob_content_sha256": hashlib.sha256(committed).hexdigest(),
                               "matches_head_content": True, "head_comparison": identity})
    archives = []
    for agent in ("1", "2"):
        spec = config["archives"][agent]
        archive = ROOT / config["data_root"] / "full_archives" / Path(spec["path"]).name
        actual = {"path": archive.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size,
                  "sha256": sha256(archive)}
        if actual["bytes"] != spec["bytes"] or actual["sha256"] != spec["sha256_expected"]:
            raise ValueError("archive identity differs from official frozen release")
        inv = read_json(f"{PRE}/members_agent_{agent}.json")
        if inv["archive"]["sha256"] != actual["sha256"]:
            raise ValueError("inventory/archive identity mismatch")
        archives.append({**actual, **inventory_summary(inv, agent), "identity_status": "PASS"})
    raw_checks = []
    for row in read_json(f"{PRE}/extraction.json")["files"]:
        prefix = f"code/artifacts/data/v2u4real_r2_01/selected_samples/train/{SCENE}/"
        if not row["path"].startswith(prefix):
            raise ValueError("extraction manifest contains out-of-scope data")
        p = (ROOT / row["path"]).resolve()
        if not p.is_relative_to((ROOT / prefix).resolve()):
            raise ValueError("sample path escape")
        if p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
            raise ValueError(f"raw input identity changed: {p}")
        raw_checks.append({"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]})
    selection = read_json(f"{PRE}/selection.json")
    local_rgb = []
    scanned_roots = []
    for folder in ("selected_samples", "partial_real"):
        base = ROOT / config["data_root"] / folder / "train" / SCENE
        scanned_roots.append(base.relative_to(ROOT).as_posix())
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                local_rgb.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)})
    camera_rows = []
    anchors = []
    metadata = read_json(f"{PRE}/metadata_field_audit.json")["files"]
    for row in metadata:
        agent, key = row["agent"], row["frame_key"]
        if agent not in ("1", "2") or key not in selection["keys"]:
            raise ValueError("unexpected inherited metadata row")
        expected_slots = {"camera0", "camera1", "camera2"} if agent == "1" else {"camera0"}
        if set(row["cameras"]) != expected_slots:
            raise ValueError("camera slot contract changed")
        for slot, camera in row["cameras"].items():
            camera_rows.append({"agent": agent, "frame_key": key, "slot": slot,
                                "basis": "inherited metadata, source bytes SHA checked",
                                **matrix_diagnostics(camera)})
        if key == selection["keys"][0]:
            relative = f'{config["data_root"]}/selected_samples/train/{SCENE}/{agent}/yaml/ouster/{key}.yaml'
            raw = yaml.load((ROOT / relative).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
            for slot in expected_slots:
                for field in ("intrinsic", "extrinsic"):
                    if raw[slot][field] != row["cameras"][slot][field]["values"]:
                        raise ValueError("raw YAML camera anchor differs from inherited audit")
            anchors.append({"path": relative, "sha256": sha256(ROOT / relative),
                            "camera_fields_match_inherited": True,
                            "raw_camera_keys": {s: sorted(raw[s]) for s in sorted(expected_slots)}})
    expected = [{"agent": a, "frame_key": key, "loader_expected_path": p,
                 "association_semantics": "expected filename only, not physical synchronization",
                 "present_in_archive_inventory": False}
                for a in ("1", "2") for key in selection["keys"] for p in camera_paths(a, key)]
    rgb_count = sum(a["rgb_file_count"] for a in archives)
    if rgb_count or local_rgb:
        raise ValueError("new RGB found: bounded decode audit needed; do not reuse no-image conclusion")
    old = read_json(inherited_paths[5])
    b = read_json("code/artifacts/audit/v2u4real_r2_04b_20261026/result.json")
    result = {
        "schema_version": "v2u4real_r2_06_modality_audit_v1", "date": "2026-10-27",
        "task_execution_status": "COMPLETED_AUDIT_ONLY",
        "data_readiness_status": "PARTIAL_LIDAR_BLOCKED_RGB_AND_FORMAL_BASELINE",
        "baseline_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset_revision": REVISION, "official_code_commit": config["official_code_revision"],
        "split": "train", "scene": SCENE, "selected_keys": selection["keys"],
        "archive_inventory": archives, "raw_input_integrity": {"status": "PASS", "files": raw_checks},
        "rgb_inventory": {"status": "BLOCKED_NO_RGB_IN_DOWNLOADED_RELEASE", "actual_rgb_count": rgb_count,
                          "local_rgb_files": local_rgb, "scanned_train_roots": scanned_roots,
                          "loader_expected_references": expected, "expected_reference_count": len(expected),
                          "decoded_image_count": 0, "decode_status": "BLOCKED_NO_SAMPLE",
                          "format": None, "resolution": None, "image_metadata": None,
                          "corruption_status": "NOT_ASSESSABLE_NO_SAMPLE",
                          "duplicate_image_status": "NOT_ASSESSABLE_NO_SAMPLE",
                          "missingness": "unreleased modality; not measured image corruption or full-dataset missing rate",
                          "scope": "one downloaded train scene, two complete archives; not all official train scenes"},
        "camera_calibration": {"status": "PARTIAL_NUMERIC_METADATA_ONLY", "instance_count": len(camera_rows),
                               "yaml_anchor_checks": anchors, "records": camera_rows,
                               "physical_direction_verified": False, "sensor_mapping_verified": False,
                               "scope": "16 Ouster YAML keys/agent only; not M1/Ruby or other scenes"},
        "time_association": {"status": "BLOCKED", "frame_key_semantics": "published_identifier_only",
                             "timestamp_raw": None, "timestamp_unit": None, "time_origin": None,
                             "rgb_exposure_time": None, "lidar_scan_time": None,
                             "per_sample_interagent_delta": None,
                             "inherited_raw_time_fields": [r["raw_time_fields"] for r in metadata],
                             "paper_only": "nominal 10 Hz; GPS/PTP/PPS; nearest LiDAR pairing; reported <=20 ms",
                             "generation_code_available": False,
                             "filename_or_sequence_index_is_not_hardware_time": True},
        "annotation_contract": {"status": "PARTIAL_3D_LABELS_ONLY", "source": "same-agent Ouster YAML vehicles mapping",
                                "fields": ["location", "center", "extent", "angle", "obj_type", "speed"],
                                "annotation_type": "3D box parameterization; no verified image 2D boxes",
                                "rgb_2d_labels": None, "rgb_projected_3d_labels_verified": False,
                                "merge_policy": "B: UAV overwrites shared ID; exclusive IDs preserved; already frozen",
                                "unresolved": ["angle order paper/code", "location/center semantics", "cross-source extent differences",
                                               "native path versus B policy", "BEV AP versus volumetric 3D AP"]},
        "geometry_diagnostic": geometry_gate(rgb_count, False, False),
        "inherited_lidar": {"r2_04b_software_transform": b["software_transform_status"],
                            "physical_alignment": b["physical_alignment_status"],
                            "duplicate_pcd_groups": old["duplicate_pcd_groups"],
                            "duplicate_cause": "BLOCKED_UNKNOWN", "pcd_decoded_this_step": 0,
                            "old_five_frame_geometry_reexecuted": False, "merge_reexecuted": False,
                            "mock_or_airfogsim_reexecuted": False,
                            "world_origin_units_axes_registration_correction": "UNVERIFIED"},
        "use_assessments": uses(),
        "formal_data_boundary": {"current_allowed": ["train parsing and interface validation", "limited documented software geometry diagnostics"],
                                 "diagnostic_only": ["camera numeric matrices", "same-stem associations", "duplicate scans", "nominal 0.1 simulation tick if explicitly assumed"],
                                 "simulation_required": ["additional sensing sources", "RSU and regions", "preset UAV/vehicle trajectory contract", "CSI/resources/queues/deadlines/service"],
                                 "rgb": "FUTURE_EXTENSION_ONLY_BLOCKED", "phase_one": "LiDAR 3D only",
                                 "formal_baseline_admission": "BLOCKED_PENDING_DATA_LABEL_EVALUATION_CONTRACT",
                                 "bounded_baseline_preparation": "PARTIAL: real train inputs exist; CPU model-entry planning possible, no training authorized"},
        "minimal_rgb_supplement_proposal": {"executed": False,
                                           "request": "3 published train keys, same scene: 3 vehicle cameras + 1 UAV camera each (12 original RGB images), paired Ouster/YAML, original exposure/scan times and pairing offsets, named directed extrinsics, distortion/rectification and calibration provenance",
                                           "bytes": None, "bytes_reason": "original image formats/sizes unavailable; no bulk download authorized",
                                           "acceptance": "decode/metadata/hash checks first; project only after directed calibration and physical timing evidence"},
        "provenance": provenance,
        "runtime": {"python": sys.version, "numpy": np.__version__, "pyyaml": yaml.__version__,
                    "script_sha256": sha256(Path(__file__))},
        "boundaries": {"gpu_used": False, "training_started": False, "model_inference": False,
                       "validation_accessed": False, "test_accessed": False, "locked_test_accessed": False,
                       "raw_data_written": False, "data_downloaded": False, "scientific_decisions_changed": False},
    }
    write_json(output / "result.json", result)
    write_json(output / "source_identity.json", {"provenance": provenance, "raw_inputs": raw_checks,
                                                "archives": [{k: a[k] for k in ("path", "bytes", "sha256")} for a in archives],
                                                "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"task": result["task_execution_status"], "rgb_count": rgb_count,
                      "camera_matrix_instances": len(camera_rows), "geometry": result["geometry_diagnostic"]["status"],
                      "formal_baseline": result["formal_data_boundary"]["formal_baseline_admission"]}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    run(args.output)
    return 0  # audit executed; never a modality/training admission PASS


if __name__ == "__main__":
    raise SystemExit(main())
