"""Five fixed train frames; official software geometry and internal evidence only.

No physical clock, calibration correction, registration, detector, or training.
R2-04A helpers and tolerances are reused unchanged. Same IDs are annotation
associations, not independently established physical correspondences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import yaml

import audit_v2u4real_r2_04a_single_frame_geometry_v1 as a

ROOT = a.ROOT
OUT = ROOT / "code/artifacts/audit/v2u4real_r2_04b_20261026"
KEYS = ("1752814590.2", "1752814590.4", "1752814590.6", "1752814591.2", "1752814591.6")
CONFIG = dict(a.CONFIG)
STATUS_KEYS = ("source_contract_status", "input_integrity_status", "software_transform_status",
               "multiframe_internal_consistency_status", "physical_alignment_status", "task_execution_status")


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify_reuse(root=ROOT):
    previous = root / "code/artifacts/audit/v2u4real_r2_04a_20261026"
    if load_json(previous / "config.json") != CONFIG:
        raise a.InterfaceBlocked("R2-04A tolerances/config changed")
    identity = load_json(previous / "source_identity.json")
    rel = "code/scripts/audit_v2u4real_r2_04a_single_frame_geometry_v1.py"
    for path, expected in identity["files"].items():
        if a.sha256(root / path) != expected:
            raise a.InterfaceBlocked(f"reused R2-04A source/reference identity changed: {path}")
    source = a.verify_source(root)
    return {"r2_04a_script": rel, "r2_04a_script_sha256": a.sha256(root / rel),
            "verified_reused_source_files": identity["files"],
            "r2_04a_config_sha256": a.sha256(previous / "config.json"), "official_source": source}


def source_contract(root=ROOT, evidence=OUT):
    basis = verify_reuse(root)
    paper_path = evidence / "paper_contract_evidence.json"
    tree_path = evidence / "official_repository_tree.json"
    inspection_path = evidence / "repository_inspection.json"
    paper, tree, inspection = map(load_json, [paper_path, tree_path, inspection_path])
    if tree.get("truncated") is not False or tree["sha"] != a.OFFICIAL_COMMIT:
        raise a.InterfaceBlocked("incomplete/wrong pinned official repository tree")
    paths = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    for f in basis["official_source"]["files"]:
        rel = f["path"].removeprefix("code/reference/V2U4Real/")
        if paths[rel]["sha"] != f["git_blob"]:
            raise a.InterfaceBlocked(f"remote tree/local snapshot mismatch: {rel}")
    if (paper["zotero_attachment"] != "JHB4X5KT" or paper["pdf_sha256"] !=
            "f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c"):
        raise a.InterfaceBlocked("paper evidence identity mismatch")
    return {"status": "DOCUMENTED_WITH_UNRESOLVED_FIELDS", "basis": basis,
            "evidence_files": {p.name: a.sha256(p) for p in [paper_path, tree_path, inspection_path]},
            "A_paper_contract": paper,
            "B_official_runtime": {
                "pcd_reader": "pcd_utils.pcd_to_np: Open3D read_point_cloud -> np.asarray(pcd.points); no axis matrix; float32 output",
                "pose": "[x,y,z,roll,yaw,pitch], degrees=True, Rotation.from_euler('ZYX',[yaw,pitch,roll])",
                "relative": "x1_to_x2(uav_pose_list, vehicle_pose_list) = inv(T_W_from_V) @ T_W_from_U",
                "separate_fixed_axis_matrix_in_audited_path": False,
                "caller": "uav_to_single.main:177 pose/list matrix; :185-186 point projection; no extra fixed axis factor",
                "box_center": "location+center in original local code frame; UAV center transformed by same relative matrix",
                "execution_boundary": "same R2-04A official function bodies; NumPy float64 point application; converter main/torch path not executed"},
            "C_sample_verifiable": ["file/decoded XYZ identity", "finite poses and fields", "software rigid geometry", "same-ID internal center/dimension comparisons"],
            "D_missing_authority": ["exact sensor-axis coefficients and export/application stage",
                                    "sample-specific raw-sensor/pose physical axes, unit, reference point and common origin",
                                    "RTK/GICP/manual corrections and application provenance", "physical timestamp pairing",
                                    "annotation angle and location/center semantics"],
            "repository_inspection": inspection,
            "working_interpretation": "use published XYZ under paper-declared convention and official pose matrix, with no additional unknown axis transform; candidate usage, not proven calibration"}


def read_frame(key, reference, root=ROOT):
    if key not in KEYS:
        raise a.InterfaceBlocked("only five researcher-fixed keys allowed")
    arrays, raw, inputs = {}, {}, {}
    for side, agent in [("vehicle", "1"), ("uav", "2")]:
        base = root / "code/artifacts/data/v2u4real_r2_01/selected_samples/train" / a.SCENE / agent
        pcd, meta = base / "ouster" / f"{key}.pcd", base / "yaml/ouster" / f"{key}.yaml"
        rows = {}
        for path, kind in [(pcd, "pcd"), (meta, "yaml")]:
            if not path.is_file():
                raise a.InterfaceBlocked(f"fixed file missing: {path}")
            digest = a.sha256(path)
            if digest != reference[side][kind]["sha256"]:
                raise a.InterfaceBlocked(f"input differs from R2-03: {path}")
            rows[kind] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256_before": digest}
        xyz, header = a.read_pcd_xyz(pcd)
        xyz_hash = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
        if xyz_hash != reference[side]["pcd"]["xyz_sha256"]:
            raise a.InterfaceBlocked("decoded XYZ identity mismatch")
        text = meta.read_bytes().decode("utf-8-sig")
        value = yaml.load(text, Loader=a.UniqueKeyLoader)
        if not isinstance(value, dict) or not isinstance(value.get("vehicles"), dict):
            raise a.InterfaceBlocked("YAML root and vehicles must be mappings")
        pose = a.strict_vector(value.get("lidar_pose"), 6, side + ".lidar_pose")
        for oid, obj in value["vehicles"].items():
            if not isinstance(obj, dict):
                raise a.InterfaceBlocked(f"{key}/{side}/{oid}: object mapping required")
            for field in ["location", "center", "extent", "angle"]:
                a.strict_vector(obj.get(field), 3, f"{key}/{side}/{oid}/{field}")
            if any(v <= 0 for v in obj["extent"]):
                raise a.InterfaceBlocked("nonpositive extent")
            if "obj_type" in obj and not isinstance(obj["obj_type"], str):
                raise a.InterfaceBlocked("class must be a string if present")
        rows["pcd"].update(header=header, decoded_xyz_sha256=xyz_hash, stats=a.point_stats(xyz))
        rows["yaml"].update(raw_text=text, root_field_types={str(k): type(v).__name__ for k, v in value.items()},
                             lidar_pose_raw=pose, lidar_pose_validation=a.vector_inventory(pose),
                             annotation_field_inventory={str(oid): {f: a.vector_inventory(obj[f]) for f in ["location", "center", "extent", "angle"]}
                                                         for oid, obj in value["vehicles"].items()},
                             vehicles_raw=value["vehicles"], annotation_count=len(value["vehicles"]))
        rows.update(agent=agent, sensor="ouster", frame_key=key,
                    identity_basis="fixed published path and R2-03 file hashes; physical time unverified")
        inputs[side], arrays[side], raw[side] = rows, np.asarray(xyz, dtype=np.float64), value
    return {"frame_key": key, "inputs": inputs, "input_status": "PASS"}, arrays, raw


def duplicate_check(frames, reference):
    groups, known_hits = [], []
    known = {g["sha256"] for g in reference["duplicate_pcd_groups"]}
    known_decoded = {r["vehicle"]["pcd"]["xyz_sha256"] for r in reference["pairs"]
                     if r["vehicle"]["pcd"]["sha256"] in known}
    for side in ["vehicle", "uav"]:
        for identity in ["sha256_before", "decoded_xyz_sha256"]:
            table = {}
            for row in frames:
                v = row["inputs"][side]["pcd"]
                table.setdefault(v[identity], []).append(row["frame_key"])
                if side == "vehicle" and ((identity == "sha256_before" and v[identity] in known)
                                          or (identity == "decoded_xyz_sha256" and v[identity] in known_decoded)):
                    known_hits.append({"side": side, "identity": identity, "key": row["frame_key"]})
            groups += [{"side": side, "identity": identity, "digest": d, "keys": ks}
                       for d, ks in table.items() if len(ks) > 1]
    return {"status": "PASS" if not groups and not known_hits else "BLOCKED",
            "within_selected_five_duplicates": groups, "R2_03_known_group_hits": known_hits,
            "scope": "five fixed frames, byte and decoded XYZ identity, known R2-03 groups; no global uniqueness claim"}


def compare_annotations(raw, transform):
    vehicle, uav = raw["vehicle"]["vehicles"], raw["uav"]["vehicles"]
    common = sorted(vehicle.keys() & uav.keys(), key=str)
    comparisons = []
    for oid in common:
        v, u = vehicle[oid], uav[oid]
        for side, obj in [("vehicle", v), ("uav", u)]:
            for field in ["location", "center", "extent", "angle"]:
                a.strict_vector(obj.get(field), 3, f"{side}/{oid}/{field}")
            if any(x <= 0 for x in obj["extent"]):
                raise a.InterfaceBlocked("positive raw extent required")
        vc = np.array(v["location"], dtype=np.float64) + v["center"]
        uc = np.array(u["location"], dtype=np.float64) + u["center"]
        ut = a.transform_points(uc[None], transform)[0]
        ve, ue = np.array(v["extent"], dtype=np.float64), np.array(u["extent"], dtype=np.float64)
        comparisons.append({"object_id": oid, "vehicle_raw": v, "uav_raw": u,
                            "vehicle_center_local_code": vc.tolist(), "uav_center_local_code": uc.tolist(),
                            "uav_center_in_vehicle_code": ut.tolist(), "delta_uav_minus_vehicle": (ut-vc).tolist(),
                            "distance_3d_code_units": float(np.linalg.norm(ut-vc)),
                            "extent_vehicle_raw": v["extent"], "extent_uav_raw": u["extent"],
                            "extent_delta_uav_minus_vehicle": (ue-ve).tolist(),
                            "extent_relative_delta_over_vehicle": ((ue-ve)/ve).tolist(),
                            "angle_vehicle_raw": v["angle"], "angle_uav_raw": u["angle"],
                            "class_values_equal": v.get("obj_type") == u.get("obj_type"),
                            "status": "official_code_interpretation_only"})
    return {"common_ids": common, "common_count": len(common), "comparisons": comparisons,
            "only_vehicle_ids": sorted(vehicle.keys()-uav.keys(), key=str),
            "only_uav_ids": sorted(uav.keys()-vehicle.keys(), key=str),
            "association_scope": "same published object ID; not independent physical matching",
            "dependence": "paper 3.2 uses coordinate transforms and IoU matching to unify IDs",
            "no_physical_error_threshold": True, "box_iou_or_heading_error_computed": False}


def scalar_distribution(values):
    p = np.asarray(values, dtype=np.float64)
    if not len(p):
        return {"count": 0, "reason": "no common-ID comparison", "mean": None, "min": None, "max": None, "quantiles": None}
    if not np.isfinite(p).all():
        raise a.InterfaceBlocked("nonfinite diagnostic distribution")
    return {"count": len(p), "mean": float(p.mean()), "min": float(p.min()), "max": float(p.max()),
            "quantile_positions": [0, .25, .5, .75, .9, .95, 1],
            "quantiles": np.quantile(p, [0, .25, .5, .75, .9, .95, 1]).tolist()}


def aggregate(frames):
    ranked = [{"frame_key": f["frame_key"], "object_id": c["object_id"],
               "center_distance_code_units": c["distance_3d_code_units"],
               "extent_delta": c["extent_delta_uav_minus_vehicle"]}
              for f in frames for c in f["annotation_comparison"]["comparisons"]]
    ranked.sort(key=lambda x: (-x["center_distance_code_units"], x["frame_key"], str(x["object_id"])))
    transitions = []
    for left, right in zip(frames, frames[1:]):
        t0, t1 = [np.array(f["transforms"]["T_vehicle_from_uav"]) for f in [left, right]]
        transitions.append({"from_key": left["frame_key"], "to_key": right["frame_key"],
                            "relative_translation_vector_change_code_units": float(np.linalg.norm(t1[:3, 3]-t0[:3, 3])),
                            "relative_rotation_matrix_frobenius_change": float(np.linalg.norm(t1[:3, :3]-t0[:3, :3])),
                            "interpretation": "ordered published keys only; not velocity, angular rate or physical-time continuity"})
    return {"common_comparison_count": len(ranked),
            "center_distance_distribution_code_units": scalar_distribution([r["center_distance_code_units"] for r in ranked]),
            "ranked_center_observations": ranked, "key_order_matrix_changes": transitions,
            "outlier_classification": "NOT_DEFINED: no independently supported physical threshold",
            "missing_or_unshared_ids_are_not_automatically_annotation_errors": True,
            "independence": "five nearby frames and repeated IDs are not independent calibration samples"}


def plot_frames(frames, clouds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = np.concatenate([p[:, :2] for row in clouds for p in row.values()])
    low, high = xy.min(0), xy.max(0)
    for frame in frames:
        for c in frame["annotation_comparison"]["comparisons"]:
            centers = np.array([c["vehicle_center_local_code"], c["uav_center_in_vehicle_code"]])[:, :2]
            low, high = np.minimum(low, centers.min(0)), np.maximum(high, centers.max(0))
    pad = np.maximum(.03*(high-low), 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    colors = {"vehicle": "#226bb0", "uav": "#d76721"}
    sampling = []
    for frame, pts, ax in zip(frames, clouds, axes.flat):
        row = {"frame_key": frame["frame_key"]}
        for side, p in pts.items():
            idx = a.selected_indices(len(p), CONFIG["plot_max_points_per_agent"])
            row[side] = {"count": len(idx), "index_sha256": hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()}
            ax.scatter(p[idx, 0], p[idx, 1], s=.35, alpha=.4, c=colors[side], rasterized=True,
                       label="Vehicle original" if side == "vehicle" else "UAV official -> vehicle")
        for c in frame["annotation_comparison"]["comparisons"]:
            vc, uc = np.array(c["vehicle_center_local_code"]), np.array(c["uav_center_in_vehicle_code"])
            ax.scatter(vc[0], vc[1], s=24, marker="+", c=colors["vehicle"])
            ax.scatter(uc[0], uc[1], s=24, marker="x", c=colors["uav"])
        distances = [c["distance_3d_code_units"] for c in frame["annotation_comparison"]["comparisons"]]
        ax.set(title=f"key {frame['frame_key']} | common IDs {len(distances)}\ncenter distances: {', '.join(f'{v:.3f}' for v in distances)} (code units)",
               xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]))
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.15)
        sampling.append(row)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    note = axes.flat[5]; note.axis("off")
    note.text(.03, .95, "Five fixed published file keys\nEach frame: independent vehicle-local code frame\n\nBlue: original vehicle XYZ\nOrange: UAV XYZ after official pose/list transform\n+ / x: same-ID code-interpreted centers\n\nCommon limits; equal X/Y scale\nUnits and physical axes not independently verified\nSame IDs depend on annotation processing\n\nNo ICP/GICP, no fitted or manual correction\nPhysical alignment: NOT ESTABLISHED\nFile key is not a verified timestamp", transform=note.transAxes, va="top", fontsize=12)
    fig.supxlabel("Vehicle-local code X (length unit unverified)")
    fig.supylabel("Vehicle-local code Y (physical direction unverified)")
    fig.suptitle(f"V2U4Real / train / {a.SCENE}\nMulti-frame software and internal-consistency diagnostic", fontsize=16)
    fig.tight_layout(rect=(.02, .02, 1, .94)); fig.savefig(output, dpi=150); plt.close(fig)
    return {"path": output.name, "sha256": a.sha256(output), "sampling": sampling,
            "range_basis": "all point XY and compared centers; shared bounds across five local frames",
            "xlim": [float(low[0]-pad[0]), float(high[0]+pad[0])],
            "ylim": [float(low[1]-pad[1]), float(high[1]+pad[1])], "equal_aspect": True,
            "frames_share_numeric_plotting_limits_not_a_single_static_physical_origin": True}


def run(root=ROOT, output=OUT, evidence=OUT):
    if not output.resolve().is_relative_to((root / "code/artifacts/audit").resolve()):
        raise a.InterfaceBlocked("outputs must stay in code/artifacts/audit")
    output.mkdir(parents=True, exist_ok=True)
    a.write_json(output / "config.json", {"fixed_keys": list(KEYS), "mathematics": CONFIG,
                                          "physical_center_threshold": None, "threshold_reason": "no independent physical basis"})
    result = {"schema_version": "V2U4Real-R2-04B-v1", "dataset_version": a.DATA_VERSION,
              "official_code_commit": a.OFFICIAL_COMMIT, "split": "train", "scene": a.SCENE,
              "fixed_keys": list(KEYS), "config": CONFIG, "frames": [], "errors": [],
              "runtime": {"device": "CPU", "gpu_used": False, "locked_test_accessed": False,
                          "python": platform.python_version(), "numpy": np.__version__, "scipy": a.scipy.__version__,
                          "training_or_simulator_started": False, "registration_executed": False},
              "executed_script_sha256": a.sha256(Path(__file__)),
              **{k: "NOT_RUN" for k in STATUS_KEYS}}
    result.update(physical_alignment_status="NOT_ESTABLISHED", task_execution_status="BLOCKED")
    stage, input_paths, raw_hashes = "source_contract_status", [], {}
    try:
        result[stage] = "CHECKING"
        result["source_contract"] = source_contract(root, evidence)
        result[stage] = result["source_contract"]["status"]
        official, _ = a.official_functions(root)
        stage = "input_integrity_status"
        ref = load_json(root / a.R2_03)
        if ref["dataset_version"] != a.DATA_VERSION or ref["scene_id"] != a.SCENE or ref["split"] != "train":
            raise a.InterfaceBlocked("R2-03 dataset identity changed")
        references = {row["frame_key"]: row for row in ref["pairs"]}
        # Capture all 20 file identities before any point transform.
        for key in KEYS:
            for agent in ["1", "2"]:
                base = root / "code/artifacts/data/v2u4real_r2_01/selected_samples/train" / a.SCENE / agent
                for sub, ext in [("ouster", "pcd"), ("yaml/ouster", "yaml")]:
                    path = base / sub / f"{key}.{ext}"
                    input_paths.append(path)
                    raw_hashes[path.relative_to(root).as_posix()] = a.sha256(path) if path.is_file() else None
        all_arrays, all_raw = [], []
        for key in KEYS:
            frame, arrays, raw = read_frame(key, references[key], root)
            result["frames"].append(frame); all_arrays.append(arrays); all_raw.append(raw)
        result["duplicate_check"] = duplicate_check(result["frames"], ref)
        if result["duplicate_check"]["status"] != "PASS":
            raise a.InterfaceBlocked("fixed samples include known/new duplicate payloads; no substitution")
        result[stage] = "PASS"
        stage = "software_transform_status"
        clouds = []
        for frame, arrays, raw in zip(result["frames"], all_arrays, all_raw):
            vp, up = raw["vehicle"]["lidar_pose"], raw["uav"]["lidar_pose"]
            tv, tu = official.x_to_world(vp), official.x_to_world(up)
            t = official.x1_to_x2(up, vp)  # Actual list/list branch; original raw list inputs.
            frame["transforms"] = {"T_world_from_vehicle": tv.tolist(), "T_world_from_uav": tu.tolist(), "T_vehicle_from_uav": t.tolist()}
            frame["matrix_checks"] = {k: a.matrix_stats(v) for k, v in frame["transforms"].items()}
            matrix_error = float(np.max(np.abs(t-np.linalg.solve(tv, tu))))
            frame["relative_matrix_check"] = {"official_call": "x1_to_x2(uav_pose_list, vehicle_pose_list)", "branch": "list/list",
                                               "source": "UAV local", "target": "vehicle local", "order": "inv(T_W_from_V) @ T_W_from_U",
                                               "independent_method": "solve(T_W_from_V,T_W_from_U)", "max_abs_error": matrix_error}
            projected, checks = a.geometry_checks(arrays["uav"], t)
            frame["point_checks"] = checks
            frame["software_transform_status"] = "PASS" if checks["status"] == "PASS" and matrix_error <= CONFIG["matrix_atol"] else "BLOCKED"
            if frame["software_transform_status"] != "PASS":
                raise a.InterfaceBlocked("numeric check failed; fixed tolerances unchanged")
            frame["annotation_comparison"] = compare_annotations(raw, t)
            clouds.append({"vehicle": arrays["vehicle"], "uav": projected})
        result[stage] = "PASS"
        stage = "multiframe_internal_consistency_status"
        result["aggregate"] = aggregate(result["frames"])
        result["visualization"] = plot_frames(result["frames"], clouds, output / "multiframe_bev.png")
        result[stage] = "INTERNAL_EVIDENCE_RECORDED_WITH_LIMITATIONS"
        result["task_execution_status"] = "COMPLETED"
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        result[stage] = "BLOCKED"; result["errors"].append({"stage": stage, "error": str(exc)})
    finally:
        changes = []
        for path in input_paths:
            rel = path.relative_to(root).as_posix()
            after = a.sha256(path) if path.is_file() else None
            before = raw_hashes[rel]
            changes.append({"path": rel, "sha256_before": before, "sha256_after": after,
                            "unchanged": before is not None and before == after})
        result["input_hashes_before_after"] = changes
        if changes and not all(x["unchanged"] for x in changes):
            result["input_integrity_status"] = "BLOCKED"; result["task_execution_status"] = "BLOCKED"
        try:
            verify_reuse(root)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["source_contract_status"] = "BLOCKED"; result["task_execution_status"] = "BLOCKED"
            result["errors"].append({"stage": "final_source_identity", "error": str(exc)})
    a.write_json(output / "result.json", result)
    write_summary(result, output / "summary.md")
    return result


def write_summary(result, path):
    lines = ["# STEP R2-04B — 五帧发布坐标与内部几何审计", "",
             "本轮只验证发布数据按官方路径的内部关系；没有独立物理标定真值。", ""]
    lines += [f"- `{k}`：`{result[k]}`" for k in STATUS_KEYS]
    lines += ["", "## 五帧实测", "", "| 发布文件键 | 车辆/UAV点数 | 共同ID | 中心距离（代码长度单位） | 正逆最大误差 |", "|---|---|---|---|---|"]
    for f in result["frames"]:
        if "point_checks" not in f:
            continue
        ann = f["annotation_comparison"]
        lines += [f"| {f['frame_key']} | {f['inputs']['vehicle']['pcd']['stats']['count']}/{f['inputs']['uav']['pcd']['stats']['count']} | {ann['common_ids']} | "
                  + ", ".join(f"{c['object_id']}: {c['distance_3d_code_units']:.9g}" for c in ann["comparisons"])
                  + f" | {f['point_checks']['roundtrip_max_abs']:.9g} |"]
    if "aggregate" in result:
        lines += ["", "全部同ID中心距离分布：", "", "```json", json.dumps(result["aggregate"]["center_distance_distribution_code_units"], indent=2), "```"]
    lines += ["", "## 坐标合同与证据边界", "",
              "A 论文声明：局部轴前/左/上，世界系NED（北/东/下）；§3.2的传感器轴统一与§4.1的UAV到ego任务坐标转换不能混为同一个已知矩阵。",
              "B 代码事实：PCD直接读取XYZ；官方pose顺序[x,y,z,roll,yaw,pitch]、角度默认度、ZYX旋转；UAV→车辆为inv(T_W_from_V)@T_W_from_U，所查执行路径无另乘固定轴矩阵。",
              "C 样本实测：原始字段、20个SHA、有限性、矩阵/全点正逆/确定性距离与同ID中心/原始尺寸差异均见result.json；绘图使用统一数值范围，但各帧是各自的车辆局部坐标，不能当作一个静止世界原点。",
              "D 未独立核验：轴变换系数/应用阶段、实际物理轴向/单位/公共原点、RTK/GICP/人工修正来源、真实时间配对及标注角度/location/center语义。",
              "", "## 内部一致性范围", "",
              "同ID由论文描述的变换与IoU匹配参与统一，中心接近只提供内部证据；没有独立匹配真值、物理误差阈值、3D框IoU或航向误差验收。",
              "extent原值和逐轴差异保留，不交换尺寸轴，不重新定义字段。only_vehicle/only_uav ID是两侧标注集合差异，不自动认定漏标。",
              "有限帧不能建立连续物理时间轨迹、全数据集唯一性或普遍对齐能力。No GPU / locked_test / training / replay / registration.",
              "", "## R2-05条件", "",
              "本轮提供固定发布数据使用方式的候选依据，不自动开放R2-05。论文§4.2与官方uav_to_single.main的同ID真值保留优先级不同，且角度/标签合同尚未裁决；固定感知基线仍需研究者冻结标签与评价合同并另行授权。",
              "", "## 错误", "", json.dumps(result["errors"], ensure_ascii=False), "",
              "实际命令与输出见run.log，分组测试见validation.json，证据检索见source_contract及配套JSON。本轮到此停止。", ""]
    path.write_text("\n".join(lines), encoding="utf8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({k: result[k] for k in STATUS_KEYS}))
    return 0 if result["task_execution_status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
