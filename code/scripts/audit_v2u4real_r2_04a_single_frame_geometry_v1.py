"""CPU-only, pinned single-frame software geometry diagnostic; no physical acceptance.

Only the published same-key association is used. Official function AST bodies
are compiled unchanged after source fingerprint checks, avoiding OpenCOOD's
unavailable Open3D imports. No imports are mocked and no third-party file is edited.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from pi_jwm.v2u4real_interface_v1 import (  # noqa: E402
    InterfaceBlocked, UniqueKeyLoader, read_pcd_xyz, rigid_matrix, sha256,
    transform_points,
)

FRAME_KEY = "1752814590.2"
SCENE = "2025-07-18-12-53_2"
DATA_VERSION = "36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a"
OFFICIAL_COMMIT = "e099a3a746c6092a614e18b74d1f5b2f3c8351e8"
REPOSITORY = "https://github.com/VjiaLi/V2U4Real"
OUT = ROOT / "code/artifacts/audit/v2u4real_r2_04a_20261026"
R2_03 = "code/artifacts/manifests/v2u4real_r2_03_20261026/real_data_contract_audit.json"
PINNED_SHA = {
    "opencood/utils/transformation_utils.py": "2c198a94ffaf9ac3087396ee364b866a817fb2a075ac7c37c618cb03091ebd80",
    "opencood/utils/uav_to_single.py": "cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b",
    "opencood/utils/box_utils.py": "820ac498be2fa586ab2c797344d55dbd3681b6e3b21536ea00412990a9777657",
}
# Software tolerances only; these do not express physical calibration accuracy.
CONFIG = {
    "dtype": "float64", "rigid_atol": 1e-10, "matrix_atol": 1e-12,
    "inverse_max_abs_atol": 1e-8, "distance_max_abs_atol": 1e-8,
    "distance_subset_count": 128, "plot_max_points_per_agent": 20000,
    "subset_rule": "unique linspace(0, N-1, count, dtype=int64)",
    "quantiles": [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1],
    "rtol": 0.0,
}
LIMITS = [
    "Same published key is a file association, not measured physical synchronization.",
    "Physical length unit, local axes, common origin and applied predefined axes transform are unverified.",
    "RTK/GICP/manual registration corrections and whether already applied remain unverified.",
    "Annotation angle order in paper differs from code; location/center semantics remain unresolved.",
    "list/matrix x1_to_x2 branch uses its matrix directly despite its docstring; only identity is passed there for boxes.",
    "Boxes are official_code_interpretation_only, not independently verified ground truth.",
    "No correction, ICP/GICP, physical acceptance, GPU, training, replay or locked_test.",
]


def write_json(path, value):
    # Canonical LF keeps artifact SHA identical before/after Git normalization.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def strict_vector(value, length, name):
    if (not isinstance(value, list) or len(value) != length
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)):
        raise InterfaceBlocked(f"{name}: numeric list[{length}] required")
    if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
        raise InterfaceBlocked(f"{name}: nonfinite values")
    return value  # Preserve list identity/type for the official list/list branch.


def vector_inventory(value):
    return {"type": type(value).__name__, "length": len(value),
            "element_types": [type(x).__name__ for x in value],
            "finite": bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())}


def verify_source(root=ROOT):
    manifest_path = root / "code/reference/V2U4Real.SOURCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest["repository"] != REPOSITORY or manifest["commit"] != OFFICIAL_COMMIT:
        raise InterfaceBlocked("official repository/commit mismatch")
    if any(manifest["files"].get(p) != h for p, h in PINNED_SHA.items()):
        raise InterfaceBlocked("official required manifest fingerprint mismatch")
    rows = []
    for rel, expected in manifest["files"].items():
        path = root / "code/reference/V2U4Real" / rel
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if actual != expected or blob != manifest["git_blobs"][rel]:
            raise InterfaceBlocked(f"official source identity mismatch: {rel}")
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": actual,
                     "git_blob": blob, "bytes": len(payload)})
    return {"repository": REPOSITORY, "commit": OFFICIAL_COMMIT,
            "manifest_sha256": sha256(manifest_path), "files": rows,
            "identity_basis": "pinned local partial snapshot + manifest SHA256/Git blobs; not a fresh remote attestation"}


def official_functions(root=ROOT):
    identity = verify_source(root)
    base = root / "code/reference/V2U4Real"
    ns = {"np": np, "Rotation": Rotation, "sys": sys, "yaml": yaml}
    provenance = []

    def load(rel, names):
        path = base / rel
        tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise InterfaceBlocked(f"official functions missing in {rel}")
        for node in nodes:
            provenance.append({"function": node.name, "path": path.relative_to(root).as_posix(),
                               "start_line": node.lineno, "end_line": node.end_lineno,
                               "file_sha256": sha256(path), "ast_body_modified": False})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)

    load("opencood/utils/transformation_utils.py", ["x_to_world", "x1_to_x2"])
    load("opencood/utils/box_utils.py", ["create_bbx", "corner_to_center"])
    ns["box_utils"] = SimpleNamespace(create_bbx=ns["create_bbx"], corner_to_center=ns["corner_to_center"])
    # CLASS_MAP is also taken verbatim from the pinned converter, not redefined.
    path = base / "opencood/utils/uav_to_single.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "CLASS_MAP" for t in n.targets)]
    if len(assignments) != 1:
        raise InterfaceBlocked("official CLASS_MAP missing")
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), ns)
    load("opencood/utils/uav_to_single.py", ["load_annotation_objects"])
    identity["execution"] = {"mode": "unchanged official function AST bodies with real NumPy/SciPy/PyYAML dependencies",
                             "functions": provenance, "converter_main_executed": False,
                             "point_application": "existing float64 transform_points; official converter uses torch projection and float32 NumPy conversion, not executed"}
    return SimpleNamespace(**ns), identity


def selected_indices(n, count):
    if n < 1:
        raise InterfaceBlocked("nonempty points required")
    return np.unique(np.linspace(0, n - 1, min(n, count), dtype=np.int64))


def point_stats(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise InterfaceBlocked("nonempty finite Nx3 required")
    return {"count": len(p), "finite": True, "min": p.min(0).tolist(), "max": p.max(0).tolist(),
            "centroid": p.mean(0).tolist(), "std": p.std(0).tolist(),
            "quantiles_xyz": np.quantile(p, CONFIG["quantiles"], axis=0).tolist()}


def matrix_stats(matrix):
    t = rigid_matrix(matrix)  # Existing strict check, never relaxed.
    r = t[:3, :3]
    orth_error = float(np.max(np.abs(r.T @ r - np.eye(3))))
    det = float(np.linalg.det(r))
    if orth_error > CONFIG["rigid_atol"] or abs(det - 1) > CONFIG["rigid_atol"]:
        raise InterfaceBlocked("rotation outside predeclared rigid tolerance")
    return {"shape": list(t.shape), "finite": True,
            "orthogonality_max_abs": orth_error,
            "determinant": det, "determinant_abs_error": abs(det-1), "status": "PASS"}


def geometry_checks(points, transform):
    matrix_stats(transform)
    p = np.asarray(points, dtype=np.float64)
    before = point_stats(p)
    projected = transform_points(p, transform)
    inverse = np.linalg.inv(transform)
    recovered = transform_points(projected, inverse)
    delta = recovered - p
    idx = selected_indices(len(p), CONFIG["distance_subset_count"])
    # Every unordered pair in this deterministic subset, not just adjacent pairs.
    i, j = np.triu_indices(len(idx), k=1)
    d0 = np.linalg.norm(p[idx[i]] - p[idx[j]], axis=1)
    d1 = np.linalg.norm(projected[idx[i]] - projected[idx[j]], axis=1)
    distance_error = float(np.max(np.abs(d1 - d0))) if len(d0) else 0.0
    max_abs = float(np.max(np.abs(delta)))
    inv_err = max(float(np.max(np.abs(transform @ inverse - np.eye(4)))),
                  float(np.max(np.abs(inverse @ transform - np.eye(4)))))
    centroid_err = float(np.max(np.abs(projected.mean(0) - transform_points(p.mean(0)[None], transform)[0])))
    checks = {
        "all_points_float64": True, "before": before, "after": point_stats(projected),
        "recovered": point_stats(recovered), "inverse_matrix": inverse.tolist(),
        "inverse_product_max_abs": inv_err, "roundtrip_max_abs": max_abs,
        "roundtrip_coordinate_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "roundtrip_point_norm_rmse": float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))),
        "centroid_transform_max_abs": centroid_err,
        "distance_subset_indices": idx.tolist(), "distance_pair_count": len(i),
        "distance_max_abs_error": distance_error,
        "distance_rmse": float(np.sqrt(np.mean((d1-d0)**2))) if len(d0) else 0.0,
    }
    checks["status"] = "PASS" if (
        max_abs <= CONFIG["inverse_max_abs_atol"]
        and inv_err <= CONFIG["matrix_atol"]
        and distance_error <= CONFIG["distance_max_abs_atol"]
        and centroid_err <= CONFIG["inverse_max_abs_atol"]
        and before["count"] == len(projected) == len(recovered)
    ) else "BLOCKED"
    return projected, checks


def read_inputs(root=ROOT):
    audit_path = root / R2_03
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    if (audit["dataset_version"] != DATA_VERSION or audit["scene_id"] != SCENE
            or audit["split"] != "train" or audit["official_code_commit"] != OFFICIAL_COMMIT):
        raise InterfaceBlocked("R2-03 identity mismatch")
    match = [r for r in audit["pairs"] if r["frame_key"] == FRAME_KEY]
    if len(match) != 1:
        raise InterfaceBlocked("fixed key absent/ambiguous in R2-03")
    duplicate_hashes = {g["sha256"] for g in audit["duplicate_pcd_groups"]}
    rows, arrays, raw = {}, {}, {}
    for side, agent in [("vehicle", "1"), ("uav", "2")]:
        base = root / "code/artifacts/data/v2u4real_r2_01/selected_samples/train" / SCENE / agent
        pcd = base / "ouster" / (FRAME_KEY + ".pcd")
        meta = base / "yaml/ouster" / (FRAME_KEY + ".yaml")
        for path, kind in [(pcd, "pcd"), (meta, "yaml")]:
            if not path.is_file() or sha256(path) != match[0][side][kind]["sha256"]:
                raise InterfaceBlocked(f"fixed input missing/identity changed: {path}")
        if side == "vehicle" and sha256(pcd) in duplicate_hashes:
            raise InterfaceBlocked("fixed vehicle PCD is in a known duplicate group")
        text = meta.read_text(encoding="utf-8-sig")
        value = yaml.load(text, Loader=UniqueKeyLoader)
        if not isinstance(value, dict):
            raise InterfaceBlocked("YAML root mapping required")
        strict_vector(value.get("lidar_pose"), 6, f"{side}.lidar_pose")
        if not isinstance(value.get("vehicles"), dict):
            raise InterfaceBlocked("vehicles mapping required")
        xyz, header = read_pcd_xyz(pcd)
        decoded_hash = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
        if decoded_hash != match[0][side]["pcd"]["xyz_sha256"]:
            raise InterfaceBlocked("decoded XYZ differs from R2-03")
        rows[side] = {"agent": agent, "sensor": "ouster", "frame_key": FRAME_KEY,
                      "identity_basis": "fixed path + R2-03 fingerprints; not an embedded physical sensor/time assertion",
                      "pcd": {"path": pcd.relative_to(root).as_posix(), "bytes": pcd.stat().st_size,
                              "sha256_before": sha256(pcd), "header": header,
                              "decoded_xyz_sha256": decoded_hash, "stats": point_stats(xyz)},
                      "yaml": {"path": meta.relative_to(root).as_posix(), "bytes": meta.stat().st_size,
                               "sha256_before": sha256(meta), "raw_text": text,
                               "root_fields": {str(k): type(v).__name__ for k, v in value.items()},
                               "lidar_pose_raw": value["lidar_pose"],
                               "lidar_pose_validation": vector_inventory(value["lidar_pose"]),
                               "object_count": len(value["vehicles"]),
                               "vehicles_raw": value["vehicles"]}}
        arrays[side], raw[side] = np.asarray(xyz, dtype=np.float64), value
    decoded = rows["vehicle"]["pcd"]["decoded_xyz_sha256"]
    duplicate_decoded = {r["vehicle"]["pcd"]["xyz_sha256"] for r in audit["pairs"]
                         if r["vehicle"]["pcd"]["sha256"] in duplicate_hashes}
    if decoded in duplicate_decoded:
        raise InterfaceBlocked("fixed vehicle decoded XYZ belongs to a known duplicate group")
    evidence = {"path": R2_03, "sha256": sha256(audit_path), "known_duplicate_pcd_hashes": sorted(duplicate_hashes),
                "fixed_vehicle_in_known_duplicate_groups": False,
                "scope": "R2-03 known groups only; not a global dataset uniqueness proof"}
    return rows, arrays, raw, evidence


def annotation_diagnostic(raw, paths, transform, official):
    result = {"interpretation": "official_code_interpretation_only", "status": "BLOCKED",
              "physical_annotation_acceptance": "NOT_ESTABLISHED", "sides": {}}
    corners = {}
    for side in ["vehicle", "uav"]:
        objs = raw[side]["vehicles"]
        for oid, obj in objs.items():
            if not isinstance(obj, dict):
                raise InterfaceBlocked(f"object {oid}: mapping required")
            for field in ["location", "center", "extent", "angle"]:
                strict_vector(obj.get(field), 3, f"{side}.{oid}.{field}")
            if any(x <= 0 for x in obj["extent"]):
                raise InterfaceBlocked("positive extent required")
            if "obj_type" in obj and not isinstance(obj["obj_type"], str):
                raise InterfaceBlocked("object class must be string")
        t = transform if side == "uav" else None
        # Exact pinned converter body, reading original YAML after strict prevalidation.
        converted = official.load_annotation_objects(str(paths[side]), transform=t)
        side_rows, side_corners = [], []
        for oid, obj in objs.items():
            pose = [obj["location"][i] + obj["center"][i] for i in range(3)] + obj["angle"]
            object_matrix = official.x1_to_x2(pose, np.eye(4))
            if t is not None:
                object_matrix = t @ object_matrix
            # Same official primitives and operations, exposing converter intermediates.
            bbx = official.create_bbx(obj["extent"]).T
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
            points = np.dot(object_matrix, bbx).T[:, :3]
            reduced = official.corner_to_center(points[None], order="lwh")[0]
            output = converted[oid]
            actual = np.asarray(output["loc"] + output["dims"] + [output["heading"]])
            if not np.isfinite(points).all() or not np.isfinite(actual).all():
                raise InterfaceBlocked("nonfinite official box result")
            error = float(np.max(np.abs(reduced-actual)))
            if error > CONFIG["matrix_atol"]:
                raise InterfaceBlocked("converter/intermediate mismatch")
            side_rows.append({"object_id": oid, "raw": obj, "object_pose_code_order": pose,
                              "field_validation": {f: vector_inventory(obj[f]) for f in ["location", "center", "extent", "angle"]},
                              "object_to_vehicle_matrix": object_matrix.tolist(),
                              "corners_vehicle_frame": points.tolist(), "official_converter_output": output,
                              "converter_intermediate_max_abs": error,
                              "note": "converter reduces tilted 8 corners to yaw-only center/dimensions; projection uses full 8 corners"})
            side_corners.append(points)
        result["sides"][side] = side_rows
        corners[side] = side_corners
    # Same ID is observed, not asserted to be an independent cross-agent match.
    v = {r["object_id"]: r for r in result["sides"]["vehicle"]}
    u = {r["object_id"]: r for r in result["sides"]["uav"]}
    result["same_id_center_separations_code_units"] = {
        str(k): float(np.linalg.norm(np.array(v[k]["official_converter_output"]["loc"])
                                    - u[k]["official_converter_output"]["loc"])) for k in sorted(v.keys() & u.keys(), key=str)}
    result["status"] = "official_code_interpretation_only"
    return result, corners


def plot_bev(arrays, corners, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"vehicle": "#226bb0", "uav": "#d76721"}
    fig, ax = plt.subplots(figsize=(12, 9))
    plotted = {}
    for side, xyz in arrays.items():
        idx = selected_indices(len(xyz), CONFIG["plot_max_points_per_agent"])
        plotted[side] = {"count": len(idx), "first_index": int(idx[0]), "last_index": int(idx[-1]),
                         "index_sha256": hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()}
        ax.scatter(xyz[idx, 0], xyz[idx, 1], s=0.5, alpha=0.45, color=colors[side], rasterized=True,
                   label="Vehicle 1: original" if side == "vehicle" else "UAV 2: official pose/list -> vehicle")
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for side, boxes in corners.items():
        for box in boxes:
            for i, j in edges:
                ax.plot(box[[i, j], 0], box[[i, j], 1], color=colors[side], lw=0.8,
                        ls="-" if side == "vehicle" else "--")
    all_xy = np.concatenate([p[:, :2] for p in arrays.values()] +
                            [b[:, :2] for boxes in corners.values() for b in boxes])
    low, high = all_xy.min(0), all_xy.max(0)
    pad = np.maximum((high-low) * 0.03, 1)
    ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]),
           xlabel="Vehicle-frame code X (length unit unverified)",
           ylabel="Vehicle-frame code Y (physical axes unverified)",
           title=f"V2U4Real train / {SCENE} / {FRAME_KEY}\nOfficial transform software diagnostic; physical alignment NOT ESTABLISHED")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if corners:
        handles += [Line2D([0], [0], color=colors[s], ls="-" if s == "vehicle" else "--") for s in corners]
        labels += [f"{s} boxes: official_code_interpretation_only" for s in corners]
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    fig.text(0.5, 0.02, "Same file key != measured synchronization. Units / axes / applied calibration unverified. No registration correction.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"path": output.name, "sha256": sha256(output), "sampled_points": plotted,
            "range_basis": "full XYZ XY bounds plus all official corners; shared limits, no clipping", "equal_aspect": True,
            "xlim": [float(low[0]-pad[0]), float(high[0]+pad[0])],
            "ylim": [float(low[1]-pad[1]), float(high[1]+pad[1])]}


def run(root=ROOT, output=OUT):
    if not output.resolve().is_relative_to((root / "code/artifacts/audit").resolve()):
        raise InterfaceBlocked("diagnostic output must stay in code/artifacts/audit")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", CONFIG)
    result = {"schema_version": "V2U4Real-R2-04A-v1", "dataset": "VJiaLi/V2U4Real",
              "dataset_version": DATA_VERSION, "split": "train", "scene": SCENE, "frame_key": FRAME_KEY,
              "config": CONFIG, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                             "device": "CPU", "gpu_used": False, "locked_test_accessed": False},
              "executed_script_sha256": sha256(Path(__file__)),
              "scope": "single real published same-key association; software mathematics only",
              "limitations": LIMITS, "errors": [],
              "input_integrity_status": "NOT_RUN", "official_source_status": "NOT_RUN",
              "software_transform_status": "NOT_RUN", "annotation_diagnostic_status": "NOT_RUN",
              "physical_alignment_status": "NOT_ESTABLISHED", "task_execution_status": "BLOCKED"}
    stage = "official_source_status"
    inputs = {}
    try:
        official, identity = official_functions(root)
        result["official_source"] = identity
        result[stage] = "PASS"
        stage = "input_integrity_status"
        inputs, arrays, raw, duplicate_evidence = read_inputs(root)
        result.update(inputs=inputs, duplicate_evidence=duplicate_evidence)
        result[stage] = "PASS"
        stage = "software_transform_status"
        vp, up = raw["vehicle"]["lidar_pose"], raw["uav"]["lidar_pose"]
        # Actual list/list call, both original lists; source=UAV, target=vehicle.
        tv, tu = official.x_to_world(vp), official.x_to_world(up)
        relative = official.x1_to_x2(up, vp)
        independent = np.linalg.solve(tv, tu)  # Independent solve vs official inv+dot.
        matrices = {"T_world_from_vehicle": tv, "T_world_from_uav": tu, "T_vehicle_from_uav": relative}
        result["transforms"] = {k: v.tolist() for k, v in matrices.items()}
        result["matrix_checks"] = {k: matrix_stats(v) for k, v in matrices.items()}
        error = float(np.max(np.abs(relative-independent)))
        result["relative_matrix_check"] = {"official_call": "x1_to_x2(uav_pose_list, vehicle_pose_list)",
                                           "branch": "list/list", "source": "UAV local code frame", "target": "vehicle local code frame",
                                           "order": "inv(T_world_from_vehicle) @ T_world_from_uav",
                                           "independent_method": "solve(T_world_from_vehicle, T_world_from_uav)",
                                           "max_abs_error": error, "atol": CONFIG["matrix_atol"],
                                           "status": "PASS" if error <= CONFIG["matrix_atol"] else "BLOCKED"}
        # Save raw poses + generated matrices before applying any points.
        write_json(output / "transform_inputs.json", {"poses": {"vehicle": vp, "uav": up}, "transforms": result["transforms"], "config": CONFIG})
        projected, checks = geometry_checks(arrays["uav"], relative)
        result["point_checks"] = checks
        if error > CONFIG["matrix_atol"] or checks["status"] != "PASS":
            raise InterfaceBlocked("software numerical checks failed; tolerances unchanged")
        result[stage] = "PASS"
        stage = "annotation_diagnostic_status"
        corners = {}
        try:
            paths = {s: root / row["yaml"]["path"] for s, row in inputs.items()}
            ann, corners = annotation_diagnostic(raw, paths, relative, official)
            result["annotation_diagnostic"] = ann
            result[stage] = ann["status"]
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            result[stage] = "BLOCKED"
            result["errors"].append({"stage": stage, "error": str(exc)})
        stage = "visualization_status"
        result["visualization"] = plot_bev({"vehicle": arrays["vehicle"], "uav": projected}, corners, output / "bev_overlay.png")
        result[stage] = "PASS"
        result["task_execution_status"] = "COMPLETED" if not result["errors"] else "PARTIAL"
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        result[stage] = "BLOCKED"
        result["errors"].append({"stage": stage, "error": str(exc)})
    finally:
        # Input integrity and official source are rechecked even after downstream failure.
        for row in inputs.values():
            for kind in ["pcd", "yaml"]:
                entry = row[kind]
                path = root / entry["path"]
                after = sha256(path) if path.is_file() else None
                entry["sha256_after"] = after
                entry["unchanged"] = after == entry["sha256_before"]
                if not entry["unchanged"]:
                    result["input_integrity_status"] = "BLOCKED"
                    result["task_execution_status"] = "BLOCKED"
                    result["errors"].append({"stage": "final_input_integrity", "error": entry["path"]})
        try:
            verify_source(root)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["official_source_status"] = "BLOCKED"
            result["task_execution_status"] = "BLOCKED"
            result["errors"].append({"stage": "final_source_integrity", "error": str(exc)})
    write_json(output / "result.json", result)
    write_summary(result, output / "summary.md")
    return result


def write_summary(result, path):
    lines = ["# STEP R2-04A — 真实单帧官方变换几何诊断", "",
             f"任务执行：**{result['task_execution_status']}**。数学检查通过也不能证明真实物理对齐。",
             "固定训练场景 `2025-07-18-12-53_2`、发布键 `1752814590.2`；agent 1 车辆、agent 2 UAV，均 Ouster。", ""]
    for k in ["input_integrity_status", "official_source_status", "software_transform_status", "annotation_diagnostic_status", "physical_alignment_status", "task_execution_status"]:
        lines.append(f"- `{k}`：`{result[k]}`")
    lines += ["", "## 输入证据（完整字段与原文见 result.json）", ""]
    for side, row in result.get("inputs", {}).items():
        lines += [f"- {side}：{row['pcd']['stats']['count']} 点；位姿 `{row['yaml']['lidar_pose_raw']}`；标注 {row['yaml']['object_count']} 个。"]
        for kind in ["pcd", "yaml"]:
            entry = row[kind]
            lines += [f"  - `{entry['path']}`：{entry['bytes']} bytes，SHA-256 `{entry['sha256_before']}`，前后不变 `{entry.get('unchanged')}`。"]
    if "point_checks" in result:
        c = result["point_checks"]
        lines += ["", "## 数学结果", "", "实际调用固定官方 `x1_to_x2(uav_pose, vehicle_pose)` list/list 分支。",
                  "矩阵将 UAV 局部列向量映射到车辆局部：`inv(T_world_from_vehicle) @ T_world_from_uav`。车辆点云保持原坐标。",
                  "", "```text", str(np.array(result["transforms"]["T_vehicle_from_uav"])), "```", "",
                  f"独立 solve 的矩阵最大差：{result['relative_matrix_check']['max_abs_error']:.15g}；全部 UAV 点正逆最大误差 {c['roundtrip_max_abs']:.15g}，坐标 RMSE {c['roundtrip_coordinate_rmse']:.15g}。",
                  f"确定性 {len(c['distance_subset_indices'])} 点的 {c['distance_pair_count']} 对距离最大差 {c['distance_max_abs_error']:.15g}。容差固定见 config，未自动放宽。"]
    if "annotation_diagnostic" in result:
        lines += ["", "## 标注诊断", "", "执行哈希验证后提取的官方函数原体；未执行 converter main，也没有合并或覆盖同 ID 框。",
                  "`location+center`、`angle` 按代码原顺序、`extent/2` 生成角点，再执行官方 corner_to_center；保存完整角点及最终简化框。",
                  f"同 ID 中心间距（代码单位，仅观察）：`{result['annotation_diagnostic']['same_id_center_separations_code_units']}`。",
                  "框只标记 `official_code_interpretation_only`；不把同 ID 当作已验证匹配，不验收跨端框对齐。"]
    lines += ["", "## 图与解释边界", "", "`bev_overlay.png` 是鸟瞰图（从上方查看 XY 平面）；每端最多 20,000 个确定性点，一致范围、等比例坐标，完整框角点投影。",
              "图像只能帮助发现异常。真实单位、轴向、公共原点、跨端同步及配准修正未建立；不得据此手调矩阵或宣布标定通过。",
              "点变换复用 float64 工具；官方转换器的 torch/float32 投影没有执行，不能宣称完整导出器逐位复现。", "",
              "## 未解决事项", ""] + [f"- {x}" for x in LIMITS]
    lines += ["", "## 错误", "", json.dumps(result["errors"], ensure_ascii=False), "",
              "CPU-only；GPU、locked_test、AirFogSim、无线服务与训练均未使用。独立测试见 validation.json；实际命令和输出见 run.log。",
              "本任务到此停止，等待研究者审阅；不自动进入 R2-04B。", ""]
    path.write_text("\n".join(lines), encoding="utf8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_status")}, sort_keys=True))
    return 0 if result["task_execution_status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
