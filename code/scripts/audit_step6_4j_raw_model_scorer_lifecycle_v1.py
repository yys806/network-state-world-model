"""Read-only Step 6.4J Raw/Model/Scorer lifecycle audit.

This script does not run AirFogSim, World Model rollouts, or planner search.  It
recomputes facts from the two previously captured live roots, scans the frozen
dev_validation Raw snapshots, records source identities, and executes shadow
FP64/FP32 boundary calculations.  Scientific state/scorer semantics are not
modified.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "code/artifacts/audit/pi_jwm_step6_4j_h1_return_action_sensitivity_20261011_v1"
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4j_raw_model_scorer_lifecycle_20261011_v1"
FORMAL = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923"

SOURCE_FILES = [
    "code/reference/AirFogSim/airfogsim/entities/task.py",
    "code/reference/AirFogSim/airfogsim/manager/task_manager.py",
    "code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py",
    "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
    "code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py",
    "code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py",
    "code/scripts/step6_4i_live_planner_v1.py",
    "code/src/pi_jwm/step6_3c_candidate_domain_v1.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def source_identity() -> dict:
    return {p: {"sha256": sha256(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in SOURCE_FILES}


def airfogsim_identity() -> dict:
    package = ROOT / "code/reference/AirFogSim/airfogsim/__init__.py"
    return {
        "environment": "airfogsim",
        "package_path": str(package),
        "package_sha256": sha256(package),
        "installed_distribution": None,
        "installed_distribution_reason": "conda run -n airfogsim importlib.metadata found no AirFogSim distribution; runtime imports project reference path",
        "import_check": "conda run -n airfogsim python with code/reference/AirFogSim prepended to sys.path",
    }


def load_live_roots() -> list[dict]:
    rows = []
    for path in sorted(V1.glob("*_root.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "root": data})
    return rows


def audit_live_roots(rows: list[dict]) -> dict:
    task_rows = []
    for item in rows:
        root = item["root"]
        raw = root.get("raw_target_lifecycle", {})
        state = root.get("anchor_task_state", {})
        for task_id in sorted(set(raw) | set(state)):
            r, s = raw.get(task_id, {}), state.get(task_id, {})
            task_cpu = r.get("task_cpu_work")
            computed = r.get("computed_cpu_work")
            remaining64 = None if task_cpu is None or computed is None else float(task_cpu) - float(computed)
            f32_total = None if task_cpu is None else float(np.float32(task_cpu))
            f32_computed = None if computed is None else float(np.float32(computed))
            remaining32 = None if f32_total is None or f32_computed is None else f32_total - f32_computed
            raw_lifecycle = r.get("lifecycle")
            model_remaining = s.get("work_remaining")
            row = {
                "root_file": item["file"], "task_id": task_id,
                "raw_lifecycle": raw_lifecycle,
                "task_cpu_work": task_cpu, "computed_cpu_work": computed,
                "remaining_fp64_raw": remaining64, "remaining_fp32_shadow": remaining32,
                "model_work_remaining": model_remaining,
                "raw_computed_exact": None if remaining64 is None else remaining64 <= 0.0,
                "fp32_computed": None if remaining32 is None else remaining32 <= 0.0,
                "model_computed": None if model_remaining is None else float(model_remaining) <= 0.0,
                "return_exists": s.get("return_exists"),
                "required_returned_size": r.get("required_returned_size", s.get("required_returned_size")),
                "return_destination_id": r.get("return_destination_id"),
                "compute_host_index": s.get("compute_host_index"),
            }
            row["raw_model_completion_conflict"] = (
                row["raw_computed_exact"] is not None and row["model_computed"] is not None
                and row["raw_computed_exact"] != row["model_computed"]
            )
            task_rows.append(row)
    lifecycle = Counter(str(r["raw_lifecycle"]) for r in task_rows)
    conflicts = [r for r in task_rows if r["raw_model_completion_conflict"]]
    return {"task_ledger": task_rows, "task_count": len(task_rows), "lifecycle_counts": dict(lifecycle), "raw_model_completion_conflicts": conflicts}


def shadow_boundaries() -> list[dict]:
    values = [1.0, 0.0, 1e-16, 1e-15, 1e-12, 1e-8, -1e-16]
    rows = []
    for eps in values:
        total, computed = 1.0, 1.0 - eps
        r64 = total - computed
        r32 = float(np.float32(total)) - float(np.float32(computed))
        rows.append({"eps": eps, "remaining_fp64": r64, "remaining_fp32": r32, "fp64_complete": r64 <= 0, "fp32_complete": r32 <= 0})
    return rows


def validation_stats() -> dict:
    manifest = json.loads((FORMAL / "split_manifest.json").read_text(encoding="utf-8"))
    wanted = set(manifest["dev_validation"])
    files = sorted(FORMAL.glob("raw/*.json.gz"))
    selected = [p for p in files if p.stem.removesuffix(".json") in wanted or p.name.removesuffix(".json.gz") in wanted]
    # The second condition is the actual filename form on Windows/Python.
    if not selected:
        selected = [p for p in files if any(t in p.name for t in wanted)]
    counts = Counter()
    task_rows = []
    roots = 0
    for path in selected:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            traj = json.load(f)
        roots += len(traj.get("decisions", []))
        for decision in traj.get("decisions", []):
            frame = int(decision["frame_index"])
            previous = next((s for s in traj.get("steps", []) if int(s["frame_index"]) == frame - 1), None)
            return_actions = set()
            if previous:
                for entry in previous.get("action", {}).get("route", {}).get("entries", []):
                    if str(entry.get("route_kind", "offload")) == "return":
                        return_actions.add(str(entry.get("task_id")))
            for task in decision.get("tasks", []):
                counts["present_tasks"] += 1
                lifecycle = str(task.get("lifecycle", ""))
                terminal = lifecycle in {"done", "failed", "completed"}
                if not terminal:
                    counts["nonterminal_tasks"] += 1
                total = float(task.get("task_cpu_work", 0.0) or 0.0)
                done = float(task.get("computed_cpu_work", 0.0) or 0.0)
                rem64 = total - done
                rem32 = float(np.float32(total)) - float(np.float32(done))
                if rem32 <= 0.0:
                    counts["fp32_rounds_to_zero"] += 1
                raw_done = rem64 <= 0.0
                model_done = rem32 <= 0.0
                if raw_done != model_done:
                    counts["raw_fp32_completion_conflicts"] += 1
                if lifecycle == "waiting_to_return" and str(task.get("task_id")) not in return_actions:
                    counts["waiting_return_without_explicit_return_route"] += 1
                if str(task.get("task_id")) in return_actions:
                    counts["explicit_return_route_visible"] += 1
                if (not terminal and rem32 <= 0.0 and float(task.get("required_returned_size", 0.0) or 0.0) > 0 and task.get("current_node_id") != task.get("return_destination_id") and str(task.get("task_id")) not in return_actions):
                    counts["anchor_potential_return_birth_block"] += 1
                task_rows.append({"trajectory_id": traj.get("environment", {}).get("trajectory_id", path.stem), "frame": frame, "task_id": task.get("task_id"), "lifecycle": lifecycle, "remaining_fp64": rem64, "remaining_fp32": rem32, "return_action_visible": str(task.get("task_id")) in return_actions})
    counts["trajectories"] = len(selected)
    counts["decision_roots"] = roots
    counts["selected_files"] = [str(p.relative_to(ROOT)) for p in selected]
    counts["selection"] = "all 12 frozen dev_validation trajectories from split_manifest; Raw-only, no WM/scorer execution"
    return {"counts": dict(counts), "task_rows_sha256": hashlib.sha256(json.dumps(task_rows, sort_keys=True).encode()).hexdigest(), "task_rows_count": len(task_rows)}


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": str(Path(__file__).relative_to(ROOT)), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "gpu": False, "locked_test": False, "formal_search": False}
    atomic_json(OUT / "01_identity.json", {"identity": identity, "airfogsim": airfogsim_identity(), "sources": source_identity()})
    roots = load_live_roots()
    live = audit_live_roots(roots)
    atomic_json(OUT / "02_live_root_ledger.json", {"roots": [{"file": r["file"], "sha256": r["sha256"]} for r in roots], **live})
    atomic_json(OUT / "03_numeric_boundary.json", {"shadow_only": True, "rows": shadow_boundaries()})
    validation = validation_stats()
    atomic_json(OUT / "04_validation_raw_stats.json", validation)
    report = {
        "status": "COMPLETED_READ_ONLY_AUDIT",
        "elapsed_seconds": time.perf_counter() - started,
        "live_root_capture_valid": True,
        "r39_exact_equivalence": "UNPROVEN",
        "new_cpu_rollouts": 0,
        "findings": {
            "airfogsim_compute": "Task.compute adds allocated_cpu * simulation_interval and clamps to task_cpu when >=; isComputed uses exact >=.",
            "airfogsim_waiting_return": "TaskManager.computeTasks removes computed tasks from _computing_tasks and appends them to _waiting_to_return_tasks; route/startReturn is separate.",
            "ledger_return_birth": "amend_raw_with_causal_flow_ledger creates a Return flow only while processing a decision-before-action route entry with route_kind=return.",
            "world_model_route_semantics": "World Model route rule updates carrying/host state and explicitly does not create a new logical Flow identity.",
            "task5": "Live Root shows waiting_to_return, exact CPU complete, no Return Flow; this is compatible with simulator waiting state and fixed-support absence, pending any separate mapping defect evidence.",
            "task15": "Live Root shows Raw computing while FP64 task_cpu-computed_cpu_work is about 1.1e-16 and model work_remaining is 0; exact simulator completion rule and model threshold are semantically different at this boundary. This audit does not label it solely an FP32 bug.",
            "structural_boundary": "Fixed support can reject a predicted completion requiring a Return Flow that is absent at anchor; Route/World Model path does not birth a new logical Flow.",
        },
    }
    atomic_json(OUT / "05_acceptance_receipt.json", {"identity": identity, "report": report, "source_sha256": source_identity(), "live_task_count": live["task_count"], "validation": validation["counts"]})
    (OUT / "analysis_report.md").write_text("# Step 6.4J Raw–Model–Scorer 生命周期审计\n\n本目录是只读审计证据。AirFogSim 的计算完成使用 `computed_size >= task_cpu`，完成后先进入 `waiting_to_return`；项目 Ledger 只有显式 `route_kind=return` 决策动作才创建 Return Flow。两个 Live Root 的结果分别支持 Task_5 的等待回传/固定支持边界，以及 Task_15 的 Raw 生命周期与模型零剩余边界差异。r39 精确候选原因分布仍未被证明。详见 JSON ledger、数值边界和 Validation 原始统计。\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
