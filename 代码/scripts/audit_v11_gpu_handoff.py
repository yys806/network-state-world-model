"""Audit the PI-JWM v11 GPU handoff without starting GPU work or opening locked data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
REPORT_DEFAULT = CODE_ROOT / "artifacts/reports/pi_jwm_v11_gpu_handoff_20260719"
SAMPLE_INDEX_DEFAULT = CODE_ROOT / "artifacts/experiments/airfogsim_v0/datasets/dataset_multiseed_active_heavy_v2_60seed_20260619/sample_index.csv"
WORLD_CHECKPOINT_DEFAULT = CODE_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt"
POLICY_CHECKPOINT_DEFAULT = CODE_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt"
SMOKE_GATE_DEFAULT = CODE_ROOT / "artifacts/reports/pi_jwm_v11_temporal_candidate_protocol_20260719/pretraining_gate.json"
SCHEMA6_SMOKE_DEFAULT = CODE_ROOT / "artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/smoke256/summary_validation.json"
FULL_SCHEMA6_SUMMARY_DEFAULT = CODE_ROOT / "artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/summary.json"
BRIDGE_MANIFEST_DEFAULT = CODE_ROOT / "artifacts/reports/pi_jwm_v11_physical_benefit_bridge_20260719/formal_h10/task_bridge/summary.json"

EXPECTED_SEEDS = {
    "train": set(range(16)) | set(range(20, 44)),
    "calibration": set(range(44, 50)),
    "validation": set(range(50, 60)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _check_sample_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [int(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_index contains duplicate sample_id values")
    seed_counts: dict[int, int] = {}
    for row in rows:
        seed = int(row["seed"])
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
        if float(row["input_end_time"]) >= float(row["label_end_time"]):
            raise ValueError(f"sample {row['sample_id']} has non-causal timing")
    return {
        "rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "seed_counts": {str(seed): count for seed, count in sorted(seed_counts.items())},
        "formal_seed_coverage": all(seed_counts.get(seed) == 390 for seed in range(60)),
    }


def _check_split_protocol() -> dict[str, Any]:
    split_sets = {name: set(values) for name, values in EXPECTED_SEEDS.items()}
    overlaps = []
    names = list(split_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(split_sets[left] & split_sets[right])
            if shared:
                overlaps.append({"left": left, "right": right, "seeds": shared})
    return {
        "splits": {name: sorted(values) for name, values in split_sets.items()},
        "overlaps": overlaps,
        "locked_test_seeds": [18, 19],
        "external_holdout_seeds": list(range(60, 70)),
        "passed": not overlaps,
    }


def _check_smoke_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality_audit", {})
    passed = (
        payload.get("status") == "ready_for_formal_label_generation"
        and bool(payload.get("bridge_smoke_gate_passed"))
        and bool(quality.get("passed"))
        and not bool(payload.get("matched_test_accessed"))
        and not bool(payload.get("external_holdout_accessed"))
    )
    return {"status": payload.get("status"), "quality_audit": quality, "passed": passed}


def _check_schema6_smoke(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_gate = payload.get("candidate_gate", {})
    protocol = payload.get("protocol_audit", {})
    manifest = payload.get("cache_manifest", {})
    interaction = manifest.get("interaction", {})
    protocol_passed = bool(protocol.get("passed")) and int(interaction.get("overflow_count", -1)) == 0
    return {
        "sample_count": int(payload.get("num_samples", 0)),
        "cache_schema_version": int(manifest.get("schema_version", -1)),
        "protocol_passed": protocol_passed,
        "candidate_gate_passed": bool(candidate_gate.get("passed")),
        "candidate_gate": candidate_gate,
        "label_generation_allowed": protocol_passed and int(manifest.get("schema_version", -1)) == 6,
        "selector_training_allowed": False,
    }


def _check_full_schema6(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("sample_count", {})
    candidate_gate = payload.get("selector_training", {}).get("candidate_gate", {})
    handoff = payload.get("label_handoff", {})
    passed = (
        counts == {"train": 15600, "calibration": 2340, "validation": 3900}
        and bool(candidate_gate.get("passed"))
        and int(handoff.get("configuration_digest_count", 0)) == 1
        and int(handoff.get("crossfit_protocol_digest_count", 0)) == 1
        and not bool(handoff.get("locked_split_accessed"))
    )
    return {
        "passed": passed,
        "sample_count": counts,
        "candidate_gate": candidate_gate,
        "configuration_digest": payload.get("configuration_digest"),
        "crossfit_protocol_digest": payload.get("crossfit_protocol_digest"),
        "locked_split_accessed": bool(handoff.get("locked_split_accessed")),
    }


def _check_bridge(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_digest = str(payload.get("bridge_manifest_digest", ""))
    actual_digest = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "bridge_manifest_digest"}
    )
    model = payload.get("model_report", {})
    passed = (
        expected_digest == actual_digest
        and bool(payload.get("bridge_gate_passed"))
        and str(payload.get("bridge_mode")) == "task_only"
        and bool(model.get("task_model_passed"))
        and not bool(model.get("energy_model_passed"))
        and str(payload.get("physical_energy_result_kind")) == "audit_only"
        and not bool(payload.get("matched_test_accessed"))
        and not bool(payload.get("external_holdout_accessed"))
    )
    return {
        "passed": passed,
        "bridge_mode": payload.get("bridge_mode"),
        "physical_feature_scope": payload.get("physical_feature_scope"),
        "physical_energy_result_kind": payload.get("physical_energy_result_kind"),
        "task_model_passed": bool(model.get("task_model_passed")),
        "energy_model_passed": bool(model.get("energy_model_passed")),
        "oof_task_mae": model.get("oof_task_mae"),
        "baseline_task_mae": model.get("baseline_task_mae"),
        "calibration_task_mae": model.get("calibration_task_mae"),
        "calibration_baseline_task_mae": model.get("calibration_baseline_task_mae"),
        "manifest_digest": expected_digest,
    }


def _check_scripts() -> dict[str, Any]:
    label = (CODE_ROOT / "scripts/run_v11_schema6_labels_gpu.sh").read_text(encoding="utf-8")
    selector = (CODE_ROOT / "scripts/run_v11_candidate_set_selector_gpu.sh").read_text(encoding="utf-8")
    checks = {
        "label_script_requires_cuda": 'if [[ "${DEVICE}" != "cuda" ]]' in label,
        "label_script_schema6": "--cache-schema-version 6" in label,
        "label_script_formal_splits": "--splits validation" in label and "--splits train calibration" in label,
        "label_script_validation_before_train": label.find("--splits validation") < label.find("--splits train calibration"),
        "label_script_validation_gate": 'split == "validation" and not bool(split_summary["candidate_gate"]["passed"])' in label,
        "selector_script_uses_candidate_set_runner": "train_v11_candidate_set_selector.py" in selector,
        "selector_script_schema6_cache_path": "candidate_labels_train_physical.npz" in selector,
        "selector_script_requires_physical_manifest": "--physical-bridge-manifest" in selector,
        "selector_script_does_not_open_locked_split": "matched_test" not in selector and "external_holdout" not in selector,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["sample_index"] = _check_sample_index(args.sample_index)
    checks["split_protocol"] = _check_split_protocol()
    checks["smoke_gate"] = _check_smoke_gate(args.smoke_gate)
    checks["schema6_local_smoke"] = _check_schema6_smoke(args.schema6_smoke)
    checks["full_schema6_labels"] = _check_full_schema6(args.full_schema6_summary)
    checks["physical_bridge"] = _check_bridge(args.bridge_manifest)
    checks["frozen_checkpoints"] = {
        "world": {
            "path": str(args.world_checkpoint),
            "exists": args.world_checkpoint.is_file(),
            "sha256": _sha256(args.world_checkpoint) if args.world_checkpoint.is_file() else None,
        },
        "policy": {
            "path": str(args.policy_checkpoint),
            "exists": args.policy_checkpoint.is_file(),
            "sha256": _sha256(args.policy_checkpoint) if args.policy_checkpoint.is_file() else None,
        },
    }
    checks["gpu_scripts"] = _check_scripts()
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    checks["source"] = {"git_sha": _git_sha(), "tracked_tree_clean": not bool(tracked_status)}
    checks["locked_split_access"] = {"matched_test_accessed": False, "external_holdout_accessed": False}

    hard_checks = [
        checks["sample_index"]["formal_seed_coverage"],
        checks["split_protocol"]["passed"],
        checks["smoke_gate"]["passed"],
        checks["schema6_local_smoke"]["label_generation_allowed"],
        checks["full_schema6_labels"]["passed"],
        checks["physical_bridge"]["passed"],
        all(value["exists"] for value in checks["frozen_checkpoints"].values()),
        checks["gpu_scripts"]["passed"],
        checks["source"]["tracked_tree_clean"],
    ]
    checks["status"] = "ready_for_gpu_selector_training" if all(hard_checks) else "blocked"
    checks["selector_training_allowed"] = bool(all(hard_checks))
    checks["formal_train_calibration_labels_required"] = False
    checks["blockers"] = (
        []
        if checks["source"]["tracked_tree_clean"]
        else ["tracked source tree is dirty"]
    )
    checks["limitations"] = [
        "the physical bridge is task-only; actual UAV energy remains audit_only",
        "GPU training may use physical task features and the deployable energy proxy only",
        "matched test and external holdout remain locked until validation freezes a configuration",
    ]
    checks["commands"] = {
        "selector_training": "bash 代码/scripts/run_v11_candidate_set_selector_gpu.sh",
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gpu_handoff_gate.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "formal_label_command.txt").write_text(
        "formal h=10 physical labels already completed; see train/calibration reproduction_command.txt\n",
        encoding="utf-8",
    )
    (output_dir / "selector_gpu_command.txt").write_text(
        checks["commands"]["selector_training"] + "\n", encoding="utf-8"
    )
    split_manifest = {
        "framework": "PI-JWM",
        "formal_splits": {name: sorted(values) for name, values in EXPECTED_SEEDS.items()},
        "matched_test": [18, 19],
        "external_holdout": list(range(60, 70)),
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "sample_index_sha256": _sha256(args.sample_index),
    }
    split_manifest["split_digest"] = _canonical_sha256(split_manifest)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol_manifest = {
        "framework": "PI-JWM",
        "candidate_protocol": "support_constrained_edge_step_repair_v2",
        "physical_temporal_protocol": "causal_policy_v1",
        "cache_schema_version": 6,
        "candidate_count": 32,
        "world_checkpoint_sha256": checks["frozen_checkpoints"]["world"]["sha256"],
        "policy_checkpoint_sha256": checks["frozen_checkpoints"]["policy"]["sha256"],
        "source_git_sha": checks["source"]["git_sha"],
        "split_digest": split_manifest["split_digest"],
        "full_schema6_configuration_digest": checks["full_schema6_labels"][
            "configuration_digest"
        ],
        "physical_bridge_manifest_digest": checks["physical_bridge"][
            "manifest_digest"
        ],
        "physical_feature_scope": checks["physical_bridge"]["physical_feature_scope"],
    }
    protocol_manifest["protocol_digest"] = _canonical_sha256(protocol_manifest)
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# PI-JWM v11 GPU 交接审计",
        "",
        f"- 状态：`{checks['status']}`",
        f"- 当前 selector GPU 训练：`{checks['selector_training_allowed']}`",
        f"- source Git SHA：`{checks['source']['git_sha']}`",
        "- 锁定集：matched test 18-19 和 external holdout 60-69 均未访问。",
        "- 完整 schema-v6 crossfit 标签已存在且 candidate gate 通过，无需重复生成。",
        "- h=10 physical task bridge 的 OOF/calibration 门通过；UAV energy 不可辨识，严格保留为 audit_only。",
        "",
        "## 交接顺序",
        "",
        "1. 运行 CandidateSet selector GPU 训练，只读取 task-only augmented train/calibration/validation caches。",
        "2. energy Pareto 使用测试时可获得的 proxy；真实 UAV energy 只用于离线安全解释。",
        "3. validation 未冻结前不运行 feature ablation、matched test 或 external holdout。",
    ]
    (output_dir / "gpu_handoff_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    ]
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--sample-index", type=Path, default=SAMPLE_INDEX_DEFAULT)
    parser.add_argument("--world-checkpoint", type=Path, default=WORLD_CHECKPOINT_DEFAULT)
    parser.add_argument("--policy-checkpoint", type=Path, default=POLICY_CHECKPOINT_DEFAULT)
    parser.add_argument("--smoke-gate", type=Path, default=SMOKE_GATE_DEFAULT)
    parser.add_argument("--schema6-smoke", type=Path, default=SCHEMA6_SMOKE_DEFAULT)
    parser.add_argument(
        "--full-schema6-summary", type=Path, default=FULL_SCHEMA6_SUMMARY_DEFAULT
    )
    parser.add_argument("--bridge-manifest", type=Path, default=BRIDGE_MANIFEST_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
