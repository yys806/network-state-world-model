"""Build the evidence-backed local gate before PI-JWM R6 uses a GPU."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_gpu_training_protocol import build_default_gpu_training_protocol  # noqa: E402


PREFLIGHT_ROOT = CODE_ROOT / "artifacts" / "preflight"
OUTPUT_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_online_gpu_readiness_v2"
SIX_MODE_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_online_six_mode_smoke_v1"
BENCHMARK_A = PREFLIGHT_ROOT / "pi_jwm_r6_online_32step_benchmark_v1"
BENCHMARK_B = PREFLIGHT_ROOT / "pi_jwm_r6_online_32step_benchmark_v2"
RESUME_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_resume_smoke_v1"
VALIDATION_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_validation_gate_smoke_v1"
LAUNCHER_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_launcher_stage1_dry_run_v1"
REWARD_SURROGATE_ROOT = PREFLIGHT_ROOT / "pi_jwm_r6_reward_surrogate_v1"
RUN_ID = "actor_critic__explicit_latent__seed_20260803"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def assess_six_mode_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, bool]:
    rows = list(summaries)
    return {
        "six_method_state_combinations": len(rows) == 6,
        "online_state_only": all(
            row.get("state_source") == "online_airfogsim_strict_dual_graph"
            for row in rows
        ),
        "real_policy_updates": all(
            int(row.get("update_count", 0)) >= 1
            and any(
                report.get("parameter_changed") is True
                and float(report.get("gradient_norm", 0.0)) > 0.0
                for report in row.get("reports", [])
            )
            for row in rows
        ),
        "nondefault_actions_executed": all(
            int(row.get("nondefault_selection_count", 0)) > 0 for row in rows
        ),
        "online_state_changed": all(
            int(row.get("distinct_explicit_state_count", 0)) >= 2 for row in rows
        ),
        "zero_hard_violations": all(
            int(row.get("hard_violation_count", -1)) == 0 for row in rows
        ),
        "checkpoint_reload_verified": all(
            row.get("checkpoint_reload_verified") is True for row in rows
        ),
        "world_model_frozen": all(
            row.get("world_model_updated") is False for row in rows
        ),
        "no_locked_test_access": all(
            row.get("locked_test_accessed") is False for row in rows
        ),
    }


def _summary(root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    return _read_json(root / run_id / "summary.json")


def main() -> int:
    protocol = build_default_gpu_training_protocol()
    expected_smoke_ids = {
        f"{method}__{mode}__seed_20260803"
        for method in protocol.methods
        for mode in protocol.state_modes
    }
    six_summaries = [
        _summary(SIX_MODE_ROOT, run_id) for run_id in sorted(expected_smoke_ids)
    ]
    six_checks = assess_six_mode_summaries(six_summaries)
    benchmark_a = _summary(BENCHMARK_A)
    benchmark_b = _summary(BENCHMARK_B)
    resume = _summary(RESUME_ROOT)
    validation = _summary(VALIDATION_ROOT)
    launcher = _read_json(LAUNCHER_ROOT / "launch_manifest.json")
    reward_surrogate = _read_json(REWARD_SURROGATE_ROOT / "summary.json")
    benchmark_checks = {
        "thirty_two_online_steps": int(benchmark_b.get("environment_steps", 0)) == 32,
        "four_updates": int(benchmark_b.get("update_count", 0)) == 4,
        "all_six_candidates_exercised": set(benchmark_b.get("candidate_selection_counts", {}))
        == {
            "airfogsim_default",
            "deadline_first",
            "priority_first",
            "load_balance",
            "rate_aware",
            "energy_conservative",
        },
        "cpu_actions_observed": int(
            benchmark_b.get("online_capture_counts", {}).get("cpu_actions", 0)
        ) > 0,
        "communication_events_observed": int(
            benchmark_b.get("online_capture_counts", {}).get("transfer_events", 0)
        ) > 0,
        "deterministic_candidate_counts": (
            benchmark_a.get("candidate_selection_counts")
            == benchmark_b.get("candidate_selection_counts")
        ),
        "deterministic_update_losses": (
            [row.get("total_loss") for row in benchmark_a.get("reports", [])]
            == [row.get("total_loss") for row in benchmark_b.get("reports", [])]
        ),
        "zero_hard_violations": int(benchmark_b.get("hard_violation_count", -1)) == 0,
    }
    validation_rows = list(validation.get("validation_reports", []))
    validation_checks = {
        "resume_from_two_to_four": (
            int(resume.get("resumed_from_environment_step", -1)) == 2
            and int(resume.get("environment_steps", -1)) == 4
            and int(resume.get("update_count", -1)) == 2
        ),
        "validation_only_gate_executed": len(validation_rows) == 1,
        "validation_checkpoint_eligible": bool(
            validation_rows
            and validation_rows[0].get("eligible") is True
            and validation_rows[0].get("improved") is True
            and int(validation_rows[0].get("hard_violation_count", -1)) == 0
        ),
        "best_checkpoint_written": (
            VALIDATION_ROOT / RUN_ID / "best_checkpoint.pt"
        ).is_file(),
    }
    launcher_checks = {
        "eighteen_isolated_runs": int(launcher.get("formal_run_count", 0)) == 18,
        "stage_one_budget_is_ten_thousand": int(
            launcher.get("target_environment_steps", 0)
        ) == 10000,
        "atomic_resume_enabled": launcher.get("atomic_resume_checkpoint") is True,
        "online_state_declared": (
            launcher.get("state_source") == "online_airfogsim_strict_dual_graph"
        ),
        "no_locked_test_access": launcher.get("locked_test_accessed") is False,
    }
    all_checks = {
        **{f"six_mode.{key}": value for key, value in six_checks.items()},
        **{f"benchmark.{key}": value for key, value in benchmark_checks.items()},
        **{f"validation.{key}": value for key, value in validation_checks.items()},
        **{f"launcher.{key}": value for key, value in launcher_checks.items()},
    }
    local_gate_passed = all(all_checks.values())
    seconds_per_step = float(benchmark_b["elapsed_seconds"]) / int(
        benchmark_b["environment_steps"]
    )
    estimate = {
        "measured_cpu_seconds_per_environment_step": seconds_per_step,
        "estimated_hours_per_100k_run_before_gpu_smoke": seconds_per_step * 100000 / 3600.0,
        "estimated_stage1_hours_at_parallel_6_before_gpu_smoke": (
            seconds_per_step * 10000 * 3 / 3600.0
        ),
        "interpretation": "wall-clock projection only; CUDA smoke must measure actual GPU memory and contention",
    }
    result = {
        "schema_version": "PIJWM-R6-online-GPU-readiness-v2",
        "protocol": protocol.to_dict(),
        "checks": all_checks,
        "local_gate_passed": local_gate_passed,
        "ready_for_gpu_smoke": local_gate_passed,
        "ready_for_formal_training_after_gpu_smoke": local_gate_passed,
        "formal_gpu_training_started": False,
        "locked_test_accessed_by_generated_evidence": False,
        "reward_surrogate_route": {
            "candidate_reward_surrogate_ready": reward_surrogate.get(
                "candidate_reward_surrogate_ready"
            ),
            "imagined_rollout_training_allowed": reward_surrogate.get(
                "imagined_rollout_training_allowed"
            ),
            "decision": "rejected; formal route uses live AirFogSim state and feedback",
        },
        "runtime_estimate": estimate,
        "evidence_roots": {
            "six_mode": str(SIX_MODE_ROOT),
            "benchmark_a": str(BENCHMARK_A),
            "benchmark_b": str(BENCHMARK_B),
            "resume": str(RESUME_ROOT),
            "validation": str(VALIDATION_ROOT),
            "launcher": str(LAUNCHER_ROOT),
        },
        "audit_note": (
            "A broad exploratory file search earlier resolved one locked-test path and timed out "
            "without returning file contents. No locked-test value was used by any artifact or check "
            "listed here; all generated evidence reads train/validation/calibration or synthetic tests only."
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(OUTPUT_ROOT / "readiness.json", result)
    lines = [
        "# PI-JWM R6 在线闭环 GPU 本地门禁",
        "",
        f"- 本地门禁：{'通过' if local_gate_passed else '不通过'}",
        f"- 可进入 GPU smoke：{str(local_gate_passed).lower()}",
        "- 正式 GPU 训练尚未启动；GPU smoke 通过后先跑 10k×18 的可续训阶段，再续到 100k。",
        "- 状态来源：每步 AirFogSim 在线严格双图；冻结教师轨迹只提供容量、归一化和协议，不提供动作分叉后的状态值。",
        "- 奖励来源：AirFogSim 实际执行反馈；反事实奖励代理因 5/6 候选无标签而弃用。",
        "",
        "## 门禁检查",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in sorted(all_checks.items())
    )
    lines.extend(
        [
            "",
            "## 成本估计",
            "",
            f"- 本地实测：{seconds_per_step:.3f} 秒/环境步。",
            f"- 单个 100k run 粗估：{estimate['estimated_hours_per_100k_run_before_gpu_smoke']:.2f} 小时。",
            f"- 6 并发、18 runs 的首个 10k 阶段粗估：{estimate['estimated_stage1_hours_at_parallel_6_before_gpu_smoke']:.2f} 小时。",
            "- 上述是 CPU 本地外推；正式并发数必须由服务器 CUDA smoke 的显存和吞吐实测确定。",
            "",
            "## 审计边界",
            "",
            f"- {result['audit_note']}",
        ]
    )
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "local_gate_passed": local_gate_passed}))
    return 0 if local_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
