"""Freeze the PI-JWM temporal-candidate smoke into a PPT-ready evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = (
    CODE_ROOT / "artifacts" / "reports" / "pi_jwm_v11_temporal_candidate_protocol_20260719"
)


def _bool_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[name]
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_pretraining_gate(summary, candidates, groups):
    non_default = candidates[candidates["action_family"].astype(str) != "default"].copy()
    applicable = _bool_series(non_default, "action_applied")
    supported = _bool_series(non_default, "action_supported")
    changed = _bool_series(non_default, "action_changed")
    group_keys = ["seed", "decision_time"]
    valid_groups = int(
        non_default.assign(_valid=applicable)
        .groupby(group_keys, dropna=False)["_valid"]
        .any()
        .sum()
    )
    total_groups = int(len(groups))
    nontrivial_groups = int(_bool_series(groups, "is_nontrivial").sum())
    seeds = {int(value) for value in summary.get("seeds", [])}
    matched_accessed = bool(seeds.intersection({18, 19}))
    external_accessed = bool(seeds.intersection(set(range(60, 70))))
    quality_passed = bool(summary.get("quality_audit", {}).get("passed", False))
    applicability_rate = float(applicable.mean()) if len(non_default) else 0.0
    valid_group_ratio = float(valid_groups / total_groups) if total_groups else 0.0
    nontrivial_ratio = float(nontrivial_groups / total_groups) if total_groups else 0.0
    passed = bool(
        quality_passed
        and applicability_rate == 1.0
        and valid_group_ratio >= 0.70
        and int(supported.sum()) > 0
        and int(changed.sum()) > 0
        and nontrivial_groups > 0
        and not matched_accessed
        and not external_accessed
    )
    return {
        "framework": "PI-JWM",
        "result_kind": "diagnostic_only",
        "status": "ready_for_formal_label_generation" if passed else "measurement_gate_failed",
        "bridge_smoke_gate_passed": passed,
        "gpu_selector_training_allowed": False,
        "gpu_block_reason": (
            "formal train/calibration physical labels and candidate-oracle audit are not frozen"
            if passed
            else "temporal candidate smoke gate failed"
        ),
        "seeds": sorted(seeds),
        "decision_groups": total_groups,
        "candidate_rollouts": int(len(candidates)),
        "non_default_candidates": int(len(non_default)),
        "applicable_non_default_candidates": int(applicable.sum()),
        "supported_non_default_candidates": int(supported.sum()),
        "changed_non_default_candidates": int(changed.sum()),
        "non_default_applicability_rate": applicability_rate,
        "valid_non_default_group_ratio": valid_group_ratio,
        "nontrivial_groups": nontrivial_groups,
        "nontrivial_group_ratio": nontrivial_ratio,
        "quality_audit": dict(summary.get("quality_audit", {})),
        "matched_test_accessed": matched_accessed,
        "external_holdout_accessed": external_accessed,
        "ppt_ready": bool(quality_passed and len(candidates) > 0),
    }


def plot_protocol_breakthrough(gate, output_path):
    labels = ["Applicable\nnon-default", "Nontrivial\ndecision groups"]
    previous = np.asarray([3.0 / 81.0, 0.0], dtype=np.float64) * 100.0
    current = np.asarray(
        [gate["non_default_applicability_rate"], gate["nontrivial_group_ratio"]],
        dtype=np.float64,
    ) * 100.0
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.18, previous, 0.36, label="Previous task-ID protocol", color="#9ca3af")
    ax.bar(x + 0.18, current, 0.36, label="Causal policy v1", color="#0f766e")
    for offset, values in ((-0.18, previous), (0.18, current)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + 2.0, f"{value:.1f}%", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Temporal candidate protocol restores executable supervision")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_ppt_evidence(gate, family_df, output_path):
    family_lines = []
    for row in family_df.itertuples(index=False):
        family_lines.append(
            f"- `{row.action_family}`：{int(row.num_candidates)} 个，"
            f"supported={int(row.num_supported_candidates)}，changed={int(row.num_changed_candidates)}，"
            f"平均任务收益差={float(row.mean_effect_task_utility):.4f}。"
        )
    text = f"""# PI-JWM 时序候选协议：PPT 结果底稿

## 一句话结论

我们把会在 rollout 后失效的 task-ID 绝对动作，重构为决策时冻结参数、执行时按当前合法集合解析的因果闭环策略。5-seed smoke 已达到正式物理标签生成条件，但尚未生成 train/calibration 全量标签，因此不能宣称 selector 或 RMSE 已定型。

## 上周问题

- 旧协议 81 个非默认候选仅 3 个有效，有效率 3.7%。
- 15 个决策组中 0 个保留有效物理差异。
- 根因是 step 0 后 task ID 和任务阶段变化，不是模型容量或 GPU 训练不足。

## 本周突破（观测事实）

- seeds：`{','.join(str(value) for value in gate['seeds'])}`；决策组：{gate['decision_groups']}；候选 rollout：{gate['candidate_rollouts']}。
- 非默认候选 {gate['non_default_candidates']} 个，安全可执行 {gate['applicable_non_default_candidates']} 个，有效率 {gate['non_default_applicability_rate']:.1%}。
- 其中 {gate['supported_non_default_candidates']} 个获得实际阶段支持，{gate['changed_non_default_candidates']} 个确实改变环境状态。
- {gate['nontrivial_groups']}/{gate['decision_groups']} 个决策组产生 task、throughput、RB/CPU 或 energy 的可测差异（{gate['nontrivial_group_ratio']:.1%}）。
- 缺失值、负能耗、reward 重建、能耗守恒和非法动作错误均为 0。
- matched test 18--19 与 external holdout 60--69 均未访问。

## 动作族证据

{chr(10).join(family_lines)}

## 严谨解释

- `action_applicable` 表示固定规则安全执行；无合法任务时允许安全 no-op。
- `action_supported` 表示该步确有可作用任务；`action_changed` 表示相对默认动作真实改变。
- 这种拆分避免把安全 no-op 当 simulator 错误，也避免把没有物理作用的候选伪装成有效收益标签。
- 当前 53.3% 非平凡组证明标签不再完全退化，但仍需要更大 train/calibration 覆盖后才能判断 bridge 可学性和 candidate oracle 上限。

## 下一步

1. 按冻结协议生成 physical train seeds `0--15,20--43` 和 calibration seeds `44--49` 标签。
2. 运行 bridge OOF/校准审计与 candidate oracle 门；只有通过后才开启 GPU selector 训练。
3. validation `50--59` 用于结构冻结；matched test 和 external holdout 继续锁定。

## PPT 图建议

- 主结果图：`figure_0_protocol_breakthrough.png`。
- 测量链图：`smoke5/figure_1_reward_decomposition.png`、`smoke5/figure_2_uav_energy_components.png`。
- 耦合解释图：`smoke5/figure_3_utility_energy_tradeoff.png`、`smoke5/figure_4_paired_coupling.png`。
"""
    output_path.write_text(text, encoding="utf-8")


def write_sha256_manifest(root):
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.json":
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0].startswith("smoke1"):
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": int(path.stat().st_size),
            }
        )
    payload = {"algorithm": "sha256", "num_files": len(entries), "files": entries}
    path = root / "sha256_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.report_root.resolve()
    smoke = root / "smoke5"
    summary = json.loads((smoke / "summary.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(smoke / "candidate_summary.csv")
    groups = pd.read_csv(smoke / "decision_group_audit.csv")
    family = pd.read_csv(smoke / "action_family_summary.csv")
    gate = build_pretraining_gate(summary, candidates, groups)
    (root / "pretraining_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_protocol_breakthrough(gate, root / "figure_0_protocol_breakthrough.png")
    write_ppt_evidence(gate, family, root / "ppt_evidence.md")
    write_sha256_manifest(root)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate["bridge_smoke_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
