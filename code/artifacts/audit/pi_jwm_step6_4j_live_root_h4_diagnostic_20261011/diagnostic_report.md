# STEP 6.4J Live Root H4 可评分性最小机制诊断（2026-10-11）

## 结论

本轮未通过真实 Live Root 身份门禁，因此没有执行评分探针。r39 的原始统计已独立复核，原始结论保持不变；逐候选根因仍无法确认。

## Implementation Fact

- r39 冻结提交为 `0586cde90c8cc244e9f5091fd51e7c77217b817a`；当前文档提交为 `5d35de12fdd32c321bf7bbe4a0de8a822771548b0`。
- 原始 receipt 共 16 份 `decision_*.json`：两个 episode 各 8 份。
- `anchor-0006` 每次 `N_complete_sequences=344, N_dead_end_branches=0`。
- `anchor-0029` 每次为 `N_complete_sequences=124, N_dead_end_branches=383`。
- 因此总计：3744 complete、3064 dead-end、0 scoreable H4。该统计按完成事件计数，不是唯一 candidate 数。

## Root identity gate

按正式 Live Planner 路径检查了重建要求：`history_tensor → build_state → context_from_current_raw → CandidateDomain.from_state → prepare_anchor`。r39 没有保存初始 live causal prefix、完整 History、候选序列或 RNG；本轮有界 CPU 重建也没有形成可核验的 root fingerprint equality receipt。故状态为 `ROOT_IDENTITY_UNVERIFIED`。

没有使用静态 `load_frozen_runtime()` 的 prepared anchor 作为真实 root，也没有把历史 raw 文件直接宣称为运行时 root 的精确替代。

## Observation / Inference / Hypothesis

- **Observation：** 16/16 搜索均 fallback A，原始完成事件中可评分 H4 为 0。
- **Inference：** 现有 r39 证据不足以区分 H1 起不支持、H1–H3 部分支持、future Return-birth 边界、动态 domain dead-end、Objective/Scorer 接线异常等机制。
- **Hypothesis：** fixed-support future Return-birth 仍是候选解释之一，但本轮没有新的逐候选证据支持其在 r39 中的频率。
- **Research Proposal：** 如需继续，最小动作是另行授权一个新的 additive live-root capture/replay 协议，在运行时保存 causal prefix、History、root fingerprints、候选身份和最多 1–4 条完整序列的 rejection ledger；这不是本轮自动实施的算法改动。

## CPU / GPU / locked_test / cost

GPU 未启动，`locked_test=false`。未调用 `score_candidate_set()`，新 WM transition 为 0。root 重建使用每 root 180 秒硬预算，整轮不超过 10 分钟；未形成可接受的完整 root receipt，故停止。

机器回执：`diagnostic_receipt.json`。

