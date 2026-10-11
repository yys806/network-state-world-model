# STEP 6.4J H1 Return-birth 动作敏感性诊断报告（2026-10-11）

## 完成工作

本轮只执行 H1 CPU one-step 诊断，不运行 H4、B512、GPU、真实策略动作或 `locked_test`。两个真实 callback root 均来自正式 `collect_trajectory()`，并复用冻结模型、normalization、CandidateDomain、动作编译和原始 `score_candidate_set()`。

## 实测结果

| Root / 目标任务 | 合法不同动作 | 动作变化 | H1成功 | Return-birth | H1支持通过 | H1可评分 | rollout异常 | Scorer异常 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `anchor-0006 / Task_5` | 6 | Mobility 6种；Comm/Comp/Route不变 | 6 | 6 | 0 | 0 | 0 | 0 |
| `anchor-0029 / Task_15` | 12 | Mobility 6种；Comp 3种，但目标Task_15分配始终为0；Comm/Route不变 | 12 | 12 | 0 | 0 | 0 | 0 |

总计 18 个不同合法动作、18 次 H1 WM transition、18 次 H1 Scorer 调用，全部返回 `OBJECTIVE_UNSCOREABLE / H_eff=0`。

## 任务状态核对

### anchor-0006 / Task_5

- Raw lifecycle：`waiting_to_return`。
- `work_remaining(anchor)=0`。
- `required_returned_size=0.0761075478`。
- Return Flow 不存在；当前计算宿主为 UAV_0，Return 目标为 UAV_1。
- H1 预测中剩余计算量仍为 0，宿主仍为 UAV_0。
- 分类：B，anchor 已经满足“计算完成但缺少必要 Return Flow”。

### anchor-0029 / Task_15

- Raw lifecycle：`computing`。
- 模型 `work_remaining(anchor)=0`，与 Raw `computed_cpu_work/task_cpu_work` 的浮点差约 `1.1×10^-16` 相符。
- Return Flow 不存在；当前计算宿主为 RSU_0，Return 目标为 UAV_0。
- 所有 12 个动作的 Comp 行都没有给 Task_15 分配正计算资源；H1 后剩余量仍为 0，宿主仍为 RSU_0。
- 分类：B。Raw lifecycle 与模型 completion flag 存在边界不一致，需要后续单独核对。

Task_5 在该 root anchor 时剩余量为正（0.2082748），分类为 C，且不是本轮 first unsupported task。

## 机制判断

**Observation：** 两个 root 的触发任务在模型 anchor 状态已经是计算完成、无 Return Flow；18 个动作没有改变触发任务的完成/宿主条件。

**Implementation Fact：** 原始 Scorer 在 H1 返回 `UNSUPPORTED_FUTURE_RETURN_BIRTH`，没有异常；模型参数保持不变；所有动作来自 CandidateDomain 并通过 `annotate_domain()`。

**Inference：** 在本次动作集合内，Return-birth 对动作不敏感，主要由当前模型状态和缺失 Return Flow 决定。该证据不能证明所有动作、所有网络状态都如此。

**Hypothesis：** Task_15 的 Raw `computing` 与模型 `work_remaining=0` 的边界差异可能来自生命周期/浮点阈值接线，而不是规划动作机制。

**Research Proposal：** 下一步优先做只读状态一致性审计（Raw lifecycle、`computed_cpu_work`、`task_work_remaining`、`task_completed`、Return slot）并决定是否修复状态表示；暂不实施动态 Return Flow。

## 证据边界与工程状态

- `R39_EXACT_EQUIVALENCE=UNPROVEN`；不能把 18/18 外推为 r39 的 3744 个 H4 事件比例。
- v2 曾成功保存完整 causal prefix 和 root，但在动作日志字段名错误处停止，未进入 action ledger；该失败证据保留。
- v1 结果已独立审计通过；GPU 未使用，`locked_test=false`。
