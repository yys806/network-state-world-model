# STEP 6.4J H4 可评分性精确诊断

## 结论先行

本轮独立复核纠正了 r39 的统计：完整 H4 是 **3744 个完成事件**，不是 468；这是每次搜索的完成事件总数，不是去重后的 candidate 数。16 次搜索均为 `FALLBACK_A / NO_SCOREABLE_H4`，scoreable H4 为 0，dead-end 为 3064。原始 GPU 闭环工程机制仍是 `COMPLETED`，但优化性能没有被证明。

已完成源码和原始落盘的可复现性审计。r39 receipt 可逐条复核预算、root、action、执行身份和汇总计数，但没有 candidate action prefix、随机状态、完整 H1-H4 trace、Objective side state 或 candidate-level rejection reason，因此不能从原始文件逐条恢复 3744 个完成事件的拒绝原因。

按授权对两个代表 root 启动了只读 CPU rejection-ledger。CPU batch=1 和 batch=16 两次均持续异常耗时且没有生成结果文件，已停止并保留“无结果”事实。因此本轮没有可报告的 CPU reason 分布、H_eff 分布或 fixed-support Return-birth 占比；不能把源码中的 `UNSUPPORTED_FUTURE_RETURN_BIRTH` 写成 r39 的逐候选统计。

## 1. Observation：原始结果与独立统计

独立读取 `code/artifacts/backups/step6_4j_r39/pilot_results/**/decision_*.json` 得到 16 份 receipt：

| root/episode 类别 | 搜索次数 | 每次 complete H4 | 每次 dead-end | 每次 scoreable H4 |
|---|---:|---:|---:|---:|
| `...2326...2426::anchor-0029` | 8 | 344 | 0 | 0 |
| `...2318...2418::anchor-0006` | 8 | 124 | 383 | 0 |
| 合计 | 16 | **3744** | **3064** | **0** |

计算为 `8×344 + 8×124 = 3744`，`8×383 = 3064`。这 3744 是完成事件次数；receipt 没有足够身份字段证明它们是 3744 个不同 candidate。

原始结果同时记录 8192 次 unique WM transition、16 次真实 `env.step`、16/16 `NO_SCOREABLE_H4`。两条 root 的状态和 cohort 不同；第二类 root 有动态 domain dead-end，第一类没有，但两者都没有 scoreable H4，所以 dead-end 不是必要条件。

## 2. Implementation Fact：源码支持条件

- `code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py:177-189` 的 `_first_unsupported()` 检查未来状态的 fixed-support Return-birth；拒绝理由包含 `UNSUPPORTED_FUTURE_RETURN_BIRTH`。
- 同文件 `:206-334` 的 `score_candidate_set()` 要求 anchor、candidate/trace identity、Planner V1 route/cohort、字段和有限性检查；使用 `H_eff=min(support.values())`。`H_eff=0` 或未满足完整 horizon 时返回不可评分状态。
- `code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py:97-120,225-232` 把 `score_h4(...) is None` 计为 unscoreable；动态 candidate domain 为空另计 dead-end。完成 H4 与可评分 H4 是不同计数。
- `code/scripts/step6_4i_live_planner_v1.py` 只有在 `SCOREABLE`、`H_eff=4` 且有分数时才提供 winner；否则回到 `None → NO_SCOREABLE_H4 → FALLBACK_A`。

这些事实证明固定支持边界**能够**导致 fallback，但不能证明 r39 的 3744 个事件全部由该理由拒绝。

## 3. Reproducibility level

- **A：可从原始保存数据逐条审计**：16 份 receipt 的 root、action、预算、fallback、执行身份和计数；SHA、checkpoint、normalization、anchor manifest、轨迹输入与 protocol 均已核对。
- **B：可确定性重算并验证身份一致**：本轮未达到。缺少 r39 的候选前缀、RNG state、完整 trace 与 Objective side state；CPU 与 CUDA 也可能有数值/批处理差异。
- **C：只能做条件相近的新诊断**：诊断脚本复用了冻结 runtime、CandidateDomain、S-CEM、Objective/Scorer 和 seed=6311，但实际运行在 CPU，且本轮两次尝试均超时无产出；不能作为原 r39 候选复核。

## 4. Interpretation and hypotheses

**已验证的 Inference：** 统一现象是“搜索预算完成且有完整 H4，但没有 scoreable H4”；固定 support 是已知的实现拒绝边界，动态 dead-end 只解释第二类 root 的部分预算损失。

**未验证的 Hypothesis：** fixed-support Return-birth 可能是主要拒绝原因；也可能有 route mapping、pending flow、cohort、字段/有限性或 Objective side-state 不一致。现有证据不能给这些原因排序，也没有实现错误的直接证据。

**不能推出：** 不能据此声称模型预测错误、Objective/Scorer 错误，或扩大 24×64 一定有意义。

## 5. CPU 诊断记录

脚本：`code/scripts/diagnose_step6_4j_cpu_rejection_ledger_v1.py`。两个授权 root 均尝试；batch=1 运行约十多分钟无输出，batch=16 仍持续占用 CPU、无 JSON 产出，随后停止。输出文件不存在，故没有任何新 reason 频率被纳入统计。脚本明确写入 `NON_EQUIVALENT_DIAGNOSTIC` 和 `NOT_POSSIBLE_FROM_ARTIFACTS` 标签（若未来运行成功）。

## 6. Research Proposal（等待研究者决定）

最小后续验证应先解决诊断可运行性，并只对两个 root 产生完整 ledger；需分别记录首个 unsupported horizon、H_eff、Return-birth、cohort、route、field/finite 检查，再与 receipt 的 complete/dead-end 分母核对。只有结果能区分 fixed-support、状态接线或 scorer 输入问题后，才讨论是否改变支持边界或扩大实验。当前直接扩大 24×64 没有足够科研意义。

## 7. 边界

本轮未启动 GPU、正式闭环、baseline、Hybrid、ablation 或 `locked_test`，未修改科学算法和冻结 r39 证据。远端 GPU 电源/计费状态没有合法管理接口，需研究者手动确认；本机 runner/SUMO 已退出。
