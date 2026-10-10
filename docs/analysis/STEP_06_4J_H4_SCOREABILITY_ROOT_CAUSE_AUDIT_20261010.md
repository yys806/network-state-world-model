# STEP 6.4J r39 H4 可评分性根因审计

## 结论先行

r39 的工程闭环已经完成：两条 live episode 各执行 8 次决策，共 16/16 次 B512 搜索、16 次真实 `env.step`，原始结果、独立审计、27/27 文件 SHA、ZIP 与本地 D: 备份一致，`independent_acceptance.json` 为 `COMPLETED/0`。这只证明真实 GPU 闭环机制和证据链可运行，不是优化性能 PASS。

16/16 次搜索均为 `FALLBACK_A`，原因均记录为 `NO_SCOREABLE_H4`；468 个完整 H4（344×8 + 124×8）全部未进入可评分集合。第一条 episode 每步没有 dead-end，第二条每步有 383 个 dead-end；两条都完成了 H4 候选，因此“候选域为空”不能解释全部结果。当前最有证据的解释是固定支持边界，尤其源码中的 `UNSUPPORTED_FUTURE_RETURN_BIRTH`；但 r39 没有保存逐候选拒绝原因，不能断言 468 个 H4 全部由这一条拒绝。

## Observation：原始结果说明了什么

证据根目录：`code/artifacts/backups/step6_4j_r39/pilot_results/`，执行身份为 `execution_config_id=74535bbe2bd07260c840666dd2fd0b68190e4800dba7159a83d485d9a2632cfc`。

| episode | 决策数 | 每次 WM 预算 | 每次完整 H4 | 每次 dead-end | 每次可评分 H4 | dispatch / reason |
|---|---:|---:|---:|---:|---:|---|
| `...2318...2418` | 8 | 512 | 344 | 0 | 0 | `FALLBACK_A / NO_SCOREABLE_H4` |
| `...2326...2426` | 8 | 512 | 124 | 383 | 0 | `FALLBACK_A / NO_SCOREABLE_H4` |
| 合计 | 16 | 8192 | 468 | 3064 | 0 | 16/16 `FALLBACK_A` |

这些数字可在各 episode 的 `decision_*.json` 的 `budget_receipt`、`h4_scoreable_count`、`best_fingerprint` 和 `fallback_reason` 字段复核；`pilot_attempt.json` 还记录 16 次、每次 512 次 unique transition、16 次真实环境步进，`actual_action_history_aligned=true`，每条 episode 有 8 个 distinct root。

两条轨迹并非同一状态：`...2318...` 从 3 个任务开始，cohort 后续增长到 7，真实动作记录只有 mobility；`...2326...` 从 23 个任务开始，后续增长到 28，含 `failed/done/computing/waiting_to_offload/waiting_to_return/to_generate` 等状态，并出现 2 条 Comp 记录。因而空 cohort、空 candidate domain 或日志接线错误都不能作为统一解释。

## Implementation Fact：源码实际如何工作

1. `code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py:206-254` 要求同一 anchor、唯一 candidate/trace identity、Planner V1 domain、H1-H4 完整 trace；Route 必须单跳且不能是 pending flow；cohort 由 anchor 中存在、未完成、非 terminal task 组成。空 cohort 返回 `OBJECTIVE_UNSCOREABLE_EMPTY_COHORT`。
2. 同文件 `:177-189` 的 `_first_unsupported()` 对每个未来状态检查 `planner_derived_return_birth_required(...)`。若完成任务需要新 Return slot，而 fixed-support 不允许出生，返回 `UNSUPPORTED_FUTURE_RETURN_BIRTH:<task>:H<h>`；`H_eff=min(support.values())`，若为 0，`score_candidate_set()` 返回 `OBJECTIVE_UNSCOREABLE`。
3. `code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py:97-120,225-232` 将 `score_h4(...) is None` 计为 unscoreable；只有 scoreable 候选进入 `scoreable`、winner 和 objective 排序。预算仍必须填满，因而“完成 H4”不等于“可评分 H4”。同文件 `:180-185` 说明空的动态 candidate domain 会计为 dead-end。
4. `code/scripts/step6_4i_live_planner_v1.py` 的 live `score()` 只在 `status == SCOREABLE`、`H_eff == 4`、有 score 且首个 `H_sup == 4` 时返回分数；否则返回 `None`。因此没有 scoreable H4 时 `best_fingerprint=null` 是预期接线结果，不代表 planner 没有执行搜索。

## 根因判断

### 已确认

- 搜索预算、真实执行和日志接线不是全局失败：16/16 预算完整，16/16 `env.step`，两条 episode 均有 distinct fresh root，动作历史对齐，独立审计 `COMPLETED/0`。
- 统一运行时结果是“有完整 H4、无 scoreable H4”，而不是统一的空 cohort 或空 domain。
- fixed-support 的 future Return-birth 是实现中的硬拒绝边界；6.4H 静态证据也记录过该边界。该边界足以产生 `H_eff<4`，从而触发 live planner 的 `None -> NO_SCOREABLE_H4 -> FALLBACK_A` 链。

### 尚未证实

- r39 没有逐候选保存 `support_boundary_reasons`，所以无法从现有落盘结果计算 468 个完整 H4 中有多少被 `UNSUPPORTED_FUTURE_RETURN_BIRTH` 拒绝，也无法排除少量候选由 Route mapping、字段/有限性或其他 fixed-support 检查拒绝。
- 没有证据显示 `score_candidate_set()`、live planner 或搜索预算实现错误；同样，不能仅凭 0/16 推断模型预测错误或 objective 数学错误。
- 6.4H 的 50% scoreability 不能外推到 r39：6.4H 是静态、配对的 Validation 起点与 seed/协议；r39 是真实 live 状态，cohort、状态序列、root、动作和反馈均不同。

## H4 边界比较与候选解释

6.4H 的固定支持边界计数（B512 49,845、B1024 162,239）和 grammar dead-end（B512 19,882、B1024 46,621）说明 fixed-support 在历史静态搜索中已经是主要筛选边界之一，但那些计数不是 r39 的逐候选计数。r39 第二 episode 的 383 dead-end/step 说明动态 domain 也有作用；第一 episode dead-end 为 0 却仍 0/344 scoreable，进一步表明 dead-end 不是必要条件。不同 cohort 和任务状态会改变 Return birth、Effort/Flow support 与 domain rebinding 的触发时机，这是 live 与静态结果不可直接比较的具体原因。

## 证据缺口与最小验证方案（Proposal）

最小验证不是扩大正式实验，而是在本地 CPU 对已有 r39 的 468 个完整 H4 做只读重放，逐候选输出：`candidate_fingerprint`、首个不支持 horizon、具体 reason、cohort size、Route mapping 检查、finite/field 检查及 H_eff。它可以区分：

- 若几乎全部为 `UNSUPPORTED_FUTURE_RETURN_BIRTH`：固定支持边界是主因；
- 若主要为 pending-flow/Route mapping：live action-to-trace 支持不一致是主因；
- 若出现 empty cohort、缺字段或 NaN：需要单独审计状态采集或 objective 输入；
- 若 CPU 重放与 receipt 的 complete/dead-end 数不一致：优先审计日志统计或 seed/状态重建接线。

该重放需要研究者单独决定是否授权；本轮未启动新实验、未修改科学算法、未访问 `locked_test`。

## 是否继续 24×64

在当前证据下，直接扩大 24×64 没有足够科研意义：它会重复产生 `NO_SCOREABLE_H4`，无法比较 winner 或优化目标，且会把“工程闭环完成”误读为“优化有效”。只有在最小 CPU rejection ledger 明确主因并由研究者决定是否修复/重新定义支持边界后，扩大实验才可能回答新的科研问题。是否修复 fixed-support、改变候选/目标定义或保留当前边界，均属于研究者决策，本报告不代替选择。

## GPU 与边界

本机未发现 PI-JWM runner 或 SUMO；本机没有可用 `nvidia-smi`，也没有合法远端实例管理接口。因此只能确认本机进程已退出，不能确认远端实例实际电源或计费状态；研究者需要手动核对并关闭实例。本次诊断不保持 GPU 运行。`locked_test=false`，未启动 baseline、ablation、Hybrid 或正式 24×64。

## 复核入口

- 原始结果：`code/artifacts/backups/step6_4j_r39/pilot_results/`
- 独立审计：`code/artifacts/backups/step6_4j_r39/independent_acceptance.json`
- scorer：`code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py`
- fixed-budget search：`code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py`
- live planner：`code/scripts/step6_4i_live_planner_v1.py`
- 6.4H 对照记录：`docs/implementation_records/STEP_06_4H_S_CEM_B512_GPU_PILOT.md`
