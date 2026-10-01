> 历史审计快照：B1已由研究者明确当前轮distinct语义、B2已在后续runner修补关闭。当前Go/No-Go请读 `docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` 和对应final acceptance。本记录和旧机器收据保留，不代表当前阻塞。

# STEP 6.3D — Validation 前科学与实现审计

状态：`PRE_VALIDATION_AUDIT=BLOCKED`、`VALIDATION_NO_GO`。这是一次完成并发现阻塞的审计，不是正式比较通过。未运行 GPU search，未读取或生成正式 Validation outcome，`locked_test=false`。

## Step Goal / Definition Basis

依据研究者 2026-10-01 本次审计指令及原 STEP 6.3D 冻结合同，独立检查方法、CEM 更新、预算、随机性、统计和 runner。目标定义继续引用 TRAIN closure 中只读 Definition 06 §3.1 及其 SHA；本次没有重新解释或修改科研定义。TRAIN closure 的身份/归档/选参 PASS 保留，不替代新审计门。

## Initial State / Files Involved

`git fetch origin` 完成；`HEAD=origin/main=b9b597a654c7c07105271b1cbb1103dc9bc10dab`，tracked clean。原有未跟踪 ZIP、TASK 和绘图脚本保留。已读项目状态、TRAIN closure、三个搜索模块、CandidateDomain、SearchNode/accountant、one-solve 和 formal matrix runner，以及治理、权威计划和进展入口。正式 Validation 目录仅检查文件名/数量，0 份；07/08 比较与方法选择收据不存在。没有打开正式 Validation outcome。

新增审计脚本 `code/scripts/audit_step6_3d_pre_validation_v1.py`、测试 `code/tests/test_step6_3d_pre_validation_audit_v1.py`。机器证据位于 `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/`，01–10 分别记录数学、elite 门、预算公平性、随机/顺序、quota、统计、预注册 schema、TRAIN 配对、总验收、SHA。原始 TRAIN、源码、模型、候选域、语法、目标、支持和 frozen config 均未改。

## Changes / Reuse

复用真实候选 Grammar/CandidateDomain、SearchNode、缓存记账和未经修改的 solver。合成 TRAIN-only CPU fixture 用 50 个合法 RB 起点、四步路径编码作为确定性一步转移，不加载正式模型、不进行训练。所有测试结果只证明实现机制和确定性，不冒充真实模型质量或正式 Validation 比较。

数学 oracle 使用独立有理数手算：q0=(1/3,1/3,1/3)，elite=(1/4,3/4,0)，eta=0.5、epsilon=0.05，一轮为 `(47/160,85/160,28/160)`；第二轮独立展开公式一致。概率和为 1、未见合法选项非零。合法选项集合进入 table key，mask 改变则创建 fresh q0；不把旧 mask 的非法质量转入新表。HRS 不更新；S 只更新 mode/count；MH 更新 mode/count/subset/assignment/start。目标严格字典序优先，fingerprint 只破相等时的并列。

## Validation / Results

- batch16、HRS K1 / CEM K4：B256/512/1024 的 quota 分别为 HRS 全预算、CEM 64/128/256 每轮。16 分支每个四步最多需要 64 次新转移，所有预算允许至少一批完成 H4；真实 Grammar 的 dead end 仍可能阻止完成，不能从 quota 可行推断真实 anchor 成功。
- 9 个 method×budget case 在原预算顺序和 1024→256→512 逆方法顺序下，插入其他 anchor/seed solve 后，所有离散 SearchOutcome 字段一致，仅排除计时。预算完整、H4 完成、retain≤5；Python global RNG 未改。result_path 不依赖执行顺序，resume 拒绝错误 budget。
- 同样显式请求 `[A,A,B]` 后 `[B,A]`：三方法都是 2 次 unique transition、3 cache hits、仅 2 个实际 forward members。预算耗尽后新 C 请求拒绝；共享 provenance+parent latent/state/graph+canonical action key，不因方法不同免费 forward。
- 六种 paired outcome、五种方法选择分支通过；64 anchor×5 seeds、10000 次、seed6316 的 cluster bootstrap 与独立逐 cluster 重采样 oracle 完全一致；seed 不拆散，lower95>0 严格门不变。该门表示“通过预注册优势判据”，不自动宣称多重比较控制。
- TRAIN 768 原始文件再次比对本机已验收 inventory SHA 和 16 frozen source SHA；K4/rho0.1 的 96 对 MH-vs-S 为 **26 胜、58 平、12 负**。逐 anchor 的三个 seed 和 mean 收入 08。这证明没有完全相同行为的迹象，不选方法、不调参。
- 07 预注册 method×budget 必需字段、六类互斥配对计数（总和必须320）和 primary 统计。额外三比较同一 anchor 重采样的 Bonferroni percentile 98.333333% 区间只作为敏感性诊断，注明 bootstrap 不提供有限样本联合覆盖保证；Holm 不启用，因为没有冻结相应 p-value 定义。主选择规则不变。

## Known Issues / Documented Intent vs Actual Implementation

### B1：新可评分 H4 的定义不一致风险 — Awaiting Researcher Decision

**Documented Intent：** 本次指令要求少于 2 个“newly scoreable H4”时不更新。**Actual Implementation：** `solve_fixed_budget` 用 `len(unique_current)>=2`，其中包括早先已评分并从 cache 重放的 H4。**Evidence：** 02 的确定性显式提议反例中，两方法首轮只发现两个可评分候选，随后没有新可评分 fingerprint，但旧候选重放仍触发 later update，且 mode 表质量确实改变。其他不可评分分支消耗正式 B256/K4/batch16 quota；未修改算法，只控制测试提议输入。

**Conflict：** 如果“新”指本次 solve 第一次发现，当前门违反冻结要求；如果指本轮重新完成的候选，可以符合。研究者尚未明确两者，本审计不自行选解释。受影响源文件为 `step6_3d_fixed_budget_search_v1.py`，proposal 模块执行收到的 elite update。保留原结果，不据此重算/撤销历史 TRAIN identity closure，也不宣称其已满足新语义门。

### B2：正式 runner 未满足本次要求的输出和阶段停止门

**Documented Intent：** 六类配对统计和320总和、Return-birth/scorer 异常单列、Stage A 只跑1024后停止让研究者审阅。**Actual Implementation：** runner 只有 win/tie/loss，未汇总 Return-birth/scorer exceptions；按256→512→1024全部完成后才统计并写07/08，没有 Stage A 参数或停止门。**Evidence：** `run_step6_3d_formal_cpu_matrix_v1.py::validation`，07 schema 逐项记录缺口。原 raw 已包含相应残余字段，但没有正式汇总和校验。**Status：** BLOCKED，需后续明确授权的实现修补与再审计；本任务遵守“发现问题不为了 PASS 自行修补”要求，保持 runner/source SHA。

## Expected vs Actual / Go–No-Go

局部机制通过，完整方法和执行合同还有 B1/B2，故 **VALIDATION_NO_GO**。推荐在解阻并再次审计后按1024→256→512安排，当前执行配置仍为原顺序；本次不激活 Stage A。未来 Stage A 为64×5×3=960 cases、983040名义转移，完成后必须停止；Stage B 256/512另行授权。Return-birth 固定支持限制保留，`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。Validation=0、方法未选、locked_test=false。

## Verification / Context / Git / Next Step

实际审计命令：`python code/scripts/audit_step6_3d_pre_validation_v1.py`，输出 NO_GO 及 TRAIN 26/58/12；数学/公平性/随机顺序/统计断言均通过，NO_GO 是审计发现而非脚本崩溃。定向测试、compileall、knowledge-index、diff 和 Context Consistency 的最终结果见过程记录及 Git 提交。Context 00/04/05/07/08、权威计划/进展、tracker/index/deferred registry 同步；其他01/02/03/06研究定义与架构无变化。SHA manifest 可逐文件复验，manifest自身 SHA 在本机计算，不放入自身以避免循环。

Git 使用单独审计提交并推送 main，不提交原始结果/大归档。唯一下一动作：研究者明确 B1 中“newly”的定义，并授权必要 runner 修补后再审计。未运行正式 Validation，完成本 Step 后停止。

实际低成本检查：focused Step 6.3D 19/19 PASS（18.101 s）；compileall exit 0；diff check exit 0。01–09及11收据均由10 SHA manifest绑定；19个测试通过不消除语义和 runner 阻塞。

最终导航/一致性检查：knowledge-index write/check 均 passed=true、mismatches=[]；git diff与staged diff --check exit0；Context Consistency PASS；12项 SHA与staged Git blob逐项一致。审计结论仍是 NO_GO。
