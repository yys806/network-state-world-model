# STEP 6.4J — r39 收口与 H4 可评分性根因审计

## Step Goal

收口冻结提交 `0586cde90c8cc244e9f5091fd51e7c77217b817a` 的 r39 GPU 工程闭环，核验原始结果与独立审计一致性，并对 16 次 live B512 搜索的 H4 可评分性做只读根因审计。

## Definition Basis

研究者授权的边界是：真实 GPU 闭环机制完成与科研搜索结论分开记录；不启动新实验，不修改 Objective、Scorer、CandidateDomain、World Model、搜索算法、训练参数或冻结结果；`locked_test=false`。

## Initial State

r39 原始备份位于 `code/artifacts/backups/step6_4j_r39/`，execution config 为 `74535bbe2bd07260c840666dd2fd0b68190e4800dba7159a83d485d9a2632cfc`。工作区已有历史未跟踪部署产物，保持不动。

## Files Involved

新增 `docs/analysis/STEP_06_4J_H4_SCOREABILITY_ROOT_CAUSE_AUDIT_20261010.md` 与本记录；同步 `task_plan.md`、`progress.md`、`findings.md`、`AI_CONTEXT/00_PROJECT_STATE.md`、`AI_CONTEXT/05_EXPERIMENTS.md`、`AI_CONTEXT/07_KNOWN_ISSUES.md`、`AI_CONTEXT/08_CHANGELOG.md`、`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`、`docs/CHANGELOG.md`、`docs/PIJWM_IMPLEMENTATION_TRACKER.md` 及知识索引。

## Changes

只新增审计报告和项目记录，明确区分工程 `COMPLETED` 与搜索优化 `NOT_ESTABLISHED`；记录 16/16 `FALLBACK_A`、468 个完整 H4、3064 个 dead-end、0 个 scoreable H4，以及 fixed-support future Return-birth 的已知边界和逐候选证据缺口。未改科学源码、配置、原始结果或备份。

## Reuse

复用 r39 `pilot_attempt.json`、16 份 `decision_*.json`、episode receipt/journal、独立审计结果、冻结 scorer/search/live planner 源码及 6.4H 静态记录。未重新运行 planner、环境、GPU 或正式实验。

## Validation

- 独立审计按正确 episode 顺序返回 `COMPLETED/0`，16 次搜索、16 次 `env.step`。
- 原始 receipt 复核：两 episode 各 8 决策、每次 512 unique transitions、动作历史对齐、每条 8 个 distinct roots。
- 只读源码审计：`score_candidate_set()` 的 H1-H4/fixed-support/Route/cohort 路径与 fixed-budget `None` 计数逻辑一致。
- 本机进程检查未发现 PI-JWM runner 或 SUMO；不具备远端实例电源/计费核验权限。

## Results

工程机制：`COMPLETED`。科研搜索：16/16 无 scoreable H4，全部 `FALLBACK_A`，搜索优化有效性 `NOT_ESTABLISHED`。fixed-support 是高可信候选根因，但不是对 468 个候选的逐条证明；当前没有实现错误的证据。

## Expected vs Actual

预期是验证真实闭环机制并保留失败分母；实际闭环完整，但 live 状态下没有 winner。该差异正是本轮 H4 可评分性审计对象，不能解释成性能 PASS。

## Known Issues

逐候选拒绝 reason 未落盘；无法从现有 r39 结果精确分解 `UNSUPPORTED_FUTURE_RETURN_BIRTH`、Route mapping、字段/有限性等路径的数量。6.4H 静态 scoreability 与 r39 live scoreability 的状态、cohort、seed、root 和协议不同，不能直接外推。

## Git

待本记录及索引/上下文同步后执行 Conventional Commit 和 push；未跟踪部署脚本、ZIP、TASK 不纳入本次提交。

## Next Step

唯一最小下一步是研究者决定是否授权本地 CPU 只读 rejection-ledger 重放，以区分 fixed-support、动态 domain、Route/状态采集和日志统计假设。未获决定前不扩大 24×64，不启动 GPU 或 `locked_test`。
