# 已确认决策

这里只记录研究者明确作出的科研或工程决策。Codex 分析、候选建议和实验现象不能自动写成 Researcher Decision。

## 2026-09-10：三方长期协作边界

**Researcher Decision**

- 研究者负责核心算法、总体架构、数学建模、科研 loss 设计、实验目的、消融变量、创新点和最终科研解释。
- ChatGPT Web 负责理论讨论、推导、架构分析、实验设计讨论和结果分析。
- Codex 作为 Research Engineer，负责实现、维护、实验执行、测试、代码事实检查和状态同步。
- Codex 可以主动报告冲突和风险，但不能自行替换科研算法。

影响范围：`AGENTS.md`、`AI_CONTEXT/`、以后所有代码与实验任务。

## 2026-09-10：GitHub 与 AI_CONTEXT 工作流

**Researcher Decision**

- GitHub `main` 是网页端可见的动态事实来源。
- `AI_CONTEXT/00_PROJECT_STATE.md` 是 ChatGPT 新对话的首入口；实现事实仍以源码/config/experiment 为最高优先级。
- 每个有效、经过合理验证的小任务应形成清晰 commit 并 push；broken state 不得推到 `main`。
- 完成本次授权更新后，Codex 不得自行修改 `AGENTS.md`，除非用户再次明确要求。

## 2026-09-09：第三 seed 与同步延期

**Researcher Decision**

- seed `20260832`、远端训练和远端同步延后。
- 现有入口保留，但不得自动启动；机器守卫见 `docs/registries/deferred_work.json`。

## 2026-09-09：项目重构优先

**Researcher Decision**

- 先完成无损知识与工程重构，以长期协作和快速找到正确信息为标准。
- 旧实验保留科研追溯价值；在引用、回归和回滚条件未闭合时不为目录整齐强行迁移。

## 尚无决策

- 最终 PI-JWM 方法是否冻结。
- P6 采用纯候选搜索、学习策略还是混合策略。
- 是否以及何时运行 seed `20260832`。

Unverified：任何未在本文件或项目权威记录中标为 Researcher Decision 的科研取舍。
