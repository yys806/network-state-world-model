# 已确认决策

这里只记录研究者明确作出的科研或工程决策。Codex 分析、候选建议和实验现象不能自动写成 Researcher Decision。

## 2026-09-18：新定义与 Step 实施工作流

**Researcher Decision**

- `D:\shen\OB\科研\PIJWM` 中最新 `00–06` 是当前目标研究定义，该目录严格只读；所有工程修改留在 `D:\shen\PKU\PIJWM`。
- 旧 P4/P6/P0–P10 计划、实验、artifact 和 checkpoint 保留为 Historical / Archived evidence，不再作为当前执行主线。
- 当前只授权 Step 1 定义—实现审计、治理和 Tracker/记录框架；禁止自动进入 Step 2、正式训练、模型重构或新 planner。
- 每个有效 Step/Substep 必须验证、记录、commit、push、固定格式汇报并停止；新定义正确性优先于旧 checkpoint 复用。

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
- 通信状态不足时是否增加 effective service/residual；外生到达/离开如何建模；planner objective/risk/fallback 的具体定义。
- 是否授权执行 Step 2。旧 `seed=20260832` 不属于当前 active queue。

Unverified：任何未在本文件或项目权威记录中标为 Researcher Decision 的科研取舍。

## 2026-09-18 Researcher Decision

Researcher explicitly fixed A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob), with A_t^Mob controlling UAV mobility only; vehicle motion remains SUMO external progression. This is a researcher decision, not an engineering inference.

## 2026-09-19：Raw 因果与 acceleration 定义

**Researcher Decision**

- AirFogSim 未来 task schedule 可以保留为 raw/internal metadata，但不得进入当前 `O_t`、History 或 input-side Entity Index。
- AirFogSim acceleration 原样保留为 raw simulator observation / audit 字段；PI-JWM canonical physical acceleration 定义为 `(v_t-v_{t-1})/delta_t`，只使用当前与历史信息。
- 首个有效时间点或缺少历史速度必须使用明确 mask，不得伪造数值；raw 与 canonical 不得混名。
- Step 2.3 只收尾 Raw Contract；后续 Dataset/Tensor 必须另行授权。

## 2026-09-20：Stateful Flow Ledger 决策

**Researcher Decision**

- Flow 是 logical end-to-end business Flow；Hop 是 carrying segment。Flow 不等于 Hop、Route 或 Communication edge。
- 使用 `AirFogSim real state/event → PI-JWM Causal Flow Ledger → Raw logical Flow state`，不修改 AirFogSim 核心传输状态机；Ledger 禁止读取 future action/outcome、target tensor、rollout prediction 或 future task schedule。
- FlowID 固定为 `(TaskID, FlowType, Epoch)`，FlowType 为 `Input/Return/DepData`；不依赖 Hop 或 RouteRevision。
- same-destination reroute 保持 FlowID/Epoch/E2E remaining，RouteRevision 增加。
- logical destination change 创建新 Epoch；旧 Epoch `SUPERSEDED`，新 source=current holder，新 total/remaining=旧 remaining。
- v1 destination change 只允许 clean hop boundary；partial active hop 必须拒绝或延迟，不在本 Step 实现 Planner。
- DepData vocabulary 保留，但当前 AirFogSim runtime instances=0；DAG 不得生成 fake DepData Flow。
- 本 Step 只授权 Ledger + Raw additive Flow contract；Sample/Tensor、Graph Builder、模型、训练、GPU 和 locked_test 均未授权。

## 2026-09-20：STEP 4.2C-C 实施授权

**Researcher Decision**

- 明确授权将 STEP 4.2C-B Raw logical Flow/Carrying state additive 贯穿到 Model-ready Sample 与 CPU Tensor；只允许 index、align、mask、train-only normalization 和 collation。
- 明确禁止 Graph Builder、Physical/Information graph、GNN/Encoder、World Model、Loss、Planner、Training、GPU、locked_test 和 formal Dataset；完成后停止等待审阅。
- 真实 Return multi-hop、same-destination reroute runtime 与 formal Flow capacity 不得因实现便利被默认为已解决；只记录机器证据和边界。

- Step 2.4（研究者明确批准）：Communication Outcome 在 Raw 层拆为 wireless、wired 和按 task 聚合 total；空 map 是已观测无服务，missing 必须是 null 加 mask/reason。该决定只冻结 Raw 语义，不授权 Dataset/Tensor 或模型实现。
- STEP 3.1（研究者明确批准）：model-ready sample 使用因果 History、严格对齐的 Future Action/Target、无 future-object leakage 的 stable input index、独立 target-side future object 表示、四类 action 和显式 presence/feature mask；本决定不授权正式数据集、模型、loss、planner 或训练。
- STEP 3.1R（研究者明确批准）：History 必须为 `[t-H+1,t]`；固定 input index/presence、真实 DAG source、typed target index、relation endpoint 和不可静默 `-1` 的 Action reference 属于修正合同。STEP 3.2 仍未授权。
- STEP 3.1F（研究者明确批准）：History 必须包含 `O_{t-H+1:t}` 和过去 `A_{t-H+1:t-1}/Y_{t-H+1:t-1}`；input index 必须覆盖 History 的因果对象 union，过去 flow/relation/DAG 必须与同一套 index 对齐；future unresolved reference 只做事实审计，不自行决定未来对象表示。STEP 3.2 仍未授权。
- STEP 3.1F-PATCH（研究者明确批准）：Future Action 先按 anchor visibility 判断是否允许引用，再使用同一 History-union `static.input_entity_index` 返回数值 index；`input_index_policy=history_causal_observable_object_union`、`future_action_index_policy=anchor_visibility_then_history_union_input_index`。STEP 3.2 仍未授权。
- STEP 4.1（研究者明确批准）：只冻结定义 03 的 Physical / Information object-field-relation mapping，必须区分真实可用、Raw 有但未暴露、Raw 不足、Derived、禁止归属和待研究者决定；禁止在本 Step 实现 graph builder、GNN、encoder、coupling、World Model、Loss、Planner、训练、GPU 或访问 `locked_test`。发现最小定义缺口时记录并停止在 mapping 层。
