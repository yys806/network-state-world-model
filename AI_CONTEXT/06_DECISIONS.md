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

## 2026-09-24：STEP 5.6A-CONFIG-FREEZE Formal Training Config v1

**Researcher Decision**

- Formal Dataset remains H=2/L=4 with 4416 train and 1104 validation windows; dataset identity, split, normalization and Definition 05 semantics are unchanged.
- Formal training seed=5601, batch_size=8, 552 steps/epoch, AdamW learning_rate=3e-4, constant schedule, weight_decay=0, betas=(0.9,0.999), eps=1e-8, gradient_clip_norm=1.0.
- Budget is max_epochs=10 and max_steps=5520. Stage 1 is steps 0–551 posterior-assisted H1; from global_step=552 the path is prior-dominant. Curriculum starts are H1 at 0, H2 at 1104, H4 at 2208.
- KL target_beta=1.0, warmup_steps=1104, free_bits=0.1 per latent dimension, overshooting OFF. Validation is prior-only over all 1104 windows and H1–H4, every 1104 completed steps, with checkpoint selector `argmin L_Val`.
- Latest checkpoints save every 552 completed steps; best checkpoints save on strict L_Val improvement; patience is 3 full validations. Resume restores model, optimizer, RNG, progress, selector state and curriculum state; sampler order is derived deterministically from formal seed and global step. FP32 is retained and AMP is not introduced.
- This freezes the formal numerical protocol. It does not authorize starting Formal Training; `formal_training=false`, `gpu_training_verified=false`, `locked_test_accessed=false`, `baseline=false`, `planner=false`, and `performance_claim=false` remain required boundaries.

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
## 2026-09-20：STEP 4.3A 实施授权

- 研究者明确授权 Frozen Tensor → typed Physical / Information dual-graph objects + Align/GeoComm。
- Physical topology 参数只允许作为 deterministic development config，必须标记 `development_only=true`、`research_frozen=false`。
- 明确禁止 Encoder/MLP/GRU/message passing/P2A/P2C/GNN/World Model/Loss/Planner/Training/GPU/locked_test/formal Dataset；完成后停止。
# 2026-09-21 STEP 4.4 communication uncertainty decision

- Researcher explicitly selected `Learned Future CSI -> Rule Nominal Rate -> Known Stochastic Outage Event -> Actual Service`.
- Comm z represents CSI/channel uncertainty only. Outage is not a latent/head/input/target leak; both seeded `sample` and marked `expectation` modes are required. `learned_service_residual=false`.
- Wired capacity may be added from causal simulator configuration; active membership/count must first be derived from Flow Carrying and checked equal to `WiredNetworkManager`.
- STEP 4.4 implementation is authorized within the untrained CPU contract only; Loss, optimizer, Training, GPU, Planner, candidate generation, formal Dataset, and `locked_test` remain forbidden.

## 2026-09-22 STEP 5.3E — CSI train-mean bias initialization

**Researcher Decision**

- PI-JWM v1 采用 raw CSI decoder，直接输出 CSI `[dB]` 并进入冻结的规则转移链；不采用 normalized-output decoder 作为 v1 主线。
- CSI decoder 最后一层 bias 使用 frozen train-only CSI normalization mean 初始化，初始化发生在 optimizer 创建前；所有 RB 使用同一 global mean。
- 当前 development provenance 为 `dev_train`；validation、Future Target 和 `locked_test` 不参与 mean fit。normalized-output bridge 仅保留为未来可选 ablation。
# 2026-09-21 STEP 5.0 — Definition 05 researcher decisions

- Prediction distribution: retain stochastic `z^Phy/z^Comm`; Vehicle Motion/CSI observation decoders are deterministic mean heads; no learned observation variance.
- Prediction loss: Motion and CSI each use independently mask-normalized normalized-space MSE; v1 combines them equally.
- Posterior teacher: only corresponding future Vehicle Motion or CSI targets may enter family-specific training-only target encoders; prior never reads Future Target; Future Graph/Flow/Task/DAG/rule-state are forbidden posterior evidence.
- KL: separate analytic diagonal-Gaussian Physical/Communication KL. Physical mask is valid Vehicle slots only; Communication mask is valid+present+wireless+CSI-target-valid. Use configurable warm-up and small free bits; no KL balancing.
- Overshooting: OFF in v1. Total objective is Prediction plus beta-weighted family KL.
- Schedule: posterior-assisted short-horizon warm-up, then prior-dominant recursive curriculum `1→2→4→L`; Stage 2 posterior is KL teacher only; validation is always prior-only.
- Fixed support: unsupported future structure uses component-level mask/exclude/classify with separate counts; valid Motion/CSI supervision remains; whole-window deletion is forbidden.
- Rule states: no Flow/Task/DAG/Lifecycle/Completion learned head or independent loss; natural differentiable rule paths may carry gradients, but discrete rules stay exact.
- Trainable modules: jointly train Dual-Graph Encoder, RSSM dynamics, Prior, Posterior, Target Encoders and Motion/CSI Decoders. STEP 4.3B FROZEN means architecture/interface, not weights.
- Validation/evaluation: checkpoint and early stopping use prior-only horizon-mean `L_Val`; KL is diagnostic. Report Motion/CSI separately with per-horizon raw-unit MAE/RMSE; uncertainty sampling is auxiliary; system metrics remain closed-loop metrics.
- These explicit decisions supersede conflicting observation-NLL/Event/Residual/overshooting clauses in the read-only older Definition 05 note. The private note remains unchanged.

## 2026-09-23 STEP 5.5 — Formal Dataset v1 protocol

**Researcher Decision**

- Formal Dataset v1 使用 `H=2`、`L=4`；每条真实 AirFogSim trajectory 含 96 个连续 transition 和 97 个 Decision，由此每条构造 92 个 windows。
- 接受 60 条完整 trajectory；simulator primary seeds 为 `2026092300..2026092359`，对应 policy seeds 为 `2026092400..2026092459`。失败条目不得部分接受，按后续未用 deterministic seed pair 替换并记录 lineage。
- 固定 `split_seed=20260923`，先按 trajectory deterministic shuffle，再分为 48 train / 12 validation；禁止 window-level random split，禁止根据后续 loss 或统计量改 split。
- Formal Physical topology v1 固定为 `radius_knn(radius=1000m,k=2)`；这是 Dataset v1 protocol choice，不是 topology optimality claim。
- 数据采集采用 causal coverage-oriented 四动作 policy：`A_t=(Route,Comm,Comp,Mobility)`，动作只能依赖当前观察/因果 History，车辆仍由 SUMO 外生推进；policy 不是 Planner，也不作 reward 最优声明。
- Dataset 不包含或物化 `locked_test`。本 Step 仅 CPU Dataset/Interface acceptance；GPU、formal training、baseline、Planner 和 performance claim 均未授权。
