# PI-JWM 当前科研状态

> 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH：`PASS` 只验收修正后 TRAIN 32 / Validation 64 锚点的静态 Objective 资格和 CPU 批量执行路径。TRAIN-only 固定 HRS seed6391/B_WM64 的 32-anchor 诊断中 16 个锚点找到可评分 H4、16 个没有；275 条不可评分完整 H4 均触发 future Return birth 支持边界。`H4_SEARCH_COMPARISON_READINESS=PENDING_RESEARCHER_DECISION_ON_MODEL_OBJECTIVE_SUPPORT`，正式 TRAIN tuning/Validation 比较未运行，方法未选。证据见 20–25 收据和 Patch 实施记录；无 GPU、`locked_test`、训练或闭环。

> 2026-09-29 STEP 6.3B/6.3C-PATCH：研究者已冻结 Comm 子集选择和 TRAIN 通信结构条件数量支持；原“所有 eligible Task 都有行”的旧限制不再适用。4416 TRAIN H1 投影中任务数量拒绝为 0，728 条依赖同决策 offload Route 的历史 Comm 行已排除并逐条记录；其他 Comp/联合结构残余没有放宽。静态域 TRAIN 14/4416、Validation 6/1104 为空。证据只涉及 CPU 候选准入与机制；优化器、排名、baseline、闭环、GPU 和 `locked_test` 均未进入。

> 2026-09-29 STEP 6.3B：候选语法与支持政策 `PASS`（CPU 合同）。正式池 Route 空、Comm 当前关系绑定、Comp 当前因果 CPU base、Mob 共享 profile，联合结构须为 TRAIN 已见签名；时间观察性只作标签。仍未选候选优化器、未排名、未进入闭环或性能验证。见 STEP 6.3B 合同与收据。

> 2026-09-28 STEP 6.3A：Formal TRAIN action-support 证据审计 `PASS`；Comp causal template 可复现，独立 family factorization 不受支持。候选算法仅有证据建议、未由研究者冻结；无 optimizer、ranking、闭环或 baseline。

> 2026-09-28 STEP 6.2B-PATCH（当前）：研究者冻结 Planner v1 Route `EXPLICIT_NOOP_ONLY`，非空 Route 准入拒绝；`STEP_6_2B=PASS`、`MPC_OBJECTIVE=FROZEN_AND_IMPLEMENTED` 只表示五项 scorer/字典序的 CPU 合同验收。候选方法待决、闭环未就绪；无 Route 优化、性能、baseline、GPU 或 locked-test 结论。下方 6.2B BLOCKED 是历史观察。

> 2026-09-28 STEP 6.2B：冻结 Objective 五项 scorer 与严格字典序已实现并通过一条 validation anchor 的 CPU checkpoint 机制验证，但 pending Route 固定支持和同路径 Host 更新与现有合同存在冲突，`STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`。不把 scorer 接通视为 Planner 方法或性能验收；candidate method、closed loop、baseline、GPU、locked_test 均未开始。

> 2026-09-28 STEP 6.1：`TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT=PASS` 限于一个 Formal Validation anchor/冻结 seed 5601 的 CPU 机制验收，四类当前因果动作的 H4 路径、对应潜变量和规则响应有机器收据。此状态不证明反事实预测准确或 Planner 控制效果；`MPC_OBJECTIVE=NOT_STARTED`、`CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`、`CLOSED_LOOP=NOT_STARTED`，baseline/locked-test/performance claim 均无。

> 2026-09-28 STEP 5.6C：正式 seed 5601 训练已完整结束并通过本地 CPU 验收，`STEP 5.6B=COMPLETE`、`FORMAL_BEST_CHECKPOINT=FROZEN`。五次完整 validation 的最终严格最低 `L_Val=0.07431338784170399`，best/latest 均为 step 5520 且模型张量逐项相同；这是 Formal Validation Observation。之前 6.0A–C 段落对 5.6B“未查询”的说法是各自历史时点，不是当前状态。仍无 baseline、locked-test、模型候选 Planner rollout 或闭环性能结果。

> 2026-09-26 STEP 6.0C：Planner v1 operational action domain 已按研究者决定冻结并做 CPU 合同测试；Search/Learned/Hybrid、目标函数、模型候选 rollout、性能、闭环与安全性仍未决定或未执行。动态可用 CPU 的 simulator source 仍不存在；本域改用静态预算。5.6B 远端状态本 Step 未查询。

> 2026-09-26 STEP 6.0B：CPU verdict=`STATIC_CAPACITY_ONLY`；UAV 配置和行为数据未构成 simulator hard bound。两项约束保持 UNKNOWN，未开始模型依赖候选推演。AirFogSim 本地源码可读，但缺独立 Git metadata，精确本地 Git SHA 不能确认；5.6B 远端未查询。

> 2026-09-26 最新增量：5.6B 正式训练由研究者先前授权在远端独立运行，本 Step 未联系远端，当前进度和 best checkpoint 未核实。6.0A 只通过 CPU 静态候选合同与合成 fixture 验收；最终候选方法、World Model 候选 rollout 和性能仍待后续授权。下文旧的“未正式训练/GPU”属于当时快照。

> STEP 5.5-PATCH 已补齐 Formal Dataset 60 trajectory shard 的全量索引与按需 CPU batch 消费；原 1+1 runtime smoke 不再承担 full-shard 证明。future Return birth 全量检测为 8828 次按窗口-未来步计数的 unsupported/fixed-support 事件，原零计数已失效。未正式训练，未用 GPU，未访问 `locked_test`。

> 本文是导航性状态摘要。当前结果必须回到原始 checkpoint、metrics、manifest 和 audit 验证。
> 截至 2026-09-23，STEP 5.5 Formal Dataset v1 已接受：60 条真实 trajectory、H2/L4、48/12 split、5520 windows、五类 package 与 CPU H=4 interface smoke 均通过。GPU 未使用，`locked_test` 未访问，formal training 未开始。

## 0. 当前实施状态

- Step 2.4 已用真实 non-locked AirFogSim 完成通信 Outcome 最终验收；Step 3.2/3.3 已冻结最小 Dataset/Tensor 合同。
- STEP 4.1 已冻结映射；STEP 4.2A/4.2C-B/4.2C-C 已闭合 graph inputs 与 Flow Raw→Tensor；STEP 4.3A 已物化 typed graph；STEP 4.3B 已实现 History temporal encoding、typed message passing 与 P2A/P2C，输出 `Z_t^{PI,L_g}`。
- 现有代码已在 STEP 4.4 contract 中接入四类动作路由、`Z_t^{PI,L_g}→xi_t^Lat`、结构化 RSSM 边界、逐步规则反馈与预测态动态图重建；5.1B/5.1D 已接入 Loss/KL/Metric，5.2 已接入 CPU training loop；真实重规划与性能验证仍未开始。
- STEP 5.0 已冻结 deterministic mean decoder、Motion/CSI mask-normalized MSE、family-specific posterior teacher、分族 KL、overshooting OFF、prior-dominant curriculum、component mask、无规则状态 loss、joint training 与 prior-only validation/evaluation。
- STEP 5.1A-PATCH 已把 Motion 改为 local one-step delta 并固定到 current physical input slots，同时把 CSI 绑定到 current model relation slot/identity/endpoint/type/RB；5.1B/5.1D 已完成 posterior/loss/metric 与 paired integration，5.2 已完成 CPU training-loop smoke。下一独立门是 STEP 5.3；这不是正式训练或性能证据。

## 1. 当前研究问题

在车联网、无人机、路侧单元和边缘计算组成的动态系统中，如何在严格区分物理连接与业务数据流的前提下，学习动作条件的多步联合状态演化，并最终支持基于预测后果的候选动作选择和滚动重规划。

## 2. 旧协议方法边界（Historical / Archived）

被审计的旧协议候选方法是 `entity_aligned_dual_graph_rssm_v1`：

- 双图底座分别表示物理网络和信息网络；
- 跨图消息只沿真实附着和承载关系传播；
- 节点、物理边、数据流和任务拥有实体级随机动力学；
- 运动状态使用因果历史差分；
- 训练时使用 posterior teacher，正式预测使用 prior-only；
- 先训练 deterministic base，再冻结 base 训练 RSSM；
- 通过 P4 非锁定数据门后，才讨论 P6 规划。

该方法及其两个 seed 结果仅保留原协议含义，不是新定义候选已经确定或实现。

## 3. 已有证据

### 3.1 方法机制和执行证据

- 第一性原理审计发现旧 global RSSM 的实体表达、链路排序和运动输入问题。
- 实体级 RSSM CPU 一致性审计通过了因果运动、动作敏感性、prior target invariant、posterior 局部性、梯度和 strict reload 等机制检查。
- GPU batch probe 选择 batch size 8；execution sentinel 只证明 CUDA 路径可执行，不承担正式性能结论。
- 冻结协议位于 `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

### 3.2 seed 20260831

正式单 seed 验收已通过，详情见 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`。

| 指标 | 数值 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| validation link-F1 delta | `+0.44415` | `>= -0.05` | 通过 |
| calibration link-F1 delta | `+0.86229` | `>= +0.05` | 通过 |
| node-x overall ratio | `0.75475` | `<= 1.25` | 通过 |
| node-x h5/h10/h20 | `0.76995/0.75071/0.75495` | 各 `<= 1` | 通过 |
| throughput/RB/task delay | `0.93555/0.47363/0.01287` | 各 `<= 1` | 通过 |

该结果只覆盖一个 seed 的 unlocked 数据，不能推出跨 seed 泛化，也不能打开 locked test。

### 3.3 seed 20260830

正式单 seed 验收已通过，详情见 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`。

| 指标 | 数值 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| validation link-F1 delta | `+0.45710` | `>= -0.05` | 通过 |
| calibration link-F1 delta | `+0.88509` | `>= +0.05` | 通过 |
| node-x overall ratio | `0.75086` | `<= 1.25` | 通过 |
| node-x h5/h10/h20 | `0.76033/0.74939/0.74960` | 各 `<= 1` | 通过 |
| throughput/RB/task delay | `0.93831/0.47523/0.01175` | 各 `<= 1` | 通过 |

最佳 RSSM checkpoint 为 epoch 40。正式 manifest 共 79 项，零缺失、零哈希差异；strict reload 为 `0/0`，机器重算最大差异为 `4.15555434507553E-09`。

## 4. 当前运行状态

远端训练进程已退出，GPU 已释放。当前没有获准继续运行的训练；旧 `20260832` 不得自动启动，新 Step 2 也未获授权。

## 5. 当前阻塞和未知

- 新定义关键模块尚未实现，不能用旧 P4/P6 状态替代。
- 通信状态充分性、外生事件、proposal 训练、planner objective/risk/fallback 和新实验门仍需冻结。
- 旧 P4 缺第三 seed 和三 seed审计，但它已经不在 active execution queue。
- `locked_test` 保持封存。
- `formal_performance_claim_ready=false`。
- 两个已完成 seed 均通过单 seed 门，但三 seed 的均值、方差、稳定性和正式泛化结论仍未知。
- 最终采用纯候选搜索、学习策略还是混合策略未知。

## 6. 历史失败的科研价值

历史 global complete RSSM、node-x-safe、edge feedback、persistence residual、概率校准等实验不能当作当前方法，但它们解释了为什么当前方案需要实体级 latent、因果运动输入、逐边修正和分阶段训练。失败结果应保留并通过 `EXPERIMENT_INDEX.md` 和原始 audit 追踪。

## 7. 当前科研决策权

- 用户决定研究问题、核心假设、方法取舍、实验目的和最终科研解释。
- AI负责实现、测试、运行、审计、整理和提出有证据的质疑。
- 当理论、代码、数据、运行配置和结果不一致时，先报告冲突，停止扩展受影响实验，不用改名或模糊表述掩盖。

## 8. 单一下一动作

研究者检查已冻结 Raw 层；若明确授权，再进入 Step 3 Dataset/Tensor Contract。在此之前不启动训练、不访问 `locked_test`。

## 2026-09-19 STEP 3.1F 状态

最小 Model-ready Sample & Tensor Contract 已冻结：History `[1,2]` 中明确包含 `O_1+A_1+Y_1+O_2`，Future Action/Target `[2,3]`，input index 来自 History union，Future Action 先做 anchor visibility 检查后引用同一 static index，固定 presence/mask、typed target index、真实 DAG 因果分区、history relation/flow 和严格 Action reference 均已验收。机器 policy 为 `history_causal_observable_object_union`；真实非 locked Raw、样本和合同测试通过；已扫描 4 个 Raw artifact 的 18 个窗口且当前无 unresolved future reference，audit JSON 的 SHA-256 已进入 manifest provenance。该扫描不代表正式 dataset 可用率；正式 batch/split、模型、训练和 `locked_test` 仍未开始。STEP 3.2 未授权。

## 9. 项目重构状态

- 人类可读的项目、代码、架构、实验、结果、冲突、归档、学习和长期协作入口已经建立。
- 机器注册表覆盖 848 个项目文件、607 个 Python 节点和 803 个 artifact 一级目录；Python 解析错误为 0，当前 artifact 控制入口读取错误为 0。
- `AI_CONTEXT/` 九个 ChatGPT 上下文文件已通过结构和证据边界检查；问题路由增至 7 类。
- 7 个重要实验使用统一完整字段，2 个正式结果已与原始 acceptance JSON 自动逐项核对，证据不一致数为 0。
- 6 类重要历史方法记录了尝试原因、实际结果、弃用原因、替代关系和原始证据；5 类常见问题已有只读路由。
- 查询工具既能按自然语言定位当前方法、历史尝试、结果来源和延后任务，也能按精确文件名或 artifact 目录名检索；输出只负责导航，正式结论仍回到原始证据核实。
- 第三个 seed 和远端同步已登记为 `deferred`、`authorization_required=true`、`auto_start=false`。
- 历史 Python 代码先做逻辑归档；因为 94 个历史节点仍被历史脚本/测试引用且全量测试基线仍有环境与 fixture 错误，本轮不做破坏 provenance 的物理移动。
## STEP 6.2A-PATCH (2026-09-28)

Objective source reconciliation and a single non-locked Formal Validation deadline sidecar passed. E2E Flow state exists. Route action and cross-hop model route semantics remain inconsistent with 4.2C-C; `STEP_6_2B_READINESS=BLOCKED`. No scorer, candidate ranking, baseline, GPU, locked_test or closed-loop result exists from this Patch.
## 2026-09-28 STEP 6.2A-CLOSURE

研究者接受现有 Formal best checkpoint、不重训；Planner v1 Route 限定为 Formal Dataset single-hop support。当前只达到 `READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`，不代表 Planner、ranking、closed loop、baseline 或 performance ready。Formal multi-hop coverage 为零，multi-hop code 保留作未来扩展/消融。
