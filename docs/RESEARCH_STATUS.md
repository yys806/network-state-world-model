# PI-JWM 当前科研状态

> 本文是导航性状态摘要。当前结果必须回到原始 checkpoint、metrics、manifest 和 audit 验证。
> 截至 2026-09-21，Raw/定义 01、最小 Dataset/Tensor/定义 02、STEP 4.3A Builder、STEP 4.3B Encoder 与 STEP 4.4 Structured RSSM World Model Contract 已冻结。STEP 5.1A 初版冻结声明因 multi-horizon Motion/slot 问题被撤回，STEP 5.1A-PATCH 已修复并冻结 target contract。STEP 5.0 的 10 项 loss/training/evaluation 决策不变；Loss/Posterior/Metric/Training 实现仍未开始。GPU 未使用，`locked_test` 未访问，`formal_dataset=false`。

## 0. 当前实施状态

- Step 2.4 已用真实 non-locked AirFogSim 完成通信 Outcome 最终验收；Step 3.2/3.3 已冻结最小 Dataset/Tensor 合同。
- STEP 4.1 已冻结映射；STEP 4.2A/4.2C-B/4.2C-C 已闭合 graph inputs 与 Flow Raw→Tensor；STEP 4.3A 已物化 typed graph；STEP 4.3B 已实现 History temporal encoding、typed message passing 与 P2A/P2C，输出 `Z_t^{PI,L_g}`。
- 现有代码已在 STEP 4.4 contract 中接入四类动作路由、`Z_t^{PI,L_g}→xi_t^Lat`、结构化 RSSM 边界、逐步规则反馈与预测态动态图重建；真实重规划、Loss/训练与性能验证仍未开始。
- STEP 5.0 已冻结 deterministic mean decoder、Motion/CSI mask-normalized MSE、family-specific posterior teacher、分族 KL、overshooting OFF、prior-dominant curriculum、component mask、无规则状态 loss、joint training 与 prior-only validation/evaluation。
- STEP 5.1A-PATCH 已把 Motion 改为 local one-step delta 并固定到 current physical input slots，同时把 CSI 绑定到 current model relation slot/identity/endpoint/type/RB；12-sample non-locked development receipt 已重建通过，19/19 focused 与 102/102 related regression 通过。现在停止并等待研究者决定是否授权 STEP 5.1B。Loss/Posterior/Metric/Training 仍未实现。

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
