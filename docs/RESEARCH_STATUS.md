# PI-JWM 当前科研状态

> 本文是导航性状态摘要。当前结果必须回到原始 checkpoint、metrics、manifest 和 audit 验证。
> 截至 2026-09-18，当前 active workflow 是最新 `00–06` 的 Step 1 实施审计；旧 P4/P6 和训练结果为 Historical / Archived。当前没有 GPU 训练，`locked_test` 未访问。

## 0. 当前实施状态

- Step 1 定义—实现审计已完成，等待研究者检查；入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/STEP_01_AUDIT.md`。
- 新定义的数据、模型、loss、训练、planner 和 closed loop 均未开始。
- 现有代码可复用时间因果、稳定索引、mask/split 和部分规则/指标工具，但严格双图、四类动作、RSSM 边界、逐步反馈和真实重规划存在结构性缺口。
- 唯一建议下一步是冻结一步轨迹字段与四类动作映射合同；研究者授权前不执行。

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

研究者检查 Step 1；若明确授权，再冻结一个决策步的原始轨迹字段与 Route/Comm/Comp/UAV 四类动作映射合同。在此之前不启动训练、不进入实现、不访问 `locked_test`。

## 9. 项目重构状态

- 人类可读的项目、代码、架构、实验、结果、冲突、归档、学习和长期协作入口已经建立。
- 机器注册表覆盖 848 个项目文件、607 个 Python 节点和 803 个 artifact 一级目录；Python 解析错误为 0，当前 artifact 控制入口读取错误为 0。
- `AI_CONTEXT/` 九个 ChatGPT 上下文文件已通过结构和证据边界检查；问题路由增至 7 类。
- 7 个重要实验使用统一完整字段，2 个正式结果已与原始 acceptance JSON 自动逐项核对，证据不一致数为 0。
- 6 类重要历史方法记录了尝试原因、实际结果、弃用原因、替代关系和原始证据；5 类常见问题已有只读路由。
- 查询工具既能按自然语言定位当前方法、历史尝试、结果来源和延后任务，也能按精确文件名或 artifact 目录名检索；输出只负责导航，正式结论仍回到原始证据核实。
- 第三个 seed 和远端同步已登记为 `deferred`、`authorization_required=true`、`auto_start=false`。
- 历史 Python 代码先做逻辑归档；因为 94 个历史节点仍被历史脚本/测试引用且全量测试基线仍有环境与 fixture 错误，本轮不做破坏 provenance 的物理移动。
