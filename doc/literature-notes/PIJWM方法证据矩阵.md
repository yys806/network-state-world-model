# PI-JWM 方法证据矩阵

更新时间：2026-08-03

这张表只记录已经在 Zotero 中有条目、或明确标记为待核验的来源。综述用于定位路线；涉及 PI-JWM 正式定义时，优先引用表中的原始论文。`当前实现` 描述代码已有证据，`目标版本` 不等于已经实现。

## 证据等级与复用方式

- **直接复用**：方法定义和训练目标可以直接采用，仍需按 PI-JWM 的状态、动作和约束重新实例化。
- **文献改造**：核心思想来自文献，但需要改造成无线双图、任务动力学或动作条件形式。
- **多文献组合**：单篇文献没有完整方案，只组合其明确支持的模块。
- **PI-JWM 候选原创**：是针对本项目的组合或约束设计，必须由后续小实验验证，不能现在当作已证明创新。
- **尚未验证**：只保留线索，不作为正式证据。

## 核心方法矩阵

| 文献（Zotero key） | 场景 | 状态/表示 | 动作或条件 | 预测目标 | 约束 | 方法 | 指标/训练信号 | 数据 | 局限 | 对 PI-JWM 的作用 | 复用方式 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Planning and Acting in Partially Observable Stochastic Domains（Kaelbling et al., 1998；本地全文，Zotero 待核验） | 部分可观测序贯决策 | 真实状态、观测、历史和belief state | 动作 | belief更新和期望回报 | POMDP转移与观测模型 | 状态估计器 + belief上的策略 | 期望折扣回报 | 理论模型 | 经典有限POMDP定义，不包含无线图和连续高维观测 | 支撑PI-JWM严格区分真实状态、观测、历史和latent belief；belief由前一belief、上一动作和当前观测更新 | 直接复用（问题定义） |
| World Models（`68SBG58R`） | 视觉控制 | VAE latent、RNN hidden state | 控制动作 | latent 下一状态和 reward | 环境动力学隐含在数据中 | VAE + MDN-RNN + controller | latent prediction、reward prediction、控制回报 | 游戏/视觉仿真 | 无无线图、无显式资源和硬约束 | 给出“表示学习 + 动力学模型 + 冻结模型规划”的基本分工 | 直接复用（概念层） |
| Dream to Control（`MXUFTZJZ`） | 连续控制 | latent belief/state | 动作序列 | imagined latent rollout、reward/continuation | 主要靠模型和策略学习，约束不显式 | latent imagination、actor-critic | imagined return、value consistency | 控制环境 | 不处理链路可行性、任务 DAG 或资源容量 | 支持 PI-JWM 使用动作条件多步 rollout 服务规划器 | 文献改造 |
| TD-MPC2（`8D6KTM8K`） | 连续控制 | latent state | 连续动作 | 短视窗 latent transition 和 value | MPC 视窗与动作边界 | task-oriented latent dynamics + TD-MPC | TD loss、multi-step consistency、return | 多种控制基准 | 没有图结构和无线传播变量 | 支持“先训练可滚动的 latent transition，再做 MPC/规划” | 文献改造 |
| When to Trust Your Model: Model-Based Policy Optimization（Janner et al., 2019；原始论文，待入库核验） | 模型式强化学习 | 环境状态与学习动力学 | 策略动作 | 短模型rollout与策略收益 | 模型偏差和rollout长度 | MBPO、从真实数据分支的短rollout | 模型泛化、样本效率、控制回报 | 控制基准 | 不处理离线无线动作支持、双图和资源硬约束 | 支撑限制rollout长度，并独立评估模型误差被规划器放大的风险 | 文献改造 |
| MOPO: Model-based Offline Policy Optimization（Yu et al., 2020；原始论文，待入库核验） | 离线模型式强化学习 | 离线数据支持内状态 | 离线行为策略动作和候选动作 | 保守的策略价值 | 数据分布偏移、支持外状态动作和模型不确定性 | 不确定性惩罚的悲观规划 | 真实MDP回报下界、离线控制回报 | 离线控制基准 | 不确定性估计可能不可靠，且不含无线结构约束 | 支撑行为策略、动作支持、OOD门控和不确定性惩罚进入PI-JWM数据与规划接口 | 文献改造 |
| COMBO: Conservative Offline Model-Based Policy Optimization（Yu et al., 2021；原始论文，待入库核验） | 离线模型式强化学习 | 数据内外状态—动作 | 策略动作 | 保守价值和策略改进 | out-of-support状态—动作 | 对模型rollout上的支持外价值进行保守正则 | 价值下界、策略改进、离线回报 | 离线控制基准 | 目标是策略价值，不直接保证状态预测或无线可行性 | 支撑把支持外动作单独标记、拒绝或保守处理，而不是依赖平均预测误差 | 文献改造 |
| I-JEPA（`37DBEF53`） | 图像表征学习 | context/target embedding | mask/位置条件 | 目标块的 latent 表示 | 非生成式，避免重建无关细节 | joint-embedding predictive architecture | embedding distance、EMA target encoder | ImageNet 图像 | 条件不是通信动作，不能直接生成网络状态 | 支持在双图或 CSI 表征上做 latent 预测而非逐字段重建 | 文献改造 |
| Learning Latent Wireless Dynamics from CSI（`T26KRVS2`） | 单用户无线链路 | CSI channel chart latent | 用户速度 | 后续 CSI latent 表示 | 条件变量是速度，未建模任务/资源动作 | wireless JEPA、channel encoder + predictor | latent prediction accuracy、look-ahead accuracy | 实测 CSI | 单用户/单链路；不含任务队列、RB/CPU动作 | 是 PI-JWM 最直接的无线 latent 动力学证据；支持传播状态预测和真实 CSI 校准 | 文献改造 |
| Model-based deep RL for channel access（`KEIK3W25`） | 无线信道接入 | 信道/接入观测 | 接入动作 | 动作后信道接入结果或回报 | 接入协议和可用信道 | model-based DRL | throughput、collision、reward | 仿真/无线接入场景 | 状态和动作维度窄，通常不含任务 DAG | 支持“预测动作后果，再由策略器选择动作”的无线闭环 | 文献改造 |
| Multilayer Networks（`Q8RWSG3X`） | 通用多层网络 | 层内节点/边与跨层关系 | 无控制动作 | 网络结构与动力学分析 | 层、节点对齐与跨层耦合定义 | multilayer network formalism | 理论性质与结构统计 | 理论/多领域网络 | 不包含无线任务、动作条件转移和资源约束 | 支撑物理图、信息图保持独立关系并通过显式跨图关系耦合；不直接证明 PI-JWM 动力学 | 直接复用（严格双图定义） |
| Relational Inductive Biases, Deep Learning, and Graph Networks（`I8DBTTIF`） | 通用图结构学习 | 带属性的节点、边和全局量 | 可将动作作为条件或全局属性 | 结构化表示与关系推理 | 图连接结构提供归纳偏置 | graph network block | 表征/推理任务指标 | 多类图任务；预印本 | 不给出双图唯一性、无线流守恒或概率世界模型 | 支撑属性图和独立节点/边特征空间；PI-JWM的三类跨图关系属于进一步改造 | 文献改造 |
| Modeling Relational Data with Graph Convolutional Networks（`HC9IWXEK`） | 多关系知识图谱 | 实体节点与带类型关系边 | 无连续控制动作 | 链路预测与实体分类 | 关系类型显式区分 | relational GCN | link prediction、entity classification | 知识图谱 | 不处理时变物理路径、任务生命周期或动作后果 | 支撑$q^{\mathrm{src}}/q^{\mathrm{host}}/q^{\mathrm{exec}}/q^{\mathrm{ret}}$等关系类型显式编码，不能替代跨图流守恒定义 | 文献改造 |
| Optimal Allocation of Interconnecting Links in Cyber-Physical Systems（`C4M4482F`） | 双网络CPS | 两个独立网络的节点、层内边和显式跨网络依赖矩阵 | 无控制动作 | 级联失效与鲁棒性 | 跨网络依赖关系 | interdependent-network model | 级联规模、鲁棒性 | 理论/仿真 | 不定义移动无线MEC的空间邻域、信息代理、任务或世界模型 | 作为PI-JWM严格双图的顶刊结构锚点：两图各自保留层内语义，再显式记录跨图关系；不直接证明本文具体实例化 | 直接复用（结构原则） |
| Graph Neural Networks for Scalable Radio Resource Management（`65HG2GWI`） | 无线RRM | 通信实体节点与带信道状态的无线边特征 | 无线资源动作 | 资源分配结果 | 无线链路、干扰与资源约束 | message-passing GNN | sum-rate等RRM目标 | 无线网络模型 | 单张无线图和单步动作映射；没有独立物理空间图、任务DAG或动作条件rollout | 作为无线语义锚点：CSI/信道增益/干扰属于通信信息边特征；PI-JWM再增加独立物理图与显式跨图耦合 | 文献改造 |
| GNNs for Wireless Communications（文献地图列出的 `10.1109/TWC.2022.3219840`，Zotero 待核验） | 无线网络 | 节点/边信道图 | 功率、RB、关联等 | 链路质量或资源分配 | 链路和功率边界 | message passing GNN | rate、SINR、资源效率 | 仿真/测量混合 | 当前条目和全文尚未完成回读 | 支撑物理图、edge update、置换等变性；正式写入前要补 Zotero 核验 | 尚未验证 |
| ENGNN（文献地图列出的 `10.1109/TWC.2023.3325735`，Zotero 待核验） | 无线资源管理 | 节点和 edge state | RRM 动作 | 资源分配相关边量 | 资源总量和链路约束 | edge-update empowered GNN | rate/分配误差 | 无线 RRM 数据 | 当前条目和全文尚未完成回读 | 直接启发物理边、信息边分别更新后再做 node-edge-node 传播 | 尚未验证 |
| RouteNet-Fermi: Network Modeling with Graph Neural Networks（`EMNT5GPV`） | 网络性能建模 | 拓扑、路由、流量和链路状态图 | 路由/配置作为条件 | 路径时延等网络KPI | 路径—链路关系、调度/QoS配置 | 关系型GNN网络模型 | 延迟预测误差、跨路由/故障/规模泛化 | OMNeT++仿真 | 不是动作条件无线世界模型；跨更大规模仍需实验证明 | 支撑跨拓扑、路由、链路故障与图规模实验，并防止把参数共享误写成泛化保证 | 文献改造 |
| A survey on GNNs for time series（`2SLIDP7J`） | 时空序列 | 图节点序列 | 通常无控制动作 | forecasting/imputation/anomaly | 由任务定义 | STGNN、时空消息传递 | MAE/RMSE、预测稳定性 | 多领域数据 | 综述，不给 PI-JWM 特定动作协议 | 支持双图历史窗口和多步预测的选择，但不能代替原始方法证据 | 多文献组合 |
| Digital twin-driven collaborative scheduling（`TEB9ARYE`） | 异构任务与边缘资源 | 任务、设备、边缘资源 | 卸载/调度 | 调度回报、完成情况 | 任务依赖、资源和时延 | DT + multi-agent DRL | delay、energy、task completion | 仿真/数字孪生 | 通常不学习可校准的无线传播 latent | 支持任务/资源动力学与数字孪生闭环的场景边界 | 文献改造 |
| Dependency-aware task offloading in multi-UAV MEC（`7XHL5935`） | 多 UAV 协同 MEC | DAG 任务、UAV/边缘节点 | 卸载、调度、可能含轨迹 | 依赖任务完成和时延 | DAG precedence、CPU、链路和截止时间 | dependency-aware offloading | completion、delay、energy | 仿真 | 未证明与 CSI latent 联合建模 | 支持信息图中任务依赖边、任务生命周期和硬约束定义 | 文献改造 |
| OpenRAN Gym（`BBURMK47`） | O-RAN 测试与 RIC | RAN KPI、UE/cell 状态 | xApp/RIC 控制 | KPI 变化和策略效果 | RAN 控制接口与资源边界 | 数据采集工具箱、闭环控制 | throughput、PRB、KPI | 测试床/真实或半真实 RAN | 任务 DAG、CPU 和 UAV 运动通常缺失 | 支持真实 RAN 外部校准与动作接口审计，不假定能直接训练完整 PI-JWM | 多文献组合 |
| Colosseum: The Open RAN Digital Twin（`5U24ZPJY`） | 大规模 O-RAN 仿真/测试 | RAN、信道、设备和网络遥测 | 可编程 RAN/场景配置 | 网络 KPI、策略后果 | 真实协议栈和测试床配置 | digital twin + wireless emulation | KPI、吞吐、时延、可靠性 | 可复现实验平台 | 不是完整 MEC 任务/资源数据集 | 用于外部验证物理通信部分和跨场景迁移 | 文献改造 |
| DeepSense 6G（`R6HEQTEH`） | V2I、V2V、行人、无人机等 | 同步多模态传感、位置和通信观测 | 通常无 RRM 动作标签 | 信道/定位/感知相关预测 | 由测量配置决定 | 多模态监督/自监督 | communication、sensing、positioning 指标 | 真实测量，40+ 场景、百万级数据点 | 没有统一任务队列、CPU/RB 和反事实动作 | 用于传播、移动性和多模态观测校准；不当作完整 PI-JWM 训练集 | 多文献组合 |
| DeepMIMO（`JLCMKGRT`） | mmWave/massive MIMO | 射线追踪 CSI、位置、天线和场景配置 | 通常无动作标签 | CSI、波束、位置或链路质量 | 射线追踪配置 | 可控信道数据生成 | rate、beamforming、prediction | 射线追踪生成 | 非真实在线测量；没有任务和资源生命周期 | 补充物理信道长尾与可控反事实，不能证明真实泛化 | 文献改造 |
| AERPAW full-stack drone measurements（`X8XZX3ZH`） | 无人机无线测试床 | UAV 位置/轨迹、无线测量、实验配置 | 可编程实验控制 | 端到端通信 KPI | 测试床功率、协议和空域约束 | real testbed measurement | throughput、latency、coverage 等 | 真实测试床 | 任务/CPU/RB 语义不完整，实验配置需逐字段核对 | 用于 UAV 传播和移动性外部验证，不直接补齐任务动力学 | 文献改造 |
| Conformal Prediction: A Unified Review（`10.3150/21-BEJ1447`，待入库核验） | 概率预测 | 预测分数/残差 | 无动作要求 | 覆盖区间 | exchangeability 或近似条件 | conformal calibration | coverage、interval width | 多领域 | 非 PI-JWM 专用；覆盖率不等于动力学正确 | 支持不确定性区间和校准指标的理论依据 | 尚未验证 |
| Learning Latent Multimodal Dynamics for Optimized Resource Planning（`10.1109/TWC.2025.3644600`，已核实 DOI，全文待获取） | 多模态无线/资源规划 | latent multimodal state | 资源规划动作 | latent dynamics 和规划效果 | 资源规划约束 | multimodal latent dynamics | planning utility、prediction | 论文元数据已核实，全文未核验 | 不能在全文核验前作为具体方法证据 | 作为候选的多模态资源动力学路线 | 尚未验证 |
| AirFogSim（`10.1109/TMC.2025.3641373`） | UAV-车联网-雾计算仿真 | 车辆/UAV/RSU、任务、资源、链路和动作派生量 | 卸载、RB/CPU 等仿真调度动作 | 下一时刻状态和任务/资源结果 | 仿真器内置运动、链路、任务和资源规则 | reference simulator/data generator | task utility、RB/CPU、energy、link/node/task errors | 仿真 | 不等于真实网络；当前数据字段并非全部真实 CSI | 作为 PI-JWM 主数据源和反事实生成器，不能写成 PI-JWM 框架 | 直接复用（数据源边界） |

## 当前结论

1. 目前有直接全文证据的核心方法链是：`POMDP观测/历史/belief边界 → World Models/Dreamer/TD-MPC2动作条件latent rollout → I-JEPA/W-JEPA latent predictive representation`，以及无线接入和图网络文献提供的动作/边结构。
2. `DeepSense 6G`、`Colosseum`、`OpenRAN Gym`、`DeepMIMO`、`AERPAW`分别只能补齐观测、测试床、射线追踪或 RAN 控制的一部分，不能假定为包含任务、资源、信道和动作的单一完整数据集。
3. MBPO、MOPO和COMBO共同说明模型偏差、短rollout、离线动作支持和支持外保守处理必须进入PI-JWM框架，但这些方法尚未完成Zotero全文回读，当前只用于确定审计接口，不能作为PI-JWM具体算法已经成立的证据。
4. TPDS双网络论文直接支撑“两张层内图＋显式跨网络关系”的结构原则，JSAC无线GNN论文直接支撑“信道状态作为通信图边特征”；物理空间邻域、复合信息代理、任务DAG辅助地位和动作条件世界模型仍是PI-JWM的多来源组合，必须通过消融验证，不提前宣称已经被单篇文献证明。
