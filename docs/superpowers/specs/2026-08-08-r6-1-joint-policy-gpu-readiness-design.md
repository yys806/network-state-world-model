# PI-JWM R6.1 联合策略 GPU 训练前协议设计

日期：2026-08-08  
阶段边界：只完成正式协议、CPU 实现与预检；不启动 GPU，不访问 locked-test，不重训世界模型，不重建 R1 数据。

## 1. 目标与非目标

R6.1 的目标不是再次优化世界模型，而是把 R5.1 冻结的双图世界模型接到一个可审计的策略闭环：

1. 从当前与历史双图观测得到显式状态和隐式 belief；
2. 在当前 AirFogSim 状态下构造一组完整、合法的联合调度候选；
3. Actor-Critic 或 PPO 对候选进行评分并选择一个候选；
4. 通过统一安全验证和 AirFogSim 原生调度接口执行卸载、RB、CPU 动作；
5. 从执行后的真实反馈重建分解奖励、约束和下一状态；
6. 形成可供 GAE/策略更新消费的严格 transition/rollout；
7. 冻结后续 GPU 公平实验矩阵和停止规则。

本阶段不把 v5 selector/ranking 诊断接口重新包装成主方法，也不宣称策略性能已经得到验证。AirFogSim 只承担仿真、数据生成、执行与反馈；PI-JWM 才是框架主体。

## 2. 为什么选择“完整可行联合方案候选集”

### 2.1 三种方案比较

| 方案 | 表达方式 | 优点 | 主要风险 | 结论 |
|---|---|---|---|---|
| A：逐任务因子化动作头 | 分别为每个任务输出卸载目标、RB 和 CPU | 理论表达最自由 | 任务数动态、动作维度巨大、三类动作容易互相冲突，必须再训练任务级编码器 | 不作为首轮正式入口 |
| B：完整可行联合方案候选集 | 每个候选都包含一套卸载、RB、CPU 绑定，策略只选择候选编号 | 动作天然成套，可统一做 DAG、邻接、RB 和 CPU 安全检查；可供 Actor-Critic、PPO 和规则基线公平复用 | 候选集是受控子集，不等于全局穷举动作空间 | 采用 |
| C：继续只学习 CPU | 卸载/RB 固定 no-op，只训练 CPU | 实现最简单 | 不满足正式联合动作定义，后续 GPU 结果仍需返工 | 拒绝 |

### 2.2 研究边界

候选集表示当前时隙中经合法性规则筛选出的**受控联合动作子集**，不是对原始组合动作空间的穷举，也不宣称包含全局最优动作。候选生成规则、上限、排序和截断必须版本化；所有比较方法接收完全相同的候选集和 mask，避免因搜索空间不同造成不公平。

## 3. 状态协议

策略输入保留两条并行状态路径，不能只保留其中一种：

- 显式状态 $$\mathbf{s}^{\mathrm{exp}}_t$$：由当前与历史的物理图、信息图、任务/DAG、资源和动作字段按 R1 冻结协议构造，保持可解释字段、mask、单位和实体映射。
- 隐式状态 $$\mathbf{s}^{\mathrm{lat}}_t$$：由 R5.1 冻结 Graph-RSSM 在仅使用当前及历史观测和历史动作的条件下生成，不能读取未来 target。

策略上下文定义为：

$$
\mathbf{x}_t = \operatorname{concat}\!\left(\mathbf{s}^{\mathrm{exp}}_t,\mathbf{s}^{\mathrm{lat}}_t\right).
$$

消融实验至少保留三种输入：仅显式、仅隐式、显式与隐式联合。状态源 checkpoint、trajectory ID、slot、split、历史窗口和哈希必须写入每条 rollout 的审计字段。

## 4. 正式联合动作协议

### 4.1 完整候选

第 $$k$$ 个候选定义为：

$$
\mathbf{a}_{t,k}=\left(\mathcal{O}_{t,k},\mathcal{B}_{t,k},\mathcal{C}_{t,k}\right),
$$

其中：

- $$\mathcal{O}_{t,k}$$：卸载绑定集合，每条记录为“任务 ID、源节点 ID、目标节点 ID”；
- $$\mathcal{B}_{t,k}$$：RB 绑定集合，每条记录为“任务 ID、RB ID 集合”；
- $$\mathcal{C}_{t,k}$$：CPU 绑定集合，每条记录为“执行节点 ID、任务 ID、分配 CPU 量”；
- $$m_{t,k}\in\{0,1\}$$：候选合法 mask；
- $$\mathbf{d}_{t,k}$$：候选描述向量，只含当前可知的规模、负载、资源、优先级、截止期裕量和动作变化信息。

候选 0 固定为“AirFogSim 默认合法调度”，作为安全回退和所有策略的共同参照。其他候选由确定性模板在同一默认调度基础上修改，首轮模板包含 deadline-first、priority-first、load-balance、rate-aware 和 energy-conservative；模板名称是候选生成规则，不是待比较论文方法。

### 4.2 合法性约束

候选生成和执行前验证必须同时满足：

1. DAG 释放：卸载任务来自 `getAllToOffloadTaskInfos(..., check_dependency=True)`；
2. 邻接：目标节点来自当前 `getNeighborNodeInfosById()`；
3. 阶段：正在计算、已完成或已失败任务不能再次卸载；
4. RB：同一 RB 在同一时隙不重复分配，总索引小于 `getNumberOfRB(env)`；
5. CPU：只给已位于执行节点且处于可计算阶段的任务分配，节点总分配不超过其 CPU 容量；
6. 实体绑定：每个 task/node/RB ID 必须在当前时隙实体表中存在；
7. 因果性：候选描述、mask 和选择只允许读取时刻 $$t$$ 及以前信息；
8. 硬约束失败：任何硬约束失败都使候选失效，不允许用负 reward 换取执行。

### 4.3 执行顺序

正式时序固定为：

1. 在默认调度前捕获因果状态和合法实体；
2. 调用 `algorithm.scheduleStep(env)` 生成 AirFogSim 默认合法调度；
3. 依次覆盖卸载、RB、CPU 绑定；
4. 再次执行统一安全验证与投影；
5. 调用 `env.step()`；
6. 从 AirFogSim 公共状态/调度接口读取本步事实反馈；
7. 写入 transition ledger。

这一路径复用现有已测试的默认调度、卸载修改、RB 剩余容量投影和 CPU 容量投影语义，不修改 AirFogSim 内核。

## 5. 策略模型

策略不直接输出每个任务的自由组合，而是对候选集合评分：

$$
e_{t,k}=f_\theta\!\left(\mathbf{x}_t,\mathbf{d}_{t,k}\right),
\qquad
\pi_\theta(k\mid\mathbf{x}_t)=
\operatorname{MaskedSoftmax}_k(e_{t,k},m_{t,k}).
$$

价值网络输出：

$$
V_\phi(\mathbf{x}_t).
$$

正式候选方法为：

- Masked Actor-Critic：单次使用 rollout 计算策略和值函数更新；
- Masked PPO：使用相同策略/价值网络、相同候选集和相同 rollout，仅替换为 clipped surrogate 多 epoch 更新。

CPU 规则基线至少包括 default、deadline-first、priority-first、load-balance、rate-aware、energy-conservative。GPU 方法和 CPU 基线必须使用同一候选生成器、同一约束验证器和同一评价器。

## 6. 奖励与约束协议

### 6.1 transition 事实

每个真实 transition 只从执行后可归因于本时隙的增量事实构造：

- $$n^{\mathrm{on}}_t$$：本步按期完成任务数；
- $$n^{\mathrm{fail}}_t$$：本步新增失败/逾期任务数；
- $$D_t$$：本步完成任务时延之和；
- $$Q_t$$：本步交付数据量；正式系统目标 NPZ 中历史字段名为 `delivered_data_total`，但已由逐步序列与轨迹报告核对为每时隙量，不再做累计差分；
- $$E_t$$：本步 UAV 能量消耗增量；
- $$h_t$$：硬约束违反数。

P95/P99 时延、Jain 公平性、资源利用率、action regret 和不确定性校准均保留为 episode/系统评价指标，不强塞进单步 reward。

### 6.2 train-only 尺度

所有尺度只从 R2 冻结的 36 条 train 轨迹估计：

$$
\sigma_D=P_{95}\!\left(\{D_t:D_t>0\}_{\mathrm{train}}\right),
$$

$$
\sigma_Q=P_{95}\!\left(\{Q_t:Q_t>0\}_{\mathrm{train}}\right),
$$

$$
\sigma_E=P_{95}\!\left(\{E_t:E_t>0\}_{\mathrm{train}}\right).
$$

validation、calibration 和 locked-test 不参与尺度计算。尺度文件必须保存数据清单哈希、轨迹数、有效样本数、分位数方法和数值。

### 6.3 分解奖励

正式 `service_first_v1` 奖励为：

$$
r_t = n^{\mathrm{on}}_t-n^{\mathrm{fail}}_t
-0.1\operatorname{clip}\!\left(\frac{D_t}{\sigma_D},0,1\right)
+0.1\operatorname{clip}\!\left(\frac{Q_t}{\sigma_Q},0,1\right)
-0.1\operatorname{clip}\!\left(\frac{E_t}{\sigma_E},0,1\right).
$$

三项次级指标绝对贡献总和不超过 0.3，因此一次按期完成或一次新增失败不会被时延、吞吐量、能耗的尺度组合反转。每条 transition 同时保存原始事实、归一化分量、权重和总 reward，禁止只保存标量。

若 $$h_t>0$$，transition 标记为 invalid 并触发失败回退；硬约束不进入可权衡的 reward。

## 7. Transition、GAE 与训练批协议

每条 transition 至少包含：

$$
\tau_t=(\mathbf{x}_t,\{\mathbf{d}_{t,k},m_{t,k}\},k_t,\log\pi_{\mathrm{old}},V_t,
\mathbf{r}^{\mathrm{raw}}_t,r_t,\mathbf{x}_{t+1},d_t,\mathcal{I}_t),
$$

其中 $$\mathcal{I}_t$$ 是 split、scenario、seed、trajectory、slot、checkpoint 和协议哈希身份信息。done 和 truncated 必须分开；环境时间上限截断允许 bootstrap，真实终止不 bootstrap。

GAE 固定为：

$$
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t,
$$

$$
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},
\qquad
\hat R_t=\hat A_t+V_t.
$$

首轮冻结 $$\gamma=0.99$$、$$\lambda=0.95$$。批构造不得跨 trajectory、seed 或 split 串联；失败 seed 必须保留在运行清单和失败日志中，不能静默删除。

## 8. GPU 公平实验矩阵（只冻结，不在本阶段运行）

| 项目 | 冻结值 |
|---|---|
| 学习方法 | Masked Actor-Critic、Masked PPO |
| 状态消融 | explicit-only、latent-only、explicit+latent |
| 正式运行数 | 2 方法 × 3 状态 × 3 seed = 18 |
| seed | 20260803、20260804、20260805 |
| 每个正式运行最大环境步 | 100000 |
| rollout 长度 | 128 |
| minibatch | 32 |
| PPO epoch | 4 |
| 学习率 | 3e-4 |
| PPO clip | 0.2 |
| value loss 权重 | 0.5 |
| entropy 权重 | 0.01 |
| gradient norm clip | 0.5 |
| 评估频率 | 每 10000 环境步，仅 validation |
| checkpoint 选择 | validation `service_first_v1` 平均回报最高且硬约束为 0；并列时按期完成率高者优先，再按平均时延低者优先 |
| 提前停止 | 5 次连续 validation 评估未改善；仍保留最大步上限和最佳 checkpoint |
| 正式运行前 | 只允许 1 seed、2000 环境步 smoke；smoke 不能作为正式结果 |
| calibration | 只用于固定安全/不确定性阈值，不参与 checkpoint 选择 |
| locked-test | R9 前保持封存 |

CPU 规则基线无需 GPU，但必须在相同场景、相同 episode/seed、相同候选集、相同执行器和相同系统评价器上运行。MPC/MPPI 属于后续规划器比较，不进入首轮 18 个 GPU 训练运行，避免把训练方法和搜索预算混成一个变量。

## 9. CPU 预检通过条件

只有同时满足以下条件，才能写出 `r6_gpu_strategy_training_ready=true`：

1. 候选 0 永远存在且合法；所有候选可重复生成；
2. 卸载、RB、CPU 三类动作均至少有一个真实非 no-op 样例被安全执行；
3. DAG、邻接、RB 唯一性、CPU 容量和实体绑定故意破坏时均被拒绝；
4. Actor-Critic 与 PPO 在候选 mask 下前向、采样、反向均为有限值；
5. 真实 AirFogSim 至少完成一段多步闭环 rollout，transition 身份连续且无未来/跨 split 数据；
6. reward 原始分量可从真实反馈重算，重算值与 ledger 一致；
7. GAE 与手算参考一致，终止/截断 bootstrap 语义通过测试；
8. R1、R2、R3、R4、R5.1 和已有 R6 回归测试继续通过；
9. 正式预检报告明确 `gpu_used=false`、`locked_test_accessed=false`、`world_model_retrained=false`、`dataset_regenerated=false`。

若任一项不满足，只能报告具体阻塞项，不能把接口就绪冒充为 GPU 就绪。

## 10. 文献和既有证据边界

- PPO 的 clipped surrogate 和多 epoch 小批更新依据 Schulman 等《Proximal Policy Optimization Algorithms》（arXiv:1707.06347）。
- GAE 的 $$\gamma$$-$$\lambda$$ advantage 递推依据 Schulman 等《High-Dimensional Continuous Control Using Generalized Advantage Estimation》（arXiv:1506.02438）。
- Actor-Critic/RSSM 的状态—策略分工沿用 PI-JWM 已冻结的 Dreamer/PlaNet 类世界模型证据，但 R6 不照搬其视觉观测或完整控制框架。
- Bou Chaaya 等 IEEE TWC 2026 的耦合 JEPA 支持“控制/资源状态与无线状态的方向性 latent 条件动力学”，不直接规定 PI-JWM 的双图对象、联合动作或候选策略。
- AirFogSim 的 DAG 释放、邻接、RB 和 CPU 公共调度接口是本项目真实执行语义的唯一来源；PI-JWM 只做适配、验证、记录和学习，不修改第三方内核。
