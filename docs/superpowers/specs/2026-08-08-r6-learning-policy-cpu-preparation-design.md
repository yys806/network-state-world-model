# R6 学习策略 CPU 准备设计

## 1. 目标

在不改变 R1 数据协议、R2 评价协议、R5 冻结世界模型和 R6 同场景配对协议的前提下，为学习策略建立统一的 CPU 可验证入口。本阶段只完成策略器输入、动作可行域、mask、安全投影、Masked Actor–Critic 与 PPO 式优化的最小前向/反向链，不做正式策略训练，不启动 GPU，不访问 locked-test，也不冻结最终方法。

## 2. 固定边界

- PI-JWM 是研究框架；AirFogSim 仅作为动作执行和结果反馈环境。
- 冻结世界模型以 `eval` 和 `stop-gradient` 方式产生显式状态与隐式 latent；策略损失不得更新世界模型参数。
- 学习策略与四个既有 CPU 规则策略共享同一动作可行域、安全壳、仿真配置、seed、指标和配对身份。
- 开发只读取 train/validation/calibration；locked-test 在 R9 前继续封存。
- 本阶段不实现采样式 MPC，不做 Actor–Critic/PPO 性能排名，不生成最终 reward 权重。

## 3. 组件与职责

### 3.1 `PolicyState`

`PolicyState` 是策略器的唯一状态输入，包含：

- 冻结世界模型产生的联合隐式 latent；
- 当前显式可观测资源状态；
- 当前任务队列与 deadline 摘要；
- 当前可行动作 mask；
- 场景、seed、时隙和协议指纹。

构造函数必须拒绝未来 target、非有限数值、空身份字段和维度不一致。世界模型输出在进入策略器前显式 `detach`，策略训练不能通过该接口回传到世界模型。

### 3.2 `ActionSpec` 与动作 mask

`ActionSpec` 固定每类动作的槽位、取值范围和合法性条件。CPU 准备阶段至少覆盖：

- 卸载目标选择；
- RB 选择；
- CPU 分配比例。

离散动作使用布尔 mask；连续 CPU 分配使用存在性 mask、非负边界和节点容量上限。没有合法离散动作时必须输出显式安全 no-op，不能从全负无穷 logits 中采样。非法 mask、空容量或动作维度漂移必须失败停止。

### 3.3 `SafetyProjector`

安全投影按固定顺序执行：有限值检查 → 实体存在性 mask → 离散动作可行性 → CPU/RB容量投影 → 任务流/依赖约束检查。输出包括：

- 最终可执行动作；
- 原始动作；
- 每项修正原因；
- 是否回退到 no-op；
- 投影前后差值和约束检查结果。

安全投影只修正动作，不修改状态，也不静默吞掉错误。能够安全回退的非法候选记录为 projected/fallback；协议结构错误直接抛出异常。

### 3.4 公共策略接口

所有学习策略实现统一接口：

```text
forward(PolicyState) -> PolicyOutput
act(PolicyState, deterministic, seed) -> ProposedAction
evaluate(PolicyState, action) -> log_prob, entropy, value
```

`PolicyOutput` 必须包含 masked logits、连续 CPU 分配参数、状态价值和可审计元数据。相同参数、输入、seed 和确定性模式必须产生相同结果。

### 3.5 `MaskedActorCritic`

使用共享状态编码器、按动作族分离的 actor heads 和单一 value head。离散 logits 在构造分布前应用 mask；CPU 输出经过非负变换并在节点内归一化，随后仍必须经过 `SafetyProjector`。CPU 门只要求：前向有限、非法动作概率为零、value 有限、单批损失可反向、策略参数发生有限更新、世界模型参数不变。

### 3.6 `PPOCandidatePolicy`

复用 `MaskedActorCritic` 的网络、动作分布和安全投影，只新增 PPO clipped surrogate、value loss 和 entropy bonus。CPU 门只运行冻结小批次的一次更新，并验证 old/new log-prob、ratio、clip、mask 和梯度均有限。PPO 不另建状态编码器，不改变动作语义。

## 4. 数据流

```text
历史显式状态与历史动作
        ↓
冻结 PI-JWM 世界模型（eval + stop-gradient）
        ↓
显式状态 + 隐式 latent → PolicyState
        ↓
ActionSpec 生成合法 mask
        ↓
MaskedActorCritic / PPOCandidatePolicy 提出动作
        ↓
SafetyProjector 生成最终可执行动作与修正台账
        ↓
AirFogSim 执行并返回下一观测、系统指标和约束记录
```

CPU 准备阶段可以使用冻结的真实非锁定样本和最小合成单元测试，但不得把单批更新结果解释为闭环收益。

## 5. 训练与评价接口

- Actor–Critic 最小损失由 policy loss、value loss 和 entropy regularization 组成；本阶段只验证数学与梯度链。
- PPO 最小损失复用相同 advantage/return 输入，额外验证 clipped ratio；advantage 和 return 必须由测试夹具或明确的数据字段提供，不能从 locked-test 或未来真实 target 泄漏。
- 正式 reward、GAE、采样预算、更新轮数和 GPU 训练矩阵留到 CPU 门通过后的独立协议冻结。
- CPU 结果只写动作合法性、数值有限性、梯度隔离、接口一致性和运行成本，不写策略优劣。

## 6. 错误处理

- 非有限状态、logit、价值、损失或梯度：立即失败。
- mask 与动作槽位不一致、活动实体缺少映射、无合法动作却未回退：立即失败。
- 投影后仍违反 CPU/RB/任务流/依赖约束：立即失败并保留失败记录。
- 世界模型参数出现梯度或更新：立即失败。
- locked-test 路径、split 或身份出现：在加载前拒绝。

## 7. 测试与验收门

1. `PolicyState` 拒绝未来字段、非有限值和身份/维度漂移。
2. 离散 mask 后非法动作概率严格为零；全非法时安全 no-op 生效。
3. CPU 动作投影后非负且节点总量不超过容量；同输入同 seed 可复现。
4. `SafetyProjector` 的每次修正有原因和差值台账，投影后硬约束为零。
5. Actor–Critic 单批前向、采样、评估和反向均有限；至少一个策略参数更新。
6. PPO 单批 ratio/clip/value/entropy 和反向均有限。
7. 世界模型参数无梯度且更新前后逐参数一致。
8. CPU smoke 只使用非锁定样本；输出 manifest、配置、失败记录和状态字段。
9. R6、R5.1 与历史双图回归测试继续通过。

只有以上门全部通过，才可把状态改为 `r6_learning_policy_cpu_ready=true`。该状态仍不等于 `r6_gpu_strategy_training_ready=true` 或 `final_method_frozen=true`。

## 8. 预期产物

- `代码/src/pi_jwm/` 下的策略契约、安全投影和学习策略模块；
- `代码/tests/` 下的定向 TDD 测试；
- `代码/scripts/` 下的 CPU 预检 runner；
- `代码/artifacts/preflight/` 下的自校验 CPU 预检 bundle；
- `文档/研究进展/` 下的设计、实施和结果记录；
- `本地计划表.md` 与权威 `PIJWM推进.md` 的状态同步。
