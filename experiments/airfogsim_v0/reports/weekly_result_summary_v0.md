# 本周实验与分析结果汇总 v0

## 1. 当前已经完成的工作

本周已经把“平台能跑”和“数据能用”推进到可评估、可建模的阶段。当前链路是：

仿真日志 -> 训练样本 -> baseline -> 扰动测试 -> 置信区间 -> 多 seed -> 跨 seed 泛化 -> 严格动作日志 -> 动作条件 world model 接口。

| 工作 | 产出 | 说明 |
|---|---|---|
| AirFogSim 数据导出 | `node_states.csv`, `link_states.csv`, `task_states.csv` | 得到节点、链路、任务三类原始日志 |
| `dataset_v0` | `dataset_v0_samples.npz` | 单 seed 历史窗口到未来标签样本 |
| baseline | `baseline_v0/` | persistence、Ridge residual、MLP residual |
| 扰动测试 | `robustness_v0/` | clean 训练、noisy 测试，观察误差随噪声变化 |
| 置信区间 | `uncertainty_v0/` | 基于验证集残差分位数输出 80%/90% 区间 |
| 状态转移机制 | `airfogsim_mechanism_report.md` | 梳理 AirFogSim 每一步如何更新状态 |
| 复杂度计时 | `timing_v0/` | 对比显式仿真 rollout 和轻量模型推理耗时 |
| 多 seed 仿真 | `multiseed_v0/` | 验证同一场景下随机轨迹会变化 |
| 多 seed 数据集 | `dataset_multiseed_v0` | 5 个 seed，共 950 个样本 |
| 跨 seed baseline | `cross_seed_baseline_v0` | 检查模型是否能从训练 seed 泛化到测试 seed |
| 动作代理接口 | `action_proxy_v0` | 第一版从状态日志反推的动作侧变量 |
| 严格动作日志 | `strict_action_v0` | 直接在 scheduler 决策时记录卸载、RB、CPU、UAV 移动动作 |
| 动作条件 baseline | `action_conditioned_baseline_v0` | 对比只用状态和状态加动作的跨 seed 预测效果 |

## 2. baseline 结果

单 seed `dataset_v0` 上，第一版 baseline 的主要结果如下：

| 模型 | 全部 RMSE | 链路 RMSE | 任务 RMSE | 解释 |
|---|---:|---:|---:|---|
| persistence | 1.345 | 1.180 | 1.396 | 直接用最后一个历史状态延续到未来，短期预测很强 |
| Ridge residual | 1.224 | 1.664 | 1.037 | 任务状态预测更好，链路速率预测不如 persistence |
| MLP residual | 8.469 | 16.786 | 1.307 | 当前小样本、高维输入下不稳定 |

结论：persistence 很强是因为当前仿真步长是 `0.1s`，预测未来 3 步只对应 `0.3s`。后续模型必须超过这个强 baseline，才算真正有效。

## 3. 扰动实验结果

扰动实验是在输入侧加入噪声，标签仍然使用原始未来状态。它模拟观测误差、链路测量误差和任务负载估计误差。

| 噪声强度 | persistence 全部 RMSE | Ridge residual 全部 RMSE | 主要现象 |
|---:|---:|---:|---|
| 0.00 | 1.345 | 1.224 | 无扰动时 Ridge 整体略好 |
| 0.05 | 1.545 | 1.724 | 两者误差上升 |
| 0.10 | 2.470 | 2.651 | 任务状态误差明显增大 |
| 0.20 | 4.145 | 5.215 | Ridge residual 退化更明显 |
| 0.30 | 5.998 | 7.270 | 强扰动下当前学习式 baseline 不够稳 |

结论：当前不能说模型已经抗干扰。现在能说明的是，扰动评估流程已经建立，且第一版 residual baseline 在强扰动下会退化。后续需要做扰动训练、结构化建模或更强正则。

## 4. 置信区间结果

当前置信区间使用验证集残差分位数法，不是复杂贝叶斯模型。做法是先训练点预测模型，再在验证集上统计残差分布，用分位数给测试预测加上下界和上界。

| 区间 | 整体覆盖率 | 平均宽度 | 解释 |
|---|---:|---:|---|
| 90% | 0.902 | 5.517 | 覆盖率接近目标 90%，区间偏宽 |
| 80% | 0.849 | 2.105 | 覆盖率略高于 80%，区间更窄 |

结论：当前已经不只是输出一个点预测，还能输出预测区间。后续可以替换成 conformal prediction、模型集成或 MC dropout。

## 5. AirFogSim 状态转移与复杂度

AirFogSim 的一步状态演化可以抽象为：

$$
s_{t+1}=\mathcal{F}_{\mathrm{sim}}(s_t,a_t,\xi_t)
$$

其中，$s_t$ 是当前系统状态，包含节点、链路和任务；$a_t$ 是调度动作，包含卸载、RB 分配、CPU 分配和 UAV 移动；$\xi_t$ 是随机因素，包含车辆到达、任务生成和信道衰落；$\mathcal{F}_{\mathrm{sim}}$ 是 AirFogSim 内部显式仿真规则。

计时实验结果：

| 项目 | 平均耗时 |
|---|---:|
| AirFogSim `scheduler + env.step` | 5.9602 ms/step |
| AirFogSim 估算 3-step rollout | 17.8807 ms |
| Ridge residual inference | 0.004364 ms/sample |

结论：在当前小场景中，显式仿真 rollout 和轻量模型推理之间存在明显在线耗时差距。这只能说明后续 world model 有在线加速空间，不能说明 Ridge baseline 已经可以替代 AirFogSim。

## 6. 多 seed 与跨 seed 结果

多 seed 仿真使用同一场景配置，改变随机种子 `[0, 1, 2, 3, 4]`。当前结果显示，不同 seed 会产生不同车辆数、任务数、成功率和链路速率。

`dataset_multiseed_v0` 张量形状：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `x_node` | `(950, 8, 37, 7)` | 过去 8 步节点状态 |
| `x_link` | `(950, 8, 188, 5)` | 过去 8 步链路状态 |
| `x_task` | `(950, 8, 9)` | 过去 8 步任务聚合状态 |
| `y_node` | `(950, 3, 37, 7)` | 未来 3 步节点标签 |
| `y_link` | `(950, 3, 188, 5)` | 未来 3 步链路标签 |
| `y_task` | `(950, 3, 9)` | 未来 3 步任务标签 |

跨 seed baseline 采用 seed `0,1,2` 训练，seed `3` 验证，seed `4` 测试：

| 测试 seed 4 | 全部 RMSE | 链路 RMSE | 任务 RMSE |
|---|---:|---:|---:|
| persistence | 1.530 | 2.280 | 1.178 |
| Ridge residual | 3.303 | 6.265 | 1.209 |

结论：当前紧凑 Ridge residual baseline 跨 seed 泛化不好，尤其是链路速率预测明显退化。这说明只靠聚合特征和线性 residual 不够，后续需要更结构化的双图编码和动作条件 world model。

## 7. 严格动作日志 v0

之前的 `action_proxy_v0` 是从状态日志里反推动作侧变量。现在新增的 `strict_action_v0` 是直接在 AirFogSim 调度器做决策时记录动作。

记录的动作包括：

- 卸载动作：任务从哪个节点卸载到哪个目标节点。
- 回传动作：任务完成后选择哪个 RSU 回传。
- 通信动作：每个 offloading 任务分配了哪些 RB。
- 计算动作：CPU callback 给每个 computing 任务分配多少 CPU。
- UAV 移动动作：每个 UAV 的速度、方向角、俯仰角和目标来源。

对齐后的动作张量：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `a_hist` | `(950, 8, 13)` | 与历史输入窗口对齐的动作变量 |
| `a_future` | `(950, 3, 13)` | 与未来标签窗口对齐的动作变量 |

每个 seed 的动作数量：

| seed | offload | return | rb | cpu | uav |
|---:|---:|---:|---:|---:|---:|
| 0 | 131 | 2 | 131 | 371 | 400 |
| 1 | 75 | 1 | 75 | 206 | 400 |
| 2 | 109 | 2 | 109 | 290 | 400 |
| 3 | 46 | 2 | 46 | 100 | 400 |
| 4 | 80 | 1 | 80 | 198 | 400 |

结论：现在已经可以把世界模型输入写成“历史状态 + 历史动作”，也可以把未来动作作为 rollout 条件。也就是后续不只是预测 $s_{t+1}$，而是预测：

$$
\hat{s}_{t+1:t+K}=f_{\theta}(s_{t-H+1:t}, a_{t-H+1:t+K})
$$

这一步比 `action_proxy_v0` 更扎实，因为动作来自调度器决策本身，而不是从结果状态里反推。

## 8. 动作条件 baseline 结果

基于 `strict_action_v0`，已经做了第一版动作条件 baseline。实验仍然使用跨 seed 划分：seed `0,1,2` 训练，seed `3` 验证，seed `4` 测试。

对比三种输入：

- `persistence`：直接延续最后一个历史状态。
- `state_only_ridge`：只输入历史节点、链路、任务状态。
- `state_action_ridge`：输入历史状态 + 严格动作张量。

测试 seed 4 的结果如下：

| 模型 | 全部 RMSE | 链路 RMSE | 任务 RMSE |
|---|---:|---:|---:|
| persistence | 1.530 | 2.280 | 1.178 |
| state_only_ridge | 3.303 | 6.265 | 1.209 |
| state_action_ridge | 2.990 | 5.824 | 0.785 |

结论：加入严格动作后，Ridge baseline 相比只用状态有提升，尤其任务状态 RMSE 从 `1.209` 降到 `0.785`。这说明动作变量确实包含有用的状态转移信息。但它仍然没有超过 persistence，所以不能说当前模型已经很好。更准确的说法是：动作条件建模方向是有信号的，但线性 compact baseline 不够，后续需要双图编码或 latent world model。

## 9. 当前创新点表达

不要说 AirFogSim 是创新点。更准确的表达是：

AirFogSim 提供可控仿真环境。当前工作的重点是把仿真日志组织成适合世界模型学习的联合状态序列，包含物理图、通信图和任务状态；同时直接记录调度动作，使样本从“状态预测”推进到“动作条件状态推演”。在此基础上，可以进一步训练动作条件 world model，用于多步状态推演和多场景预训练迁移。

可以拆成三层：

- 数据组织：把原始 `node/link/task` 日志整理成联合时间序列。
- 动作接口：把卸载、RB、CPU、UAV 移动动作直接记录并对齐到样本窗口。
- 评估体系：不只看预测误差，还看扰动稳定性、置信区间、在线耗时和跨 seed 泛化。

## 10. 还没有完成的部分

当前还没有完成正式 world model 训练。

下一步优先级：

1. 在严格动作日志基础上，搭建动作条件 baseline。
2. 做扰动训练，而不只是 clean 训练、noisy 测试。
3. 搭建双图编码 baseline，对比单图/普通序列输入。
4. 在此基础上再做动作条件 latent world model。
