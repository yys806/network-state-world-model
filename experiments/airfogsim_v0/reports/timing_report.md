# timing_v0 阶段小结

## 任务设置

- 目的：补充“复杂度/计算量”问题的第一版实测证据。
- 仿真侧：运行 AirFogSim demo 场景，统计 `scheduleStep + env.step()` 的逐步耗时。
- 模型侧：复用 `dataset_v0`，训练 Ridge residual baseline，统计测试集推理耗时。
- 预测设置：历史窗口 `H=8`，预测步长 `K=3`。

## 当前结果

| 项目 | 平均耗时 |
|---|---:|
| AirFogSim scheduler | 0.4099 ms/step |
| AirFogSim env.step | 5.5504 ms/step |
| AirFogSim scheduler + env.step | 5.9602 ms/step |
| AirFogSim 估算 3-step rollout | 17.8807 ms |
| Ridge residual baseline | 0.004364 ms/sample |
| Persistence baseline batch | 0.001973 ms/29 samples |

按当前小场景和 Ridge baseline 估算，AirFogSim 的 3-step 显式 rollout 平均耗时约为 Ridge 单样本推理的 `4097.4` 倍。这个倍数只用于说明当前小场景下存在在线推理加速空间，不能直接外推到更大场景或最终模型。

## 怎么解释

AirFogSim 的每一步会显式执行交通、任务、通信、计算、存储和能量等模块，因此多步 rollout 需要重复调用完整仿真流程。学习式模型在训练完成后，可以把历史窗口编码为特征，并用一次前向推理输出未来 `K` 步预测，所以在线推理可能更快。

需要强调的是：这不是说当前 Ridge baseline 已经可以替代 AirFogSim。当前结果只说明“复杂度/耗时评估流程已经建立”，后续还需要在误差可接受的前提下，对更强的双图模型或动作条件 world model 做同样计时。

## 输出文件

- `timing_comparison_logscale.png`
- `airfogsim_step_timing.png`
- `airfogsim_step_timing.csv`
- `timing_summary.json`
