# PI-JWM v8 文献支撑与模块映射

这份文档只记录 v8 设计中实际要用到的文献依据。原则是：优先使用正式发表的期刊/会议论文；只有 arXiv 或未确认正式出版的信息，只作为概念补充，不作为核心支撑。

## v8 总目标

v8 是 PI-JWM 的“满血版第一阶段”，目标不是继续堆 head 或 gate，而是把 v6/v7 暴露出的瓶颈转成模块化升级：

- M0：复现 v7.6 best，建立 v8 入口。
- M1：edge-update 双图消息传递。
- M2：latent rollout。
- M3：specialist-distilled active-rate head。
- M4：Mixture-of-Experts active-rate head。
- M5：multi-task loss balancing。
- 后续诊断：跨 seed、扰动、置信区间和复杂度。

## M1：edge-update 双图消息传递

核心依据：

- Y. Shen, J. Zhang, S. H. Song, K. B. Letaief, “Graph Neural Networks for Wireless Communications: From Theory to Practice,” IEEE Transactions on Wireless Communications, 2023. DOI: 10.1109/TWC.2022.3219840.
- Y. Wang, Y. Li, Q. Shi, Y.-C. Wu, “ENGNN: A General Edge-Update Empowered GNN Architecture for Radio Resource Management in Wireless Networks,” IEEE Transactions on Wireless Communications, 2024. DOI: 10.1109/TWC.2023.3325735.

对 PI-JWM 的作用：

- 支撑把无线网络建成图，而不是普通 MLP 特征表。
- 支撑 edge state 本身需要更新，因为链路速率、干扰和资源分配都发生在边上。
- v8-M1 应实现 physical edge update、information edge update、action-conditioned edge update，再做 node-edge-node 传播。

验证标准：

- 单元测试检查 shape、梯度、置换等变性。
- CPU smoke 跑通。
- same-split 小规模训练不比 v7.6 明显退化。

## M2：latent rollout 与时空图预测

核心依据：

- G. Jin et al., “Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey,” IEEE Transactions on Knowledge and Data Engineering, 2024. DOI: 10.1109/TKDE.2023.3333824.
- M. Jin et al., “A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. IEEE Computer Society document: 10636792.
- L. U. Khan et al., “Digital Twin of Wireless Systems: Overview, Taxonomy, Challenges, and Opportunities,” IEEE Communications Surveys & Tutorials, 2022. DOI: 10.1109/COMST.2022.3198273.
- Z. Tao et al., “Wireless Network Digital Twin for 6G: Generative AI as a Key Enabler,” IEEE Wireless Communications, 2024. DOI: 10.1109/MWC.002.2300564.

对 PI-JWM 的作用：

- 支撑 v8 不能只做 one-step supervised predictor，而要学习隐状态转移。
- 支撑 world model/digital twin 的用途：快速模拟、评估候选动作、服务闭环优化。
- v8-M2 应先做 1-step latent transition，再扩展到 3-step rollout。
- 当前本地实现状态：
  - 已新增 `history_encoder=temporal_conv` 候选路径，用一维时间卷积从历史 latent 序列中抽取当前隐状态。
  - 这只是 STGNN/STGCN 思想的轻量候选，不是完整标准 STGCN；当前没有实现标准 STGCN 的“图卷积 + 时间卷积”堆叠结构。
  - 小样本 CPU smoke 显示 temporal-conv 没有改善 active-rate，因此暂不作为默认 full 初版配置。
  - 已新增 `latent_transition_mode=recurrent` 候选路径：未来每个动作步先做双图 message passing，再用 node/edge GRUCell 更新隐状态。
  - recurrent latent rollout 已通过单测和小样本 CPU smoke，但当前没有改善 active-rate，因此暂不作为默认正式配置。
  - 已新增 `history_encoder=stgcn_light` 候选路径：历史每个时刻先做双图消息传递，再对传播后的 node/edge 序列做时间卷积。
  - STGCN-light 比 `temporal_conv` 更接近“图传播 + 时间卷积”的 STGNN 思路，但当前小样本 test active-rate 仍未改善，因此同样暂不作为默认正式配置。

验证标准：

- 1-step 指标接近 v7.6。
- 3-step rollout 不出现误差爆炸。
- node/link/task 多目标不能单独崩掉。

## v7 有效组件回接：cross-attention 与 active-selected rate path

核心依据：

- v7 真实实验中，`cross_attention + direct + active_mixed` 曾将 active-rate RMSE 从 v6 dual baseline 的 228.318 降到 209.778。
- v7.6 真实实验中，`cross_attention + aux_soft_zero + aux_weight=0.3` 进一步将 active-rate RMSE 降到 206.641，同时 link-rate RMSE = 6.809。
- 注意力机制的一般依据可由 transformer/cross-attention 系列方法支撑；在 PI-JWM 中它不是单独提出的新理论，而是已验证有效的 v7 工程组件。

对 PI-JWM 的作用：

- 保证 v8 不是推翻 v7，而是在 v7 最优路径上继续加入 graph message passing、latent rollout 和 rate head 增强。
- `cross_attention` 在 v8 中用于 physical edge、information edge 和 action 三个 per-edge token 的融合。
- `aux_soft_zero` 用 activity probability 对 active-rate auxiliary head 做软选择，使 inactive link 接近 0，同时保留可微训练路径。

当前本地实现状态：

- 已在 v8 中接回 `fusion_mode=cross_attention`。
- 已在 v8 中接回 `active_rate_auxiliary=True` 和 `rate_output_mode=aux_soft_zero`。
- 小样本 CPU smoke 已跑通，但 active-rate 仍弱，需 full same-split GPU 训练判断是否有效。

## M3：specialist-distilled active-rate head

核心依据：

- J. Gou et al., “Knowledge Distillation: A Survey,” International Journal of Computer Vision, 2021. DOI: 10.1007/s11263-021-01453-z.

本地诊断依据：

- 当前 active-rate specialist 在真实 active link-step 上的 random forest test RMSE = 92.862。
- 当前最佳端到端 neural world model 的 active-rate RMSE = 206.641。
- 这说明 active-rate 幅值里存在可学习信号，但神经世界模型没有充分吸收。

对 PI-JWM 的作用：

- 将 specialist 作为 teacher，只对 active link-step 蒸馏 rate amplitude 或 residual。
- 最终部署仍然是 PI-JWM neural world model，不是两阶段 random forest。

验证标准：

- active-rate RMSE 低于 206.641。
- activity F1 和 task RMSE 不出现明显恶化。
- 如果 teacher loss 只改善 teacher target 而不改善最终 link_rate，立即回退。

## M4：Mixture-of-Experts active-rate head

核心依据：

- S. E. Yuksel, J. N. Wilson, P. D. Gader, “Twenty Years of Mixture of Experts,” IEEE Transactions on Neural Networks and Learning Systems, 2012. DOI: 10.1109/TNNLS.2012.2200299.
- W. Cai et al., “A Survey on Mixture of Experts in Large Language Models,” IEEE Transactions on Knowledge and Data Engineering, 2025. DOI: 10.1109/TKDE.2025.3554028.

对 PI-JWM 的作用：

- active-rate 的分布很异质：不同链路类型、距离范围、动作条件、rate 区间可能对应不同函数形态。
- MoE 可以让不同 expert 处理不同 rate 区域或链路类型，gate 使用 edge latent、action 和 activity probability。

验证标准：

- expert gate 不塌缩到单一 expert。
- active-rate RMSE 低于 M3 或至少低于 v7.6。
- 记录 low/mid/high rate 分桶误差，确认提升来自 active-rate 幅值，不是 inactive link 稀释。

当前本地实现状态：

- 已在 v8 中实现 `active_rate_head_mode=moe`，先作用于 active-rate auxiliary head，而不是替换主 `link_rate_head`。
- 已输出 `rate_expert_weights`，用于后续检查 expert 使用率和 gate collapse。
- 小样本 CPU smoke 已跑通，但没有优于普通 MLP auxiliary head：MoE test active-rate RMSE = 670.010，同口径普通 auxiliary test active-rate RMSE = 666.369。
- 因此 MoE 当前是候选模块，不进入默认正式配置；后续需要 full same-split training、expert 使用率和 rate 分桶误差一起判断。

## M5：multi-task loss balancing

核心依据：

- Y. Zhang, Q. Yang, “A Survey on Multi-Task Learning,” IEEE Transactions on Knowledge and Data Engineering, 2022. DOI: 10.1109/TKDE.2021.3070203.
- A. Kendall, Y. Gal, R. Cipolla, “Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics,” CVPR, 2018. DOI: 10.1109/CVPR.2018.00781.
- Z. Chen et al., “GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks,” ICML, 2018.

对 PI-JWM 的作用：

- PI-JWM 同时预测 node、activity、rate、task。v7.6 已经出现 rate 改进但 task 退化的问题。
- M5 用于约束多任务训练，避免只优化 active-rate 而破坏世界模型整体 rollout。

验证标准：

- active-rate RMSE 不能以明显牺牲 task/node 为代价。
- 记录每个 loss head 的梯度或动态权重。
- 如果复杂方法不稳定，先使用固定权重网格做保守版本。

## 置信区间与校准诊断

核心依据：

- M. Fontana, G. Zeni, S. Vantini, “Conformal Prediction: A Unified Review of Theory and New Challenges,” Bernoulli, 2023. DOI: 10.3150/21-BEJ1447.
- A. N. Angelopoulos, S. Bates, “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,” arXiv:2107.07511.

对 PI-JWM 的作用：

- 用 residual/conformal-style interval 检查 90% 覆盖率和区间宽度。
- active link 的区间要单独评估，不能被大量 inactive link 掩盖。

## 参考仿真器

- Z. Wei et al., “AirFogSim: A Light-Weight and Modular Simulator for UAV-Integrated Vehicular Fog Computing,” IEEE Transactions on Mobile Computing, 2025. DOI: 10.1109/TMC.2025.3641373.
- AirFogSim 在 PI-JWM 中只作为参考仿真器和数据来源，不作为框架主线。
# 2026-06-14 full GPU ablation 状态更新

v8 已完成 same-split 200 epoch GPU full ablation。结果说明：文献支撑的模块不是都应该进入默认模型，必须按指标取舍。

| 模块 | GPU full 结果 | 当前状态 |
|---|---|---|
| cross-attention fusion | 相比 gated，active-rate RMSE 230.302 -> 219.564，但 activity/node/task 变差 | 保留为 rate-side 主候选 |
| temporal_conv history encoder | active-rate RMSE 219.564 -> 208.722，task RMSE 5.190 -> 2.980，但 activity/node 变差 | 候选，可用于 balanced/rate-side 对比 |
| MoE active-rate head | active-rate RMSE 219.564 -> 236.997，整体不占优 | 不进入默认，仅保留诊断接口 |
| recurrent latent rollout | active-rate RMSE 219.564 -> 204.915，超过 v7.6 best 206.641；但 link-rate/node/activity 明显退化 | rate-side best 候选，需要 M5 loss balancing |
| STGCN-light history encoder | active-rate RMSE 219.564 -> 216.956，task 改善但 node 明显变差 | 弱候选，不作为默认 |
| STGCN-full history encoder | active-rate RMSE 219.564 -> 218.493，link-rate 基本持平，但 activity/node/task 不占优 | 已完成严格候选验证，不作为默认 |
| temporal_conv + recurrent | active-rate RMSE 211.340，弱于 recurrent；activity/node/task 比 recurrent 更稳 | 稳定性组合候选 |

当前结论：

- v8 不是推翻 v7，而是在 v7.6 effective path 上继续做 latent dynamics 和时序建模。
- `recurrent latent rollout` 是目前唯一刷新 v7.6 active-rate RMSE 的端到端 v8 模块，但它牺牲了 link-rate、node 和 activity。
- `STGCN-full` 已经补齐“时间卷积 + 图卷积 + 时间卷积”的严格候选，但 full 结果没有超过 temporal_conv/recurrent，因此当前不作为主模型。
- 因此 M5 multi-task loss balancing 的优先级上升：下一步不应继续盲目叠模块，而应约束 recurrent latent 的副作用。

# 2026-06-15 M5 fixed loss balancing 结果更新

M5 先实现固定 loss 权重和 composite checkpoint selection，而不是直接上 GradNorm/uncertainty weighting。原因是当前优先级是可审计地验证“recurrent latent 的副作用能否被约束”，而不是引入新的动态权重不确定性。

实现状态：

- `compute_v8_loss` 支持 node/activity/rate/task 显式权重。
- 训练脚本支持 `--best-metric val_composite`。
- `val_composite = active_rate_rmse + link_rate_rmse + 0.1 * node_rmse + task_rmse - 10 * activity_f1`。
- 本地 67 项单测通过，远程 M5 目标 26 项单测通过。

GPU same-split 200 epoch 结果：

| 配置 | activity F1 | active-rate RMSE | link-rate RMSE | node RMSE | task RMSE | 状态 |
|---|---:|---:|---:|---:|---:|---|
| old recurrent rate-best | 0.846 | 204.915 | 9.181 | 65.953 | 4.316 | rate-side best ablation |
| M5 recurrent composite balanced | 0.964 | 214.680 | 6.042 | 35.489 | 5.553 | balanced/stability candidate |
| M5 recurrent rate checkpoint | 0.920 | 215.103 | 6.607 | 48.217 | 4.580 | M5 codepath control |
| M5 recurrent activity high | 0.962 | 235.996 | 6.654 | 42.584 | 5.672 | 不推荐 |
| M5 temporal recurrent composite | 0.899 | 235.995 | 6.675 | 47.905 | 3.516 | task-friendly but not rate-useful |

结论：

- M5 fixed balancing 能显著修复 old recurrent 的 activity/link/node 副作用，但会牺牲 active-rate 峰值。
- 当前不应把 M5 写成 active-rate 刷新结果；它是稳定性候选和后续约束框架。
- 下一步如果继续 M5，应在 balanced 配置下加强 active-rate amplitude，例如 specialist distillation、active-rate residual calibration 或 active-link 分桶重加权；不是继续简单提高 activity/node/task 权重。

# 2026-06-15 combined stack 互补假设结果

为验证“STGCN、recurrent、MoE、M5 可能联合互补”的假设，已完成 same-split 200 epoch GPU 组合实验。

| 组合 | activity F1 | active-rate RMSE | link-rate RMSE | node RMSE | task RMSE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| STGCN-full + recurrent + M5 | 0.893 | 218.939 | 7.345 | 52.452 | 2.960 | task 友好，但非 balanced/rate 最优 |
| STGCN-full + recurrent + MoE + M5 | 0.898 | 218.066 | 9.172 | 53.991 | 5.228 | MoE 未带来互补收益 |
| STGCN-light + recurrent + M5 | 0.846 | 243.614 | 6.850 | 50.490 | 3.282 | 不推荐 |

结论：组合堆叠没有超过 old recurrent 的 active-rate 204.915，也没有超过 M5 balanced 的整体稳定性。STGCN-full 可以作为 task-friendly ablation 写入论文，但当前不应作为 PI-JWM 主模型。
