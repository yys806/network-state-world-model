# P4 node-x RSSM 修正诊断与单变量候选

日期：2026-09-06
状态：诊断已完成；修复方法为候选定义，尚未实现、尚未训练。

## 1. 已确认的现象

使用最终有效的 `formal_complete_rssm_v1_1` v3 checkpoint、原冻结的 128 个 validation sample IDs，对正式输出做如下分解：

\[
\hat{x}_{full}=\hat{x}_{base}+\Delta x_{rssm}.
\]

CPU 只读诊断满足 strict checkpoint reload、样本身份复用和逐值分解精确性。结果为：

| 输出 | node-x MAE | 相对 persistence 比值 | 是否满足 1.25 保护线 |
|---|---:|---:|---|
| persistence | 11.85200 m | 1.00000 | 是 |
| formal dual-graph base | 13.71936 m | 1.15756 | 是 |
| base + RSSM correction | 15.26231 m | 1.28774 | 否 |

`RSSM correction` 的绝对值平均为 `3.04245 m`，只有 `23.79%` 的有效节点因该修正而降低误差；h1 到 h20 的每个步长，完整输出 MAE 都高于 base。由此可以确认：本轮 node-x No-Go 的直接来源是 RSSM 连续修正的可靠性不足，不能归因于 base 已经越过保护线。

该分解只定位关联来源，不证明下面的候选修复一定有效。

## 2. 推荐的唯一新变量

候选名称：`node_x_residual_non_degradation_v1`。
类型：训练期单一新增损失项；不新增网络、不改变部署输入。

对训练样本中的每个有效节点和预测步，定义：

\[
L_{safe-x}=\operatorname{mean}\left[\max\left(0,
|\hat{x}_{base}+\Delta x_{rssm}-x^*|-|\hat{x}_{base}-x^*|
\right)\right].
\]

它只处罚“RSSM 修正让 node-x 比 base 更差”的部分。已有 node-state 重构损失继续负责奖励准确预测；新项不替代原损失，也不读取 validation 或 `locked_test`。首个候选固定 `lambda_safe-x=1.0`，不做参数网格；若冻结 sentinel 失败，按停止规则保留失败证据，不临场改权重。

## 3. 保持不变的主线内容

- 正式 physical-information 双图、数据划分、train-only 统计量和 h20 rollout 不变。
- RSSM 的 action-conditioned prior、observation-conditioned posterior teacher、balanced KL、overshooting、prior/teacher 重构不变。
- validation 和部署仍是 prior-only，不能读取未来 target。
- 连续状态及事件输出仍由完整 RSSM 路径产生；新损失不是把 node-x 直接换回 persistence 或绕开 latent。
- seed、样本数、训练预算、link 阈值候选、性能门和保护线均不变。
- `locked_test` 保持封存；P4 未通过前不进入 P6。

## 4. 实现前后的验收顺序

1. TDD 先证明新损失只在修正劣化 node-x 时为正，在修正改善或相等时为零，并正确处理缺失节点 mask。
2. 证明梯度到达 RSSM node-x correction decoder，不改变 base 参数的直接梯度语义。
3. CPU consistency audit 证明训练期可使用 target 计算该损失，但 validation/部署输出仍 target-invariant；checkpoint 必须记录候选名称和 `lambda_safe-x`。
4. 重新冻结协议并做独立 Go/No-Go 审计。
5. 只有上述门全部通过，才允许同一 seed `20260831` 的一次 GPU sentinel。先检查正式性能门；失败立即停止，不运行其他 seed、概率门或 `locked_test`。

## 5. 当前结论

P4 仍为 blocked，`formal_performance_claim_ready=false`。现有完整 RSSM 已证明接口和训练闭环成立，但未通过 node-x 性能保护门。当前唯一待决策事项是是否实施 `node_x_residual_non_degradation_v1`；在确认前不继续训练。

## 6. GPU sentinel 结果（2026-09-06）

该候选已经按冻结协议完成唯一一次 seed `20260831` sentinel。源码、tensor 与协议哈希匹配，训练样本/validation/calibration=`256/128/128`，8 epochs，checkpoint strict reload 和 19 项 manifest 均通过，`locked_test_accessed=false`。

正式 validation 结果：link-F1 delta=`-0.01241 >= -0.05`，throughput/RB/task-delay delta=`-0.33768/-0.64604/-1.35343`，这些保护项通过；node-x MAE=`18.02197 m`，persistence=`11.85200 m`，ratio=`1.52058 > 1.25`，故 sentinel=`no_go`。

同一 sample IDs 的 CPU 分解显示：RSSM correction 平均绝对值已由旧 v3 的 `3.04245 m` 降至 `0.07621 m`，说明约束确实压制了 correction；但 formal base MAE 从旧 v3 的 `13.71936 m` 恶化至 `17.97475 m`，base ratio=`1.51660`，已经单独越过保护线。旧、新 run 的初始化哈希完全相同，因此差异来自新增训练损失改变了联合优化轨迹，不是随机初始化变化。

结论：`node_x_residual_non_degradation_v1` 作为“从头联合训练”的单变量方案失败并停止。不得补跑、调权重或启动其他 seed。可供下一次独立评审的主线候选仅是复用旧 v3 已通过位置门的 base，冻结 base 后再训练 correction；该候选尚未实现、未审计、未获准训练。
