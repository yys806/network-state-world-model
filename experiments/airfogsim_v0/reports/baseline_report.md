# baseline_v0 阶段小结

## 任务设置

- 数据：`dataset_v0_samples.npz`
- 样本数：190
- 划分：训练 133，验证 28，测试 29
- 输入：过去 `H=8` 步节点、链路、任务状态
- 输出：未来 `K=3` 步链路平均速率和任务状态统计
- 模型：`persistence`、`ridge residual`、`mlp residual`

## 当前结果

| 模型 | 全部 MAE | 全部 RMSE | 链路速率 MAE | 链路速率 RMSE | 任务状态 MAE | 任务状态 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.719 | 1.345 | 0.233 | 1.180 | 0.881 | 1.396 |
| ridge residual | 0.723 | 1.224 | 0.802 | 1.664 | 0.697 | 1.037 |
| mlp residual | 2.182 | 8.469 | 6.198 | 16.786 | 0.843 | 1.307 |

## 北大汇报口径评估

这一项已经完成“最小 baseline 搭建”，可以作为下周汇报的第一版实验结果，但还不能作为最终方法效果结论。

当前最重要的发现是：由于仿真步长为 0.1 秒、预测步长为未来 3 步，短期状态变化较小，`persistence` 是很强的基线。Ridge 残差模型在任务状态统计上优于 persistence，但在链路速率上仍不如 persistence；MLP 在当前小样本和高维输入下不稳定，说明后续需要更强正则、更合理特征或更大数据量。

汇报时可以说：第一版 baseline 已经跑通，证明 `dataset_v0` 可以进入训练-评估流程；同时也暴露出当前数据规模较小、短期预测中 persistence 很强的问题。下一步会基于这个 baseline 做扰动实验和置信区间，而不是直接宣称复杂模型已经有效。

## 输出文件

- `baseline_metrics.csv`
- `baseline_summary.json`
- `baseline_link_rate_predictions.png`
- `baseline_task_state_predictions.png`
- `mlp_training_history.csv`
