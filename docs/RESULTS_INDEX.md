# PI-JWM 结果索引

> 结果索引连接“数字—实验—配置—代码—数据—研究问题”。数字本身不是最终真相，必须回到原始 metrics、checkpoint、manifest 和 audit。

当前可引用数字的结构化副本见 `docs/registries/results_registry.json`；它只负责定位，最终仍以对应原始 metrics 和独立 audit 为准。`build_project_knowledge_index_v1.py --check` 会把 seed、最佳 epoch、9 项门控指标、audit SHA-256、验收状态和 `locked_test` 边界与原始 `single_seed_acceptance.json` 自动比较，不一致时直接失败。

STEP 5.5 只有 Dataset/CPU interface 验收结果，没有预测性能结果：60 trajectories、5520 windows、四动作 coverage、package hashes 与 machine receipts 定位在 `code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`，不得写入性能比较。

> 2026-09-18：以下数字仍可在旧协议边界内引用，但全部属于 Historical / Archived evidence。新 `00–06` 尚无性能结果，不能用这些数字证明新双图、四类动作、目标 RSSM 或 planner 已实现。

## 1. 旧协议可引用结果

### P4 entity RSSM，seed 20260831

唯一正式单 seed 验收入口：

`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`

该文件记录了 79 项 manifest、strict reload、门控重算和 `locked_test_accessed=false`。核心数值：

| 指标 | 值 | 证据范围 |
| --- | ---: | --- |
| validation link-F1 delta | `+0.4441475764` | seed 20260831，unlocked validation |
| calibration link-F1 delta | `+0.8622921256` | seed 20260831，unlocked calibration |
| node-x overall ratio | `0.7547533605` | seed 20260831 |
| node-x h5/h10/h20 ratio | `0.7699514616 / 0.7507086696 / 0.7549541058` | seed 20260831 |
| throughput ratio | `0.9355544639` | seed 20260831 |
| RB occupancy ratio | `0.4736271290` | seed 20260831 |
| task delay ratio | `0.0128686245` | seed 20260831 |
| selected checkpoint | RSSM epoch 39 | base 20 + RSSM 40 epoch |

正确表述是：

> 实体级双图 RSSM 在 seed 20260831 的正式 unlocked 单 seed 验收中通过 9 项数值门。

不能表述为：最终性能已确定、跨 seed 泛化已证明、P4 已关闭或 locked test 已通过。

## 2. 第二个已验收结果

### P4 entity RSSM，seed 20260830

正式 run 位于 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1/`；独立验收入口为：

`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`

最佳 RSSM checkpoint 为 epoch 40。validation/calibration link-F1 delta=`+0.45710/+0.88509`；node-x 总体/h5/h10/h20 ratio=`0.75086/0.76033/0.74939/0.74960`；throughput/RB/task-delay ratio=`0.93831/0.47523/0.01175`。9 项单 seed 数值门全部通过，79 项 manifest 零缺失、零哈希差异，strict reload 为 `0/0`。

正确边界是：两个固定 unlocked seed 已分别通过单 seed验收；第三 seed 和三 seed 审计仍缺失，不能据此关闭 P4、开放 P6 或访问 `locked_test`。

## 3. 机制和执行结果

| 结果 | 说明 | 证据 |
| --- | --- | --- |
| causal motion | 运动特征由历史位置因果差分得到 | `formal_motion_state_v1.py`、P4 第一性原理审计 |
| entity-local stochastic state | node/physical edge/flow/task 分别维护 latent | 实体级模型与 CPU consistency audit |
| prior-only deployment | 正式预测不读取未来目标 | 模型代码、consistency audit、冻结协议 |
| two-stage training | base 训练后冻结，再训练 RSSM | runner、protocol、checkpoint metadata |
| batch 8 execution | CUDA batch probe 在预算内通过 | `pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/` |

这些是机制/执行证据，不等于完整性能结论。

## 4. 历史结果的引用规则

历史失败结果可以用于回答“为什么改方法”，例如 global RSSM 的实体广播限制、运动输入缺失和训练目标冲突；但必须同时标记：

- 使用的旧方法身份；
- 数据和协议范围；
- 失败门或诊断指标；
- 是否被后续方法替代；
- 不能把它和当前实体 RSSM 数字混成同一实验。

## 5. 双向追踪模板

新增结果时按下列链路登记：

```text
研究问题
  → 方法/假设
  → 代码入口及版本
  → 配置和 tensor manifest
  → seed/split/checkpoint
  → 原始 metrics
  → 独立 audit
  → 结论和边界
```

反向查询时，从数字先找到 audit 或原始 metrics，再回到 run summary、checkpoint、manifest、配置和代码；不要只依赖 PPT、README 或摘要表。
