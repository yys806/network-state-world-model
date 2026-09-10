# 当前与历史实验

本文件只记录客观状态，不自动解释科研意义。Source of truth：完整字段见 `docs/registries/experiment_registry.json`，正式数字见 `results_registry.json` 和对应 acceptance JSON。

## 当前 P4 正式实验

| ID | Seed | 数据/协议 | 状态 | 客观边界 |
| --- | ---: | --- | --- | --- |
| `P4-EARSSM-SEED-20260831` | 20260831 | v5 causal-motion h20 / frozen v2 | 单 seed 通过，best epoch 39 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260830` | 20260830 | 同上 | 单 seed 通过，best epoch 40 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260832` | 20260832 | 同上 | deferred，未运行 | 没有结果；需要用户明确授权 |

共同训练入口：`code/scripts/run_formal_p4_entity_rssm_gpu_v1.py`。共同 tensor：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`。冻结协议：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

## 正式验收证据

- seed 20260831：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`
- seed 20260830：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`
- 两份报告均为 `status=passed`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 完整 9 项指标保存在 `docs/registries/results_registry.json`，并由索引生成器自动与 acceptance 核对。

## 前置门

- `P4-EARSSM-CPU-CONSISTENCY`：机制、梯度和 strict reload 证据通过；不证明性能。
- `P4-EARSSM-GPU-BATCH-PROBE`：选择 batch=8，未执行 optimizer step；不证明性能。
- `P4-FIRST-PRINCIPLES-20260906`：历史 global RSSM 表达限制的只读诊断；动机证据，不是新方法性能。

## 重要历史方法

以下均不是当前方法，详情与原始路径见 `docs/registries/historical_method_registry.json`：

- aggregate dual-graph residual baseline：通信/资源诊断有部分通过，node 位置和逐 RB/实体级边界未闭合。
- physical-edge feedback GRU：冻结 sentinel 的 validation link-F1 门失败。
- link persistence residual：单独修链路头不足以闭合完整 P4。
- global complete RSSM：补齐 prior/posterior/KL 语义，但全局池化/广播不能区分实体，且 sentinel node-x 门失败。
- node-x non-degradation loss：修正幅度下降但共享 base 变差，sentinel No-Go。
- v11 selector/ranking：历史决策诊断，不是逐候选世界模型 rollout planner。

## P6 实验边界

`P6-CANDIDATE-ROLLOUT-AUDIT-20260826` 状态为 `blocked_prototype_only`。CPU 原型存在，但 P6 未开放，不得运行正式 planner GPU 或给出规划收益结论。

Unverified：第三 seed 结果、三 seed 均值/方差、locked test、最终泛化和正式 planner 收益均不存在。
