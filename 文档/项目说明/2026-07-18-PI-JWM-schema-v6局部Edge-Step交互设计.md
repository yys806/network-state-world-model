# PI-JWM schema-v6 局部 Edge-Step 交互设计

日期：2026-07-18  
状态：设计已确认，待实施计划复核  
方法身份：`v11 candidate` 的可辨识性增强，不是 v12，也不是已定型 selector

## 1. 背景与设计结论

schema-v5 候选收益可辨识性正式审计完成 6 个固定特征组与 linear/RF/HGB/XGBoost 共 24 个组合。最优 `full_schema_v5__rf` 的 validation active-rate RMSE 为 `233.9031`，未超过 best fixed `230.8556`；只执行 3 个样本，全部集中在 seed 50 的 compute 阶段，只改善 1/10 validation seeds。

full-schema RF 的 candidate-sign PR-AUC 为 `0.4524`、benefit Pearson 为 `0.1992`，但 sample 内排序 Spearman 只有 `0.0862`。这说明 schema-v5 对“是否可能存在收益”有弱信号，却不能稳定识别同一状态下哪个候选更好。

根因假设是 schema-v5 将被修改边和三个 rollout step 压缩为 mean/max/sum，丢失了局部 state--action--response 的配对关系。本设计采用：

> 稀疏 edge-step token 作为无损局部表示，同时生成 step × action-channel 的可解释聚合特征；简单模型先通过 CPU 可辨识性门，之后才训练 GPU candidate-set/token encoder。

不采用仅增加更多全局统计量的方案，因为若再次失败，无法区分“真实无信号”和“聚合继续丢信号”。不先做纯 stage/family routing，因为现有 schema-v5 在各动作族内仍缺少局部交互输入。

## 2. 保持冻结的研究条件

本轮保持以下部分不变：

- PI-JWM world model checkpoint 和推理配置；
- 32 候选动作库、动作投影、applicability 和 action-applied 语义；
- train seeds `0--15,20--43`、calibration `44--49`、validation `50--59`；
- actual-rollout SSE、link/activity 指标及默认 ranked-allocation；
- matched seeds `18--19` 和 external seeds `60--69` 的锁定状态。

schema-v6 只改变 selector 的测试时可观测输入表示和缓存格式，不能将 actual outcome、future truth、oracle choice、真实 benefit/regret 或 seed 身份写入 token。

## 3. Token 单位、容量与顺序

token 单位是：

\[
(\text{sample},\text{candidate},\text{rollout step},\text{edge})
\]

仅当候选动作相对 ranked-allocation baseline 在该 edge-step 的任一动作维发生变化时，生成 token。

冻结容量为每个 sample-candidate 最多 `72` 个 token。现有 schema-v5 统计为：

- train 最大 modified edge-step 数 `67`；
- calibration 最大值 `64`；
- validation 最大值 `64`。

因此 72-token 容量可以无截断覆盖全部现有正式 split。token 按 `(step, edge_index)` 稳定排序；不足部分使用全零 padding 和独立 mask。

质量规则：

- train/calibration/validation 任一候选超过 72 tokens 时，schema-v6 正式缓存生成直接失败，不做静默 top-k 截断；
- smoke 必须输出最大 token 数、p95/p99、padding ratio 和 overflow count；
- edge index 单独保存用于追溯与 token 对齐，不进入模型特征。

## 4. Token 特征合同

每个 token 固定包含以下测试时可观测字段：

1. `step_one_hot_0..2`：3 维；
2. `default_action_0..5`：ranked baseline 在该 edge-step 的六维动作；
3. `action_delta_0..5`：候选相对默认动作的六维差；
4. 当前链路状态：`distance`、`rate_sum`、`csi_mean`、`active_task_count`、`allocated_rb_count`，5 维；
5. `default_predicted_activity`、`default_predicted_rate`，2 维；
6. `predicted_activity_delta`、`predicted_rate_delta`，2 维；
7. `action_delta_l1`，1 维。

共 25 维。候选绝对预测响应可以由默认预测加 delta 重建，不重复保存。task stage、action family 和全局 task/resource/energy proxy 继续保留在 candidate/context 层，不复制进每个 token。

六维动作顺序必须与现有 `edge_action_features` 一致：offload count、RB task count、RB total、CPU task count、CPU total、return count。缓存 manifest 必须显式保存该顺序，不能依赖调用者暗猜。

## 5. 可解释聚合特征

为在 GPU 之前验证信号，schema-v6 从 token 生成固定维 `interaction_pooled` 特征。

按 3 个 rollout step × 6 个 action channel 分组。每组计算：

- modified count；
- signed delta sum；
- absolute delta sum；
- absolute delta max；
- 以该 channel 的 `|action_delta|` 为权重，对 5 个 current-link 字段做加权均值；
- 对 default predicted activity/rate 做加权均值；
- 对 predicted activity/rate delta 做加权均值。

每个 step-channel 共 13 个量，总计 `3 × 6 × 13 = 234` 维。无修改的组全部为零，并通过 modified count 区分“真实零”和“无 token”。

CPU 审计固定比较：

- `schema_v5_full`；
- `interaction_pooled_only`；
- `schema_v5_plus_interaction_pooled`；
- 去掉 current-link、predicted-response、stage/family 的固定消融。

不得在看过 validation 后继续创建新的统计组合。

## 6. 缓存和稳定接口

schema-v6 缓存继续使用一个 split 一个 NPZ 和相邻 manifest，新增：

- `interaction_tokens`：`float32 [sample,candidate,72,25]`；
- `interaction_token_mask`：`bool [sample,candidate,72]`；
- `interaction_token_edge_index`：`int32 [sample,candidate,72]`，padding 为 `-1`；
- `interaction_token_feature_names`：25 个稳定名称；
- `interaction_pooled_features`：`float32 [sample,candidate,234]`；
- `interaction_pooled_feature_names`：234 个稳定名称；
- `token_overflow_count`、token 数分布和 action-feature 顺序写入 manifest。

新增独立 `CandidateInteractionBatch`，不把四维 token 强塞进现有 `CandidateBatch.candidate_features`。现有 schema-v1--v5 loader 保持兼容；schema-v6 loader 必须要求 token、mask、edge index 和名称同时存在，禁止部分降级。

稳定接口：

- `build_candidate_interaction_tokens()`；
- `pool_candidate_interactions()`；
- `save_candidate_interaction_cache()` / 扩展后的统一 cache 保存接口；
- `load_candidate_interaction_cache()`；
- `audit_candidate_interaction_protocol()`。

## 7. 本地 CPU 验证顺序

1. 纯数组单测验证 token 内容、排序、padding、overflow、默认/候选响应重建和候选置换等价性。
2. schema-v6 cache round-trip、SHA、split、名称、动作顺序和 locked-feature 审计。
3. 固定 synthetic tiny end-to-end。
4. 本地真实 64-sample 标签 smoke：只验证 world-model rollout → token → cache → audit 全链路。
5. 重复 64-sample smoke，除 runtime/path 外的 token/cache/result 哈希必须一致。
6. 本地 256-sample smoke，运行 linear/RF/HGB 的 pooled-feature 可辨识性审计；XGBoost 可用时作为对照。

smoke 结果不得宣称指标突破，也不得据此调整 72-token 容量或统计定义。

## 8. CPU→GPU 硬门与停止规则

只有完整 schema-v6 train/calibration/validation 标签生成后，才能做正式 validation 结论。由于完整 32-candidate actual-rollout 标签生成成本高，GPU 的首次用途是生成完整 schema-v6 标签，而不是直接训练新 selector。

完整 schema-v6 简单模型必须同时满足：

- validation active-rate RMSE `<230.8556`；
- 至少 7/10 validation seeds 改善；
- executed positive precision `>=65%`；
- negative-selection rate `<=20%`；
- sample 内排序 Spearman `>=0.20`；
- activity F1 下降不超过 `0.002`；
- link RMSE 相对恶化不超过 `2%`。

若过门，才训练使用 raw token 的 permutation-invariant candidate/token encoder，并保持 calibration defer。若不过门：

- `interaction_pooled` 有改善但未过门：检查 stage/family routing；
- raw-token diagnostic 仍无 sample 内信号：回到 PI-JWM candidate-conditioned forecast 或候选动作构造；
- 禁止通过打开 matched/external、放宽安全阈值或增加无界模型网格追分。

## 9. 解释产物

本地 smoke 输出目录：

`代码/artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/`

至少包含：

- token distribution/overflow audit；
- pooled feature manifest；
- schema-v6 cache manifest 和 SHA-256；
- 64/256 smoke summary；
- 每个 token 可追溯到 sample、candidate、step、edge 和源字段的检查样例；
- 完整 GPU 标签生成命令草案；
- “观测事实、合理解释、待验证假设”三段式状态说明。

本轮本地停止点是：schema-v6 代码、测试、64/256 smoke 和 GPU 完整标签命令全部就绪；不得在完整标签尚未生成时提前训练 selector 或宣称 RMSE 改善。
