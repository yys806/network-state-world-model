# PI-JWM schema-v6 局部 Edge-Step 交互实施计划

> 对应设计：`文档/项目说明/2026-07-18-PI-JWM-schema-v6局部Edge-Step交互设计.md`  
> 执行方式：TDD、本地 CPU 优先、保持 world model/候选库/split 冻结

## Task 1：建立 token 数据结构和构造函数

**文件：**

- 新增：`代码/src/pi_jwm/v11_interactions.py`
- 新增：`代码/tests/test_v11_schema6_interactions.py`

**步骤：**

1. 先写 `CandidateInteractionBatch` 失败测试，固定 token `[sample,candidate,72,25]`、mask、edge index、feature names 和 pooled feature合同。
2. 测试 padding token 必须全零、padding edge index 必须为 `-1`、有效 edge index 必须非负、所有特征有限。
3. 先写 `build_candidate_interaction_tokens()` 内容测试：
   - 只对相对 ranked baseline 发生变化的 edge-step 生成 token；
   - 稳定按 `(step, edge)` 排序；
   - default action + action delta 可重建 candidate action；
   - default prediction + response delta 可重建 candidate prediction；
   - current link state 与 edge 对齐。
4. 写 overflow 测试：第 73 个 token 必须 hard fail，不能截断。
5. 写候选置换测试：置换候选后 token 同步置换，内容不变。
6. 实现最小代码并运行：

   ```powershell
   cd D:\shen\网络组\代码
   python -m unittest discover -s tests -p "test_v11_schema6_interactions.py"
   ```

## Task 2：实现 234 维可解释聚合

**文件：**

- 修改：`代码/src/pi_jwm/v11_interactions.py`
- 修改：`代码/tests/test_v11_schema6_interactions.py`

**步骤：**

1. 先写 `pool_candidate_interactions()` 失败测试，固定 3 step × 6 action channels × 13 statistics。
2. 验证 count、signed/absolute sum、absolute max 和 9 个 `|delta|` 加权均值可由原 token 重算。
3. 验证无修改 step-channel 的 13 维全部为零，count 显式为零。
4. 固定 234 个名称和顺序；名称必须包含 step、action feature name 和 statistic。
5. 新增 `append_interaction_pooled_features()`，返回新的 `CandidateBatch`，不修改输入对象；interaction 名称统一加 `interaction_` 前缀。
6. 运行 token/pooled 单测并确认候选置换等价性。

## Task 3：增加 schema-v6 缓存协议并保持 v1--v5 兼容

**文件：**

- 修改：`代码/src/pi_jwm/v11_labeling.py`
- 修改：`代码/tests/test_v11_selector_finalization.py`
- 修改：`代码/tests/test_v11_schema6_interactions.py`

**步骤：**

1. 先写 `save_candidate_interaction_cache()` / `load_candidate_interaction_cache()` round-trip 失败测试。
2. schema-v6 manifest 必须包含 token capacity/dimension、token/pooled feature names、token 数分布、overflow count 和 action-feature 顺序。
3. 缺 token、mask、edge index、pooled features 或名称时必须失败，禁止部分降级。
4. 修改公共保存内部实现以支持可选 interaction payload；原 `save_candidate_label_cache()` 继续写 schema-v5。
5. `load_candidate_label_cache()` 继续兼容 schema-v1--v6，但只返回基础 `CandidateBatch/CandidateOutcome`；需要 token 的调用者必须显式使用 interaction loader。
6. 验证 cache SHA、configuration digest、sample count 和 candidate order。
7. 运行 schema-v6、selector-finalization 和 objective-aligned 相关测试。

## Task 4：接入 actual-rollout 标签生成脚本

**文件：**

- 修改：`代码/scripts/run_v11_selector_candidate_labels.py`
- 修改：`代码/tests/test_v11_schema6_interactions.py`

**步骤：**

1. CLI 新增 `--cache-schema-version {5,6}`，默认 5，保持旧命令兼容。
2. schema-v6 分支复用已在内存中的 `library.actions`、`predictions_by_candidate` 和 current link state 构建 token；不得重复运行 world model。
3. 将 `edge_action_features` 顺序显式传入 builder 和 manifest。
4. schema-v6 使用 interaction cache 保存接口；schema-v5 路径结果 SHA/字段不变。
5. split summary 新增 token distribution/overflow audit。
6. synthetic tiny runner 测试验证 schema-v5/v6 分支和 CLI，不接触 locked split。

## Task 5：扩展可辨识性审计以读取 schema-v6 pooled features

**文件：**

- 修改：`代码/src/pi_jwm/v11_benefit_identifiability.py`
- 修改：`代码/scripts/audit_v11_candidate_benefit_identifiability.py`
- 修改：`代码/tests/test_v11_benefit_identifiability.py`
- 修改：`代码/tests/test_v11_schema6_interactions.py`

**步骤：**

1. schema-v5 特征组顺序和既有测试保持不变。
2. 当 candidate features 含 `interaction_` 前缀时，新增：
   - `interaction_pooled_only`；
   - `full_schema_v6`。
3. `full_schema_v5` 在 schema-v6 batch 中必须排除 interaction 列，提供同缓存内公平对照。
4. 审计 CLI 增加 `--required-schema-version {5,6}`；schema-v6 使用 interaction loader 并调用 append helper。
5. 保持 train/calibration/validation 精确 seed 集、locked split 无 CLI、固定阈值和 3-fold group CV。
6. summary 新增 token protocol SHA、sample-rank Spearman `>=0.20` 门和 schema-v5/v6 对照。
7. tiny schema-v6 end-to-end 输出必须包含 pooled feature manifest、trace、summary 和 SHA-256。

## Task 6：本地真实 64-sample smoke 与确定性

**输出：**

`代码/artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/smoke64/`

**步骤：**

1. 使用冻结 world/policy checkpoint、32 候选和 schema-v6 运行 train/calibration/validation 各 64 个 seed-balanced 样本。
2. 检查 token 最大值 `<=72`、overflow `0`、padding/重建/名称/缓存 SHA 全通过。
3. 使用 linear 对 `schema_v5_full`、`interaction_pooled_only`、`full_schema_v6` 做执行链审计。
4. 重复相同命令到 `smoke64_repeat/`，比较除 runtime/path 外的 cache、token audit、CSV 哈希。
5. smoke 只验证链路，不用于修改特征定义或宣称指标改善。

## Task 7：本地真实 256-sample smoke

**输出：**

`代码/artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/smoke256/`

**步骤：**

1. 运行每 split 256 个 seed-balanced 样本的 schema-v6 标签缓存。
2. pooled 审计运行 linear/RF/HGB/XGBoost 固定模型。
3. 报告 token 分布、cache 大小、构建时间、sample-rank Spearman、top-1 positive ratio 和安全选择结果。
4. 若本地单 split 预计超过 60 分钟，可停止 256 标签扩展，但必须已有 64 deterministic smoke；不得因此改 token 或缩减候选。

## Task 8：全量验证、状态文档和 GPU 交接

**文件：**

- 新增：`代码/scripts/run_v11_schema6_labels_gpu.sh`
- 修改：`本地计划表.md`

**步骤：**

1. GPU 脚本只生成完整 schema-v6 train/calibration/validation 标签和输入 SHA，不训练 selector、不访问 matched/external。
2. 固定输出目录：

   `代码/artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/label_cache_schema6/`

3. 脚本预检 CUDA、checkpoint、源码 SHA、三 split seed 规格；生成后验证 schema=6、token overflow=0、三个 cache configuration digest 一致。
4. 运行全量测试：

   ```powershell
   cd D:\shen\网络组\代码
   python -m unittest discover -s tests -p "test_*.py"
   ```

5. 更新 `本地计划表.md`，记录本地 smoke、是否达到 GPU 停止点和严格的下一步命令。
6. 本轮停止于“完整 schema-v6 标签生成需要 GPU”；未生成完整标签前不训练 selector、不报告正式 RMSE。

## 验收边界

- 本地代码必须通过 token 内容/overflow/padding/重建、cache round-trip、泄漏、确定性和现有全量测试。
- 64-sample deterministic smoke 是最低交付；256-sample smoke 在本地时间门允许时完成。
- 任何 schema-v6 正式结论都必须等待完整 train/calibration/validation 标签；smoke 结果统一标记 `diagnostic_only`。
- matched seeds 18--19 和 external seeds 60--69 全程保持未访问。
