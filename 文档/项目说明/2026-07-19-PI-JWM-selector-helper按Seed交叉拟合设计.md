# PI-JWM v11 Selector Helper 按 Seed 交叉拟合设计

日期：2026-07-19

状态：设计已确认，待书面复核后进入实施计划

方法身份：`v11 candidate` 的标签协议修复；不是 v12，也不代表 selector 已定型

## 1. 背景、证据与设计结论

schema-v6 的完整 train/calibration/validation 标签、局部 edge-step token 和 234 维交互聚合特征已经生成并通过协议审计。候选库仍有充分上限：validation sample-oracle active-rate RMSE 为 `105.835`，约 `75.8%` 的 validation 样本存在优于默认动作的候选。因此，当前失败不能归因于“候选库完全没有改善空间”。

但正式 selector 训练和可辨识性审计均未通过：

- CandidateSetBenefitRanker validation RMSE 为 `233.9876`，与默认动作相同，defer ratio 为 `1.0`；
- 去掉 uncertainty 后 RMSE 为 `234.0443`，说明问题不是单纯由回退阈值过严造成；
- schema-v6 pooled-feature 审计最优 validation RMSE 为 `233.5657`，仅改善约 `0.42`，且只改善 `2/10` seeds；
- 不使用 defer 的 rank-only 选择在 validation 上约为 `286--300`，被选候选实际改善率不足 `9%`，说明候选排序方向本身不可靠。

进一步检查候选分布发现，helper-dependent 候选存在明显的 train--validation 反转。例如 `rb_repair__k16__value_head__persistent` 在 train 上改善 `40/40` seeds、RMSE 约 `154.929`，但到 calibration/validation 不再稳定；validation 最稳的固定候选仍只有约 `230.485`。代码路径确认，当前 `_fit_helper_models()` 在全部 40 个 selector train seeds 上拟合 score/value helper，随后用同一个 helper 给 train、calibration、validation 生成候选。于是：

- train 候选是 helper 的样本内输出；
- calibration/validation 候选是 helper 的样本外输出；
- selector 训练面对的是一种不可部署的候选分布，验证面对的是另一种候选分布。

本轮采用以下设计结论：

> 对 selector train seeds 做固定五折、按 seed 分组的 out-of-fold 候选生成；每个训练样本的候选只能由未见过该 seed 的 helper 产生。calibration 和 validation 由全部 40 个 train seeds 拟合的最终 helper 生成。三个 split 统一绑定到一个交叉拟合协议清单，在重新训练 selector 之前先验证候选分布和收益可辨识性是否恢复。

该方案优于两个备选方向：leave-one-seed-out 更严格但需要 40 次 helper 拟合，计算与审计成本没有必要；直接删除 helper-dependent 候选最快，但 validation 固定候选的现有最好结果约为 `230.485`，不足以支撑 RMSE `<200` 的目标。

## 2. 冻结边界

本轮只修正候选标签生成协议，不改变以下研究条件：

- PI-JWM world model checkpoint、推理配置和 rollout 语义；
- 32 个候选的名称、顺序、动作投影、applicability 和 action-applied 语义；
- schema-v6 edge-step token、交互聚合特征和 outcome 指标定义；
- selector train/calibration/validation/test/external 的 seed 边界；
- actual-rollout SSE、active-rate RMSE、activity F1、link RMSE 和 task--energy 安全审计口径；
- ranked-allocation baseline 和现有 defer/Pareto 规则。

本轮不得通过改变 world model、扩大候选数量、打开 matched test、放宽安全阈值或加入 future/oracle 特征来掩盖标签协议问题。

## 3. 固定五折划分

selector train seeds 固定为 `0--15,20--43`，共 40 个。按排序后的 seed 列表做 round-robin 五折划分，以同时覆盖早期和后期 seed：

| fold | held-out seeds | helper train seeds |
|---|---|---|
| 0 | `0,5,10,15,24,29,34,39` | 其余 32 个 selector train seeds |
| 1 | `1,6,11,20,25,30,35,40` | 其余 32 个 selector train seeds |
| 2 | `2,7,12,21,26,31,36,41` | 其余 32 个 selector train seeds |
| 3 | `3,8,13,22,27,32,37,42` | 其余 32 个 selector train seeds |
| 4 | `4,9,14,23,28,33,38,43` | 其余 32 个 selector train seeds |

每个 fold 中，helper 只能读取该 fold 之外的 32 个 seed；候选和 actual-rollout 标签只为该 fold 的 8 个 held-out seeds 生成。五个 fold 合并后必须恰好覆盖全部 40 个 train seeds 和 15,600 个 train samples，每个 `sample_id` 恰好出现一次。

calibration seeds `44--49` 和 validation seeds `50--59` 不参与 helper 拟合。它们统一使用在全部 40 个 selector train seeds 上拟合的 final helper 生成候选。world-model validation seeds `16--17`、matched test seeds `18--19` 和 external seeds `60--69` 保持锁定。

## 4. 数据流与职责边界

### 4.1 交叉拟合协议层

在 `代码/src/pi_jwm/` 中提供独立的协议函数，负责：

- 根据冻结 seed 规格构建并验证五折划分；
- 检查 train fold 互斥、并集完备和每折 8 个 held-out seeds；
- 为每个 sample 解析其 fold 身份，但不把 fold 或 seed 身份写入 selector 特征；
- 生成一个全局、规范化、可哈希的 cross-fit protocol manifest。

协议清单至少包含：split seed 规格、五折 held-out/helper-train seeds、`train_helper_mode=seed_crossfit_5fold`、`evaluation_helper_mode=full_selector_train`、helper 超参数、候选协议、world/policy checkpoint SHA、schema 版本和源码 Git SHA。

### 4.2 候选标签生成层

现有候选构造、projection、actual rollout 和 schema-v6 特征生成逻辑继续复用。脚本新增显式运行模式：

- `crossfit_train_fold`：拟合某一 fold 的 32-seed helper，只生成该 fold held-out train samples；
- `full_train_eval`：在全部 40 个 selector train seeds 上拟合 final helper，只生成 calibration/validation；
- 原有非交叉拟合 train 生成模式不得作为正式 selector 输入，但保留为历史兼容和诊断用途，并在 manifest 中标记 `in_sample_helper`。

每个 fold 的输出保留独立 NPZ、manifest、helper 训练 seed、held-out seed、样本数、配置 digest、文件 SHA-256 和复现命令。生成器不得暗中根据 validation 结果改变 helper 参数或候选配置。

### 4.3 确定性合并层

五个 train fold 缓存通过独立合并函数生成正式 OOF train cache：

1. 校验所有 fold 使用相同的全局 cross-fit protocol digest、schema、候选顺序、特征名和 checkpoint；
2. 校验 helper-train seeds 与 held-out seeds 无交叉；
3. 校验五折样本无重复、无缺失，且与 `sample_index.csv` 中冻结的 train samples 完全一致；
4. 按 `sample_id` 稳定排序后合并所有数组；
5. 写入每个样本的 fold provenance 供审计，但 provenance 不进入模型特征；
6. 记录五个源缓存及合并缓存的 SHA-256。

正式 train、calibration、validation manifest 共享同一个全局协议 digest。digest 表示完整实验协议，而不是要求三个 split 使用同一组 helper 训练 seed；split-specific helper mode 和来源在各自 manifest 中显式记录。

## 5. 运行顺序与质量门

### 5.1 本地 CPU smoke

先执行 synthetic 单测，再用每 fold 64 个样本做真实 smoke。必须验证：

- held-out seed 不进入对应 helper 训练集；
- 五折 fold 生成和合并的候选顺序、shape、schema-v6 token 合同一致；
- 相同参数重复运行时，除路径和 runtime 外的结果与哈希一致；
- calibration/validation 的 final helper 只读取 40 个 selector train seeds；
- matched test 和 external holdout 没有被读取。

任一泄漏、覆盖、确定性或 cache integrity 检查失败时停止服务器正式生成。

### 5.2 服务器正式标签

服务器依次生成五个 OOF train folds，再生成 calibration 和 validation，最后合并正式 OOF train cache并执行 schema-v6 handoff audit。由于每个 train sample 仍只执行一次 actual rollout，本轮主要增加五次 helper 拟合和分 fold 调度，不会把 actual-rollout 成本扩大五倍。

### 5.3 先审计、后训练

新缓存完成后先重复以下只读审计，不立即训练复杂 GPU selector：

- train/calibration/validation 的固定候选逐 seed 表现；
- helper-dependent candidate 的收益符号、排名和跨 split 稳定性；
- schema-v5、interaction-pooled-only 和 full-schema-v6 可辨识性；
- candidate gate、非平凡样本比例、identity win 和 action-applied；
- train 与 validation 的 stage/family/context 漂移。

只有可辨识性审计显示正向变化，才进入 CandidateSetBenefitRanker 重训。若 OOF train 上 value-head persistent 不再呈现“几乎所有 seed 都最佳”的异常，同时 calibration/validation 的方向一致性提高，即说明协议修复达到了预期机制效果。

## 6. Selector 训练与冻结门槛

协议修复不改变 selector 的既有比较框架：full-label RF/GB、pointwise benefit regressor、CandidateSetBenefitRanker 和 feature-group ablation 使用同一份新 OOF train cache；defer 阈值仍只由 calibration 确定。

validation 进入 matched test 前必须同时满足：

- active-rate RMSE `<230.8556`；
- 至少 `7/10` validation seeds 优于同口径 ranked baseline；
- executed positive precision `>=65%`；
- negative-selection rate `<=20%`；
- activity F1 下降不超过 `0.002`；
- link RMSE 相对恶化不超过 `2%`；
- 配置、checkpoint ensemble、defer 阈值和 Pareto 规则全部冻结。

sample 内 rank Spearman `>=0.20` 作为解释性目标和失败定位指标，但不单独授权打开 test。validation 未通过全部硬门时，不读取 seeds `18--19`，也不以 sample-oracle、test-best 或取消 defer 的结果代替 deployable 结论。

若通过门槛，matched test 只打开一次，并继续使用原 A/B/未通过验收：`<200` 才能称 A 级突破，`[200,213.160874)` 只能称 B 级改善，`>=213.160874` 不得称 selector 定型。external holdout 只在最终结构冻结后生成和评价。

## 7. 失败分支与停止规则

本设计将不同失败定位到不同层次：

1. **OOF 候选 train oracle 明显恶化，但 validation oracle 不变**：这是去除样本内乐观后的正常结果，继续看收益规律是否更一致，不能用旧 train oracle 作为回退理由。
2. **OOF 后固定候选和收益符号跨 split 更一致，但简单模型仍未过门**：允许训练现有 raw-token CandidateSetBenefitRanker，不创建新的无界模型网格。
3. **OOF 后收益符号、排名和 validation 仍不可辨识**：停止 selector 扩模；结论指向 helper/candidate-conditioned rollout 本身，而不是继续调整 defer。
4. **validation 仅被某一个 seed 拉动或安全指标失败**：保持 `v11 candidate`，不打开 matched test。
5. **协议或锁测试审计失败**：作废对应缓存并重新生成，不手工修改 manifest 或跳过检查。

## 8. 稳定接口

计划新增或扩展以下接口，具体文件边界在实施计划中落实：

- `build_seed_crossfit_folds()`：构建冻结五折；
- `audit_seed_crossfit_folds()`：验证互斥、覆盖和锁定 split；
- `build_crossfit_protocol_manifest()`：生成规范配置和 digest；
- `fit_helper_for_fold()` / 复用后的等价接口：仅从指定 helper-train indices 拟合；
- `merge_crossfit_label_caches()`：确定性合并五折 schema-v6 缓存；
- `audit_crossfit_label_protocol()`：审计样本 provenance、特征泄漏和 cache 完整性。

脚本只负责解析参数、调度和落盘；seed 规则、合并和审计逻辑放在 `代码/src/pi_jwm/`，以便单测和后续复用。

## 9. 测试设计

测试至少覆盖：

- 五折的 held-out/helper-train 互斥、并集完备、固定顺序和确定性；
- calibration/validation/matched/external 不得出现在任何 fold helper train 中；
- 每个 train sample 只由一个 held-out fold 生成；
- 缺 fold、重复 sample、未知 sample、错误 seed、候选顺序不一致、schema 不一致和 digest 不一致均明确失败；
- 合并后数组按 `sample_id` 稳定排序，所有字段与源 fold 一致；
- fold provenance 不进入 candidate/context/interaction 模型特征；
- schema-v6 token、mask、edge index 和 pooled features round-trip 不回退；
- 同一 fold helper 和标签生成同 seed 可复现；
- test lock 在配置冻结前持续生效；
- 现有 selector finalization、schema-v6、benefit-identifiability 及全量测试不回归。

## 10. 产物与解释口径

正式产物放入：

`代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/`

至少包含：

- `crossfit_protocol.json` 和五折 seed 清单；
- 五个 OOF fold cache、manifest、helper provenance 和复现命令；
- 合并后的正式 OOF train cache；
- final-helper calibration/validation cache；
- split/candidate/distribution/identifiability 审计；
- selector 对比、decision trace、seed/stage/family/coupling ablation；
- `summary.json`、Git SHA、配置 digest 和 SHA-256 manifest。

结果叙述严格分为：

- **观测事实**：OOF 与旧样本内标签的分布差异、validation 指标和安全指标；
- **合理解释**：helper 样本内候选导致 selector concept shift 的证据链；
- **待验证假设**：交叉拟合是否足以让 selector 达到 RMSE `<200`。

交叉拟合是必要的协议修复，但设计本身不保证指标突破。只有通过冻结 validation 门、一次性 matched test 和 external holdout 后，才能决定 selector 是否定型。
