# PI-JWM 两阶段目标对齐 Selector 设计

日期：2026-07-18  
状态：待实施  
方法身份：`v11 candidate`，不是 v12，也不是已定型主方法

## 1. 背景与问题定义

schema-v5 正式实验只使用 train、calibration 和 validation。validation ranked-allocation baseline 的 active-rate RMSE 为 `234.2529`，32 候选 sample oracle 为 `105.9892`，候选库质量门全部通过；当前 `CandidateSetBenefitRanker` 的 rank-only RMSE 为 `268.1722`，使用不确定性回退后 defer `99.974%`，最终 RMSE 仍为 `234.2529`。

当前损失主要对有效 sample 等权，而最终指标为：

\[
\operatorname{RMSE}=\sqrt{\frac{\sum_i \operatorname{SSE}_{i,c_i}}{\sum_i N_i}}.
\]

validation 中收益最大的 10% 有效样本贡献 `55.32%` 的 oracle 总 SSE 收益。现有目标没有显式强调这些高影响 sample，并且独立 rank score、predicted improvement 与 uncertainty 三个输出可能产生不一致决策。

本设计不修改冻结 PI-JWM world model、不扩充候选库、不重新生成 schema-v5 train/calibration/validation 标签。目标是学习与最终全局 SSE/RMSE 一致、可解释且可安全回退的 selector。

## 2. 方法选择

采用两阶段目标对齐结构：

1. `OpportunityHead` 判断当前 sample 是否存在足够大的可实现 SSE 收益。
2. `CandidateBenefitHead` 预测每个候选相对 ranked-allocation baseline 的 SSE benefit，并直接用该 benefit 排序。
3. ensemble 不确定性和 calibration 阈值共同决定执行或 defer。

保留两个对照：

- 现有等权 `CandidateSetBenefitRanker` 的冻结 validation 结果，不重复训练。
- 使用相同合法特征和 sample weight 的 XGBoost candidate-benefit baseline。若服务器现有环境没有 XGBoost，则明确记录 `skipped_dependency_unavailable`，不在正式运行中临时安装依赖，也不影响主方法验收。

不采用只修改 defer 阈值的方案，因为 `without_uncertainty` 已证明放宽回退会将 validation RMSE 恶化到 `241.2321`。不采用完整 offline RL，因为本轮已有固定离散候选和 actual-rollout 监督标签，扩大问题定义会削弱解释链。

## 3. 标签与目标对齐

对 sample \(i\) 和候选 \(c\)，定义：

\[
B_{i,c}=\operatorname{SSE}_{i,d}-\operatorname{SSE}_{i,c},
\]

其中 \(d\) 为 ranked-allocation baseline。定义 sample opportunity：

\[
G_i=\max\left(0,\max_{c\in\mathcal C_i} B_{i,c}\right).
\]

所有量仅由 train/calibration/validation actual-rollout 标签构造，绝不作为测试时输入。训练集正 opportunity 的中位数记为 \(S_G\)，用于稳定缩放；统计量只从 selector train seeds 计算并随 checkpoint 冻结。

sample impact weight 为：

\[
w_i=0.25+\min\left(\frac{G_i}{S_G}, w_{\max}\right).
\]

基础权重 `0.25` 保证模型仍学习低收益和有害候选；高收益样本获得更高权重；截断避免单个异常 sample 控制梯度。只搜索 `w_max∈{5,10}`。

以下情况必须显式处理：

- `active_count=0`：保留审计记录，但不进入 SSE 排序损失和 RMSE。
- sample 没有正 benefit：`G_i=0`，训练 opportunity 和 candidate benefit，跳过 listwise 排序项。
- 不适用或被 mask 的候选：不进入 softmax、回归或选择。
- 非有限值、负 SSE、候选维度错位：训练前直接失败，不做静默修补。

## 4. 模型结构与损失

新增 `OpportunityBenefitRanker`，复用现有候选 encoder、candidate-set pooling、stage/context encoder 和合法特征协议。

输出统一为：

- `predicted_candidate_benefit`：每个候选的归一化 SSE benefit；它同时是排序分数。
- `candidate_uncertainty`：每个候选的正值风险估计。
- `predicted_opportunity`：sample 最大可实现 benefit。
- `opportunity_uncertainty`：sample opportunity 风险估计。

不再设置与 benefit 无物理对应关系的独立 rank score。

固定训练目标：

\[
\mathcal L=
\mathcal L_{\text{weighted-listwise}}
+0.5\mathcal L_{\text{candidate-benefit}}
+0.5\mathcal L_{\text{opportunity}}
+0.10\mathcal L_{\text{worst-seed}}
+0.05\mathcal L_{\text{uncertainty}}.
\]

- weighted listwise 只对 `G_i>0` 的 sample 计算，使用 \(w_i\) 加权。
- candidate-benefit 使用所有合法候选的 weighted Huber loss。
- opportunity 使用所有 `active_count>0` sample 的 Huber loss。
- worst-seed 使用 selector train seed group，只约束训练稳健性。
- uncertainty 使用 benefit 残差的异方差 NLL，不能接触 validation outcome 调参。

固定 temperature 为 `0.25`、dropout 为 `0`、训练 200 epochs。只搜索：

- hidden dimension `{64,128}`；
- weight cap `{5,10}`；
- training seeds `{17,29,41}`。

共 12 个神经 checkpoint。validation 按 active-rate RMSE、worst-seed regret、训练 seed 标准差依次打破并列。

## 5. Calibration 与部署决策

三个训练 seed 组成 ensemble。候选选择只基于 ensemble mean predicted benefit；candidate LCB 定义为 ensemble mean 减去 `1.64 × total std`。

calibration seeds `44–49` 独立确定唯一 opportunity LCB 阈值：在预先固定的预测分位点集合 `{0, 0.25, 0.5, 0.75, 0.9}` 中选择 calibration RMSE 最低者，并以更高 defer 比例作为并列裁决。validation 和任何 locked split 都不得调整该阈值。

sample 仅在以下条件全部满足时执行候选：

1. opportunity LCB 高于 calibration 阈值；
2. 被选候选的 candidate-benefit LCB 大于 0；
3. 候选适用且真实改变动作张量；
4. 候选未被测试时可观测 task–energy proxy Pareto 规则支配。

否则 defer 到 ranked-allocation baseline，并记录单一、可机读的 defer reason。

## 6. 数据边界与泄漏防护

复用既有 schema-v5 缓存：

- train：seeds `0–15,20–43`；
- calibration：seeds `44–49`；
- validation：seeds `50–59`。

原 matched test seeds `18–19` 已在旧冻结配置中访问一次，本设计禁止再次使用。external seeds `60–69` 在 validation 达标并冻结配置前保持关闭。

测试时特征仍只能包含 current state、历史动作、候选动作摘要、PI-JWM forecast、stage、资源/任务/能耗代理和候选 mask。以下字段继续禁止：seed 身份、future truth、actual counterfactual outcome、oracle choice、真实 future task/energy。

所有 normalization、benefit scale、weight scale、模型参数、calibration 阈值和候选/特征顺序写入冻结 manifest，并绑定源码 Git SHA 与缓存 SHA-256。

## 7. 验收门与停止规则

进入服务器正式训练前必须通过：

- target 构造、权重截断、零 opportunity、mask 和 permutation invariance 单测；
- 64/256 sample 本地 CPU smoke；
- 无 future/oracle/seed 特征泄漏审计；
- 同 seed CPU 重跑一致；
- 现有全量测试和脚本测试。

validation 分三级：

- 成功：deployable active-rate RMSE `<200`，三个训练 seed RMSE 标准差 `≤5`，至少 7/10 validation seeds 优于各自 baseline，activity F1 下降不超过 `0.002`，link RMSE 相对恶化不超过 `2%`。
- 部分改善：RMSE `[200,230.8556)`；保留为 `diagnostic_only`，只允许一次针对已记录失败模式的设计复核，不打开 external。
- 失败：RMSE `≥230.8556`；停止当前结构，不通过放宽 defer 或增加网格继续追分。

只有成功级别才冻结唯一配置并生成 external seeds `60–69` 标签。external 只运行一次，至少 7/10 seeds 优于同口径 baseline 才能报告跨 seed 突破；由于原 matched test 已被访问，新结果必须标记为 `external_holdout`，不得冒充新的 matched-test A 级结果。

## 8. 解释产物

输出目录固定为：

`代码/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/`

必须包含：

- train-only benefit scale、sample weight 和 gain concentration 审计；
- opportunity calibration curve 与阈值选择记录；
- candidate predicted/actual benefit calibration；
- validation 逐 seed、stage、family 和 gain decile 结果；
- 每次选择的 candidate、预测 benefit、LCB、opportunity、defer reason；
- 与 ranked baseline、best fixed candidate、旧等权 ranker 和可用 XGBoost baseline 的比较；
- 去掉 opportunity、impact weight、uncertainty、stage/task/resource/energy 特征的消融；
- `summary.json`、完整复现命令、checkpoint SHA-256 和文件清单。

结论继续分为“观测事实、合理解释、待验证假设”，不把 sample oracle、calibration-best、diagnostic 或 external 结果混成同口径排行榜。

## 9. CPU/GPU 执行边界

本地 CPU 负责目标构造、开发、测试、64/256 sample smoke、审计和图表。已存在的 schema-v5 标签直接复用，不重新运行 world-model rollout。

RTX 4090 只负责 12 个神经 checkpoint、XGBoost 可用时的对照、validation 预测和固定消融，预计 1–2 小时。不重新训练 PI-JWM world model，不重复运行旧 RF/GB baseline。服务器运行必须在本地质量门全部通过后启动。
