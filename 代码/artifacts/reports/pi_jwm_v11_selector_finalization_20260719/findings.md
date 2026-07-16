# Findings

## 已知基线与上限

- 当前 13-template matched test sample-oracle 约为 `210.9754`，所以只重训旧 selector 不可能达到 `<200`。
- 当前 expanded55 selector matched test 约为 `235.42337`。
- 当前 deployable reference 为 `213.160874`。
- 将 predicted-active `rb_total` 替换为 truth 的诊断可达到 val/test 约 `119.52/103.58`，说明 magnitude repair 存在显著 headroom，但该结果不是 deployable。

## 上周物理诊断

- seeds 0–4、15 个决策组、66 个候选；H=3/10/20 均只有 7/15 非平凡组。
- CPU-scale 与 return-route 候选未产生可区分状态变化。
- 当前 perturbation 主要发生在第一步，后续 rollout 回到共享 default action。
- 能耗、reward 和 coupling 字段只能作为可解释特征、安全约束或训练审计；future simulator outcome 不可成为测试时输入。

## 当前工作区

- 分支：`main`；用户明确允许直接在当前工作区处理，不创建隔离 worktree。
- HEAD：`68cf68c`。
- 上周能耗/reward/horizon 的代码、测试、文档与图仍未提交，必须先验收并独立冻结。

## 口径复核

- `210.975399–210.975402` 是 13-template strict512 三组实验的 test sample-oracle；expanded55 的 test sample-oracle 是 `215.394402`，两者不得混称。
- `235.423373` 是 expanded55 按 validation 选型后的 matched-test 结果；其 deployable selector 结果为 `227.017662`。
- `213.160874` 是 ranked-allocation GPU gate 的较好信号，但更大 freeze 回到 `220.225805`；本轮把它作为突破参考线时，必须同时保留“不稳定历史信号”的说明。
- 上周正式报告目录和 H=3/10/20 horizon study 产物均存在，阶段 1 需要验证可重算性和测试状态后再提交代码/文档。

## 旧 selector 实现审计

- 旧 `compare_v11_rollout_reward_template_selector.py` 已能复用冻结 world model，产生 candidate-specific actual PI-JWM rollout SSE，并提供 RF/GB/HGB pairwise、error/improvement regressors 和 calibration；新工作应复用其 rollout 评估，不重写模型加载链。
- 旧 expanded55 本质是 3 个高度相似 score family × `top-k/alpha/cap/ecap` 网格；候选动作仍只改 `rb_total`，candidate-set 关系、stage 和物理耦合没有进入主 selector。
- expanded55 的 1024 个训练样本中 identity oracle-win 为 813（约 79.4%），超过新 gate 的 `<65%` 要求；这直接说明旧库监督覆盖失衡，不能靠更大 selector 解决。
- 旧 candidate feature 主要是 RB/activity/rate 的全局统计或逐步统计；没有明确的 split/test-lock 协议，也没有 permutation-invariant candidate-set encoder。

## 冻结数据事实

- world-model 数据集含 23,400 samples，60 seeds，每 seed 390 samples；主要 NPZ 已包含 `sample_seed`。
- frozen world model 原训练 seed 是 0–15、20–59，原 validation 是 16–17，原 matched test 是 18–19。
- 数据张量：history 8 steps、future 3 steps、314 edges、6 action dims；这与本轮 edge-step repair 的三步动作协议一致。
- 原始 selector full-label 路线通过现有 `evaluate_raw_actions` 逐候选运行 frozen PI-JWM；完整 15,600 × 32 候选会是主要算力成本，标签缓存和分块/续跑是必要接口。

## 新实现边界

- 候选库固定为 28 个：identity、ranked-allocation baseline、18 个 `K×magnitude×pattern` RB repair、6 个阶段耦合候选、2 个稳定历史 cap 控制；没有继续扩大旧 alpha/cap 网格。
- 所有 repair 保留 step 0，修改 step 1–2；persistent 两步保持目标，decayed 在第二个干预步衰减到 0.5。
- compute/return/offload 候选用 candidate mask 表示当前样本不可用；不可用动作回到 baseline，不把“未生效”伪装成有效候选。
- `CandidateSetBenefitRanker` 使用共享候选编码、masked DeepSets pooling、context/stage encoder，以及 score/improvement/uncertainty 三头；对候选重排保持输出等变。

## 64-sample candidate smoke

- 64 个均衡 validation 样本只有 11 个含 active target，因此绝对 RMSE 仅用于 smoke，不可与 full validation 的 200+ 数值直接比较。
- offload-RB low/high 是唯一显著改善的全局候选（约 48.30 → 44.49）；它把上周“offload–RB 耦合有实际任务效应”的诊断转化成了本周可用候选。
- RB value-head repair 多数与 identity 完全相同，表明当前支持集合/预测幅值没有真正改变 rollout；这与上周 CPU/return 不可区分的结论一致。
- sample-oracle 42.33 显示少数样本存在很大 headroom，但 identity 在 10/11 active 样本中仍为 oracle/tie；selector 监督仍高度稀疏，必须先看 256-sample 覆盖而不是开始深模型训练。

## 256-sample residual candidate gate

- 固定 4 个 benefit-guided residual 将 sample-oracle 从 102.07 降到 71.27，并把 identity oracle/tie 从 69.23% 降到 53.85%；说明“围绕受支持 baseline magnitude 做局部 expand/shrink”比重复预测 value-head 更有效。
- 新增 residual 的逐样本 oracle wins：expand25-k8 为 2、shrink25-k8 为 5、shrink50-k16 为 2；它们虽然不是最佳单一全局候选，却显著扩大 sample-specific headroom，正适合 selector。
- nontrivial 仍是 18/26=69.23%，只差一个有效样本达到 70%；由于 active 样本仅 26 个，不能继续按这 26 个样本微调候选，下一判断点必须是 full validation seeds 50–59。
- 当前已满足 oracle `<190`、identity `<65%`、action-applied `100%` 三项；唯一未确认项是 full-validation nontrivial coverage。

## 代码审查后的协议修正

- `candidate_mask` 原先把“不适用”和“适用但未生效”混在一起，会把 no-op 从动作生效率分母排除；schema-v4 现在独立保存 `action_applicable` 与 `action_applied`，正式 gate 对所有适用非 identity 候选要求 100% 生效。
- selector 原先训练 listwise `score`、部署却按 improvement head 排序；现改为按 ensemble listwise score 排名，再用 calibration-normalized improvement 的 90% 单侧下界决定执行，风险同时包含三模型方差和预测 aleatoric uncertainty。
- improvement/regret 由 raw SSE 差改为逐样本 active-rate RMSE 差，并按 train positive-regret 中位数归一化，保证温度网格具有实际分辨率。
- 冻结协议新增 `selector_freeze_digest`：绑定候选/特征顺序、train/calibration/validation cache SHA、唯一配置、checkpoint SHA、calibration bias、target scale、world/policy checkpoint SHA、源码 commit 和固定 defer/Pareto 规则。
- matched-test/external-holdout 访问写入 append-only ledger；同一冻结 selector 不能重复打开同一锁定 split。
- external 7/10 与 AirFogSim 安全门不得用 CLI 整数手填；必须由同 configuration/freeze digest 的逐 seed external summary 和逐行实际 task-energy audit 重算。
- 上周能耗候选与新 32 候选没有一一对应关系，旧 `candidate_summary.csv` 不能直接冒充本轮安全证据；新增严格 sample/seed/time/candidate 对齐工具，缺映射或缺物理行时直接拒绝生成审计。

## 首次 full validation GPU 证据

- RTX 4090 上完成 3,900 validation samples × 32 candidates，运行约 480 秒；matched test 未访问。
- sample-oracle active-rate RMSE `105.57994`，nontrivial ratio `83.7625%`，identity oracle/tie `24.6781%`，说明候选库上限和监督覆盖已经明显达到 selector 训练前提。
- 初次 action-applied ratio `99.9841%` 来自 105,300 个 applicable 非 identity 候选中的 25 个 no-op：20 个 q50 repair 的目标值恰等于 baseline，5 个 offload-low 同理。这些候选在构造阶段没有请求任何数值变化，不属于“适用但执行失败”。
- 修正后 applicability = stage/support 条件 ∧ raw candidate 相对 baseline 确有变化；若 raw candidate 请求变化但投影/执行后仍 no-op，继续按 action-applied 失败计数，因此没有降低 100% 质量标准。

## 正式冻结结果与 selector 根因

- 修正后的 full validation 为 oracle `105.19929`、nontrivial `83.0472%`、identity oracle/tie `24.8212%`、action-applied `100%`，candidate gate 正式通过。
- 12 个 listwise 配置和 RF/GB 对照在 validation 上未形成可部署增益：冻结配置 `h64_t0.1_d0` 的 defer ratio 为 `100%`，RMSE 等于 ranked baseline `233.94402`。冻结后的一次性 matched test 为 `223.88910`，同样 `100%` defer，结果等级为 `not_passed`；matched-test ledger 已落盘，不允许为新配置重复使用。
- matched-test sample oracle 为 `99.90324`，validation sample oracle 为 `105.19929`；候选库不是当前上限。固定模板中 validation 最好的是 historical cap105，仅到 `230.5828`；calibration 选出的 stage-wise 规则在 validation 为 `231.7815`，也不能替代样本级 selector。
- 数值诊断发现候选特征未标准化，最大尺度超过 `42,000`，三个冻结模型 calibration bias 约为 `936/3045/551`，uncertainty 达数千，所有 LCB 为负；但仅标准化后 rank-only 仍为 `297.60`，说明尺度问题不是唯一原因。
- 当前 `CandidateBatch.context` 实际只是 identity 候选的 46 维预测摘要，没有原计划中的 current state；候选特征又以全网统计为主，会把 8--32 条被修改边上的变化稀释。加入合法 current node/link/task 全局统计并训练 500 次更新后，rank-only 改善到 `259.05`，仍差于默认，支持下一轮必须增加 selected-edge 条件耦合特征，而不是继续扩大 epoch 或放宽 defer。
- 新 selector 只能继续使用 train/calibration/validation 开发；seeds 18--19 已经消费为本次冻结模型的一次性测试。后续结构冻结后，以尚未访问的 external seeds 60--69 作为独立验收，不得把旧 matched-test 缓存用于新模型选型或指标宣称。
