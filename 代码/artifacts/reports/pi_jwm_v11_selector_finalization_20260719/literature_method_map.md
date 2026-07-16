# PI-JWM v11 Selector 文献方法矩阵

更新时间：2026-07-16
范围：候选集合排序、收益/风险选择、不确定性回退、跨 seed 稳健性。以下只用于 v11 selector 定型；PI-JWM 世界模型保持冻结。

| 一手论文 | 核心问题与方法 | 对本项目的数据要求 | 可解释性 | 本轮决策 |
|---|---|---|---|---|
| [Mandi et al., 2022, Decision-Focused Learning Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) | 将决策质量写成 pointwise、pairwise、listwise 排序损失，直接关注解的 regret | 每个 sample 需要多个可行候选及 actual-rollout regret | 可按候选 regret、排序概率解释 | **采用**：listwise 主损失 + pairwise 辅助损失 |
| [Kamran et al., 2024, Learning to Rank for Optimal Treatment Allocation](https://proceedings.mlr.press/v238/kamran24a.html) | 资源受限时不必精确回归绝对收益，可直接学习收益排序 | 候选的配对增益标签和资源占用 | 可解释为单位资源下的收益排序 | **采用思想**：candidate benefit/risk 排序；不照搬其因果树模型 |
| [Zaheer et al., 2017, Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | 对集合输入构造 permutation-invariant 表示 | 每个样本对应可变语义但固定上限的候选集合 | 结构约束明确，可做候选置换测试 | **采用**：共享 candidate encoder + masked set pooling |
| [Lee et al., 2019, Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | 用注意力显式建模集合元素间交互，同时保持排列不变性 | 候选集合及 mask | 可查看候选间注意力，但额外复杂 | **部分采用/暂缓 attention**：本轮 32 候选先用 DeepSets；若验证显示集合交互不足再升级 |
| [Mozannar and Sontag, 2020, Consistent Estimators for Learning to Defer](https://proceedings.mlr.press/v119/mozannar20b.html) | 将 defer 作为带成本的决策而不是强制输出 | calibration split 和稳定默认动作 | defer 原因与覆盖率可直接报告 | **采用原则**：不确定时回退 ranked-allocation baseline；不声称复现其一致性定理 |
| [Yang et al., 2022, Offline Policy Selection under Uncertainty](https://proceedings.mlr.press/v151/yang22a.html) | 用候选价值的不确定性分布做离线策略选择，而不只看点估计 | 多训练 seed/ensemble 的候选收益预测 | 可报告均值、方差、LCB 和 defer | **采用简化版**：3 模型 ensemble，执行条件为 mean−1.64·std>0 |
| [Yu et al., 2024, Empirical Group DRO](https://proceedings.mlr.press/v235/yu24a.html) | 最小化群组最大经验风险，防止平均结果掩盖弱组 | 明确的 seed/stage group | 能按最差 seed/stage 报告 | **采用简化版**：固定 0.10 worst-group loss；不实现完整双层 ALEG 优化器 |
| [Geifman and El-Yaniv, 2019, SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) | 联合训练预测与拒绝，优化 risk–coverage 权衡 | 明确 coverage 目标和足够拒绝样本 | risk–coverage 曲线清楚 | **作为对照依据**：本轮 calibration 后置定阈值，避免把覆盖率目标再加入超参搜索 |
| [Qiao and Valiant, 2019, A Theory of Selective Prediction](https://proceedings.mlr.press/v99/qiao19a.html) | 分析选择性预测在在线/分布条件下的可行边界 | 需要与其理论设置相符的数据序列 | 理论边界强，但与当前离线候选任务不完全同构 | **不直接采用**：仅用于约束“defer 不是免费提升”的表述 |
| [Blondel et al., 2020, Fast Differentiable Sorting and Ranking](https://proceedings.mlr.press/v119/blondel20a.html) | 用 permutahedron 投影实现可微排序 | 连续排序目标 | 排名算子清楚，但引入额外数值层 | **本轮不采用**：32 候选的 softmax listwise 已足够，避免无收益的复杂化 |
| [Ma et al., 2021, Plackett–Luce Listwise LTR](https://proceedings.mlr.press/v130/ma21a.html) | 针对部分偏好标签高效计算 listwise likelihood | 只知道分区/部分排序时有优势 | 概率排序可解释 | **本轮不采用**：当前 actual rollout 给出完整 regret，不存在部分偏好缺失 |
| [Devic et al., 2024, Stability and Multigroup Fairness in Ranking with Uncertain Predictions](https://proceedings.mlr.press/v235/devic24a.html) | 研究不确定预测下排名稳定性和多群组性质 | 群组定义与预测不确定性 | 强调稳定性而非单次最优 | **采用评价思想**：报告逐 seed 方向一致率、worst-seed regret，不引入公平约束 |
| [Liu et al., 2023, Constrained Decision Transformer](https://proceedings.mlr.press/v202/liu23m.html) | 在离线轨迹上联合建模 reward–cost 约束并支持阈值条件化 | 大规模完整轨迹、return/cost 条件和重新训练策略 | reward/cost 权衡明确，但归因链更长 | **本轮排除**：会同时改变策略与 selector，无法把提升归因到候选/选择器 |

## 方法收敛结论

1. 本轮主问题是“在一个 sample 的候选集合中，选择 actual-rollout regret 最小且风险可接受的动作”，因此 listwise decision-focused ranking 比普通最佳候选分类更匹配。
2. 候选顺序不应改变输出，主结构先采用更稳、更容易验证的 DeepSets；Set Transformer attention 不是当前必要复杂度。
3. defer 必须只由 calibration 和 ensemble 不确定性决定；validation 只选结构，matched test 只打开一次。
4. 上周的 task、resource、energy 和 stage 耦合指标分别进入可解释特征、候选约束、Pareto 审计与 ablation；AirFogSim actual outcome 不进入测试时特征。
5. 完整 offline RL/Decision Transformer 会改变研究问题和归因口径，本轮明确不做。
