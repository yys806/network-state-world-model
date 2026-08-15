# PI-JWM R5 多 seed 统计与方法分析

日期：2026-08-06

## 1. 统计对象与结论边界

R5正式训练比较五个有限组合：A为Graph-GRU参考模型，B为Graph-RSSM，C为Graph-RSSM加异方差头，D为Graph-RSSM加显式DAG消息，E为Graph-RSSM加soft presence。每个组合使用训练seed `20260803/20260804/20260805`，统一采用validation选checkpoint、calibration独立评价、最多100 epoch、patience 10和有效batch size 32。

15/15个run完成，失败数为0，总训练时间`40287.76 s`；所有最佳checkpoint均可加载，最大checkpoint复现误差为`1.44e-8`，locked-test未访问。统计采用同seed配对：低值指标的正收益表示“参考值减候选值”，高值指标的正收益表示“候选值减参考值”。同时报告三seed均值、样本标准差、均值95% t区间、胜/负seed数和双侧精确符号翻转检验。

三组配对样本的双侧精确符号翻转检验最小只能达到`p=0.25`，因此本轮只能提供方向一致性和效应大小证据，不能宣称统计显著，也不自动指定winner。

## 2. Validation多seed结果

| 组合 | 综合分数↓ | 链路活动AUPRC↑ | 活动链路速率MAE↓ | 连续状态归一化误差↓ | 任务生命周期Macro-F1↑ |
|---|---:|---:|---:|---:|---:|
| A Graph-GRU | 4.4958±0.0367 | **0.02253±0.03409** | 449.312±3.576 | 0.68836±0.01210 | 0.06605±0.05985 |
| B Graph-RSSM | **4.4507±0.0360** | 0.002008±0.000151 | 445.612±2.351 | 0.67652±0.02665 | **0.12856±0.09115** |
| C RSSM＋异方差头 | 4.4508±0.0379 | 0.002021±0.000152 | **445.600±2.544** | 0.67669±0.02634 | 0.12776±0.09054 |
| D RSSM＋显式DAG | 4.5177±0.0611 | 0.002227±0.000730 | 452.471±5.479 | **0.66917±0.02102** | 0.08770±0.13604 |
| E RSSM＋soft presence | 4.4757±0.0622 | 0.002951±0.001778 | 446.134±5.866 | 0.68367±0.03472 | 0.05249±0.03109 |

表中的“±”为三seed样本标准差，不是置信区间。

## 3. 同seed配对结论

### 3.1 B/C相对A

- B的validation综合分数平均降低`0.04513`，即约`1.00%`，3/3个seed改善；calibration平均降低`0.04719`，3/3个seed改善。对应双侧精确`p=0.25`，95%区间跨0。
- C的validation综合分数平均降低`0.04499`，即约`1.00%`，3/3个seed改善；calibration平均降低`0.04735`，3/3个seed改善。对应双侧精确`p=0.25`，95%区间跨0。
- B和C的活动链路速率MAE均在3/3个seed优于A，平均改善约`0.82%`。B的该配对均值95%区间为`[0.297, 7.103]`，但三对样本的精确检验仍只能得到`p=0.25`。
- B/C的连续状态误差和任务生命周期Macro-F1只在2/3个seed改善，方差较大，不能写成稳定全面领先。
- 最关键的负结果是链路活动AUPRC：A均值为`0.02253`，B/C仅约`0.00201/0.00202`，B/C均有2/3个seed下降。Graph-RSSM的综合分数改善没有转化为稀疏链路活动识别改善。

### 3.2 C/D/E相对B的模块增量

- C相对B的validation综合变化为`-0.000135`，calibration变化为`+0.000166`，均接近0且只有1/3个seed改善；四项公开指标也没有稳定方向。当前证据不能说明异方差头改善点预测。
- D相对B的连续状态误差平均改善约`1.02%`，但活动链路速率MAE在0/3个seed改善，综合分数在0/3个seed改善并平均恶化`1.50%`，任务Macro-F1也下降。显式DAG消息只表现为定向状态特征收益，不适合作为整体替换。
- E相对B的综合分数平均恶化约`0.56%`，只有1/3个seed改善；速率、连续状态和任务指标均不稳定。soft presence当前不进入主路径。

## 4. 收敛与运行成本

| 组合 | 平均最佳epoch | 平均执行epoch | 跑满100 epoch | 最佳点位于100 epoch | 参数量 | 平均运行时间 |
|---|---:|---:|---:|---:|---:|---:|
| A | 17.3 | 27.3 | 0/3 | 0/3 | 43,414 | 21.5 min |
| B | 80.3 | 87.0 | 1/3 | 1/3 | 49,564 | 65.6 min |
| C | 74.3 | 81.0 | 1/3 | 1/3 | 50,222 | 61.9 min |
| D | 40.7 | 50.7 | 0/3 | 0/3 | 51,740 | 42.7 min |
| E | 34.0 | 40.7 | 1/3 | 1/3 | 49,564 | 32.1 min |

B/C各有一个run在第100 epoch取得最佳值，存在训练预算截断迹象。这不破坏“相同预算比较”的公平性，但说明当前证据不能支持“RSSM已经充分收敛”的结论。当前不立即追加GPU预算：应先确认链路活动退化和评价缺口是否来自目标权重、事件头或指标接口，再决定是否对所有保留候选统一扩展训练预算。

## 5. 当前方法判断

1. **保留B作为工作候选，保留A作为强制参考。** B在validation与calibration综合分数上均3/3改善，结构也比C/D/E简单，是进入下一轮诊断最合理的RSSM基线；但它还不是最终PI-JWM世界模型。
2. **C暂不并入主模型。** C与B的点预测结果基本等价，而当前R5报告没有直接给出NLL、区间覆盖率、区间宽度或共形校准结果，尚未真正评价异方差头的专属价值。
3. **D只保留为DAG定向消融。** 它稳定改善连续状态聚合误差，却稳定损害速率并降低整体分数，说明当前DAG消息可能需要任务相关目标或更精确的传播位置，而不是直接加入全局主干。
4. **E停止作为主路径候选。** soft presence没有稳定收益，后续只在动态拓扑误差被单独定位后再考虑。
5. **现在不能进入最终方法冻结。** 所有组合的链路活动AUPRC都很低，B/C还明显低于A；三seed置信区间普遍较宽，精确检验分辨率不足。

## 6. R5完整报告复评与评价缺口

已对15个正式run的validation和calibration报告重新统计，不再只看综合分数和四项公开指标。现有报告中保存的14项指标已经全部完成3-seed均值、离散程度和同seed配对分析，包括10项连续状态/任务误差、连续状态归一化误差、链路活动AUPRC、活动链路速率MAE和任务生命周期Macro-F1。完整结果位于`代码/artifacts/formal_training/pi_jwm_r5_multi_seed_analysis_v4/`。

对R2冻结的43项指标逐项审计后，覆盖情况为：14项可由现有R5报告直接比较，3项稀疏事件F1需要重新执行预测后处理，3项不确定性指标需要读取预测分布重新评价，22项系统/资源/安全/决策指标必须等策略器真实闭环执行，推理时延P95需要独立计时评估。以下内容仍未闭合：

- 不确定性NLL、95%覆盖率、区间宽度和校准误差；
- 分horizon的1/5/20步误差与误差增长率；
- 任务完成率、平均/P95/P99时延、吞吐量、能耗、RB/CPU利用率和公平性；
- 动作敏感性、结构合法性、任务/资源守恒和推理时延的组合级汇总。

因此当前状态应写为：`r5_gpu_training_complete=true`、`r5_saved_metric_analysis_complete=true`、`r5_method_freeze_ready=false`、`r6_ready=false`。时延、吞吐量、能耗和资源利用率不能由离线世界模型报告直接推出，必须留到策略器执行相同动作协议后的闭环对比。

## 7. 双图耦合、图编码器与旧架构复验计划

R4只对耦合和图编码器进行了单seed初筛，尚不足以冻结PI-JWM核心结构。下一轮采用Graph-RSSM组合B作为控制：F移除`CIP/CEP/CFL`跨图消息，G使用受`CIP/CEP/CFL`约束的Cross-Attention，H只把图编码器改为Edge-conditioned MPNN。另设J作为旧架构控制：保留旧`coupled_directed_dynamic_residual_v2`的有向动态、以上一观测为固定锚点的残差状态预测和预测拓扑思想，但输入重新绑定当前R1的物理图、信息图与`CIP/CEP/CFL`，输出使用当前R2接口；旧张量、旧图语义和旧checkpoint均不复用。J不是单模块消融，也不冒充论文baseline。

B的3个正式run已逐一核对原manifest条目、文件大小、checkpoint哈希、输入绑定、训练seed和冻结协议，确认可复用。F/G/H已在R1真实窗口上通过CPU前向与反向；J也已在当前`ExplicitStateBatch`上完成多步rollout、R2 objective与有限非零梯度。正式矩阵共15份同协议证据，其中B复用3份、F/G/H/J新增12份。运行器具备CUDA前置门、best/last checkpoint、输入指纹绑定、断点续跑、失败保留、远程状态文件及下载后逐文件验哈希。冻结矩阵位于`代码/artifacts/preflight/pi_jwm_r5_module_confirmation_v2/`；这些结果仍只证明接口与实验协议就绪，不代表任一方法优越。

## 8. 可复现产物

- 正式训练：`代码/artifacts/formal_training/pi_jwm_r5_gpu_training_v1/`
- 完整14项指标统计及43项覆盖审计：`代码/artifacts/formal_training/pi_jwm_r5_multi_seed_analysis_v4/`
- 双图耦合/图编码/旧架构复验矩阵：`代码/artifacts/preflight/pi_jwm_r5_module_confirmation_v2/`
- R5.1正式训练入口：`代码/scripts/run_r5_module_confirmation_training.py`
- R5.1远程续跑与状态入口：`代码/scripts/run_r5_module_confirmation_remote.sh`
- 下载结果验哈希入口：`代码/scripts/verify_r5_module_confirmation_bundle.py`
- 早期五项统计包：`代码/artifacts/formal_training/pi_jwm_r5_multi_seed_analysis_v2/`
- 初始统计包：`代码/artifacts/formal_training/pi_jwm_r5_multi_seed_analysis_v1/`，仅缺少相对B的模块增量表，已由v2取代
- 统计入口：`代码/scripts/analyze_r5_multi_seed.py`
- 统计模块：`代码/src/pi_jwm/r5_analysis.py`
- 定向测试：`代码/tests/test_r5_multi_seed_analysis.py`与`代码/tests/test_analyze_r5_multi_seed.py`
