# AirFogSim稀疏事件诊断v2设计

日期：2026-08-01

## 1. 目标与边界

本阶段回答一个具体问题：当前PI-JWM最小rollout模型对链路活动、信息流存在和任务演化的预测，是否超过不学习的简单规则；类别不平衡处理是否带来可核对的改善。

本阶段仍使用`airfogsim_tensor_v2_dev`的三个开发split：seed 0为`dev_train`，seed 1为`dev_validation`，seed 2为`dev_calibration`。不建立或打开锁定测试集，不接入JEPA，不进行正式模型选择，不把开发结果表述为泛化收益。

## 2. 四个对照臂

| 实验臂 | 定义 | 作用 |
|---|---|---|
| `zero_activity` | 未来链路活动、速率、RB量、信息流存在和任务存在均预测为0；节点与物理边的其他连续状态复制历史末时隙；任务生命周期预测为`dev_train`中的多数类 | 检查类别不平衡下“全预测负类”能达到什么表面结果 |
| `last_persistence` | 将历史最后一个时隙的节点、物理边、信息流、任务状态、存在性和任务生命周期复制到未来3步 | 检查学习模型是否超过最简单的时序持续性规则 |
| `learned_unweighted` | 当前action-conditioned双图rollout模型，新增链路活动和任务生命周期输出，但不使用正类加权 | 作为同结构、同训练轮数的公平对照 |
| `learned_balanced` | 与`learned_unweighted`完全相同，只增加由`dev_train`统计得到的正类权重 | 单独判断类别不平衡处理的作用 |

两个学习臂使用相同初始化seed、batch size、学习率、5个epoch和数据顺序。任何阈值均固定为0.5，不根据seed 1或seed 2结果调节。

`zero_activity`对目标时隙真实存在的信息流和任务使用全零连续状态计算MAE，不能通过“预测不存在”逃避状态误差；`last_persistence`只有在历史末时隙对象存在时才复制其状态，否则保持全零。两个基线的节点和物理边有效槽位均沿用历史末时隙的拓扑存在性。

## 3. 模型补充

在现有四类状态与存在性输出之外增加两项：

1. `link_activity_logits`：对每条物理边未来每个时隙预测是否存在实际业务承载，标签为真实`active_task_count>0`，只在该物理边当时可观测时计算。
2. `task_lifecycle_logits`：对每个真实任务预测`to_offload/computing/returning/finished/failed`五类生命周期，只在任务已到达、存在且生命周期标签有效时计算。

加权臂的正类权重只由`dev_train`的有效窗口统计，定义为负样本数除以正样本数，并限制在$[1,50]$，防止极稀疏标签产生不稳定梯度。链路活动、信息流存在和任务存在使用各自独立权重；节点和物理边拓扑存在性不加权。

总损失固定为

\[
L=L_{state}+0.1L_{presence}+0.1L_{activity}+0.1L_{lifecycle}.
\]

四项损失分别先在自己的有效mask内求平均，不按槽位数量直接相加，避免306条物理边压倒其他对象。

## 4. 评价口径

每个实验臂在seed 1和seed 2分别报告：

- 链路活动precision、recall、F1和AUPRC；
- 真实活动边上的速率MAE、RMSE和样本数；
- 信息流存在F1与任务存在F1；
- 任务生命周期accuracy和macro-F1，并报告每类support；
- 节点、物理边、信息流和任务连续状态的反归一化字段级MAE。

零活动基线的active-only速率误差按预测速率0直接计算，不能因它没有预测活动正类而跳过。任务生命周期在任务不存在或标签为$-1$时不参与评价。
对常数分数或没有预测正类的情况仍按完整precision-recall排序计算AUPRC；评价集中确实没有真实正样本时才标记为`not_applicable`。

## 5. 产物与验收

代码继续放在`代码/src/pi_jwm/`和`代码/scripts/`，新产物写入`代码/artifacts/small_experiments/exp08_airfogsim_sparse_event_diagnostic_v2/`，至少包含配置、训练集类别统计、逐臂逐split指标、训练历史、模型权重、Markdown报告和SHA-256清单。

验收条件：

1. 四个实验臂使用相同窗口和评价代码，seed之间不泄漏；
2. 单元测试覆盖权重只取训练split、padding不参与损失、生命周期无效标签被忽略、简单基线定义和AUPRC计算；
3. 所有损失与指标为有限值或有明确的`not_applicable`原因；
4. 报告直接给出学习模型是否超过`zero_activity`与`last_persistence`，失败也保留为结果；
5. 本阶段只产生go/no-go诊断：基础模型未稳定超过简单基线时，不进入耦合JEPA结构比较。
