# P4 link低召回edge GRU单次因果核验设计

状态：已实施；真实结果为`not_sufficient_to_explain_dominant_low_recall`，路线停止
日期：2026-09-04
当前主线：P4 blocked

## 1. 目的

本设计只回答一个问题：

> 当前checkpoint为什么会大量漏报上一时刻已经活跃、下一时刻仍然活跃的物理链路；edge GRU更新是否足以解释这一主要低召回现象？

本设计不是模型修复，不训练、不保存新checkpoint、不改变正式模型定义，也不把干预结果当作性能结果。

## 2. 已冻结的诊断证据

输入证据为：

- checkpoint：`code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/checkpoints/coupled_dual_gnn_residual__best.pt`；
- checkpoint SHA-256：`0d203fa46b89e3b4267f387371e4a8c38682810482d30af7c8058f2ea2ac141f`；
- canonical tensor manifest SHA-256：`d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`；
- validation sample IDs SHA-256：`1f01dade9a0a673fd95bb3d5953a00617c09f124cdc019673765983ea7cde582`；
- raw threshold：`0.9`；
- train-only link `pos_weight`：`50.0`；
- 原始candidate TP/FP/FN：`1902/557/5855`；
- candidate/persistence F1：`0.3723570869/0.5900903873`；
- 诊断报告：`code/artifacts/audit/pi_jwm_p4_link_recall_diagnosis_20260904/link_recall_diagnosis.json`；
- 诊断报告 SHA-256：`ee03931c27ddccf4b8967247db3f06dcd3db7f2efe546917ca5e6d7bf1556e4a`。

诊断中的关键事实：

- 7,757个真实正样本中，6,050个是持续活跃链路，1,707个是新激活链路；
- 持续活跃链路的candidate/persistence recall为`0.2648/0.7205`；
- persistence正确而candidate漏报的2,806个样本中，2,791个属于持续活跃链路；
- candidate overall recall在h1、h7、h20分别为`0.0000/0.5778/0.0106`，呈现“中间上升、两端塌陷”；
- h1还没有上一预测步产生的rule feedback，因此rule feedback不能单独解释h1召回为零；
- 20步使用同一个link activity head，明显的逐步形状必须来自送入head的edge latent轨迹，而不是一个静态head参数独立产生。

## 3. 唯一被怀疑的机制

唯一被核验的模块是：

```text
FormalDualGraphWorldModel.edge_transition
```

它是物理边latent的GRU更新单元。现有前向顺序是：历史编码得到edge latent，接收物理图消息、信息流耦合消息和已有规则反馈消息，经`edge_transition`更新后，再由同一个`link_activity_head`输出logit。

当前证据支持优先检查它，因为低召回主要是“已有活跃信息丢失”，且h1在第一次edge更新之后立即变为零召回。当前证据不允许直接断言GRU参数已经被证明有错。

## 4. 单次只读干预

只在CPU推理期间给`edge_transition`注册一个临时forward hook，使它每次返回调用前的hidden state：

```text
intervened_edge_state = previous_edge_hidden_state
```

等价含义是：在这次反事实推理中旁路edge GRU更新，检查保留历史编码得到的物理边记忆后，主要漏报是否被找回。

以下内容全部保持不变：

- checkpoint及全部参数；
- physical graph、information graph和两图对象定义；
- node、flow、task和agent更新；
- physical message、CFE耦合消息和rule feedback的计算；
- link activity head；
- dataset、tensor、normalization、sample IDs和validation顺序；
- physical-edge有效性与`aggregate_link_activity_mask`；
- raw threshold=`0.9`及float32 legacy decision语义；
- persistence基线；
- calibration和validation隔离；
- `locked_test_accessed=false`。

消息仍会被原模型计算，但只阻断这些消息通过`edge_transition`改变edge hidden state的结果；不得同时清零任何消息、关闭规则层或改变其他GRU。

## 5. 输出

新增一个独立CPU-only intervention runner和对应测试，输出到新的、不允许覆盖的目录：

```text
code/artifacts/audit/pi_jwm_p4_edge_gru_bypass_intervention_20260904
```

报告至少包含：

- 原始路径与干预路径的overall TP/FP/FN、precision、recall、F1；
- 两条路径的持续活跃、新激活、previous-unobserved分组；
- h1至h20完整结果，以及h1/h5/h10/h20重点汇总；
- persistence正确但candidate漏报集合中，被干预找回的数量；
- 该集合在干预前后的raw logit分位数；
- checkpoint、class weights、tensor manifest、sample IDs、mask counts的路径和SHA-256；
- strict checkpoint reload结果；
- hook只命中`edge_transition`且命中次数与20步rollout一致的证明；
- `gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`；
- 独立manifest，使用`size_bytes`和SHA-256。

报告不得覆盖原始recall诊断、threshold replay、概率校准失败记录或sentinel产物。

## 6. 预注册判定

只有以下条件全部满足，才记录：

```text
edge_gru_transition_intervention = sufficient_to_explain_dominant_low_recall
```

1. 在原始2,791个“persistence正确、candidate漏报”的持续活跃样本中，至少找回1,396个；
2. 持续活跃链路FN相对原始4,448至少减少25%，即干预后不高于3,336；
3. h1持续活跃recall相对原始`0.0000`至少提高`0.20`；
4. h20持续活跃recall相对原始`0.0000`至少提高`0.20`；
5. overall F1相对原始`0.3723570869`至少提高`0.10`，即不低于`0.4723570869`；
6. overall FP不超过原始557的两倍，即不高于1,114；
7. 原始路径必须在同一次runner中精确复现TP/FP/FN=`1902/557/5855`，否则判为输入或实现漂移并停止。

任一条件不满足，统一记录：

```text
edge_gru_transition_intervention = not_sufficient_to_explain_dominant_low_recall
```

不得因为“部分指标看起来有改善”而放宽门槛。

## 7. 如何解释结果

若判定为`sufficient_to_explain_dominant_low_recall`，只说明：旁路edge GRU更新足以找回主要漏报，下一步可以设计一个只修改edge GRU信息保持方式的最小训练修复。它不证明永久旁路GRU是正确修复，也不允许直接把干预结果当作正式性能。

若判定为`not_sufficient_to_explain_dominant_low_recall`，P4继续blocked并停止当前路线；不得自动尝试CFE置零、rule feedback关闭、换loss、换阈值或第二个模型模块。Sol形成不支持说明后，必须获得用户确认才能另立不同干预。

## 8. TDD与审查

实施顺序固定为：

1. Sol冻结本设计和精确接口；
2. Terra只实现已经写死的hook、统计、provenance和runner；
3. 测试先在无实现状态准确RED；
4. 测试覆盖hook仅影响edge transition、命中20次、原始路径不变、float32阈值边界、分组和逐步统计、严格复载、实际tensor哈希、原子发布和失败清理；
5. Terra完成GREEN后，由Sol逐项规格审查；
6. fresh Sol或主Sol做代码质量审查；
7. 主Sol独立复跑定向回归；
8. 只有以上全部通过，才运行一次真实CPU intervention；
9. 主Sol读取原始JSON并按第6节机械判定。

## 9. 边界

- 当前不需要GPU；
- 不运行任何训练；
- 不启动seeds `20260830/20260832`；
- 不拟合温度、不选择阈值、不改变方案B；
- 不进入P6；
- 不访问`locked_test`；
- P4在本次干预前后都保持`blocked`，除非未来新checkpoint按完整双门和三seed协议通过；
- 本设计文档不执行Git提交，原因是当前工作树包含用户既有未提交P4变更，必须原样保留。
