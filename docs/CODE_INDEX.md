# PI-JWM 代码状态索引

> 本页区分“当前正式代码、支撑代码、原型和历史兼容代码”。精确依赖以 `registries/generated/python_dependency_map.json` 为准。

## 当前正式 P4 核心

| 模块 | 职责 |
| --- | --- |
| `formal_dual_graph_world_model_v1.py` | 物理图、信息图、任务和跨图关系的确定性编码与递推 |
| `formal_entity_aligned_rssm_world_model_v1.py` | 节点、物理边、数据流和任务的实体级 prior/posterior 动力学 |
| `formal_deterministic_rule_layer_v1.py` | 容量、服务量、剩余量、生命周期和 DAG 的规则更新 |
| `formal_motion_state_v1.py` | 只用历史位置派生因果运动状态 |
| `formal_rb_targets_v1.py` | RB 动作与运行时传输结果的严格对齐 |
| `formal_world_model_loss_v1.py` | 正式训练损失 |
| `formal_world_model_metrics_v1.py` | 正式状态、链路和运营指标 |
| `formal_world_model_baselines_v1.py` | persistence 等比较基线 |
| `formal_p4_gate_v1.py` | 单 seed P4 数值门 |

正式运行入口是 `code/scripts/run_formal_p4_entity_rssm_gpu_v1.py`，底层训练器是 `run_formal_dual_graph_gpu_train_v1.py`。第三个 seed 仍使用同一入口，但已在 `registries/deferred_work.json` 中锁定为必须重新获得用户授权，不能自动启动。

## 当前支撑代码

数据采集、正式 window、双图合同、图操作、审计和序列化模块仍被当前入口依赖。它们不是“当前方法名称”，但不能因为没有出现在上表就移动。依赖图中 `imported_by` 或 `test_files` 非空的文件，归档前必须逐项处理引用。

项目知识维护使用 `code/scripts/build_project_knowledge_index_v1.py`；常见问题只读检索使用 `code/scripts/query_project_knowledge_v1.py`。两者是工程治理工具，不是研究方法或实验结果。

## P6 原型

- `formal_candidate_rollout_planner_v1.py`
- `formal_candidate_rollout_planner_audit_v1.py`
- `run_formal_candidate_rollout_planner_audit_v1.py`

这些文件只证明“逐候选调用世界模型”的机制骨架存在。合法候选生成、正式风险/代价、当前 RSSM 闭环和真实执行反馈尚未冻结，因此不能标为当前正式策略器。

## 历史兼容代码

`r3_*`、`r4_*`、`r5_*`、`r6_*`、`v6_*` 至 `v11_*` 主要属于历史候选、诊断或旧策略路径。它们暂时保留在原路径，是因为部分脚本、测试和证据复现仍引用这些 import；保留不代表继续采用。

当前采用“逻辑归档、物理保留”：AI 默认不读历史模块，只有追查旧实验或失败原因时才通过索引进入。物理迁移必须先通过 `ARCHIVE_CANDIDATES.md` 中的门。
