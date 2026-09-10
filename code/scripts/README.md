# PI-JWM 脚本导航

## 当前正式 P4

- `build_formal_causal_motion_tensor_v1.py`：构建因果运动正式 tensor。
- `freeze_formal_p4_entity_rssm_protocol_v1.py`：冻结实体级 RSSM 协议。
- `run_formal_entity_aligned_rssm_consistency_audit_v1.py`：CPU 一致性审计。
- `run_formal_p4_entity_rssm_gpu_batch_probe_v1.py`：GPU batch 探测。
- `run_formal_p4_entity_rssm_gpu_v1.py`：正式 seed 守卫与入口。
- `run_formal_dual_graph_gpu_train_v1.py`：底层两阶段训练器。

## 项目维护

- `build_project_knowledge_index_v1.py`：重建文件、依赖和 artifact 注册表；使用 `--check` 可只读检查是否漂移。
- `query_project_knowledge_v1.py`：先从人工注册表检索当前方法、实验、结果、历史方案和延期任务，再给出需要打开的原始证据路径。中文查询时先设置 `PYTHONUTF8=1`。

## 延后任务

第三个 seed 的现有入口、冻结 protocol、前序证据和允许变化字段记录在 `docs/registries/deferred_work.json`。该文件明确 `auto_start=false`；文档中的接口不是运行授权。

其余 `r*`、`v*`、分析、诊断和绘图脚本应先通过 `docs/CODE_INDEX.md` 确认状态。脚本能启动不等于当前方法或结果已通过。
