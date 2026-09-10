# `pi_jwm` 源码导航

当前正式 P4 核心先看：

1. `formal_dual_graph_world_model_v1.py`
2. `formal_entity_aligned_rssm_world_model_v1.py`
3. `formal_deterministic_rule_layer_v1.py`
4. `formal_motion_state_v1.py`
5. `formal_world_model_loss_v1.py`
6. `formal_world_model_metrics_v1.py`
7. `formal_p4_gate_v1.py`

数据合同和图构建模块属于支撑层；candidate rollout planner 属于 P6 原型；`r3_*` 至 `r6_*`、`v6_*` 至 `v11_*` 属于历史兼容层。完整状态与反向引用见 `docs/CODE_INDEX.md` 和 `docs/registries/generated/python_dependency_map.json`。

历史文件暂不物理移动，因为仍存在复现实验和测试 import。保留路径不代表当前采用，新增正式模型代码仍必须直接放在本目录，并同步更新代码和项目索引。
