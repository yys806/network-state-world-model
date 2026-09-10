# PI-JWM 用户学习路径

这份路径的目的不是让用户一次记住全部历史，而是按“问题—直觉—实现—实验—边界”逐层重新掌握项目。

每一层都可以直接向 AI 提问。AI 应按 `COLLABORATION_GUIDE.md` 先解释问题和直觉，再连接数学、代码与实验；用户不需要先知道精确文件名。

## 第一层：先掌握我们在研究什么

先读 `RESEARCH_STATUS.md` 的研究问题和方法边界，再读 `ARCHITECTURE.md` 的第 1–3 节。

要能用自己的话回答：

1. 为什么物理网络和信息网络需要分开表示？
2. 为什么模型不仅要看过去状态，还要看未来候选动作？
3. 为什么当前 P4 只验收世界模型，尚不能宣称规划器已经完成？

## 第二层：掌握数据如何变成模型输入

阅读顺序：

1. `code/src/pi_jwm/full_dual_graph_collector_contract_v1.py`：仿真器中的决策、执行和结果如何被审计。
2. `code/src/pi_jwm/formal_airfogsim_window_v1.py`：8 步历史、20 步预测和 split 如何组织。
3. `code/src/pi_jwm/formal_motion_state_v1.py`：速度和加速度为什么只能从历史位置向后计算。
4. `code/src/pi_jwm/formal_rb_targets_v1.py`：RB 动作和真实传输事件为什么必须逐项匹配。

掌握标准：能够解释“空值 + mask”和“真实的零”为什么不同，以及未来目标为什么不能进入 prior 预测。

## 第三层：掌握当前模型

阅读顺序：

1. `formal_dual_graph_world_model_v1.py`：双图确定性底座如何编码实体和关系。
2. `formal_deterministic_rule_layer_v1.py`：哪些状态由守恒、生命周期和 DAG 规则更新。
3. `formal_entity_aligned_rssm_world_model_v1.py`：节点、物理边、数据流和任务为什么各自维护隐状态。
4. `formal_world_model_loss_v1.py`：训练目标如何组合。
5. `formal_p4_gate_v1.py`：模型最终必须通过哪些门。

先理解直觉，再看公式和代码：确定性底座负责可解释的关系传播；RSSM 负责不确定的时间变化；规则层防止预测违反容量和任务生命周期。

## 第四层：掌握实验和结果

先在 `EXPERIMENT_INDEX.md` 或 `registries/experiment_registry.json` 找到实验，再到 `RESULTS_INDEX.md` 或 `registries/results_registry.json` 找数字，最后打开对应 artifact 验证。

当前必须能区分：

- 机制测试通过：说明接口和因果语义成立；
- 单 seed 数值门通过：说明一个固定随机种子满足门槛；
- 三 seed 审计通过：当前尚未完成；
- `locked_test`：当前没有访问，不能引用任何锁定测试结果。

## 第五层：参与下一次科研决策

当第三个 seed 完成并且 P4 审计通过后，才进入 P6。届时需要由用户决定：合法候选如何生成、任务代价和风险如何定义、纯搜索/学习策略/混合策略如何公平比较。

用户不需要亲自完成所有工程细节，但应能对每个候选回答：它解决什么问题、前提是什么、哪个实验能证伪、结果支持到哪一层。

追查旧方案时不要从目录名猜结论。先查询 `registries/historical_method_registry.json`，再打开其中列出的实验和 audit。
