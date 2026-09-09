# P4 link activity 持久性残差状态转移实施计划

> 这是已确认候选方法的单变量实施计划。执行者必须按顺序完成，每一门失败即停止，不追加第二种方法。

## 目标与范围

只新增`link_activity_persistence_residual_v1`：保留现有edge latent和单路link head，让head输出变化量，在内部递推未加权事件logit，再加固定train-only `pos_weight`形成既有`link_activity_logits`。不修改数据、双图、GRU、消息、loss、阈值协议或其他heads。

## Task 1：接口规格与测试RED

文件：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`、`code/scripts/run_formal_dual_graph_gpu_train_v1.py`、对应两个测试文件。

先新增失败测试，覆盖：有效历史边的零delta保持已有`+20/-20`加权raw persistence；h2读取上一预测而非target；未来target和target mask不影响输出；`z=u+log(pos_weight)`；缺失history使用预先绑定的train-only prior；旧方法身份不接受旧checkpoint续训；既有默认模型输出不变。运行定向测试，确认因缺少新配置/方法失败。

## Task 2：最小模型实现

在`FormalWorldModelConfig`加入显式方法身份、link persistence开关、pos_weight、有限persistence logit和缺失历史先验字段。forward只在该开关开启时维护`u_prev`，h1从history最后活动和mask初始化，随后递推`u_prev + delta`，输出`u + log(pos_weight)`。未来target不参与forward。默认开关关闭，保持旧方法行为。

## Task 3：训练runner与版本证据

增加唯一learned method spec，构建时传入冻结配置计算出的train-only活动先验和pos_weight；在config、method registry、checkpoint metadata、summary中写明方法身份和递推语义。旧checkpoint严格拒绝语义续训/正式评价。runner仍只从train split计算class weight，方案B仍使用原始raw logits。

## Task 4：CPU契约和回归

运行新方法测试、正式模型测试、loss/metrics/calibration/runner测试和compileall。执行一个极小CPU micro-smoke，仅验证有限输出、保存、严格重载、manifest和target隔离，不做完整训练。Sol检查diff与关键字段，任何漂移停止。

## Task 5：Go/No-Go与GPU边界

CPU所有门通过后，Sol审查唯一修改和精确配置；向用户报告后才开启GPU。只跑seed `20260831`。先检查性能门，再检查方案B概率门。性能或概率任一失败立即保留证据并停止。

## Task 6：后续seed与最终审计

只有sentinel同时通过性能门和概率门，才按原协议依次运行`20260830`、`20260832`。任一seed失败停止。三seed全过才由Sol运行统一审计；否则P4继续blocked。

## 当前状态

Task 1--4已完成：RED有效，最小实现、runner/checkpoint证据、CPU定向回归和真实canonical tensor micro-smoke均通过。P4仍blocked，follow-up seeds不开放，`locked_test_accessed=false`。单一下一动作是用户确认后仅运行GPU sentinel seed `20260831`，先过性能门再过方案B概率门。
