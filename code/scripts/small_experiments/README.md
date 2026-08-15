# PI-JWM训练前小实验

本目录只保存训练前的小型验证脚本，不保存PI-JWM主线模型代码，也不保存生成结果。

当前实验：

| 编号 | 脚本 | 目的 | 是否需要GPU |
| --- | --- | --- | --- |
| 00 | `strict_dual_graph_validity.py` | 验证非空DAG、MN/ME/EP跨图关系、路径与流守恒以及规则级动作敏感性 | 否 |
| 01 | `data_contract_field_audit.py` | 审计现有AirFogSim数据、严格动作、NPZ与候选真实数据是否满足PI-JWM v1字段契约 | 否 |
| 02 | `airfogsim_strict_dual_graph_preflight.py` | 从实际AirFogSim运行导出非空DAG、MN/ME/EP证据并检查严格双图是否就绪 | 否 |
| 03 | `airfogsim_cross_graph_evidence_closure.py` | 监听实际AirFogSim逐任务信道事件，并按PI-JWM共享父输出契约闭合ME/EP证据 | 否 |
| 04 | `task_resource_conservation_audit.py` | 审计任务/依赖/RB/CPU/UAV能量守恒与指标可计算性 | 否 |
| 05 | `paired_action_causal_sensitivity.py` | 用同seed完整重放验证卸载目标和RB集合的真实后继效应 | 否 |

运行方式：

```powershell
cd D:\shen\PKU\PIJWM\code\scripts\small_experiments
python strict_dual_graph_validity.py
python data_contract_field_audit.py
conda run -n airfogsim python airfogsim_strict_dual_graph_preflight.py --seed 0 --max-time 8
conda run -n airfogsim python airfogsim_cross_graph_evidence_closure.py --seed 0 --max-time 12
conda run -n airfogsim python task_resource_conservation_audit.py --seed 0 --max-time 12
conda run -n airfogsim python paired_action_causal_sensitivity.py --seeds 0 1 2 --max-time 12
```

结果统一写入`code/artifacts/small_experiments/`。本目录中的受控夹具不能替代AirFogSim或真实数据验证。
