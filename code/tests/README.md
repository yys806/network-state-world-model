# PI-JWM 测试导航

## 测试层次

- 当前正式 P4：`test_formal_*.py` 和当前实体级 RSSM runner 的定向测试。
- 项目结构与索引：`test_project_configuration.py`、`test_build_project_knowledge_index_v1.py`、`test_project_knowledge_registry_contract.py`。
- 快速问答检索：`test_query_project_knowledge_v1.py`，覆盖当前方法、历史失败、正式结果和延期任务。
- ChatGPT 上下文合同：`test_ai_context_contract.py`，覆盖九个入口、事实边界、三方规则、冲突模板和索引接入。
- AirFogSim 实机测试：需要 `airfogsim` Conda 环境、`PYTHONUTF8=1`、SUMO/TraCI 依赖。
- R3–R6、v6–v11：历史回归和旧证据复现，不代表当前研究主线。

## 运行边界

本机核心测试可以使用：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
$env:PYTHONUTF8='1'
python -m unittest discover -s .\code\tests -p 'test_formal_*.py'
```

全量测试包含需要 AirFogSim 环境和本地历史 artifact 的测试，不能把缺少外部依赖产生的错误伪装成代码通过，也不能因此放宽正式合同。已知基线见 `docs/KNOWN_CONFLICTS.md`。
