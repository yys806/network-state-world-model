# PI-JWM 机器注册表

本目录把人类可读索引补充为可由脚本检查的结构化入口。它们用于定位，不替代原始代码、配置、checkpoint、metrics、manifest 或 audit。

ChatGPT 网页端先读根目录 `AI_CONTEXT/00_PROJECT_STATE.md`。`document_authority.json` 的 `chatgpt_entrypoint` 和九个上下文文件承担稳定路由，本目录继续提供机器可查证的细粒度索引。

## 人工维护的注册表

- `document_authority.json`：文档权威关系和推荐读取顺序。
- `experiment_registry.json`：当前重要实验的统一完整字段；未知/不适用项必须用 `null + field_notes`。
- `results_registry.json`：正式可引用数字到实验和 audit 的连接，数值与原始 acceptance 自动核对。
- `historical_method_registry.json`：重要旧方法的尝试原因、结果、弃用原因、替代关系和原始证据。
- `question_routes.json`：常见自然语言问题到权威入口和验证证据的路由。
- `deferred_work.json`：已推迟且必须重新获得用户授权的工作；当前包括第三个 seed 和远端同步。

## 自动生成的注册表

运行：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python .\code\scripts\build_project_knowledge_index_v1.py
python .\code\scripts\build_project_knowledge_index_v1.py --check
```

检索常见问题：

```powershell
$env:PYTHONUTF8='1'
python .\code\scripts\query_project_knowledge_v1.py --query "某个实验之前做过没有"
```

生成内容：

- `generated/tracked_file_inventory.csv`：项目文件、大小、SHA-256、职责和生命周期状态。
- `generated/python_dependency_map.json`：Python 模块的本地依赖、反向引用和直接测试。
- `generated/artifact_catalog.csv`：artifact 一级目录及其控制文件，不哈希 checkpoint 或 predictions。
- `generated/archive_candidate_registry.csv`：历史 Python 文件逐项哈希、反向引用和回滚路径；当前只作逻辑归档。
- `generated/registry_summary.json`：数量、读取错误和科研边界摘要。

`--check` 只比较当前仓库与注册表，不写文件，同时验证九个 `AI_CONTEXT` 文件、人工注册表 schema、证据路径、正式结果原始值和 SHA-256。新增模块、实验、结果、路径或重要文档后，应先做 Context Consistency Check，再更新人工注册表并重新生成。
