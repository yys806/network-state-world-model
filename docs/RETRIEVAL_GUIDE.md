# PI-JWM AI 检索指南

## 固定检索顺序

1. 先读 `PROJECT_INDEX.md` 和 `RESEARCH_STATUS.md`，确定当前门和结论边界。
2. 查 `CODE_INDEX.md`、`EXPERIMENT_INDEX.md` 或 `RESULTS_INDEX.md` 定位对象。
3. 需要精确结论时，读取 `registries/` 中的机器注册表。
4. 最后回到代码、配置、原始 metrics、manifest、checkpoint 或 audit 验证。

也可以先运行只读机器检索：

```powershell
$env:PYTHONUTF8='1'
python .\code\scripts\query_project_knowledge_v1.py --query "之前的完整 RSSM 为什么不用了"
```

输出只给候选入口和证据路径，不直接建立科研结论。

不得跳过第 4 步就声称“已实现”“已经通过”或“当前最好”。

## 常见问题到入口的映射

| 用户问题 | 第一入口 | 最终核验 |
| --- | --- | --- |
| 当前研究到哪里 | `RESEARCH_STATUS.md` | 最新权威记录和 acceptance JSON |
| 当前模型在哪里 | `CODE_INDEX.md` | 模型 registry、runner、checkpoint identity |
| 某模块被谁调用 | `registries/generated/python_dependency_map.json` | 对应 import 和测试 |
| 某实验做过没有 | `EXPERIMENT_INDEX.md` | `experiment_registry.json` 和 artifact catalog |
| 某个数字从哪里来 | `RESULTS_INDEX.md` | `results_registry.json`、raw metrics、audit |
| 旧方法为何废弃 | `registries/historical_method_registry.json`、`ARCHIVE_CANDIDATES.md` | 失败 audit 和当时配置 |
| 第三个 seed 怎么续接 | `registries/deferred_work.json` | 冻结 protocol 和两个前序 acceptance |
| 文档冲突信谁 | `KNOWN_CONFLICTS.md` | `document_authority.json` 和更新机器证据 |

## 检索边界

- 文件名带 `formal`、`best`、`final` 或 `latest` 不等于正式有效。
- `code/src/pi_jwm/` 中保留历史兼容模块；必须读取生命周期标签，不能默认都是当前主线。
- `code/artifacts/` 中的目录很多；先看 artifact catalog，再看控制 JSON，不扫描或重算全部 checkpoint。
- meeting、paper 和历史日志只能提供语境，不能单独建立实现或性能结论。
