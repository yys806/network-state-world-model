# PI-JWM Artifacts

本目录保存 PI-JWM 的数据、模型、指标、图表和可复核报告。AirFogSim 产物仅作为仿真与数据来源，不代表项目框架。

## 目录约定

- `experiments/`：按运行或研究阶段组织的数据、checkpoint、指标和摘要。
- `reports/`：跨实验汇总、论文/PPT 证据索引和研究报告。
- `literature/`：研究过程中形成的文献检索与阅读材料。
- `packages/`：需要保留的可分发归档；临时补丁包不放在这里。
- `manifests/`：实验保留、隔离和引用索引。

## 关键证据

- 主数据集：`experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v2_60seed_20260619/`
- 冻结 v10：`experiments/pi_jwm_v10_action_aligned_20260619/V10_FREEZE_MANIFEST.json`
- v6-v9：保留关键基线、消融和结构化结果，作为 PI-JWM 状态 rollout 的演进证据。
- v11：保留严格匹配和最新候选证据，但方法状态仍是 `v11 candidate`，不是最终 v11 或 v12。

## 结果口径

实验结论必须区分 `deployable`、`true_future_reference`、`sample_oracle` 和 `test_best_diagnostic`。真实未来动作、逐样本 oracle 与测试集事后最优不能写成自主可部署结果。详细定义见 `文档/项目说明/实验结果口径.md`。

## 清理规则

1. 被代码、文档或证据索引引用的路径不得自动清理。
2. 含唯一 checkpoint 或结构化 JSON/CSV/Markdown 结果的目录默认保留。
3. 未引用的 smoke/probe 只标记为 `manual-review`，不因名称自动删除。
4. 仅空目录、无结构化结果的失败运行和已被正式运行替代的纯 dry-run 可进入可恢复隔离区。
5. `.log`、`.pid`、缓存和本机虚拟环境不作为研究证据；日志只有在同目录已有结构化摘要时才可隔离。

查看 `manifests/retained_experiments.csv`、`manifests/quarantined_experiments.csv` 和 `manifests/referenced_paths.txt` 可追踪本次治理决策。
