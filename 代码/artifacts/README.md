# PI-JWM Artifacts

本目录保存 PI-JWM 的数据、模型、指标、图表和可复核报告。AirFogSim 产物仅作为仿真与数据来源，不代表项目框架。

## 目录约定

- `audit/`：一致性、数据契约、分布和闭包审计；结论必须同时读取状态字段与阻断项。
- `preflight/`：CPU、接口、采集器和运行前门；通过只证明对应门，不等于正式数据或完整方法完成。
- `formal_training/`：历史正式预算训练、checkpoint与汇总；旧协议结果不得直接升级为新协议结论。
- `analysis/`：跨运行统计、门控和诊断结果。
- `experiments/`：按运行或研究阶段组织的历史数据、checkpoint、指标和摘要。
- `reports/`：跨实验汇总、论文/PPT证据索引和研究报告。
- `protocols/`、`evaluation/`：冻结协议、指标映射和评价产物。
- `literature/`、`research_notes/`：研究过程中形成的文献检索与阅读材料，不替代Zotero权威库。
- `manifests/`：允许入Git的治理、保留、隔离和引用索引。
- `tmp/`、`.codex_work/`及迁移/传输目录：临时或本机辅助内容，不作为权威证据入口。

## 当前关键证据

- P0一致性审计：`audit/pi_jwm_p0_consistency_audit_v1/`。
- P1信息边契约：`audit/pi_jwm_p1_information_edge_contract_v4/`。
- P2-A CPU规则：`preflight/pi_jwm_cpu_inner_rule_v1/`。
- P2单步/多步smoke：`preflight/pi_jwm_p2_single_step_collector_v1/`与`preflight/pi_jwm_p2_multistep_collector_v1/`。
- P2-B v1：`preflight/pi_jwm_p2_full_dual_graph_collector_v1/`。
- P2-C v1：`audit/pi_jwm_p2c_scale_distribution_audit_v1/`。
- P2-B/P2-C v2候选：分别位于带`v2_candidate_20260814`和`v2_pre_document_closure_20260814`的目录；在最终文档闭包完成前仍是候选。

`formal_training/`和旧`experiments/`中的R3-R6、v6-v11结果保留为历史模型或策略证据，不能证明正式v4数据、新协议模型或候选rollout规划器已经完成。

## 结果口径

实验结论必须区分`deployable`、`true_future_reference`、`sample_oracle`和`test_best_diagnostic`。真实未来动作、逐样本oracle与测试集事后最优不能写成自主可部署结果。当前理论和阶段边界以根目录最新handoff、`本地计划表.md`和对应机器manifest共同为准。

## 清理规则

1. 被代码、文档或证据索引引用的路径不得自动清理。
2. 含唯一 checkpoint 或结构化 JSON/CSV/Markdown 结果的目录默认保留。
3. 未引用的 smoke/probe 只标记为 `manual-review`，不因名称自动删除。
4. 仅空目录、无结构化结果的失败运行和已被正式运行替代的纯 dry-run 可进入可恢复隔离区。
5. `.log`、`.pid`、缓存和本机虚拟环境不作为研究证据；日志只有在同目录已有结构化摘要时才可隔离。

查看 `manifests/retained_experiments.csv`、`manifests/quarantined_experiments.csv` 和 `manifests/referenced_paths.txt` 可追踪本次治理决策。

除`README.md`和`manifests/**`治理元数据外，本目录默认由`.gitignore`排除。忽略只控制Git发布，不代表本地文件可以删除。
