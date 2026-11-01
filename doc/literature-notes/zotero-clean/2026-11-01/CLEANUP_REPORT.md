# STEP ZOTERO-20261101 — PIJWM 文库分类、索引与冲突审计

日期：2026-11-01（Asia/Shanghai）。状态：**获批分类/修正与文库验收完成；跨项目合并及少数版本/贡献归属待审；全仓知识索引既有问题阻塞推送。**

## 结论

- 实时根集合：PKU → PIJWM，PXDR9FBZ；收件箱GVMEXXZE保留，11条已分类、剩余0。121个唯一主条目各有且仅有一个固定主分类，未增设第11个主类。
- 两次逐项批准后完成12条修正：8次现有主分类调整、2条正式题名恢复、1条摘要恢复、1条错误DOI及论文身份整体修复。
- 保留全部121个主条目key、121个PDF附件key、1条笔记key、父子关系、标签和其他项目集合关系。Zotero随26MWELTC年份改正自动把附件9AHMZ4KD文件名年份从2026改为2025，已单独记录；未直接操作存储文件。
- 项目内部精确DOI及题名/作者/年份重复均为0；全库只读比对确认3对PIJWM—RRM独立重复记录。未合并、删除或移动RRM记录。
- 所有111条当前DOI取得标识符权威响应：93条Crossref、18条DataCite；10条无DOI保留。此处的核验不等于所有出版信息、指标或全文均正确。

## Step Goal / Definition Basis

按用户明确调用的zotero-clean完成收件箱分类、全项目只读审计、索引与重复/冲突处理计划；只对获批key执行额外修正。项目现状以2026-10-27任务书/场景合同和10-28 R2-09A证据为准：车与UAV提供LiDAR感知，逻辑RSU处理；当前v1仅特征包选择和RB分配，编码、计算、DAG、Route固定。R2-09A只是合成CPU服务闭环；不能据此宣称真实无线/感知、新世界模型或正式实验完成。10-21/10-23旧口径保留为历史，不当作当前传感器/实施边界。读取范围及哈希见source_identity.json；未修改Obsidian笔记或图片。

## Initial State / Files Involved

原工作区已有AGENTS、AI_CONTEXT、导航索引、根过程文件/旧文献/论文删除和未跟踪脚本等修改；保留这些变更，不恢复已删除文件。起始清单git_status_before.txt留在本地。根task_plan/progress/findings已删除，按现行指引在记录/本地计划表.md与记录/工作日志/代记过程。新增报告、索引、计划与执行回执均位于本目录；只对必要导航文件追加本任务记录。

## 分类前后

| 固定主分类 | 前 | 后 |
|---|---:|---:|
| 01_系统场景与问题定义 | 5 | 5 |
| 02_严格双图与图学习 | 8 | 8 |
| 03_世界模型与状态预测 | 27 | 25 |
| 04_任务卸载与资源优化 | 21 | 28 |
| 05_数据集、测量与仿真 | 12 | 12 |
| 06_评价、不确定性与决策诊断 | 6 | 6 |
| 07_AI业务场景与任务模型 | 5 | 2 |
| 08_Token与语义通信 | 11 | 17 |
| 09_网络QoS与任务质量映射 | 6 | 4 |
| 99_综述与奠基性文献 | 11 | 14 |

主分类依据是论文主要贡献，不是题名关键词或当前是否热门。项目用途另列：当前感知/服务对照、可迁移方法机制、历史卸载参考、未来RGB/大模型扩展。现有分类也没有被一概当作已核实：逐篇索引区分“本轮摘要/原文核对”与“保留既有归属”，没有冒充121篇已精读。

## Changes / Reuse

1. classification_plan.json：11条收件箱计划与理由；classification_execution.json及classification_verification.json保存写入和回读结果。2条原本已在03，保留其03只移除收件箱；9条获得主类。
2. verified_corrections_plan.json / verified_corrections_execution.json：用户第一批批准的3条07→08、2条正式题名与EdgeCooper摘要修复。
3. remaining_corrections_plan.json / remaining_corrections_execution.json：用户第二批批准的FT88WZND 09→04、JC8WK6WY 09→08、3篇跨领域综述03→99，以及26MWELTC预印本身份修复。
4. 26MWELTC原DOI 10.1109/ecnct70535.2026.11661443实际指向Zhu/Wang的另一篇会议论文。附件印有arXiv:2510.20149v2；DataCite正式题名和五位作者全部匹配，故恢复2025年预印本、正确DOI和arXiv来源，清理误配会议字段，并在Extra保留冲突历史。更换类型时Zotero自动把旧会议字段塞入Extra，已按获批计划再次清除并回读。
5. 全程复用已安装skill的本地HTTP API、分页、分类及导出方法；不使用UI自动化、数据库或存储目录访问。API密钥不进入记录或Git。

## 索引入口

- PIJWM_ZOTERO_DETAILED_INDEX_20261101.md：按十类组织的121篇阅读导航。
- PIJWM_ZOTERO_DETAILED_INDEX_20261101.csv / .json：key、题名、年份、DOI、来源、主类、当前用途、证据状态、复核标记、重复链接、附件/笔记数量。
- 简版PIJWM_ZOTERO_INDEX_20261101.md / .csv与前后版本一并本地保留；主条目数按唯一key计数，不把多重成员关系加成篇数。

## 重复与待审冲突

跨项目精确重复的拟保留主key与另一key：
- 4PFMSEHY ↔ GAJYIJWW：Learning latent multimodal dynamics for optimized resource planning；两个PDF child key均须保留，拟取两项目集合并集。
- 8L7P5YNN ↔ H6CMFRKK：Mastering diverse control tasks through world models；两个PDF child key均须保留，拟取两项目集合并集。
- 2BTIMUCB ↔ 83AZUBUR：Learning latent wireless dynamics from channel state information；两个PDF child key均须保留，拟取两项目集合并集。

详见duplicate_review_plan.json。其状态是待逐项合并批准，且当前HTTP helper没有已验证的原生合并实现；不能用“删除一个记录”冒充无损合并。原文索引文本可对照，但未验证PDF二进制哈希相同。

近似题名筛选的8对候选见near_duplicate_candidates.json，均保留；不同DOI/作者不等于绝对排除版本关系，不凭相似分数合并。AirFogSim的JOSS/TMC及Colosseum相关出版物也保留为相关文献，不按工具同名合并。

尚有5条书目信息版本复核：
- 3MFDIRK8：Graph Neural Networks for Wireless Communications: From Theory to Practice；date_year。
- MU9TZL9S：Modeling relational data with graph convolutional networks；venue_or_record_type。
- 7B8X48TK：ENGNN: A General Edge-Update Empowered GNN Architecture for Radio Resource Management in Wireless Networks；date_year。
- KQXI4GWD：AirFogSim: A Light-Weight and Modular Simulator for UAV-Integrated Vehicular Fog Computing；date_year。
- UW2UJ3X5：Deepmimo: A generic deep learning dataset for millimeter wave and massive mimo applications to vehicular communications；version_repository。

其中3条涉及在线年份与卷期年份（3MFDIRK8、7B8X48TK、KQXI4GWD），1条涉及预印本/会议版本（MU9TZL9S），1条涉及SSRN DOI与arXiv来源关系（UW2UJ3X5）。这些尚不能直接判错，未擅改。

另有2条主要贡献归属仍有歧义：Y63ZQVRG、VHJSJSQL，现有09归属保留；摘要同时覆盖编码/压缩与资源优化，候选08或04，见remaining_classification_reviews.json。没有新增永久复核分类。

6QKKAHEZ的卷期日期为2026年12月，在本次2026-11-01审计时仍属未来卷期；DOI记录与题名一致，不能因未来卷期就判伪造或删除，也未把未来卷期说成今日已出刊。

未发现可确认的撤稿/撤回证据；未逐篇独立查询出版商全部撤稿公告，不能声称撤稿检查全面通过。未重跑期刊分区/影响因子与翻译流水线，不把已有Extra指标提升为本轮新核验。

## Validation / Actual Results

- final_acceptance.json：121个主key、121个PDF child key、1个note key完整；主类唯一、收件箱0；批准字段按计划一致，标签和其他项目成员关系保留。
- 本地API恢复快照：api_snapshot_before.json及.zip已解码回读、压缩包CRC与父子关系验证，recovery_gate.json保存SHA-256。仅支持本轮元数据/成员关系恢复，不是PDF二进制灾备；不据此授权删除/合并。原始快照/全文读取缓存/笔记内容保留本地且不提交Git。
- 附件/笔记关系：只读枚举597条全库记录，0个缺失parent key，0个PIJWM无父附件/笔记；未操作其他项目。121份PDF的字节签名未重验，因为API/file跳转到存储路径，遵守skill不越界读取。
- zotero-clean离线测试11/11通过，quick_validate通过，实际命令/输出在skill_validation.json。
- helper最终审计0个缺刊名/来源候选；旧helper误用期刊字段会把会议/预印本误报，修正为按条目类型取字段，并支持旧archive别名。
- 全仓知识索引--check仍为既有5项不同步，见repository_index_check.json；未覆盖重建或夹带其他工作区变更，推送门保持关闭。

## Skill Updates / Context Consistency

按用户授权并参考skill-creator，更新全局zotero-clean：贡献分类与当前用途分离；按类型核对来源；DOI规范化；过期计划停止；未知子条目数为null；跨项目重复/备份范围/限流/二进制核验边界；标签顺序及类型转换Extra与自动附件名的回读规则。修改前skill压缩快照留在本地，skill_changes.patch为零上下文补丁，可审阅或用git apply --unidiff-zero重建；未改AGENTS.md。

Context Consistency Check：当前场景、架构、数据流、模块、实验与科研决策均未改变；只在AI_CONTEXT 00/08及项目索引/变更记录增加文献入口和未决事项。无GPU、训练、远端运行或locked_test访问。

## Expected vs Actual

符合预期：未分类清零、完整索引与获批修正完成。部分保留待审：3对跨项目重复不具备逐key合并批准及已验证合并工具，5条版本差异和2条贡献歧义未被强行“修正”。这不等于文库所有论文已精读或全部出版指标已复核。

## Completion Report

1. 本次做了什么：按当前LiDAR车—UAV—RSU工程范围维护十类；分类11条，完成12条获批修正、全库只读比对和多格式索引。
2. 得到了什么结果：121篇各一主类，收件箱0；发现并修复1条确实错配的DOI；3对跨项目重复有逐项保留/合并候选计划。
3. 是否符合预期：部分符合；授权的可执行清理已完成，合并/少数书目及贡献归属需后续审阅。
4. 当前还存在的问题：跨项目合并工具与逐key批准、5条版本信息、2条贡献归属、全仓既有索引门；二进制PDF未重验。
5. 记录位置：本报告、详细索引、final_acceptance.json、各计划/执行回执与记录/本地计划表.md。
6. Git状态：单独提交本任务可审阅材料；不夹带旧改动。当前知识索引5项失败，按仓库规则不推送；提交结果另见git_delivery.json。
7. 建议下一步：仅审阅duplicate_review_plan.json的3对跨项目重复，决定是否另行批准并建立可验证的无损合并操作；不自动扩展到模型或实验。
