# R2-05A 审计摘要

有限审计完成，正式训练BLOCKED、基线NEEDS_RESEARCHER_DECISION、物理对齐NOT_ESTABLISHED。

完整V2U4Real包含LiDAR/RGB；本轮只核验五帧20个train PCD/YAML，不代表完整视觉车-RSU/通信/截止时间系统已适配。

固定75份源码/配置身份通过；原清单/第三方目录未改。原生Car-only、首次去重与冻结B有差异；论文3D AP与官方XY面积IoU有差异。人工探针默认AP0.5/全局排序AP1仅说明排序规则，不是真实检测性能。

19项新增和77项相关测试共96项通过，编译通过。知识索引只读检查仍5项不同步，不重建、不绕过、不推送。最终状态详见validation.json和acceptance.json。

合同保持DRAFT — AWAITING RESEARCHER REVIEW；唯一下一步为研究者决定是否将V2U4Real限定为LiDAR三维感知模块验证。本轮停止，无GPU/locked_test/训练/推理服务/正式标签。
