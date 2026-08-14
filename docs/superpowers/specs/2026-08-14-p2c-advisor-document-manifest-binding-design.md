# P2-C研究进展文档Manifest绑定设计

## 目标

将`文档/研究进展/2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md`纳入P2-C机器产物的现有源码/证据manifest，使该文档发生任何字节变化后，既有canonical的`--verify-only`必须失败，直到按当前文档重新发布审计bundle。

## 当前问题

P2-C runner当前的`CANONICAL_SOURCE_PATHS`只包含：

- 审计库；
- runner；
- 两个测试；
- P2-C实施计划；
- P2-B全双图采集器设计。

P2-C研究进展文档没有进入`source_hashes`。因此，文档可以改变“source closure已恢复”“当前只剩四项阻断”等结论，而既有P2-C manifest仍能通过验证。这与PI-JWM的理论—实现—证据一致性原则冲突。

## 采用方案

复用现有单一manifest，不增加新的文档manifest：

1. 在`代码/scripts/run_p2c_scale_distribution_audit_v1.py::CANONICAL_SOURCE_PATHS`中加入P2-C研究进展文档的项目内绝对`Path`；
2. 由现有`_portable_source_key`转换为项目相对POSIX键；
3. 由现有`_source_hashes`写入SHA-256；
4. 由现有`verify_audit_bundle`逐项复算并拒绝缺失或哈希不一致的文档；
5. 重新发布P2-C canonical，使manifest绑定当前已核验的恢复结论。

不为文档增加专用验证分支，不复制哈希逻辑，不改变P2-C审计算法、正式配置候选或剩余四个阻断。

## 未采用方案

### 独立文档manifest

会形成机器审计manifest与叙述文档manifest两套验证入口，增加漂移风险，且现有`source_hashes`已经能表达该依赖。

### 维持文档不绑定

会允许advisor-facing结论绕过机器验证，直接违反项目永久一致性约束。

### 仅在文档中手写artifact哈希

这是单向引用，不能让机器验证感知文档被篡改，也不能关闭双向证据链。

## 测试设计

在`代码/tests/test_run_p2c_scale_distribution_audit_v1.py`中扩展现有发布/验证测试：

1. 发布临时P2-C bundle；
2. 断言`manifest["source_hashes"]`恰含键`文档/研究进展/2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md`；
3. 断言该键的值等于研究进展文档当前SHA-256；
4. 保留现有artifact篡改拒绝测试；
5. 新增独立测试，使用临时文档路径替换`CANONICAL_SOURCE_PATHS`中的目标文档，发布后修改临时文档，再验证`verify_audit_bundle`返回`passed=false`且错误包含`source hash mismatch`。

RED阶段必须因manifest缺少目标文档键而失败。GREEN阶段只增加一个canonical source路径，不修改验证算法。

## 发布与归档

当前canonical不能原地覆盖。实施时：

1. 先在新的候选目录发布并运行`--verify-only`；
2. 比较新旧报告和formal config，要求两者字节不变，只有manifest的source hash集合、source count及manifest自身字节发生预期变化；
3. 将当前canonical移动到带`pre_advisor_doc_binding_20260814`后缀的可恢复归档；
4. 将已验证候选提升为canonical；
5. 提升后再次运行P2-B/P2-C双`--verify-only`、83项AirFogSim SHA-256复算和P1/P2 focused suite。

## 验收标准

- 自动测试先RED后GREEN；
- P2-C manifest包含目标文档的唯一portable key和正确SHA-256；
- 修改绑定文档会使验证稳定失败，恢复文档后验证通过；
- P2-C审计报告仍为`blocked`，且只含四项既有阻断；
- `formal_data_approved=false`不变；
- P2-B/P2-C verify-only、83/83外部依赖哈希和focused suite全部通过；
- 不启动GPU、不访问locked test、不生成正式轨迹、不修改AirFogSim第三方源码。
