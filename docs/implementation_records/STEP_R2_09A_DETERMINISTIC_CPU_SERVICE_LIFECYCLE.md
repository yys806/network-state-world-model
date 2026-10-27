# STEP R2-09A — 确定性 CPU 多时隙通信与任务生命周期闭环

日期：2026-10-28（Asia/Shanghai）。本轮是研究者明确授权的 R2-09A 工程实现；结果只适用于 R2-08 synthetic contract fixture，不代表真实无线、物理同步、真实感知质量或正式科研参数已通过。

## 1. Step Goal / Definition Basis

目标是依据最新版《新场景下训练前任务书.md》任务 4、`组会笔记/1028组会.md`、R2-05B/R2-07/R2-08 记录和 `docs/contracts/PIJWM_SCENARIO_CONTRACT_V1.md`，在不调用 AirFogSim 无线物理模型的条件下，完成可复现的 CPU 服务与任务生命周期闭环。

本轮继承的事实边界：R2-05B 只把感知 payload 放入应用等待队列；R2-07 的速率读取和容量换算是条件性瞬时容量，不能当实际交付；R2-08 已校验六类记录、100 ms 外部观测周期、截止时间锚定 `target_observation_time + allowed_total_latency`，但没有队列推进、跨时隙累计、完成或超时引擎。本轮没有修改这些历史合同的正式科研参数。

## 2. Initial State / 当前门与保全

- 起始分支：`main`；起始 HEAD：`6988a91 feat(scenario): add versioned scenario and service contracts`；起始相对 `origin/main` 为 ahead 10。
- 起始工作区已有 `AGENTS.md`、AI_CONTEXT、索引、文献/论文删除、根过程文件删除和多份未跟踪脚本等修改。本轮保留这些内容，没有 reset、clean、覆盖重建或移动历史证据。
- 起始知识索引门沿袭 R2-08：生成索引中 `tracked_file_inventory.csv`、`python_dependency_map.json`、`artifact_catalog.csv`、`archive_candidate_registry.csv`、`registry_summary.json` 不同步。本轮仅做 `--check` 只读检查，不运行覆盖式重建。
- GPU、AirFogSim 真实无线、V2U4Real 原始数据、validation/test/`locked_test`、正式检测和训练均未访问。

## 3. Files Involved / Changes

### 新增

1. `code/src/pi_jwm/deterministic_cpu_service_lifecycle_v1.py`
   - `DeterministicLifecyclePolicy` 显式接收 RB 数量、速率单位、固定 RSU 处理时长、截止相等策略和同刻事件顺序；这些是工程测试策略，不是正式研究决策。
   - `DeterministicCpuServiceLifecycle` 注册 R2-08 `PerceptionTask`/`FeaturePacket`，分别维护 task request 与每个 `packet_id` 的 transfer；只在请求送达后接受特征服务。
   - 服务容量按 `Σ rate × unit_factor × interval / 8` 计算，区间不得超过截止时刻；实际交付使用带小数 credit 的整数 byte，始终不超过请求/包大小。服务记录用 `ServiceResult`，并固定 `physical_service_status=SYNTHETIC_ONLY`；请求服务用独立 `RequestServiceReceipt`。
   - 按 `packet_id` 记录部分/完整交付、跨 100 ms 边界、RSU 处理开始/完成和模拟检测成功/失败/未执行；不实现空间切块、重组、缺块感知或融合。
   - 拒绝负/非有限速率、非法时间、越界/重复 RB、同目标重叠区间、重叠 RB、重复事件、请求未送达时的包服务和截止后服务。
2. `code/scripts/configs/pi_jwm_r2_09a_deterministic_cpu_v1.json`
   - 版本化 synthetic-only 工程夹具；沿用 R2-08 50 RB、30 ms 调度语义的测试背景，另显式给出 40 ms 固定 RSU 处理、截止/同刻测试策略和边界状态。
3. `code/tests/test_deterministic_cpu_service_lifecycle_r2_09a.py`
   - 15 项 CPU 专项测试，覆盖单/多时隙、fractional byte、边界、并发任务、多包、失败原因、单位/RB/重复防护、固定空间块接口和确定性重放。
4. `code/scripts/audit_deterministic_cpu_service_r2_09a.py`
   - 独立运行专项测试并写出 `execution.log`、`result.json`、源码指纹和 `PASS/BLOCKED/NOT_RUN` 矩阵。

### 同步更新

AI_CONTEXT `00/03/04/05/07/08`、`docs/PROJECT_INDEX.md`、`ARCHITECTURE.md`、`RESEARCH_STATUS.md`、`EXPERIMENT_INDEX.md`、`RESULTS_INDEX.md`、`CHANGELOG.md`、`记录/本地计划表.md` 和 `记录/工作日志/{task_plan,progress,findings}.md` 增加本轮入口、数据流、验证边界和停止条件。没有更新 `AGENTS.md`，没有改变 formal pending 配置、世界模型、checkpoint、loss、训练协议或第三方源码。

## 4. Reuse / Theory–Implementation Boundary

- 复用 `scenario_contract_v1.validate_scene_config`、`record`、`validate_record` 的 R2-08 schema/来源/单位校验；FeaturePacket 保留 `granularity`、`block_id`、`spatial_block_contract_ref`。
- 复用 R2-07 的十进制 Mbps/bit/s→byte 容量公式语义，但不把 R2-07 瞬时容量当交付；实际 simulated receipt 由本模块独立生成并标记 synthetic。
- 复用 R2-05B 的 payload/TaskManager 语义边界，但没有调用 AirFogSim `offloadTask`、无线速率、传播或原生任务完成字段。
- `fixed_spatial_block` 测试只确认按包 ID 的接口可迁移；没有实现真实空间切块、重组、缺块检测或特征融合。

## 5. Validation / 实际命令与结果

| 检查 | 命令 | 结果 |
|---|---|---|
| R2-09A 专项 | `python -m unittest discover -s code/tests -p 'test_deterministic_cpu_service_lifecycle_r2_09a.py'` | **PASS，15 tests** |
| R2-08 定向回归 | `python -m unittest discover -s code/tests -p 'test_scenario_contract_v1.py'` | **PASS，25 tests** |
| R2-07 定向回归 | `python -m unittest discover -s code/tests -p 'test_airfogsim_service_interface_r2_07.py'` | **PASS，8 tests** |
| R2-05B 定向回归 | `python -m unittest discover -s code/tests -p 'test_v2u4real_r2_05b_interface_v1.py'` | **PASS，7 tests** |
| 内存编译 | `compile()` 检查本轮 3 个 Python 文件，`PYTHONDONTWRITEBYTECODE=1` | **PASS，3 files**；避免现有 `__pycache__` ACL 问题 |
| 机器验收 | `python code/scripts/audit_deterministic_cpu_service_r2_09a.py` | **PASS**；15/15，失败 0，错误 0 |
| 知识索引门 | `python code/scripts/build_project_knowledge_index_v1.py --check` | **BLOCKED（继承）**；五项生成索引不同步；本轮未重建/覆盖 |

最终机器回执：`code/artifacts/audit/pi_jwm_r2_09a_deterministic_cpu_service_20261027/result.json`；执行日志：同目录 `execution.log`。回执记录 Python 3.13.5、`gpu=false`、`locked_test=false`、`airfogsim_wireless_service=false`、`real_detector=false`，并保存 7 个源码/配置 SHA-256 指纹。

## 6. Results / Expected vs Actual

**符合本轮工程预期，科研边界仍是 synthetic-only。**

- 单时隙完成、多时隙部分交付、跨时隙小数容量累计、完整交付、剩余字节更新均 PASS；交付不会超过包大小，也不会重复计入同一重叠区间或 RB。
- 请求和特征包服务进度独立；请求未完成时，特征服务被拒绝；任务跨 100 ms 观测边界时 ID、进度、截止时间不重置。
- 任务终止原因可分别读取：`REQUEST_NOT_DELIVERED`、`FEATURE_PACKET_NOT_DELIVERED`、`RSU_PROCESSING_TIMEOUT`、`DETECTION_FAILED`；另有 `DETECTION_NOT_EXECUTED` 和 `COMPLETED`。四类失败没有合并成一个笼统失败。
- 截止恰好相等的 inclusive/exclusive、`processing_before_deadline`/`deadline_before_processing` 均显式运行并通过；它们只属于工程测试策略，未冻结 formal pending 合同。
- 同一来源多个独立包和 `fixed_spatial_block` 的包 ID 状态均 PASS；真实空间切块/重组/缺块感知仍 `NOT_RUN`。
- 理论容量和模拟交付分别落账；所有服务结果是 `SYNTHETIC_ONLY`，不能宣称真实无线传输成功、物理覆盖或真实感知性能。

## 7. Known Issues / 当前阻塞

1. R2-07 原生步内时间错位、SINR/中断/高度/快衰落可信度和原生 Mbit—payload byte 绑定仍未闭合；因此本轮引擎不接入 AirFogSim 原生服务。
2. 正式区域概率、真实/合成节点映射、正式包粒度、截止预算、内部调度和同刻事件顺序仍待研究者确认；当前数值只在 synthetic fixture 中有效。
3. 固定 RSU 处理时长和 Mock 检测是可控桩，不是测量延迟或真实检测结果；没有任务质量/AP/3D AP 证据。
4. 知识索引 `--check` 的五项既有不同步仍阻塞推送；未以重建索引掩盖该状态。

## 8. Git / Context Consistency

本轮只准备提交本报告、引擎、工程夹具、专项测试、验收脚本及本轮 AI_CONTEXT/索引/计划增量；起始工作区既有修改、删除、未跟踪文件和未推送提交不纳入本轮提交。提交前将以 staged diff 再核对一次，避免夹带既有工作。知识索引门未通过时不强推；本轮不会自动启动 R2-09B、真实无线或训练。

## 9. Completion Report / 停止条件

1. **本次做了什么**：实现确定性 CPU 多时隙服务和请求—特征—RSU—检测生命周期，保留包粒度扩展入口，完成专项验收和机器回执。
2. **得到了什么结果**：15 项 R2-09A、25 项 R2-08、8 项 R2-07、7 项 R2-05B 定向测试通过；四类失败可区分；真实无线/检测未运行。
3. **是否符合预期**：符合预期（工程闭环）；不等于真实无线或正式科研方法通过。
4. **当前还存在的问题**：真实 AirFogSim 服务、正式参数/同刻策略、空间切块与真实检测仍待决；知识索引五项不同步。
5. **记录位置**：本报告、`code/artifacts/audit/pi_jwm_r2_09a_deterministic_cpu_service_20261027/`、AI_CONTEXT 00/03/04/05/07/08、计划/进度/发现记录。
6. **Git 状态**：起始 HEAD `6988a91`；本轮提交和推送在验证后执行，既有脏工作区不夹带。
7. **建议下一步**：研究者审阅本报告和 `result.json`，只决定一个最小后续动作（R2-09B 或另行授权真实无线接入），本轮停止。
