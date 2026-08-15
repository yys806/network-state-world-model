# PI-JWM-DG-Contract-v3 R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从现有AirFogSim正式轨迹生成老师口径的PI-JWM物理—信息双图v3映射、固定张量和可复核验收产物。

**Architecture:** `airfogsim_teacher_graph_v3.py`负责纯函数重映射与结构验证；`airfogsim_teacher_tensor_v3.py`负责时间张量和mask；构建脚本只编排54条非锁定轨迹、locked-test完整性继承、train-only统计和manifest。所有新行为先由unittest失败用例固定。

**Tech Stack:** Python 3、NumPy、标准库JSON/CSV/hashlib、unittest、现有PI-JWM正式数据目录。

---

### Task 1: 冻结v3图协议

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_teacher_graph_v3.py`
- Test: `代码/tests/test_airfogsim_teacher_graph_v3.py`

- [x] 先用`find_spec`测试证明模块尚不存在。
- [x] 创建最小模块后运行测试，确认模块发现用例通过。
- [x] 增加重映射、字段分层、`CIP/CEP/CFL`和破坏输入拒绝测试，确认因缺失接口而失败。
- [x] 实现`remap_teacher_aligned_graph`、`audit_v3_source_fields`和`validate_teacher_aligned_graph`，只满足上述测试。
- [x] 运行`python -m unittest test_airfogsim_teacher_graph_v3.py`，预期全部通过。

### Task 2: 冻结v3张量协议

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_teacher_tensor_v3.py`
- Test: `代码/tests/test_airfogsim_teacher_tensor_v3.py`

- [x] 先写模块发现失败测试并运行。
- [x] 增加物理/信息特征隔离、缺失mask、`CIP/CEP/CFL`索引、padding和有限值测试，确认缺失行为失败。
- [x] 实现`TeacherAlignedTensorContract`、容量推断、单轨迹张量化和验证函数。
- [x] 运行`python -m unittest test_airfogsim_teacher_tensor_v3.py`，预期全部通过。

### Task 3: 构建正式v3数据与张量

**Files:**
- Create: `代码/scripts/build_airfogsim_teacher_aligned_v3.py`
- Test: `代码/tests/test_build_airfogsim_teacher_aligned_v3.py`
- Output: `代码/artifacts/datasets/airfogsim_teacher_aligned_v3/`

- [x] 先写三轨迹fixture，验证locked-test loader绝不被调用、只输出两条非锁定张量并使用train-only统计；运行并确认失败。
- [x] 实现构建器、协议JSON、trajectory index、逐轨迹mapping/NPZ/report、validation、dataset summary和manifest。
- [x] 运行专项测试并确认通过。
- [x] 在60条正式轨迹上执行构建；字段审计无必需缺口，未调用AirFogSim。

### Task 4: 全量验收和文档同步

**Files:**
- Modify: `D:/禹尧珅/人工智能知识库/北大科研/PIJWM/PIJWM推进.md`
- Modify: `本地计划表.md`
- Modify: `progress.md`
- Modify: `findings.md`

- [x] 复算artifact manifest并核对54/6 split、15660个非锁定窗口和locked-test零张量目录。
- [x] 运行三个R1专项测试、相关旧正式数据测试和源码编译。
- [x] 更新R1状态、实际容量、字段覆盖、缺失边界、是否重跑和验收命令。
- [x] 运行`git diff --check`并回读所有机器报告后再声明R1完成。

### R1验收结果（2026-08-03）

- 正式输出：`代码/artifacts/datasets/airfogsim_teacher_aligned_v3/`。
- 数据规模：54条非锁定轨迹张量、6条locked-test完整性记录、15,660个非锁定窗口；固定容量为44节点、1,892条物理空间边、1,892条信息通信边、588条业务流、481个任务和1,236条DAG边。
- 构建边界：复用既有AirFogSim轨迹，`airfogsim_rerun_required=false`；可选增强信道字段使用零值加false mask，未伪造观测。
- 完整性：278个受管文件的文件集合、大小和SHA-256独立复算全部匹配；locked-test无张量目录、未读取标签内容。
- 调试证据：全局归一化统计由大张量拼接改为逐轨迹流式累加，并增加校验后续构；两个行为均经过失败—通过测试。
