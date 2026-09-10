# 已知问题与冲突

Source of truth：本文件是入口；具体事实必须回到列出的代码、配置、测试或 artifact。

## 1. P4 尚未闭合

- Actual Implementation：当前正式 runner 支持三个冻结 seed，前两个已完成。
- Evidence：两份 `single_seed_acceptance.json` 均通过；`docs/registries/deferred_work.json` 将第三 seed 标为 deferred。
- Conflict：两个单 seed 证据不足以形成三 seed 结论。
- Status：等待研究者决定是否授权 `20260832`。
- Boundary：`locked_test_accessed=false`、`formal_performance_claim_ready=false`。

## 2. 全量测试存在历史/环境错误

- 2026-09-10 基线：1643 项为 0 assertion failure、17 errors。
- 已确认类别：缺少 AirFogSim `traci` 环境；旧 teacher fixture 与严格 RB 事件合同不符；旧 directed-dynamic fixture 无有效 calibration link 样本。
- 相比 2026-09-09 的 21 errors，历史 R5/R6 artifact 权限相关 4 个错误在当前权限下消失；这不是科研代码或实验结果变化。
- 影响：不能用全量套件证明物理迁移旧代码完全无回归。
- 不应采取：放宽当前合同、删除测试或修改科研逻辑来制造表面全绿。

## 3. 历史 artifact 控制文件读取曾受限

2026-09-09 受限环境下机器 catalog 曾记录 14 个历史 manifest/control file 不可读；不可读不等于文件不存在或实验失败。2026-09-10 在当前权限下重新生成后，802 个 artifact 记录的读取错误数为 0。该差异属于运行环境权限变化，不是实验状态变化。

## 4. 冻结协议状态字段是历史时刻

`protocol.json` 的 `status=ready_for_gpu_batch_probe` 表示协议冻结时状态；当前运行完成情况必须看后续 run 与 acceptance，不能只读这个字段。

## 5. 组会 PPT 结果滞后

`meeting/PI-JWM_组会汇报.pptx` 的结果页使用 seed 20260831 epoch 22 中间证据，晚于它的正式 epoch 39 结果和第二 seed 结果。PPT 不能作为当前两 seed 的最新验收来源。

## 6. 图术语仍需研究者最终统一

- Documented Intent：部分较早材料把通信关系统称为信息边。
- Actual Implementation：`physical_edge_state` 表示有向通信链路；`flow_state` 表示任务数据流信息边；agent 没有独立观测张量。
- Evidence：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`、`formal_airfogsim_window_v1.py`、当前 tensor contract。
- Affected Files：较早理论材料、旧 PPT 与后续论文表述。
- Conflict：最终理论术语如何命名仍属于科研决策。
- Status: Awaiting Researcher Decision

## 冲突记录模板

遇到新重大冲突时必须记录：`Documented Intent`、`Actual Implementation`、`Evidence`、`Affected Files`、`Conflict`、`Status: Awaiting Researcher Decision`，并停止相关科研逻辑修改。

Unverified：没有代码、config、experiment 或可读 audit 支持的问题只能标记为待核验，不能写成确认缺陷。
