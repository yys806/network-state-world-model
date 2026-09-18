# 研究背景与问题

> 2026-09-18 当前目标定义来自研究者只读目录中的 `00–06`；本文件下方的旧 P4 描述只用于说明被审计的当前代码。实现差异见 `docs/PIJWM_IMPLEMENTATION_TRACKER.md`。

## 研究对象

PI-JWM 研究由车辆、无人机、路侧单元、边缘服务器和云节点构成的动态通信—计算系统。目标是学习动作如何影响物理连接、数据传输、任务处理和资源状态随时间共同演化。

Source of truth：理论边界见 `记录/PIJWM主文档.md`；实际实现见 `code/src/pi_jwm/`；当前状态见原始实验与 audit。AirFogSim 只提供参考仿真和数据，不是研究框架。

## 核心问题

给定过去 8 步系统历史和未来动作，预测随后 20 步的联合状态，包括物理节点、通信链路、任务数据流和任务状态；在世界模型通过 P4 后，目标才是用预测后果比较候选动作并滚动重规划。

## 问题建模

- 物理图：物理节点与有向通信链路。
- 信息图：附着在物理节点上的 agent 与任务数据流。
- 跨图关系：agent—node 附着、flow—physical-edge 承载，以及任务到节点/数据流的映射。
- 动作条件：未来任务动作及其显式源/目标节点进入逐步 rollout。
- 预测对象：连续状态、实体存在性、链路活动、任务生命周期、DAG 状态、吞吐量、RB 占用、时延和不确定性。

实现入口：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`、`code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py`。

## 当前代码中的历史候选方法

当前候选为 `entity_aligned_dual_graph_rssm_v1`：先用确定性双图模型编码实体关系，再为 node、physical edge、flow、task 分别维护随机状态；训练时 posterior 可以看目标用于学习，验证和部署只使用 prior 预测未来。

这只是旧协议下获得两个单 seed 非锁定验收的方法。它已进入 Historical / Archived 边界，不等于新定义采用的方法，也不等于最终科研方法已经冻结。

## Research Rationale

以下属于研究动机，不是代码本身可以证明的结论：实体级随机状态旨在避免历史 global RSSM 把一个全局修正广播给所有实体，从而更好地区分不同节点运动和不同链路排序。该动机由 `code/artifacts/audit/pi_jwm_p4_first_principles_audit_20260906_v1/first_principles_audit.json` 支持，但最终科研意义仍需三 seed、消融和后续研究者解释。

## 历史工作假设与当前审计结论

- 历史假设：实体对齐 latent、因果运动输入、逐边链路表示和分阶段冻结训练可满足旧 P4 门；两个正式 seed 单独通过。
- 当前审计：新定义改变双图语义、动作空间、随机状态范围、规则反馈和训练边界，因此旧证据不能作为新定义验收。
- 未验证：新定义的数据合同、模型可学习性、跨 seed 泛化、locked test、正式 planner 收益和最终创新结论。

## 目标闭环与当前边界

目标定义是“同一 belief 下逐候选调用世界模型 → 预测未来状态/任务/成本/风险 → 选择动作 → 只执行首动作 → 接收真实反馈后重规划”。当前 `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` 仅为 CPU 原型；合法候选生成、目标函数、风险定义和真实执行反馈尚未冻结。

Unverified：最终采用纯候选搜索、学习策略或混合策略，尚无研究者决策。
