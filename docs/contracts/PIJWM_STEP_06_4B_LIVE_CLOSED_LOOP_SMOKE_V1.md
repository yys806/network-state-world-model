# STEP 6.4B 实时历史输入与最小闭环 smoke 合同

## 研究者决定与边界

2026-10-03 本轮授权冻结 `SYNCHRONOUS_PAUSED_SIMULATION`：到决策点暂停仿真，等待搜索完成；0.1 秒仿真间隔不构成墙钟期限，不能声称实时控制。

Smoke-only `FAIL_CLOSED_AND_STOP_EPISODE`：无可评分 H4、空域、缺真实字段、slot 不支持、动作桥拒绝、setter 失败都记录并结束 episode。没有 H3/NOOP/旧行为策略 fallback；这不是最终 fallback 决定。

MH-CEM K4/rho0.1，CPU FP32，B64，batch1：每轮16次配额容纳完整H1–H4。它是独立机制配置，不能作为正式Stage A batch16身份，也不能用于Stage B性能比较。不实现warm start、learned proposal或Route扩展。MH仍仅为pure-search backbone，最终hybrid未冻结。

## 实时输入

`step6_4b_live_bridge_v1.causal_prefix/build_live_sample`只白名单复制真实观测字段、当前时刻以前实际动作/结果；隔离内部未来审计字段。完整因果前缀用于重建Flow ledger，最近两个真实观测形成History；输入对象槽沿用 `history_causal_observable_object_union`。不调用要求future_steps的离线sample builder；target/future_action命名空间为空，不伪造未来标签。

张量化沿用正式TRAIN base/extension/flow统计，只按既有冻结容量补mask=false/数值0/索引-1/类别0；超容量立即拒绝。typed graph从当前History构建，posterior由当前观测编码初始化；不能用上一轮预测的state/graph/latent作为下一轮root。

`live_deadline_sidecar`从当前真实Task getters读取arrival/deadline并与当前观测对齐。首轮必须与已冻结TRAIN对应sidecar逐任务相同，不能使用未来目标恢复deadline。

## 搜索输出与执行

SearchOutcome新增实际 `winner_sequence` / `winner_first_action`，只保留已评分候选对象，不反解fingerprint、不增加抽样/rollout/打分；固定旧源码oracle验证旧离散输出不变。正式旧JSON序列化剔除此新增对象，历史结果不改写。

Route显式NOOP；先重新做原Grammar admission及真实ID/presence/eligibility/RB/CPU/UAV单位检查，再调用原生Comm/Comp/Traffic setters。只执行winner第一动作；H2–H4仅是预测候选。决策buffer在setter异常时恢复并终止episode，不推进env；env.step异常不尝试回滚世界、不重试。setter事务只在本次单线程暂停仿真、已审计三个原生buffer的范围成立，不宣称一般环境原子性。

正常路径原生env.step恰好一次，再采集真实O_t+1，更新真实action/outcome History，重新构建输入/graph/posterior/domain并调用第二次搜索；第二次不执行动作。

## 冻结fixture与验收

fixture在首次B64 outcome之前冻结：已知旧TRAIN MH K4rho0.1可评分记录中anchor最早者，frame3/seed6301。第一轮B64无H4则记录 `SUCCESS_PATH_NOT_OBTAINED_AT_SMOKE_BUDGET` 并停止，不能提预算或换样本。

接口、旧搜索不变性、真实反馈两轮、failure path、防泄漏、归一化/参数不变都需独立机器证据。未通过的接口不得以静态调用链、合成测试或目标图代替。最终状态见本轮acceptance，不由本合同自行宣称PASS。

Stage B/正式闭环性能/baseline/ablation/locked test均未授权；GPU禁用。
