# STEP 6.4E — 当前动作 fallback 接口（科研验收 BLOCKED）

## 研究者决定与范围

2026-10-04 本轮授权正式冻结 `PLANNER_V1_CLOSED_LOOP_B_WM=512`，MH-CEM K4/rho0.1、H4-only、同步暂停仿真不变。仅 Planner v1 pure-search；B1024 是高预算参考，不是等价/最优证明。6.4D 中观察到 Burden/Effort 质量损失，前三项未观察到退化；不推出闭环性能。原 Stage B 未运行，`DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE`。最终 fallback 仍 `RESEARCHER_DECISION_PENDING`。

## 接口

实现 `code/src/pi_jwm/step6_4e_fallback_v1.py`：

- `prepare_fallback(reason, candidate, domain, decision, runtime_tasks, behavior_offer=None)`：调用者必须显式指定本次审计候选，没有默认策略。输入为当前真实观测、由因果历史重建的当前 Domain 和当前任务句柄；拒绝有预测分支历史的 Domain、不同帧/时刻、必要观测缺失。没有世界模型评分、未来目标、预测未来 root 或未来仿真排程入口。
- A：消费 `domain.iter_bound()` 的第一个规范排序结果，沿用 mode/count/subset/row/RB-start 层级，不物化整个候选空间。用冻结 grammar 再准入，再用当前真实观测检查执行条件。
- B：只把已生成的同帧同时间 offer 的 Route 置空，Comm/Comp/Mob 不修补；再次准入并验证。真实现有 collector 尚无独立、不写 setter 的 current offer provider，不能因为投影接口存在就说 B 已接好。
- C：终止 episode，不调用 setter，不推进环境。
- `execute_fallback(...)`：统一复用 6.4B `apply_commands`，和 winner 使用相同三组原生 setter；本函数不调用 `env.step()`。成功后的真实一步推进应由获授权的 orchestration 显式执行。

Reason 覆盖 `NO_SCOREABLE_H4 / DOMAIN_EMPTY / REQUIRED_LIVE_OBSERVATION_MISSING / LIVE_SLOT_UNSUPPORTED / ACTION_BRIDGE_REJECTED / SETTER_FAILURE`。当前审计实现对除 NO_SCOREABLE_H4 外的关键错误均终止；没有级联尝试另一个策略。

## 失败语义

先完整验证，再写 Comm/Comp/Mob decision buffers。原生 setter 异常恢复 RB 字典、CPU callback、UAV pattern 三个原缓冲，停止 episode，不重试、不 step；这不承诺任意外部 setter 副作用可全局回滚。测试注入 setter 失败证明时间不变和三缓冲恢复，不能表述为真实复杂环境任意失败的原子性证明。

## 真实执行阻塞与结论边界

冻结 TRAIN `formal-v1-sim-2026092300-policy-2026092400::anchor-0041`，当前 t=4.5s。按 dev_train 轨迹 ID / 帧升序，只依据当前 Flow 与 Domain 合法性选择，在任何新执行前冻结两个独立一步场景。Domain 给出 Task_1/Task_6/Task_24 的无线关系；Task_1、Task_6 的当前真实 lifecycle 已为 failed，Task_24 为 offloading。首个非空 Comm 为 Task_1→RB0，grammar 准入但 live bridge 拒绝。没有本轮动作 setter 或授权 env.step；为重建 fixture 执行的历史 41 步不算本轮一步成功。第二个 fallback 一步场景未启动，没有换样本、换动作或重试。

当前 `_wireless_bindings` 检查已知/存在/活跃 Flow、task presence、合法无线关系，但不检查 Task lifecycle；因此不能把 Flow 活跃直接解释为真实任务仍可执行。此问题没有通过修改 CandidateDomain、mask、Objective 或 Return semantics 来隐藏。记录状态为 Awaiting Researcher Decision。

`FALLBACK_INTERFACE=READY_CPU_CONTRACT`；`REAL_NONEMPTY_COMM_EXECUTION=BLOCKED`；`CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED`。A 当前 replay 可准备合法空 Comm 动作，但本轮尚无它真实执行一步的证据；B provider 缺失；C 无动作终止路径通过。推荐仅为未来完成一致性修复与真实执行证据后：NO_SCOREABLE_H4 考虑 A，关键错误考虑 C；不是研究者选择。

## 验证与证据

固定 receipt 位于 `code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/`。`01` 冻结 fixture；`01b` 仅记录环境导入前控制台 GBK→UTF-8 的工程续接身份，原 fixture/动作不变；`03a` 保留创建环境前的编码错误，无真实动作重试。`04` 原始拒绝回执；`06` Domain/bridge 冲突；`07` 空域/缺字段/bridge/setter 负路径；`08` 三候选比较；`09` 预算与保护源码；`10` BLOCKED 验收。

CPU tests、Future Target poison/current input 不变、保护源码字节比较及历史实验文件不变是本轮证据；没有新 World Model forward/search transition。GPU 未用，Stage B 未启动，正式闭环性能未启动，locked_test=false。
