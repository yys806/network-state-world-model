# STEP 6.4F — 当前通信资格合同

## Researcher Decision / Definition Basis

本轮研究者明确冻结 E_comm(t) 为有效唯一活跃无线Flow绑定与当前可通信Task生命周期的交集，仅动作资格修复。Task必须存在，lifecycle只能是offloading/transmitting；若task_completed适用则必须为false。failed、completed、unknown、缺失lifecycle均不得产生新Comm。旧Flow可留在模型状态，不删除Ledger对象。

## Source / Inputs / Outputs

`step6_4f_comm_eligibility_v1.py::comm_current_task_eligible`为raw-live与tensor共享Task谓词；`tensor_comm_task_eligible`解析冻结vocab。`step6_3b_candidate_grammar_v1.py::_wireless_bindings`保留原unique Flow、known/presence、current carrying hop active、wireless relation validity及端点一致规则，再加共同Task谓词。`step6_4b_live_bridge_v1.py::validate_command`同时验证当前tensor绑定及raw lifecycle/completed，关系必须匹配，不允许旧行为覆盖。

只用当前state/raw observation与因果History，不使用Future Target。缺失/无效字段fail closed；不放宽grammar/support。Comp、Mob、Route=EXPLICIT_NOOP_ONLY保持原代码。Dataset/Tensor/normalization/FlowLedger/WorldModel/checkpoint/Objective/CEM/RNG/B_WM不改。B512是研究者working-budget，不因本修复声称旧预算实验已验证新域。

## Failure / Fallback

当前无资格Task的Comm不能生成，显式外来非法动作bridge拒绝。域空、缺字段、非法bridge和setter异常维持6.4E smoke-only fail closed；setter不原子时有partial mutation风险，停止episode、不静默重试。候选A对非空合法域采用层级canonical排序，不评分不物化全集，并走winner同一个bridge；候选B仍无独立current behavior offer provider；C为不执行动作终止。最终fallback由研究者决定。

## Scientific evidence boundary

根域未改不自动证明未来域未改。重新验证范围与7个充分不变量证书见07 receipt：不活跃Flow不能被Route NOOP重新生为active；活跃任务return requirement未知且return slot=-1时，原固定支持规则不可能令其completed，因此新的Task过滤不会影响任何H1–H4分支。其他57个Validation anchors需重新验证旧winner/ranking；raw没有完整trace。合成旧rule oracle证明completed+activeFlow确实可能存在，不能只审计根域。

## Validation

53 CPU tests、TRAIN4416/Validation1104及32/64anchor同输入pre/post静态审计、两个预冻结TRAIN真实一步执行、future poison、旧raw SHA/历史6.4E字节保护。PASS仅为资格/执行机制，不是Planner性能证明。本Step未GPU、搜索或locked_test。
