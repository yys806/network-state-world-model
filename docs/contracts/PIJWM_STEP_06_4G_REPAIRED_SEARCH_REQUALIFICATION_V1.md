# STEP 6.4G — 修复域正式搜索重新资格协议

## Researcher Decision / Fixed scientific boundary

完整TRAIN32×3 seeds×2 CEM×4(K3/4,rho.1/.2)×B512=768；原tune_cem_config规则success/paired net/smaller K/rho。独立验收并冻结新S/MH config，commit/push后才A。

A完整64×5×3方法B1024=960，原anchor-cluster bootstrap10000/seed6316/95% lower>0选法；六类互斥计数每组320，独立oracle重算与simultaneous diagnostic不改变主规则。不得预设旧MH结论。新选择非MH立即STOP，B不运行，闭环预算待研究者。

仅新MH获选才B51264×5=320；配对本轮A320 fresh MH B1024，不复用任何旧case。B gate：320身份完整、scorer0、scoreability set完全一致、PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0、mean/median搜索耗时均降低、严格预算。J_Burden/Effort可损失但须报告，失败budget PENDING，不自动改1024。

WM/checkpoint/Dataset/normalization/Objective/Return/Route NOOP/H4/cache/RNG/comparator/Comm修复规则不改。TRAIN support catalog定义是behavior observed structures，与current legality不同；只读全量4416行为独立复算一致，保持原catalog。

## Fallback formally frozen

CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1。仅NO_SCOREABLE_H4且输入完整/当前域非空允许A一次canonical lazy choice。其他列举reason直接C；A preparation/admission/bridge/setter失败直接C，不级联、不改候选、不H3、不behavior fallback。current real only，RouteNOOP，原winner bridge；setter不原子风险只终止不重试。未来闭环必须记录trigger count/rate、原因、A execution和C termination；本轮没有闭环性能。

## Provenance / execution / resume

最终协议00_protocol_r2.json；早00仅CPU draft从未执行，原样保留并明确superseded。每phase config由协议SHA、全部冻结源码、checkpoint、eligibility/support/objective hashes及新父receipt SHA确定。T config已知；A/B身份在新选参/选法后按预冻结公式构造，不能用旧configs冒充。case identity涵盖phase/anchor/seed/method/K/rho/budget/device/FP32/batch16/source/checkpoint/eligibility/catalog/parents；错误resume在model forward前拒绝。GPU执行原始源码bytes必须等于Git LF SHA，CPU canonical hash不表示GPU资格。

每phase启动HEAD=origin/main exact phase gate commit且tracked clean；commit源码blob逐项匹配协议。阶段间只改receipts/docs，不改科研源码。fcntl排他锁防止重复runner；逐case原子保存，停止异常写STOP receipt不重启。完整matrix和时间诊断保留；低scoreability不改变矩阵/选法。没有额外免费forward，不改batch。

## Persistent backup and acceptance

每phase原raw保留，inventory/per-file SHA、ZIP/SHA回读、independent raw/math recompute；SFTP本地D:备份，`audit_step6_4g_phase_v1.py --verify-accepted-local`重新从本地raw复算并出local_backup_acceptance。下一phase parent强制包含该SHA及已提交新selected receipt。remote临时盘不冒充D:备份。

## Scope

不跑B256/旧三方法StageB/正式closedloop/performance/baseline/ablation/retraining/hybrid/locked_test。GPU若不可达，CPU协议commit后STOP READY_FOR_GPU_LAUNCH=true。真实GPU资源与软件/disk/checkpoint/source结果数量须重复核验，不能只看SSH网关。
