# STEP 6.4G — Repaired-domain formal search requalification

## Step Goal / Definition Basis

研究者最新粘贴文本授权整个pure-search链：完整新TRAIN→完整新A→条件新B与fallback冻结。本轮替代6.4F的1140case targeted proposal，不复用7未变anchors的旧结果。docs/contracts/PIJWM_STEP_06_4G_REPAIRED_SEARCH_REQUALIFICATION_V1.md为工程合同。

## Initial State

HEAD=origin/main=1f2b9fafeeb00251383ae4551a78b1a9ef5bc48a，tracked clean。6.4F机制PASS但旧搜索需重新资格；服务器此前用户确认关机，当前只读SSH认证实际发现RTX3080Ti空闲，remote仍旧6.4D协议commit9b7f50e58d410e012d9b27ed04661e8c88191f2a，无search进程。本轮未操作开机。

## Files Involved / Changes / Reuse

新增6G phase/fallback模块、CPU冻结脚本、单phase runner、独立raw统计/本地备份auditor、SSH交互probe/deploy/launch/snapshot、meaningful tests。复用6F repaired predicate/3B grammar/3C Domain/3D solver/proposal/method-selection/6E合法动作准备与winner执行桥；这些既有源码不修改。支持目录完整TRAIN4416复算全部joint/start-width/task counts/temporal counts等同原值，保持原定义不重建。

## Validation (actual commands / outputs)

- TDD先运行6G tests因缺模块失败，实施后通过；runner synthetic六类test揭示summary调用少budget参数，在GPU前修复新orchestration，既有科研实现不改。
- focused 6G/6F/3B/3C/3D/6B/6E：67 tests in42.196s，OK；其中旧FallbackTests被import，计数如实包含其重复执行。
- CPU protocol preflight：原32/64固定anchor所有current domain非空、原全量静态source/input SHA再次匹配；checkpoint SHA和normalization/sample packages明确冻结。早版CPU draft原样保留，r2在protocol commit前补充输入与本地备份父链，未写任何正式solve result。
- 新phase identity/resume测试拒绝phase/source/device/checkpoint/seed/K/rho/batch/eligibility/support等错误；NaN/scorer/quota故障在保存前STOP；synthetic六类sum320和独立bootstrap/selection一致。不是正式Validation结果。
- compile/index/diff/Context命令及摘要见06_cpu_checks_receipt；全部通过后才能protocol commit/push/remote ff/GPU launch。

## Results / Expected vs Actual

STEP_6_4G=IN_PROGRESS；CPU preflight/协议冻结PASS，完整旧TRAIN行为支持catalog复算一致，无需重建；原32/64anchors在修复域均非空，未替换。Phase T NOT_STARTED（须精确Git/device/source gate才开GPU），目标768；Phase A NOT_STARTED（目标960），Phase B NOT_STARTED（仅新A选MH才320）。旧调参/选法/预算均historical-under-pre-6.4F-domain；不复用任何旧raw，包含先前7不变量anchors也完整重跑。FINAL_FALLBACK_POLICY=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1，研究者本轮明确冻结：NO_SCOREABLE_H4且当前域/输入完整才一个canonical A，A任何失败及空域/缺观测/slot unsupported/bridge/setter失败均C终止，不级联、不换动作、不重试。SEARCH_METHOD_REQUALIFIED=PENDING，S/MH新K/rho=PENDING；旧MH/K4rho0.1/B512只为历史/working决定，修复域最终预算待T→A→条件B gate。B512必须scoreability set一致、前三项目标劣化0、scorer0、mean/median耗时更低和严格budget。FORMAL_SEARCH_REQUALIFICATION=PENDING；CLOSED_LOOP_PRE_FORMAL_READINESS=PENDING_SEARCH_REQUALIFICATION；READY_FOR_FORMAL_CLOSED_LOOP_PROTOCOL=false。3080Ti只读认证核验空闲、11912MiB free、tracked clean，无其他搜索进程；未启动实例，不换4090。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN，Stage B=NOT_STARTED / DEFERRED，locked_test=false。唯一下一动作：commit/push协议后服务器fast-forward到精确commit，重复preflight后只启动PhaseT。

证据：`code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004/00_protocol_r2.json`、`01_support_catalog_semantic_audit.json`、`02_frozen_anchor_manifest.json`、`04_CPU_preflight_receipt_r2.json`；实现记录`docs/implementation_records/STEP_06_4G_REPAIRED_DOMAIN_FORMAL_SEARCH_REQUALIFICATION.md`。新目标evidence_class=repaired-domain formal evidence，尚无新正式科研结论。

## Known Issues / Stop gates

真正新K/rho、选法、预算结论PENDING；不能宣称67测试或source一致即正式requalification PASS。Return-birth fixed-support边界保留，hard anchors不得删除。B512 gate失败待决定，A非MH停B。长GPU期间源码异常、crash/scorer/身份异常停止不修算法不自动重启。

## Git / Execution identity

protocol commit通过`git log -- docs/implementation_records/STEP_06_4G_REPAIRED_DOMAIN_FORMAL_SEARCH_REQUALIFICATION.md`与06/07 gate receipts查询；scientific hash绑定Git LF，阶段间只提交新的已验收receipt与docs。每phase实际config独立保存于T/A/B execution_config.json；case gate commit不冒充旧3B/6C/6D执行身份。

## Next Step

Protocol commit+push，服务器ff该精确commit并重复preflight，才T；不可达则STOP。
