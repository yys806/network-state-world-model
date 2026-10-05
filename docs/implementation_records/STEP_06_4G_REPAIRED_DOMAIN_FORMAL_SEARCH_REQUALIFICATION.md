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

STEP_6_4G=IN_PROGRESS；Phase T=PASS，768/768全部新结果独立验身份/预算/选参与本地D:归档通过；名义/实际unique一步转移393216，正式矩阵32.5955小时。修复域新冻结 S-CEM=(K4,rho0.2)，MH-CEM=(K3,rho0.1)，两方法四配置均48/96 H4可评分；这只属于TRAIN选参观察，不能据此选择搜索方法。scorer exception/inconsistency=0。Phase A=NOT_STARTED（新960 B1024）；Phase B=NOT_STARTED（只有新A选MH才320 B512）。SEARCH_METHOD_REQUALIFIED=PENDING，FORMAL_SEARCH_REQUALIFICATION=PENDING，修复域最终budget待新证据；旧MH/K4rho0.1/B512只在historical-under-pre-6.4F-domain/working决定范围保留，不复用任何旧raw。FINAL_FALLBACK_POLICY=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1；只尝试一次canonical A失败则C，不重试。future Return-birth fixed-support限制保留。CLOSED_LOOP_PRE_FORMAL_READINESS=PENDING_SEARCH_REQUALIFICATION，READY_FOR_FORMAL_CLOSED_LOOP_PROTOCOL=false；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN，旧Stage B=NOT_STARTED / DEFERRED，locked_test=false。唯一下一动作：T阶段Git门完成后，同步服务器至精确新commit并启动PhaseA；科学源码SHA不变。

证据：`code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004/T/acceptance.json`、`T/local_backup_acceptance.json`、`T/selected_configs.json`、`T/inventory.json`、`T/archive.json`；新证据=repaired-domain formal evidence。

## Known Issues / Stop gates

新K/rho已从修复域完整TRAIN冻结；新选法和预算结论PENDING；不能宣称67测试或source一致即正式requalification PASS。Return-birth fixed-support边界保留，hard anchors不得删除。B512 gate失败待决定，A非MH停B。长GPU期间源码异常、crash/scorer/身份异常停止不修算法不自动重启。

## Git / Execution identity

protocol commit通过`git log -- docs/implementation_records/STEP_06_4G_REPAIRED_DOMAIN_FORMAL_SEARCH_REQUALIFICATION.md`与06/07 gate receipts查询；scientific hash绑定Git LF，阶段间只提交新的已验收receipt与docs。每phase实际config独立保存于T/A/B execution_config.json；case gate commit不冒充旧3B/6C/6D执行身份。

## Next Step

只读监控T、SHA备份至768；独立验收+Git后才新A，依序执行授权链。

## 2026-10-05 Phase T independent closure

原始768、归档ZIP逐文件SHA、模型parameter digest、current source/input SHA、原tune_cem_config与独立数学oracle一致；原结果不覆盖。正式运行2026-10-04T05:12:19Z至2026-10-05T13:48:00Z，32.5955h。S4/rho.2、MH3/rho.1；八配置各48/96 scoreable，不能推断方法相同。

实际CPU命令：python code/scripts/audit_step6_4g_phase_v1.py --phase T --verify-accepted-local → PASS/raw_count768；原67 focused unittest → 67 tests in43.626s OK；python -m compileall -q code/src code/scripts code/tests → exit0。index write/check、Context与diff在本阶段Git前执行。SFTP 10054仅备份连接恢复事件，原GPU无重启且自然完成；第二次会话56512已正常结束并SHA补齐。下一步新A960，T source保持原冻结；A身份由NEW T四父收据SHA构造，不能沿旧execution config。
