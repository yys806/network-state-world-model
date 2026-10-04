## 2026-10-04 STEP 6.4D 全量B512最终验收（当前）

STEP_6_4D_FULL_COHORT_B512=PASS。完整64 anchors×5 seeds，B512覆盖320/320（48原6.4C结果按SHA引用、272新增），原Stage A B1024父结果320/320身份与SHA通过且未重跑。新增名义/实际独立一步转移139264，完整B512实际163840，B1024参考327680。两预算均160/320可评分H4（50%），六类配对0/0/30/109/21/160合计320，B512对B1024为30胜/181平/109负；64anchor×5seed cluster bootstrap95% CI=[-0.3375,-0.15625]。PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0，前三项N_DDL/A_DDL/J_Delay在双方可评分160对中均相同；首差J_Burden为B512更好1/B1024更好8，J_Effort为29/101，全等21。独特可评分候选15668/33505，B512减少53.24%；内部搜索mean57.641/119.746秒、median43.564/90.506秒，分别节省51.863%/51.866%。两预算困难anchor均32，交集32、差集0；future Return-birth固定支持限制保留，scorer exception/inconsistency=0。新增272总墙钟24692.314秒（6h51m32s），加载/准备开销单列。640raw独立核验、本地D:持久备份、新raw ZIP/SHA归档PASS；旧raw未复制/修改。GPU搜索已硬停止，研究者已在本聊天确认手动关机（Codex未取得UI关机核验）。SEARCH_METHOD仍MH-CEM纯搜索骨架，非最终hybrid冻结。证据支持将B512作为节省计算的预算候选，但完整词典序Objective有损失，不能声称等价、最优或真实闭环性能。CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。唯一下一动作研究者审阅完整预算取舍并决定预算，不自动执行后续实验。

证据：`code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/17_independent_acceptance_receipt.json`、`19_archive_acceptance_receipt.json`、`20_formal_runtime_receipt.json`、`21_instance_shutdown_receipt.json`。执行协议commit9b7f50e58d410e012d9b27ed04661e8c88191f2a；分析/文档收口属于后续独立commit，不覆盖历史运行source。

以下均为历史gate；较早RUNNING/未启动表述不是当前状态。

---

## Final Changes / Reuse / Validation / Expected vs Actual / Git

实际GPU272已完成，48复用与320父结果原路径保留；新增只读监控、标准库独立auditor及文档收口工具，不进入GPU科学source closure。独立auditor从640raw重建全部320pair/首差/counters/runtime/困难anchors及seed6316、10000replication bootstrap，逐项与runner一致。新增272raw+运行收据和log归档，291个entry逐项SHA通过；640原文件清单保留引用血缘。符合本Step完整扩样与配对验收预期；科学观察是计算节省与后两项目标质量损失并存，预算仍待研究者决定。只读SSH两次失活，原GPU进程未重启；最终新连接SFTP补齐59文件、runner已退出且log为PASS; STOP。computer-use因不能识别Edge URL被安全机制停止；研究者随后确认手动关机，此事实仅人类确认，未冒充UI验收。CPU回归/compileall/index/diff/Context一致性最终命令和输出见22_closure_checks_receipt.json。协议9b已push main；最终分析收口commit以Git历史定位，提交推送后停止。

## STEP 6.4D GPU运行（当前）

STEP_6_4D_FULL_COHORT_B512=RUNNING（尚未最终验收）。协议提交9b7f50e58d410e012d9b27ed04661e8c88191f2a已push，服务器fetch/ff到该精确commit，tracked clean及source/config/checkpoint/320父结果/48复用SHA核验PASS后才启动。执行身份1c3cd986dbda4559e94fa37013b2d70e96adf85911c44e88cc22cfe3f99fd843；RTX3080Ti CUDA FP32 batch16。正式start2026-10-03T13:38:52.692909+00:00；新增272 B512，旧48按SHA引用，B1024不重跑。进度以09_runtime_status及raw为准；已启用只读SSH/SFTP身份监控、逐case SHA核验与本地D:原子备份，无中途质量结论。监控脚本属于独立部署工具，不进入冻结科学source closure，不改科学运行源码。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false；Return-birth固定支持限制保留。唯一下一动作继续监控至272新case完成，再独立验收与Git收口；不启动B256/环境动作/其他实验。

证据：`code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/08_launch_acceptance_receipt.json`。以下为历史gate，当前状态以本段和runtime/raw为准。

---

# STEP 6.4D — Full-cohort MH-CEM B512 Expansion

## Step Goal / Definition Basis
研究者2026-10-03本聊天授权：完整既有64×5 MH-only B512，严格配对原Stage A B1024；只比较预算质量与计算开销，不选择方法或冻结闭环预算。MH-CEM K4/rho0.1、H4、Objective、CandidateDomain、Grammar、Return/cache/RNG/prior/service/checkpoint不变。

## Initial State
origin/main=cb19ca74064a76bb28b98b1fa126b444259676e3；tracked clean；6.4C PASS；三原有untracked保留；Stage B/正式闭环/locked test未运行。

## Files Involved / Changes / Reuse
独立helper `code/src/pi_jwm/step6_4d_full_cohort_b512_v1.py`；prepare/run/manage三个script；focused tests。仅新增预算执行、复用验收、统计与SSH SHA备份。既有solver/CandidateDomain/Grammar/Objective/WM与6.4C冻结全source closure逐字节一致。320父结果逐文件身份、StageA inventorySHA验证；48B512按6.4C已验收inventory SHA引用，不复制或改写原raw。新272，无B256或B1024模型forward。

## Validation — Actual Commands / Outputs
RED：新helper缺失导致import失败；GREEN：`python -m unittest discover -s code/tests -p 'test_step6_4d*.py'` 6/6 PASS(0.454s)，6.4C 4/4 PASS，6.3D 34/34 PASS(30.508s)。compileall PASS，部署脚本三处regex字符串产生非致命SyntaxWarning（正则运行正确，不属于科学问题）。prepare与CPU runner均PASS，64/320父配对和48复用完整，新增272、新raw=0。check source/checkpoint身份PASS。

## Protocol / Scientific Statistical Contract
01完整static manifest原序，不采样；seeds6311..6315。02冻结唯一预算B512、配对None六类合计320。64anchor clusters保持每anchor5seeds，frozen bootstrap seed6316/10000replications。每pair首差分量：N_DDL/A_DDL/J_Delay/J_Burden/J_Effort/all_equal，分别记录左/右更优；前三项先劣的loss定义PRIMARY_OBJECTIVE_DEGRADATION_COUNT，后两项loss定义BURDEN_EFFORT_ONLY_DEGRADATION_COUNT。每预算320cases完整H4/独特scoreable候选/unscoreable/Return/grammar/scorer/budget/cache/proposal/runtime。原raw in-solve runtime含mean/median/P90/P95/max；准备与matrix overhead另记。困难anchors全保留并比较交集/两差集。

## Results / Expected vs Actual
当前是协议/CPU门PASS；GPU未开始，48/320已验复用覆盖，正式完整结果待272新case。不写实验最终PASS或预算结论。

## Known Issues / Claim Boundary
future Return-birth fixed-support limitation保留；B512是否值得冻结须研究者判定。禁止等价/最优/真实闭环性能/系统收益声称。Stage B、B256、正式closed loop、baseline、ablation、locked test均未启动。

## Git / Provenance / Execution Gate
05独立执行身份绑定530源码/输入条目（确切数以config为准）、冻结checkpoint、64×5 manifest、320父证据与48复用。protocol-owning commit必须先push；为避免commit自身引用循环，精确commit通过--protocol-commit指定并在launch/runtime/result保存。服务器须fetch+ff至该精确commit、tracked clean；runner再次核验Git blobs逐源码与config SHA，CUDA3080Ti FP32 batch16。每case原子保存；错误resume/scorer/NaNInf/source drift拒绝并停止，不自动重启。最终commit由Git历史定位。

## Next Step
协议commit/push后同步精确commit并核验，执行仅剩余272 B512；完成后独立验收/归档/Context/Git收口并停止。
