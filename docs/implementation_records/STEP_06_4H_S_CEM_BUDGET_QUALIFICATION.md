# STEP 6.4H — S-CEM 在线搜索预算资格：正式收口

## Step Goal / Definition Basis

研究者明确授权修复域S-CEM K4/rho0.2仅新增B512完整64×5，与6.4G A既有同方法B1024严格配对。预注册资格门为同可评分集合、前三项首差劣化0、scorer0、mean/median内部搜索耗时均降低及身份预算完整；不据此声称等价、最优或真实闭环收益。

## Initial State / Files Involved / Changes / Reuse

CPU协议freeze dc0e6c94b2353bdaf1c671e52bde2a978a991bf8；研究者随后开机并授权续跑。RTX3080Ti/CUDA/FP32/batch16，config5cf6685959ad8847150c1680ba55c399102858611e5c2d955fead8f32f6c46bf，checkpoint941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9。父320逐SHA与ZIP lineage完整，不重跑。无旧runner、tracked clean、43.8GiB可用容量及launch gate PASS。远端fetch超时以SHA验证的增量Git包精确快进，未改源码或实验定义。唯一PID1615于2026-10-08T03:50:21.905759UTC启动，2026-10-08T15:40:34.197641UTC自然结束，不重启，不运行其他实验。

科研solve/CandidateDomain/Objective/WM/参数/RNG/cache文件与GPUfreeze完全不变。新增本地只读远端管理helper、正式small receipts、结果registry数值验收分支及反例测试；更新AI_CONTEXT00–08、权威/过程/导航索引。所有raw/ZIP留在D:原namespace，不提交大归档，不覆盖历史6.4G和MH证据。

## Validation / Results / Expected vs Actual

python code/scripts/step6_4h_s_cem_budget_v1.py --verify-local退出0：从320新raw和320父raw独立重建统计，ZIP逐文件SHA及接受receipt绑定PASS，local_backup_acceptance.json PASS。新结果实际转移163840=名义320×512，320身份完整，无重复/缺失/混GPU。资格所有门PASS，符合预注册资格预期，但完整Objective有Effort损失，必须报告。

六类（only512/only1024/both512better/both1024better/bothtie/neither）0/0/11/76/73/160，总和320；win/tie/loss=11/233/76。bootstrap以64anchors为整组、每组保留5seeds，10000次seed6316，CI[-0.290625,-0.11875]，mean=-0.203125。双方160scoreable首差：N_DDL/A_DDL/J_Delay/J_Burden均0；Effort11好/76差，全等73。PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0。完整配对整体优势为负，不称两个预算等价。

| 指标 | B512 | B1024父结果 |
|---|---:|---:|
| scoreable cases | 160/320 | 160/320 |
| complete H4 | 96373 | 375172 |
| distinct scoreable H4 | 30209 | 72587 |
| unscoreable/Return boundary | 49845 | 162239 |
| grammar dead-end | 19882 | 46621 |
| scorer exception/inconsistency | 0/0 | 0/0 |
| unique transitions | 163840 | 327680 |
| cache hits | 287932 | 1289621 |
| proposed/admitted/rejected | 462461/462461/0 | 1625748/1625748/0 |
| runtime mean/median seconds | 74.548/51.416 | 175.249/126.359 |
| runtime P90/P95/max seconds | 140.693/194.663/376.998 | 343.668/435.796/1079.521 |
| transitions/sec in-solve | 6.868 | 5.843 |

平均/中位内部搜索时间减少57.46%/59.31%。矩阵总elapsed42613.116s，内部23855.506s，case加载setup10899.799s，其余核验持久化7857.811s。模拟0.1s不是wall-clock deadline。32hardanchors集合不变，future Return-birth限制保留。

## Archive / Known Issues / Claim Boundary

raw ZIP SHA256=8133c9707454a4c279be3007fd918937e60dce1d46d49d5e8fa39996fca38a5f；acceptance SHA256=939ab785b4a96610ed1c270a8e0cd378de657065d9e65bda18a654963bdb3df7。D:持久原目录与inventory逐文件hash可验证。原runner已停止，备份验收后告知研究者可以关GPU，云电源状态由研究者处理。本轮不再需要GPU。

RECOMMENDED_CLOSED_LOOP_B_WM=512仅为工程/研究建议；FINAL_CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING。正式闭环NOT_STARTED，hybrid NOT_FROZEN，Stage B未运行/DEFERRED，6.4G MH PhaseB条件停止不变，locked_test=false。待研究者接受预算取舍，不能把模型预测指标写成实际闭环性能。

## Git / Next Step

收口测试、compileall、index write/check、diff check与Context Consistency Check见08_closure_validation.json。提交并push main后停止。唯一下一动作：研究者决定S-CEM pure-search最终闭环预算。

---

## 以下是此前CPU gate历史记录

# STEP 6.4H — S-CEM 在线搜索预算资格（CPU gate）

## Step Goal

准备唯一320-case S-CEM B512矩阵，与已验收B1024严格配对。本轮完成CPU门后停止，等待研究者开GPU；不运行搜索或真实环境动作。

## Definition Basis / Initial State

研究者2026-10-08显式授权及末尾CPU-only指令。HEAD=origin/main=aea391f280e5933898cf1804d06179c0d9ec5f2f，git fetch成功，tracked clean；已有未跟踪参考ZIP/TASK/plot脚本保留。6.4G T/A accepted，选S-CEM K4/rho0.2，MH专用B条件停止。

## Files Involved / Changes / Reuse

新增code/scripts/step6_4h_s_cem_budget_v1.py、code/tests/test_step6_4h_budget_v1.py、独立protocol/manifest/config/parentrefs/CPU receipts、合同及本记录。复用已测试的科学solve、6.4G validate_result/atomic、6.4D预算配对与首差统计、6.4G独立数学oracle；不调用或修改旧Phase B条件门。所有code/src/pi_jwm科学文件不变，旧6.4F/6.4G/旧MH结果不变。同步AI_CONTEXT00–08、权威/过程/导航及experiment/question registry。

## Validation

先加测试并确认缺新模块时失败，再实现。新增资格反例、320六类统计/目标首差、完整cohort约束、resume身份/NaN/scorer/预算拒绝、raw summary oracle tests。真实B1024父文件320/320及ZIP内逐SHA、accepted manifest/source/config/checkpoint/模型参数验证PASS。CPU不运行WM搜索，未来字段poison回归使用既有因果bridge/Comm资格测试。

最终focused CPU tests、compileall、knowledge-index write/check、Context Consistency Check、git diff --check输出见05_cpu_validation.json；GPU运行门尚未执行，不把CPU PASS写成GPU运行PASS。

## Results / Expected vs Actual

符合本轮CPU准备预期：320父raw PASS，B512=0/320，READY_FOR_GPU_LAUNCH=true。资格NOT_RUN、final budget待研究者决定。所有性能、scoreability差异和CI待GPU320cases后统计，当前不填预测数值。源身份由03 config绑定；精确Git freeze commit以本次commit为准，启动强制HEAD/origin/main相同。

## Known Issues / Boundaries

future Return-birth fixed-support仍保留。远端GPU型号/容量/无旧runner须续跑前核验，当前未连接/开机。正式闭环NOT_STARTED，hybrid NOT_FROZEN，locked_test=false。预算资格不代表真实闭环性能；0.1s模拟步不是GPU墙钟期限。

## Git / Next Step

本门测试通过后提交并推送main；commit本身为protocol-freeze gate，无自引用SHA。唯一下一动作：研究者开启原RTX3080Ti实例并通知继续，再做远端精确commit/身份/容量/runner检查；不擅自开关服务器。
