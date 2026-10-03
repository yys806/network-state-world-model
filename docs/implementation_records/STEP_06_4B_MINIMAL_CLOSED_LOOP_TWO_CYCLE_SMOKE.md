# STEP 6.4B — MINIMAL CLOSED-LOOP INTERFACE + TWO-CYCLE REAL-FEEDBACK SMOKE

## Step Goal / Definition Basis / Initial State

研究者2026-10-03授权CPU-only最小接口和两轮真实反馈机制；fetch后HEAD=origin/main=f9b7c05fb016403b776065074de8d791126661f0，tracked clean，三个既有untracked保留。原6.4A readiness BLOCKED，Stage A已选择MH-CEM，仅pure-search backbone；GPU服务器关闭，不联系/启动。Definition06只读依据沿用6.4A的§1.1/5.1/5.2及文件SHA，不写私人笔记；本轮用户明确冻结同步暂停与smoke-only失败即终止。详见合同。

## Files Involved / Changes / Reuse

新增限定于本轮证据目录的.gitattributes -text规则，避免Git换行转换破坏已执行证据SHA；新增live bridge、CPU smoke runner、独立CPU inference/provenance audit、七项live tests、合同与本记录。Solver仅增加实际winner对象输出，formal JSON剔除新增字段；collector增加策略RNG/任何setter之前的decision hook。原CandidateDomain/Grammar/Objective/RSSM/rollout/comparator/normalization fit/参数/预算定义/Krho/checkpoint不改。旧Stage A/TRAIN config、raw、05/06/07/08 receipts不覆盖；历史config对新source继续严格拒绝。

复用真实observer、causal Flow ledger/当前projection、正式TRAIN统计、tensor/typed graph、frozen encoder/current posterior、MH-CEM、原生Comm/Comp/Traffic setters、原环境step与transfer/outcome聚合。没有复制环境动力学、伪造target或使用预测root。对历史loader统计重算仅作既有TRAIN身份验证，live值绝不参与fit，校验后与checkpoint frozen identity完全一致。

## Validation（实际执行）

- 新winner assertion首先RED（旧SearchOutcome无winner属性），随后旧源码f9b7c05和新源码在三方法×batch1/4小型精确H4 fixture比较，除新增winner和墙钟外整个离散SearchOutcome完全相同。
- `python -m unittest discover -s code/tests -p test_step6_4b_live_bridge_v1.py`：7/7 PASS，约4.6秒。
- `python -m unittest discover -s code/tests -p 'test_step6_3d_*.py'`：34/34 PASS，21.849秒；原历史config正例显式mock其历史source，另验证当前新source会拒绝该旧config，未放宽runner gate。CLI error/Stage-A STOP日志来自合成测试，非新Validation。
- `D:/miniconda/envs/airfogsim/python.exe code/scripts/run_step6_4b_two_cycle_smoke_v1.py --recover-pre-action-receipt`：PASS，真实AirFogSim/SUMO，两轮、仅一次Planner环境step。
- `D:/miniconda/envs/airfogsim/python.exe code/scripts/audit_step6_4b_closure_v1.py`：Independent input/posterior/provenance acceptance True；只做CPU inference，0新search/0 rollout transitions。
- compileall、index write/check、diff check和Context Consistency Check见12检查收据。

## Results / Expected vs Actual

STEP_6_4B=PASS，TWO_CYCLE_REAL_FEEDBACK_SMOKE=PASS，CLOSED_LOOP_MECHANISM_READINESS=PASS。固定TRAIN fixture、MH-CEM K4/rho0.1、CPU FP32 batch1、B64：仅执行首轮winner第一动作，仿真0.6→0.7秒，真实新观测重建state/graph/current posterior并完成第二次规划；下一轮root不是上一轮预测。两轮各64独特转移（合计128）；另保留一次动作执行前回执异常的64次尝试，实际总计192。live tensor/deadline、旧搜索不变性、防未来泄漏及参数/归一化不变全部PASS。SEARCH_METHOD=MH-CEM仍仅pure-search backbone；READY_FOR_BUDGET_CALIBRATION=true只是机制就绪，不是运行授权。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN；future Return-birth fixed-support limitation保持。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并单独授权预算校准协议，不自动运行。

首轮0.6秒，root state b0e35b42…/graph97e734f5…/latent42329d2f…；首动作Task_8在RSU_0分配CPU 2.5843526422977448，两个UAV HOLD速度0/角度0.05，Route NOOP，Comm空。只执行此一步，不执行预测H2-H4，不让采集behavior同一步覆盖。真实反馈后0.7秒，root state f2f46224…/graph67d2e829…/latent36c01d7d…，不同于第一轮预测H1 state b82fa57f…；第二次规划成功，不执行其动作。每轮64 unique、30 cache hits、18完整/16 distinct可评分，墙钟28.358/28.588秒，仅机制诊断。

符合预期：真实反馈闭合、当前posterior重新初始化、fresh domain、暂停与第一动作边界、deadline和未来毒化门全部PASS。不能推出闭环收益、realtime或B64在线预算充分性。

## Engineering preflight / failure preservation

输入预验最初的padding类别/索引与CPU observed_mask差异按原冻结合同修复，尚未搜索；UTF-8输出配置和airfogsim环境psutil7.2.2是工程依赖修复。首次实际B64找到16可评分候选后，回执frozenset序列化失败；EpisodeStopped又被collector异常包装覆盖原原因。发生在回执/任何setter之前，episode结束、环境未推进。修复仅JSON序列化及停止控制（BaseException绕过历史异常包装），保留preflight_attempts原SHA证据，同fixture/seed/预算显式恢复，首轮root/目标/指纹/完整与可评分计数完全一致。两轮机制128，加已中止一次64，实际共192；绝未提预算或换样本。成功后runner拒绝自动再跑。

## Known Issues / claim boundary

单一预注册TRAIN fixture的机制验收；真实Comm为空，非空Comm映射仅合成因果CPU测试，不能声称真实所有动作/状态已验收。setter本身不是跨组件事务；wrapper在单线程暂停、原生setter只改三个buffer的边界完整预验/恢复，env.step前没有外部世界部分执行。未知第三方副作用/并发或env.step异常不保证回滚，必须终止且另行BLOCK。future Return-birth限制、final fallback、warm start/learned proposal/Route扩展、hybrid/正式performance均未解决或未授权，不属于本轮机制结论。

## Records / Git / Next Step

证据目录：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003`，包含fixture/source identity、live parity/deadline、winner/action桥、两轮、failure、leakage、搜索不变性、最终acceptance与SHA inventory；本地原始TRAIN/Stage A/raw继续保留。AI_CONTEXT00–08、authority三文件、process/index/registry同步。Commit由本记录所属Git提交确定；常规commit+push origin/main，推送状态以最终汇报为准，避免自引用SHA。

唯一下一动作：研究者审阅本机制证据并单独决定预算校准协议。完成停止，不自动Stage B/GPU/正式闭环/baseline/ablation/locked test。
