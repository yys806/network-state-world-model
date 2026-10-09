# STEP 6.4I — Pure-search正式闭环CPU准备

## Step Goal / Definition Basis
按研究者2026-10-09请求，仅CPU审计、必要接口、真实反馈smoke和可审阅Protocol Draft。定义依据为本轮研究者决定及现有6.4B/F/H源代码、合同和accepted evidence；不把目标定义当实现事实。

## Initial State
fetch后HEAD=origin/main=10c70763c1aaf44b6be8ffa8b1051f11dd30dadc，tracked clean；既有无关untracked ZIP/TASK/plot脚本原样保留。6.4H预算资格PASS，但此前final预算待决定。本轮研究者正式接受512，保留历史receipt原值。

## Files Involved / Changes / Reuse
新增step6_4i_episode_v1：原子journal、仅第一动作、一次A否则C、一次step、重复token保护、partial setter与step异常硬停。episode_runner提供有界episode入口/批准门/固定决策分母，不自动恢复。real_metrics提供真实Task getter初始cohort草案。live_planner提供fresh history/state/graph/posterior/domain→S-CEM，CUDA复用旧CPU缓存桥但本轮未GPU验证。两个smoke脚本与只读audit脚本、四组tests。复用6.4B真实capture/执行、6.4F当前Comm资格、6.4Ecanonical A、冻结WM/rollout/scorer/S-CEM；protected科学源码与基线无diff。

## Validation — actual commands / outputs
- git fetch；HEAD/origin对齐；protected source diff empty。
- 默认D:/miniconda/python.exe首次preflight缺osmnx，尚未创建真实环境、无search/action；失败原namespace保留。不是科学失败后换样本重试。
- D:/miniconda/envs/airfogsim/python.exe import torch,osmnx,traci PASS；在独立r2 --freeze先冻fixture，--smoke PASS steps2；fallback --freeze/--execute PASS steps1。
- python unittest集中运行test_step6_4i*.py及6.4B/E/F、6.2B、6.3D、context/index/registry/query：73 tests in20.032s OK；provider独立2tests3.224s OK。
- python code/scripts/audit_step6_4i_cpu_preparation_v1.py PASS（不执行环境或WM）。
- final compileall、index write/check、Context Consistency Check、git diff --check结果见14_validation_receipt.json；收口时全部必须PASS。

## Results
STEP_6_4I=PASS（仅CPU正式闭环准备）；SEARCH_METHOD=S-CEM，K=4/rho=0.2；FINAL_CLOSED_LOOP_BUDGET=512，PLANNER_V1_CLOSED_LOOP_B_WM=512，H=4。研究者本轮明确接受6.4H所见Effort质量损失换取计算节约；B1024保留参考，不声称等价/最优，不冻结hybrid预算。Route=EXPLICIT_NOOP_ONLY；fallback=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次A失败即C，无重试）；同步暂停仿真、非实时。

独立CPU真实S-CEM机制：固定TRAIN anchor0003，engineering-only B64/batch1，两次winner第一步真实执行0.6→0.7→0.8s；每次实际64 unique transitions、13个不同可评分H4。第二root由新真实观测/History/后验重新构建，未复用预测root，模型与TRAIN normalization不变。固定TRAIN anchor0041注入NO_SCOREABLE_H4，canonical fallback A经同桥真实执行4.5→4.6s一次；不是搜索性能结论。74项focused CPU tests通过，覆盖空域、缺字段、bridge/setter/step异常、部分写入、重复决策保护、Return支持与Future Target poison。

CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_PROTOCOL_REVIEW；真实CPU链READY，通用episode组件已测试，live CUDA provider=IMPLEMENTED_NOT_GPU_VERIFIED。正式episode来源/数量/seed/长度、指标分母、timeout及baseline仍待研究者批准；GPU_LAUNCH_AUTHORIZED=false，READY_FOR_GPU_LAUNCH=false。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED/DEFERRED；GPU=NOT_USED；locked_test=false。

证据：docs/implementation_records/STEP_06_4I_PURE_SEARCH_CLOSED_LOOP_PREPARATION.md；docs/contracts/PIJWM_STEP_06_4I_FORMAL_CLOSED_LOOP_PROTOCOL_DRAFT_V1.md；code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json。唯一下一动作：研究者审阅Protocol Draft，批准最小GPU pilot与首个live CUDA资格检查。本轮停止。以下较早内容均为历史时点，旧pending不覆盖本轮明确接受预算决定。

## Expected vs Actual
符合CPU准备预期。normalS-CEM两cycle各B64、unique64/cache29、distinct scoreable13，只有winner第一动作执行；两次内部搜索约51.04/53.48s是CPU工程机制耗时，不是正式B512指标。fallback注入不冒充实际search失败。

## Metrics correction / provenance
真实Task getter可以读取因果arrival/deadline/finish/resultdata；初始cohort包含11at-riskTasks，短窗口不解释性能。原09的new-birth计数误将初始failedTask计新，原09不改、执行时旧metrics源码保存executed_metric_source.py.txt与原configSHA匹配；10独立用initial/freshIDs重算14→2。当前metric修复有回归测试，无action/search重跑。所有正式指标与分母仍DRAFT。

## Protocol Draft / cost / Known Issues
pilot建议2源episode×8决策；正式建议12源Validation轨迹×2search seeds×64决策（24episodes）。均未批准。6.4H离线B512内部耗时映射：pilot平均加工程余量约0.66h，P95/观测max情景1.33/2.34h；正式平均42.16h，慢状态情景106.22/203.47h，不是最坏上界。动态闭环须pilot重估。价格P未确认，费用=P×实际实例小时。存储估算0.04/4.01GiB、建议20GiB空间。RouteNOOP、Return支持、live slots、Effort损失保留；没有真实性能或实时claim。来源trajectory级统计/固定失败分母、公平baseline约束在Draft。CUDA provider未实际资格验证，不可据CPU PASS直接长跑。

## Git / Next Step
本Step最终Conventional commit+push main；最终SHA见Git提交（不构造自引用source身份）。当前12_interface_audit及16_inventory引用实际源码SHA，独立CPUexecution身份保留。下一步仅研究者批准pilot协议与live CUDA资格门，本轮停止；不启动GPU/StageB/baseline/locked_test。

实现边界补充：实际真实smoke执行的是固定脚本内的planner与EpisodeController；可复用LiveSCEMPlanner/generic run_episode只做组件与输入一致性测试，尚未整合运行真实正式episode，更未执行CUDA。首次index注册校验指出deferred status枚举须为deferred，已修正元数据并保留protocol待批准门，未放宽科学规则。

最终复核：指标report异常终止C计数补充test先RED（0≠1）、最小记录修复后集中74 tests/22.416s PASS；仅通用runner记录逻辑修改，无真实action/search重跑。Context/index同步后额外22 tests/1.995s PASS。
