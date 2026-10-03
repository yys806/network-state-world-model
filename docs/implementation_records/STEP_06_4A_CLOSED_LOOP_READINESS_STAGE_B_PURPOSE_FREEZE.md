# STEP 6.4A — CLOSED-LOOP READINESS & STAGE-B PURPOSE FREEZE

## Step Goal / Definition Basis / Initial State

2026-10-03研究者授权：CPU/静态审计当前真实闭环距离、fallback、在线代价及Stage B用途/成本。不实现闭环，不启动服务器/GPU/Stage B，不改科研定义。初始fetch确认HEAD=origin/main=afbd7c82bfd549df2e64a5f16be15c1a29741e67、tracked clean；既有三个untracked保留。Stage A已PASS，960/960，MH-CEM K4/rho0.1选为pure-search backbone，非最终hybrid planner。Stage B未运行、locked_test=false；服务器关闭为研究者报告，本次不SSH核验或开机。

只读Definition06 `D:/shen/OB/科研/PIJWM/06策略器与候选动作规划.md` §§1.1、5.1、5.2：只执行winner第一步，下一轮必须用真实反馈重新构建state/graph/latent；规则＋learned proposal＋rollout＋objective是目标结构。文件SHA在06收据，不复制私人笔记。

## Files Involved / Changes / Reuse

新增CPU审计脚本 `code/scripts/audit_step6_4a_closed_loop_readiness_v1.py`、两项focused测试、当前记录、Stage B用途合同及01–07机器收据。原16项科研source SHA、960 raw/Stage A收据、TRAIN/checkpoint/参数/anchors全部保持。只复用已有原始结果，不调用搜索/模型/仿真运行；CPU grammar fixture只绑定动作，没有rollout或env.step。

证据目录：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/`。01含每段文件、函数、行号、源码SHA、缺接口和科研决定标记；02为fallback及CPU反例；03为320逐case耗时和区间来源；04为Stage B三方案；05为smoke要求；06为判定；07为SHA清单。

## Results — 闭环链路（7 IMPLEMENTED / 3 PARTIAL / 5 MISSING）

这些状态描述局部代码能力，不表示整个闭环已实现：

| 链路 | 状态 | 真实路径 / 缺口 |
|---|---|---|
| 真实O_t采集 | IMPLEMENTED | `run_step2_3_real_airfogsim_raw_contract_finalization_v1._capture`；包含隔离的simulator内部未来审计字段，必须白名单剥离后入planner |
| History→state/graph/latent | PARTIAL | `prepare_anchor`调用encoder并初始化posterior，`build_state`已有；缺实时history-only tensor/slot/normalization/sidecar桥 |
| 输入sample构造 | PARTIAL | `model_ready_sample_contract_v1.build_sample`需要future_steps，不能实时使用未来标签或伪造target来凑离线接口 |
| CandidateDomain | IMPLEMENTED | `CandidateDomain.from_state`；真实domain可为空，观测缺失不猜值 |
| MH-CEM、H1–H4、Objective | IMPLEMENTED | `solve_fixed_budget`、`rollout_one_step_batch`、`score_candidate_set`；H4/Return边界保持 |
| winner动作序列 / 第一动作 | MISSING / MISSING | SearchOutcome只返回目标/指纹和统计，没有candidate.steps；指纹不是可执行动作，需另行授权输出桥及离散不变性验证 |
| 动作编译成仿真命令 | PARTIAL | `compile_candidate`输出模型action tensor，不是AirFogSim setters；真实执行器接收另一动作类型，缺真实ID、单位及执行前重验桥 |
| 环境一步/下一真实capture | IMPLEMENTED | AirFogSimEnv.step和collect_trajectory已有；后者是采集behavior循环，不是MH闭环 |
| 真实反馈刷新并第二轮MH规划 | MISSING | load_frozen_runtime读取固定离线sample/shard/sidecar；无live rolling history入口或两轮MH证据 |
| NO_SCOREABLE_H4策略分支 | MISSING | 可返回best=None，但没有闭环fallback决定/调用/回执 |
| MH warm start | MISSING | shift_warm_start工具已有且需fresh context、caller tail；solver无注入参数，不是MH集成。learned proposal也只有wrapper接口 |

规则：预测S_hat/graph/latent只用于当前候选比较，禁止成为下一轮真实root。新O_t+1必须fresh capture，History只含O≤t、A/Y<t；固定TRAIN归一化，不能refit，剥离future/target/internal_future审计字段。当前没有live接口，因而不能用离线无泄漏测试宣称闭环无泄漏。

## Fallback options / Researcher Decision Required

| 现有候选 | 合法性和限制 | 是否需要H4评分 / Future Target |
|---|---|---|
| rule_fallback空动作＋显式零CPU/空RB/当前方向speed0 | 空candidate可构造；无MH执行桥。存在eligible compute时NOOP被grammar拒绝，缺UAV方向不得构造；不保证所有状态、任务超时或安全 | 不需要H4评分；Future Target=NO |
| Comm-NOOP＋合法Comp alpha＋PROFILE_HOLD | bind_structured_step已有。当前TRAIN CPU fixture下alpha .5/.75/1.0均admitted，全NOOP反例被拒；joint支持/观测/RB50/domain非空必须成立，非通用fallback | 不需要H4评分；Future Target=NO |
| 采集behavior / CpuPolicyAllocator | 仿真执行路径真实存在；完整behavior含offload/Return route，违反当前Route explicit-NOOP边界，不可直接接入。CPU allocator只是计算分量，不是四族策略 | 不需要H4评分；Future Target=NO |

上述候选均不依赖World Model score或Future Target；unsupported future Return birth不直接阻止真实动作执行，但仍可能阻止H4评分。需要研究者明确正常无H4、domain-empty/缺观测、setter失败三种处理。NONE adopted。本次不H3降级、不改支持域。候选存在和执行primitive存在，不构成闭环安全/性能证据。

## Runtime / Decision Interval

从带SHA验证的320 MH-CEM B1024 raw，采用(n−1)p线性插值：median=90.505966秒，P90=204.915859秒，P95=284.152412秒，max=421.700734秒。这是in-solve，未包含online input重建或模型加载。MH总in-solve=38318.632593秒，327680 transitions / 总in-solve =8.551453/s；cache hits200005，accounted transition requests中cache占37.902347%。完整H4尝试111943、不同可评分33505、不可评分完成55999；不同统计单位不可强制相加。

整个Stage A wall-clock146186.680770秒；三方法in-solve之和113363.124496秒，未归属开销32823.556273秒（9.11765h）。它混合模型准备/加载/参数hash、编排、原子保存、summary/bootstrap；没有分项timer，不能称全部为I/O或全部为模型加载，更不能将其均摊当作实测online端到端延迟。

AirFogSim examples/config.yaml中的simulation_interval=traffic_interval=0.1秒。_build_environment调用build_preflight_config，后者不覆盖该间隔；正式raw记录slot=simulation_interval及连续决策网格。env.step一般按traffic_interval推进，可含多个simulation_interval。**0.1秒是仿真时间，不是已冻结的墙钟计算deadline**；DECISION_LATENCY_REQUIREMENT=UNRESOLVED。若要求每0.1秒墙钟实时决策，中位搜索单项约905倍超时；若暂停仿真慢速规划则可能运行，但不是实时证据。B1024无真实在线执行证据，尚不能冻结为闭环在线预算。

## Stage B Purpose Freeze / Cost / Resources

研究者本次明确：Stage B只研究budget与objective quality、scoreability、candidate diversity、runtime/compute的取舍，为在线预算选择提供证据。SEARCH_METHOD仍MH-CEM，Stage B不得改变Stage A选择。原runner三方法B256/512计划仍保留，本次未调用。

| 方案 | cases / transitions | 3080Ti规划场景耗时 | 用途/决策边界 |
|---|---:|---|---|
| A 原计划64×5×3×两预算 |1920 /737280|30.46–41.85小时|完整三方法budget sensitivity，在线MH预算问题额外成本较大 |
| B MH-only64×5×两预算 |640 /245760|10.26–14.06小时|NEW RESEARCH PROPOSAL，需新runner/provenance/授权，不替换原Stage B |
| C 建议16×3×MH×两预算 |96 /36864|1.54–2.11小时|NEW RESEARCH PROPOSAL，未预注册/未选subset/seed；应保留困难层，匹配现有B1024，曲线仅小样本，不可代表完整闭环 |

这些是两个透明成本情景，不是置信区间或保证上下界：情景1整个耗时按转移数缩放；情景2in-solve按转移数、未归属开销按case数缩放。低预算cache/分支、anchor复杂度和summary开销可非线性。MH开销分摊是假设，03/04中与实测分开。

建议先补live接口并确定fallback/latency，再由研究者预注册C；有用再扩展B，A仅在确需三方法预算敏感性证据时考虑。C不能充分证明完整64锚点或未来闭环分布；评价使用配对anchor/seed，unscoreable不能填零目标；候选数/指纹重复率不是语义多样性证明。

现有短native batch16 benchmark：4090=8.433192/s、3080Ti=7.312050/s，约1.1533倍。不是正式MH cpu_cache_and_prefix完整同协议paired资格；batch32推荐也不可替代batch16。因此4090_MIGRATION_NOT_YET_JUSTIFIED；不启动服务器，不凭理论规格建议迁移。

## Bounded Smoke Requirements / Expected vs Actual / Known Issues

CLOSED_LOOP_READINESS=BLOCKED。审计本身完成，目标是找出缺口而非让readiness通过。缺口不是仅现成函数接线：还涉及winner输出、history-only schema/新实体slots、current deadline sidecar、fallback/latency研究者决定及真实ID执行验证。

最小待授权任务：history-only实时输入→state/graph/posterior；winner sequence/first action输出并保持原排名/预算；四族动作ID/单位桥和setter前重验；研究者指定fallback、domain-empty停止行为、smoke预算及时间模式。然后另行授权CPU bounded smoke，记录真实capture→MH规划→只执行第一步→one real env.step→独立新capture→belief刷新→second planning cycle，覆盖NO_SCOREABLE_H4、domain-empty、setter失败和future poison/drop。本Step不执行该链路。首个纯搜索smoke不要求实现learned proposal或warm start；warm start未来仍需合法重评/预算记账研究定义，不能自动接入。

## Validation / Context / Git / Next Step

命令：`python code/scripts/audit_step6_4a_closed_loop_readiness_v1.py`，960逐文件SHA和16 frozen source通过；01–07重建成功。CPU fixture证明eligible compute全NOOP拒绝，而三个合法alpha/HOLD组合通过，仅单synthetic状态证据。focused `test_step6_3*.py`、`test_step6_4a*.py`、compileall、knowledge-index write/check、diff和Context检查结果见最终08收据。

同步AI_CONTEXT当前状态/科研背景/架构/数据流/模块/实验/研究者决定/问题/变更及authority/process/index/registry。审计证据可接受，闭环执行BLOCKED；没有推送失败实现或修改科研算法。Git commit/push身份以本记录所属提交为准。唯一下一动作：研究者决定fallback及decision latency模式，再授权最小接口集成；Stage B/闭环均不自动执行。
