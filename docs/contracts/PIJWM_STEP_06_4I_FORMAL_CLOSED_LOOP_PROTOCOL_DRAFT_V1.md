# STEP 6.4I — Pure-search正式闭环 Protocol Draft v1

**DRAFT / NOT APPROVED FOR GPU EXECUTION**。本文件是研究者可审阅建议，不是已冻结episode规模、seed、长度、baseline、metric或停止阈值。CPU机制已验收，但正式闭环性能NOT_STARTED，locked_test=false。下一轮研究者批准pilot协议后，须冻结新GPU身份、通过真实live输入CUDA/batch16首轮资格门；本轮不能启动。

## 1. 已批准的工作配置 / 范围

研究者2026-10-09明确决定：先慢Planner真实反馈闭环，再learned proposal/快Planner。S-CEM K4/rho0.2、B_WM512、H4、Route EXPLICIT_NOOP_ONLY、冻结Formal WM/Dataset/normalization/Objective，fallback CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1。非实时同步暂停仿真。B512接受6.4H Effort损失后的计算取舍，不等价、不最优，B1024只读参考。此决定只适用于pure-search v1，不冻结未来hybrid预算。

## 2. Episode来源与初始状态（建议，待批准）

- 最小pilot建议2个独立dev_validation轨迹，每个8个决策步（0.8秒模拟窗口）、一个search seed6311；用于设备/完整记录/循环集成验证，不作性能比较。
- 正式一级建议现有dev_validation的12个独立轨迹×2个search seeds（建议6311/6312）×64决策步（6.4秒），24episodes、最多1536次规划。不是64anchors×5的替代矩阵，不访问locked_test。
- 每个轨迹只取一初始起点，候选规则为该轨迹原静态anchor manifest中最早frame；同frame按sample_id排序，**不读6.4H scoreability/Objective筛选**。pilot从按固定hash排序的轨迹取前2；最终IDs/seed/初始frame必须批准后先写manifest再运行。无需替换困难起点。
- Simulator/prefix policy seeds直接来自原raw environment identity；只重放到预注册起点（含既有warmup和相同因果前缀），核验初始O_t、History和Task cohort SHA。之后behavior退出，Route不再由behavior补动作；车辆由SUMO外生推进。
- 无learned proposal/warm start、无Route扩展。新任务等待Route可能持续积压；结论只能针对受限pure-search接口。不能把完全不具备任务投递能力的Route NOOP系统表述成完整PI-JWM优化系统。
- 遇slot容量超限、DOMAIN_EMPTY、missing字段、非法动作、setter/env.step/观测一致性/指标一致性失败立即终止该episode，保留seed、原分组、失败原因与实际步数。没有替换样本、重试或隐式生命周期修复。

## 3. 每个真实决策步

真实O_t与因果History → 已冻结slot/mask/Train normalization → 双图/State → current-observation posterior mean → repaired CandidateDomain → fresh S-CEM proposal/RNG/cache、B512 H4 rollout/prior mean/service expectation → strict Objective排序 → winner第一步；无scoreable H4时只尝试一次canonical current-domain A，A无法准备/执行即C终止。不存在H3、隐式NOOP、旧behavior恢复。

下一轮root只用env.step后的真实fresh capture重建。每个solve用独立random.Random(search seed)，建议同episode固定base search seed、每轮fresh solve；最终seed方案待批准。不改变既有RNG抽样实现，不共享上轮proposal/cache或winner轨迹。记录真实frame/capture/time以及state/graph/latent指纹，预测H1仅作诊断不能输入下一root。

当前EpisodeController/runner提供：决策token提前消费、setter前整组验证、step intent日志、仅一次env.step、真实capture核验、同输出目录拒绝重复、失败停止。Native三buffer恢复不代表任意setter绝对原子；任何setter失败均终止，记录部分写入风险。step失败可能已有环境部分突变，禁止续跑或换动作。

## 4. 真实任务指标草案（待研究者批准）

所有指标从真实AirFogSim Task和真实observer/transfer轨迹计算，模型预测Objective只作搜索诊断。

**建议主cohort**：初始真实O_t中非completed/failed的所有任务ID，所有baseline使用同一列表。已终止任务记录但不在at-risk分母；初始列表为空时指标null，不填0。新到达任务另报次cohort，按首次真实可观测出现，不读未来schedule。初始terminal tasks不得算作新birth。

- 完成率 = 初始cohort中在观察窗口完成的Task数 / 固定初始cohort总数。失败和pending保留分母。
- Deadline：读取当前Task.getTaskDeadline相对期限与getTaskArrivalTime；按时完成率=真实completion_time≤arrival+deadline的初始Task数/固定cohort总数。另报真实failed、completed-but-late与pending；pending超过期限的诊断不能替换终止标签。
- Delay：真实完成Task的getLastOperationTime−getTaskArrivalTime，必须同时有真实completed lifecycle及isFinished一致证明。只对完成Task做mean/P50/P95，并明确完成子样本数；不得给pending或failed填预测Delay/0，也不得把这一有选择的均值单独声称更好。
- End-to-End Useful Throughput建议定义：**窗口内按时完成的初始Task交付结果总getReturnedSize / 预注册完整模拟窗口时长**。单位是AirFogSim原生data-unit/s；getReturnedSize是所需结果量，只有真实isFinished完成交付才记credit。无返回需求的本地Task仍计完成，但贡献结果量0。不是每hop transmitted bytes、输入任务size或模型输出。若论文希望采用“输入数据量/有效完成任务/实际bytes”等其他定义，必须重新批准，不擅自换口径。
- 早停保持episode原分组及固定主Task分母，吞吐主分母使用计划模拟时长，另报actual exposure；不能缩短时间分母提高表现。早停后未观察到的未来新任务不可推定，birth-cohort结果标记观察不完整，不能同全窗口run直接混比。
- 后续若希望评价所有到达任务的长期服务率，需批准生成窗口/排空窗口、可表示slot、Route边界和censoring规则；本草案不自行增加drain步骤。

来源代码：airfogsim/entities/task.py::isFinished/getLastOperationTime/getReturnedSize/getTaskDeadline；airfogsim_full_dual_graph_observer_v1::_extract_tasks；step6_4i_real_metrics_v1::RealTaskLedger。real ledger为草案实现，不代表研究者已批准metric定义。工程real fallback两观测已验证getters可读取，零完成的短smoke不作性能结论。

## 5. 规划诊断 / 时间 / GPU

每步记录H4 distinct scoreable/complete/unscoreable、Return boundary、grammar dead-end、scorer异常、预算unique/cache/proposed/admitted/rejected、WINNER/A/C、fail reason、只执行first action证明。分母分别明确：scoreable率以所有实际attempted search为分母；另报attempted/全部预定decisions覆盖率，未尝试步数保留；A/C按episode全分组报告，不能只统计幸存episode。

内部search时间（含注明是否包含posterior prepare）、端到端决策时间（live rebuild+planner+validate+setter+step+capture+日志）、episode墙钟和模型/环境一次setup分开。均报告mean/P50/P95/P99/max，定义线性分位数(n−1)p。GPU消耗记录实例运行elapsed（付费时间）、真实GPU型号/batch/precision、模型前向和环境CPU等待时间，GPU utilization只作诊断。0.1秒模拟步不是墙钟deadline。

Runtime参考仅6.4H 320静态S-CEM B512 raw：mean74.548s、median51.416s、P95 194.663s、max376.998s；不是closed-loop测量。窗口中状态/缓存/scoreability会变化，不能保证按这些速度运行。

建议成本（未冻结）：pilot16规划，内部平均0.331 GPU-instance小时；加25%工程余量和0.25h setup建议预留0.66h。P95全步情景1.33h、每步取已观测max情景2.34h。正式1536规划，内部平均31.81h；加25%余量和24×0.1h冷启动建议42.16h；P95全步情景106.22h、观测max全步情景203.47h。后二者是规划情景，不是置信区间或真正最坏上界。具体动态状态可能更慢，需pilot测量后重估。

费用=实际实例单价P（元/h）×实例小时。本轮未访问云价格或启用GPU；P尚未确认，不能编造固定人民币金额。pilot大致0.66P~2.34P，正式42.16P~203.47P。

存储：依据本轮实际JSON/journal大小生成13_cost_storage_receipt.json。至少预留20GiB可用容量，保留原观测/动作/转移/Task ledger/逐case receipt、清单SHA和独立D:备份；不逐candidate保存巨大H4 latent轨迹。建议2份副本+ZIP余量，真实增长速率由pilot复核。

建议（待批准）停止门：scorer/NaNInf/预算/identity/source/重复run立即停止整个pilot；setter/step异常停止episode并阻止自动后续；规划单步>10min或episode>12h作为候选工程hard-stop阈值，须研究者批准，不返回提前best、不加budget、不自动降H。超时保存partial budget/phase及失败分组，不静默重试。

## 6. Baseline公平性和统计（只设计，不运行）

baseline方案未批准。建议以后在同一源轨迹/初始seed/prefix/cohort/time window、同外生SUMO配置、同observer/failure与metrics下配对；主pure-search比较应保持Route NOOP与当前Comm资格一致。若完整旧behavior含Route，必须标记不同动作能力，不能把差异全部归因于搜索器。NOOP/旧behavior不可自动用作本轮baseline或fallback。

episode级完整保留Task计数与失败，先每episode计算再配对。若每源trajectory含2search seeds，统计cluster为**源trajectory（12组）**，保留组内seed，不把1536决策当独立样本。建议10000次cluster bootstrap，seed6316（待批准），输出原始配对/均差/percentile95% CI；小pilot只作机制/成本诊断，不声称显著收益。不同任务终止率/完成样本量同时报告，不以零填补未定义Delay。

## 7. 下一步精确启动门

研究者批准pilot的episode IDs选择规则、初始prefix/seed、步数、metric定义、timeout/异常后是否继续剩余episodes、允许GPU及实际小时价格。然后先冻结批准protocol/manifest和新execution config、commit+push；远端精确FF同commit、tracked clean、无旧runner、目标namespace空/严格resume同身份、checkpoint/norm/catalog/source/parent SHA一致、RTX3080Ti CUDA FP32 batch16、容量和D:备份通道PASS。第一live CUDA决策资格通过后只运行已批准pilot；pilot验收后再次停止审阅，不能自动运行正式24episodes或baseline。

当前状态：READY_FOR_PROTOCOL_REVIEW；GPU live provider=IMPLEMENTED_NOT_GPU_VERIFIED；GPU_LAUNCH_AUTHORIZED=false；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=DEFERRED；locked_test=false。
