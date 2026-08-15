# PI-JWM v11 Selector GPU Iteration Plan

## 2026-08-15 本地文献库迁移与Zotero退役

### Goal

删除用户级`using-superpowers`技能；将PIJWM现有本地论文与Zotero PIJWM集合的元数据、附件和分类完整归档到`文档/文献/`，自动补齐可公开获取的PDF，生成需要用户手动下载的清单，验证后删除Zotero中的PIJWM集合树。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 全局技能与工作区基线审计 | complete | 唯一用户级`using-superpowers`目录已送入回收站；Git状态和官方技能加载规则已核对 |
| 2. Zotero与本地文献清点 | complete | 56个PIJWM条目、32个附件记录、30个原有本地PDF和哈希重复关系已清点 |
| 3. 分类目录与本地归档 | complete | 七类目录、64个唯一PDF、BibTeX/JSON/CSV索引和哈希清单已验收 |
| 4. 缺失PDF公开来源补全 | complete | 禁用Sci-Hub后自动新增18篇；14篇进入`需要手动下载.md` |
| 5. 本地归档验收与Zotero退役 | complete | 验收0错误；云端/本地8个PIJWM集合键均消失，个人条目和RRM集合保留 |
| 6. 文档治理、Git提交与推送 | complete | README/AGENTS/计划表已同步，提交`7828748`并推送；最终状态提交随后复核 |

### Fixed Boundaries

- `D:\shen\PKU\RRM`始终是独立参考项目，不把其文献、代码或证据静默混入PIJWM。
- 每篇PDF只保留一个主存放目录；多分类归属保存在机器可读索引中。
- 自动补全仅使用公开合法来源并设置`PAPER_FETCH_NO_SCIHUB=1`；无法自动获取的条目交给用户手动下载。
- Zotero删除仅针对PIJWM根集合`MZ9JQ2I6`及其子集合；不删除整个个人库、不批量删除条目、不清空回收站。
- 只有本地元数据、现有附件、哈希和缺失清单均完成验收后，才执行不可逆的集合删除。

## 2026-08-15 Repository governance and GitHub publish

### Goal

在不改变研究语义、执行行为、机器证据字节和历史可追溯性的前提下，修复迁移后的worktree路径，完善仓库导航与忽略边界，分类并验证当前源码/测试/文档，最后非强制推送到现有GitHub `origin/main`。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 发布边界与基线冻结 | complete | 记录dirty tree、169 commits ahead、三处迁移worktree和候选发布边界 |
| 2. worktree迁移修复 | complete | 三个Git行政路径与ledger AirFogSim junction恢复到当前根；P2-B/P2-C verify-only均通过 |
| 3. 导航与忽略治理 | complete | 根README、文档索引、artifact/reference说明和.gitignore与当前权威状态一致；空误嵌套目录已移除 |
| 4. 当前源码/测试/文档分类提交 | complete | `ac7d10b`提交框架/脚本/测试，`ae03b0e`提交现行研究文档与归档迁移；治理/计划/交接文件单独提交 |
| 5. P2 ledger证据闭包 | complete | `d6f776a`绑定P2-C v2预文档计数与三个阻断；final candidate生成并逐字节/manifest核验；`e5ad8e4`非破坏性合并到`main` |
| 6. 最终验证与GitHub推送 | complete | 编译/秘密/体积/worktree审计完成；远端拓扑0个remote-only提交；非强制push成功且GitHub ref与本地`main`一致 |

### Fixed Boundaries

- 采用已确认的保守治理方案，不物理移动或批量删除研究资料，不重写Git历史。
- 既有tracked deletion不得仅因出现在`git status`中就视为用户确认删除；必须逐项完成引用和替代关系分类。
- `.worktrees/`、AirFogSim第三方checkout、生成artifact、文献、组会二进制、缓存和临时文件保持本地，不进入发布集。
- 不生成正式v4数据、不训练、不启动GPU、不访问locked test、不改变P2-C三个正式数据阻断。
- 成文设计：`docs/superpowers/specs/2026-08-15-repository-governance-and-github-publish-design.md`；实施计划：`docs/superpowers/plans/2026-08-15-repository-governance-and-github-publish.md`。

## 2026-08-13 P1-MVS 信息边协议实施与证据审计

### Goal

在不重建正式数据、不训练、不读取locked-test轨迹的前提下，实现v4字段registry、旧18槽迁移、Mask/缺失/COO契约和非锁定v3证据审计包。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 成文设计复核 | complete | 用户确认2026-08-12 v4均衡协议设计 |
| 2. TDD契约实现 | complete | 29字段registry、9种缺失原因、18槽迁移、Mask/边类型/COO验证器 |
| 3. TDD审计实现 | complete | split加载前封锁、present边分母、场景/边类型coverage、诚实micro sample、原子manifest |
| 4. 真实非锁定审计 | complete | train 36 / validation 12 / calibration 6；5维有效、13维无有效观测 |
| 5. 自审修正与回归 | complete | legacy/v4命名隔离、固定配置registry、padding分母修正 |
| 6. 独立审查整改与证据重跑 | complete | 补齐协议顺序/schema、失败包、数值守恒、代码哈希和locked I/O门；54项相关回归通过 |
| 7. P1-A CPU动作边界冻结 | complete | 核心动作固定为卸载+RB；CPU为每候选通信后执行的确定性工作守恒内层规则 |
| 8. P2-A CPU内层规则实现 | complete | 纯函数、AirFogSim真实Task源码接口callback、原子预检bundle和manifest；20项测试通过 |
| 9. P2单步非训练集成 | complete | 双远端严格单因素候选、真实AirFogSim通信→CPU/能耗闭环、原子证据bundle；正式v4数据仍未批准 |
| 10. P2多步时序局部门 | complete | setter前CSI勘误、三帧真实step、E1回填、有效零值、append-only已观测边词表和原子bundle通过 |
| 11. 正式v4全双图采集器设计 | in_progress | 核查完整E0/CEP、动态presence、多任务/多流、多seed的直接来源和拒绝门；设计确认前不写实现代码、不训练 |

## 2026-08-13 P2采集器契约与单步非训练集成设计

| Phase | Status | Deliverable |
|---|---|---|
| 1. 现有路径核查 | complete | AirFogSim runtime、事件记录、v4字段契约、严格双图与manifest复用边界 |
| 2. 方案比较与边界冻结 | complete | 单步输入/输出、时序、失败门和证据能力边界已成文 |
| 3. TDD纯契约与适配器 | complete | setter前COO验证、真实scheduler/step、CPU before/after ledger |
| 4. 严格单因素真实集成 | complete | 同seed/config/state/RNG、同50-RB COO，只改变远端目标；两候选均正传输并进入目标CPU集合 |
| 5. 机器证据与回归 | complete | action-pre字段时序勘误后重放；8产物/21 source依赖闭包哈希独立复算0不匹配；不启动GPU、不访问locked test |
| 6. 多步非训练smoke | complete | seed 0三帧真实step、E1上一帧回填、有效零值、append-only已观测边词表、原子失败与7产物/26 source传递闭包哈希通过；正式数据仍未批准 |

### Evidence Boundary

- 正式产物：`代码/artifacts/audit/pi_jwm_p1_information_edge_contract_v4/`。
- `training_eligible_micro_sample_count=0`、`v4_collector_implemented=false`、`v4_dataset_complete=false`、`v4_model_trained=false`。
- 早期自审前产物可恢复保存在`代码/artifacts/audit/pi_jwm_p1_information_edge_contract_v4_pre_self_review_20260813/`，不作为正式证据。
- 独立审查前产物可恢复保存在`代码/artifacts/audit/pi_jwm_p1_information_edge_contract_v4_pre_independent_review_fix_20260813/`，不作为正式证据。
- 未启动GPU；locked-test仅使用既有索引身份元数据，正式manifest的110个输入路径中locked路径为0；6个输出、110个输入和2个代码文件哈希独立复算0不匹配。
- P2-A正式产物：`代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/`。它只验证`PIJWM-CPU-Inner-Rule-v1`纯规则和AirFogSim `TaskManager.computeTasks` callback接口同构；没有执行完整AirFogSim轨迹、没有实现v4采集器或数据集、没有训练。
- P2单步正式产物：`代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/`。setter前decision-time CSI与动作后outcome CSI已分离；旧时序缺陷bundle可恢复归档，不再作为action-pre证据。
- P2三帧正式产物：`代码/artifacts/preflight/pi_jwm_p2_multistep_collector_v1/`。它只证明单条seed 0 fixture上的三步真实执行、已观测通信边词表和E1回填，不证明完整严格双图、分布覆盖、正式v4数据或训练资格。

## 2026-08-12 P1 信息边均衡协议设计

### Goal

冻结一个不追求机械补齐维度、在可观测性、物理语义、动作因果时序、预测价值、计算开销与真实数据迁移性之间可验收的信息边协议；设计确认前只读核查，不改生产代码、不重建数据、不训练模型。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 现有18槽与数据来源核查 | complete | 每槽来源、时序、单位、有效性、获取代价与真实数据可迁移性 |
| 2. 均衡目标与边界确认 | complete | 核心集允许少于18维；先过语义/时序/单位/缺失硬门，再做Pareto选择 |
| 3. 候选方案比较 | complete | E0结构动作、E1五维核心、E2历史逐RB增强、E3当前逐RB上界诊断 |
| 4. 设计与验收协议 | complete | v4字段契约、Mask/异常、消融、统计门、P1-MVS产物和自审记录已成文 |
| 5. 用户审阅设计 | complete | 用户已确认成文设计；实施计划和P1-MVS已于2026-08-13完成 |

### Fixed Boundaries

- 字段数量不是优化目标；不可得字段不得用常量、零值、代理量或改名伪装为真实观测。
- 决策前输入、动作和动作后结果必须分开；动作后结果不能泄漏进同一时刻决策输入。
- 设计确认前不修改生产代码、不启动GPU、不访问locked test、不重建正式数据。
- 成文设计：`文档/研究进展/2026-08-12-PI-JWM-v4信息边均衡协议设计.md`；P1-MVS实施状态见上节，不能据此宣称v4正式数据完成。

## 2026-08-12 P0 理论—实现—证据一致性审计

### Goal

在不启动长GPU任务、不访问locked test、不清理脏工作树、不修改模型机制的前提下，对PI-JWM的固定理论、组会/PPT口径、代码路径、数据字段、测试、artifact和阶段结论进行逐项核查，输出可追溯的claim matrix、冲突分级、处置决议和P1/P4/P6阻塞门。

| Phase | Status | Deliverable |
|---|---|---|
| 0. 恢复上下文与冻结审计边界 | complete | 读取现有规划文件；旧“完成”状态仅作为待核陈述；冻结禁止长GPU、locked test和工作树清理 |
| 1. 权威文档与组会材料清单 | complete | 已核对8.11 PPT有效区段191—203页、问答、讲稿、主文档和8.12推进 |
| 2. 代码—测试—artifact证据抽取 | complete | 已定位关键入口、输入输出、测试、正式结果与manifest；106项定向测试通过 |
| 3. 逐项claim matrix与冲突分级 | complete | 25项claim已分为verified / partial / contradicted / not-implemented / not-verifiable |
| 4. 关键机制深审计 | complete | 18/5信息边、规则递推、RSSM、因果/反事实、CPU动作空间、候选Rollout闭环已形成结论 |
| 5. 处置与串行门 | complete | P1-MVS为下一步；P2/P4/P6/P7及100k按依赖阻塞 |
| 6. 产物、manifest与文档同步 | complete | 审计报告、机器矩阵、输入/输出哈希、8.12推进与本地计划已同步 |

### Fixed Boundaries

- “文档写了”“类名存在”“张量宽度正确”“checkpoint可加载”“smoke通过”均不等于理论机制已实现。
- 没有取得可核验PPT源文件时，只能记录“PPT未取得/不可核验”，不得根据讲稿反推PPT内容。
- P0只读审计现有模型和数据；只允许新增审计产物及同步状态文档，不修模型、不重训、不访问locked test。
- 旧R1—R6结论逐项重审，不继承旧对话的技术判断；发现冲突立即阻断相应扩训。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 错误调用不存在的wait cell | 1 | 确认为工具调用错误且未产生副作用；不重复调用，直接按真实任务继续 |
| Default模式误调用request_user_input | 1 | 工具明确返回不可用；无需用户补充即可从本地证据继续 |
| 首次三文件补丁假定了错误的progress标题 | 1 | 补丁验证失败且未写入；重新读取真实文件头后拆成最小补丁 |
| 首次PPT检查未先初始化artifact-tool workspace，脚本错误解析到项目内不存在的runtime依赖 | 1 | 未将失败当作PPT证据；按技能规范先运行setup脚本并指定已加载的bundled Node路径后重试 |
| 第二次PPT初始化仍从当前工作目录推导runtime，工作目录设为Node目录导致重复拼接依赖路径 | 2 | 读取工具源码确认其在HOME缺失时使用cwd；不修改HOME，改从真实用户目录`C:\Users\Lenovo`启动 |
| 第三次模板检查在artifact-tool初始化成功后因系统`unzip -Z1`不可用失败 | 3 | 停止重复模板检查脚本；P0不编辑PPT且不需要媒体清单，改用已初始化的artifact-tool直接import、inspect和逐页render，并记录此偏离 |
| 自写逐页inspect传入`slide.id`后仍返回全deck快照并截断 | 1 | 不采用这些页级ndjson作证据；使用完整`deck-inspect.ndjson`按slide字段过滤，PNG仍由真实slide对象逐页导出 |
| 首次表格读取使用不存在的`table.columns.length`，循环未执行而得到空行 | 1 | 不猜表格数值；检查真实table对象/proto或只读解析PPT表格XML，并与渲染截图交叉验证 |
| P1检索假定存在`airfogsim_online_graph_v3.py` | 1 | `rg`准确报告该文件不存在；不据此推断在线数据缺失，改从实际`r6_online_observation.py`和信道管理器路径追踪 |
| P1并行检索中至少一个`rg`模式无匹配使聚合命令退出1 | 1 | 按项目规则解释为“未找到”；拆分检索并保留其他命中，不误报权限或工具故障 |
| 用`rg --files | rg`按文件名筛选artifact未命中 | 1 | 只说明该管道的路径编码/模式没有返回结果；不据此否定产物，后续从P0已核验的明确目录读取 |
| P1阈值证据聚合检索因一个目录/模式无匹配退出1 | 1 | 已有命中仅用于确认可测指标；不从旧v3数值拟合v4门槛，阈值按新协议事前冻结 |

## 2026-08-08 R6.1正式GPU训练前协议与CPU预检

### Goal

在不启动GPU、不访问locked-test、不重训世界模型和不重建数据的前提下，冻结正式多目标reward与硬/软约束、卸载/RB/CPU联合动作语义、真实闭环transition与GAE训练批次、正式训练矩阵及停止条件，并以真实非锁定AirFogSim执行完成CPU入口验收。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 现有执行链与数据字段审计 | complete | 明确可复用runtime、调度回调、动作/结果台账、R2指标和缺口，不重复实现已有能力 |
| 2. R6.1设计与协议冻结 | complete | 设计文档、机器可读reward/action/rollout/training协议和明确证据边界 |
| 3. TDD实现联合动作、reward和GAE契约 | complete | fail-fast候选验证、硬约束拒绝、reward分量、transition/trajectory/GAE和Actor–Critic/PPO更新 |
| 4. 真实闭环采样适配与CPU正式预检 | complete | validation seed 507真实执行；4条连续transition，卸载/RB/CPU实际改变1/2/1次，世界模型哈希不变 |
| 5. 冻结GPU实验矩阵与停止门 | complete | 2方法×3状态×3 seed共18-run预算、validation选模、失败保留和早停已机器化；未启动GPU |
| 6. 全量回归与文档同步 | complete | 58项定向和75项历史测试通过；371项扩展测试仅1项用户删除目录失败；产物/输入哈希、结果说明、两份主文档均完成 |

### Fixed Boundaries

- 直接复用R1数据、R2评价协议、R5.1候选B及R6规则配对闭环，不重新生成或训练这些资产。
- AirFogSim只提供真实动作执行和反馈；新方法代码继续位于`代码/src/pi_jwm/`。
- validation/calibration只用于开发与阈值；locked-test在R9前不读取。
- 本阶段完成后最多置`r6_gpu_strategy_training_ready=true`，不会启动GPU、不会宣称策略性能或最终方法定型。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 首次同时补丁假定`findings.md`首行为`# Findings`，上下文不匹配 | 1 | 重新读取真实首部，拆分为基于真实内容的最小补丁；未产生部分修改 |
| 并行执行3个Zotero CLI检索时外层24秒超时 | 1 | 改为单查询60秒上限；确认PPO和GAE条目当前未在本地库中，未重复写入或暗猜查询失败原因 |
| 首次按字段名把`delivered_data_total`解释成累计量，正式尺度冻结因序列下降而fail-fast | 1 | 核对seed 0原始数组与报告：300步数组求和等于47.137 MB且字段逐步归零，确认其为每时隙交付量；同步修正文档、测试和实现，不做差分 |
| 首次从`代码/tests`运行语法编译时使用了错误相对路径，单测仍通过但编译目标未找到 | 1 | 在`代码/scripts`重新独立运行`py_compile`并确认通过；不把后续命令成功误当作前一命令成功 |
| `airfogsim`环境缺PyTorch，base环境又缺AirFogSim几何依赖 | 1 | 不混用不同Python版本的二进制包；在现有`airfogsim`环境安装与checkpoint环境一致的`torch 2.8.0+cpu`并验证`cuda_available=false` |
| PyTorch 2.4 CPU wheel的`fbgemm.dll`依赖缺失的`libomp140.x86_64.dll` | 1 | 用PE依赖表确认根因后卸载2.4并升级2.8 CPU wheel；未复制/伪装DLL |
| 首次真实仿真使用通用`BaseAlgorithm`触发卸载断言 | 1 | 复用正式数据路径的DAG-ready调度、返回路由、观测型通信环境和非变异传输查询，不放宽断言 |
| 冻结张量的float32时间`8.100000381`使循环误走到8.2 | 1 | 先按AirFogSim仿真间隔量化到8.1，再保持严格误差检查 |
| 标准AirFogSim环境与正式观测环境在时隙80任务数不一致（128对58） | 1 | 复用正式v1的Observed通信路径和`nonmutating_transmission_tasks`，随后live/frozen任务与节点事实对齐 |
| 相同seed协议文件首次复跑哈希变化 | 1 | JSON差异定位到unittest墙钟时间；成功时不写入时长tail，连续两次重跑协议SHA-256完全一致 |
| 371项扩展回归中目录治理测试失败1项 | 1 | 确认为用户工作区已删除`文档/项目说明/`的既有改动；不擅自恢复，记录为非R6功能回归；其余370项及75项历史指定测试通过 |

### Current findings

- 现有正式runtime只提供可注入的CPU allocator；卸载/RB联合动作注入需要复用已有候选执行与严格动作辅助接口，而不能暗猜新的AirFogSim API。
- 现有`airfogsim_diagnostics.reward_components()`只是历史诊断型任务效用，不能直接作为R6正式多目标reward；正式定义必须绑定R2 canonical指标、归一化尺度和硬约束门。
- 已形成`docs/superpowers/specs/2026-08-08-r6-1-joint-policy-gpu-readiness-design.md`：采用完整可行联合方案候选集、显式/隐式双状态、service-first分解reward和真实transition/GAE；GPU矩阵只冻结不运行。

## 2026-08-08 R6学习策略CPU准备

### Goal

在不修改R1/R2协议、不访问locked-test、不启动GPU且保持R5世界模型冻结的前提下，完成显式＋隐式策略状态、统一动作契约、逐节点CPU安全投影、Masked Actor–Critic/PPO数值更新和真实非锁定状态预检，为后续单独冻结GPU策略训练协议建立可审计入口。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 冻结策略状态与动作契约 | complete | `PolicyState`、`ActionSpec`、样本身份、未来target封锁、逐节点CPU容量与任务—节点映射 |
| 2. TDD实现安全投影和学习策略 | complete | Masked Actor–Critic、精确非法动作mask、CPU连续分布、逐节点残差安全投影及审计记录 |
| 3. 实现Actor–Critic/PPO CPU更新 | complete | 两种目标均完成有限loss、有限梯度和策略参数变化检查；世界模型保持stop-gradient |
| 4. 真实冻结状态正式预检 | complete | R5候选B真实checkpoint和非锁定validation窗口完成正式预检；CPU任务435个、活动节点38个，硬约束违规0 |
| 5. 独立验收与文档同步 | complete | 新增22项、既有R6 17项、历史回归75项通过；7个bundle文件独立哈希复算一致 |

### Fixed Boundaries

- 本轮只通过CPU动作学习入口；正式状态中的卸载和RB被限定为安全no-op，不冒充已经冻结联合动作训练协议。
- Actor–Critic/PPO只用常数advantage/return做数值smoke，不作性能、收敛或方法优劣结论。
- 策略状态只来自历史观测、历史动作和当前mask；未来target与locked-test均不可进入。
- 当前状态是`r6_learning_policy_cpu_ready=true`，但仍是`r6_gpu_strategy_training_ready=false`和`final_method_frozen=false`。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 将Graph-RSSM belief误按R3 `.joint`接口读取 | 1 | 区分R3与RSSM belief；RSSM使用deterministic＋stochastic拼接，并补定向测试 |
| 正式状态构造缺少`PolicyIdentity`导入 | 1 | 增加真实资产集成测试并修复已有类型导入 |
| `float32`缩放后两个节点残留`7.15e-7/2.38e-7`容量超量 | 1 | 增加边界回归；缩放后回收舍入残差，严格保证逐节点不超容量 |
| `evaluate()`会为外部传入的masked离散动作返回极大有限负log-prob | 1 | 收尾审查新增失败测试；训练评价入口现按当前effective mask拒绝非法卸载/RB动作 |
| 独立哈希脚本的中文绝对路径在PowerShell管道中被替换 | 2 | 按准确编码根因改为从`代码/`使用纯ASCII相对路径；7/7哈希复算通过 |

### Formal result

- 正式产物位于`代码/artifacts/preflight/pi_jwm_r6_learning_policy_cpu_preflight_v1/`，结果说明位于`文档/研究进展/2026-08-08-PI-JWM-R6学习策略CPU预检结果.md`。
- 真实状态绑定候选B、training seed `20260803`、validation轨迹`load_high__density_dense__r07`、环境seed `507`和时隙`298`；24维显式状态和32维隐式belief均来自冻结历史输入链。
- Actor–Critic/PPO两种更新均有有限梯度且策略参数发生变化；世界模型参数更新前后SHA-256相同，失败记录0、硬约束违规0、locked-test未访问、GPU未启动。

## 2026-08-07 R6 CPU-only同场景配对闭环

### Goal

在不改世界模型、不改R1/R2协议、不访问locked-test且不启动GPU的前提下，使用同一场景、同一环境seed和同一仿真配置，对三种规则CPU策略与一个确定性局部搜索CPU启发式进行配对闭环比较，形成可审计的动作合法性、真实系统指标、paired delta和action regret入口。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 设计与协议冻结 | complete | 配对身份、四个策略臂、局部搜索邻域、指标语义、验收门已写入R6设计文档和实施计划 |
| 2. TDD策略/配对契约 | complete | 策略权重、CPU投影、局部搜索确定性、同状态指纹、locked-test封锁和N/A语义测试 |
| 3. 实现可注入runtime与配对runner | complete | 复用AirFogSim真实执行链，默认runtime行为不变，四臂配对运行与bundle写出 |
| 4. CPU smoke与正式非锁定运行 | complete | airfogsim环境下完成1个validation smoke和54个非锁定场景—seed的216次配对仿真 |
| 5. 独立验收与文档同步 | complete | 哈希复算、配对完整性、硬约束0、locked-test未访问、paired delta边界和结果文档均完成 |

### Fixed Boundaries

- 四个CPU策略臂为`equal_share`、`deadline_aware`、`feasible_exploration`和`local_search`；只有CPU回调可变，卸载、RB、移动、信道、任务到达和DAG先后不变。
- local search只在当前时隙可观测队列和deadline上生成有限邻域，不读取未来结果；它是CPU启发式下界，不是学习策略。
- 每个pair必须绑定相同`scenario_id`、环境seed、配置指纹、最大仿真时间和协议版本；缺少完整反事实时指标为`not_computable`。
- train/validation/calibration可用；locked-test在R9前完全封存。完整R6仍不等于最终策略器定型。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 默认`base`环境缺少AirFogSim依赖`osmnx`，4臂smoke全部失败 | 1 | 保留`pi_jwm_r6_cpu_paired_smoke_v1/v2`失败bundle；按项目规范切换到`airfogsim`环境，未安装或修改依赖 |
| 外层工具等待上限先于conda子进程返回 | 1 | 通过进程树和目标文件确认子进程继续完成；未重复启动正式run，并对最终bundle做独立验收 |

### Formal result

- `airfogsim`环境正式完成54个非锁定base spec × 4个CPU策略臂，共216/216 runs；`train/validation/calibration=144/48/24`。
- 同一`scenario_id`、环境seed、配置指纹、最大仿真时间和协议版本完成54个完整pair group；`locked_test_accessed=false`、`world_model_updated=false`、`gpu_started=false`。
- `action_legal_rate_min=1.0`、硬约束违规0、失败run 0；指标状态为`available=3888`、`not_applicable=216`、`not_computable=648`。
- 8个bundle文件独立重算SHA-256全部一致。paired deltas仅作为同状态描述性比较；没有据此冻结最终策略，也没有声称action regret已可计算。

## 2026-08-07 R6 CPU策略预检

### Goal

严格复用R1—R5.1冻结资产，在不访问locked-test、不重新生成数据、不启动GPU的前提下，完成R6候选角色、动作台账、硬约束、安全壳接口和真实AirFogSim闭环指标的CPU预检，为后续受控规则/局部搜索与学习策略比较建立可审计入口。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 冻结R6设计、输入和验收门 | complete | R6 CPU设计与实施计划；B/G/J/F角色、54条非锁定轨迹、硬约束和指标门固定 |
| 2. TDD实现角色/轨迹/指标审计模块 | complete | 5项测试先因模块缺失而失败，再通过；候选角色漂移、locked-test、约束违例和N/A语义均严格拒绝 |
| 3. 实现并运行正式CPU预检 | complete | 54条轨迹级审计、三种CPU策略真实系统指标和自校验manifest已生成 |
| 4. 独立验收与文档同步 | complete | 75项既有回归、5项R6测试、6/6输出哈希复算、go/no-go及主计划/主文档同步完成 |

### Fixed Boundaries

- AirFogSim只作为仿真与数据源；R6方法代码仍位于`代码/src/pi_jwm/`。
- 现有三种CPU策略轨迹只形成闭环下界和执行链证据；不同seed/轨迹上的均值不作严格因果优胜结论。
- R5世界模型在本阶段只读取冻结角色，不进行联合微调；学习策略GPU训练必须等CPU门通过后另定预算。
- validation/calibration可用于开发诊断；locked-test在R9前保持封存。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 首次正式runner命令的shell超时设为1秒，进程在产物写出前被终止 | 1 | 先确认输出目录不存在，再以120秒上限重跑；正式run一次完成，不拼接首次中止结果 |
| 从`代码/tests`运行位于`代码/scripts`的两个历史测试时模块未找到 | 1 | 按真实文件位置在`代码/scripts`重跑75项历史回归；R6的5项测试继续从`代码/tests`独立运行 |

## 2026-08-07 R5.1本地统计与R6候选冻结

### Goal

在不重新训练、不访问locked-test、不改变R1数据协议或R2评价定义的前提下，严格完成已下载R5.1结果的本地独立验收、B/F/G/H/J三seed与1/5/20步诊断，并形成可审计的R6工作候选集合。该集合只冻结R6开发输入，不冒充R9最终PI-JWM方法。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 本地证据恢复与独立验收 | complete | 解包远端结果；正式验证器在本地验证12个新checkpoint、3个复用run、72个manifest文件及locked-test边界 |
| 2. 统计接口审计与TDD设计 | complete | 复用现有R5统计/评价接口；配对统计、分horizon、主指标门、N/A和候选角色规则已在结果产生前冻结 |
| 3. 实现并运行R5.1本地分析 | complete | 15个checkpoint CPU重放、三seed配对差值、45个1/5/20步单元、权衡与稳定性报告均已生成 |
| 4. 冻结R6工作候选集合 | complete | B主候选/控制、G任务专长、J连续状态专长、F去耦消融及H no-go已写入自校验manifest |
| 5. 同步计划与权威主文档并最终验证 | complete | `本地计划表.md`、`PIJWM推进.md`及研究进展记录一致；30项定向测试、Python编译、8项manifest复算和文档检查通过 |

### Fixed Boundaries

- 不重新生成AirFogSim数据，不重新训练B/F/G/H/J，不启用GPU。
- validation只用于模型/候选诊断，calibration只用于校准诊断；locked-test在R9前保持封存。
- 综合分数只作为冻结R2复现指标，不能覆盖链路活动、活动速率、任务生命周期、连续状态和跨horizon稳定性的分项门。
- J是当前协议下适配的旧架构控制，不是论文baseline；B/G/J是否进入R6必须由同seed、同horizon、同口径证据决定。
- 当前工作区包含未提交的R3—R5真实依赖，新worktree会缺失这些资产；因此在当前分支原位追加，不移动、不清理、不覆盖用户改动，也不创建提交。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| `apply_patch`首次按旧版Phase文本定位失败 | 1 | 重新读取真实表格内容后，按当前行做最小补丁；未产生中间文件或重复改写 |
| 首次CPU冒烟以1秒shell时限启动，进程被超时终止 | 1 | 确认未产生输出后使用可等待执行cell重跑；未重复正式批处理 |
| PowerShell管道中的中文路径在Python stdin代码页变成`?` | 2 | 改由`Path.cwd().glob()`定位脚本和代码根目录，真实CPU冒烟随后通过 |

## 2026-08-05 R5 CPU组合预检

### Goal

在不启动GPU、不访问locked-test、不改变R1数据和R2指标定义的前提下，冻结R5有限组合、正式三seed预算和独立指标门；复用R4公共模型/目标/checkpoint/窗口接口，在真实train/validation/calibration窗口上完成组合前向、反向、1/5/20步rollout、动作敏感性、结构/守恒、校准接口和严格重载预检，形成可直接交给GPU runner的机器可读产物。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 审计R4接口并固化R5设计/实施计划 | complete | 复用边界、A-E组合、三seed预算、独立指标门和GPU交接契约 |
| 2. TDD实现R5组合注册与协议 | complete | 受控多模块配置、非法组合拒绝、三seed预算和locked-test封锁 |
| 3. TDD实现CPU组合预检runner | complete | 真实非锁定窗口的forward/objective/backward/1-5-20 rollout/checkpoint与诊断 |
| 4. 运行正式CPU预检并独立验收 | complete | A-E×3 seed共15/15通过；26个manifest文件哈希复算一致；`r5_gpu_ready=true` |
| 5. 同步主计划、权威文档和GPU交接说明 | complete | R5 CPU结果、限制、正式GPU预算与交接文件均已记录 |

### Fixed Boundaries

- 候选固定为A参考Graph-GRU、B Graph-RSSM、C Graph-RSSM+异方差头、D Graph-RSSM+显式DAG消息、E Graph-RSSM+soft presence；CPU门不新增其它候选。
- 正式训练预算固定至少3个训练seed、最多100 epoch、patience 10；validation选模型，calibration只做阈值/不确定性校准，locked-test继续封存。
- R4综合分数保留用于可复现排序，但R5不得只凭该分数定型；链路活动、活动速率、任务生命周期和连续状态必须分别过门。
- CPU结果只证明组合接口、梯度、因果性、结构合法性、重载与报告链路正确，不作收敛或性能优越性声明。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Windows下把`test_r4_*.py`直接作为`rg`路径导致卷标语法错误 | 1 | 改用测试目录加`-g 'test_r4_*.py'`过滤，后续不重复该命令形式 |
| 单一1步验证窗口没有活跃链路和数据流，四项公共指标出现`not_computable` | 1 | 保留指标的严格N/A语义；CPU门统一使用1/5/20步最低充分验证窗口集合，不将缺失样本伪造为0 |
| 正式runner首次外层超时设置过短而被终止 | 1 | 核对未形成目标产物后，以600秒命令时限完整重跑；正式运行93.3秒完成 |

## 2026-08-04 R4 GPU单模块筛选

### Goal

在R4 CPU门通过后，严格复用R1数据、R2评价协议和R4受控候选矩阵，在RTX 4090上完成单seed短预算筛选。预算固定为seed `20260803`、最多30 epoch、patience 5、batch size 32；所有候选使用相同窗口、优化步数和checkpoint选择口径，不访问locked-test，不把单seed筛选结果写成最终模型定型。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 服务器与环境审计 | complete | RTX 4090、CUDA 12.8、PyTorch 2.8.0、磁盘和独立运行目录均已核验 |
| 2. TDD实现GPU训练与评价runner | complete | 同预算训练、validation选模、失败保留、严格checkpoint与恢复接口 |
| 3. 同步最小代码/数据并执行远端smoke | complete | CUDA forward/backward、20步rollout、checkpoint复现和locked-test封锁均通过 |
| 4. 运行12臂短预算筛选并持续监控 | complete | 12/12候选完成、0失败；逐epoch记录、最佳checkpoint、成本和四项公共指标齐全 |
| 5. 回传产物、独立验收并同步计划 | complete | 本地46/46清单哈希一致；R4结论和R5候选边界已固化 |

### Fixed Boundaries

- 远端使用独立目录；不覆盖旧PI-JWM训练目录，不修改AirFogSim第三方代码。
- GPU筛选只读取train/validation/calibration；任何locked-test请求必须在加载数据前失败。
- R2冻结的`validation_protocol_score`是checkpoint选择口径；calibration只能校准阈值或分布参数，不能选架构。
- 方向性JEPA、GATv2、Transformer和Deep Ensemble仍是reserve/deferred，不因服务器已开启而越过CPU门直接训练。

## 2026-08-04 R4 CPU接口与预检实现

### Goal

在冻结R1数据、R2评价和R3参考接口的前提下，以单模块控制变量方式实现R4候选注册、统一模型/目标/checkpoint接口和真实非锁定窗口CPU预检。CPU阶段只证明接口、梯度、封锁和复现正确，不选择最终模型，不访问locked-test，不启动GPU。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 审计R3真实接口和R4书面计划 | complete | R3 22项定向基线通过；确认在当前`codex/`功能分支增量实现且不修改R3 |
| 2. TDD实现候选注册表与严格配置 | complete | 七个模块族、参考/计划/储备/推迟状态、单模块配置与非法配置拒绝 |
| 3. TDD实现统一模型、目标和checkpoint接口 | complete | 与R3同输入/输出的R4工厂、12个受控候选、概率辅助目标和严格checkpoint |
| 4. TDD实现CPU runner和封锁报告 | complete | 9个真实非锁定窗口、1/5/20步、forward/objective/backward/reload和机器可读报告 |
| 5. R3回归、R4全量CPU验收与计划同步 | complete | 12/12候选通过，`r4_cpu_preflight_ready=true`、`gpu_screening_ready=true` |

### Fixed Boundaries

- R3源码、R1数据契约、R2指标和split不改；当前工作区已有未提交用户资产不移动、不清理。
- 每个新生产接口必须先有按预期失败的测试，再写最小实现；未实现候选不得静默回退。
- CPU产物不得给出模型优越性结论；GPU只在全部CPU门通过后开始。

## 2026-08-04 R4候选方法文献门

### Goal

在任何R4模型实现或GPU实验之前，系统审计字段编码、图编码、双图耦合、潜在动力学、输出与不确定性、DAG/动态拓扑六类方法。每个进入实验的候选必须有原始文献依据、明确机制与假设、PI-JWM接口映射、可证伪假设和公平比较方案；不凭直觉预先固定最终网络。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 冻结R4实验并建立方法问题清单 | complete | 六类候选的审计字段与准入门 |
| 2. 审计Zotero并检索原始来源 | complete | 去重后的核心文献集及证据等级 |
| 3. 方法级精读与PI-JWM映射 | complete | 机制、公式、假设、输入输出、风险和实现接口 |
| 4. 候选矩阵与实验顺序 | complete | 一次只改变一个模块的候选组、排除项和递进预算 |
| 5. 文档与Zotero固化 | complete | R4方法调研文档、计划状态和文献记录；云端已验证，本地等待Zotero同步 |

### Fixed Boundaries

- R3参考模型、R1数据协议和R2评价协议保持不变；本阶段不训练、不访问locked-test、不宣称候选优劣。
- 文献只能证明一般机制或设计先例，不能替代PI-JWM上的控制实验；最终模块必须由同场景、同数据、同预算的消融结果决定。
- 优先正式顶刊/顶会原始论文；奠基方法与尚无正式替代的最新预印本必须明确标注证据等级。
- 基座模型预训练、跨场景迁移、策略器和规划器不进入本轮R4结构筛选。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Windows命令参数链路和首次Web API正文更新把三条Zotero中文标签写成乱码；HTTP状态成功但云端内容校验失败 | 2 | 停止信任成功码与CLI回显；使用显式UTF-8 `StringContent`、当前版本并发保护和完整数据回写，随后逐项云端回读，作者、集合及期望标签均恢复且无额外乱码标签 |

## 2026-08-03 R2返修前正确性与文献审计

### Goal

对已完成的R2评价协议做一次独立的定义—代码—数据—统计—文献闭环审计；发现真实问题即在R2内修正并补测试，避免把歧义带入R3—R9。审计不访问locked-test标签、不训练模型、不预先固定候选网络。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 逐项审计36项注册表、v3评价器、22项真实sidecar及公平实验协议 | complete | 公式/单位/mask/命名映射/聚合/数据泄漏问题清单 |
| 2. 用原始论文和官方资料核验关键评价口径 | complete | 结构化文献证据表及适用边界 |
| 3. 独立代码审查与交叉质疑 | complete | Critical/Important/Minor审查意见 |
| 4. 对确认问题实施最小修正并补红绿测试 | in_progress | 无歧义的R2协议v3产物 |
| 5. 全量重建、独立哈希与文档同步 | pending | R2返修前审计报告和R3 go/no-go结论 |

### Fixed Boundaries

- 只用54条非锁定轨迹和既有真实sidecar；6条locked-test继续只做完整性核验。
- 文献只能支撑评价原则和指标选择，不能替代对PI-JWM具体字段、公式和实现的代码审计。
- 不声称存在“一篇论文完整证明PI-JWM全部评价体系”；每项证据写清直接支持与PI-JWM工程选择的边界。
- 若注册表与已有22项真实指标未建立一一映射、统计层级含混或checkpoint复合分数不可复现，视为R2必须修正的问题。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 本地单位审计命令引用了不存在的`代码/src/pi_jwm/airfogsim_adapter.py`，`rg`返回路径错误 | 1 | 不重复猜文件名；先用`rg --files`枚举真实适配器路径，再读取AirFogSim速率源码和实际字段映射 |
| 四路只读批次整体返回退出码1，聚合输出未标明具体失败分支 | 1 | 保留已成功返回的证据；把缺失的归一化统计和测试搜索拆成独立命令，不猜测权限或文件状态 |

## PI-JWM统一阶段门路线图文档化（2026-08-03）

### Goal

将老师口径下的严格双图、显式/隐式双状态、正式数据协议、模块消融、世界模型定型、策略比较、论文baseline和闭环评价统一为一条R0—R9阶段门路线，并明确复用资产、CPU/GPU需求、预期产物和进入下一阶段的验收条件；路线截至基座预训练与跨场景迁移之前。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 审计现有定义、代码、数据、指标和实验资产 | complete | 可直接复用、需语义适配、必须重跑三类边界 |
| 2. 冻结R0—R9统一路线结构 | complete | 阶段依赖、资源类型、预期结果和停止门 |
| 3. 同步PI-JWM权威主文档 | complete | 统一方法路线和证据边界 |
| 4. 重写本地执行计划表 | complete | 完整逐步计划、状态和权威索引 |
| 5. 一致性与Markdown验证 | complete | R0—R9阶段、资源、预期结果和状态检查全部通过 |

### Fixed Boundaries

- 已完成的60条AirFogSim原始轨迹、split、动作/结果台账、指标实现、训练入口和实现级测试优先复用，不重复生成或重写。
- 旧数据/张量把无线信道放在物理图中，因此旧checkpoint和旧数值只作历史证据；teacher-aligned v3映射、张量和正式结果仍必须重新生成。
- 基座模型预训练、外部预训练微调和跨场景迁移不进入R0—R9执行范围，只在R9正式方法冻结后另立计划。
- locked-test继续封存，直到数据、方法、超参数、停止规则和报告模板全部冻结。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 路线阶段计数把Mermaid节点和“立即执行顺序”误算为计划表行 | 1 | 验证表达式收紧为只匹配`| **R0...R9`正式表格行后重跑；不修改路线内容 |

## 正式双图模型与完整评价本地阶段（2026-08-01）

### Goal

基于54条非锁定正式AirFogSim张量，在CPU上先跑通新双图模型、统一评价指标和公平baseline对比的最小闭环；本地门通过后再迁移长训练到GPU，locked-test继续封存。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 盘点现有模型、评价和baseline接口 | complete | 复用边界、缺口和首轮成功标准 |
| 2. 确认并冻结本地CPU smoke设计 | complete | 模型输入输出、baseline、指标、split与预算 |
| 3. 编写逐文件实施计划 | complete | TDD任务、运行命令和验收门 |
| 4. 实施并运行正式数据CPU smoke | complete | 新双图模型与baseline同口径结果 |
| 5. 冻结GPU训练入口 | in_progress | CPU门已通过；长训练预算、完整论文baseline和多seed成功门待冻结 |

### Fixed Boundaries

- 数据只使用train/validation/calibration；locked-test在模型与训练协议冻结前不读取。
- AirFogSim是数据来源，PI-JWM是模型主线；不恢复任务/DAG充当信息图节点或边的旧语义。
- baseline与新模型使用相同窗口、归一化、标签、指标和计算预算；先比较状态rollout，不提前进入selector/ranking。

## AirFogSim新双图数据重构（2026-07-31）

### Goal

在不修改AirFogSim仿真内核的前提下，通过PI-JWM适配层把真实设备/信道、信息代理/信息流、任务生命周期与DAG、卸载/RB动作、资源和结果统一导出为新双图契约，并提供可计算真实评价指标与优化目标的基础台账。

### Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. 审计旧适配器、数据字段和测试 | complete | 旧语义复用边界与v2文件映射 |
| 2. 冻结v2数据契约和合法性规则 | complete | 机器可读schema及失败测试 |
| 3. 实现代理—信息流新构图 | complete | 物理图、信息图、DAG辅助结构、CIP/CFE |
| 4. 生成AirFogSim最小样例 | complete | 九方向物理快照、代理—信息流双图和$t=2.6$严格同一时隙样例 |
| 5. 接入真实指标基础台账 | complete | 任务、链路、资源、能量、约束数值及不可计算状态报告 |
| 6. 全量验证与文档同步 | complete | 相关153项测试通过；全量749项中746项通过，3项既有文档路径问题已记录；编译和差异检查通过 |

### Scope Decisions

- AirFogSim只作为仿真与原始数据来源；新代码放在`代码/src/pi_jwm/`，运行脚本放在`代码/scripts/`，测试放在`代码/tests/`，产物放在`代码/artifacts/`。
- 首版信息节点与活动物理节点一一对应；任务和DAG保持辅助结构，只有真实输入、结果或跨代理依赖数据才生成信息边。
- 当前核心动作只包含卸载与RB；CPU继续使用固定规则，不扩展selector/ranking，不启动正式训练。
- 当前导出只确认V2U、V2I和U2I；其余AirFogSim信道方向先审计接口，再决定是否纳入v2导出。
- 所有缺失字段使用显式`null + mask/status`，不得以0伪造。

### Acceptance Gates

- 新测试先失败后通过，且现有适配器与小实验测试不回归。
- 每个信息代理具有唯一物理附着；每条信息流具有真实端点、正剩余量和唯一`flow_id`。
- 每条活动承载关系指向真实存在的物理边；等待流允许没有CFE承载。
- DAG边只有在前驱完成、后继位于不同代理且真实释放数据时才转为`dependency_data`信息流。
- 指标输入能追溯到任务、链路、资源或约束原始字段，并报告分子、分母、样本数、单位和可计算状态。

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 组合只读审计命令返回退出码1 | 1 | 确认是某个搜索无匹配/路径缺失导致，拆分查询并继续；未修改代码或产物 |
| 独立exp06运行在AirFogSim导入阶段触发`UnicodeEncodeError` | 1 | 根因为PowerShell子进程GBK无法打印依赖中的Unicode符号；固定`PYTHONIOENCODING=utf-8`后重跑 |
| UTF-8重跑使用base Python并缺少`osmnx` | 2 | 已验证既有`airfogsim` Conda环境包含`osmnx 2.0.7`；后续通过`conda run -n airfogsim`执行 |
| 全量749项测试有3项失败 | 1 | 746项通过；失败为既有`文档/项目说明`目录缺失1项及两个文档审计脚本路径乱码，均不涉及本次构图/指标代码；保留现场并单独报告 |

## AirFogSim多seed开发数据集v2（2026-07-31）

### Goal

在不恢复旧任务节点语义、不启动正式训练的前提下，将已验证的单轨迹v2流水线扩展为3-seed开发数据集，冻结逐轨迹双图/资源/指标、seed级split和8→3窗口索引，为下一步张量化与训练smoke提供可审计输入。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 冻结多seed数据集与窗口契约 | complete | seed级隔离、时间连续、右删失和字段来源规则 |
| 2. 实现聚合器与合法性测试 | complete | `airfogsim_dataset_v2.py`及测试 |
| 3. 实现单次运行生成器 | complete | 同一次仿真同时产出源双图证据、逐时隙任务状态、动作账本和资源台账 |
| 4. 生成seeds 0/1/2开发样本 | complete | 三条12秒轨迹、指标表、窗口索引与manifest |
| 5. 分布审计与文档同步 | complete | 跨seed统计、数据缺口、formal-training就绪结论 |

### Scope Decisions

- 首轮只做`development_smoke`，seed 0/1/2分别标为`dev_train/dev_validation/dev_calibration`，不创建或冒充锁定test。
- 每个窗口历史8步、未来3步，只在单个seed内部切片；不跨轨迹、不随机打散相邻时隙到不同split。
- 先保存可变规模图与窗口索引，不强制转成旧`x_node/x_link/x_task`固定张量；张量化必须另行冻结节点/边/流的padding与mask契约。
- 每个seed只运行一次以控制本周成本；seed 0已有独立同seed复现证据，多seed包本身标明`single_pass_development_smoke`。

### Result

- `代码/artifacts/datasets/airfogsim_multiseed_v2_dev/`已生成seeds 0/1/2、330个seed隔离的8→3窗口，全部开发数据门通过。
- 定位AirFogSim `getOffloadingTasksWithNumber()`的列表别名副作用；PI-JWM采集适配层使用无副作用合并视图，未修改第三方核心源码。seed 1独立复现和三seed重跑通过。
- 三seed任务流守恒、CPU容量和UAV能量方程违例率均为0；`development_dataset_ready=true`。后续张量化已完成，锁定测试集和正式复现仍缺，因此`formal_training_ready=false`。

## AirFogSim双图张量v2与训练smoke（2026-08-01）

### Goal

将三seed可变规模双图开发数据转成无未来泄漏、可训练、可审计的固定张量，并完成一轮action-conditioned多步rollout训练与真实指标计算。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 修正断连cloud与多seed语义 | complete | 18/16/17个实际接入节点，不再把无物理边cloud计入图 |
| 2. 冻结张量契约 | complete | 18节点、306物理边、216信息流、163任务、422 DAG边，8→3窗口 |
| 3. 实现懒加载与训练统计 | complete | seed按需加载、train-only归一化、padding保持为0 |
| 4. 生成三seed张量数据 | complete | `airfogsim_tensor_v2_dev`、校验报告、manifest |
| 5. 实现最小双图rollout模型 | complete | 四类状态及存在性输出、mask-safe损失、反向传播测试 |
| 6. 运行一轮smoke并同步文档 | complete | seed0训练、seed1验证、seed2校准及字段级评价报告 |

### Result

- 张量开发集330个窗口全部通过引用、有限值、padding、窗口和容量检查；seed0训练统计含11,106个节点、137,516个物理边、1,160个信息流和49,817个任务有效观测。
- 一轮训练损失为0.701065；seed1/seed2链路活动F1均为0，active-only速率MAE分别为354.606536/452.419688，信息流存在F1为0.016876/0.026939，任务存在F1为0.388395/0.186794。
- 结论仅为`smoke_ready=true`：流水线成立，但稀疏链路活动、信息流和任务演化尚未学好。下一步先做持久性/零活动基线、类别不平衡处理和任务生命周期预测，再比较耦合JEPA结构。

## AirFogSim稀疏事件四臂诊断v2（2026-08-01）

### Goal

在不接入新数据源和JEPA的前提下，用零活动、末值保持、未加权学习和训练集加权学习四个实验臂，判断最小PI-JWM rollout是否真正超过简单规则，并把链路活动、信息流、任务生命周期和物理单位状态误差纳入同一评价口径。

| Phase | Status | Deliverable |
|---|---|---|
| 1. 训练集稀疏标签统计 | complete | 显式链路活动标签、正类比例、封顶权重和生命周期多数类 |
| 2. 模型输出与损失补充 | complete | 链路活动头、五类生命周期头和mask-safe多任务损失 |
| 3. 简单基线与统一评价 | complete | 零活动、末值保持、AUPRC及物理单位指标 |
| 4. 四臂公平运行器 | complete | 相同初始化、样本顺序、5轮配置和统一split评价 |
| 5. 实测与文档同步 | complete | exp08产物、四臂结果表和go/no-go结论 |

### Result

- `dev_train`链路活动与信息流存在正类比例仅0.007538和0.008899，正类权重均触发50上限；任务存在比例0.391950，权重1.551345。
- 加权臂在验证/校准上的链路AUPRC为0.044855/0.045419，均高于未加权臂和末值保持；但链路F1、active-only速率MAE、任务存在F1和生命周期macro-F1仍未稳定超过末值保持。
- 状态冻结为`diagnostic_ready=true`、`formal_training_ready=false`、`jepa_comparison_ready=false`。下一阶段先强化基础状态转移和持续性建模，不提前扩展耦合JEPA。

## Goal

Use train/calibration/validation only to diagnose and improve the candidate selector, freeze one defensible method, then evaluate the locked method once on external holdout seeds 60-69. Historical matched test seeds 18-19 remain locked.

## Acceptance Gates

- Validation must improve over ranked default RMSE 233.7162005 on a clear majority of seeds.
- Selected candidates must have positive realized-benefit precision and a controlled negative-selection rate.
- The method must retain task-energy safety and use only deployable features at inference.
- External holdout is opened only after configuration freeze.
- Any result reported to the advisor is labelled deployable, sample_oracle, diagnostic_only, or external_holdout.

## Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Reconstruct the completed GPU run | complete | 36 checkpoints and formal validation result available locally |
| 2. Attribute failure to ranking, uncertainty, Pareto, or defer | complete | 12-config x 6-policy validation audit |
| 3. Form and test one root-cause hypothesis | complete | Opportunity is identifiable; candidate ranking needs token-level interactions |
| 4. Implement the selected method with TDD | complete | Tests, reusable selector code, runner changes |
| 5. Local smoke and full validation checks | complete | Formal phase-selector reproduction plus 722 main and 84 script tests |
| 6. Sync and run the necessary GPU experiment | complete | CUDA smoke plus no-phase/phase-aware 3-seed probes completed |
| 7. Freeze on validation and evaluate external holdout once | complete | A gate not met; external was correctly kept locked and unaccessed |
| 8. Update report artifacts and PPT data placeholders | complete | CSV/JSON/NPZ/figures, manifest, and six-page PPT planning text updated |

## 2026-08-14 P2-C Advisor-Document Manifest Binding

| Task | Status | Evidence |
|---|---|---|
| RED: prove the progress document is not bound | complete | 2/2 focused tests failed for the expected missing-key and tamper-not-detected reasons |
| GREEN: add one canonical source path | complete | P2-C test pair passed 9/9; Python compile, Ruff, and diff checks passed |
| Rebuild and promote canonical audit | complete | Core audit/config JSON stayed byte-identical; old canonical was archived; promoted `--verify-only` passed |
| Complete evidence gate | complete | P2-B/P2-C verify passed, AirFogSim 83/83 matched, focused suite passed 159/159 |

The four P2-C blockers remain unchanged: `action_rejection_rate_not_observed`, `scenario_matrix_not_frozen`, `formal_scale_not_frozen`, and `formal_split_not_frozen`. No GPU task, formal trajectory generation, or locked-test access is authorized by this closure.

## Outcome

- Best validation result: B-grade RMSE 207.5399 versus 233.7162 ranked baseline.
- All 10 validation seeds improved; positive execution precision was 93.85% with zero Pareto violations.
- The pre-registered A-grade RMSE <200 gate was not met. Configuration remains a v11 candidate, and external seeds 60-69 remain unaccessed.

## Documentation Synchronization (2026-07-20)

- [x] Record the phase-conditioned benefit LCB validation result and its evidence boundary in the research progress document.
- [x] Add the selector root-cause analysis, method definition, validation table, and limitations to the paper draft.
- [x] Recompile both XeLaTeX documents and inspect the generated PDFs before treating the documentation update as complete.

## New Conversation Handoff (2026-07-21)

- [x] Create a root-level project handoff document that preserves evidence, data locks, code entry points, and current research boundaries.
- [x] Keep the next method explicitly open instead of prescribing the current phase-conditioned selector.
- [x] Include a starter prompt that a new conversation can use to reconstruct context before proposing new ideas.

## Fixed Protocol

- Train seeds: 0-15, 20-43.
- Calibration seeds: 44-49.
- Validation seeds: 50-59.
- Historical matched test seeds 18-19: never reopen in this iteration.
- External holdout seeds 60-69: one evaluation after validation freeze.
- Actual UAV energy is audit-only; online decisions use physical task LCB and deployable energy proxy.

## Errors Encountered

- 2026-08-03：首次正式重建候选评测包时，验证失败于 `checkpoint_continuous_term_executable`。根因不是数据损坏，而是错误地在“单轨迹”层要求十个连续分量全部可计算；多数轨迹没有连续两时隙均存在的数据流，导致 persistence 的 `state.flow.remaining_data.mae` 合理地为 `not_computable`。修正方向：先在完整环境轨迹集合上汇总每个分量，再计算验证集 checkpoint normalized error。
- 2026-08-03：按项目规范尝试通过本地 `zotero-cli` 对四篇评价协议依据做 DOI/题名去重检索，CLI 连续 60 秒无输出并超时，进程随后已退出。未执行任何 Zotero 写入；本轮先以出版社/会议/标准组织的一手页面固定证据，避免重复导入。
- 2026-08-03：全仓 `unittest discover` 共运行 907 项，904 项通过、3 项失败。失败均来自工作开始前已存在的用户侧文档删除：`文档/项目说明/` 不存在，以及 `文档/研究进展/audit_tables.py`、`audit_table_numbers.py` 不存在；与本轮 R2 代码无关，未擅自恢复用户删除内容。R2 的 13 项定向测试全部通过。
- 2026-08-03：增加lifecycle上界防护后的正式重建发现，真实张量允许`task_present=true`但lifecycle为`-1`（已出现但阶段未知）；最初的验证错误地把它当损坏数据并中止。修正为：`-1`作为unknown由mask排除，`0..4`参与五类评价，`>=5`或`<-1`才拒绝。
- 2026-08-03：首次正式目录提升的只读前置门错误使用`$obj.PSObject.Properties.Count`，在PowerShell中未得到预期整数，因而安全中止，未移动任何目录。改为`@($obj.PSObject.Properties).Count`后再执行。

| Error | Attempt | Resolution |
|---|---:|---|
| Formal CandidateSet grid deferred 100% on validation | 1 | Root-cause attribution is in progress; no threshold change yet |
| Parallel XeLaTeX recompilation hit a shared PDF lock | 1 | Recompiled the two documents sequentially; both completed with exit code 0 |
| Inline attribution could not import `pi_jwm` because the piped Python process did not use the requested working directory | 1 | Use explicit absolute source/script paths for the diagnostic process |
| Explicit Unicode source path was still not visible to the piped Python process | 2 | Stop retrying the attribution command; probe cwd/path/encoding and use an environment-level `PYTHONPATH` or ASCII launcher |
| Piped script body converted the Chinese `代码` path segment to `??` before any model load | 3 | Abandon Unicode literals in stdin; inject the complete code root through `PI_JWM_CODE` and keep the Python body ASCII-only |
| `git bundle create` rejected a bare commit range as an empty ref set | 1 | Export the named `main` ref while excluding the remote base commit |
| Candidate-expert prototype requested unsupported HGB regressor loss `huber` | 1 | Use the installed sklearn's supported robust `absolute_error` loss |
| Phase-table prototype could not import the shared script metric helper | 1 | Add the repository scripts directory to `PYTHONPATH`; no experiment rows were evaluated before failure |
| Dense phase-table search could not JSON-serialize NumPy scalar types after selecting calibration parameters | 1 | Keep the already fixed calibration parameters and rerun validation formatting with scalar conversion only |

## Project Understanding Review (2026-07-21)

### Goal

Reconstruct PI-JWM from the repository, local research documents, implementation, artifacts, and related literature; distinguish established evidence from inherited assumptions and develop independent research directions.

### Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Repository and document inventory | complete | Authoritative entry points, document set, and evidence map |
| 2. Framework and experiment reconstruction | complete | Data-model-objective-evaluation chain grounded in code |
| 3. Local literature review | complete | Prior-work matrix and novelty boundaries |
| 4. Independent critique and alternatives | complete | Bottlenecks, hidden assumptions, and new hypotheses |
| 5. Synthesis and verification | complete | Detailed project explanation with source pointers |

## Four-Day Research Definition Sprint (2026-07-22 to 2026-07-25)

### Goal

Define the final PI-JWM optimization problem, map it to observable data and defensible theory, assess realistic external data, and close a minimal multi-metric evaluation loop without reopening locked test sets.

| Phase | Status | Deliverable |
|---|---|---|
| 1. Verify current state/action/target semantics | in_progress | Evidence-backed system boundary and objective candidate |
| 2. Formalize constraints and proof route | pending | Assumptions, constraints, propositions, and evidence boundary |
| 3. Survey realistic datasets and literature | pending | Dataset-field compatibility and use decision |
| 4. Define and audit multi-layer metrics | pending | Metric formulas, field availability, and missing-data map |
| 5. Minimal metric computation and preregistration | pending | Reproducible validation-only metric smoke and next experiment protocol |

### Sprint Rules

- Daily blocks are guidance, not mechanical gates; proceed as soon as dependencies are clear.
- Explain purpose, inputs, operation, and observed output before treating a step as understood.
- Do not access matched test seeds 18--19 or external seeds 60--69.
- Do not promote future-action references, oracle results, or selector diagnostics into autonomous-control evidence.

### Review Rules

- Treat AirFogSim only as a simulator/data source; PI-JWM is the research framework.
- Separate implemented behavior, measured evidence, planned claims, and speculation.
- Preserve train/calibration/validation/test locks when interpreting experiments.
- Do not promote v5/v11 selector diagnostics to the main method.
- Do not modify project code or `本地计划表.md` during this read-only review.

### Review Errors

| Error | Attempt | Resolution |
|---|---:|---|
| PDF skill path was resolved under the system skill root and did not exist | 1 | Read the plugin-cached PDF skill from the declared `r3` root |
| PowerShell displayed UTF-8 Chinese README text as mojibake | 1 | Re-read text with explicit UTF-8 console/output handling |
| Parallel literature inventory returned exit code 1 because the TeX grep pattern did not match | 1 | Replaced the grep with a PowerShell reference-section inspection and listed literature separately |
| `web-access` CDP preflight could not connect to Chrome | 1 | Started Chrome in the background as directed and retried |
| Chrome/CDP retry waited until the command timeout | 2 | Avoid browser automation; use public academic APIs and first-party paper pages without login state |
| First OpenAlex batch command had an invalid PowerShell pipeline after `foreach` | 1 | Materialize result rows before formatting; no request was made before the parse failure |
| OpenAlex PowerShell batch returned empty result arrays despite a successful response | 2 | Verified the API with raw `curl`; switch to metadata-only responses to avoid parser ambiguity |
| arXiv Atom response for locally cited 2026 IDs contained no entries | 1 | Treat those preprints as unverified and cross-check exact titles in independent indexes |
| PowerShell `ConvertFrom-Json` rejected OpenAlex works with case-insensitive duplicate abstract keys | 1 | Request only id/title/year/DOI fields using OpenAlex `select` |
| Decision-trace grouping used nonexistent `selected_candidate` and `deferred` columns | 1 | Recompute with actual `candidate_name` and `executed` columns from the CSV header |
| Expected dataset metadata filename did not exist | 1 | Read the actual `world_model_dataset_v0_summary.json` listed in the directory |
| Base world-model NPZ did not contain `sample_id` | 1 | Validate 60 contiguous seed blocks of 390 samples and compute within-seed local indices explicitly |

## PI-JWM Literature Library and Workspace Consolidation (2026-07-22)

### Goal

Consolidate PI-JWM literature in the user's Zotero `PIJWM` collection, classify and deduplicate the local and previously surveyed corpus, acquire lawful full text where available, and reorganize the workspace so that only `代码/` and `文档/` remain as user-content roots without losing authoritative project context.

| Phase | Status | Deliverable |
|---|---|---|
| 1. Read-only inventory | in_progress | Zotero connectivity, local paper corpus, root/document tree, and git-state audit |
| 2. Classification and retention proposal | pending | Literature taxonomy plus keep/archive/delete candidates with reasons |
| 3. User confirmation for destructive cleanup | pending | Explicitly approved deletion/move scope |
| 4. Literature metadata verification and lawful acquisition | pending | Deduplicated recent/top-venue and foundational paper set |
| 5. Zotero import and classification | pending | Items, attachments, collections/tags, and duplicate audit in `PIJWM` |
| 6. Workspace reorganization | pending | Approved moves/deletions, root reduced to the requested structure |
| 7. Verification and handoff | pending | Zotero counts, attachment checks, filesystem manifest, and future-ingest rules |

### Guardrails

- Do not delete or move files during the read-only inventory.
- Treat “outdated” as a review label, not deletion authorization for individual files, until the candidate manifest is shown to the user.
- Preserve Git metadata and configuration required for the repository to function, even though they are not user-content folders.
- Do not write directly to Zotero's SQLite database while Zotero is running; prefer supported local API, translators, or UI import.
- Download only lawful open-access copies or copies available through user-authorized institutional access; disable Sci-Hub fallback.
- Do not mix PI-JWM's main world-model literature with the historical v5/v11 selector-diagnostic branch without explicit tags.

### Audit Errors

| Error | Attempt | Resolution |
|---|---:|---|
| Recursive search for `zotero.sqlite` across the user profile and entire `D:\` drive exceeded 30 seconds | 1 | Stop the broad scan; resolve the active profile from `profiles.ini`/`prefs.js` and query only configured paths |
| Zotero local API requests to `127.0.0.1:23119` were closed unexpectedly | 1 | Inspect the sockets owned by Zotero and retry only against the actual bound endpoint; no database mutation attempted |
| First Zotero documentation browser call parsed the CDP response as `.id`, but the current proxy returns `.targetId` | 1 | Inspect the raw `/new` response, switch to `.targetId`, and close the created tab after extraction |
| First 27-paper OA batch returned partial success (12 PDFs, 15 unresolved) | 1 | Preserve all verified metadata, check author/repository OA locations, and label any remaining items as metadata-only; do not use illicit sources |
| Direct OpenAlex `/works/{encoded DOI URL}` calls failed for every unresolved DOI | 1 | Switched to the documented `filter=doi:` query form, which returned OA status and locations correctly |
| DOM evaluation on Chrome's built-in PDF viewer returned `Uncaught` | 1 | Use target info/screenshot and normal viewer controls; do not retry DOM access against the extension page |

## RRM Review-Based PI-JWM Reconstruction (2026-07-24)

### Goal

Read the two RRM/world-model review documents from beginning to end, extract their defensible guidance on scenario, state, action, objective, hard/soft constraints, prediction targets, and evaluation, then compare that guidance with PI-JWM before proposing any project restructuring.

### Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Folder and document inventory | complete | Two PDFs, PPTX, and speaker notes identified |
| 2. Full review extraction and chapter map | complete | Both language versions read through conclusion and references; shared section structure confirmed |
| 3. RRM-to-PI-JWM comparison | complete | Implemented-versus-missing matrix for scenario/state/action/objective/constraints/metrics |
| 4. Rebuild proposal | pending | Minimal coherent project definition and decisions requiring user confirmation |
| 5. Documentation/code impact plan | pending | Files and experiments affected; no edits until approved |

### Guardrails

- Do not modify PI-JWM definitions, code, experiments, or the user's knowledge-base Markdown during this review phase.
- Treat AirFogSim as a simulator/data source, not as the PI-JWM framework.
- Separate what the reviews recommend from what PI-JWM currently implements.
- Do not assume the project should become cell-free, channel-only, or an RRM benchmark without checking the review's actual scope and the current data fields.

## 2026-08-03 Strict Dual-Graph Literature Freeze

| Phase | Status | Deliverable |
|---|---|---|
| Verify the two-network structural precedent | complete | Yağan et al., IEEE TPDS 2012: two intra-layer networks plus explicit inter-network dependency links/matrix |
| Verify wireless edge semantics | complete | Shen et al., IEEE JSAC 2021: channel states are wireless communication-graph edge features |
| Separate direct evidence from PI-JWM-specific design | complete | Physical spatial-neighborhood rules, composite agents, DAG auxiliary status, and action-conditioned rollout remain experimental PI-JWM choices |
| Persist the evidence | complete | Main document, evidence matrix, Zotero audit, and R0 plan updated; Zotero cloud verified, local sync pending |

## 2026-08-03 R2 Final Audit Status

| Phase | Status | Deliverable |
|---|---|---|
| Metric/data/statistical audit | complete | 43 canonical metrics, 22 factual mappings, executable checkpoint rule |
| Primary-literature grounding | complete | Sparse-event, proper-score, calibration, split, fairness, and network-KPI evidence with explicit scope limits |
| Semantic contract binding | complete | Upstream hashes plus exact ordered checks for four R1 graph feature lists and the frozen dual-graph protocol |
| Formal rebuild and independent review | complete | 54 non-locked evaluated, 6 locked sealed, 62 inputs/7 code files/13 outputs verified; final DAG column-semantics correction and rebuild passed |
| Documentation and R3 gate | complete | Canonical artifact promoted and R2 marked complete; R3 may begin without changing the frozen R2 protocol |

## 2026-08-04 R3 New-Semantics CPU Preflight Status

| Phase | Status | Deliverable |
|---|---|---|
| Read-only dynamic windows and explicit batch | complete | 8-step history plus 1/5/20-step targets, train-only normalization, raw activity labels, locked-test rejection |
| Explicit/latent reference model | complete | Separate physical/information encoders, `CIP/CEP/CFL` coupling, explicit predictions and four latent belief groups |
| Objective and strict checkpoint | complete | Masked continuous/presence/activity/lifecycle terms, N/A semantics, gradient coverage, provenance-bound strict reload |
| Formal nonlocked CPU run | complete | 9 real windows, 2 controls, 18 objectives, 18 R2 interface records, all finite and hash verified |
| Claim boundary and next gate | complete | `r3_cpu_preflight_ready=true`; no convergence/superiority claim; proceed to controlled R4 GPU module screening |

## 2026-08-04 R3 Literature and Correctness Re-Audit

### Goal

Re-audit R3 against primary literature and the frozen R1/R2 contracts so that every implemented design choice has an explicit support boundary, every PI-JWM-specific choice is labeled as experimental, and any code/claim mismatch is fixed before R4.

| Phase | Status | Deliverable |
|---|---|---|
| 1. Claim-to-code inventory | complete | Claim/code/test matrix completed; historical-action, relation-validation, DAG-summary and fixed-presence boundaries identified |
| 2. Primary-literature verification | complete | PlaNet, GNS, JSAC wireless GNN, TPDS interdependent networks and TWC coupled JEPA checked from original/authoritative sources |
| 3. Gap and risk audit | complete | Frozen principles separated from analogies and PI-JWM hypotheses; DAG-edge and predicted-topology work explicitly assigned to R4 |
| 4. Corrective implementation and rerun | complete | TDD added historical-action belief and strict relation gates; formal 9-window/2-control canonical artifact regenerated |
| 5. Evidence persistence | complete | Audit document, design, main document, local plan, findings and Zotero records updated; cloud and local Zotero reread passed |
| 6. Independent verification | complete | 22/22 R3 tests pass; 930/933 full-repo tests pass with only the same three pre-existing user-deletion failures; compile, manifest/source hashes and document consistency pass; verdict Ready within the frozen R3 boundary |

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Windows `rg` rejected the literal wildcard path `代码/tests/test_r3_*.py` | 1 | Use `rg` on the tests directory with `-g 'test_r3_*.py'` or enumerate files explicitly; do not repeat the invalid path pattern |
| First formal rerun used an intentionally short shell timeout and was terminated | 1 | Treat as an interrupted run, not model evidence; rerun in a fresh candidate directory with a 180-second bound |
| Canonical runner refused a non-empty output directory | 1 | Preserve the guard; generate a fresh candidate, verify it, then copy its exact manifest-bound files into the canonical directory |
| Recursive and fixed-file cleanup commands for the verified candidate duplicate were blocked by the execution policy | 2 | Do not bypass the policy; leave the noncanonical duplicate outside the documented formal path and report it explicitly |

## 2026-08-06 R5多seed统计收口

| Phase | Status | Deliverable |
|---|---|---|
| 1. 正式产物验收 | complete | 15/15 run、86项清单、15个最佳checkpoint、locked-test边界全部通过 |
| 2. 配对统计实现 | complete | 相对A的结构比较、相对B的模块增量、三seed描述统计、t区间、胜负数和精确符号翻转检验 |
| 3. 正式统计产物 | complete | `pi_jwm_r5_multi_seed_analysis_v2`生成JSON/CSV/Markdown及自校验manifest |
| 4. 方法判断 | complete | B保留为工作候选、A保留为控制、C/D转诊断、E退出主路径；不自动指定winner |
| 5. R2全指标复评 | pending | 不确定性、分horizon、任务/资源系统结果、动作/守恒和运行时指标闭合后再审查R6入口 |

## 2026-08-08 R6在线GPU入口

| Phase | Status | Deliverable |
|---|---|---|
| 1. 状态/reward语义审计 | complete | 旧冻结状态闭环作废；反事实reward代理因候选支持1/6而no-go |
| 2. 在线双图实现 | complete | 每步AirFogSim重采集、最近8步严格双图、显式＋冻结隐式状态 |
| 3. 动作与日志正确性 | complete | 实际卸载/RB/CPU日志、CPU回调单次执行、硬约束回归 |
| 4. CPU闭环门禁 | complete | 六组合、32步、确定性、真实事件、checkpoint重载均通过 |
| 5. 恢复与选模 | complete | 原子2→4步续训、validation-only best checkpoint通过 |
| 6. 正式矩阵准备 | complete | 18-run隔离启动器、6并发默认、10k阶段dry-run和汇总协议完成 |
| 7. GPU smoke | pending | 开GPU后先跑单run 2k并测显存/吞吐；通过后启动10k×18 |
# 2026-08-13 P2 多步时序与跨步词表门

| Phase | Status | Deliverable |
|---|---|---|
| 1. 单步采样时序复核 | complete | 证实 event attenuation 在动作 setter 与当前槽快衰落更新之后采集，不能宣称 action-pre |
| 2. 语义修订设计 | complete | 冻结 decision-time CSI 与 outcome-side channel snapshot 的分离口径，设计已获用户书面确认 |
| 3. 单步证据修订 | complete | TDD修正字段来源，旧bundle可恢复归档，修正版原子bundle通过 |
| 4. 多步纯契约 | complete | append-only node/edge/flow vocabulary、E1 prev outcome、事务提交和失败原子性通过 |
| 5. 真实非训练轨迹 smoke | complete | seed 0三帧CPU trajectory、双verify-only、77项回归、依赖闭包与独立SHA-256复算通过；不批准正式数据集或训练 |

Implementation plan: `docs/superpowers/plans/2026-08-13-p2-multistep-temporal-contract.md` (executed on 2026-08-13; Tasks 1-5 complete).

## 2026-08-14 P2-C AirFogSim source-closure recovery

| Phase | Status | Deliverable |
|---|---|---|
| 1. Reproduce manifest gap | complete | P2-B expects 83 AirFogSim files; current project reference directory is empty |
| 2. Evaluate local archive | complete | `AirFogSim_clean_runnable.zip` matches 82/83; only `energy_manager.py` differs |
| 3. Recover exact remaining file | complete | Historical patch replay reproduces expected `energy_manager.py` SHA-256 exactly |
| 4. Restore external checkout | complete | Restored into ordinary workspace directory; target independently matches 83/83 |
| 5. Re-verify evidence chain | complete | 83/83, P2-B/P2-C verify-only, and UTF-8 158/158 focused tests passed |
| 6. Bind the P2-C advisor-facing audit document | complete | RED/GREEN implementation, canonical rebuild, document-tamper rejection, dual verify, 83/83 hashes and 159/159 focused tests passed |

Guardrails: no GPU, no locked test, no formal trajectory generation, no dirty-worktree cleanup, and no source-closure claim before 83/83 exact hashes.

### Recovery errors

| Error | Attempt | Resolution |
|---|---:|---|
| Combined `rg` expression for AirFogSim Git state had an unclosed character class | 1 | Treat the command failure as a regex error only; switch to fixed-string queries and do not infer absence from the failed search |
| `rg` interpreted search text `--verify-only` as a flag | 1 | Add the `--` end-of-options separator; the corrected fixed-string search located the commands |
| P2-C verify-only omitted required `--bundle`/`--output-dir` | 1 | Read the CLI usage, rerun with canonical input/output paths; verification passed |
| First restored 151-test run had 5 `UnicodeEncodeError` failures at AirFogSim's emoji import print under GBK | 1 | Do not alter third-party source; rerun the identical suite with `PYTHONUTF8=1` to test the environment-root-cause hypothesis |
| Historical-session extraction used a mistyped session filename | 1 | Corrected the exact archived filename and extracted the original 151-test command |
| Inspection assumed a top-level `evidence_gates` object in the P2-C report | 1 | Read actual top-level keys; source closure is represented by `blocking_reasons`, not an `evidence_gates` property |
# 2026-08-15 新对话迁移接续文档

- [x] 读取现有接续说明、磁盘计划和工作树状态。
- [x] 核验 P2-B v2、P2-C v2 artifact、证据文档和主计划表的最新事实。
- [x] 新建 2026-08-15 接续说明，明确权威入口、分支状态、已证实事实、阻断与下一步。
- [x] 校验新文档路径、内容、差异和 Git 状态，不触碰现有用户改动。

错误记录：首次读取 P2-C v2 时沿用摘要中的通用文件名 `audit_report.json` 与 `formal_dataset_candidate_config.json`，真实文件名不同；已根据目录清单改用 `p2c_scale_distribution_audit_v2.json` 与 `p2c_formal_data_config_candidate_v2.json`，不重复错误命令。首次登记本条记录的补丁因多文件 hunk 格式错误被 `apply_patch` 拒绝，已改为合法的独立 hunk。
