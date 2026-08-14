# PI-JWM P2联合动作Attempt/Reject Ledger v1设计

日期：2026-08-14
状态：已获用户分节确认，待实施计划
范围：CPU-only、非训练、非locked-test、P2-B v2采集证据与P2-C v2审计

## 1. 背景与问题

当前P2-B canonical只保留成功发布的自然frame、coverage、replay、validation和quarantine状态。它能够证明已保存frame中的动作通过合同并完成真实AirFogSim step，但不能证明候选构造、合同验证、setter调用或`env.step()`失败是否曾在发布前被丢弃。

P2-C当前仅尝试从`validation_report.json`读取`action_rejection_count`。旧bundle没有完整attempt分母，审计也无法从0个quarantine或120个成功frame反推出拒绝率为0。因此`action_rejection_rate_not_observed`是真实阻断，不能通过补一个汇总数字、回填旧frame或假造120条成功记录关闭。

本设计建立逐联合动作候选的机器可读ledger，使每个attempt拥有稳定身份、可验证状态路径、真实runtime调用证据和唯一终态；成功与失败发布均绑定manifest。该ledger是数据采集审计基础设施，不是世界模型、策略器或候选rollout规划器。

## 2. 已确认边界

### 2.1 Attempt计数单位

一个attempt是一帧内提交给采集执行链的一份完整联合动作候选，而不是单个任务、RB、flow、setter或CPU分量。当前P2-B每个时隙只允许一个候选，即`candidate_ordinal=0`。

未来真实候选rollout规划器如需每帧评估多个候选，必须另行升级协议；本设计不提前把单候选采集器描述为规划器。

### 2.2 指标分层

`run_role`严格分为：

- `natural_reference`：主拒绝率分母，与发布的自然frame一一对应；
- `natural_replay`：只用于可复现性诊断；
- `fixture`：受控边界验证动作；
- `bootstrap`：为fixture准备真实环境状态的动作。

主拒绝率只由`natural_reference`计算。replay、fixture和bootstrap必须分别报告，不得混入自然分布指标。

### 2.3 二元守恒与详细终止原因

顶层`disposition`只有`accepted`和`rejected`，必须满足：

```text
attempt_count = accepted_count + rejected_count
```

详细失败位置由`terminal_stage`、runtime调用记录、mutation状态和quarantine状态表达。setter或step之后的失败仍计为rejected，但不能伪装成执行前拒绝。

## 3. 方案比较与选择

### 3.1 选择：独立ledger状态机

新增纯PI-JWM ledger模块，由v2 runner在候选构造前创建attempt，v2执行适配器记录验证、setter、step和结果采集边界。它覆盖构造失败、验证拒绝、部分setter失败和step后失败，并能被未来多候选协议复用。

### 3.2 拒绝：只扩展`FullCollectorStepResult`

候选构造或验证可能在返回result前抛异常，因此仅扩展result仍会丢失失败attempt，无法建立完整分母。

### 3.3 拒绝：runner外层只记异常

外层异常捕获看不到真实setter调用顺序、成功次数、`env.step()`边界和环境副作用，只能猜测失败阶段，不满足理论—实现—证据一致性。

## 4. 版本隔离与文件边界

加入`action_attempts.jsonl`会改变P2-B受管文件矩阵，因此必须升级artifact schema，不能继续称为v1。

旧P2-B v1代码和artifact保持不动，继续作为历史preflight证据；不得向旧bundle回填ledger。新实现使用独立v2文件，避免修改P2-B v1 manifest绑定的源码后让当前v1 source closure失效：

- 新建`代码/src/pi_jwm/action_attempt_ledger_v1.py`：纯状态机、记录类型、验证与汇总；
- 新建`代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v2.py`：对真实v1 executor做实例级runtime观测，记录setter、step和observer调用；
- 新建`代码/src/pi_jwm/full_dual_graph_artifact_v2.py`：v2受管文件、语义验证、成功/失败原子发布；
- 新建`代码/scripts/run_p2_full_dual_graph_collector_preflight_v2.py`：v2自然reference/replay、fixture/bootstrap编排；
- 新建对应v2测试，不改写v1测试含义；
- 新建`代码/src/pi_jwm/p2c_scale_distribution_audit_v2.py`与v2 runner/test，读取P2-B v2；
- v2可导入v1纯合同、observer、frame builder、vocabulary、coverage和frame验证逻辑，但不得修改v1模块全局对象或用module-level monkeypatch替换executor。

v2执行适配器通过实例级、可恢复的scheduler代理、`env.step`包装和observer包装记录实际调用；所有包装均在`finally`中恢复。它记录真实方法调用，不以推断量代替runtime证据。v2 runner只复制无法通过显式依赖注入复用的边界编排，不能复制或改写理论合同。

新schema与目录为：

```text
PIJWM-Full-Dual-Graph-Artifact-v2
代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814/
代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2/
```

P2-C v2通过候选目录验证后再提升；旧P2-C canonical先归档，不原地伪造ledger。

## 5. Ledger类型与字段

### 5.1 Attempt身份

身份字段为：

```text
run_role
episode_id
trajectory_id
frame_index
candidate_ordinal
```

`attempt_id`由上述字段的规范化UTF-8 JSON生成SHA-256，并带可读前缀。相同身份必须产生相同ID，不同run role必须产生不同ID。所有字符串非空，`frame_index`与`candidate_ordinal`为非布尔非负整数。

### 5.2 Terminal记录

每条`action_attempts.jsonl`至少包含：

```text
schema_version
attempt_id
run_role
episode_id
trajectory_id
frame_index
candidate_ordinal
candidate_digest
stage_trace
setter_calls
env_step_called
env_step_completed
disposition
terminal_stage
rejection_code
rejection_detail
environment_mutation_status
quarantined
training_eligible
```

`candidate_digest`是完整联合动作经dataclass/enum/plain-value转换、稳定键排序和UTF-8编码后的SHA-256。候选未成功构造时必须为`null`，不得根据计划值伪造。

`setter_calls`逐次记录：

```text
ordinal
setter_kind
task_id
call_started
call_completed
succeeded
error_type
error_detail
```

`setter_kind`覆盖CPU callback安装、offload、return route和RB setter。汇总调用数和成功数必须从明细重新计算，不单独信任手填计数。

### 5.3 Mutation三态

`environment_mutation_status`只允许：

- `none`：任何runtime action call前拒绝，可证明动作没有修改环境；
- `confirmed`：至少一个setter完成成功或已经调用`env.step()`；
- `unknown_after_runtime_call`：setter调用开始后异常，但无法证明其是否部分修改环境。

一旦调用过runtime action接口，失败记录不得声明`none`。任何非`none`的rejected attempt必须`quarantined=true`。

## 6. 状态机与一致性规则

合法成功路径为：

```text
begun
→ candidate_built
→ contract_validated
→ pre_setter_revalidated
→ setters_applied
→ env_step_started
→ env_step_completed
→ outcome_captured
→ accepted
```

无setter动作的合法帧仍需经过`setters_applied`，但`setter_calls`可为空。`stage_trace`必须是合法路径的连续前缀，不能跳步、倒退或在terminal后追加。

一致性门如下：

1. accepted必须有candidate digest、通过两次验证、step started/completed、execution和outcome证据；终止于`outcome_captured`，无rejection code，非quarantine。
2. 构造、首次验证或setter前复验失败时，setter列表为空、step未调用、mutation为`none`且非quarantine。
3. setter调用后失败时，mutation为`confirmed`或`unknown_after_runtime_call`，必须quarantine，不得step或重试。
4. step或step后取证失败时，记录真实called/completed组合，mutation为`confirmed`，必须quarantine。
5. 每个attempt只能终结一次；重复ID、重复terminal和非法转换均拒绝。
6. 每个run role独立满足二元守恒。
7. 当前natural reference每个`trajectory_id/frame_index`只能有`candidate_ordinal=0`；任何rejected attempt使本次v2成功发布失败。

## 7. 数据流

自然reference流程为：

```text
decision snapshot
→ begin attempt
→ runtime inputs
→ build candidate
→ candidate digest与首次合同通过
→ v2 runtime adapter
→ setter明细
→ env.step边界
→ execution/outcome
→ terminal attempt
→ frame/attempt对齐
```

frame builder内部已有首次合同验证；runner只在builder正常返回后记录`candidate_built`与`contract_validated`。若builder抛`CollectorContractError`，终止阶段为`contract_validation`；其他构造异常终止于`candidate_build`。executor内部的第二次验证由v2适配器观察，任何异常终止于`pre_setter_revalidation`。

decision snapshot在`begin attempt`之前采集；如果连决策状态都无法形成，则属于run-level observation failure，而不是动作候选拒绝，不能进入attempt分母。失败报告必须记录该run-level错误且ledger可为空。

reference和replay使用不同attempt ID但允许相同candidate digest。除现有frame replay比较外，还必须验证attempt数、对应digest和accepted状态一致，且两侧均无隐式重试。

fixture准备动作标记`bootstrap`，真正边界动作标记`fixture`。两者进入总ledger和各自汇总，但不进入natural-reference主拒绝率。

## 8. 成功与失败Artifact

P2-B v2成功bundle受管文件为：

```text
collector_config.json
vocabularies.json
frames.jsonl
action_attempts.jsonl
coverage_report.json
validation_report.json
replay_report.json
status_flags.json
manifest.json
```

`validation_report.json`可包含由ledger计算的分层摘要，但v2 verifier和P2-C v2必须从`action_attempts.jsonl`重新计算，不信任汇总数字。

任何当前协议内的rejected attempt都停止该环境和轨迹，不允许更换候选继续采集。成功目录不生成；可捕获异常原子发布到：

```text
<output-dir>_failed/
├── action_attempts.jsonl
├── failure_report.json
└── manifest.json
```

`failure_report.json`必须包含失败attempt ID、run role、terminal stage、rejection code、quarantine、异常类型/信息以及固定false的训练和正式批准状态。成功目录或`_failed`目录任一存在时，runner必须在创建AirFogSim环境前拒绝覆盖。

失败bundle的manifest绑定两个artifact以及ledger、v2执行器、v2 runner、测试、设计和实际依赖源码。已生成临时目录只有在内容与manifest复算通过后才原子提升。该保证覆盖可捕获的Python、合同、setter、step和取证异常；不宣称对操作系统强制关机或外部kill提供crash-safe journal。

## 9. P2-C v2审计

P2-C v2必须：

- 拒绝没有ledger的P2-B v1作为“已观测拒绝率”证据；
- 验证attempt schema、ID唯一性、状态路径、terminal唯一性和二元守恒；
- 验证accepted natural-reference attempt与frame按`trajectory_id/frame_index/candidate_digest`一一对应；
- 验证rejected attempt没有frame；
- 验证reference/replay对应digest和attempt结构；
- 分离fixture/bootstrap；
- 独立计算natural-reference attempt、accepted、rejected和拒绝率；
- 独立报告quarantine与mutation三态计数；
- 忽略并拒绝用手写`action_rejection_count`替代ledger；
- 绑定P2-B v2输入、P2-C v2代码、测试、设计和研究进展文档。

只有ledger完整且所有门通过时，才能移除`action_rejection_rate_not_observed`。即使该阻断关闭，仍保留：

```text
scenario_matrix_not_frozen
formal_scale_not_frozen
formal_split_not_frozen
```

P2-C v2仍为候选审计，不批准正式数据、训练、GPU、locked test、候选rollout规划器或最终方法。

## 10. 测试矩阵

### 10.1 纯ledger

测试重复ID、重复terminal、非法跳转、字段类型、accepted缺step、rejected缺reason、mutation矛盾、quarantine矛盾以及各run role守恒。

### 10.2 v2 executor

使用现有FakeEnv/SpySchedulers覆盖：RB越界无setter/no-step；首setter异常为unknown并quarantine；部分setter成功后失败为confirmed并quarantine；step异常；step完成后outcome失败；正常动作只step一次并accepted。

### 10.3 v2 runner与artifact

覆盖角色分层、ordinal=0、无重试、成功九文件矩阵、失败三文件矩阵、拒绝覆盖、ledger/artifact/source篡改、atomicity与portable manifest key。

### 10.4 P2-C v2

覆盖缺ledger、重复/删除attempt、伪造summary、frame映射错误、role混入、digest不一致、mutation矛盾、有效0拒绝率和仍保留三个后续阻断。

全部生产实现严格采用RED→确认正确失败→最小GREEN→相关回归。测试必须调用真实ledger/validator；除隔离AirFogSim边界所需的现有FakeEnv外，不用mock证明核心语义。

## 11. CPU候选与提升门

测试和静态门通过后，才运行短CPU P2-B v2 candidate。冻结请求为3 seeds×2 arms×20 frames，理论请求120个natural-reference attempts，但机器报告必须读取实际ledger，不能把120常量写成观测结果。

成功提升要求：

- 实际natural-reference attempt数与发布frame数一致；
- accepted/rejected守恒，accepted与frame一一对应；
- 无隐藏retry、reference/replay对齐、quarantine为0；
- 原有双图、E1、任务、RB、fixture和replay门继续通过；
- manifest artifact/source复算0错误；
- `formal_data_approved=false`、`training_eligible=false`；
- P2-B v1与旧P2-C目录保留，不覆盖或删除。

若真实运行出现rejection，不修改记录或丢弃失败来追求0拒绝率；保留`_failed`并先诊断。v2与v1统计若不同，逐项解释真实差异，不强行修改v2使数字相同。

## 12. 已有实验影响与非目标

P2-B v1的120帧仍可作为结构、五维E1、任务生命周期和runtime接口的preflight证据，但不能支持拒绝率声明。由于当前正式v4数据和训练尚未启动，本设计不使已有模型checkpoint自动失效。

本阶段不做：

- 正式scenario matrix、scale或split冻结；
- 正式轨迹生成；
- GPU训练或扩训；
- locked-test访问；
- 世界模型、策略器或候选rollout规划器实现；
- 用fixture rejection替代自然拒绝率；
- 向旧bundle回填attempt；
- 修改AirFogSim第三方源码。

## 13. 验收结论边界

实现完成后最多可以表述：

> PI-JWM P2联合动作attempt/reject ledger v1已实现并在P2-B v2 CPU preflight中形成可验证证据；P2-C v2能够从natural-reference ledger独立计算拒绝率。

在scenario matrix、formal scale和formal split冻结前，禁止表述P2-C正式通过、v4正式数据集完成、模型训练可启动或最终方法已固定。
