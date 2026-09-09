# P4 Link Score/Threshold 方法级一致性决策实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `cost-aware-model-routing`; use `executing-plans` only after the user approves the single decision route. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不增加新实验，只复用当前 P4 证据，确定 link activity head 输出究竟是“可校准事件概率”还是“类别加权后的决策分数”，并形成一个理论、代码、数据、指标一致的唯一方法级决策。

**Architecture:** 执行链固定为“冻结证据身份 → 对齐理论定义 → 对齐 loss/score/threshold 实现 → 对齐当前机器结果 → 三态判定 → 用户确认”。本计划只交付 decision memo，不改模型、loss、阈值协议或评价门；任何实现必须在用户确认唯一方法定义后另写单变量补充计划。

**Tech Stack:** Markdown、PowerShell、Python/PyTorch 源码、JSON/CSV 机器产物、SHA-256。

---

## 一、当前门与唯一交付物

- 当前粗粒度主线仍是 `P0 -> P1 -> P2 -> P4 -> P6 -> P7+`，本计划只是 P4 内部决策，不新增阶段。
- 当前状态：P4 blocked；Task 8 sentinel=`no_go`；Task 9 未开放。
- 硬边界：`formal_performance_claim_ready=false`、`locked_test_accessed=false`；GPU follow-up seeds `20260830/20260832` 未启动。
- 唯一交付物：`记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md`。
- 本计划完成的含义仅为“方法定义决策可供用户确认”，不等于 P4 通过，不等于实现完成，不等于允许训练。

## 二、已冻结且不得重复验证的事实

| 事实 | 当前证据 | 本计划用法 |
| --- | --- | --- |
| 方案 A 已实现 | physical-edge correction 改为 edge GRU input message；参数量 `83750` | 作为受保护实现，不重新设计或重测 |
| 旧高置信 FP 机制已消除 | 新 checkpoint validation FP=`557`，h20 FP=`2` | 作为已解决历史机制，不再重复因果干预 |
| sentinel 已 No-Go | candidate/persistence F1=`0.3723571/0.5900904` | 作为当前失败事实，不重跑当前配置 |
| 冻结阈值 `0.9` 低 recall | TP/FP/FN=`1902/557/5855`，recall=`0.2452` | 用于解释当前失败形态 |
| 预注册候选内仍无通过点 | validation 候选最优 `0.7`，F1=`0.4573437`，delta=`-0.1327467` | 证明不能靠事后换候选阈值过门 |
| 位置和运营保护项未回退 | node-x ratio=`1.0988`；throughput/RB/task-delay delta 均小于零 | 后续任何方法决策必须保护 |
| 安全边界未突破 | `locked_test_accessed=false`；follow-up seeds 未启动 | 全计划持续保持 |

## 三、范围控制

### 本计划允许

- 读取当前理论文档、源码、tests、sentinel JSON/CSV、既有 threshold audit。
- 重算 JSON/CSV 中已经存在的计数、比率和哈希。
- 写一份 decision memo，并同步 root 过程记录和现有权威进展记录。
- 使用 Luna 做带精确字段路径的机械抽取，Terra 做固定验收矩阵审查，Sol 做冲突处理和最终判定。

### 本计划禁止

- 不训练，不做 forward replay，不做新阈值扫描，不跑新 seed，不启动 GPU。
- 不访问、生成或推断 `locked_test`。
- 不改 `formal_dual_graph_world_model_v1.py`、loss、metrics、runner、tensor、sample IDs 或评价门。
- 不把 validation 最优 `0.7` 提升为正式阈值。
- 不扩展到 planner、P6、robustness、uncertainty ensemble、paper baseline 或参数 sweep。
- 不把 2026-08-28 历史 threshold audit 的性能数值当作当前 sentinel 性能。
- 不同时规划“改 loss”“概率修正”“temperature scaling”“新阈值网格”四条实现路线；decision memo 只能推荐一个方法定义方向。

## 四、文件职责

### 只读理论与治理入口

- `AGENTS.md`：理论--实现--证据一致性和主线停止规则。
- `记录/PIJWM主文档.md:1594`：链路活动的目标定义和 hurdle 结构。
- `记录/PIJWM主文档.md:1635`：链路活动作为离散事件概率输出。
- `记录/PIJWM主文档.md:1734`：二元目标使用 Bernoulli 负对数似然。
- `记录/PIJWM主文档.md:1763`：类别权重允许，但必须报告概率校准。
- `记录/PIJWM主文档.md:1865`：概率校准与训练损失边界。
- `记录/PIJWM主文档.md:1867`：calibration split 只确定概率阈值。
- `记录/PIJWM主文档.md:1953`：链路活动与活跃速率分开评价。
- `记录/PIJWM主文档.md:1955`：precision/recall/F1/AUPRC/Brier/ECE 最低指标要求。
- `记录/PIJWM主文档.md:2194`：链路活动与速率的 hurdle 预测头。
- `记录/PIJWM主文档.md:2196`：链路活动使用 Bernoulli logit head。
- `记录/PIJWM主文档.md:2282`：离散事件概率需要独立校准。

### 只读实现入口

- `code/src/pi_jwm/formal_world_model_loss_v1.py:66`：带 `pos_weight` 的 binary BCE 实现。
- `code/src/pi_jwm/formal_world_model_loss_v1.py:167`：link activity target、mask 和 loss 路径。
- `code/src/pi_jwm/formal_world_model_loss_v1.py:344`：train-only class weight 统计与上限 `50.0`。
- `code/scripts/run_formal_dual_graph_gpu_train_v1.py:184`：raw sigmoid score、候选阈值和 calibration-only 选择。
- `code/scripts/run_formal_dual_graph_gpu_train_v1.py:245`：validation/calibration 共用已选事件阈值评价。
- `code/src/pi_jwm/formal_world_model_metrics_v1.py:270`：顶层默认阈值与事件专用阈值覆盖。
- `code/src/pi_jwm/formal_world_model_metrics_v1.py:335`：link event 使用 `thresholds['link_activity']`。
- `code/src/pi_jwm/formal_binary_calibration_v1.py:8`：weighted score 到 unweighted posterior 的现有数学变换。
- `code/tests/test_formal_binary_calibration_v1.py:13`：该变换的数值反演测试。

### 当前机器证据

- `code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/`
- `code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_sentinel_20260831/sentinel_audit.json`
- `code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_link_diagnosis_20260831/link_bias_latent_diagnosis.json`
- `code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_threshold_replay_20260831/link_activity_threshold_replay.json`
- `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`，仅作历史机制证据。

### 执行阶段允许创建或修改

- Create: `记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md`
- Modify: `task_plan.md`
- Modify: `progress.md`
- Modify: `findings.md`
- Modify: `记录/本地计划表.md`
- Modify: `记录/8.12之后推进.md`
- Modify only if method wording changes after user approval: `记录/PIJWM主文档.md`
- Modify handoff only after decision is confirmed: `PROJECT_CONTEXT.md`

## 五、执行任务

### Task 1：冻结证据身份与可比性

**Files:**

- Read: 当前 sentinel run、tensor manifest、三个 audit。
- Write later: decision memo 的“证据身份”表。

- [x] **Step 1：重算冻结输入和当前报告 SHA-256**

Run:

```powershell
cd D:\shen\PKU\PIJWM
$files = @(
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/config.json',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/sample_ids.json',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/checkpoints/coupled_dual_gnn_residual__best.pt',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/class_weights.json',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/metrics/coupled_dual_gnn_residual__threshold_selection.json',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/metrics/coupled_dual_gnn_residual__validation.json',
  'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/comparison.csv',
  'code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827/manifest.json',
  'code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_sentinel_20260831/sentinel_audit.json',
  'code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_link_diagnosis_20260831/link_bias_latent_diagnosis.json',
  'code/artifacts/audit/pi_jwm_p4_edge_feedback_gru_input_threshold_replay_20260831/link_activity_threshold_replay.json'
)
foreach ($file in $files) {
  $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $file
  "$($hash.Hash.ToLower())  $file"
}
```

Expected：分别得到当前已核验哈希：

```text
c4382e53e0abd271500af9f003046c3834e70eea17366f729393c58c3f3ce6c1  config.json
1f01dade9a0a673fd95bb3d5953a00617c09f124cdc019673765983ea7cde582  sample_ids.json
0d203fa46b89e3b4267f387371e4a8c38682810482d30af7c8058f2ea2ac141f  checkpoint
6953516d9f44acdc2a6a675915a7b5b042024531cb8e8b8f8525aa2605040841  class_weights.json
6de7f8dd66a5d9751caef9a760a2f77261d4af28c4eeec2ed94d67cfad2a084b  threshold_selection.json
bd65e08d42771f23696e0e74874d8c8e2179b04cc5190637dc6e7815e82144b4  validation.json
e175593a4d0dd5363c2b92e44a1e742a2ef6a90864e7d1f1f8e410b1707001d2  comparison.csv
d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781  tensor manifest
8dddf747b19f6b06681e0e431270473696ed1a52adc8555ce737ad9ab2168d0a  sentinel audit
91cd0b54d146bd0f316236bbeb2206fb112697c6961261e9eb2eb1096129a528  link diagnosis
756f63eead1847c94b9e53024c10af4d3499d25403bb34d333e6938b9a4a7f43  threshold replay
```

- [x] **Step 2：核对运行边界**

必须逐项确认：seed=`20260831`、data seed=`20260823`、history/horizon=`8/20`、train/validation/calibration=`256/128/128`、`locked_test_accessed=false`、另外两 seed 不存在有效 run。

**Stop condition:** 任一哈希或合同字段不一致，立即将 decision 状态记为 `evidence_incomplete`；不通过补跑实验修复证据身份。

### Task 2：建立理论定义表

**Files:**

- Read: `记录/PIJWM主文档.md` 上述精确段落。
- Write: decision memo 的“理论要求”表。

- [x] **Step 1：逐条摘录而不改写含义**

表格固定包含：对象、理论术语、训练允许项、校准要求、评价指标、split 边界、当前是否已实现。

- [x] **Step 2：明确必须回答的唯一语义问题**

```text
当前 link_activity_logits 经 sigmoid 后的输出，在正式方法中究竟被定义为：
A. 可解释的链路活动事件概率；
B. 受 pos_weight 影响、只用于排序和阈值判定的 cost-sensitive score。
```

不得把 A 和 B 混用。若选 A，概率校准、Brier/ECE 和阈值语义必须与代码一致；若选 B，主文档和 advisor-facing 文字不得继续把 raw score 直接称作校准概率。

**Pass condition:** 每条理论说法都能指向精确文档段落；没有用历史实验替代理论定义。

### Task 3：建立 loss—score—threshold 实现链

**Files:**

- Read: loss、runner、metrics、binary calibration utility 及对应 tests。
- Write: decision memo 的“实现链”表。

- [x] **Step 1：确认训练 score 的来源**

必须记录：link activity 使用 `binary_cross_entropy_with_logits`；class weight 仅由 train split 统计；当前 `pos_weight.link_activity=50.0`；mask 为有效 physical edge 与 aggregate activity mask 的交集。

- [x] **Step 2：确认正式阈值路径**

必须记录：runner 对 raw logits 直接 `sigmoid`；候选固定为 `0.1/0.3/0.5/0.7/0.9`；calibration 上最大 F1 选择；同 F1 时选更接近 `0.5`；validation 只使用已选阈值评价。

- [x] **Step 3：区分默认阈值与事件专用阈值**

Fresh check:

```powershell
$path = 'code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831/metrics/coupled_dual_gnn_residual__validation.json'
$report = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
[PSCustomObject]@{
  top_level_default = $report.threshold
  link_effective = $report.thresholds.link_activity
  flow_effective = $report.thresholds.flow_present
  task_effective = $report.thresholds.task_present
} | Format-List
```

Expected：顶层默认=`0.5`；link effective=`0.9`；flow=`0.9`；task=`0.7`。任何 worker 把顶层默认值当作 link effective threshold 时必须拒绝该抽取。

- [x] **Step 4：确认概率修正的真实接入状态**

`correct_positive_weighted_probability` 已存在且有反演测试，但必须用 `rg` 证明它是否进入当前正式 runner：

```powershell
rg -n -F 'correct_positive_weighted_probability' code/src code/scripts code/tests
```

Expected：当前 utility、test 和历史 threshold audit 引用可见；正式 GPU runner 的阈值选择路径不调用它。

**Stop condition:** 理论称为概率、实现却只用未修正 weighted score，且没有明确方法边界时，状态必须进入 `protocol_theory_mismatch`；不得用改名掩盖。

### Task 4：建立当前 sentinel 的同口径证据矩阵

**Files:**

- Read: sentinel threshold selection、validation metrics、comparison、threshold replay、persistence validation。
- Write: decision memo 的“当前证据”表。

- [x] **Step 1：写正式 calibration 选择**

link candidates 的 calibration F1：

```text
0.1 -> 0.0355007
0.3 -> 0.1504954
0.5 -> 0.1943651
0.7 -> 0.2129944
0.9 -> 0.2702278  [正式选中]
```

- [x] **Step 2：写只读 validation 诊断，不产生新正式阈值**

```text
threshold  TP    FP      FN    precision  recall  F1
0.1        7757  199530  0     0.0374     1.0000  0.0721
0.3        7749  39178   8     0.1651     0.9990  0.2834
0.5        7731  26137   26    0.2283     0.9966  0.3715
0.7        7221  16600   536   0.3031     0.9309  0.4573
0.9        1902  557     5855  0.7735     0.2452  0.3724
```

- [x] **Step 3：写 persistence 同口径汇总**

```text
TP=8944, FP=5856, FN=6570
precision=0.6043, recall=0.5765, F1=0.5901
```

- [x] **Step 4：只允许得出两个并存结论**

1. calibration 选出的 `0.9` 在 validation 上对应低 recall；
2. 预注册候选内最优 `0.7` 仍低于 persistence，不能靠换候选阈值过门。

**Pass condition:** 所有数值标明 `calibration formal selection`、`validation read-only diagnosis` 或 `validation formal evaluation`；不得混淆单 horizon 和全 20 步 aggregate。

### Task 5：限定历史 threshold audit 的用途

**Files:**

- Read: `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`。
- Write: decision memo 的“历史证据边界”。

- [x] **Step 1：只复用机制事实**

允许引用：`p = s / (50 - 49s)` 的 weighted-score 反演关系、calibration-only 原则、不同 seed 的 threshold 迁移曾经存在。

- [x] **Step 2：明确禁止继承性能数值**

历史 audit 对应旧 CPU checkpoints 和旧模型路径；不得用其 F1、threshold 或 seed 结论替代当前 sentinel。

**Stop condition:** 若当前结论依赖历史 checkpoint 性能才成立，则记为 `evidence_incomplete`，不进行新推理补齐。

### Task 6：执行三态方法级判定

**Files:**

- Read: Tasks 1--5 的 decision memo 草稿。
- Write: memo 的“最终判定与唯一推荐”。

- [x] **Step 1：按以下互斥条件分类**

| 状态 | 必要条件 | 允许结论 | 下一动作 |
| --- | --- | --- | --- |
| `protocol_coherent_candidate_failed` | 理论明确把 raw weighted sigmoid 定义为 cost-sensitive score；代码、阈值、指标文字全部一致 | 当前方案 A 分支在冻结协议下失败 | 保留方案 A 接口证据，停止该候选，不改协议救结果 |
| `protocol_theory_mismatch` | 理论要求事件概率或概率校准，但正式 runner 直接用 raw weighted sigmoid，且校准/概率指标链未闭合 | 当前性能 No-Go 有效，但方法概率语义未闭合 | Sol 只推荐一个定义修订方向，提交用户确认；不实现 |
| `evidence_incomplete` | provenance、mask、aggregation 或 score 路径存在无法消解的冲突 | 不能作方法选择 | 报告精确缺失字段并停止，不训练补猜测 |

- [x] **Step 2：Sol 只输出一个推荐方向**

decision memo 必须包含：被选状态、支持证据、反证、被拒绝解释、受保护基线、会改变的唯一方法定义、不会改变的内容。不得列出多条并行实验路线。

- [x] **Step 3：设置用户确认门**

只有当结论涉及修改理论定义、loss、概率变换、阈值协议或评价门时才向用户请求确认。确认问题只包含一个推荐方案及其代价，不要求用户在多个实验套餐中选择。

### Task 7：写 decision memo 并同步记录

**Files:**

- Create: `记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md`
- Modify: root planning files、`记录/本地计划表.md`、`记录/8.12之后推进.md`

- [x] **Step 1：memo 使用固定结构**

```text
1. 结论
2. 当前 P4 门与边界
3. 冻结 provenance
4. 理论—loss—score—threshold—metric 对照表
5. 当前 sentinel 同口径证据
6. 历史证据适用边界
7. 三态判定
8. 唯一推荐与受保护基线
9. 用户确认问题
```

- [x] **Step 2：同步状态但不改理论正文**

在用户确认前，只在过程记录和进展记录中写“候选决策/待确认”。不得提前修改 `PIJWM主文档.md` 的理论定义。

- [x] **Step 3：运行格式与禁词检查**

Run:

```powershell
$patterns = @(('T' + 'BD'), ('T' + 'ODO'), ('implement' + ' later'), ('后续' + '再定'), ('多路线' + '并行'))
rg -n ($patterns -join '|') '记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md'
git diff --check -- task_plan.md progress.md findings.md '记录/本地计划表.md' '记录/8.12之后推进.md' '记录/设计/2026-08-31-P4-link-score-threshold方法级一致性决策.md'
```

Expected：`rg` 无匹配、退出码 `1`，按“未找到禁词”处理；`git diff --check` 退出码 `0`。

## 六、用户确认后的固定边界

本计划在 decision memo 和用户确认问题处停止。用户确认前不触碰代码。

若用户确认唯一方法定义，下一份实施补充必须继续遵守：

1. 只改变一个方法变量；不得同时改 loss、概率变换和候选阈值网格。
2. 先 TDD 和现有 checkpoint/metrics 的 CPU 只读或 micro-smoke；不做 CPU 大训练。
3. 先证明不会丢失方案 A 已消除旧 FP 尾部、node-x 和运营指标保护项。
4. 只有实现门、reload、manifest 和独立 Go/No-Go 全通过，才向用户报告 GPU 必要性。
5. GPU 仍只先运行 sentinel seed `20260831`；通过全部冻结门后才允许 `20260830/20260832`。
6. 不访问 `locked_test`，不进入 P6，不做 sweep。

## 七、模型路由与验收责任

- Luna：机械提取精确文件行号、JSON 字段路径和哈希；不得解释概率语义或作最终选择。
- Terra：检查每一步输入、输出、停止条件和受保护基线；不得提出新模型或实验扩展。
- Sol：回查决策关键字段，处理理论--代码--指标冲突，拒绝错误口径，写最终 memo 和唯一推荐。
- 本计划制定时已发生真实路由：Luna 提取一次出现“顶层默认阈值与 link 事件阈值混淆”，Sol 依据原始 `thresholds.link_activity=0.9` 拒绝该错误；Terra 的计划门审查通过后由 Sol 收敛。

## 八、计划自审结果

- [x] 主线覆盖：只处理 P4 link score/threshold 定义，不新增 top-level phase。
- [x] 零重复：不重复 GPU sentinel、旧 causal intervention、v2 diagnosis 或 threshold replay。
- [x] 证据分层：当前 sentinel 性能、历史机制、理论定义和候选决策分别记录。
- [x] 停止门：任何 provenance/口径冲突立即停止；用户确认前不实现。
- [x] GPU 边界：本计划完全不需要 GPU。
- [x] `locked_test` 边界：始终不访问。
- [x] 占位符扫描：计划没有常见未完成标记或未定义的实现步骤。
