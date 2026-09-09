# P4 Link Score/Threshold 方法级一致性决策

更新时间：2026-08-31

状态：`confirmed direction / calibration-boundary spec written`

## 1. 结论

当前三态判定为：

```text
protocol_theory_mismatch
```

通俗地说：当前 GPU sentinel 的 No-Go 结论有效，但 link head 的“分数含义”没有和 PI-JWM 主文档完全对齐。

- 主文档把 link activity 定义为离散事件概率，并要求在独立 calibration split 上做概率校准，至少报告 Brier score 和 ECE。
- 当前模型使用 train-only `pos_weight=50.0` 的 weighted BCE 训练 link logit；正式 runner 直接对该 logit 做 sigmoid，把结果当作 threshold score。
- weighted BCE 下的 raw sigmoid 不是未经类别加权的事件后验概率；代码库虽已有反演 utility，但正式 runner 没有调用它，也没有输出 Brier/ECE。
- 因此当前 raw sigmoid 可以作为“冻结协议下的 cost-sensitive decision score”完成 F1 gate，但不能被表述为已经校准的链路活动概率。

准确边界：

1. sentinel validation link-F1=`0.3724`、低于 persistence `0.5901`，所以本次候选仍然是 No-Go；方法语义不一致不能用来推翻失败结果。
2. validation 候选内最优 `0.7` 的 F1=`0.4573`，仍打不过 persistence；不能事后换阈值救结果。
3. 方案 A 已消除旧的长步高置信 FP 机制，这项接口和机制证据继续保留；它不等于完整 P4 性能通过。

## 2. 当前 P4 门与边界

| 项目 | 当前状态 | 含义 |
| --- | --- | --- |
| 当前阶段 | P4 blocked | 不进入 P6 |
| GPU sentinel | `no_go` | 不运行另外两个 seed |
| follow-up seeds | `20260830/20260832` 未启动 | 不补跑、不重复当前配置 |
| formal performance | `formal_performance_claim_ready=false` | 不作正式性能声明 |
| locked test | `locked_test_accessed=false` | 不访问、不推断 |
| 本决策范围 | 只定义 link score/probability 边界 | 不改代码、不训练 |

## 3. 冻结 provenance

本决策只使用以下当前 sentinel 与只读报告：

| 对象 | SHA-256 |
| --- | --- |
| `config.json` | `c4382e53e0abd271500af9f003046c3834e70eea17366f729393c58c3f3ce6c1` |
| `sample_ids.json` | `1f01dade9a0a673fd95bb3d5953a00617c09f124cdc019673765983ea7cde582` |
| checkpoint | `0d203fa46b89e3b4267f387371e4a8c38682810482d30af7c8058f2ea2ac141f` |
| `class_weights.json` | `6953516d9f44acdc2a6a675915a7b5b042024531cb8e8b8f8525aa2605040841` |
| threshold selection | `6de7f8dd66a5d9751caef9a760a2f77261d4af28c4eeec2ed94d67cfad2a084b` |
| validation metrics | `bd65e08d42771f23696e0e74874d8c8e2179b04cc5190637dc6e7815e82144b4` |
| `comparison.csv` | `e175593a4d0dd5363c2b92e44a1e742a2ef6a90864e7d1f1f8e410b1707001d2` |
| canonical tensor manifest | `d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781` |
| sentinel audit | `8dddf747b19f6b06681e0e431270473696ed1a52adc8555ce737ad9ab2168d0a` |
| link diagnosis | `91cd0b54d146bd0f316236bbeb2206fb112697c6961261e9eb2eb1096129a528` |
| threshold replay | `756f63eead1847c94b9e53024c10af4d3499d25403bb34d333e6938b9a4a7f43` |

合同字段已 fresh 核验：

```text
seed=20260831
data_seed=20260823
history/horizon=8/20
train/validation/calibration=256/128/128
training_run_complete=true
gpu_execution=true
locked_test_accessed=false
followup_seeds_authorized=false
```

## 4. 理论—loss—score—threshold—metric 对照

| 层级 | 主文档要求 | 当前实现/产物 | 判定 |
| --- | --- | --- | --- |
| 预测对象 | 链路活动与条件活跃速率采用 hurdle 结构分别建模 | 独立 link activity logit 与 active-only rate 指标存在 | 一致 |
| 输出语义 | 链路活动属于离散事件概率 | `sigmoid(link_activity_logits)` 被直接当 threshold score | 语义未闭合 |
| 训练损失 | 二元事件使用 Bernoulli NLL；类别不平衡可用 train-only class weight | `binary_cross_entropy_with_logits(..., pos_weight=50.0)` | 加权训练本身允许 |
| class weight 来源 | 只能来自 train split | `source_split='train'`，256 samples | 一致 |
| weighted score 概率含义 | 最终仍需概率校准 | raw weighted sigmoid 未在正式路径修正或校准 | 不一致 |
| threshold split | calibration 应在已校准概率域确定正式阈值 | 当前只在 calibration 上选择 raw weighted-sigmoid score 的阈值，validation 只评价 | split 所有权一致；阈值坐标的概率语义未闭合 |
| threshold 候选 | 必须预先冻结 | `0.1/0.3/0.5/0.7/0.9` | 当前协议内部一致 |
| 事件专用阈值 | 每个事件使用自己的阈值 | 顶层 default=`0.5`，effective link=`0.9` | 一致；不得混淆字段 |
| 分类指标 | precision/recall/F1/AUPRC/Brier/ECE | 当前有 precision/recall/F1/AUPRC；未发现 Brier/ECE 正式输出 | 不完整 |
| 概率校准 | 离散头需独立检查可靠性；主文档候选口径为 temperature scaling | 当前正式 runner 未接任何概率 calibrator | 未实现 |

### 实现链的精确事实

1. `formal_world_model_loss_v1.py` 用有效 physical edge 与 `aggregate_link_activity_mask` 的交集构造 link mask。
2. link loss 使用 `binary_cross_entropy_with_logits`，当前 train-only 正负计数为 `16269/1755657`，pos weight 被上限截为 `50.0`。
3. `run_formal_dual_graph_gpu_train_v1.py` 对 raw logits 直接 `sigmoid`，然后在 calibration 上比较五个候选 F1。
4. `formal_world_model_metrics_v1.py` 使用 `thresholds['link_activity']`，当前有效 link threshold=`0.9`。
5. `correct_positive_weighted_probability` 实现了

```text
p = s / (w - (w - 1)s)
```

其中当前 link `w=50`。该 utility 有反演测试，但只被历史 threshold audit 使用，未进入正式 GPU runner。

## 5. 当前 sentinel 同口径证据

### 5.1 Calibration 正式选择

| raw threshold | TP | FP | FN | F1 | 身份 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | 3903 | 212077 | 0 | 0.0355 | calibration candidate |
| 0.3 | 3873 | 43694 | 30 | 0.1505 | calibration candidate |
| 0.5 | 3846 | 31826 | 57 | 0.1944 | calibration candidate |
| 0.7 | 3293 | 23725 | 610 | 0.2130 | calibration candidate |
| 0.9 | 688 | 501 | 3215 | 0.2702 | formal selected |

`0.9` 是 calibration F1 唯一最高值，不是 tie-break 偶然选出。

### 5.2 Validation 只读候选诊断（overall：合并 k=1--20）

以下 TP/FP/FN、precision、recall、F1 均基于同一 validation sample IDs 与有效 mask 的 overall 汇总；不是任一单独 horizon 的数值。

| raw threshold | TP | FP | FN | precision | recall | F1 | 身份 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | 7757 | 199530 | 0 | 0.0374 | 1.0000 | 0.0721 | read-only diagnosis |
| 0.3 | 7749 | 39178 | 8 | 0.1651 | 0.9990 | 0.2834 | read-only diagnosis |
| 0.5 | 7731 | 26137 | 26 | 0.2283 | 0.9966 | 0.3715 | read-only diagnosis |
| 0.7 | 7221 | 16600 | 536 | 0.3031 | 0.9309 | 0.4573 | diagnostic best only |
| 0.9 | 1902 | 557 | 5855 | 0.7735 | 0.2452 | 0.3724 | formal evaluation |

### 5.3 Persistence 同口径汇总（overall：合并 k=1--20）

以下 TP/FP/FN、precision、recall、F1 均基于同一 validation sample IDs 与有效 mask 的 overall 汇总；不是任一单独 horizon 的数值。

```text
TP=8944
FP=5856
FN=6570
precision=0.6043
recall=0.5765
F1=0.5901
```

因此：

- `0.9` 确实对应低 recall；
- 把 threshold 降到 `0.7` 会用 `16600` 个 FP 换回 recall，F1 仍只有 `0.4573`；
- 候选最优相对 persistence delta=`-0.1327 < -0.05`；
- 不能把 threshold migration 写成 sentinel No-Go 的唯一原因。

## 6. 历史证据适用边界

历史报告：`code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`，SHA-256=`c9104036b226d380cfc5718db82e771d7d717ad0280298474456a8d7b8f02657`。

它允许复用的只有：

- weighted score 的反演公式；
- calibration-only、validation-only-evaluation 的 split 原则；
- raw score 与 ordinary probability 不是同一阈值坐标系。

它不能证明：

- 当前 sentinel 采用了 probability correction；
- 当前 sentinel 在修正后的概率域会达到某个 F1；
- 历史三个 CPU checkpoint 的阈值或性能可迁移到当前 GPU checkpoint。

本决策不继承历史性能数值，因此不依赖旧 checkpoint 才成立。

## 7. 三态判定

### 7.1 未选择：`protocol_coherent_candidate_failed`

不满足必要条件。主文档没有把 raw weighted sigmoid 固定定义为纯 decision score；相反，它明确要求事件概率、概率校准、Brier/ECE。因此不能只把现有字段改名为 score，就宣称理论和实现一致。

### 7.2 已选择：`protocol_theory_mismatch`

满足全部条件：

- 理论要求事件概率和校准；
- weighted BCE 使 raw sigmoid 带有类别代价偏移；
- 正式 runner 直接用 raw sigmoid；
- correction utility 未接入正式路径；
- Brier/ECE 和正式 calibration adapter 未实现。

当前性能 No-Go 仍有效；不一致发生在“输出语义与概率评价链”，不是样本、mask、checkpoint 或 F1 计算身份。

### 7.3 未选择：`evidence_incomplete`

不满足条件。当前 provenance、mask、aggregate、事件专用阈值和 persistence 对比均可核验；已知缺口足以定位为方法语义不一致，不需要用新训练补猜测。

## 8. 唯一推荐与受保护基线

### 8.1 唯一推荐

保持 PI-JWM 的理论目标：link activity head 的正式输出应是可审计的事件概率，不把 raw weighted score 降格改名后充当概率。

下一步只允许设计一个方法组件：

```text
post-training link event probability calibration boundary
```

中文解释：保持模型、weighted BCE、数据和方案 A 不变，只设计“训练后的 link 分数如何转换为正式事件概率、如何只用 calibration split 定阈值、如何输出 Brier/ECE/F1”的统一边界。

本决策不预先实现 correction、temperature scaling 或新阈值网格，也不同时比较多条实验路线。下一份设计必须先在数学与接口层选定一个方案，再向用户确认；未确认前不修改代码。

### 8.2 为什么不推荐把 raw score 直接改名

- 会与主文档的事件概率、Brier/ECE 和独立校准要求冲突；
- 会丢失 PI-JWM 后续风险与不确定性接口所需的概率语义；
- 属于为了适配当前实现而降低理论目标，不符合项目永久一致性约束。

### 8.3 必须保护的既有证据

后续任何设计都不得丢失：

- 方案 A 已消除旧 checkpoint 的长步高置信 FP 尾部（历史机制证据，不作为当前 sentinel 性能基线）；
- 模型参数结构和历史 checkpoint strict load 能力；
- canonical tensor、sample IDs、train-only class weights 和 mask；
- node-x ratio=`1.0988 <= 1.25`；
- throughput/RB/task-delay 不回退；
- calibration-only threshold、validation-only evaluation；
- `locked_test_accessed=false`。

### 8.4 GPU 边界

当前不需要 GPU。

若用户确认该方向，下一步仍先写数学与接口设计、TDD 和 CPU 极小/只读验证。只有新的实现门、reload、manifest、受保护基线和独立 Go/No-Go 全部通过，才重新申请 GPU；届时仍只先跑 seed `20260831`。

## 9. 用户确认结果

用户于 2026-08-31 确认以下唯一方向：

> 保留 link activity 的“事件概率”理论目标，冻结模型、weighted BCE、数据、方案 A 和现有性能门；下一步只设计 post-training link probability calibration boundary，不同时修改 loss、模型或开展阈值 sweep。

具体方案已经收敛为“固定类别权重解析反演 + calibration-only scalar temperature”，书面规格为 `记录/设计/2026-08-31-P4-link事件概率校准边界设计.md`。在用户复核该书面规格前不修改实现；P4 blocked、follow-up seeds 未启动、`formal_performance_claim_ready=false`、`locked_test_accessed=false` 保持不变。
