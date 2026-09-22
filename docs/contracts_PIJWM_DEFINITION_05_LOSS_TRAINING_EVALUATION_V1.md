# PI-JWM Definition 05 — Loss / Training / Evaluation Contract v1

状态：**RESEARCHER DECISION FROZEN；PRIMITIVES AND STEP 5.2 CPU DEVELOPMENT LOOP IMPLEMENTED**
冻结日期：2026-09-21
适用起点：STEP 5.1 及其后的 Definition 05 子步骤

## 1. 权威依据与覆盖关系

- 本合同记录研究者在 STEP 5.0 明确确认的 10 项科研决定；Codex 不得自行替换或扩展。
- 只读定义源：`D:\shen\OB\科研\PIJWM\05模型训练、Loss与评价.md`，SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`。
- 该只读文件中的 observation NLL、learned Event/Residual loss 和 latent overshooting 条款，被本轮研究者明确决定更新。私人文件保持原样；工程实现以本合同的较新显式决定为准。
- STEP 4.4 的 learned/rule boundary、fixed current support、prior-only rollout 和 known stochastic outage 继续有效。

## 2. 冻结的 10 项 Researcher Decision

### Decision 1 — Prediction distribution

保留 `z^Phy` 与 `z^Comm` stochastic latent。Vehicle Motion 与 CSI decoder 只输出确定性 point/mean prediction，不增加 learned observation variance，不建立 `(mu, sigma)` observation head。预测不确定性由 stochastic prior sampling 和 multi-sample rollout 辅助表达。

### Decision 2 — Prediction loss

Motion 与 CSI 分别采用 mask-normalized MSE：

`L_Mot = sum(mask_Mot * squared_error_Mot) / sum(mask_Mot)`

`L_CSI = sum(mask_CSI * squared_error_CSI) / sum(mask_CSI)`

`L_Pred = lambda_Mot * L_Mot + lambda_CSI * L_CSI`

v1 默认 `lambda_Mot=lambda_CSI=0.5`。两个 family 必须先在各自有效 mask 内独立归一化，再组合。训练使用冻结的 train-only normalized representation；最终评价反归一化到真实单位。

### Decision 3 — Training-only posterior teacher

Future Target 只能进入 training-only posterior branch：Motion Target Encoder 只编码 Vehicle Motion target，CSI Target Encoder 只编码 CSI target。禁止向 posterior 提供 Future full graph、Flow、Task、DAG、lifecycle 或其他 rule-driven future state。Prior 在训练、验证和 rollout 中均禁止读取 Future Target。STEP 5.1 必须提供 leakage negative tests。

### Decision 4 — Family-specific analytic KL

`L_KL_Phy` 与 `L_KL_Comm` 使用 diagonal-Gaussian analytic KL 并分别 mask-normalize。Physical KL 只覆盖有效 Vehicle stochastic slots；UAV、RSU、cloud 不计入。Communication KL 只覆盖 `valid AND present AND wireless AND CSI-target-valid` relation；wired relation 不计入。v1 使用 KL warm-up 与 small free bits，但 `beta_KL`、threshold 和 warm-up schedule 是训练超参数，不在本合同写死。禁止复杂 KL balancing / stop-gradient balancing。

### Decision 5 — Overshooting OFF

v1 固定 `L_Over=OFF`：

`L_Total = L_Pred + beta_KL * (L_KL_Phy + L_KL_Comm)`

Overshooting 只可作为未来另行授权的 ablation，不能从旧代码或旧文档自动恢复。

### Decision 6 — Training schedule

训练顺序固定为 posterior-assisted warm-up → prior-dominant recursive multi-step curriculum。Stage 1 从 one-step/short-horizon teacher-assisted training 建立 Encoder、Posterior、Decoder、Prior。Stage 2 的实际 rollout state 只能由 prior 递归产生；posterior 只作 KL teacher/reference，GT 只用于 Motion/CSI loss 与 KL teacher。Horizon curriculum 概念为 `1 → 2 → 4 → L`，具体 schedule 属于实验超参数。Validation 始终 prior-only。

### Decision 7 — Unsupported future structure

保持 `fixed_current_object_support=true` 与 `future_return_birth_supported=false`。Unsupported future structure 采用 component-level mask/exclude/classify，不能删除整个 window，也不能计为普通 prediction error。只要 Motion/CSI target 有效，其监督继续保留。必须分别统计 `unsupported`、`unresolved`、`fixed-support-blocked` 数量。

### Decision 8 — No downstream rule-state loss

v1 不增加 `L_Flow`、`L_Task`、`L_DAG`、`L_Lifecycle` 或 `L_Completion`。直接监督只有 Motion、CSI、KL。Flow/Task/DAG 等由 `F^Trans` 恢复并用于 rollout、consistency checking 和 evaluation。允许梯度自然穿过可微规则，但不得人为 stop-gradient；离散 completion、DAG release、Flow status 和 stochastic outage sample 保持真实不可微规则，禁止 soft approximation。

### Decision 9 — Joint trainable modules

v1 端到端 joint training 参数集合为 Dual-Graph Encoder、RSSM deterministic dynamics、Prior、Posterior、Training-only Target Encoder、Vehicle Motion Decoder、CSI Decoder。STEP 4.3B 的 FROZEN 是 architecture/interface frozen，不是随机初始化权重 frozen。规则、mask/vocab/semantics、train-only normalization statistics 和 known equations 不训练。不采用“预训练 Encoder 后冻结”的额外流程。

### Decision 10 — Validation / checkpoint / evaluation

Validation 只使用 prior recursive rollout。每个 horizon `k` 独立计算 mask-normalized normalized-space `L_Mot_k`、`L_CSI_k`，默认 `L_Pred_k=0.5*L_Mot_k+0.5*L_CSI_k`，并定义 `L_Val=(1/L)*sum_k L_Pred_k`；checkpoint selector 固定为 `argmin L_Val`，early stopping 同样依据 `L_Val`。KL 只作 diagnostic。最终 Motion、CSI 分开按 horizon 报告反归一化真实单位 MAE/RMSE；不得只报告整体平均。Observation NLL 不是 v1 核心指标。Multi-sample prior uncertainty 只作辅助 sample-based evaluation，不参与主 checkpoint selection。World Model prediction metrics 与 Planner/closed-loop system metrics 必须分开。

## 3. STEP 5.1 必须建立的数据与张量接口

- Motion target 必须与 STEP 4.4 的 `vehicle delta_xyz + next speed` learned-head boundary 对齐，并使用冻结 train-only normalization lineage。
- CSI target 必须是 future wireless per-RB CSI，保留 relation presence/validity、wireless type、RB 和 CSI observability masks。
- Target Encoder 输入白名单只能包含上述对应 family 的 target 与 mask；黑名单必须覆盖 Flow、Task、DAG、lifecycle、future graph 和 rule-driven state。
- Unsupported-structure mask 必须是 component-level side metadata，不得改写 Motion/CSI validity，也不得删除 window。
- 当前 STEP 4.2A sample 中 future position 只有 raw value，尚无 normalized target；当前冻结 tensor 没有 target position tensor，也没有 future CSI target namespace。STEP 5.1 必须用 additive、可追溯方式补齐，不得重新拟合 normalization statistics。
- 当前 STEP 4.4 real adapter 使用 raw position，而 CSI 来自 normalized graph feature。STEP 5.1 必须显式冻结 decoder output、loss target 与 rule transition 之间的 normalized/raw bridge，禁止单位混用。

## 4. 禁止实现

- learned observation variance、observation NLL 主目标；
- learned outage/rate/service residual、Event head；
- Flow/Task/DAG/lifecycle/completion learned head 或独立 loss；
- v1 latent overshooting 或 KL balancing；
- Future Target 进入 prior；
- Future full graph 或 rule-state 进入 posterior；
- posterior-assisted validation；
- whole-window deletion 代替 component mask；
- development result 冒充 formal/locked-test result。

## 5. 当前证据边界

本合同的研究决定已冻结；5.1B/5.1D 已实现并验收 Loss/Posterior/Metric primitives，STEP 5.2 已实现少量 CPU development training-loop/optimizer smoke。该证据不证明 tiny-data overfit、full training、GPU readiness、预测精度、校准、baseline 公平比较或最终性能。当前固定：`training_loop_implemented=true`、`cpu_optimizer_smoke=true`、`full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test=false`、`performance_claim=false`。
