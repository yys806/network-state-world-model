# STEP 6.4J — Live H4 可评分性原位诊断（2026-10-11）

## 完成工作

在项目规定的 `airfogsim` 环境中，使用正式 `collect_trajectory()`，在两个冻结 episode 的 decision frame 原位停止：

- `anchor-0006`：frame 6，capture-012；
- `anchor-0029`：frame 29，capture-058。

callback 在 policy command、policy RNG 和任何新环境步进之前采集完整 causal prefix，并复用 `history_tensor → build_state → context_from_current_raw → CandidateDomain.from_state → objective_side → prepare_anchor`。每个 root 最多尝试两个合法域枚举候选；第一个候选完成 H4，第二个候选在 rollout 前触发 mobility domain 约束异常。没有运行完整 B512。

## 实测证据

独立审计：`independent_audit.json`，通过。

| root | LIVE_ROOT_CAPTURE_VALID | domain | H4完成 | Scorer调用 | scoreable H4 | 主要结果 |
|---|---:|---:|---:|---:|---:|---|
| anchor-0006 | 是 | 6 | 1 | 1 | 0 | `OBJECTIVE_UNSCOREABLE`, `H_eff=0`, `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1` |
| anchor-0029 | 是 | 17 | 1 | 1 | 0 | `OBJECTIVE_UNSCOREABLE`, `H_eff=0`, `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_15:H1` |

合计：2 个不同 root、4 个不同 candidate fingerprint、2 个完整 H4、2 个 Scorer 结果、0 个 scoreable H4、0 个 Scorer exception、2 个 rollout/domain exception、8 次新 WM transition。每个 root 的新 transition 为 4，低于 32 上限。

两个 H4 候选都实际生成了 H1–H4 prediction trace；不可评分发生在 Scorer 的 H1 support 检查，而不是 WM rollout 未完成。

## 确认机制

本次新证据支持：在两个新的真实 live root 上，冻结 support 规则会在 H1 因 future Return-birth 拒绝候选，导致 `H_eff=0` 和 `OBJECTIVE_UNSCOREABLE`。这解释了本次新探针的 2/2 结果。

第二候选的 `ValueError` 是候选 rollout 的 mobility domain 约束异常，独立记录为 rollout/domain exception，没有转换为 scorer rejection，也没有伪造 score。

## 未确认问题

- `R39_EXACT_EQUIVALENCE=UNPROVEN`：新 root 不是 r39 逐候选精确重放。
- 不能把 2/2 的 future Return-birth 频率外推到 r39 的 3744 个 complete 事件。
- 本轮没有观察到 `H_eff=1/2/3`、合法完整可评分 H4 或 Scorer exception，因此不能排除其他机制在更广候选集合中的作用。
- callback 设计在真实 env.step 前停止；环境时间未被 WM 推演推进，且没有执行真实动作。

## 推荐科研方案

下一步最小科学区分实验应由研究者与 ChatGPT 决定：比较保持 fixed-support 的候选与显式支持 future Return 的候选，判断是支持边界定义还是模型预测造成不可评分。该实验需要改变科学语义，Codex 不在本轮自动实施。

GPU 未启动，`locked_test=false`。原 r39 证据未覆盖写入；`R39_EXACT_EQUIVALENCE` 保持 `UNPROVEN`。

诊断脚本 SHA256：`35A5F7A05A742B6C9364DA981C8A2A926BC7B3BE24C8C85DA26F9182684D1CF6`。
