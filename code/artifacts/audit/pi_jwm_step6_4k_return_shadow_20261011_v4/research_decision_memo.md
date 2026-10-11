# STEP 6.4K Research Decision Memo

| 方案 | 当前证据 | 影响判断 |
|---|---|---|
| A 纯规则扩展 | Ledger/Shadow 规则通过；冻结 WM H4 Shadow 通过 | 可作为隔离原型；正式接入仍需状态容量、Graph 和 Scorer 资格验证 |
| B 规则+状态/图适配 | Shadow 追加 slot/zero hidden 可执行 4 步，但属于分布外输入 | 可能保留权重，但不能宣称无需重训；需要重新验证 normalization、latent 初始化和历史 checkpoint |
| C Planner Return Route | 正式 Planner Route 仍 NOOP；正式 scoreability 未测试 | 独立阻塞，需研究者决定动作空间、CandidateDomain 和 Objective 合同 |
| D 数值状态合同 | 12 条 Validation 中 3 条 FP64/FP32 冲突 | 可能是纯工程修复，但会改变输入状态和历史实验资格，需重新验证 |
| E 微调/重训 | 当前没有冻结模型失败证据，只有分布外接口证据 | 目前 `UNDETERMINED`；只有正式适配后出现递归/评分失败才评估重训 |

推荐顺序：D 的状态合同 shadow 决策 → C 的合法 Return Route 设计 → B 的正式动态 slot/Graph 合同 → 重新资格验证；不自动修改科学语义。
