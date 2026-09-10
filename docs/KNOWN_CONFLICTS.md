# PI-JWM 已知冲突与限制

## 当前未解决

### 1. 三 seed 证据尚未闭合

两个正式 seed 已分别通过，但 `20260832` 未运行，三 seed 汇总与稳定性结论不存在。影响：P4 不能关闭，P6 和 `locked_test` 不能开放。

### 2. 理论术语与当前张量对象命名仍需最终统一

代码把有向通信链路放在 `physical_edge_state`，把任务数据流放在 `flow_state`；较早理论材料曾把通信关系统称为信息边。当前实现事实已经写清，但“哪种理论划分最终采用”仍需用户做科研决定，重构不能替用户改定义。

### 3. 全量测试基线不是全绿

2026-09-10 建立 AI_CONTEXT 后在 `PYTHONUTF8=1` 下运行 1643 项测试，得到 0 failure、17 errors。新增的上下文、注册表、证据防漂移和问答检索测试均已纳入。主要类别：

- 当前 Python 环境缺少 AirFogSim 所需 `traci`；
- 历史 teacher fixture 没有满足后来加入的严格 RB 动作—传输事件合同；
- 两个旧 directed-dynamic fixture 没有有效 calibration link 样本；
- 项目结构测试曾错误地要求根目录只能有两个 Markdown 文件；本轮已按治理规则更新，定向结构测试通过。

影响：其余环境/fixture 错误仍阻止使用全量套件证明物理迁移无回归。当前 AI_CONTEXT 范围 6 项、项目知识/结构范围 28 项、P4 `test_formal_*` 范围 210 项分别通过，但它们只覆盖各自路径；正式结果与原始 acceptance 的自动核对为 2 项通过、0 项不一致。

2026-09-09 受限环境下，机器 artifact catalog 曾记录 14 个历史 manifest 无法读取；生成器当时保留了路径和错误。2026-09-10 当前权限下重新生成后，802 个 artifact 记录的读取错误数为 0。该变化只说明当前可读，不改变任何历史实验状态。

### 4. 冻结协议中的状态字段是历史时刻

`protocol.json` 的 `status=ready_for_gpu_batch_probe` 表示冻结当时的状态，不表示 GPU probe 或正式训练现在尚未执行。实时状态必须看后续 run、acceptance 和权威记录。

### 5. 组会结果页可能滞后

现有组会 PPT 的结果页最初使用 seed `20260831` 的中间 checkpoint；不能把旧 PPT 验收自动当作两枚正式 seed 的最新结果验收。

## 冲突处理规则

先定位冲突属于理论、实现、数据、运行、结果还是文档层，再按 `registries/document_authority.json` 找权威来源。不得为了让摘要看起来一致而改代码、阈值或科研结论。
