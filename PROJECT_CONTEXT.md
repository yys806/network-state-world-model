# PI-JWM 项目上下文交接

> 更新时间：2026-09-09（Asia/Shanghai）
> 项目根目录：`D:\shen\PKU\PIJWM`
> 当前分支 / HEAD：`main` / `0630515bf8c43d8212f14fcc449291bb62860c67`
> 交接性质：基于当前权威记录、代码、冻结协议、两枚正式 seed 验收产物和远端终态形成的续接快照。后续若有新运行，以新生成的原始产物和机器审计为准。

## 1. 项目概述

PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）面向车联网、无人机、路侧单元与边缘计算共同构成的动态系统。主线是：先把物理网络和信息网络表示为严格对齐的双图，再学习动作条件下的多步状态演化，最后基于世界模型进行候选动作推演、选择和滚动重规划。

AirFogSim 仅作为场景仿真和原始轨迹生成工具，不是 PI-JWM 框架本身。

当前粗粒度路线固定为 `P0 -> P1 -> P2 -> P4 -> P6 -> P7+`。当前仍在 **P4：正式非 locked 世界模型精度与泛化验收**。P6 的候选动作规划器尚未开放。

## 2. 当前状态快照

### 2.1 已验证状态

- 当前候选方法：`entity_aligned_dual_graph_rssm_v1`。
- 模型版本：`formal_entity_aligned_rssm_v1`。
- latent dynamics：`entity_aligned_complete_rssm_prior_posterior_v1`。
- 因果运动 tensor、第一性原理审计、CPU 方法一致性门、GPU batch 探测和 GPU execution sentinel 均已完成。
- GPU batch size 已冻结为 `8`；三种子不得更换 batch。
- 首个正式 seed `20260831` 已完成 deterministic base 20 epoch + entity RSSM 40 epoch；按 `p4_gate_aware_v1` 选择 epoch 39。
- seed `20260831` 的 79 项正式 manifest 大小与 SHA-256 一致，checkpoint 方法身份、strict reload 和原始指标独立重算通过；重算最大浮点差约 `3.37e-09`。
- seed `20260831` 的 9 项单种子数值门全部通过：
  - validation link-F1 相对 persistence：`+0.4441475764`；
  - calibration link-F1 相对 persistence：`+0.8622921256`；
  - node-x MAE ratio：overall/h5/h10/h20 = `0.754753/0.769951/0.750709/0.754954`；
  - throughput/RB occupancy/task delay MAE ratio = `0.935554/0.473627/0.012869`。
- `locked_test_accessed=false`；正式 locked test 仍未 tensorize、未访问。

### 2.2 第二个正式 seed 与当前运行状态

- seed `20260830` 已按与 `20260831` 完全相同的冻结配置完成 deterministic base 20 epoch + entity RSSM 40 epoch；选择 epoch 40。
- 正式 run 位于 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1/`，独立验收位于 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`。
- 79 项正式 manifest 零缺失、零哈希差异，strict reload 为 `0/0`，原始指标重算最大差为 `4.15555434507553E-09`。
- 9 项单 seed 数值门全部通过：validation/calibration link-F1 delta=`+0.45710/+0.88509`；node-x overall/h5/h10/h20 ratio=`0.75086/0.76033/0.74939/0.74960`；throughput/RB/task-delay ratio=`0.93831/0.47523/0.01175`。
- 远端训练进程已退出，GPU 显存占用为 `0 MiB/24564 MiB`。当前没有正在运行或获准启动的正式训练。

### 2.3 尚未完成与当前阻塞

- seed `20260832` 尚未启动，启动权限由用户保留，不得根据前两枚 seed 的通过结果自动续跑。
- 三种子独立正式审计尚未完成，所以 P4 仍为 `blocked`，P6 未开放。
- `formal_performance_claim_ready=false`。两枚 seed 分别通过仍不能宣称三 seed 泛化或 PI-JWM 性能最终通过。

### 2.4 单一下一动作

**等待用户明确决定是否授权运行最后一个冻结 seed `20260832`；在此之前不启动新训练、不进入 P6、不访问 `locked_test`。**

## 3. 当前架构

```text
AirFogSim 原始轨迹（数据源）
        |
        v
严格单时隙物理图 + 流式信息图 + 任务/DAG + 动作/规则字段
        |
        v
因果滑动窗口 tensor（history=8, horizon=20）
        |
        v
确定性耦合双图基础模型
  - 物理图：物理节点 <-> 物理通信边
  - 信息图：信息代理 <-> 数据流信息边
  - 跨图：agent-node 附着；flow-physical-edge 承载
        |
        | 先训练 20 epoch，然后冻结
        v
实体级双图 RSSM
  - node / physical_edge / flow / task 各自维护 h 与 z
  - 训练：动作条件 prior + 观测条件 posterior teacher
  - 验证/部署：严格 prior-only rollout
        |
        v
确定性规则层逐步反馈
  - 容量、剩余工作、任务生命周期、DAG 等合法状态递推
        |
        v
多步预测 + P4 正式指标 + gate-aware checkpoint 选择
        |
        v
P4 三种子通过后，才进入候选动作推演与滚动重规划
```

双图编码和 RSSM 是前后相接的两个模块，不是互相竞争的“方法一/方法二”。双图编码负责建立实体及其物理—信息关系，RSSM 负责在该表示上学习动作条件的时间演化。

## 4. 仓库地图

| 路径 | 职责 | 证据边界 |
| --- | --- | --- |
| `AGENTS.md` | 永久治理和一致性规则 | 最高优先级约束 |
| `记录/本地计划表.md` | 唯一粗粒度路线和当前阶段 | 当前状态权威入口 |
| `记录/PIJWM主文档.md` | 理论、对象、接口、方法和评价定义 | 方法边界权威记录 |
| `记录/8.12之后推进.md` | 动态进展、失败与停止规则 | 最新运行进展权威记录 |
| `task_plan.md` / `progress.md` / `findings.md` | 当前过程、结果与发现 | 不单独建立最终结论 |
| `code/src/pi_jwm/` | 模型、数据、规则和审计逻辑 | 证明实现，不单独证明性能 |
| `code/scripts/` | tensor 构建、审计、训练和验收入口 | 启动成功不等于门通过 |
| `code/tests/` | 单元、契约、回归和机制测试 | 仅证明对应断言 |
| `code/reference/AirFogSim/` | 第三方仿真器 | 只作数据源/参考环境 |
| `code/artifacts/` | tensor、协议、训练、审计和 manifest | 机器证据核心目录 |
| `literature/` | 本地权威文献库 | 不证明代码已实现 |
| `meeting/` | 组会材料 | 不能反向替代实验验收 |
| `paper/` | 论文材料 | 不得超出已验证证据 |

当前工作树包含跨越多轮历史开发的较大变更集。本次 GitHub 快照已先排除训练产物、凭据类文件和无关本地截图；后续仍不得使用 reset/clean 破坏本地证据。

## 5. 技术栈与运行环境

- 主要语言：Python；深度学习：PyTorch。
- 本机当前探测：Python `3.13.5`、PyTorch `2.8.0+cpu`、`torch.cuda.is_available() == false`。
- 正式训练远端：RTX 4090；seed `20260830` 已结束并释放 GPU。
- 优化：AdamW，learning rate `3e-4`，weight decay `1e-5`，gradient clip `5.0`。
- 模型：hidden dimension `32`，stochastic dimension `16`。
- RSSM 损失：KL balance `0.8`，KL weight `0.1`，overshooting distance/weight `5/0.1`，teacher reconstruction weight `0.5`。
- 数据量：train/validation/calibration = `9828/3276/1638` unlocked windows；history/horizon = `8/20`。
- 正式种子顺序：`20260831 -> 20260830 -> 20260832`。

冻结协议：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

该协议是 GPU 执行前冻结的不可变合同，其中 `status=ready_for_gpu_batch_probe` 等字段记录冻结当时状态；实时进展必须读取后续训练目录、acceptance JSON 和权威进度记录，不能修改协议来“更新状态”。

## 6. 功能模块

### 6.1 数据与因果运动合同

- 正式 tensor：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`。
- 节点主状态：`x/y/z/speed/acceleration/cpu/storage`。
- 额外因果运动状态：`vx/vy/vz/ax/ay/az`，只由当前及更早位置按真实时隙差分得到；信息不足处使用 mask。
- 轨迹 split、窗口 ID、RB、任务、链路和规则字段沿用冻结合同；不读取未来位置构造输入。

### 6.2 Strict Physical–Information Coupled Dual-Graph Message Passing

中文名称：**严格物理—信息耦合双图消息传递**。

- 物理节点：车辆、UAV、RSU、边缘设备等真实设备。
- 物理边：设备间有向通信链路，状态为 `distance/csi_mean/rate_sum/active_task_count/allocated_rb_count`。
- 信息节点：一一附着到活动物理节点的 agent。当前没有独立 `information_node_state`；agent encoder 从对应物理节点的七维历史初始化确定性 latent。
- 信息边：任务输入流、结果回传流和显式 payload 的依赖数据流；`flow_state` 为 `total_data/remaining_data/delivered_cumulative/delivered_this_slot/age`。
- 图内传播：node—physical-edge 与 agent—flow。
- 跨图传播：agent—physical-node 附着、flow—physical-edge 承载。
- task 与 DAG 是业务辅助对象，参与消息传播和规则递推。

### 6.3 Entity-Aligned Dual-Graph RSSM

中文名称：**实体对齐双图循环状态空间世界模型**。

- 为每个物理节点、物理边、信息流和任务分别维护 deterministic state `h` 与 stochastic state `z`，保留实体维度。
- agent 是双图基础模型中的确定性中间 latent，当前没有单独随机状态；不要表述成“五类对象都有独立 RSSM”。
- prior 只使用上一时刻实体状态、当前图消息与合法动作。
- posterior 仅在训练阶段读取对应实体的目标观测，作为 teacher。
- validation 和部署严格使用 prior-only rollout；改写未来 target 不得改变 prior prediction。
- 节点位置从因果历史运动的确定性 proposal 出发，RSSM 学习逐节点残差。
- link head 从逐边 latent 解码，可以改变同一步不同边的排序，修复旧 global broadcast RSSM 无法改变边间排序的结构限制。
- 复用 balanced KL、prior reconstruction、teacher reconstruction 与 overshooting；residual head 零初始化。

### 6.4 确定性规则层

规则层把明确的物理/业务约束逐步反馈到后续状态，例如容量、剩余数据、任务阶段、生命周期和 DAG 依赖。它与学习模型共同递推，但不能用规则代理量冒充模型已学会的能力。

### 6.5 候选动作规划器（尚未开放）

`code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` 已有 CPU 可审计机制原型：过滤预构造合法候选、为每个候选注入未来动作、分别调用世界模型、比较预测状态/任务/成本/风险、选择首动作并支持 `replan()`。

它尚不是完整策略器，缺少正式候选生成器、领域合法性检查、冻结目标/风险定义、与最终实体 RSSM 的正式集成、AirFogSim 执行反馈闭环和训练后的策略。P4 三种子通过前不得扩展或宣称该模块完成。

## 7. 核心逻辑与数据流

1. AirFogSim 生成原始时隙轨迹。
2. collector/adapter 将轨迹规范化为物理节点、物理边、信息 agent、数据流、任务、DAG、动作和规则字段。
3. tensor builder 按完整轨迹划分 train/validation/calibration/locked test，并构造 history 8、horizon 20 窗口；train-only statistics 用于归一化。
4. 确定性双图基础模型编码历史，在物理图、信息图和两类跨图关系上交换消息并预测未来 proposal。
5. 使用全部 train windows 训练 base 20 epoch。
6. 冻结 base，训练实体级 prior、posterior、transition 和 decoder，最多 40 epoch；前 20 个 RSSM epoch 不早停，之后 patience 10。
7. 训练期 posterior teacher 参与 KL 与重构；正式 validation/calibration prediction 只走 prior。
8. 每个 epoch 保存 checkpoint，按字典序选择：失败硬门数最少、最大归一化超限最小、validation state NLL 最低。
9. 单种子结束后，从原始 checkpoint 与 prediction/metrics 独立重算门，不使用训练 summary 代替验收。
10. 三个种子都通过后才能记录 P4 non-locked accuracy gate passed，并开放候选动作推演阶段。

## 8. 数据、持久化与证据

### 8.1 当前正式输入

- tensor 根：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`
- manifest SHA-256：`85e50b5ecf1742cfaecb859522587efaed3412d03a74f1a00e322164ae8861f6`
- unlocked counts：train `9828`，validation `3276`，calibration `1638`
- locked test：未 tensorize、未访问

### 8.2 关键正式证据

- 第一性原理审计：`code/artifacts/audit/pi_jwm_p4_first_principles_audit_20260906_v1/first_principles_audit.json`
- CPU 一致性：`code/artifacts/audit/pi_jwm_p4_entity_rssm_cpu_consistency_20260906_v2/`
- GPU batch probe：`code/artifacts/audit/pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/`
- GPU sentinel：`code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_sentinel_20260906_v1/`
- seed `20260831` 正式运行：`code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260831_v1/`
- seed `20260831` 独立验收：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`
- seed `20260830` 运行中证据：`code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_live_evidence_20260908/`

seed `20260831` acceptance 的状态是 `passed`，但 scope 明确为 `single-seed`、`cross-seed pending`。这两个字段必须同时保留。

### 8.3 失败证据的作用

历史失败不能删除或被新结果覆盖。旧 aggregate/global RSSM 的关键问题包括：全局 pooled context 向同类实体广播相同 correction，link correction 无法改变边间排序；旧运动字段 speed/acceleration 全零；从头联合训练使 base 与 RSSM 目标互相干扰；短样本预算和只按 aggregate loss 选 checkpoint 与正式门错位。

这些失败和反事实审计构成当前新方法设计依据。若后续种子失败，应先比较失败属于跨种子泛化、训练轨迹、某个实体/视野/指标，还是数据可达上界问题；不得重新运行相同配置或追加临时 loss 补丁。

## 9. API、接口与外部集成

项目当前以离线脚本和本地/远端文件 artifact 为主，没有对外 HTTP API。

主要代码接口：

- `FormalDualGraphWorldModel`：确定性耦合双图基础模型。
- `FormalEntityAlignedRSSMWorldModel`：正式实体级 RSSM 包装与 rollout。
- `DeterministicRuleLayer`：确定性规则递推。
- `run_formal_p4_entity_rssm_gpu_v1.py`：GPU sentinel / formal 训练入口。
- `run_formal_entity_aligned_rssm_consistency_audit_v1.py`：CPU 方法一致性审计入口。
- `FormalCandidateRolloutPlanner`：后续候选推演机制原型，当前不可作为正式模块结论。

远端 GPU 通过 SSH 使用。仓库根目录存在本地 askpass 辅助文件 `.codex_askpass_30339.cmd`；它可能包含敏感凭据，**不得读取、打印、提交或写入交接文档**。续接时只可把它作为已有认证辅助，并优先执行只读状态检查。若认证失效，应由用户提供新的连接信息。

## 10. 构建、运行、测试与验收

### 10.1 本机通用检查

```powershell
cd D:\shen\PKU\PIJWM
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python -m compileall -q .\code\src .\code\scripts .\code\tests
python -m unittest discover -s .\code\tests -p 'test_*.py'
```

本次交接没有重跑全量测试；当前通过状态来自正式 CPU consistency artifact、GPU sentinel 与 seed `20260831` acceptance。若修改代码，必须重新执行与修改范围对应的测试和一致性审计，不能沿用旧通过结论。

### 10.2 当前远端监控原则

- 只读查看进程、GPU、staging checkpoint 和日志。
- 不向训练进程发送信号，不修改远端 workspace、tensor、配置或 checkpoint。
- 正式输出在运行结束前位于隐藏 staging 目录；最终原子发布后才出现正式 run 目录。
- 中间 checkpoint 只作证据保全，不执行临时阈值选择或性能否决。

可复用认证方式只引用本地 askpass 文件路径，不显示其内容：

```powershell
$ask = (Resolve-Path '.\.codex_askpass_30339.cmd').Path
$env:SSH_ASKPASS = $ask
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'codex'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 30339 root@connect.nmb1.seetacloud.com "ps -eo pid,etimes,cmd; nvidia-smi"
```

### 10.3 seed 结束后的验收顺序

1. 确认远端训练进程自然退出且正式输出目录原子发布。
2. 回传完整 run 目录，保留原始目录结构、大小和 SHA-256。
3. 检查 method/model/latent identity、seed、batch、样本数、训练完成标志、locked-test 边界。
4. strict reload 最佳 checkpoint。
5. 从原始 validation/calibration prediction/metrics 独立重算 9 项门，浮点容差沿用 `20260831` 验收。
6. 生成单独 acceptance JSON 与 manifest；失败也必须完整保存。
7. `20260830` 通过后才启动 `20260832`；否则停止并做一次基于已有证据的失败归因，不换 seed、不调门。
8. 三种子完成后运行独立三种子审计，才判断是否关闭 P4。

## 11. 关键决定与约束

- P4 门槛固定，不根据结果放宽。
- 256/128/128 的短训练只验证执行路径，不能用于性能 No-Go。
- 正式结论必须使用全部 unlocked 数据训练至协议停止点。
- 合法历史位置可以用于因果运动派生；未来位置、未来路线和未来标签不得进入 prior prediction。
- 理论定义、数据字段、代码读取路径、运行配置、指标和机器验收必须逐项一致。
- 完整 RSSM 必须具备动作条件 prior、训练期 observation-conditioned posterior、KL 与 prior-only deployment；只有类名或 latent shape 不构成证明。
- 双图基础模型与实体 RSSM 分两阶段训练；base 在 RSSM 阶段冻结。
- 不引入 Transformer、JEPA、event model、focal loss、PCGrad、新数据源或规划器扩展来绕过当前门。
- 同一正式配置只运行一次。失败后必须形成新证据和明确方法定义，不能重复试验碰运气。
- `locked_test` 继续封存；P4 non-locked 三种子通过也不等于 final locked 性能声明。

## 12. 开发状态

### 已完成（Verified）

- 正式物理—信息双图数据合同与 h20 unlocked tensor。
- 因果运动状态修复及 tensor manifest。
- 第一性原理审计：可观测性、实体表达、梯度冲突和 checkpoint 选择错位。
- 实体级双图 RSSM 实现及 CPU 一致性门。
- GPU 显存探测、batch=8 冻结与 execution sentinel。
- seed `20260831` 全量正式训练、严格复载、manifest 与独立单种子验收。
- 组会 PPT 已在 `meeting/PI-JWM_组会汇报.pptx` 后追加第 206–215 页，1–205 页受保护内容保持不变；验收记录为 `meeting/2026-09-09-PI-JWM组会汇报PPT验收.json`。

### 正在进行（Active）

- seed `20260830` 全量正式训练；截至交接快照已完成 base 20 epoch，进入 entity RSSM epoch 1。
- 对该运行做只读证据保全。

### 已同意但尚未执行（Agreed）

- 若用户明确授权，再按冻结协议原样运行 `20260832`。
- 三种子完成后从原始 checkpoint 与 predictions 独立重算正式门并形成跨种子审计。
- P4 通过后开放 P6，完成候选生成、合法性过滤、逐候选世界模型 rollout、成本/风险比较、首动作执行和反馈重规划。

### 候选/原型（Proposed）

- `FormalCandidateRolloutPlanner` 是 CPU 机制原型，不是已经完成的策略器。
- Actor–Critic / PPO 仅是后续可能的策略学习或对照候选，不属于当前已实现主线结论。

### 未知（Unknown）

- `20260832` 是否能通过全部门。
- 三种子均值、方差以及正式跨种子泛化结论。
- 进入 P6 后最终采用纯候选搜索、学习策略还是混合机制；必须等世界模型冻结后再公平定义。

## 13. 精确续接点

新会话应按以下顺序继续，不要重新调研或重做已经通过的 CPU/GPU 门：

1. 读取本文件、`task_plan.md`、`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`。
2. 读取 seeds `20260831` 和 `20260830` 的 acceptance JSON，确认两次参照验收合同一致。
3. 等待用户明确决定是否授权运行 `20260832`；没有授权时只做只读检查。
4. 若授权，除 seed 与输出目录外不得改变冻结配置；完成后按相同入口做独立单 seed 验收。
5. 若失败，保全全部 checkpoint、history、raw metrics、predictions、runtime、manifest 和哈希并停止，不重复运行同一 seed。
6. 三种子全部通过后，执行独立跨种子 P4 审计；只有审计通过才记录 `P4 non-locked accuracy gate=passed` 并开放 P6，`locked_test` 仍保持关闭。

## 14. 文件与文档索引

### 必读治理与权威记录

- `AGENTS.md`
- `记录/本地计划表.md`
- `记录/PIJWM主文档.md`
- `记录/8.12之后推进.md`
- `记录/文件树与证据分层_20260826.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

### 当前方法代码

- `code/src/pi_jwm/formal_dual_graph_world_model_v1.py`
- `code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py`
- `code/src/pi_jwm/formal_motion_state_v1.py`
- `code/src/pi_jwm/formal_deterministic_rule_layer_v1.py`
- `code/src/pi_jwm/formal_world_model_loss_v1.py`
- `code/src/pi_jwm/formal_p4_gate_v1.py`

### 当前执行与审计入口

- `code/scripts/build_formal_causal_motion_tensor_v1.py`
- `code/scripts/run_formal_p4_first_principles_audit_v1.py`
- `code/scripts/run_formal_entity_aligned_rssm_consistency_audit_v1.py`
- `code/scripts/freeze_formal_p4_entity_rssm_protocol_v1.py`
- `code/scripts/run_formal_p4_entity_rssm_gpu_batch_probe_v1.py`
- `code/scripts/run_formal_p4_entity_rssm_gpu_v1.py`

### 后续规划器原型

- `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py`
- `code/src/pi_jwm/formal_candidate_rollout_planner_audit_v1.py`
- `code/scripts/run_formal_candidate_rollout_planner_audit_v1.py`

### 汇报材料

- `meeting/PI-JWM_组会汇报.pptx`
- `meeting/2026-09-09-PI-JWM组会汇报PPT验收.json`

组会 PPT 的结果页是在 seed `20260831` epoch 22 中间状态时制作的。当前 seed `20260831` 已完成并由 epoch 39 正式通过，因此 PPT 结果页相对当前正式证据已经滞后；只有在不影响训练主线时，才可用最终单种子产物更新该页，并必须继续保持“跨种子尚未完成”的边界。

## 15. 未知项、冲突与检查限制

- 旧版 `PROJECT_CONTEXT.md` 曾把“远端 GPU 不可用、link persistence residual 候选”为续接点；该状态已被 2026-09-06 至 09-08 的实体级 RSSM 正式证据覆盖。本文件以更新 artifact 和权威记录重建。
- `记录/PIJWM主文档.md` 中较早段落仍保留历史方法和失败状态；应按日期及“当前覆盖”段读取。其 2026-09-06 段仍写首种子正在训练，已被 2026-09-08 acceptance JSON 和最新计划/进度记录更新。
- 冻结协议 JSON 的状态字段记录协议冻结时刻，不能用来判断当前运行是否已开始或结束。
- 本次仅执行只读仓库、artifact 与远端状态检查，没有重跑全量测试，也没有改动训练进程。
- 工作树大量未提交变更的逐文件来源未完全追溯。版本整理是单独任务，不能夹在当前正式训练验收中进行。
- 本地 askpass 文件未读取；其有效期和凭据来源不在交接中记录。

## Final Audit

- [x] 当前阶段、阻塞和单一下一动作已明确。
- [x] 已验证、正在进行、已同意、候选和未知状态已分开。
- [x] 理论名称、实现对象、训练协议、数据合同、指标和证据路径已对齐。
- [x] seed `20260831` 的结论限定为单种子通过，没有扩展为三种子或最终性能声明。
- [x] seed `20260830` 的状态来自 2026-09-08 12:59 远端只读快照，并标记为运行中。
- [x] P6、策略器与候选生成器标记为尚未开放/未完成。
- [x] `locked_test_accessed=false` 与 `formal_performance_claim_ready=false` 明确保留。
- [x] 历史失败和旧方案保留为分析证据，没有被删除或重命名成成功结果。
- [x] 未记录密码、token 或 askpass 内容。
- [x] 本次交接只更新 `PROJECT_CONTEXT.md`，未修改代码、协议、实验产物或正在运行的训练。

## 2026-09-09 当前覆盖：seed 20260830 已完成

- seed `20260830` 已完成 base `20` + RSSM `40` epoch，最佳 RSSM epoch=`40`；9项单 seed 数值门全部通过。
- 正式产物：`code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1`。
- 独立验收：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`；manifest `79`项零差异，strict reload=`0/0`，原始指标重算最大差=`4.16e-09`。
- 远端进程已退出，GPU显存占用=`0 MiB`，可以释放。
- 当前按用户要求暂停，不启动 seed `20260832`。P4仍等待第三 seed和三 seed审计；P6、`locked_test`和正式性能声明继续关闭。
- 单一下一动作：等待用户明确决定是否运行最后一个冻结 seed `20260832`。
