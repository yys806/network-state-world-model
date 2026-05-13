# AirFogSim 状态演化、随机性与复杂度分析 v0

## 1. 这部分解决什么问题

这份文档回答三个问题：

- AirFogSim 内部每一步到底怎么从当前状态推进到下一时刻状态。
- 当前仿真里哪些变量本来就有随机性，哪些扰动是我们后续额外加的。
- 如果后续做学习式 world model，它和 AirFogSim 原始逐步仿真的关系是什么，复杂度优势应该怎么严谨表述。

一句话概括：AirFogSim 是“按规则逐步算”的仿真环境，我们现在做的是把它产出的 `node/link/task` 日志整理成世界模型可训练的联合时间序列；后续模型不是取代场景定义，而是学习一个更快、更可迁移的状态预测器。

## 2. AirFogSim 每一步怎么演化

源码入口是 `airfogsim_env.py` 里的 `AirFogSimEnv.step()`。它不是直接预测未来，而是把一个时间步拆成多个模块顺序执行：

| 顺序 | 模块 | 主要更新内容 | 对 dataset_v0 的影响 |
|---:|---|---|---|
| 1 | 交通更新 | 车辆由 SUMO 推进；UAV 根据速度、角度和俯仰角更新位置 | 决定节点位置、速度、加速度 |
| 2 | 任务更新 | 按 Poisson/Uniform/Normal/Exponential 等模型生成任务，并检查任务生命周期 | 决定任务数量、大小、计算量、deadline、状态 |
| 3 | 任务/调度决策 | 默认算法给出卸载、RB、CPU、返回路径等决策 | 影响链路激活、资源占用、任务状态转移 |
| 4 | 无线通信 | 更新 path loss、shadowing、fast fading，计算 SINR/rate，并推进任务传输进度 | 决定 `rate_sum`、`csi_mean`、RB 占用 |
| 5 | 有线通信 | 处理 RSU/Cloud 之间的回传链路 | 影响回传阶段任务进度 |
| 6 | 计算更新 | 根据 CPU 分配推进任务计算进度，完成后进入返回阶段 | 影响 `computing/returning/finished` 状态 |
| 7 | 存储、能量、区块链等 | 更新缓存、UAV 能耗和可选系统模块 | 当前 dataset_v0 只直接使用其中部分节点资源字段 |
| 8 | 状态记录 | 导出节点、链路和任务日志 | 形成后续训练样本来源 |

可以把一次状态转移写成：

$$
s_{t+1}=\mathcal{F}_{\mathrm{sim}}(s_t,a_t,\xi_t)
$$

其中，$s_t$ 是时刻 $t$ 的系统状态，包含节点状态、链路状态和任务状态；$a_t$ 是调度动作，例如卸载、RB 分配、CPU 分配和 UAV 移动；$\xi_t$ 是仿真中的随机因素，例如车辆到达、任务生成和信道衰落；$\mathcal{F}_{\mathrm{sim}}$ 是 AirFogSim 内部的规则集合。

这条公式的意思很简单：AirFogSim 的下一步不是凭空来的，而是由“当前状态 + 动作 + 随机扰动 + 仿真规则”共同决定。

![AirFogSim state transition flow](airfogsim_state_transition_flow.png)

## 3. 交通侧具体怎么变

车辆侧：

- 在 SUMO 模式下，`TrafficManager.stepSimulation()` 会先按 Poisson 分布生成新车辆数量。
- 新车会随机选取起点边和终点边，调用 SUMO 的路径搜索生成 route。
- 然后 SUMO 执行 `simulationStep()`，车辆位置、速度、路线随路网交通规则推进。

UAV 侧：

- UAV 初始位置在地图范围和高度范围内随机生成。
- 每个时间步根据速度、水平角、俯仰角显式更新位置：

$$
\mathbf{p}_{i,t+1}
=
\mathbf{p}_{i,t}
+
\Delta t
\begin{bmatrix}
v_{i,t}\cos\theta_{i,t}\cos\phi_{i,t}\\
v_{i,t}\sin\theta_{i,t}\cos\phi_{i,t}\\
v_{i,t}\sin\phi_{i,t}
\end{bmatrix}
$$

其中，$\mathbf{p}_{i,t}$ 是 UAV $i$ 在时刻 $t$ 的三维位置，$v_{i,t}$ 是速度，$\theta_{i,t}$ 是水平运动方向角，$\phi_{i,t}$ 是俯仰角，$\Delta t$ 是仿真步长。

这说明 UAV 运动不是神经网络学出来的，而是仿真器按运动学公式更新；我们后续模型要做的是从历史观测中学习这种状态变化规律。

## 4. 通信侧具体怎么变

通信侧主要由 `ChannelManagerCP` 负责，包含三类核心计算：

- 大尺度路径损耗：由节点距离、频率、通信类型决定，例如 V2I、V2U、U2I。
- 阴影衰落：反映建筑物、遮挡和环境带来的慢变化。
- 快衰落：每个时隙更新，反映小尺度信道波动。

链路速率不是手填的，而是由信道状态、发射功率、噪声、干扰和 RB 分配共同计算。可以抽象为：

$$
r_{ij,t}
=
B\sum_{r\in\mathcal{R}_{ij,t}}
\log_2\left(1+\mathrm{SINR}_{ij,r,t}\right)
$$

其中，$r_{ij,t}$ 是节点 $i$ 到节点 $j$ 在时刻 $t$ 的链路速率，$B$ 是单个 RB 的带宽，$\mathcal{R}_{ij,t}$ 是分给链路 $(i,j)$ 的 RB 集合，$\mathrm{SINR}_{ij,r,t}$ 是该链路在 RB $r$ 上的信干噪比。

在我们的导出数据里：

- `distance` 来自节点位置。
- `csi_mean` 来自信道矩阵的均值。
- `rate_sum` 来自各 RB 上速率求和。
- `active_task_count` 和 `allocated_rb_count` 来自当前任务卸载和 RB 分配记录。

## 5. 任务侧具体怎么变

任务由 `TaskManager` 管理。它主要做三件事：

- 生成任务：任务数量可以来自 Poisson、Uniform、Normal、Exponential 等分布。
- 生成任务属性：任务大小、计算需求、deadline、priority、返回数据大小可以来自 Uniform 或 Normal 分布。
- 更新生命周期：任务会在 `to_offload -> offloading -> computing -> returning -> finished/failed` 之间转移。

任务生命周期可以写成：

$$
q_{\tau,t+1}
=
\mathcal{T}_{\mathrm{task}}
\left(q_{\tau,t},a_{\tau,t},r_{ij,t},c_{n,t},d_{\tau}\right)
$$

其中，$q_{\tau,t}$ 是任务 $\tau$ 的生命周期状态，$a_{\tau,t}$ 是任务相关动作，$r_{ij,t}$ 是传输速率，$c_{n,t}$ 是分配到的计算资源，$d_{\tau}$ 是任务 deadline。

这个公式说明任务状态变化同时受通信和计算影响，所以它天然把物理侧、通信侧和任务侧耦合在一起。

## 6. 当前 dataset_v0 已经覆盖什么

当前 `dataset_v0_samples.npz` 已经形成了历史窗口到未来标签的训练样本：

| 字段 | shape | 含义 |
|---|---:|---|
| `x_node` | `(190, 8, 37, 7)` | 190 个样本，每个样本看过去 8 步，37 个节点，每个节点 7 个特征 |
| `x_link` | `(190, 8, 188, 5)` | 190 个样本，每个样本看过去 8 步，188 条候选链路，每条链路 5 个特征 |
| `x_task` | `(190, 8, 9)` | 190 个样本，每个样本看过去 8 步，任务聚合状态 9 个特征 |
| `y_node` | `(190, 3, 37, 7)` | 未来 3 步节点状态标签 |
| `y_link` | `(190, 3, 188, 5)` | 未来 3 步链路状态标签 |
| `y_task` | `(190, 3, 9)` | 未来 3 步任务状态标签 |

样本构造可以写成：

$$
\left(
\mathbf{X}^{node}_{t-H+1:t},
\mathbf{X}^{link}_{t-H+1:t},
\mathbf{X}^{task}_{t-H+1:t}
\right)
\rightarrow
\left(
\mathbf{Y}^{node}_{t+1:t+K},
\mathbf{Y}^{link}_{t+1:t+K},
\mathbf{Y}^{task}_{t+1:t+K}
\right)
$$

其中，$H=8$ 表示历史窗口长度，$K=3$ 表示预测未来 3 步。当前版本已经能训练“从历史状态预测未来状态”的 baseline；严格动作变量 $a_t$ 还没有作为单独张量导出，这是后续要补的。

## 7. 随机性来源

AirFogSim 本身已经包含随机性，但当前 `export_dataset_demo.py` 里设置了：

```python
np.random.seed(0)
random.seed(0)
```

所以当前这一版数据是可复现的。也就是说，它“机制上有随机性”，但“本次导出的轨迹固定”。

主要随机来源如下：

| 来源 | 代码位置 | 随机内容 | 当前意义 |
|---|---|---|---|
| 车辆到达 | `TrafficManager.stepSimulation()` | Poisson 生成新车辆数量 | 影响节点数量和交通密度 |
| 车辆路线 | `_generateRandomRoute()` | 随机选择起点/终点道路边 | 影响车辆轨迹和网络拓扑 |
| UAV 初始位置 | `_initialize_UAVs()` | 在地图范围内随机初始化 | 影响空地距离和通信质量 |
| 任务生成数量 | `TaskManager._generateTasks()` | Poisson/Uniform/Normal/Exponential | 影响负载强度 |
| 任务属性 | `_generateCPU/_generateSize/_generateDeadline/_generatePriority()` | Uniform/Normal | 影响计算、传输和时延 |
| 任务依赖 DAG | `generate_random_dag()` | 随机边生成 | 影响任务依赖结构 |
| 信道阴影/快衰落 | `all_channels.py` | LogNormal/Rayleigh 等 | 影响 CSI 和链路速率 |

本周我们额外做的扰动实验，是在 dataset_v0 输入侧加入合成噪声，用来测试预测模型面对观测扰动时是否稳定。这和 AirFogSim 内生随机性不是一回事：

- AirFogSim 内生随机性：仿真过程自己生成的随机事件。
- 合成扰动：我们在训练样本输入上额外加噪声，用于鲁棒性测试。

## 8. 复杂度怎么说才严谨

AirFogSim 的逐步仿真需要按模块显式计算。粗略地，主要计算量来自：

| 部分 | 主要变量 | 复杂度直觉 |
|---|---|---|
| 交通推进 | 车辆数 $N_v$、路网规模 | SUMO 负责，随车辆和路网规模增加 |
| UAV 运动 | UAV 数 $N_u$ | 近似 $O(N_u)$ |
| 候选链路构造 | 节点数、链路数 $E$ | 需要遍历候选通信对 |
| 信道矩阵/速率 | 车辆、UAV、RSU、RB 数 $R$ | 常见项类似 $O(E R)$，部分矩阵接近 pairwise |
| 任务生成和状态检查 | 活跃任务数 $M$ | 近似 $O(M)$ 到 $O(M+E_{task})$ |
| 调度与资源分配 | 活跃任务数 $M$、RB 数 $R$ | 取决于具体策略，可能随 $M R$ 增长 |

学习式 world model 的潜在优势不是“现在已经证明更快”，而是：

- 训练阶段：需要离线训练成本。
- 推理阶段：给定历史窗口后，可以用一次或少量前向传播预测未来 $K$ 步。
- 如果未来要做大量候选动作评估，学习模型可能比反复调用完整仿真器 rollout 更快。

更稳妥的对比公式是：

$$
C_{\mathrm{sim}}(K)
\approx
K\cdot
\left(
C_{\mathrm{traffic}}
+C_{\mathrm{channel}}
+C_{\mathrm{task}}
+C_{\mathrm{scheduling}}
\right)
$$

$$
C_{\mathrm{wm}}(K)
\approx
C_{\mathrm{encode}}(H)
+K\cdot C_{\mathrm{latent\ rollout}}
+C_{\mathrm{decode}}(K)
$$

其中，$C_{\mathrm{sim}}(K)$ 是 AirFogSim 做 $K$ 步显式仿真的成本，$C_{\mathrm{wm}}(K)$ 是学习式 world model 做 $K$ 步预测的成本。我们后续要验证的是：在目标误差可接受的前提下，$C_{\mathrm{wm}}(K)$ 是否能低于 $C_{\mathrm{sim}}(K)$。

## 9. 这周已经完成的实验位置

| 内容 | 输出位置 | 当前结论 |
|---|---|---|
| baseline 训练 | `baseline_v0/` | dataset_v0 已能进入训练-测试流程；短期预测中 persistence 很强 |
| 扰动实验 | `robustness_v0/` | 已建立扰动评估流程；当前 residual baseline 在强扰动下退化 |
| 置信区间 | `uncertainty_v0/` | 已能输出 80%/90% 区间；方法是验证集残差分位数法 |
| 机制分析 | `airfogsim_analysis_v0/` | 已梳理 AirFogSim 状态演化、随机性和复杂度口径 |

## 10. 下一步应该怎么做

下周可以按这个顺序推进：

1. 补严格动作变量导出：把卸载动作、RB 分配、CPU 分配、UAV 移动动作整理成 $a_t$。
2. 做计时实验：同一个 $K$ 步预测任务，统计 AirFogSim rollout 时间和 baseline/world model 推理时间。
3. 做内生随机性实验：不同 seed 生成多条轨迹，而不仅是在输入上加合成噪声。
4. 做扰动训练：用 noisy input 训练，再测试 noisy input，看鲁棒性是否真正改善。
5. 升级模型：从 persistence/Ridge/MLP baseline 过渡到双图编码器或动作条件 latent world model。

当前最适合对外展示的表述是：

> 本周已经把数据构造、baseline、扰动评估、置信区间和仿真机制分析这几条链路跑通。当前还不是最终模型效果，而是为后续动作条件 world model 和复杂度/鲁棒性验证建立了实验基础。
