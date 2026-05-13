# AirFogSim 字段映射表

## node_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `time` | $t$ | 离散时间步 |
| `node_id` | 节点编号 | UAV、vehicle、RSU、cloud 的唯一标识 |
| `node_type` | 节点类型 | 区分无人机、车辆、RSU、云节点 |
| `x, y, z` | $\mathbf{p}_{i,t}$ | 节点空间位置 |
| `speed` | $\|\mathbf{v}_{i,t}\|$ | 节点速度大小 |
| `acceleration` | 加速度状态 | 节点运动变化趋势 |
| `cpu` | $c_{i,t}$ 或 $C_{r,t}^{\mathrm{edge}}$ | 本地或边缘可用算力 |
| `storage` | 缓存/存储资源 | 节点可用存储资源 |

## link_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `tx_id, rx_id` | $(i,j)$ | 链路发送端和接收端 |
| `link_type` | 链路类型 | V2U、V2I、U2I |
| `distance` | $d_{ij,t}$ | 节点间距离，可用于物理边 |
| `rate_sum` | $r_{ij,t}$ | 链路传输速率 |
| `csi_mean` | $\gamma_{ij,t}$ | CSI/SINR 摘要的近似输入 |
| `active_task_count` | 任务占用强度 | 当前链路上激活任务数 |
| `allocated_rb_count` | $\eta_{ij,t}$ | 资源块占用数 |

## task_states.csv

| 原始字段 | 建模符号 | 含义 |
|---|---|---|
| `task_id` | 任务编号 | 单个任务唯一标识 |
| `task_node_id` | 任务源节点 | 任务产生位置 |
| `current_node_id` | 当前所在节点 | 任务当前执行或传输位置 |
| `assigned_to` | 卸载目标 | 当前任务被分配到的节点 |
| `task_size` | $D_{m,t}$ | 任务数据量 |
| `task_cpu` | $C_{m,t}$ | 任务计算需求 |
| `deadline` | $\tau_{m,t}$ | 任务时限 |
| `priority` | $\rho_{m,t}$ | 任务优先级 |
| `lifecycle_state` | $\mathbf{Y}^{\mathrm{task}}$ | 任务生命周期状态 |
