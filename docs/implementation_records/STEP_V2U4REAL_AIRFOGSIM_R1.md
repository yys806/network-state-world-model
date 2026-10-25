# STEP R1 — V2U4Real × AirFogSim 第一轮接口可行性验证

日期：2026-10-25。性质：研究者本次明确授权的独立 CPU 接口实验，不是旧 Step 6.4 的续跑，也不是新研究方法验收。

## Step Goal / 本轮结论

**软件接线可以继续做，但真实数据接入尚未通过。** 在真实 `AirFogSimEnv` 中，人工指定的车辆/UAV 绝对位置经独立适配层传到实体、地图网格和信道；两次各 3 步运行可重复。真实 V2U4Real 样本未就绪，坐标/时间语义未核验；原生回放和无线计算也有明确缺口。因此本轮不能宣称 V2U4Real 已成功接入，更不能开始通信服务或世界模型实验。

| 实验 | 总状态 | 已验证的较小范围 | 阻塞原因 |
|---|---|---|---|
| A 数据读取与同步 | BLOCKED | 官方源码格式审计；人工 PCD/YAML 读取与异常测试 PASS | 未找到可用训练子集；真实字段值、时间单位、周期、异常率均未知 |
| B 公共坐标与 RSU | BLOCKED | 人工刚体变换、逆变换、固定虚拟 RSU 变换 PASS；官方 pose–pose 分支数值检查 PASS | 缺真实点云/标注/标定；物理坐标轴与传感器外参未核验；官方混合参数分支契约有歧义 |
| C 轨迹回放 | BLOCKED | 独立适配层驱动真实环境的位置/环境时钟/网格检查 PASS | 原生 real 模式漏新出现的车辆；原生车辆 z=0；UAV setter 不是绝对位置接口；实体更新时间标签落后一帧；真实位姿/时间合同缺失 |
| D V2I/U2I | BLOCKED | 真实环境初始化、数组尺寸、位置传递、有限性、距离响应、固定种子重复 PASS | V2I 高度/水平距离语义冲突；中断概率单位问题；未分配资源块伪速率；快衰落实现限制 |
| E 特征包服务 | NOT_RUN | 无 | C、D 未整体通过，按前置条件停止；未实现服务验证器 |

**PASS 子项只证明相应的软件行为。** A–D 的总状态均为 BLOCKED，退出码 `2` 是预期的有阻塞完成，不是成功退出 `0`。GPU、训练、正式数据协议、旧模型/检查点、locked_test 均未使用或修改。

## Definition Basis / 依据与适用边界

1. 研究者当前消息直接授权 A–E 接口范围、固定编码与压缩、记录轨迹、V2I/U2I 和固定逻辑 RSU；没有授权研究算法、传播模型或正式协议变更。
2. 请求中的 `D:\shen\OB\PIJWM\组会笔记\1028组会.md` 不存在。实际只读文件为 `D:\shen\OB\科研\PKU\PIJWM\组会笔记\1028组会.md`，SHA-256 `4eb9fd6ee41855b6bfc74df204eb45ba678f9680e33449f207d151bc9ff6d8b5`。本轮依据 `# 一、这周要解决的问题` 之后的正文，尤其 `二、具体实现` 的场景组成、任务定义、数据集与仿真器分工；不采用其前面 `# 讲稿` 中“相机、多车、UAV 中继”的旧描述。不复制整份笔记进仓库。
3. 当前消息对本实验的场景口径优先于较早 10.23 导航；没有据此冻结世界模型、任务价值、编码器、特征大小、损失或检测质量指标。
4. 官方代码：[VjiaLi/V2U4Real](https://github.com/VjiaLi/V2U4Real/tree/e099a3a746c6092a614e18b74d1f5b2f3c8351e8)，版本 `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`。只获取 9 个审计所需源码文件，逐一核对官方 Git blob 和 SHA-256；不是完整安装。无数据集或模型下载。官方数据入口 [Hugging Face](https://huggingface.co/datasets/VJiaLi/V2U4Real/tree/main) 只查看目录页面。

## Initial State / 初始状态

- 工程分支 `main`，基线 `bd8af505aeeb95c489903e078b3d36fb0713bdb8`。起始已有修改、删除和未跟踪脚本，包括 AGENTS、AI_CONTEXT/00、索引、计划，以及过程文件、论文资料删除。本任务不恢复、不清理、不混入提交。初始清单本地存于 `.tmp/v2u4real_round1/initial_status.txt`。
- `task_plan.md`、`progress.md`、`findings.md` 起始已删除；遵照项目记忆 C7，不重新创建，以 `记录/本地计划表.md` 和本记录替代。
- 可访问工程路径中未找到 V2U4Real 代码或 PCD；用户未提供已有数据路径，并允许在工程内部建目录。未遍历其他项目、私有笔记目录或外盘寻找数据；历史产物部分路径存在访问限制。因此“未找到”不等于断言整台电脑没有数据。
- `code/reference/AirFogSim/` 存在，但此目录的 Git 命令向上解析到 PIJWM 仓库，不能把返回的 PIJWM commit 当成 AirFogSim 版本。本轮以逐文件 SHA 固定所运行的 vendored 源码，未修改它。
- 默认 base Python 无法直接导入完整环境（缺地图依赖）；最终使用已存在的 `D:\miniconda\envs\airfogsim\python.exe` 成功导入真实环境。该环境没有 SciPy，官方变换单独用 base Python 的 SciPy 做 CPU 检查。曾向仓库 `.tmp` 安装地图依赖，但该目录仍不可读/不可用，最终实验不依赖它，不修改全局环境。
- 起始索引 `--check` 已失败：`tracked_file_inventory.csv`、`artifact_catalog.csv`、`registry_summary.json` 与本地状态不一致；这是本轮新文件入索引前的实测，不归因于新方法。

## Experiment A / 字段、文件和时间同步审计

### 官方代码契约与真实观测分开记录

| 对象 | 官方代码所读取/生成的契约 | 真实样本观测 |
|---|---|---|
| 两个来源 | `1` 车辆、`2` UAV；本轮共同使用 `ouster`；车辆另有 `ruby/m1` | 未观测 |
| 文件关联 | `train/<scene>/<agent>/ouster/<key>.pcd` 和 `yaml/ouster/<key>.yaml` | 0 个实际样本；场景、时间范围、帧数均 `null/UNKNOWN` |
| `<key>` | `basedataset.extract_timestamps` 直接提取文件名；它的注释称 mocked timestamps | 不能将 `000001` 当秒或毫秒；不能由它推断 100 ms |
| `lidar_pose` | 变换函数消费 6 元列表 `[x,y,z,roll,yaw,pitch]`；默认角度为度；`Rotation.from_euler('ZYX', [yaw,pitch,roll])` | 真实类型/数值、米制、原点和轴向均未核验 |
| `vehicles` | 标注字典；对象读取 `location/angle/center/extent`，可选 `obj_type`；包围框函数以 `extent/2` 构造角点 | 对象 ID、维度含义、标注所在坐标系与物理时间均未核验 |
| 变换矩阵 | `reform_param` 运行时生成 `transformation_matrix/gt_transformation_matrix/spatial_correction_matrix` | 不能声称这些矩阵原本就在 YAML 中 |
| 物理时间戳、传感器到机体外参 | 审计代码不能从官方通用加载逻辑确定其真实字段/单位 | BLOCKED；需要原始元数据或官方说明 |
| 点云强度 | 官方 `pcd_to_np` 使用 Open3D 点颜色第一通道，无颜色时补零 | 本轮只实现并测试 XYZ 读取；强度语义未核验，不作为编码器输入 |

审计脚本只读取 `train`，按两端 YAML 总数选择最小有配对目录，最多读取配置指定的 16 个帧编号，保留完整文件名集合用于编号缺口检查。它输出实际字段类型/列表长度、原始时间字段、PCD 头、点数、范围、文件 SHA、匹配/缺失编号、重复原始时间值和解析异常；**即使文件能读，也不会自行宣布物理时间或坐标已核验**。当前无样本，以上实际统计为空，不报告虚构的采样间隔。

自动化反例覆盖重复 YAML 键、重复时间值、缺失配对、帧编号跳号、坏 PCD、NaN、ASCII/binary 读取、非训练入口和截断审计。`binary_compressed` 明确 BLOCKED，需在获取样本后用官方 Open3D 读取路径补验，不静默返回空点云。

官方 `basedataset.py` 按相同 key 取另一端帧，同时假设各端帧数相同；`uav_to_single.py` 对排序后的两个列表直接 `zip`。后者在缺失文件时可能错配。本适配审计逐 key 核对，不采用 zip 作为同步证据。官方异步延迟仍有 `//100` 的遗留换算，不能据此确认本数据集实际为 10 Hz。

### 下一轮所需最小样本（尚未下载）

- 一个已确认属于 **train** 的同一场景，车辆 `1` 与 UAV `2` 的 **3 个连续共同帧**，各帧包含两端 `ouster PCD + 原始 YAML`：最低 **6 个 PCD + 6 个 YAML**。保留原文件名、路径和校验值，不重写位姿或时间字段。
- 同场景完整文件名清单、原始采集时刻/帧号映射及单位说明；若 YAML 无物理时间，需官方时间清单或相应记录。只有三个编号不足以验证整个场景的缺失率或固定周期。
- LiDAR 到车辆/UAV 机体、机体到公共系的标定和轴向/单位/旋转顺序说明，及标注坐标与采集时间的依据。若数据只提供 LiDAR 全局位姿，不能未经说明把它当通信天线或机体位姿。
- 三帧足够做首个往返与相邻间隔检查；若要统计周期抖动/缺帧，应再取连续 16 帧及全序列时间清单，仍不需要整个数据集。

## Experiment B / 坐标与虚拟 RSU

独立函数接受显式的 `4×4` 刚体矩阵，不猜 YAML 含义。对列向量有 `p_common = R p_sensor + t`，RSU 系使用 `T_rsu_to_common` 的逆矩阵。`R` 是旋转矩阵，`t` 是平移；要求旋转正交、行列式为 +1，拒绝缩放、镜像和非有限数。

人工例中，绕 z 轴旋转 90° 并平移 `(20,30,4)` 后，点 `(1,2,3)` 应到 `(18,31,7)`；已验证这个独立已知结果以及正逆变换。RSU `(300,300,25)`、Z-up、米制及地图 `[0,600)×[0,600)` **全是人工仿真设定**，与 V2U4Real 的真实坐标定义无关。实测往返误差为 `0`，但真实点云与 3D 框空间对齐 **NOT_RUN/BLOCKED**，未用人工框冒充真实标注。

官方 `x_to_world` 的 pose–pose 分支人工数值与往返检查通过。另一个 `list + matrix` 分支直接乘入第二个矩阵，与该函数文档“第二个参数是到世界系矩阵”的说明不同：平移 `(4,5,6)` 的人工例与文档意义下的逆矩阵最大元素差为 `12`。这说明该分支需按实际调用语义核验；不证明真实数据所有路径都错，不擅自修复或选择相反的坐标解释。源码的 NED 注释也不足以单独建立真实数据的北/东/下方向。

## Experiment C / 原生与适配回放

真实执行 `TrafficManager` 和 `TrafficScheduler`，得到：

1. 原生 `real` 模式先按旧时刻选 ID，再推进时钟，再按新时刻筛数据。只有一条 `t=0.1` 的新车辆记录时，在 `0.1` 与 `0.2` 均没有该车；已记录反例。连续时间零起始记录可到后续帧，但不是所有数据都能正确回放。原生 CSV 使用半开窗口 `(t-dt,t]`，窗口重复记录也可能按迭代顺序覆盖；不能直接对不规则采集时间宣称兼容。
2. 原生车辆只读取 x/y，z 强制为 0，acceleration 强制为 0；无法保留任意三维实测位姿。原生速度/angle 字段不等于完整机体姿态。
3. `setUAVMobilityPatterns` 丢弃额外的绝对位置字段。速度 `2 m/s`、角度/仰角为 0、`dt=0.1 s` 后，位置增量实测 `(0.19999999999998863,0,0)`，符合积分公式，不能当作绝对位置设置。
4. 独立子类只在实验实例内安装，不改第三方源码。它要求显式秒、零起点、均匀间隔、固定一个车辆/UAV、合法地图和高度范围；直接逐帧安装三维位置，调用原生网格更新，再由真实 `env.step` 更新实体/信道。拒绝不规则时间、越界、机动命令、数据耗尽后的外推；reset 回到首帧。没有将原始 YAML 自动转换成已核验轨迹。
5. 人工样本车辆 `10 m/s`，UAV 约 `11.18 m/s`，经过 x=350 m 的网格边界；固定 RSU 保持 `(300,300,25)`。两次各 3 个真实环境 step 的位置、环境/TrafficManager 时刻、网格成员相同。
6. 原生实体 `_last_updated_time` 使用推进前的 `env.simulation_time`，而位置来自推进后的 TrafficManager：在环境 `0.1 s` 时，该字段仍为 `0.0 s`。适配实验如实记录，未重写此标签。因此软件位置检查 PASS 不等于全链路时间语义 PASS。

最小替代路径已经实现为子类适配层。若下一轮要改 vendored 源码，先报告：车辆 ID/时钟查询顺序、z/acceleration 丢失、更新时间标签三个具体修改点，并比较独立适配层方案；**本轮不修改第三方源码**。

## Experiment D / 无线接口与数值语义

在真实环境中使用一个车辆、一个 UAV、一个 RSU；保留原生 `50` 个资源块，每块 `2 MHz`、起始 `2.4 GHz`。数据任务生成关闭；CPU 分配仅提供环境要求的空分配回调。没有把感知特征包转换为原计算卸载任务。

路径损耗是随距离变化的基础衰减；阴影是较慢的随机变化；快衰落是短时随机变化。当前最终信道数组表达的是 **dB 衰减**，不是复数信道测量：`final = PathLoss + Shadow - FastFading`。二者的 PathLoss/FastFading/final/rate 为 `[1,1,50]`，Shadow 为 `[1,1]`；资源激活数组也是 `[1,1,50]`。逐项检查有限值和公式分解；V2I 激活 RB0、U2I 激活 RB1。返回速率单位从源码确认是 **Mbps**；未来换字节需 `Mbps × 10^6 / 8 × 秒`，本轮未执行特征包传输。

固定 `random` 和 NumPy 种子 `20261025` 后，两次完整运行的逐数组结果相同；这只验证当前环境版本可重复。移动后路径损耗改变，固定相对高度并将水平距离从 `50/100/200/400 m` 增大时，两链路的基础损耗均单调增加；快衰落后的最终衰减不要求单调。

| 接口事实或问题 | 实测/源码证据 | 解释与边界 |
|---|---|---|
| U2I 自由空间路径损耗 | `74.9927/80.3075/86.1322/92.1024 dB`（第一个频点） | 符合此实现的三维距离公式，不能证明真实场景传播准确 |
| V2I UMa_LOS | 当前实现 `74.8250/80.1359/86.3655/93.2177 dB` | 单调检查 PASS；物理几何仍 BLOCKED |
| V2I 高度重复/默认值 | 三维位置差的范数被当作 `d_2d`，随后又加入默认 `h_BS=25/h_UT=1.5` 的高度差；100 m 例 `80.135909` 对文档水平距离公式 `79.861021 dB`，相差 `0.274889 dB` | 当前节点 z 没有一致传入模型高度参数；不得用调 RSU 坐标掩盖，需研究者决定独立严格模型还是最小修复 |
| 中断概率单位 | `computeRate` 将 dB SINR 传给 `1-exp(-threshold/snr)` | 历史合同 `docs/contracts_PIJWM_STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL_V1.md` 已明确记录此源码事实；本轮不是新发现旧合同错误，也不回写旧结果 |
| 未分配 RB 伪速率 | 仅诊断时暂设 outage threshold=0，随后恢复；未激活 RB 仍约 `2.0000000003 Mbps` | 来自 SINR 的正值下限。不能把全数组求和当可服务字节数；当前只检查指定 RB |
| 快衰落限制 | `Rayleigh_fastfading` 对一次采样计算 PDF 值再取 log，并跨所有 RB 广播 | 不是独立逐 RB 的标准瑞利功率抽样；数组有限不等于传播统计正确 |

没有修改任何传播模型、阈值协议或旧信道/结果文件。最后三项不能通过掩码或改名宣称无线服务已经符合理论。数据集不提供本实验的真实无线 CSI；全部信道状态、速率和资源分配均为仿真/人工配置。

## Experiment E / 条件停止

`NOT_RUN`。未实现队列、跨时隙进度、截止时间和完成事件；不声称它们已经通过测试。未来若 C、D 通过，应使用独立的逻辑特征包层，复用 AirFogSim 的几何、信道与指定 RB 的速率计算接口，另外核验 Mbps 到字节的换算、准备时刻、跨隙累计、一次完成和逾期判定。不得直接套用原计算卸载 Task 的语义。

## Files Involved / Changes / Reuse

- 可复用工具：`code/src/pi_jwm/v2u4real_interface_v1.py`，仅严格 XYZ/元数据审计、显式刚体变换和隔离绝对回放子类。
- 入口：`code/scripts/run_v2u4real_airfogsim_interface_v1.py`；官方变换 CPU 检查 `probe_v2u4real_official_transform_v1.py`；小源码恢复 `fetch_v2u4real_audit_sources_v1.py`。
- 精简配置/人工地图：`code/scripts/configs/v2u4real_airfogsim_interface_v1.json`、`v2u4real_tiny.net.xml`。
- 测试：`code/tests/test_v2u4real_interface_v1.py`。第三方锁定清单：`code/reference/V2U4Real.SOURCE.json`；源码目录被忽略，不提交第三方资产、数据或模型。
- 复用原生真实环境生命周期、实体更新、地图网格、ChannelManagerCP、路径损耗/阴影/快衰落/干扰/速率接口；没有复用旧世界模型、调度器或感知模型。
- 更新本记录、必要导航/AI_CONTEXT、实验/问题/后续工作注册表；正式结果注册表不新增科学性能数字。

## Validation / 实际命令与输出

在 `D:\shen\PKU\PIJWM` 执行，先设 `PYTHONUTF8=1`，保留环境原有 SUMO tools 路径，再将 `code/src;code/scripts` 加入 PYTHONPATH。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts;'+$env:PYTHONPATH
D:\miniconda\envs\airfogsim\python.exe code/scripts/run_v2u4real_airfogsim_interface_v1.py
# A/B/C/D BLOCKED；C adapted_synthetic_status=PASS；D interface_status=PASS；E NOT_RUN；exit 2
D:\miniconda\envs\airfogsim\python.exe -m unittest discover -s code/tests -p test_v2u4real_interface_v1.py -v
# 17 tests OK
D:\miniconda\envs\airfogsim\python.exe -m unittest discover -s code/tests -p test_airfogsim_contract_adapter.py -v
# 9 tests OK
D:\miniconda\envs\airfogsim\python.exe -m unittest discover -s code/tests -p test_raw_single_decision_step_contract_v1.py -v
# 9 tests OK
D:\miniconda\envs\airfogsim\python.exe -m unittest discover -s code/tests -p test_airfogsim_cpu_inner_rule_v1.py -v
# 5 tests OK
python code/scripts/probe_v2u4real_official_transform_v1.py
# pose_pose_branch PASS；mixed_branch_error 12.0；dataset_transform_status BLOCKED；exit 0
python code/scripts/fetch_v2u4real_audit_sources_v1.py
# 9 files VERIFIED；只恢复源码，绝不下载数据/权重
python -m compileall -q code/src code/scripts code/tests
python code/scripts/build_project_knowledge_index_v1.py
python code/scripts/build_project_knowledge_index_v1.py --check
```

原始运行目录：`code/artifacts/audit/v2u4real_airfogsim_round1_20261025/`，包含 `data_audit.json`、`geometry.json`、`official_transform_probe.json`、`native_replay.json`、`adapted_environment.json`、`distance_probe.json`、`runtime_config.json`、`source_identity.json`、`acceptance.json` 和命令日志。精简验收与 SHA 清单位于 `code/artifacts/manifests/v2u4real_airfogsim_round1_20261025/`，随代码提交，不含真实数据。

## Results / Expected vs Actual / Known Issues

**部分符合预期。** 按授权完成当前条件可做的审计和真实 CPU 接口验证；未获得真实数据，也未把已发现的语义冲突改成通过。真实点云读取、真实跨源同步/标注空间对齐、真实轨迹及特征服务未验收。独立适配层仅是候选工程接口；不产生检测质量、网络真实性、世界模型效果或方法创新结论。

Context Consistency Check：新条目限定本实验范围，旧 Step、双图/RSSM/Planner/Scorer、数据划分、checkpoint 和历史结果保留；`06_DECISIONS.md` 不添加 AI 自行作出的研究决策。

## Git / Next Step

目标提交：`feat(interface): audit V2U4Real and probe AirFogSim replay channels`，分支 `main`。实际提交与 push 状态在完成汇报及 Git 历史核验；不将任务前的修改/删除纳入。索引处理的实际结果见本记录后续补充。

2026-10-26 收尾说明：暂存前逐一核对交付证据和代码在 Git 中的字节指纹，统一换行后重新执行探针与 17 项专项测试，均符合本轮限定预期。四份原有修改的共享文档只暂存本轮新增段落；AGENTS、旧资料删除和未跟踪旧脚本不纳入。普通进程读写生成索引/部分历史元数据被 Windows 拒绝，使用获准的权限后索引生成返回 `passed: true`；没有修改历史证据或权限。为避免知识索引混入原有删除和未提交文本，另在仓库 `.tmp/v2u4real_round1/index_view/` 建立暂存内容的只读审阅视图，复用实际产物作只读索引输入；提交索引以该视图重建。工作区与暂存视图的复核日志分别保存在 `.tmp/v2u4real_round1/index_check.log` 和 `index_staged_check.log`，最终状态见完成汇报。

唯一建议下一动作：取得上述 **3 个连续 train 配对帧与官方时间/标定说明**，只重跑 A/B 审计。传播模型修正需要单独提供最小修改方案并由研究者决定；本轮在这里停止，不自动启动下一轮或 E。
