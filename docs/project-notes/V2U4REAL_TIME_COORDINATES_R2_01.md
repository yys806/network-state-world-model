# V2U4Real 官方时间、坐标与标定依据 — R2-01

核查日期：2026-10-26。**本页区分官方定义、源码行为和真实样本观测；不是时间同步算法或坐标变换验收。** 本轮没有取得 YAML，不能把论文中的原则自动绑定到某个原始字段。

## 1. 来源、版本与证据位置

| ID | 权威来源与版本 | 本轮证据 | 限制 |
|---|---|---|---|
| CODE | [官方仓库固定提交](https://github.com/VjiaLi/V2U4Real/tree/e099a3a746c6092a614e18b74d1f5b2f3c8351e8)，`e099a3a746c6092a614e18b74d1f5b2f3c8351e8` | 当前 upstream main 与 R1 相同；原 9 文件指纹全部匹配，R2 增加 4 个官方源码/依赖文件的只读审计 | 部分源码快照，不是完整训练安装 |
| HF | [官方训练集固定版本](https://huggingface.co/datasets/VJiaLi/V2U4Real/tree/36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a/train)，`36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，最后修改 2026-07-21 | 公开、非 gated；50 个训练包，25 个场景各 2 个来源；许可元数据为 CC BY-NC 4.0 | 许可名称是发布页事实，不将源码 MIT 许可套给数据；没有逐帧下载接口/包内索引 |
| LEGACY | [作者旧版辅助清单](https://huggingface.co/datasets/VJiaLi/V2U4Real/tree/8908a33d43ca05ddc82d89c1014d8b0906b67bc8)，`8908a33d43ca05ddc82d89c1014d8b0906b67bc8` | `FILES.tsv` 和 `SHA256SUMS`；选中训练包大小、SHA 与当前 LFS 对应；train 树未变 | 只有整包信息，不是逐文件校验或随机读取索引；清单已从当前根目录移除 |
| PAPER | [作者预印本 v1，含补充材料](https://arxiv.org/html/2603.25275v1)，2026-03-26 | 从 Zotero `PKU→PIJWM/05` 读取原文；条目 `4SKU5M6X`、附件 `JHB4X5KT`；正文 PDF 第 4 页、附录第 11 页已渲染核对 | 不等同于已核验 CVPR 最终版；没有复制论文 PDF 到仓库 |
| OWNER-1 | [仓库作者关于元数据的说明](https://github.com/VjiaLi/V2U4Real/issues/1#issuecomment-4294187896) | 作者明确说明逐时刻 GPS 定位、位姿次序、速度和相机参数等 | 没有给出本次样本的数值标定、时钟字段名或时间原点 |
| OWNER-7 | [作者发布说明](https://github.com/VjiaLi/V2U4Real/issues/7#issuecomment-5031674615) | 当前是部分发布；图像未公开；已发布训练/验证数据对应元数据、标定和标注；测试元数据/标注不公开 | 作者声明不等于本轮已下载 YAML；不使用其他用户评论代替作者结论 |
| CVF | [正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_V2U4Real_A_Real-world_Large-scale_Dataset_for_Vehicle-to-UAV_Cooperative_Perception_CVPR_2026_paper.pdf)及[正式补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_V2U4Real_A_Real-world_CVPR_2026_supplemental.pdf) | 官方仓库链接可确认地址 | 本轮直接访问返回 HTTP 403，标为 BLOCKED；不绕过访问限制，不宣称其内容与预印本逐字相同 |

PDF 原件 SHA-256：`f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c`，与 Zotero 附件 MD5 核对一致。论文仍只保留在 Zotero；本地页图只作只读核对，不纳入 Git。

完整来源记录见 `code/artifacts/manifests/v2u4real_r2_01_20261026/official_sources_index.json`；包目录响应、HTTP Range 回执、旧清单及指纹位于同目录 `official_sources/`。源码指纹见 `code/reference/V2U4Real.SOURCE.json` 和 `V2U4Real.R2_SOURCES.json`。

## 2. 时间语义：能确认什么

| 问题 | 已有依据 | 状态与不能推出的结论 |
|---|---|---|
| 名义采集频率 | PAPER §3.1 / PDF p4：10 Hz、统一时间参考 | **PASS（官方描述）**；不能推出每对相邻文件都严格相隔 100 ms |
| 同步机制 | PAPER 附录 C.3 / PDF p11：计算单元先对齐 GPS 时间，传感器结合 PTP（精确时间协议）与 PPS（每秒脉冲）硬件同步 | **PASS（官方描述）**；本轮未取得设备原始日志来复核 |
| 两端采集配对 | 同上：选择时间最近的两端 LiDAR 帧，再关联相机；作者报告跨端偏差控制在 20 ms 内 | **PASS（官方规则）**；这是允许小幅错时的配对，不是绝对同时采集的证明 |
| 文件标识 | CODE `basedataset.extract_timestamps()` 仅从文件名去掉后缀，返回字符串；按同一 key 查询两端内容 | **PASS（源码事实）**；变量名 timestamp 不自动赋予物理时间单位 |
| 实际见到的标识 | 两个完整 PCD 成员及相邻未完整成员的文件名分别含 `1752814590.2`、`1752814590.3` | **PASS（文件名观测）**；不将它们转换成 Unix/GPS 日期，不把差值直接称为秒 |
| 原始时间字段、单位、原点 | 没有取得 YAML 或逐帧原始时间清单；PCD 头没有本轮可解释的时间字段 | **BLOCKED**；`timestamp_raw`、`timestamp_unit`、参考原点均为 `null` |
| 设备时间→统一时间→文件名映射 | 原理见 PAPER，样本级映射未获取 | **BLOCKED**；不能以计划的 100 ms 或浮点外观替代映射证据 |
| 缺失/重复帧处理 | PAPER 表示发布前剔除了外部异常；CODE 假设各端 key 集合一致；转换脚本存在排序后 zip 的做法 | 没有找到适用于本样本的重采样/补帧规则，**BLOCKED**；不把 zip 当同步证明 |
| 异步通信实验 | PAPER D.2 与 CODE 的延迟计算通过较早帧模拟通信等时延；代码保留 `//100` 换算 | **PASS（基准设置/源码）**；模拟延迟、采集错时和真实丢帧是三件事，不作混用 |
| 后续仿真环境时间 | 本任务不定义真实时间到 AirFogSim 秒时钟的映射 | **NOT_RUN**；不做帧号×0.1 s，不启动轨迹回放 |

尚无完整三帧序列，本轮不报告实测采样周期、抖动、整场景缺失率或重复率。相同文件名只构成待核验的关联线索。

文献内部还需区分统计口径：§3.1 将场景时长概述为 30–80 秒，但附录表 7 中与所选训练场景相对应的 `2025-07-18-12-53-2` 一行报告 876 Frames、21.9 秒。它与公开包的下划线命名也不同。这里保留差异，不把表中汇总 Frames 当成双端配对时刻数，也不用任一时长反推当前文件的周期或缺失帧。

## 3. 坐标与标定：来源层级

| 对象 | 官方定义/源码行为 | 当前缺口 |
|---|---|---|
| LiDAR 局部轴 | PAPER §3.2 / p4：预定义变换统一为 x 前、y 左、z 上；安装方向可不同 | 预定义变换的数值、是否已应用于本次 PCD、传感器原始轴到统一轴的字段，均待 YAML/标定材料确认 |
| 世界系 | PAPER 为 NED，即北、东、下 | 本场景原点、地理参考、GPS 到局部 NED 的具体转换和单位绑定未取得 |
| 手性 | 上述轴向可在数学上解释为右手系 | 这是由轴向作出的推断，不是 YAML 的实测属性；不得据此擅加翻转矩阵 |
| 长度单位 | PAPER 检测范围和传感器量程以 m 表述 | 不能据此把未知 GPS/位姿字段全部当米；本轮 PCD 头不声明单位 |
| 车辆/UAV 位姿 | OWNER-1 给出 `(x,y,z,roll,yaw,pitch)`；CODE `x_to_world` 同序读取、角度默认度、内部按 ZYX 组成旋转 | 原始 `lidar_pose` 数据类型、物理参考点、角度符号/取值及其与 GPS/IMU 机体的关系未观察 |
| 传感器到机体 | 论文图 3 可见多个安装方向，§3.1 描述标定过程 | 不能把 LiDAR 原点、车辆/UAV 机体原点或无线天线位置视为同一点；数值外参未获得 |
| 跨端空间对齐 | PAPER §3.1 描述 RTK 初始化，再经 GICP 和人工调整 | 论文说明过程，不提供本场景每帧的数值矩阵；未核验实际矩阵的源/目标及是否包含注册修正 |
| 相机标定 | PAPER 描述内参、2D–3D 对应点与外参优化；OWNER-1 说明有相机内外参 | 不据别人的 issue 内容编造 YAML 中的矩阵字段或方向；本任务不需要下载 RGB |
| 3D 框 | PAPER §3.2：各 LiDAR 局部系独立标注，中心、l/w/h、欧拉角；跨源做 ID 一致性处理 | 实际 YAML `location/center/extent/angle` 的映射未获得；文本中列出的欧拉角名称不能直接替代数组顺序 |
| 框中心与尺寸顺序 | CODE `create_bbx` 对 extent 各分量除 2，沿 x/y/z 构角点；其 docstring 又写 width/height/length。`vehicle_to_single` 使用 location，而 `uav_to_single`/`project_world_objects` 使用 location+center | **文档/调用差异待样本核验**；不自行选择 lwh 或 whl，不猜 center 恒为零 |
| PCD VIEWPOINT | 两个文件头均是 `0 0 0 1 0 0 0` | 只记录头值，不当作 GPS 位姿、标定外参或车辆/UAV 悬停证明 |

实际 YAML 未获得，所以矩阵字段名、类型与尺寸没有“真实样本 PASS”。相机内外参、机体外参、跨端注册矩阵也不能相互替代。

## 4. R1 变换分支差异的进一步定位（只审计源码）

所有行号对应固定官方提交：

| 输入/路径 | 实际计算与调用 | 解释边界 |
|---|---|---|
| list + list | `transformation_utils.py:62`：先将两组位姿转矩阵，再用第二个的逆矩阵左乘第一个 | 这与“两个位姿都在世界系”的函数文档一致 |
| matrix + matrix | 同文件 `:75`：`inv(x2) @ x1` | 同上，前提是两者确实是到同一世界系的矩阵 |
| list + matrix | 同文件 `:69`：直接 `x2 @ x_to_world(x1)`，没有求逆 | 通用 docstring 没有解释这份额外契约，保留文档/实现冲突记录 |
| 真实数据主调用 | `intermediate_fusion_dataset.py:233`、`early_fusion_dataset.py:175`：`isSim=False` 传 `transformation_matrix`；`base_postprocessor.py:197` 再传给 `box_utils.project_world_objects`，最终进混合分支 | 静态调用链表明该参数是当前 LiDAR→ego 的直接变换，不是在此处把第二个位姿解释为“到世界系” |
| 仿真调用 | 上述条件的 `isSim=True` 分支改传 `ego_pose` 列表 | 走另一种参考参数契约；不把旧仿真对象坐标语义套到真实标签 |
| 独立转换脚本 | `uav_to_single.py:93` 用单位矩阵构局部对象，再在外层乘 UAV→ego 变换 | 单位矩阵不能用于辨别是否应求逆；它不证明所有非单位矩阵调用都正确 |

R1 的平移反例中，把第二矩阵按通用文档理解为“到世界系”时，两种结果最大元素差为 `12`，原证据保留在 R1 的 `official_transform_probe.json`。**这证明通用文档不能覆盖所有分支，不证明真实数据调用链必然算错。** 本轮只定位调用语义，没有重跑真实坐标变换、修改函数或替研究者选分支。闭环仍需真实 YAML、明确的矩阵源/目标和标注叠加核验。

## 5. 虚拟 RSU 后续需要独立配置的项目

本轮不确定最终位置，也不复用 R1 人工例中的 `(300,300,25)`。后续至少需要：公共系名称/原点/轴/单位；RSU 在该系的平移；RSU 感知轴方向；变换的源和目标命名；RSU 与逻辑服务器/天线参考点的关系；LiDAR 到机体/天线偏置；地图覆盖范围；时间原点与回放映射。数值均待后续批准。所有 RSU 数值只能标为 **仿真配置**，不能声称是 V2U4Real 实测设施。

## 6. 进入真实解析/变换前的解除条件

1. 同一个 train 场景的至少 3 个连续共同 Ouster PCD/YAML 对，或可恢复这些文件的官方最小包；优先 16 对。
2. 文件名、设备/采集/统一时间之间的字段映射、单位和参考原点；检查实际间隔，不预设 100 ms。
3. 实际位姿、注册/外参及标注的源/目标坐标定义；明确中心、尺寸和角度顺序；对每个矩阵逐项核验。
4. 对实际出现的无效点给出可审计的处理口径。当前 M1 文件不满足 R1 严格有限性要求；不得自动滤掉后宣称通过。

这些是后续条件，不是本轮获授权的实现队列。总状态和最小获取方案见 [R2-01 实施记录](../implementation_records/STEP_R2_01_V2U4REAL_DATA_PREPARATION.md)。
