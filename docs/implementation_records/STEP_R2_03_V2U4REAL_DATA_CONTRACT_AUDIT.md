# STEP R2-03 — V2U4Real 数据合同与证据核验

日期：2026-10-26。**文件读取和文件键序列可用；真实时间语义、跨端物理配对、坐标/标定合同及重复 PCD 成因仍有阻塞。R2-04 真实坐标变换尚不具备准入条件。**

## Step Goal / Definition Basis

本轮只核实固定 V2U4Real 训练样本的时间字段、官方配对实现、坐标和标定依据、重复点云及下游数据合同。没有运行真实坐标变换、点云/标注融合、AirFogSim 回放、无线服务或训练；没有修改数据、第三方代码或 PI-JWM 方法。没有使用 GPU、validation/test/locked_test。

数据来源及范围：数据集固定版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，`train/2025-07-18-12-53_2`，车辆 agent `1` Ouster 与 UAV agent `2` Ouster；只审计此前选出的 16 个文件键 `1752814590.2`–`1752814591.7`。原始包、选中 PCD/YAML 均留在 Git 忽略的数据目录。

论文使用 Zotero 原文附件 `JHB4X5KT`，题名 *V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception*，对应 arXiv `2603.25275v1`（2026-03-26）。本轮直接读取本地 PDF：26,393,888 bytes、15 页，SHA-256 `f3257de22a9e438c3e755237bef901dd0b889b0d58b4a76a1040e3faa8a6350c`；核对 PDF 第 4–6、11 页。原文入口：[arXiv v1 HTML](https://arxiv.org/html/2603.25275v1)。CVPR 最终版 PDF 的访问在既有官方来源索引中记为 HTTP 403；本报告的论文结论限于上述 v1，不声称已逐字比较最终版。

官方代码来源为 [VjiaLi/V2U4Real](https://github.com/VjiaLi/V2U4Real)，固定 commit `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`，身份依据 `code/reference/V2U4Real.SOURCE.json` 与 `V2U4Real.R2_SOURCES.json`。该目录不是独立 Git checkout；不得把父仓库 HEAD 当作官方代码 commit。本轮对关键文件重新计算 SHA-256，均与固定来源清单一致：

| 官方源码 | SHA-256 | 核验 |
|---|---|---|
| `opencood/data_utils/datasets/basedataset.py` | `aea708b371977015deb5fbe45ad3ed19a912b34d2cc66cf0482f7feff63f19aa` | PASS |
| `opencood/utils/transformation_utils.py` | `2c198a94ffaf9ac3087396ee364b866a817fb2a075ac7c37c618cb03091ebd80` | PASS |
| `opencood/utils/uav_to_single.py` | `cab7b48fe7c343cbb356db131cd1b87f64249f9cfcb7b4073fdeac18ed51ec4b` | PASS |

初始 PIJWM 仓库分支为 `main`、HEAD `f05bbe693aea23f31de839fda473ce717bb60fe2`。工作区有本轮开始前的修改、删除和未跟踪文件；均保留，最终只选择性提交 R2-03 文件与独立上下文增量。

## Initial State / Files Involved / Reuse

本轮实际读取 R2-02 实施记录、时间/标定说明、官方来源与样本清单、16 对本地 PCD/YAML、官方代码及 arXiv v1 原文。R2-02 已取得 16 个发布文件键上的 32 个 Ouster PCD 和 32 个 YAML；其完整整包 SHA 与样本来源仍以 R2-02 记录为准。本轮没有重复下载或展开数据。

复用 `code/scripts/audit_v2u4real_r2_03_v1.py` 和原 `pi_jwm.v2u4real_interface_v1` 中的 `read_pcd_xyz`、`UniqueKeyLoader`、SHA 工具。审计结果写入 `code/artifacts/manifests/v2u4real_r2_03_20261026/real_data_contract_audit.json`。没有另造点云解码器，也没有降低严格读取规则。

## A. 时间语义与样本配对证据

| 结论 | 直接依据 | 状态与边界 |
|---|---|---|
| 名义采集频率及统一时间参考 | 论文 §3.1（PDF 第 4 页）称所有传感器以 10 Hz 记录并使用统一时间参考。 | **仅有论文依据**；不证明所选相邻文件的实际物理间隔严格为 100 ms。 |
| 采集同步机制和最近帧规则 | 论文附录 C.3（PDF 第 11 页）称计算单元先同步到 GPS 时间，再以 PTP/PPS 硬件同步 LiDAR 与相机，并按跨 agent 时间最近的 LiDAR 帧配对；论文报告每个采集样本的跨端时间差在 20 ms 内。 | **仅有论文依据**；本地没有逐帧原始设备时钟、GPS/PTP/PPS 日志或配对误差。不能把论文总体声明当作这 16 对的测量结果。 |
| 官方主数据读取器的关联方式 | `basedataset.py:173–203` 按排序后的 YAML 文件名去扩展名取得字符串键，使用同一键定位同 agent 的 YAML 和 PCD；读取其他 agent 时沿用由首个 agent 取得的键。`extract_timestamps()`（`:314–336`）注释称其为 “mocked timestamps”，实现只是取文件名 stem；`return_timestamp_key()`（`:339–362`）按序号返回该键。 | **代码事实**；这是发布文件关联规则，不是最近物理时刻搜索。读取器还假设各 agent 时间键数量相同。 |
| 官方导出工具的另一种关联方式 | `uav_to_single.py:161–163` 对 ego/UAV YAML 列表分别排序后直接 `zip`，输出编号来自列表序号。 | **代码事实**；有缺帧或序列差异时，按位置 `zip` 可能把不同 stem 配成一项。对未来真实数据处理，应显式核对原始键，不沿用序号配对作为物理同步证据。 |
| 本地 YAML/PCD 的实际时钟字段 | 本轮对 32 份 YAML 递归检查 `time/stamp/timestamp` 类字段，均为空；根字段是 `ego_speed`、`lidar_pose`、相机标定和 `vehicles`。Ouster PCD 头字段为 `x y z rgb`，没有采集时钟字段；`VIEWPOINT` 是恒等值，不是采集位姿。 | **本地样本已核验**：没有能恢复设备时间或物理时间的字段。 |
| 16 个选中键 | 两端 PCD/YAML 均存在；键顺序连续，从 `1752814590.2` 到 `1752814591.7`，键的十进制文本相邻差为 `0.1`。 | **文件键连续 PASS**；数值差单位、参考原点和含义均未核实，不能当作秒。 |
| 车辆/UAV 物理采集配对及偏差 | 两端相同 YAML stem 是本地可追溯的文件关联；没有本地时钟值可套用论文“最近帧”规则。 | **BLOCKED**；不能证明相同 stem 对应最近采集时刻，也不能验证偏差 ≤20 ms。 |

因此可建立一个**相对排序索引** `sim_frame_index=0..15`。若后续接口冒烟测试显式配置 `sim_time_s = sim_frame_index × 0.1`，其身份只能是“依据论文名义 10 Hz 的仿真假设”，不能写入 `timestamp_raw`、不能用于宣称真实采样间隔，也不能验证同步误差。当前无法建立可信的物理时间轴。

### 16 个候选键逐项状态

下表中的 `文件层 PASS` 仅表示对应车辆/UAV PCD 与 YAML 都存在、可解析，PCD 结构一致且 XYZ 有限；每一行的 `物理配对` 均为 BLOCKED。相邻键没有跳过选中发布序列项；这不代表经时钟验证的连续采集时刻。路径前缀为 `code/artifacts/data/v2u4real_r2_01/selected_samples/train/2025-07-18-12-53_2/{1,2}/`，车辆路径在 `1/ouster/` 和 `1/yaml/ouster/`，UAV 路径在 `2/ouster/` 和 `2/yaml/ouster/`，文件名均为 `<frame_key>.pcd/.yaml`。

| frame key | 车辆 PCD bytes | UAV PCD bytes | 文件读取 | 车辆 PCD 字节重复 | 物理配对 |
|---|---:|---:|---|---|---|
| `1752814590.2` | 2,259,016 | 1,128,694 | PASS | 否 | BLOCKED |
| `1752814590.3` | 2,280,584 | 1,126,006 | PASS | 否 | BLOCKED |
| `1752814590.4` | 2,286,040 | 1,132,198 | PASS | 否 | BLOCKED |
| `1752814590.5` | 2,294,888 | 1,129,238 | PASS | 否 | BLOCKED |
| `1752814590.6` | 2,292,216 | 1,131,894 | PASS | 否 | BLOCKED |
| `1752814590.7` | 2,296,504 | 1,132,374 | PASS | **与 `.8` 完全重复** | BLOCKED |
| `1752814590.8` | 2,296,504 | 1,134,054 | PASS | **与 `.7` 完全重复** | BLOCKED |
| `1752814590.9` | 2,293,704 | 1,130,966 | PASS | **与 `4591.0` 完全重复** | BLOCKED |
| `1752814591.0` | 2,293,704 | 1,132,326 | PASS | **与 `4590.9` 完全重复** | BLOCKED |
| `1752814591.1` | 2,286,936 | 1,130,086 | PASS | 否 | BLOCKED |
| `1752814591.2` | 2,310,680 | 1,129,446 | PASS | 否 | BLOCKED |
| `1752814591.3` | 2,306,152 | 1,129,238 | PASS | 否 | BLOCKED |
| `1752814591.4` | 2,301,432 | 1,130,630 | PASS | 否 | BLOCKED |
| `1752814591.5` | 2,307,400 | 1,131,222 | PASS | 否 | BLOCKED |
| `1752814591.6` | 2,304,216 | 1,129,734 | PASS | 否 | BLOCKED |
| `1752814591.7` | 2,306,824 | 1,128,838 | PASS | 否 | BLOCKED |

## B. 坐标系、位姿、标定与标注

| 项目 | 论文 / 代码 / 本地文件证据 | 状态与边界 |
|---|---|---|
| LiDAR 局部轴与世界轴 | 论文 §3.2（PDF 第 4 页）称各 LiDAR 通过预定义变换统一为 x 前、y 左、z 上；世界坐标使用 NED（北、东、下）。 | **仅有论文依据**。本地 PCD 没有声明轴变换矩阵、源/目标轴或是否已经应用该变换。不能因源码提到 NED 就认定点云轴向正确。 |
| 论文描述的空间配准流程 | 论文 §3.1–3.2（PDF 第 4–5 页）称 GPS/IMU 提供初始相对位姿，车辆/UAV 点云对齐使用 RTK 初始化，再以 GICP 精化并人工调整；相机与 LiDAR 用标定板及逐帧 5 对 2D/3D 点求相机外参。 | **仅有论文依据**。未见选中 YAML 或包内独立文件记录本场景跨平台注册矩阵、GICP 输出、人工修正值或应用状态。论文流程不等同于已验证的发布样本标定。 |
| `lidar_pose` 字段形状与数值 | 32 份 YAML 均含有限 `list[6]`。例如首键车辆 `[162.4558620813,-195.5474155993,-3.0822680281,-179.1582303045,128.6537437347,0.4917033974]`；UAV `[146.0642330580825,-178.5034348014285,-51.65063285498502,-179.0519502518403,131.9188901918,0.9285541649206536]`。 | **本地结构 PASS**；数值存在不代表已知道地理原点、参考点、轴向或单位。 |
| `lidar_pose` 的代码消费方式 | `transformation_utils.x_to_world()`（`:13–40`）按 `[x,y,z,roll,yaw,pitch]` 取值；角度默认 `degree=True`，通过 `Rotation.from_euler('ZYX',[yaw,pitch,roll])` 生成旋转，平移原样放入 4×4 矩阵。 | **代码事实**；没有对单位、GPS→局部原点、机体/天线偏置或本地 PCD 的轴约定做转换。作者 issue #1 说明位姿基于 GPS 且顺序如上，但仍不提供当前场景原点与矩阵合同。 |
| `x1_to_x2()` 分支语义 | 同文件 `:43–78`：list/list 实际为 `inv(T_world_x2) @ T_world_x1`；list/matrix 实际为 `x2 @ T_world_x1`；matrix/matrix 为 `inv(x2) @ x1`。但 docstring 把第二个矩阵说明成 `x2->world`，list/matrix 分支却把其直接当作 `world->x2` 使用。 | **代码文档/实现语义冲突，仍未裁决**。调用者必须绑定具体分支及矩阵方向；不能只凭函数名推断。 |
| 官方数据集调用链 | `basedataset.reform_param()`（`:479–521`）用不同 pose 组合生成 `transformation_matrix`、`gt_transformation_matrix`、`spatial_correction_matrix`；这些值是运行时计算，不是选中 YAML 的原始标定字段。`uav_to_single.py:177` 对 UAV 与 ego pose 调用 list/list 分支。 | **代码事实**；可证明代码如何构造矩阵，不能证明真实样本的点云/标注已经正确对齐。 |
| 相机标定 | YAML 中有相机 `intrinsic` 3×3 与 `extrinsic` 4×4（车辆 3 组、UAV 1 组）；值有限且矩阵结构可读。 | **数值结构 PASS**；矩阵源/目标方向及是否已包含局部轴变换未确认。这些不是 LiDAR→机体或跨端配准矩阵。 |
| 3D 标注 | 每帧 YAML 的 `vehicles` 映射包含 object ID 与 `location/center/extent/angle/obj_type/speed` 等字段。论文称框中心 `(x,y,z)`、尺寸 `(l,w,h)`、角度 `(roll,pitch,yaw)`，并说明标注分别在各 LiDAR 局部系生成。 | **字段结构已核验；几何合同部分阻塞**。官方 `uav_to_single.load_annotation_objects()`（`:62–101`）把 `location + center` 组成 object pose，再将 `angle` 作为 `[roll,yaw,pitch]` 传入 `x1_to_x2()`；`create_bbx()` 使用 `extent/2` 构造角点。论文所列角度次序是 roll/pitch/yaw，与该代码消费顺序存在待裁决差异；`location` 与 `center` 的精确数据字典含义也需结合发布定义确认。不能据此验收标注与点云对齐。 |
| 虚拟 RSU | 本场景是车与 UAV 数据；样本无 RSU 实体位置或测量值。 | RSU 只能是下游仿真独立配置。必须在已确认的公共坐标框架下另设 `T_world_to_rsu`；不得写成数据集真实测量。 |

需要按项目冲突规则保留的事项：**Documented Intent**：论文给出的 annotation Euler 顺序为 `(roll,pitch,yaw)`，并声明 LiDAR 局部轴已通过预定义矩阵统一；**Actual Implementation**：官方转换器把 annotation `angle` 原样按 pose 顺序传入，`x_to_world` 解释为 `[roll,yaw,pitch]`，没有从样本字段显式读取轴统一矩阵；**Evidence**：论文 PDF 第 4–5 页、`uav_to_single.py:76–100`、`transformation_utils.py:13–78`、32 份 YAML；**Affected Files**：上述论文/代码/样本；**Conflict**：角度索引、轴矩阵是否预先应用及字段合同不能由当前证据统一；**Status: Awaiting Researcher Decision / official clarification**。本轮没有自行选择分支或改标定值。

## C. 重复 PCD 与样本完整性

`audit_v2u4real_r2_03_v1.py` 重新读取 16 对文件，并对字节 SHA 与复用读取器解码后的 XYZ 数组分别比较。选中车辆 PCD 有两组精确重复：

| 帧键 | 实际文件 | 每个文件大小 | 文件 SHA-256 | 解码后 XYZ |
|---|---|---:|---|---|
| `1752814590.7` / `.8` | `1/ouster/1752814590.7.pcd`、`1/ouster/1752814590.8.pcd` | 2,296,504 bytes | `054aebfbbd28685a665c2cb3c215293f26baa93b198bffa92aba88f8dd5939df`（相同） | `143520×3`，逐元素相同，最大绝对差 `0.0` |
| `1752814590.9` / `1752814591.0` | `1/ouster/1752814590.9.pcd`、`1/ouster/1752814591.0.pcd` | 2,293,704 bytes | `1e07de7bbe3494ba7b237e4136617165f4169eac56c423812ad835639e1eaebd`（相同） | `143345×3`，逐元素相同，最大绝对差 `0.0` |

比较还显示：相应 YAML 字节 SHA 和语义 SHA 均不同，不是重复 YAML。每组前后帧的车辆 `lidar_pose`、`ego_speed` 及 `vehicles` 标注数值均有变化，车辆 ID 集合均为 `1,2,3,4,6`；UAV 侧选中 PCD 没有字节重复。故证据支持“两个不同文件键复用了完全相同的车辆 PCD 字节”，不支持“对应标签/位姿也被重复复制”。原因**仍未知**：可能是源采集/导出中的重复扫描，也可能是文件复用或数据索引/序列问题；仅凭当前文件无法判定，且不得删除、替换或覆盖原始数据。论文 §3.1 称其排除了外部数据丢失/异常，但不能据此推翻本地逐文件哈希结果，也不能单凭重复字节判定发布错误。

16 对的四份文件均存在、PCD/YAML 读取成功；所有 PCD 头部与数据体尺寸匹配，XYZ 有限；YAML 无重复键，`lidar_pose` 为有限 `list[6]`，`ego_speed` 为有限数值。故**文件/读取层 16/16 PASS**，但严格的“完整物理配对”仍为 0；重复的 4 个车辆帧键不能计作 4 次独立点云观测。机器回执记录 `selected_pair_count=16`、`read_errors=0`、`duplicate_pcd_groups=2`、`repeated_yaml_byte_groups=0`、`status=BLOCKED`。

## D. 下游数据合同

| 字段/约束 | 下游允许的含义 | 状态 |
|---|---|---|
| `dataset_version` / `split` / `scene_id` | 精确固定为 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a` / `train` / `2025-07-18-12-53_2`。 | **已核验** |
| 实体唯一键 | `(dataset_version, split, scene_id, agent_id, sensor, frame_key_string)`；本样本 vehicle=`1`、UAV=`2`、sensor=`ouster`。保留原始字符串键，不先转浮点数。 | **文件合同已核验** |
| 跨端候选键 | `(dataset_version, split, scene_id, frame_key_string)` 可索引一组同 stem 候选；另保留两端 agent/sensor/PCD/YAML 各自路径。 | **仅发布文件关联已核验**；不能命名为物理同步 pair。 |
| `frame_key` 与 `timestamp_raw` | `frame_key` 是 YAML 文件名 stem；当前 `timestamp_raw=null`、`timestamp_unit=null`、时间参考原点 `null`。禁止把 stem 转成 Unix/GPS 时间。 | **已核验 / 物理时钟阻塞** |
| 相对仿真时间 | 可用整数 `sim_frame_index` 表达选中键的顺序；若另设 `sim_dt_s=0.1`，必须标为“名义 10 Hz 的接口仿真假设”。 | **仿真假设**；不是真实采集时间或配对差。 |
| PCD 与 YAML 关联 | 同 agent、同原始 stem 的 Ouster PCD/YAML；PCD 字段 `x,y,z,rgb`，实际 PCD 局部轴向和单位尚待确认；不要把 PCD `VIEWPOINT` 当 `lidar_pose`。 | 文件对应关系 **已核验**；物理坐标 **未核验** |
| `lidar_pose` | 原始值必须无损保留；官方代码消费次序 `[x,y,z,roll,yaw,pitch]`、角度默认度、ZYX。 | 字段存在和代码合同 **已核验**；位置单位/世界原点/参考点/轴向及发布值是否按此合同生成 **阻塞** |
| `ego_speed` | 原始有限标量。 | 字段存在 **已核验**；单位未核验，不得直接用于物理回放。 |
| 标注 | 原样保留同帧 YAML `vehicles` 与 object ID；若要变换框，必须先解决 `location+center`、extent 顺序和 angle 顺序。 | 样本内关联 **已核验**；几何解释/对齐 **阻塞** |
| 相机矩阵 | `intrinsic` 3×3、`extrinsic` 4×4。 | 数值结构 **已核验**；方向及与 LiDAR 变换的关系 **未核验** |
| LiDAR 跨端/世界标定 | 当前包没有独立的跨端注册矩阵或其 GICP/人工修正 provenance；由 `lidar_pose` 在运行时计算出来的矩阵必须标成“代码派生”，不能冒充发布文件实测矩阵。 | **BLOCKED** |
| 虚拟 RSU | 由实验配置提供固定公共坐标位置及其局部框架，携带配置来源/单位/坐标系。 | **实验设定**；V2U4Real 未提供真实 RSU 测量 |
| 重复观测标记 | 对两组重复车辆 PCD 保留文件 SHA、帧键和 `duplicate_payload=true` 诊断，不去重、不覆盖。 | 重复范围 **已核验**；原因与后续样本政策 **BLOCKED** |

因此当前 16 对候选文件**可用于有限的解析器、目录映射、原始字段读取和纯接口输入冒烟检查**；不能用于声称真实跨端同步、车辆连续运动采样、点云/标注空间对齐、协同感知融合或物理时间服务评估。文件层连续性 PASS 不得升级成数据集完整验收 PASS。

## 已解决、未解决与 R2-04 准入

**已解决（仅本地选中样本范围）**：固定版本/训练场景和文件键明确；16 组两端 Ouster PCD/YAML 均存在且可读；字段类型、PCD 头/数据体、有限 XYZ 已检查；官方 v1 论文和关键源码版本/哈希可追溯；两个重复 PCD 组的确切文件范围、大小、哈希与解码后相同性已核验。

**仍 BLOCKED**：原始设备/统一时间戳、时间单位和参考原点；本地文件键到物理时刻的映射；16 对各自真实最近帧配对和 ≤20 ms 的逐样本测量；`lidar_pose` 的实际世界原点、单位、参考点与发布时轴变换合同；预定义轴矩阵、RTK/GICP/人工修正是否已落入发布 PCD/位姿及其数值/方向；`x1_to_x2` 矩阵分支文档与实现差异；标签角度排列和 location/center 语义；车辆重复扫描的上游原因。

**R2-04 当前准入：不满足。** 在用真实点云/标注作坐标变换验收前，至少需要：

1. 获得权威数据字典或发布代码证据，明确 PCD 的实际局部轴与长度单位、`lidar_pose` 的物理参考点/单位/公共原点/角度顺序，以及预定义轴变换是否已应用。
2. 获得跨端注册矩阵及源/目标坐标框架、矩阵方向和 GICP/人工修正 provenance；或由研究者明确批准一个只作诊断的、假设显式标记的变换方案，不能冒充已标定事实。
3. 解决 annotation `(roll,pitch,yaw)` 与代码 pose `[roll,yaw,pitch]` 的差异及 `location+center` 的标注合同；否则变换后的框无法作为对齐验收基准。
4. 为涉及时间序列的验收取得文件键到采集时间的权威映射及两端原始时刻；若只进行单帧几何冒烟，报告需明确它不验时间同步。
5. 由数据来源证据或研究者决定如何标注两组重复 PCD 的独立观测资格；不得自行删除、替换或计作独立扫描。

这些条件具备后，R2-04 才能在独立适配层中执行前向/逆向一致性、点云和框空间对齐检查；本报告没有启动这些操作。用户审阅后决定下一任务，本轮在此停止。

## Validation / 实际命令与输出

CPU，工作目录 `D:\shen\PKU\PIJWM`，以项目已有脚本与固定样本运行：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
$env:PYTHONUTF8='1'
python code/scripts/audit_v2u4real_r2_03_v1.py
# {"status": "BLOCKED", "selected_pair_count": 16, "read_errors": 0,
#  "duplicate_pcd_groups": 2, "repeated_yaml_byte_groups": 0, ...}
```

同一脚本直接读取本地 PCD/YAML 并重新生成 `real_data_contract_audit.json`；退出码 2 表示阻塞状态。实际回执为 `status=BLOCKED`、16 个候选键、读取错误 0、重复 PCD 组 2、重复 YAML 字节组 0。`python -m py_compile code/scripts/audit_v2u4real_r2_03_v1.py` 通过。

定向测试命令（隔离 AirFogSim 依赖类）及结果：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts;D:\shen\PKU\PIJWM\code\tests'
$env:PYTHONUTF8='1'
python -m unittest test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests test_v2u4real_interface_v1.GeometryTests -v
# Ran 29 tests — OK
git diff --check
# exit 0
```

另尝试包含 `AirFogSimTests` 的完整 V2U4Real 测试发现；该类初始化因环境缺少 `traci` 而报 `ModuleNotFoundError`，其余 29 个测试方法通过。项目知识索引 `python code/scripts/build_project_knowledge_index_v1.py --check` 未通过，报告 `tracked_file_inventory.csv`、`artifact_catalog.csv`、`registry_summary.json` 三项不同步。工作区同时包含任务前已有的生成文件修改、删除和未跟踪资产；本轮没有在主工作区运行写入式生成器，以免覆盖或混入这些既有内容。另在干净临时工作树尝试生成索引时，因实验注册表引用的本机忽略文件缺失而中止：`code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt`。没有据此生成或暂存不完整索引。当前索引检查仍为未解决项。

Zotero MCP 未在本会话注册；只读本地 REST 请求不可用，本机 `zotero.sqlite` 也返回数据库锁定。本轮没有绕过锁或重取附件。论文页码和 SHA-256 记录来自此前对 Zotero 保存的 arXiv v1 PDF 的直接只读检查；当前会话无法再次经 Zotero 接口独立复核。没有对真实样本做新变换。

## Results / Expected vs Actual / Known Issues

**部分符合预期，总状态 BLOCKED。** R2-02 提出的 16 个候选文件键均通过文件和读取层检查，因此对有限的解析/接口验证可用；这不解决原物理时间、同步、跨端标定问题。重复 PCD 的范围现已明确，成因没有查明；对应位姿和标注并未重复。论文官方采集机制有明确表述，但它不能替代发布样本级时钟与标定证据。

## Git / Next Step

本记录、机器审计回执、审计脚本和必要导航更新已按独立 Conventional Commit `docs(v2u4real): audit R2-03 data contract` 提交；任务开始前的工作区变更没有纳入。当前提交未推送：必需的项目知识索引检查仍报告 `tracked_file_inventory.csv`、`artifact_catalog.csv`、`registry_summary.json` 不同步；在干净临时工作树重建索引又因注册表引用的本机忽略 checkpoint 缺失而中止。依据仓库规则，验证未通过时不推送。提交 hash、当前分支和工作区状态见本轮完成汇报。唯一建议下一步：研究者审阅 R2-04 准入缺口，优先确定时间戳映射、姿态/标注合同和重复 PCD 的处理证据来源；未经新授权不执行真实变换。
