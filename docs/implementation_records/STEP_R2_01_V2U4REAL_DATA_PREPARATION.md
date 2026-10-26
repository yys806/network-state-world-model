# STEP R2-01 — V2U4Real 真实数据与官方资料准备

日期：2026-10-26。**结论：官方训练包可访问，官方时间/坐标原则已找到；最小完整训练样本仍为 BLOCKED。** 已取得 2 个完整真实 PCD 成员，但只有一个文件名时刻，传感器不同且没有 YAML，不能算作 1 对已核验样本，更未达到至少 3 对的目标。未使用人工结果替代真实验收。

## Step Goal / 研究者本次授权

只做当前仓库恢复、官方来源/许可/版本核查、最小训练样本准备与文件读取审计。复用 R1 读取器，不做坐标变换验收、时间同步算法、轨迹接入、AirFogSim 修复、服务实现、模型训练或正式训练数据生成。完成本 Step 不授权下一任务。

## Definition Basis / Initial State

- 依据研究者本次 R2-01 消息及 `AGENTS.md`、项目记忆、协作规则；方向不在本轮重定。工程仍以 PI-JWM 命名，AirFogSim 只作后续参考仿真器。
- 本地 `main` 与 GitHub main 均为 `9bffecfa75e57a12bba2632d77af3e58d85490c0`。R1 工具真实存在：严格 ASCII/binary PCD、YAML/帧编号审计，显式刚体变换及隔离回放；本轮只复用读取/字段审计部分。
- R1 官方版本 `e099a3a746c6092a614e18b74d1f5b2f3c8351e8` 仍是 upstream main，原 9 文件 SHA-256 全部匹配。没有重做 R1 人工回放或无线测试，也没有修改 R1 读取器、正式数据协议、模型/检查点或第三方源码。
- 初始已有 AGENTS、AI_CONTEXT/00、计划/索引修改，旧资料和过程文件删除，以及未跟踪旧脚本。原样保留，不纳入本轮提交。缺失的 task_plan/progress/findings 不重建，过程记于本记录和 `记录/本地计划表.md`。
- 检查 `code/artifacts/data`、`datasets`、相关 tmp/reference 入口及 Hugging Face 缓存中的 V2U 名称，未发现已有真实子集；只找到 R1 官方源码快照。没有扫描其他项目或私有笔记目录。初始状态与共享文件副本保存在 `.tmp/v2u4real_r2_01/`。

## Results / 逐项状态

| 交付/问题 | 状态 | 真实证据与限制 |
|---|---|---|
| 官方训练数据能否访问 | PASS | 官方公开、非 gated；固定版本目录 HTTP 200，两个训练包 Range HTTP 206 |
| 至少 3 个连续 PCD/YAML 配对时刻 | BLOCKED | 完整合格对数为 0；没有车辆 Ouster、两端 YAML，也没有连续序列 |
| 官方同步资料 | PASS | 作者原文 §3.1 与附录 C.3：名义频率、GPS/PTP/PPS、最近时刻配对；不等于样本级实测 |
| 原始时间字段/单位/映射 | BLOCKED | 只见文件名 `1752814590.2` 等；没有 YAML/逐帧时间清单，字段和单位保持 null |
| 官方坐标与标定资料索引 | PASS | 作者原文 §3.1/3.2、作者澄清、固定代码调用链；真实分支的矩阵直接组合语义已静态定位 |
| 样本数值标定、矩阵方向和标签映射 | BLOCKED | 缺实际 YAML、外参、注册矩阵和明确原点；未执行变换验收 |
| UAV PCD 读取 | PASS | 一个真实 Ouster binary PCD，70,532 点，R1 检查 XYZ 有限 |
| 车辆 PCD 读取 | BLOCKED | 包头首先是 M1 ASCII PCD，头声明 78,750 个采样位；含非有限坐标，R1 严格拒绝；不是所需的车辆 Ouster |
| YAML 读取 | BLOCKED / NOT_RUN | 文件可用性 BLOCKED；没有文件，解析执行 NOT_RUN，不能报告字段存在 |
| 完整变换/同步算法/轨迹/信道/服务/训练 | NOT_RUN | 全部超出本轮范围；GPU 与 locked_test 均未使用 |

## A. 官方发布方式、版本、许可与最小下载量

官方代码 README 指向 [VJiaLi/V2U4Real](https://huggingface.co/datasets/VJiaLi/V2U4Real)。本轮锁定数据提交 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`，读取 train 目录 50 条，无下一页链接，共 25 个双端场景包。按两包压缩字节数合计排序，最小场景是 `2025-07-18-12-53_2`：

| 来源 | 官方路径 | 压缩字节数 | 官方 LFS/旧清单期望 SHA-256 |
|---|---|---:|---|
| 车辆 1 | `train/2025-07-18-12-53_2__agent-1.tar.zst` | 1,070,684,502 | `34e03eb7a34c896aab55b44f32797a1e66a62386a5a7c4e58074abfb6741fa75` |
| UAV 2 | `train/2025-07-18-12-53_2__agent-2.tar.zst` | 172,308,593 | `64ae659bca67144c8713d5a689ce310474328a3c4ffd7a6f35948358b72b61f3` |
| 合计 | 一个训练场景、两个来源 | **1,242,993,095（约 1.16 GiB）** | 本轮没有下载整包，因此没有执行整包校验 |

这里只是按下载量选择最小训练场景，不声称它具有最少帧或最适合科研评价。源包大小不能当成特征包大小。

数据页许可元数据为 **CC BY-NC 4.0**；代码仓库为 **MIT**。分别按原来源记录，不混用。作者在 [issue #7 的本人回复](https://github.com/VjiaLi/V2U4Real/issues/7#issuecomment-5031674615) 说明当前发布不含图像，训练数据的元数据/标定/标注在发布范围内，测试元数据/标注不公开。本轮只读训练包，不用验证或测试数据替代。

官方读取器 `pcd_utils.pcd_to_np` 推荐 Open3D；固定 requirements 为 `open3d==0.13.0`，其余含 NumPy、SciPy、PyYAML 等训练依赖。**本轮不安装 CUDA/Torch/spconv**：已有 R1 NumPy/PyYAML 读取器覆盖本次 ASCII/binary 格式，另使用已有 zstandard 解压前缀。R1 的非有限值拒绝是实际触发的严格检查，不是“工具只支持人工数据”，无需放宽读取器或重写一套解析器。

## B. 实际取得了什么

只对每个训练包请求 `Range: bytes=0-1048575`，各读取 1 MiB，共 **2,097,152 字节**压缩前缀；另读取 UAV 包尾部 **32 字节**检查是否存在标准 seekable-zstd 索引。没有取得完整压缩帧，不能验证整包 zstd 校验或官方整包 SHA。HTTPS 来源、固定版本/包身份、局部内容 SHA、成员长度和本地文件 SHA 分开记录。

已取得的原始文件位于：

```text
D:\shen\PKU\PIJWM\code\artifacts\data\v2u4real_r2_01\
├── prefixes\agent-1.first1m.zst.part
├── prefixes\agent-2.first1m.zst.part
└── partial_real\train\2025-07-18-12-53_2\
    ├── 1\m1\1752814590.2.pcd
    └── 2\ouster\1752814590.2.pcd
```

两份 PCD 都是包内完整成员，提取字节数与 tar 声明一致；各自 SHA 不同，不是同文件复制。车辆 M1 为 `FIELDS x y z intensity / TYPE F F F F / SIZE 4 4 4 4 / DATA ascii`，宽 5、高 15,750。ASCII 正文检出 98,058 个 `nan/inf` 无效数字标记（**不是点数**），严格 XYZ 读取 BLOCKED；不自动去除、补零或重新写文件。UAV 为 `FIELDS x y z rgb / DATA binary`，70,532 点；数据体长度与头一致，XYZ 全部有限。

有限性 PASS 只覆盖本轮读取器检查的 XYZ，不为打包 rgb 或强度字段的数值/物理含义背书；这些字段没有送入编码器。

两端 PCD 头 `VIEWPOINT` 均为单位值，但不能据此认定它是世界位姿。UAV 的 XYZ 数值范围已原样落盘，不从其正负推断真实轴、飞行高度或已经变换到了哪个坐标系。

包头还出现下一文件名 `1752814590.3.pcd`，其内容未完整取得。前缀解析到末尾出现 `unexpected end of data` 是本轮主动限量截断，不是原压缩包损坏的证据。本轮没有完整目录清单，无法判断全场景的缺失率、重复率或实际帧间隔。

## C. 小样本获取的具体限制

1. 官方发布粒度是每场景/来源一个 tar.zst，当前目录没有逐帧文件或小样本包入口。
2. HTTP 支持字节 Range，但 zstd 压缩 tar 不能因此直接按 YAML 文件名随机取成员。探测的 UAV 包尾部没有标准 seekable-zstd 页尾；这不证明一切自定义随机读取都不可能，只是未找到官方支持的低成本路径。
3. 作者旧版本 `8908a33d43ca05ddc82d89c1014d8b0906b67bc8` 的 `FILES.tsv/SHA256SUMS` 只提供包级大小/哈希；所选包与当前版本一致，但没有成员路径偏移或标定内容。
4. 车辆包前部是 M1，UAV 包前部是 Ouster。没有取得车辆共同 Ouster 或两端 YAML；不将混合传感器、同名文件当作已核验配对。

因此不继续扩大读取量、不下载最小整包或整个数据集；最小完整样本仍 **BLOCKED**。当前准备脚本硬限制每包前缀最多 1 MiB；服务器若忽略 Range，读取正文前就拒绝。它无法被配置成整包下载器。

### 用户可选择的最小获取方案（NOT_RUN）

优先请作者/已有完整数据持有人提供同一 train 场景 **3 对共同 Ouster PCD+YAML**，优先 16 对，并附原始时间/标定说明。**本轮没有联系作者或向任何人发消息。**

若选择按当前公开发行方式获取，则最低独立包组合为上表两包，压缩总量约 **1.16 GiB**；解压量未实测，不能承诺。以下是研究者确认后可执行的方案，**本轮未执行**，不是后续自动任务：

```powershell
$v2uDir='D:\shen\PKU\PIJWM\code\artifacts\data\v2u4real_r2_01\full_archives'
New-Item -ItemType Directory -Force -Path $v2uDir | Out-Null
$v2uBase='https://huggingface.co/datasets/VJiaLi/V2U4Real/resolve/36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a/train/'
$v2uNames=@('2025-07-18-12-53_2__agent-1.tar.zst','2025-07-18-12-53_2__agent-2.tar.zst')
foreach($v2uName in $v2uNames){
  $v2uTarget=Join-Path $v2uDir $v2uName
  if(!(Test-Path -LiteralPath $v2uTarget)){
    Invoke-WebRequest -Uri ($v2uBase+$v2uName) -OutFile $v2uTarget
  }
  Get-FileHash -Algorithm SHA256 -LiteralPath $v2uTarget
}
```

回读 SHA 必须逐一等于上表；不能仅因下载命令退出就接受。随后只提取同名共同 Ouster 及对应 YAML 的 3–16 个连续 key，保留原字节与相对路径；额外时间表/标定文件按真实目录取得，不能事先编造字段。完整取包及提取流程需后续单独确认，不在本轮自动执行。

## D. 时间说明、坐标/标定资料与未解决问题

详见 [官方时间与坐标资料说明](../project-notes/V2U4REAL_TIME_COORDINATES_R2_01.md)。本轮能够确认官方名义采样、硬件同步、最近帧配对、局部/世界轴向、作者给出的位姿次序，以及真实/仿真源码调用差异。尚不能确认原始时间字段单位/原点、样本级数值矩阵和源/目标、机体/天线外参、标注中心/尺寸/角度字段的实际对应。

R1 混合参数分支的 `12` 差异是通用文档语义下的人工反例。新增静态调用链表明真实路径传入直接 LiDAR→ego 矩阵，仿真路径传入 ego_pose 列表；不能据此把 R1 反例扩大成“真实路径一定有错”。原文与代码各自的边界均保留，本轮没有选分支或变换真实点云。

## E. 样本清单与复用审计

机器清单 `sample_manifest.json` 保留用户要求的全部字段。当前唯一记录的 `frame_id=1752814590.2`，`timestamp_raw/unit=null`，两端 metadata path 为 null，车辆 sensor=m1、UAV sensor=ouster，`qualified_complete_pair=false`。已核验完整合格配对数为 **0**；不生成虚构的三行成功样本清单。

复用 `pi_jwm.v2u4real_interface_v1.read_pcd_xyz` 分别检查两个真实文件，复用 `audit_dataset` 检查其目录。后者如实返回 `No paired train scene with 1/2 yaml/ouster found`，说明 YAML/共同传感器条件未满足。没有复制 R1 的 PCD 解码器、创建替代数据集或放宽有限性要求。

## Files Involved / Changes / Reuse

- 新增 `code/scripts/prepare_v2u4real_r2_01_v1.py`：固定版本训练包、严格限量获取、只提取完整首个 PCD、复用 R1 只读检查、输出诚实的部分清单。默认离线；`--fetch` 最多补取两个 1-MiB 前缀和小型训练目录元数据。
- 配置 `code/scripts/configs/v2u4real_r2_01.json`；测试 `code/tests/test_v2u4real_preparation_v1.py`。未改 R1 读取器、AirFogSim 或模型。
- 官方新增只读源码的版本与哈希在 `code/reference/V2U4Real.R2_SOURCES.json`；完整原文件仍在被忽略的 reference 目录，不提交第三方大资产。
- 原始数据/压缩前缀/论文页图只留在被忽略的 artifacts；Git 只纳入实施/资料说明、清单、来源回执、测试日志、指纹和必要导航。

## Validation / 实际执行与预期

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\tests;D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts;'+$env:PYTHONPATH
python code/scripts/prepare_v2u4real_r2_01_v1.py
# exit 2；官方包可访问 PASS，UAV PCD PASS，车辆 PCD/完整三对/数值合同 BLOCKED
python -m unittest discover -s code/tests -p test_v2u4real_preparation_v1.py -v
# 8 tests OK：训练边界、硬下载限额、不覆盖原文件、路径/链接/截断保护、无资产不伪造 PASS
python -m unittest test_v2u4real_interface_v1.DatasetAuditTests -v
# 8 tests OK；只运行 R1 的读取审计用例，不执行坐标、回放或信道用例
python -m compileall -q code/scripts/prepare_v2u4real_r2_01_v1.py code/tests/test_v2u4real_preparation_v1.py
python code/scripts/build_project_knowledge_index_v1.py
python code/scripts/build_project_knowledge_index_v1.py --check
```

网络检查均固定官方版本、记录 URL/响应/字节/哈希：HF 目录和小清单 HTTP 200、前缀与尾部 HTTP 206；正式 CVF PDF 为 403。原始日志与资料回执在 `code/artifacts/audit/v2u4real_r2_01_20261026/`；精简可共享证据在 `code/artifacts/manifests/v2u4real_r2_01_20261026/`。源码与交付指纹必须和 Git 中的实际字节比对，不以文件名或成功退出代替证据。

## Expected vs Actual / Known Issues

**部分符合预期，总状态 BLOCKED。** 原先“真实数据是否公开可访问”已得到肯定证据，官方同步/坐标原则和调用链资料也补齐；“至少三对真实 PCD/YAML 可直接审计”没有达到。源码的推荐读取依赖与实际 PCD 格式已建立部分对应，但真实元数据与代码的字段级兼容性尚未建立。NaN/Inf 的原因和处理政策也未决定。完整变换、同步算法、轨迹接入、无线修复、特征/计算服务、PI-JWM/正式数据生成均 NOT_RUN。

## Git / Context Consistency Check / Next Step

本任务独立 Conventional Commit，实际 hash 与 push 状态见完成汇报及 Git 历史；不纳入起始未提交改动。共享文档只新增本轮段落，必要 AI_CONTEXT 更新只说明当前真实证据与阻塞，不新增 AI 自行作出的研究决策。索引应同时对工作区和隔离的暂存视图检查，避免把原有删除/修改混入提交；详细收尾结果见日志和完成汇报。

唯一建议下一动作：研究者审阅**最小场景获取方案（1.16 GiB）或作者提供的小样本方式**并决定获取路径。未经新授权，不继续扩大下载、不自动进入真实同步/变换或其他任务。
