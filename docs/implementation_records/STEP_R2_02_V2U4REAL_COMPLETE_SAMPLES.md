# STEP R2-02 — V2U4Real 最小完整真实样本获取与验收

日期：2026-10-26。**部分完成，总状态 BLOCKED：两个完整官方训练包 SHA 校验通过；取得 16 个发布文件序列中连续的共同 Ouster/YAML 时刻，64 份文件读取与结构检查通过；物理同步和关键标定语义仍未达到最终验收条件，合格完整配对数为 0。**

## Step Goal / Definition Basis

执行研究者本次 R2-02 的独立数据准备授权，固定 train 场景 `2025-07-18-12-53_2`、数据版本 `36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a`。只获取两个指定完整包，流式检查目录并筛选最多 16 个候选共同 key 的 Ouster/YAML，再复用 R1 严格读取器。真实文件取得、字段结构、物理语义三个层次分别验收。

本轮没有改变研究方案、模型、双图/RSSM/Planner/Scorer、Loss、协议或 checkpoint。没有时间同步算法、真实坐标变换、AirFogSim/无线/服务接入、训练或正式训练数据生成；GPU、validation/test/locked_test 均未使用。RSU 未设定，没有联系作者或第三方。

依据：当前 `AGENTS.md`、`docs/PROJECT_MEMORY_AND_SOURCES.md`、协作规则、AI_CONTEXT 00/02/03、R1/R2-01 实施记录、已有脚本与真实源码。AI_CONTEXT 仅作导航，不作验收依据。

## Initial State

- 本地 `main` 为 `82cad06e4646228201a844bf0e07176a3c8644b1`；`git ls-remote origin refs/heads/main` 实际回读相同。普通沙箱网络曾出现 schannel/套接字权限错误，在已授权网络访问下完成；既有下载前缀与续传文件保留复用。
- R2-01 的原始清单只有一个混合传感器文件 key：车辆 M1 含非有限 XYZ、UAV Ouster 可读，无 YAML。其原始文件/报告/清单本轮全部保持不变；不能把 M1 异常外推为车辆 Ouster 不可用。
- 官方代码仍采用 R1/R2-01 固定提交 `e099a3a746c6092a614e18b74d1f5b2f3c8351e8`；13 份本地源码/依赖快照 SHA 全部匹配，无第三方代码修改。
- 起始有 AGENTS、AI_CONTEXT/00、计划、CHANGELOG/PROJECT_INDEX、两份生成索引修改，旧文献/论文/过程文件删除和多份未跟踪脚本。初始状态及共享文件副本在 `.tmp/v2u4real_r2_02/`。本轮不清理、不恢复这些删除、不提交无关修改；缺失的 task_plan/progress/findings 按项目记忆入口规则继续记于本地计划表。

## A. 完整包获取与来源

两包保存在被 Git 忽略的 `code/artifacts/data/v2u4real_r2_01/full_archives/`。官方 URL 均为 `https://huggingface.co/datasets/VJiaLi/V2U4Real/resolve/36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a/train/<下表文件名>`；完整 URL、各 Range 请求及最终来源主机、时间、旧网络错误与整包核验见 `archive_agent_1.json`、`archive_agent_2.json`。

| 来源 / 文件 | 实际大小 bytes | 实际 SHA-256（与官方期望完全一致） | 状态 |
|---|---:|---|---|
| vehicle / `2025-07-18-12-53_2__agent-1.tar.zst` | 1,070,684,502 | `34e03eb7a34c896aab55b44f32797a1e66a62386a5a7c4e58074abfb6741fa75` | PASS |
| UAV / `2025-07-18-12-53_2__agent-2.tar.zst` | 172,308,593 | `64ae659bca67144c8713d5a689ce310474328a3c4ffd7a6f35948358b72b61f3` | PASS |

合计 1,242,993,095 bytes，约 1.16 GiB。仅指定一个场景，不是全数据集。复用每包原有 1 MiB 前缀、按精确 HTTP 206 Range 续传；最终以完整大小和 SHA 为准。脚本拒绝身份变化、服务器忽略 Range、空间不足、哈希错误和覆盖不同原文件。数据许可沿用 R2-01 官方 CC BY-NC 4.0 证据，源码 MIT 许可不替代数据许可。

## B. 包目录与最小提取

| 包 | 完整成员数 | 实际文件结构 | 异常 |
|---|---:|---|---|
| agent 1 | 1,314 | `train/<scene>/1/{m1,ouster,ruby,yaml/m1,yaml/ouster,yaml/ruby}/`，每类 219 文件 | 无重名成员、链接或额外独立时间/标定文件 |
| agent 2 | 438 | `train/<scene>/2/{ouster,yaml/ouster}/`，每类 219 文件 | 同上 |

`members_agent_*.json` 是完整包成员名/大小/类型清单。只读取压缩流，不把所有成员落盘；只写出 `selected_samples/train/<scene>/<agent>/ouster/<key>.pcd` 与 `yaml/ouster/<key>.yaml`。

选择四个已发布序列共同 key 的最早 16 项：`1752814590.2` 至 `1752814591.7`，各序列索引 0–15，无发布文件跳项。完整 Ouster/YAML key 集合均为 219 项，从 `1752814590.2` 至 `1752814612.0`，每一数值差均为 `0.1`；**单位未核验，不能称为实际秒间隔**。

实际提取 64 份原始文件、54,876,775 bytes，保留原名、路径与字节。`extraction.json` 记录逐文件包成员、大小、SHA；`selection.json` 记录四序列及索引。没有提取 M1/Ruby、其他时刻或其他数据划分，也没有修复/过滤原始数据。

## C. 实际读取与异常

使用未修改的 `pi_jwm.v2u4real_interface_v1.audit_dataset`，内部复用 `read_pcd_xyz` 与拒绝重复键的 `UniqueKeyLoader`。新增流程仅补充元数据形状/有限数值、原字节身份、跨文件重复和资格分层记录，没有另写 PCD 解码器或降低校验要求。

- 32 份 Ouster PCD 均为 binary、`FIELDS x y z rgb`，头部 POINTS/WIDTH/HEIGHT 与数据体长度一致，XYZ 全有限。车辆每帧 141,177–144,406 点，UAV 70,364–70,867 点。打包 rgb 的物理含义不在本轮有限性验收内。
- 32 份 YAML 解析成功，无重复键；`lidar_pose` 为有限 float list[6]，`ego_speed` 为有限 float；相机外参 4×4、内参 3×3；标注 location/center/extent/angle 为有限 list[3]，extent 为正。
- 每个选中 key 两端 PCD/YAML 均存在，没有缺文件或包成员损坏证据。文件存在/读取 PASS 不代表同步/空间对齐 PASS。
- **两组车辆 PCD 重复内容**：`1752814590.7` 与 `.8`；`1752814590.9` 与 `1752814591.0`。各组 SHA 完全相同但对应 YAML 的 SHA 不同；UAV 选中 PCD 无字节重复。原因未知，不删除、不改名、不把文件时刻数等同独立观测数。
- R1 `audit_dataset` 总状态始终保留 BLOCKED，物理时间与坐标未核验；32 个 `read_status` 全 PASS。本轮资格计数器不会将可读文件、同名 key、重复 PCD 或未核验语义计作合格物理配对。

## D. 时间、位姿、标定与最终状态

详见 [字段级时间/标定说明](../project-notes/V2U4REAL_TIME_CALIBRATION_R2_02.md)，其中有实际矩阵、位姿与官方调用位置。32 份 YAML 没有 time/stamp 字段；包中没有独立时间表。已有位姿和相机矩阵，但未见独立 LiDAR→机体或跨端注册修正矩阵；官方 `transformation_matrix` 等字段由读取器运行时根据位姿生成，并非本次原始 YAML 字段。

| 最终验收项 | 状态 | 证据 / 仍缺内容 |
|---|---|---|
| 两个官方整包完整性 | PASS | 实际大小与 SHA 均精确匹配 |
| 至少 3 个、优先 16 个发布序列连续共同 Ouster/YAML key | PASS（文件层） | 16 个候选，64 份文件，四序列索引连续 |
| PCD/YAML 读取和结构 | PASS | 32 次端侧 PCD/YAML 读取 + 数值字段审计 |
| 数值位姿、相机内外参取得 | PASS（字段层） | 全部实际数值和维度保存在审计中 |
| 物理采集时间、单位、原点、实际周期与两端偏差 | BLOCKED | 没有字段到采集时钟的映射；名义 10 Hz/最近帧规则不够 |
| 样本绑定的外参方向、预定义轴变换与注册修正来源 | BLOCKED | 论文过程、代码消费合同、原始字段仍需逐项绑定 |
| 不同 key 下车辆 PCD 重复原因 | BLOCKED | 两组字节重复，尚无官方解释或采集日志 |
| 合格完整物理配对数 ≥3 | BLOCKED | **0**；16 是候选可读文件对数，不是验收通过数 |
| 同步算法/真实变换/回放/无线/服务/训练 | NOT_RUN | 均超出本轮范围 |

## Files Involved / Changes / Reuse

新增 `code/scripts/prepare_v2u4real_r2_02_v1.py`、`code/tests/test_v2u4real_full_sample_v1.py`、本记录与字段说明、新版本清单及来源/测试证据。复用 `configs/v2u4real_r2_01.json` 的固定身份、R2-01 官方来源和 JSON 输出、R1 PCD/YAML 审计；R2-01 原清单不改写。`.gitattributes` 将本轮回执固定为 LF 换行，避免 Windows 检出改变审计字节。

更新本地计划/进展、实施与实验/结果导航、受影响 AI_CONTEXT 00/03/04/05/07/08、实验注册表/问答路由/CHANGELOG 与生成索引；AI_CONTEXT 01/02/06 和 PIJWM 主文档的理论边界未变，不将工程判断记为研究者决策。所有原始包/PCD/YAML 留在忽略目录，只提交脚本、审计摘要和清单。

## Validation / 实际命令与输出

在 `D:\shen\PKU\PIJWM`，CPU，设置 `PYTHONUTF8=1`，`PYTHONPATH=code/src;code/scripts;code/tests` 的绝对路径后执行：

```powershell
python code/scripts/prepare_v2u4real_r2_02_v1.py download
# 两个完整包通过大小/SHA 验证；已有完整文件时不重复下载，exit 0
python code/scripts/prepare_v2u4real_r2_02_v1.py inventory
# vehicle 1314 / UAV 438 members，duplicate_names=[]，exit 0
python code/scripts/prepare_v2u4real_r2_02_v1.py extract
# extracted 64 unchanged files for 16 candidate keys，exit 0
python code/scripts/prepare_v2u4real_r2_02_v1.py audit
# candidate_file_pairs=16, readable_file_pairs=16, qualified_complete_pairs=0,
# extracted_file_count=64, duplicate_payload_groups=2, status=BLOCKED，exit 2
python -m unittest test_v2u4real_full_sample_v1 test_v2u4real_preparation_v1 test_v2u4real_interface_v1.DatasetAuditTests -v
# 25 tests OK；只跑读取/获取相关测试，人工 fixture 不作为真实验收证据
python -m compileall -q code/scripts/prepare_v2u4real_r2_02_v1.py code/tests/test_v2u4real_full_sample_v1.py
# exit 0
python code/scripts/build_project_knowledge_index_v1.py
python code/scripts/build_project_knowledge_index_v1.py --check
```

精确执行日志在新清单目录的 `verify_archives.txt/real_sample_audit.txt/tests.txt/compile.txt`；目录与提取的机器回执为 `members_agent_*.json/extraction.json`。测试日志中 R2-01 的“无本地前缀”输出来自隔离临时目录的缺数据拒绝用例，不是当前真实样本状态。生成索引另对隔离暂存视图复核，以防工作区已有删除/修改混入提交；结果见 `index_validation.json`。

## Expected vs Actual / Known Issues

**部分符合预期。** 原有“缺车辆 Ouster、无连续 YAML、无数值相机标定”阻塞已经解除；16 个完整候选文件对可直接被严格读取器读取。最终目标要求物理时钟与标定语义可核验，这些条件尚未满足，不能将本任务标记 PASS。发现重复 PCD 是新增事实；未决定处理政策。现有原始文件足够进行后续只读证据核对，但不自动授权真实变换或轨迹接入。

## Git / Context Consistency Check / Next Step

本轮采用独立 Conventional Commit `feat(data): acquire and audit pinned V2U4Real scene samples`，以 Git 历史和完成汇报记录实际 hash/push。只暂存本轮新增文件及共享文档中的本轮段落；不纳入 AGENTS、原有删除、无关脚本或起始索引差异。工作区与提交视图分别核验，原有修改保留。

唯一建议下一动作：研究者审阅本轮**时间字段映射、标定方向和重复 PCD 的证据缺口清单**，决定下一轮最小核验授权。本任务结束即停止，不下载其他场景、不联系作者、不进入下一实验。
