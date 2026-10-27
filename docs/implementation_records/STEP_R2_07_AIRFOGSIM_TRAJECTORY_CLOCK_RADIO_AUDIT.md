# STEP R2-07 — AirFogSim 轨迹、时钟与无线服务接口核验

日期：2026-10-27。**审计完成并保留阻塞：合成轨迹、步后观测和有效资源块服务读取的软件接口通过；原生内部时刻错位、无线公式和物理模型可信度未通过。没有真实数据回放或无线交付通过的结论。**

## Step Goal / Definition Basis

执行最新版《新场景下训练前任务书》任务2及本轮用户授权。车辆和UAV均为感知源、UAV采用预设轨迹、单个逻辑RSU；第一阶段LiDAR，不优化计算资源、DAG或Route。AirFogSim只是参考仿真工具，研究框架是PI-JWM。

只读依据：

| 文件 | SHA-256 | 本轮使用范围 |
|---|---|---|
| 私有笔记 PIJWM/新场景下训练前任务书.md | 9df6fe58b9fb563f87bded2744cfe2243f67b7f0c57d0dd1e7a24bdbdefafa40 | 任务2；任务3/4的前置条件和停止边界 |
| 私有笔记 组会笔记/1028组会.md | 5399a4b4519b7b5b1b6427a057e1837e622208a614fb9ca33c14163906fa3144 | 车辆/UAV感知、预设UAV轨迹、单逻辑RSU和固定资源范围 |

私有笔记根为 D:/shen/OB/科研/PKU/PIJWM，只读、不复制全文。最新版任务书优先于旧日期的导航摘要。R1、R2-05B、R2-06源文件和历史回执直接继承，不以GitHub是否已包含本地提交判断其存在。

## Initial State / 当前门

- 基线分支main，HEAD=66faa8fdfab517adf8b1a68bc14543027205fb41；相对本地origin/main有8个未推送提交。未刷新远端，本轮不依据缓存分支声称远端最新状态。
- 任务前40项修改/删除/未跟踪路径及哈希见本轮 baseline.json。包含AGENTS、状态/变更导航、生成索引的原有修改，以及旧论文/文献和三个过程文件的删除。均不恢复、不清理、不夹带提交。
- 继承R1：原生real模式新车选择滞后、UAV原生setter是速度积分、实体标签落后一帧；V2I高度、SINR与中断、快衰落问题未闭合。
- 继承R2-05B：两个payload可进入应用等待队列，无线交付BLOCKED_NOT_ATTEMPTED；原生finished字段不作交付证据。
- 继承R2-06：固定train包RGB=0；LiDAR物理时间/物理配准未建立。本轮真实数据读取、PCD/YAML解码、五帧/B标签/完整Mock审计均为NOT_RUN。
- 既有知识索引5项不一致仍是推送门；本轮只运行只读检查，不重建生成索引绕门。

## Files Involved / Changes / Reuse

1. code/src/pi_jwm/v2u4real_interface_v1.py：为既有绝对轨迹工厂增加显式dynamic_vehicle_ids选项；默认仍拒绝动态车辆，保留R1合同。输入按给定秒数逐帧安装，不插值、不外推、不猜真实采样时间。
2. code/src/pi_jwm/airfogsim_service_interface_v1.py：新增步后观测和单时隙服务读取；检查实体/网格/时钟/数组索引/RB掩码，并把原生落后时间标签原样保留。
3. code/scripts/audit_airfogsim_trajectory_clock_radio_r2_07.py：复用R1 prepare_config，实际运行AirFogSim CPU；两次各8步合成动态轨迹，以及独立公式诊断；不用虚构仿真器或替代传播回调。
4. code/tests/test_airfogsim_service_interface_r2_07.py：8项针对性单元测试。R1两个轨迹合同测试、R2-05B两个应用入队测试按需回归。
5. 机器结果、日志、报告、必要AI_CONTEXT/权威进度/索引注册增量。无研究决策写入AI_CONTEXT/06_DECISIONS，无第三方源码修改。

## 1. 轨迹、实体与时钟证据

所有位置均为人为设定的合成米制地图位置；时间是明确给定的仿真秒，不是V2U4Real文件名推导出的真实时间。

| 检查 | 结果 | 证据和限制 |
|---|---|---|
| 动态车辆进入/退出 | PASS | 步后0.1至0.8秒车辆数量1→3→2→1→2→0→1→1；检验ID、顺序索引、速度/加速度/角度及实体位置；退出ID不留在网格，零车辆时V2I形状为[0,1,50] |
| UAV逐时隙绝对位置 | PASS | x=349,350,349,351,350,349,350,351；z=50.5,51,50,50.5,51,50,50.5,51；有反向移动，逐项等于输入快照，不通过速度积分近似 |
| step前后与地图网格 | PASS | 每步记录before_step、after_step；独立用地图原点和网格宽度计算格号；跨x=350边界；RSU固定[300,300,25] |
| 步后环境/TrafficManager时钟 | PASS | 0.1秒等间隔配置下二者均等于快照时刻；两次同seed的完整轨迹和无线数值逐项一致 |
| 自有观测采样时刻 | PASS | observe_post_step同时核对原生实体位置和TrafficManager快照，给出position_time_s=traffic_time_s=env_time_s；不同步、错误成员、位置、格号或不同时间步长拒绝 |
| 原生实体_last_updated_time | BLOCKED | 原生标签仍是t-0.1；自有观测保留native_last_updated_time_s与lag字段，绝不覆盖原生字段后宣称修复 |
| 原生步内无线使用时刻 | BLOCKED | 实际记录_updateWirelessCommunication入口：env时间t、TrafficManager时间t+0.1、位置来自t+0.1。原生先推进轨迹再按旧env时间更新实体并做无线；步后对齐并不修复步内时序 |
| 真实V2U4Real轨迹回放 | BLOCKED | 真实物理时间、坐标/地图绑定仍无充分证据；本轮不使用真实数据 |
| 控制/越界/非均匀时刻/耗尽 | PASS | 拒绝机动控制、600米上边界、非均匀时刻和耗尽后的外推；reset恢复零时刻；旧默认合同仍拒绝动态车辆 |
| 不同traffic/simulation步长 | NOT_RUN | 新观测接口明确拒绝，不能将等步长结果推广到多子时隙 |

源码依据：airfogsim_env.py step先_updateTraffics，内部循环后才增加simulation_time；_updateTraffics先traffic_manager.stepSimulation，再用旧simulation_time调用Vehicle/UAV.update。Vehicle.update/UAV.update直接赋值位置与_last_updated_time。R1原生real模式反例和原生UAV setter反例继承，不重新跑整套。

时间零点的边界：安装TrafficManager零时刻快照时，原生env的车辆实体尚未同步；首次env.step后才同步至0.1秒。本轮没有声称原生零时刻可直接形成完整观测。任务3须明确初始快照/首个可观测时刻规则。

机器证据：synthetic_snapshots.json、trajectory_radio_trace.json、runtime_config.json和result.json；每一步包含原生无线入口的时间和位置。随机seed=20261027，实际解释器D:/miniconda/envs/airfogsim/python.exe，useCUPY=False（NumPy CPU）。

runtime_config的tripinfo CSV只用于原生构造启动；实际逐帧位置来自安装的独立TrafficManager及synthetic_snapshots.json，不能把启动CSV当作完整轨迹输入。

## 2. V2I/U2I几何与无线资源

V2I是车辆到RSU，U2I是UAV到RSU；RB（资源块）是可分配的一小段无线频带。原生50个RB、每RB2 MHz，频率2.4至2.498 GHz。数组布局为[发送节点索引, RSU索引, RB索引]，不是节点ID直接作下标。

| 检查 | 结果 | 本轮直接证据 |
|---|---|---|
| V2I距离实际实现 | PASS（行为核实） | pathloss_callback以位置所有列计算欧氏距离，三维输入先得到sqrt(d_xy²+Δz²)，再加默认(25-1.5)²。源码变量名d_2d不改变实际三维计算 |
| V2I高度物理解释 | BLOCKED | 节点z与默认天线高度关系未统一，已有高度可能重复计入；all_channels保存h_bs/h_ut但调用时未传入，实际落入callback默认值 |
| U2I距离实际实现 | PASS（行为核实） | free_space用完整三维距离，未额外加一次默认高度；整体z平移100米损耗不变 |
| 已知几何与路径损耗变化 | PASS（行为核实） | 水平50/100/200/400米，V2I为74.824980/80.135909/86.365456/93.217726 dB；UAV z=60、RSU z=25时U2I为75.755488/80.546088/86.195831/92.118549 dB。独立重建原生分段公式一致；不等于物理模型验收 |
| 实体位置到路径损耗调用 | PASS | 真实env每步PathLoss数组等于以当前实体位置调用当前native callback的结果；不是手写覆盖PathLoss |
| 动态节点索引和RB激活 | PASS | 车辆数量0–3时数组维度始终对应当前ID列表；不同车RB0/2/4，UAV RB1/7；下一步无任务时激活掩码归零 |
| 激活掩码与速率读取 | PASS | 自有接口按当前ID解析tx/rx，严格比对预期RB与原生布尔掩码；拒绝重复/越界RB、错误端点、错误维度和不一致分配 |
| 未激活RB服务屏蔽 | PASS | 只对掩码为True的速率求和；合成[2,2,8] Mbps、掩码[False,False,True]只算8 Mbps，全未激活为0。该正速率反例是算术保护测试，不冒充本轮默认信道实测 |
| 速率单位与换算 | PASS | 实际速率逐项等于2 MHz×log2(1+10^(native_SINR_dB/10))并应用原生outage掩码，单位Mbps；与原生getter一致；bits=Σactive rate_Mbps×10^6×Δt，bytes=bits/8 |
| 无分配/零速率/小于1字节 | PASS | 空分配实际env服务0；人工正速率无激活仍0；0速率0服务；1e-6 Mbps×0.5秒=0.0625字节，不擅自取整或累计 |
| 包完成、完整交付与截止时间 | NOT_RUN | 无offload任务、无跨时隙累计、无数据包完成、无期限判定；这些属于任务4 |

例如：第一步车辆RB0原生速率29.783152 Mbps，乘假设0.1秒得372289.405419字节；UAV RB1/7合计65.246546 Mbps，得815581.825970字节。**这些是当前公式输出下的条件性单时隙容量，不是已发送/已收到的字节，也不是可信实际无线吞吐。**

采样语义：先完成env.step，再明确激活诊断profiles并computeRate；采样时刻为步后t。输出是此瞬时速率乘一个假设的Δt，不能冒称已完成的[t-Δt,t]区间服务，更不能由此推断原生步内无线时序正确。没有创建/执行真实传输任务。接口要求同一次采样的速率/掩码，尚无多时隙调度或防陈旧状态的生命周期令牌。

原生任务单位尚未闭合：airfogsim_env.py以sum(rate_Mbps)×simulation_interval直接作为trans_data传给Task.transmit_to_Node，数值单位是Mbit；Task._task_size声明只有“size”，没有明确byte/MB单位。R2-05B的perception_payload_bytes是独立应用属性，原生mission任务size=0，未与真实包长度连接。例如8 Mbps×0.1秒在原生路径是0.8 Mbit，在本轮自有接口是100000 byte，不能把两者数值直接互换。native_task_unit_boundary.json记录源码指纹和算术关系；这一任务4合同缺口为BLOCKED，不在本轮补写累计/完成逻辑，也不据无明文单位直接认定固定8倍软件错误。

## 3. 数学语义、软件问题与研究者待决项

所有相关AirFogSim源码保持原样。PASS只表示观察到并复现了实际行为；下列物理接受状态均为BLOCKED。

| 问题与源码依据 | 数学/软件结论 | 候选处理及影响（未采纳） |
|---|---|---|
| channel_manager_cp.py计算10log10(S/I)，再把小于1e-9的dB数置为1e-9 | 负dB本来有合法含义；该截断将线性SINR最小值抬至约1，而非接近0。无outage时每个2MHz RB产生约2.000000000332 Mbps下限 | 可讨论在线性域作数值保护、只对有效RB计算；会改变低SINR服务结果，须统一无线合同后另行授权修改vendor |
| outage_callback.py返回1-exp(-threshold/snr)，调用者传入dB SINR | 比值形式需要同域的非负比率，现状单位不一致。20dB、threshold数值10时，直接代入得0.393469，先将SINR转100得0.095163；10dB处二者碰巧都是10不能据此通过 | 选项包括明确线性阈值/平均SINR的中断模型，或在瞬时衰落下直接判门限。是否同时再抽一次Rayleigh中断、阈值究竟用dB还是线性，都由研究者确定；不能只改一行宣称物理正确 |
| V2I UMa将完整三维位置差再加默认高度 | 源码的d_2d命名与输入维度冲突；100米水平、z=0/25先得到103.077641米，再得到105.722514米。重复高度/天线位置语义未闭合 | 候选A：明确水平距离、地面/天线高度后按模型传参；候选B：统一三维天线位置并选择匹配的公式。两者均涉及模型合同，不自动选择 |
| fading_callback.py先抽r，再求p(r)，最后10log10(p(r))，50RB广播 | 可复现的是“样本处概率密度的对数”，不等于标准瑞利幅度/功率增益抽样；所有RB完全相同。std文档声称dB，却用于高斯幅度参数；omega还由同批链路均值确定 | 候选：规定归一化复高斯/功率增益及方差、RB/时域相关性；或明确将当前实现限制为软件占位。不能假设独立RB或当前PDF就是可信衰落 |
| env步内使用t+Δt位置、t时钟；实体标签也是t | 明确的软件时序/标签不一致。外部步后观测已隔离标签误读，原生内部没有被修复 | 后续可讨论按区间起点/终点采样，或项目环境适配层统一内部时刻；涉及任务到达/服务先后次序，须在任务3冻结时明确，不能偷偷改时钟 |
| 原生UAV setter只接受速度/角度等 | 这是原生接口能力限制，不能把position额外键当成功能 | 自有逐帧绝对轨迹已解决本轮合成输入需求，不声称修复了原生setter |

本轮没有改中断阈值来制造正速率，没有替换callback，没有修改发射功率、传播参数或公式。R1既有threshold=0反例仅作为历史证据；本轮的默认threshold始终为10。四个已知几何点是源码与数学行为诊断，不是无线覆盖、精度或物理校准实验。单逻辑RSU存在不代表全部区域均被实际无线覆盖。

## 4. Validation / 实际命令与结果

完整命令和退出码另见 validation.json/reproduce_commands.md，日志均在本轮审计目录。

| 验证 | 结果 | 范围 |
|---|---|---|
| 新增专项unittest | PASS，8 tests | 动态合同、默认兼容、reset/耗尽/控制拒绝、观测错误拒绝、RB/单位/边界 |
| 实际AirFogSim探针 | PASS，exit0 | 两次各8步，完整结果相等；动态车辆含归零；保留原生错位和物理BLOCKED |
| R1针对性回归 | PASS，2 tests | 旧轨迹非法时间/边界及reset/耗尽/控制；仅将测试临时目录移到本轮自有目录，未修改旧测试文件 |
| R2-05B针对性回归 | PASS，2 tests | payload ID重复一致与应用入队；不重读真实帧、不重新测试完整Mock链 |
| 编译检查 | PASS，4份源码 | 对本轮4份Python源码采用内存compile，不写pycache；不重复已知旧compileall缓存ACL失败 |
| Context Consistency Check / 注册表结构 | PASS，9份上下文 | 逐项核对AI_CONTEXT 00–08、权威记录和证据边界；新实验不记正式性能 |
| 知识索引只读--check | BLOCKED，exit1，原5项不同步 | 若仍失败则本地提交、不推送；不运行覆盖式重建 |

保留的实际失败：

1. 首次单元启动采用模块名但未能导入测试路径，0个有效测试，exit1。改用明确discover目录并使测试自行设置源码路径后8项通过。attempt1_unit_tests.log保留，不计首次通过。
2. 首次实际探针轨迹重复比较已通过，但诊断器错误地把400米也当作UMa第一分段；公式断言失败，exit1。修复的是自有探针的分段预期，不是传播模型。attempt1_result.json及日志保留。
3. 第二次启动Windows默认GBK不能输出vendor的扳手字符，连带结果打印也失败，exit1。自有脚本显式设置UTF-8流后运行通过；attempt2_result.json及日志保留。未安装包或修改环境。

保护：机器结果保留13份相关vendor源码及64份R1/R2-05B/R2-06历史回执的运行前后哈希；均须unchanged。原始数据本轮完全未读写，未下载、未访问非train拆分、未启GPU/训练/检测器；没有通过重新读取原始数据来制造物理证据。基线40项工作区内容另由收尾保全检查逐项核对。

## 5. Results / Expected vs Actual / 当前准入

**部分符合预期。** 软件侧需要的动态轨迹、清晰步后采样和有效服务量读取已具备；原生内部时间/物理模型仍有真实阻塞，不能签发完整无线服务验收。

| 后续任务 | 本轮已提供的前置条件 | 仍缺什么；本轮状态 |
|---|---|---|
| 任务3：场景合同冻结与可扩展接口 | 车辆动态进出、UAV预设绝对快照、单RSU、节点/格号索引、明确合成秒和步后观测；R2-06数据边界继承 | 真实/合成节点的组合、零时刻规则、任务到达/截止时间、特征包粒度和内部时间合同仍待冻结；无线覆盖不能由逻辑服务推断。任务3本轮NOT_RUN，不自动冻结 |
| 任务4：多时隙传输与生命周期 | 继承payload/TaskManager入队；新增有效RB筛选、Mbps→bit/byte换算、零服务/部分字节/错误分配的检查。足以为“先用确定性速率验证”提供算术接口依据 | 跨时隙累计、部分/完整交付、多轮调度、超时、请求/包/计算/检测分层状态尚未实现；原生Mbit数值与payload字节长度尚未绑定，AirFogSim物理服务和步内时序未接受。任务4本轮NOT_RUN，不把本轮容量当交付 |
| 正式训练/真实感知基线 | 无新增准入 | 仍受R2-05A/R2-06数据、标签/评价、物理时间和研究决策等门限制；本轮未执行 |

当前可复用：合成CPU接口测试、步后位置观测、当前公式下带掩码的瞬时容量诊断。当前禁止推断：真实数据同步、真实轨迹回放、可信无线吞吐/覆盖、真实包交付、真实检测精度或PI-JWM方法已完成。

## 6. Context Consistency / 记录更新

只更新受影响的状态、数据流、模块、实验、已知问题和变更导航；研究方向、核心架构和研究决策不变。权威计划/推进记录将本轮标为有限接口审计完成并停止。起始已删除的task_plan/progress/findings不恢复，在本地计划表代记本轮计划、进度和发现。注册表新增独立软件审计条目，metrics/checkpoint/protocol等不适用字段明确null并给原因；不生成正式检测/无线性能数值。

## 7. Git

本轮从40项脏工作区中隔离提交。共享脏文件只暂存本轮新增段落与HEAD内容，不暂存已有修改；其他改动明确按文件加入。具体提交SHA以最终git log/回报为准，机器证据记录执行时基线HEAD和源码SHA，避免在提交内容内自引用提交哈希。

暂存时实际发现根级text/eol规则会把本轮JSON回执的CRLF转换为LF，导致source_identity的原字节SHA与暂存内容不符。仅在本轮新审计目录添加局部.gitattributes（-text），原字节保持不变，再逐文件比对暂存blob与本地原字节；不修改全局Git配置、根属性或历史证据。

知识索引门未通过时：保留独立本地提交，不推送，不强推；不覆盖原有生成索引。本轮没有自动启动任务3、任务4或正式训练。

## Next Step / 唯一建议

研究者审阅本报告的待决表，明确无线公式与步内时间的统一合同；在裁决和另行授权前，仅复用本轮已通过的软件接口证据，不继续扩展实验。
