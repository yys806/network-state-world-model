# Ubuntu WSL 环境与存储记录

任务：PIJWM-WSL-CLEANUP-001；核验日期：2026-10-24。

结论：完成授权范围内的安全清理、轻量环境核验和记忆更新。20 GiB 目标未达成，主因是非 PIJWM 研究目录和环境；未扩大删除范围。

## 1. 实测环境与统计边界

- Ubuntu 24.04.3 LTS，WSL 2；WSL 2.6.3.0，Linux 6.6.87.2-microsoft-standard-WSL2；Windows 10.0.26100.4946。仅检测到 Ubuntu 一个注册发行版。
- 使用 du -x 和 statvfs 统计 Linux 根文件系统；不计 /mnt/c、/mnt/d，也不重复计 WSLg 的挂载视图。du 在同次遍历中按工具默认规则避免硬链接重复计数；子目录表为层级关系，不可全部相加。
- 字节为原始单位，GiB=2^30 字节，GB=10^9 字节。初次容量快照与清理前即时快照时间不同，系统日志正常增长会造成差异。

| 项目 | 初次检查 | 最终快照 |
|---|---:|---:|
| ext4 总容量 | 1,081,101,176,832 B | 1081101176832 B |
| ext4 已用 | 46,164,344,832 B（42.993896 GiB） | 46064521216 B（42.900928 GiB） |
| ext4 可用 | 979,944,476,672 B | 980044300288 B |
| VHDX 文件长度 / 实际分配 | 71,251,787,776 B | 71251787776 B / 71251787776 B |
| VHDX 虚拟最大容量 | 1,099,511,627,776 B（1024 GiB） | 未改动 |

最终 Linux 快照：2026-10-24T07:50:09Z。ext4 已用净减少 99823616 B；删除文件的精确逻辑字节数为 113,250,795 B，分配块为 113,258,496 B。二者不同，不能把系统日志变化算成清理量。Windows VHDX 物理占用未减少，本轮没有执行 TRIM、压实或缩容。

## 2. 原始主要占用与保留边界

| 路径 | 原始分配字节 | 归属 / 本轮处理 |
|---|---:|---|
| /home/shen | 43,047,768,064 | 总计；不清空 home |
| /home/shen/shen | 28,110,213,120 | 下列研究目录的父目录；保留 |
| /home/shen/shen/datasets | 14,521,335,808 | exiD/highD/inD/rounD/DRIFT 等其他研究数据；未验证备份，保留 |
| /home/shen/shen/mixed_autonomy_lab | 13,429,850,112 | README 确认为 Mixed Autonomy Lab，非 PIJWM 主项目；完整保留 |
| 上述目录 /outputs | 7,078,424,576 | 其他项目结果，保留 |
| 上述目录 /.venv | 5,350,871,040 | 其他项目环境，保留 |
| /home/shen/miniconda3 | 14,929,113,088 | 共享/其他项目环境，未删包或环境 |
| /home/shen/miniconda3/envs/cityflow310 | 7,405,645,824 | Python 3.10.19，包元数据显示 torch 2.10.0/CUDA 组件；仅元数据核验，未运行 GPU |
| /home/shen/miniconda3/envs/pg-mad | 7,046,402,048 | Python 3.9.25，包元数据显示 torch 2.8.0/CUDA 组件；非 PIJWM 专属，保留 |
| /usr | 2,099,576,832 | 系统组件，保留 |
| /var | 1,012,162,560 | 系统状态和日志；只处理下述两个缓存 |
| /var/log/journal | 634,298,368 | 系统日志；未授权清理，保留 |
| /root / /opt / /usr/local / /tmp | 20,480 / 4,096 / 102,400 / 57,344 | 无高收益低风险项目缓存；保留 |

其他项目代码及数据没有与 D 盘 PIJWM 进行伪去重，也没有据此删除。ControlTraj 和 flow 分别有 9、314 项状态条目，DiffAIL 为干净状态；全部未改动。本轮以 shen 身份只读核验，未同步远端或更改 Git 关系。按名称搜索仅发现 /home/shen/PIJWM；不据此宣称其他项目中绝无未知 PIJWM 内容。私人文件和凭据内容未读取。

## 3. 实际删除清单

| 路径 | 逻辑字节 | 已分配字节 | 理由 / 恢复 |
|---|---:|---:|---|
| /var/cache/apt/srcpkgcache.bin | 56,614,966 | 56,619,008 | APT 二进制索引缓存，可从保留的 /var/lib/apt/lists 重建 |
| /var/cache/apt/pkgcache.bin | 56,635,829 | 56,639,488 | 同上 |

删除前核对精确路径、普通文件、链接计数、大小和打开文件描述符，未发现占用；删除后确认均不存在。没有运行 apt 重建缓存；以后需要时由 APT 生成，或另行批准 apt-cache gencaches。未清理环境、系统日志、其他项目输出、checkpoint 或科研数据。

## 4. WSL 工作区与可用性

- /home/shen/PIJWM 原已存在且为空；本轮只新增 README_WSL.md（1256 B），未迁移其他项目文件，未复制 D 盘代码或大型数据，未初始化 Git。目录最终分配 8192 B。
- shen（UID 1000）登录、工作目录访问及标准库 JSON/数学/文件读写自检通过；临时目录已清理。
- /usr/bin/python3 3.12.3；pip、yaml 模块可发现；系统环境无 NumPy、torch、pytest。其他项目 Conda 中有科学计算包，不视为 PIJWM 可直接复用的环境，也未替其执行运行验证。
- Git 2.43.0；WSL 可只读读取 /mnt/d/shen/PKU/PIJWM 的 HEAD，未通过 Linux 修改该仓库。docker/nvcc 命令未发现，不表示硬件驱动或所有安装路径均已完整审计。
- DNS 与 PyPI HTTPS HEAD=200；GitHub HTTPS HEAD 超时。WSL 启动提示 localhost 代理未映射到 NAT WSL；代理环境变量为空。未修改网络设置；GitHub 连接需要后续单独处理。
- 当前适合轻量标准库实验，不是已准备好的 PIJWM 训练环境。将来须先批准具体代码、依赖、数据、空间预算；没有擅自安装/删除必要组件。

## 5. swap、C 盘与容量决策

- VHDX 真实路径：D:\WSL\Ubuntu\ext4.vhdx；非稀疏属性，Windows 分配大小通过 GetCompressedFileSizeW 核验，虚拟容量通过只读 GetVirtualDiskInformation 核验。
- 未发现用户 .wslconfig。swap 虚拟容量 4 GiB，当次使用 0；宿主文件长及分配均 37,748,736 B（36 MiB）。最终观察路径：C:\Users\Lenovo\AppData\Local\Temp\68BED469-2CB0-4ED7-9BCB-260FD955F0FF\swap.vhdx。会话 GUID 会变化，须重新发现，不能把历史完整路径当固定配置。
- Windows 默认 TEMP 为 C:\Users\Lenovo\AppData\Local\Temp；本任务宿主进程 TEMP/TMP/TMPDIR 指向 D 盘项目既有 tmp 目录，但未在那里生成项目产物。仅系统管理的 WSL swap 仍使用 C 盘；未主动向 C 盘创建项目数据或备份。
- 当前已用 42.900928 GiB，超过 20 GiB 22.900928 GiB。尚未满足常驻 15 GiB + 临时 5 GiB 的预算；df 的大可用数只是现有大容量文件系统的逻辑余量。
- home 之外已用约 2.809566 GiB；这不是精确最小安装需求。不能声称 Linux/PyTorch 必然需要 43 GiB；本轮主要阻塞是不得删除的其他项目和环境。科学计算最小环境需求仍为 UNKNOWN，需按将来实验实测。
- 核对本机 WSL --help：有 --manage/--resize、--set-sparse、--move 等入口。微软磁盘文档说明 --resize 扩容；本轮不把它当成已验证的安全缩容办法。仅更改数字不能把现有 43 GiB 数据塞入 20 GiB。
- 待批准方案一：研究者先决定其他项目及环境如何长期保留，再重新评估 20 GiB。不得据本报告自行删除这些内容。
- 待批准方案二：在备份、Ubuntu 停止、VHDX 独占及 Windows 压实工具可用性核验后，评估 Linux discard 与宿主压实，回收空闲物理块；这不降低文件系统最大容量，当前实际内容仍约 43 GiB，不会因此达到 20 GiB。具体压实工具尚未执行/验证。
- 待批准方案三：若需迁移 swap 到 D 盘，先批准 Windows 全局 .wslconfig 的具体改动和 WSL 重启影响。没有擅自写配置。
- 20 GiB 最大容量涉及缩容/迁移/重建/重注册时，须另行批准完整备份和恢复方案；本轮不实施。

参考：本机 WSL 2.6.3.0 帮助；Microsoft Learn https://learn.microsoft.com/en-us/windows/wsl/disk-space 与 https://learn.microsoft.com/en-us/windows/wsl/wsl-config （2026-10-24 只读核对）。

## 6. D 盘保护、记忆与 Git 交付

- D 盘科研资产未再次清理、压缩、迁移或覆盖；code/artifacts 基线为 21395 个普通文件、37569160502 B，元数据指纹 d6f06968e79db79ebc0c6e0d513cdcdf47bcd16a5a0569d9ed81c4d89d8620f9。最终再比对。
- AGENTS.md、AI_CONTEXT/00_PROJECT_STATE.md、docs/PROJECT_MEMORY_AND_SOURCES.md 的原有用户内容完整保留，只追加本任务段落；CHANGELOG 同样追加。未更新科研方向、算法或实验结论。
- 原有 main HEAD：5431f2170614da0225745e751e002629ad9a1cfe，5 项修改、23 项删除、11 项未跟踪。仅隔离提交本任务新增/追加的文档；不把原有修改、删除、未跟踪文件混入提交。
- PROJECT_MEMORY_AND_SOURCES.md 原已未跟踪，只追加本节，暂不将整个原文件加入 Git。AGENTS/AI_CONTEXT/CHANGELOG 若提交，使用 HEAD 原文加本任务独立追加块，保留工作区此前修改。
- 提交/推送结果以最终执行报告与对应 docs(wsl) Git 记录为准；这里不预先宣称推送成功。没有更新 AI 研究事实、没有执行训练/仿真/GPU/封存测试评估，也没有执行其他项目清理或 VHDX 操作。
