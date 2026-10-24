# Ubuntu WSL 环境与存储记录

任务：PIJWM-WSL-CLEANUP-001；核验日期：2026-10-24。

结论：研究者明确授权的四目录已永久删除。Linux 内部已用 3.432888 GiB，已满足 20 GiB 用量预算；Windows VHDX 仍分配 66.354492 GiB，未回收大部分宿主空间，也未设置 20 GiB 硬上限。没有继续扩大删除范围。

## 1. 环境、口径与边界

- Ubuntu 24.04.3 LTS / WSL 2；WSL 2.6.3.0，内核 6.6.87.2-microsoft-standard-WSL2，Windows 10.0.26100.4946。仅注册 Ubuntu。
- 字节为原始单位，GiB=1073741824 B，GB=1000000000 B。Linux 使用 du -x / statvfs，不计 /mnt/c、/mnt/d 或其他挂载视图；硬链接同一次统计只算一份。
- 没有操作 VHDX、全局 WSL 配置或系统环境变量；不在 C 盘主动创建项目数据。宿主进程 TEMP/TMP/TMPDIR 指向 D 盘项目既有 tmp，Python 关闭字节码写入；系统自身日志和 swap 不等同项目写入。
- 没有访问 locked_test 样本，没有训练、仿真、GPU 或正式评价。D 盘科研资产没有再次清理、压缩或搬迁；只更新授权的记忆文档。

## 2. 容量实测

| 指标 | 四目录删除即时之前 | 删除即时之后 | 最新核验 |
|---|---:|---:|---:|
| ext4 已用字节 | 46074892288 | 3671642112 | 3686035456 |
| ext4 总容量字节 | 1081101176832 | 1081101176832 | 1081101176832 |
| ext4 可用字节 | 980033929216 | 1022437179392 | 1022422786048 |
| Windows VHDX 文件长度/分配字节 | 71251787776 | 未在该瞬间测量 | 71247593472 / 71247593472 |
| VHDX 虚拟最大字节 | 1099511627776 | 未修改 | 1099511627776 |

- 删除即时净减少 42403250176 B（39.491104 GiB）。四目录经硬链接去重占用 42403233792 B，差 16384 B 属文件系统元数据口径，不能伪装成额外数据收益。后续日志增长使最新已用略高。
- 最新快照 2026-10-24T08:08:55.696446+00:00：已用 3.432888 GiB，20 GiB 预算中尚余 16.567112 GiB，足够预留约 5 GiB 轻量实验空间；不是已实施的容量限制。科学计算依赖安装前仍须评估空间。
- Windows VHDX 相比之前仅少 4194304 B（4 MiB），未主动压实，不能把 Linux 减少的约 39.49 GiB 报成 Windows 已释放空间。Windows 磁盘总空闲量还会受其他进程影响，本轮不将其变动归因清理。
- Ubuntu 根文件系统仍约 1006.854 GiB，VHDX 最大 1024 GiB；宿主物理占用约 66.354492 GiB，尚未达到 20 GiB。压实只回收空块，不自动降低最大容量；压实/缩容均须另行批准，未执行。

## 3. 实际永久删除清单

研究者在了解内容包含其他研究数据、结果和环境后明确表示“直接删除吧，允许永久删除”。此授权仅用于下表四个精确目录，不构成删除其他项目的通用授权。没有已验证的完整备份，不保证科研数据、结果或源码可恢复；环境以后需重新安装。

| 精确路径 | 单独 du 分配字节 | 普通文件数 | 处理结果 |
|---|---:|---:|---|
| /home/shen/shen/datasets | 14521335808 | 986 | 已永久删除 |
| /home/shen/shen/mixed_autonomy_lab | 13429850112 | 31112 | 已永久删除 |
| /home/shen/miniconda3/envs/cityflow310 | 7459119104 | 22286 | 已永久删除 |
| /home/shen/miniconda3/envs/pg-mad | 7046402048 | 23003 | 已永久删除 |

- 共 77387 个普通文件、2722 个符号链接、5926 个目录。目录内含自身 .venv/outputs 的一并删除，只在父目录统计一次。
- 两套 Conda 环境有 2762 个共享硬链接 inode，共 53473280 B。单独 du 之和 42456707072 B 不是真实净收益，应采用合并去重 42403233792 B。
- 删除前核对真实路径、所有者、挂载边界、文件数与元数据，并检查进程占用；未发现占用，不跨文件系统，不删除 locked_test。四目标已重新确认不存在。
- 四目录外 /home/shen 保留区 14833 条元数据，删除前后 SHA-256 均为 15aa325a41146cd49459c2915dacfa01c44a7806b64a04e979e919f732ae1194。保留 SSH、Shell 配置、个人文件、Conda base、ControlTraj/DiffAIL/flow 等 Git 仓库；这是元数据核验，不声称对全部私人内容逐字节读取。

## 4. 初始安全缓存阶段（历史记录）

此前仅删除 /var/cache/apt/srcpkgcache.bin（56614966 B）及 /var/cache/apt/pkgcache.bin（56635829 B），合计逻辑 113250795 B、分配 113258496 B，可从保留的 /var/lib/apt/lists 重建。2026-10-24T07:50:09Z 的已用为 46064521216 B；当时 20 GiB 未达标。该结论已被本次明确授权后的新状态取代。

初次系统审计、两阶段完整快照与删除前元数据指纹保存在 JSON，不能将旧表中的“保留”误认为四目录目前仍存在。

## 5. 当前工作区与环境

- /home/shen/PIJWM 仅有本任务 README_WSL.md，非 Git 仓库；未复制 D 盘代码、正式数据或 checkpoint。D 盘仍是工程主项目。旧目录无需归并，已按后续授权删除。
- shen（UID 1000）正常登录并访问该目录；系统 Python 3.12.3、Git 2.43.0 可运行；此前标准库文件读写自检通过且临时文件已清理，删除后再次验证基本标准库执行与目录权限。
- Conda base 保留：/home/shen/miniconda3/bin/python 为 Python 3.13.11，conda 25.11.1；仅核验版本，未安装依赖。cityflow310、pg-mad 及 mixed_autonomy_lab 的 .venv 已删除。
- 系统 Python 可发现 pip/yaml，不能发现 numpy/torch/pytest。适合标准库轻量运行；不代表 PIJWM 正式训练环境可用。需要旧依赖时须依据保留的外部配置或重新制定安装方案，未保证已删环境可原样重建。
- 删除后目录分配：/home/shen 644538368 B，miniconda3 477065216 B，/home/shen/shen 159027200 B，工作目录 8192 B。它们为层级关系，不应相加。/var/log 的 shen 复测权限不足，未用不完整 du 冒充精确目录总量；整盘 statvfs 不受该限制。
- 初次网络检查 DNS、PyPI HTTPS HEAD=200，GitHub HEAD 超时；WSL 仍提示 Windows localhost 代理未镜像至 NAT WSL。未修改网络，未把连接问题谎称已解决。docker/nvcc 初检未发现。

## 6. swap、VHDX 与未授权操作

- VHDX：D:\WSL\Ubuntu\ext4.vhdx；文件长度与 GetCompressedFileSizeW 分配相同，GetVirtualDiskInformation 只读确认最大 1099511627776 B。
- 无 .wslconfig 改动。/proc/swaps：虚拟 4 GiB、已用 0；历史宿主实测 37748736 B（36 MiB），路径 C:\Users\Lenovo\AppData\Local\Temp\<会话GUID>\swap.vhdx；GUID 会变，不能将旧会话路径当配置。
- 本机 WSL --help 提供 --manage/--resize、--set-sparse、--move；没有执行任何相关变更。微软文档描述 --resize 扩容，不能把它当作当前已有数据的安全缩容许可。
- 若要回收 Windows 物理空间，下一步只能先提交具体压实方案、停机影响、备份与恢复条件，等待批准；不得自行 TRIM、压实、迁移、导出导入、重注册、卸载或重装。20 GiB 硬上限目前未实现。

参考：本机 WSL 2.6.3.0 帮助；Microsoft Learn https://learn.microsoft.com/en-us/windows/wsl/disk-space 与 https://learn.microsoft.com/en-us/windows/wsl/wsl-config，初始阶段已只读核对。

## 7. D 盘保护、记忆和 Git

- D 盘既有科研资产：21395 个普通文件、37569160502 B，路径/大小/修改时间指纹 d6f06968e79db79ebc0c6e0d513cdcdf47bcd16a5a0569d9ed81c4d89d8620f9。核验排除且任务结束前清理 Git 凭据工具在指定项目 TEMP 生成的 539 B dotnet-suggest 注册标记；不读取或清理其他科研文件内容。
- 三个 worktree 保持干净，未移动或删除。main 原有 5 项修改、23 项删除、11 项未跟踪保留；本次只隔离提交任务自己的环境文档和段落，不提交原有修改。
- AGENTS/AI_CONTEXT/CHANGELOG 仅替换本任务追加段落；PROJECT_MEMORY_AND_SOURCES 原已未跟踪，仅本地更新任务段落。未重建用户已删 task_plan/progress/findings，没有运行生成器或改研究索引/算法。
- 本轮不修改研究方向、方法或结果。实际 commit/push 以 Git 历史和最终交付为准，记录不预先宣称成功。完成后停止，不继续清理或操作虚拟磁盘。
