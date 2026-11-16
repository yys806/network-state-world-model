# 对第三方代码的本地改动清单（V2U4Real / OpenCOOD）

`code/reference/V2U4Real/` 已被 `.gitignore` 排除，不进版本库。这里记录我们**改了什么、为什么改**，换机器或重下代码时照着改回来即可。

建立：2026-11-16

---

## 1. `opencood/tools/inference.py`（第 12 行附近）

**改法**：把顶层 `import open3d as o3d` 改成 try/except，失败时置 `o3d = None` 并警告。

**为什么**：可视化只在 `--show_vis` / `--save_vis` 分支用到，但顶层导入会让**整个评测**在无显示环境（WSL 缺 libusb）直接崩掉。

---

## 2. `opencood/visualization/vis_utils.py`（第 9 行附近）

**改法**：同上，`import open3d as o3d` 包 try/except，失败置 None。

**为什么**：`bev_postprocessor.py`、`voxel_postprocessor.py` 都在模块层导入 `vis_utils`，导致只要 import 到后处理就被 open3d 卡死。

---

## 3. `opencood/data_utils/post_processor/voxel_postprocessor.py`（第 21 行）

**改法**：注释掉顶层的 `from opencood.visualization import vis_utils`，改为在唯一使用它的函数（`visualize_single_sample_output_gt` 调用处）内局部导入。

**为什么**：全文件只有一处用它，且是调试可视化，不该绑定到评测路径上。

---

## 4. `opencood/data_utils/datasets/basedataset.py`（核心改动）

**改了三处**：

1. **`__init__` 末尾**：解析 `wild_setting.our_delay`，得到 `our_delay_enable / mode / mean_frames / max_frames / table_csv / seed`；`mode == 'table'` 时把 CSV 读成 `{(cav_id, frame): delay_frames}`。
2. **`time_delay_calculation`**：签名从 `(self, ego_flag)` 改为 `(self, ego_flag, cav_id=None, frame_idx=0)`；函数体在「ego 返回 0」之后插入我们的分支：
   - `table` 模式：直接查表
   - `random` 模式：以 `(frame_idx, crc32(cav_id), seed)` 为种子抽指数分布，截断到 `[0, max_frames]`
3. **`__getitem__` 调用处**：改为 `self.time_delay_calculation(cav_content['ego'], cav_id=cav_id, frame_idx=timestamp_index)`。

**为什么**：官方异步是「所有源一律 `async_overhead` 帧」，无法表达「这个源近所以快、那个源被挡所以慢、这个干脆别传」。改成可逐源逐帧给定之后，我们的调度决策才有地方落地。

**备份**：同目录 `basedataset.py.bak-prePIJWM`（改动前原版）。

**注意**：`our_delay` 只在 `wild_setting.async: true` 时生效（函数里判了 `self.async_flag`）。

---

## 5. 配置侧（不属于代码，但必须同时存在）

`code/artifacts/ckpt_where2comm/config.yaml` 的 `wild_setting` 下：

```yaml
  async: true
  our_delay:
    enable: true
    mode: 'random'     # 或 'table'
    mean_frames: 8
    max_frames: 16
    seed: 20261116
    table_csv: ''      # mode=table 时填 CSV 绝对路径
```

要回到官方行为：把 `our_delay.enable` 设 `false`（保留 `async: true` 即官方固定 8 帧）；要同步基线：`async: false`。

---

## 6. 环境侧的补偿（不是改代码，但缺了就起不来）

- `box_overlaps` 需要编译：`python opencood/utils/setup.py build_ext --inplace`（`run_inference_wsl.sh` 会自动做）
- libusb 放在仓库 `.syslib/`，靠 `LD_LIBRARY_PATH` 指过去（WSL 里 sudo 要密码，装不进系统）
- numpy 必须 <2（老代码用 `np.Inf`，已锁 1.26.4）
