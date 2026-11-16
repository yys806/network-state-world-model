"""把 V2U4Real 原始压缩包转成 OpenCOOD 能读的目录（时间戳 -> 帧号）。

OpenCOOD 按 <split>/<scene>/<agent>/ouster/000001.pcd 索引，
我们的文件按时间戳命名（1752745655.4.pcd），必须重命名。

输出：
  code/artifacts/v2u4real/val/<scene>/1/ouster/000001.pcd + yaml/ouster/000001.yaml
  code/artifacts/v2u4real/val/<scene>/2/ouster/000001.pcd + yaml/ouster/000001.yaml
"""
from __future__ import annotations

import os
import re
import shutil
import tarfile
import time

import zstandard  # noqa: F401  （若缺失就用 tar --zstd 子进程，见下）

SRC = r"D:\hermes-yan\cache\pijwm-v2u4real"
DST = r"D:\shen\PKU\PIJWM\code\artifacts\v2u4real\val"
SCENE = "2025-07-17-17-42_7"
TARS = {
    1: os.path.join(SRC, f"{SCENE}__agent-1.tar.zst"),
    2: os.path.join(SRC, f"{SCENE}__agent-2.tar.zst"),
}


def extract_members(tar_path, want):
    """从 .tar.zst 里挑出需要的成员，返回 {成员名: bytes}。"""
    import subprocess
    import tempfile
    out = {}
    # 先用 tar 列出成员，再按需解到临时目录（比全解省空间）
    with tempfile.TemporaryDirectory() as td:
        names = subprocess.run(["tar", "--zstd", "-tf", tar_path],
                               capture_output=True, text=True).stdout.splitlines()
        picked = [n for n in names if want(n)]
        print(f"  {os.path.basename(tar_path)}: 命中 {len(picked)} 个成员")
        if not picked:
            return out
        # 分批解压，避免命令行过长
        B = 200
        for i in range(0, len(picked), B):
            batch = picked[i:i + B]
            subprocess.run(["tar", "--zstd", "-xf", tar_path, "-C", td] + batch,
                           capture_output=True)
        for n in picked:
            p = os.path.join(td, n)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    out[n] = f.read()
    return out


def main():
    t0 = time.time()
    for agent, tp in TARS.items():
        if not os.path.exists(tp):
            print(f"缺压缩包 {tp}")
            continue
        print(f"== agent{agent}")
        # agent1: 只要 ouster 的 pcd 与 yaml；agent2 同理
        data = extract_members(tp, lambda n: ("/ouster/" in n) and (n.endswith(".pcd") or n.endswith(".yaml")))
        if not data:
            continue
        # 按时间戳分组
        pcds, yamls = {}, {}
        for n, blob in data.items():
            ts = re.search(r"([\d.]+)\.(pcd|yaml)$", n)
            if not ts:
                continue
            t = float(ts.group(1))
            (pcds if n.endswith(".pcd") else yamls)[t] = blob
        frames = sorted(set(pcds) & set(yamls))
        print(f"  时间戳配齐 {len(frames)} 帧（pcd {len(pcds)} / yaml {len(yamls)}）")

        od = os.path.join(DST, SCENE, str(agent), "ouster")
        yd = os.path.join(DST, SCENE, str(agent), "yaml", "ouster")
        os.makedirs(od, exist_ok=True)
        os.makedirs(yd, exist_ok=True)
        for i, t in enumerate(frames, 1):
            name = f"{i:06d}"
            with open(os.path.join(od, name + ".pcd"), "wb") as f:
                f.write(pcds[t])
            with open(os.path.join(yd, name + ".yaml"), "wb") as f:
                f.write(yamls[t])
        print(f"  写出 {len(frames)} 帧 -> {od}")

    print(f"\n完成，用时 {time.time()-t0:.0f}s")
    print("输出根:", DST)


if __name__ == "__main__":
    main()