"""阶段一判据二（几何版）：到手信息量 与 目标可见性 的关系。

不依赖检测器，先回答一个更基础的问题：
  一个目标在点云里有多少点，跟它「能不能被看出来」是不是单调关系？
  如果「少给一部分数据」会让一部分目标的点数掉到不可见，那调度就有价值。

做法：
  1. 读一帧 pcd（binary 或 ascii 自动判）+ 对应 yaml 的目标框
  2. 对每个目标，数落在它 3D 框内的点数（点数 = 该目标贡献的信息量）
  3. 按「保住多少比例的目标」模拟交付，统计还能看见的目标数
  4. 输出：点数分布、以及「信息保留率 vs 仍可见目标比例」曲线

阈值说明：可见判据用「框内点数 >= N_MIN」，N_MIN 取全体目标点数的某个分位，
不改检测器也能做，是代理指标，需在报告里标明。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np
import yaml

ROOT = r"D:\hermes-yan\cache\pijwm-v2u4real"
OUT = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "info_visibility_" + time.strftime("%Y%m%d"))
OBJ_TYPES = {"Car", "Cyclist", "Truck"}


def read_pcd(path):
    """读 PCD：支持 ascii / binary，返回 (N,3) float。"""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("#") or not s:
                continue
            parts = s.split()
            header[parts[0]] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        fmt = header.get("DATA", ["ascii"])[0].lower()
        fields = header.get("FIELDS", ["x", "y", "z"])
        n = int(header.get("POINTS", ["0"])[0])
        if fmt == "ascii":
            arr = np.loadtxt(f, dtype=np.float64, max_rows=n)
            if arr.ndim == 1:
                arr = arr[None, :]
            ix = [fields.index(k) for k in ("x", "y", "z")]
            return arr[:, ix]
        # binary: 按 SIZE 求每点字节数
        sizes = [int(x) for x in header.get("SIZE", ["4"] * len(fields))]
        types_ = header.get("TYPE", ["F"] * len(fields))
        stride = sum(sizes)
        buf = f.read(n * stride)
        dt = []
        for sz, tp in zip(sizes, types_):
            dt.append((np.float32 if tp.upper() == "F" else np.int32))
        arr = np.frombuffer(buf, dtype=np.dtype([(f"f{i}", d) for i, d in enumerate(dt)]))
        pts = np.stack([arr[f"f{i}"] for i in range(len(dt))], axis=1).astype(np.float64)
        ix = [fields.index(k) for k in ("x", "y", "z")]
        return pts[:, ix]


def box_mask(pts, loc, ext, yaw):
    """点是否落在旋转框内（局部坐标判 3 轴各半长）。"""
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx = pts[:, 0] - loc[0]
    dy = pts[:, 1] - loc[1]
    dz = pts[:, 2] - loc[2]
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return ((np.abs(lx) <= ext[0] / 2) & (np.abs(ly) <= ext[1] / 2) & (np.abs(dz) <= ext[2] / 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    pcds = sorted(glob.glob(os.path.join(ROOT, "pcd", "**", "1", "ouster", "*.pcd"), recursive=True))
    print(f"点云帧 {len(pcds)}")
    if not pcds:
        raise SystemExit("没有 pcd，先解压")

    rows = []
    for fp in pcds:
        ts = os.path.basename(fp)[:-4]
        yp = os.path.join(ROOT, "parsed", "val", "2025-07-17-17-42_7", "1", "yaml", "ouster", ts + ".yaml")
        if not os.path.exists(yp):
            continue
        y = yaml.safe_load(open(yp, encoding="utf-8", errors="ignore"))
        pts = read_pcd(fp)
        pts = pts[np.isfinite(pts).all(axis=1)]
        for oid, v in (y.get("vehicles") or {}).items():
            if v.get("obj_type") not in OBJ_TYPES:
                continue
            loc, ext, ang = v.get("location"), v.get("extent"), v.get("angle")
            if not loc or None in loc or not ext:
                continue
            yaw = float(np.deg2rad(ang[1])) if ang else 0.0
            m = box_mask(pts, [float(x) for x in loc], [float(x) for x in ext], yaw)
            rows.append(dict(t=ts, id=oid, type=v["obj_type"], n=int(m.sum()),
                             dist=float(np.hypot(loc[0], loc[1]))))
        if len(rows) and len(rows) % 200 < 10:
            print(f"  {ts}: 累计 {len(rows)} 个目标", flush=True)

    ns = np.array([r["n"] for r in rows])
    print(f"\n=== 目标点数分布（{len(rows)} 个目标实例）===")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  {q:3d}% 分位: {np.percentile(ns, q):.0f} 点")
    print(f"  为 0 点的目标: {(ns == 0).sum()} ({(ns==0).mean():.1%})")
    print(f"  < 10 点的目标: {(ns < 10).sum()} ({(ns<10).mean():.1%})")

    # 两种丢包策略对比：随机丢（基线） vs 按价值保留（我们的方法思路）
    order_desc = np.argsort(-ns)     # 点数多 -> 少
    rng = np.random.default_rng(20261116)
    curve = []
    print("\n=== 信息保留率 vs 仍可见目标比例（阈值 20 点）===")
    print("  保留 | 随机丢包 | 按价值保留(多点数优先) | 提升")
    thr = 20
    for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        k = max(int(len(order_desc) * keep), 1)
        # 随机保留
        rand_sel = rng.choice(len(ns), size=k, replace=False)
        rand_vis = float((ns[rand_sel] >= thr).mean())
        # 按价值保留（点数多者优先）
        smart_sel = order_desc[:k]
        smart_vis = float((ns[smart_sel] >= thr).mean())
        curve.append(dict(keep=keep, thr=thr, random=rand_vis, value_first=smart_vis))
        print(f"  {keep:3.0%} | {rand_vis:7.1%} | {smart_vis:12.1%} | {smart_vis-rand_vis:+.1%}")

    json.dump(dict(n_objects=len(rows), pct=dict(zip(
        ["0", "10", "25", "50", "75", "90", "100"],
        [float(np.percentile(ns, q)) for q in (0, 10, 25, 50, 75, 90, 100)])),
        zero_ratio=float((ns == 0).mean()), curve=curve),
        open(os.path.join(OUT, "info_visibility.json"), "w", encoding="utf-8"), indent=2)
    print("\n产物:", OUT)


if __name__ == "__main__":
    main()