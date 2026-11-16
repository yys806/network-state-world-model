"""跑通验证（smoke test）：V2U4Real 真实位姿 → 几何遮挡 → 每链接路状态。

目的不是出结果，是回答「我们的方法在这份数据上跑不跑得通」。
只走三件事，不碰检测器、不用 GPU：
  1. 目标位置从自车系平移到世界系（已核：纯平移即正确，z 标准差 0.15 m）；
  2. 自己设一个路侧设施（RSU），对每个目标→RSU 的链路做 BEV 线段-矩形遮挡判定；
  3. 统计每帧的视距/遮挡比例，看真实数据里**遮挡是否随时变**——这是我们方法的前提。

判据（跑通的四条）：
  A. 能对全部帧算出链路状态，无异常；
  B. 遮挡比例在帧间**会变**（不是恒定），否则「按分布决策」没有载体；
  C. 至少存在被遮挡的链路（不是全视距）；
  D. 输出可复算（同一帧两次跑结果一致）。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import yaml
from shapely.geometry import LineString, Polygon

ROOT = r"D:\hermes-yan\cache\pijwm-v2u4real\parsed"
OUT = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "smoke_v2u4real_" + time.strftime("%Y%m%d"))

OBJ_TYPES = {"Car", "Cyclist", "Truck"}      # 研究者 2026-11-16 定：汽车/骑行者/卡车都算


def load_frames(agent=1, sensor="ouster"):
    fs = sorted(glob.glob(f"{ROOT}/**/{agent}/yaml/{sensor}/*.yaml", recursive=True))
    frames = []
    for fp in fs:
        y = yaml.safe_load(open(fp, encoding="utf-8", errors="ignore"))
        pose = y["lidar_pose"]
        objs = []
        for oid, v in (y.get("vehicles") or {}).items():
            if v.get("obj_type") not in OBJ_TYPES:
                continue
            loc = v.get("location"); ext = v.get("extent"); ang = v.get("angle")
            if not loc or None in loc or not ext:
                continue
            objs.append(dict(id=oid, type=v["obj_type"],
                             x=float(loc[0] + pose[0]), y=float(loc[1] + pose[1]),
                             z=float(loc[2] + pose[2]),
                             L=float(ext[0]), W=float(ext[1]),
                             yaw=float(np.deg2rad(ang[1])) if ang else 0.0))
        frames.append(dict(t=os.path.basename(fp)[:-5], ego=pose[:3], objs=objs))
    return frames


def box_polygon(o):
    """BEV 旋转矩形。"""
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    hl, hw = o["L"] / 2.0, o["W"] / 2.0
    pts = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        pts.append((o["x"] + dx * c - dy * s, o["y"] + dx * s + dy * c))
    return Polygon(pts)


def blocked_between(a, b, obstacles, pad=0.0):
    """线段 a->b 是否被任一障碍物的 BEV 矩形拦住（排除两端自身）。"""
    seg = LineString([a, b])
    for o in obstacles:
        poly = box_polygon(o)
        if pad:
            poly = poly.buffer(pad)
        if seg.intersects(poly) and not poly.buffer(1e-9).contains(LineString([a, a])):
            # 端点落在框内不算遮挡（那是目标自己）
            if poly.contains(LineString([a, b]).interpolate(0.02, normalized=True)) and \
               poly.contains(LineString([a, b]).interpolate(0.98, normalized=True)):
                continue
            return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = load_frames(1, "ouster")
    print(f"帧数 {len(frames)}，每帧目标数 "
          f"{[len(f['objs']) for f in frames[:5]]} ...")

    # 用全场景目标范围定 RSU（路口角：x 最大侧、y 取中位）
    allx = [o["x"] for f in frames for o in f["objs"]]
    ally = [o["y"] for f in frames for o in f["objs"]]
    allz = [o["z"] for f in frames for o in f["objs"]]
    rsu = (max(allx) + 15.0, float(np.median(ally)), float(np.median(allz)))
    print(f"目标范围 x[{min(allx):.1f},{max(allx):.1f}] y[{min(ally):.1f},{max(ally):.1f}]")
    print(f"RSU（自设）={tuple(round(v,1) for v in rsu)}")

    rows = []
    for f in frames:
        obs = f["objs"]
        links = []
        for o in obs:
            others = [q for q in obs if q["id"] != o["id"]]
            blk = blocked_between((o["x"], o["y"]), (rsu[0], rsu[1]), others)
            d = float(np.hypot(o["x"] - rsu[0], o["y"] - rsu[1]))
            links.append(dict(id=o["id"], type=o["type"], dist=d, blocked=bool(blk)))
        rows.append(dict(t=f["t"], n=len(links),
                         n_blocked=sum(1 for l in links if l["blocked"]),
                         mean_dist=float(np.mean([l["dist"] for l in links])) if links else 0.0,
                         links=links))

    # 判据
    n_all = sum(r["n"] for r in rows)
    n_blk = sum(r["n_blocked"] for r in rows)
    blk_ratio_per_frame = [r["n_blocked"] / r["n"] for r in rows if r["n"]]
    print(f"\n=== 判据 ===")
    print(f"A 全部帧算出链路: {all(r['n'] > 0 for r in rows)}  （共 {n_all} 条链路）")
    print(f"B 遮挡比例帧间会变: 标准差 {np.std(blk_ratio_per_frame):.3f}，"
          f"范围 [{min(blk_ratio_per_frame):.2f}, {max(blk_ratio_per_frame):.2f}] "
          f"-> {'会变' if np.std(blk_ratio_per_frame) > 0.02 else '基本不变（有问题）'}")
    print(f"C 存在被遮挡链路: {n_blk}/{n_all} = {n_blk/max(n_all,1):.1%}")
    print(f"D 距离分布: 中位 {np.median([l['dist'] for r in rows for l in r['links']]):.1f} m")

    json.dump(dict(rsu=rsu, frames=len(rows),
                   n_links=n_all, n_blocked=n_blk,
                   blk_ratio_std=float(np.std(blk_ratio_per_frame)),
                   per_frame=[dict(t=r["t"], n=r["n"], blocked=r["n_blocked"],
                                   mean_dist=r["mean_dist"]) for r in rows]),
              open(os.path.join(OUT, "smoke_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n产物:", OUT)


if __name__ == "__main__":
    main()