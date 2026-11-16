"""V2U4Real 接入第一步：解析 yaml，核验能否作为模块 1（观测装配）的输入。

只解 yaml、不碰点云，先把三件事核清楚：
  1. 两路（车 agent-1 / 机 agent-2）时间戳是否逐帧对齐；
  2. 目标 ID 是否跨帧连续（决定能否做轨迹、能否算目标级截止期）；
  3. 每帧目标数量与类型分布（决定多源互补空间有多大）。

产物写到 code/artifacts/v2u4real_ingest_<日期>/
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import yaml

ROOT = r"D:\hermes-yan\cache\pijwm-v2u4real\parsed"
SCENE = "2025-07-17-17-42_7"
OUT = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "v2u4real_ingest_" + time.strftime("%Y%m%d"))


def load_yaml(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 收集每个 agent / 雷达的 yaml
    files = glob.glob(os.path.join(ROOT, "**", "*.yaml"), recursive=True)
    by_stream = defaultdict(dict)     # (agent, sensor) -> {timestamp: path}
    for p in files:
        m = re.search(r"/(\d+)/yaml/([^/]+)/([\d.]+)\.yaml$", p.replace("\\", "/"))
        if not m:
            continue
        agent, sensor, ts = int(m.group(1)), m.group(2), float(m.group(3))
        by_stream[(agent, sensor)][ts] = p

    print("=== 数据流 ===")
    streams = {}
    for (a, s), d in sorted(by_stream.items()):
        ts = sorted(d)
        streams[(a, s)] = ts
        print(f"agent{a}/{s}: {len(ts)} 帧, {ts[0]} ~ {ts[-1]}, 间隔中位 "
              f"{sorted([round(ts[i+1]-ts[i], 3) for i in range(len(ts)-1)])[len(ts)//2] if len(ts)>1 else '-'} s")

    # 1) 两路时间对齐
    veh = sorted(set().union(*[set(v) for (a, s), v in streams.items() if a == 1]))
    uav = sorted(set().union(*[set(v) for (a, s), v in streams.items() if a == 2]))
    common = sorted(set(veh) & set(uav))
    print(f"\n=== 时间对齐 ===")
    print(f"车端 {len(veh)} 帧, 机端 {len(uav)} 帧, 共同 {len(common)} 帧 "
          f"({len(common)/max(len(veh),1):.1%} of 车端)")
    print(f"仅车有 {len(set(veh)-set(uav))}, 仅机有 {len(set(uav)-set(veh))}")

    # 2) 目标 ID 连续性与数量——取车的 m1 与机的 ouster
    print(f"\n=== 目标统计 ===")
    summary = {}
    for (a, s) in sorted(streams):
        d = by_stream[(a, s)]
        ids_per_frame = []
        types = Counter()
        for ts in sorted(d):
            y = load_yaml(d[ts])
            veh_obj = (y or {}).get("vehicles") or {}
            ids = list(veh_obj.keys())
            ids_per_frame.append(len(ids))
            for v in veh_obj.values():
                types[(v or {}).get("obj_type", "?")] += 1
        all_ids = set()
        for ts in d:
            y = load_yaml(d[ts])
            all_ids |= set(((y or {}).get("vehicles") or {}).keys())
        mean_n = sum(ids_per_frame) / len(ids_per_frame) if ids_per_frame else 0
        print(f"agent{a}/{s}: 每帧目标 均值 {mean_n:.1f} 最小 {min(ids_per_frame)} "
              f"最大 {max(ids_per_frame)} | 出现的不同 ID {len(all_ids)} | 类型 {dict(types)}")
        summary[f"agent{a}/{s}"] = dict(frames=len(d), mean_objs=mean_n,
                                        ids=len(all_ids), types=dict(types))

    # 3) 同一目标跨帧位置连续性（用车的 m1，按 ID 建轨迹）
    key = (1, "m1")
    if key in by_stream:
        d = by_stream[key]
        traj = defaultdict(list)
        for ts in sorted(d):
            y = load_yaml(d[ts])
            for oid, v in ((y or {}).get("vehicles") or {}).items():
                loc = (v or {}).get("location")
                if loc and None not in loc:
                    traj[oid].append((ts, [float(x) for x in loc]))
        print(f"\n=== 轨迹连续性（agent1/m1）===")
        lens = sorted((len(v) for v in traj.values()), reverse=True)
        print(f"目标数 {len(traj)}，轨迹长度分布 前5 {lens[:5]}，只出现1帧的 {sum(1 for x in lens if x==1)} 个")
        # 速度核验：用同 ID 相邻帧位移估速度
        vels = []
        for oid, pts in traj.items():
            for i in range(len(pts) - 1):
                dt = pts[i+1][0] - pts[i][0]
                if dt <= 0:
                    continue
                dx = (pts[i+1][1][0]-pts[i][1][0]); dy = (pts[i+1][1][1]-pts[i][1][1])
                vels.append((dx*dx+dy*dy) ** 0.5 / dt)
        if vels:
            vels.sort()
            import statistics
            print(f"目标速度（同 ID 相邻帧估）: 中位 {statistics.median(vels):.2f} m/s, "
                  f"90分位 {vels[int(len(vels)*0.9)]:.2f}, 最大 {max(vels):.2f} m/s")

    json.dump(dict(streams={f"agent{a}/{s}": len(v) for (a, s), v in streams.items()},
                   aligned=len(common), veh_only=len(set(veh)-set(uav)), uav_only=len(set(uav)-set(veh)),
                   summary=summary),
              open(os.path.join(OUT, "ingest_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n产物:", OUT)


if __name__ == "__main__":
    main()