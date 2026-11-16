"""坐标变换核验：V2U4Real 的 vehicles 在雷达系，需转到世界系才能放 RSU。

lidar_pose 六维顺序按 OpenCOOD 约定 [x, y, z, roll, pitch, yaw]，角度单位度。
判定标准：转到世界系后，所有地面目标（Car/Cyclist/Truck）的中心 z 应聚在同一高度附近
（地面高度），离散度小说明变换对；离散度大说明顺序或单位错了。
"""
import glob
import numpy as np
import yaml

ROOT = r"D:\hermes-yan\cache\pijwm-v2u4real\parsed"


def rot(roll, pitch, yaw):
    """度 -> 旋转矩阵，顺序 Rz(yaw) Ry(pitch) Rx(roll)。"""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def to_world(loc, pose, order="zyx"):
    R = rot(pose[3], pose[4], pose[5])
    return R @ np.asarray(loc, dtype=float) + np.asarray(pose[:3], dtype=float)


def check(files, label):
    zs, xy = [], []
    for fp in files:
        y = yaml.safe_load(open(fp, encoding="utf-8", errors="ignore"))
        pose = y["lidar_pose"]
        for v in (y.get("vehicles") or {}).values():
            loc = v.get("location")
            if not loc or None in loc:
                continue
            w = to_world(loc, pose)
            zs.append(w[2]); xy.append((w[0], w[1]))
    zs = np.array(zs)
    xy = np.array(xy)
    print(f"== {label} ({len(zs)} 个目标)")
    print(f"   世界系 z: 均值 {zs.mean():.2f}  标准差 {zs.std():.2f}  "
          f"[{zs.min():.2f}, {zs.max():.2f}]")
    print(f"   世界系 x: [{xy[:,0].min():.1f}, {xy[:,0].max():.1f}]  "
          f"y: [{xy[:,1].min():.1f}, {xy[:,1].max():.1f}]")
    return zs.std()


if __name__ == "__main__":
    for agent, sensor in [(1, "m1"), (1, "ouster"), (1, "ruby"), (2, "ouster")]:
        fs = sorted(glob.glob(f"{ROOT}/**/{agent}/yaml/{sensor}/*.yaml", recursive=True))
        if fs:
            check(fs, f"agent{agent}/{sensor}")
