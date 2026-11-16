"""可复现的信道构建：替换 demo_end2end_v2.channel_means_from_airfogsim。

问题（2026-11-16 查实）：AirFogSim 的 LogNormal_Shadowing 用全局 np.random.normal 抽阴影，
没有种子入口，导致同一个配置连续两次调用信道差 7.9 dB，进而整轮实验不可复现
（allocator v1 两次运行、同一 seed，贪心分数 17.15 vs 15.15）。

本模块只做一件事：用固定种子的 numpy Generator 复刻同一条 3GPP 对数正态阴影公式，
其余（路损）照旧走 AirFogSim，保证结果可复算。
"""
from __future__ import annotations

import sys
import os

import numpy as np

PIPE = r"D:\hermes-yan\projects\PIJWM\pipeline"
sys.path.insert(0, PIPE)
sys.path.insert(0, r"D:\hermes-yan\projects\PIJWM\module6")

import demo_end2end_v2 as D   # noqa: E402


def channel_means_deterministic(offset_db=0.0, block_db=0.0, shadow_seed=20261116,
                                shadow_std=4.0, n_veh=None):
    """与 demo_end2end_v2.channel_means_from_airfogsim 同语义，但阴影可复现。

    路损：仍然调 AirFogSim 的 V2IChannel（确定性，无随机）。
    阴影：用固定种子复刻 LogNormal_Shadowing 的首帧形式（preShadowing=0, delta=0）。
         公式取自 code/reference/AirFogSim/airfogsim/channel_callback/shadowing_callback.py：
         shadow = 10*log10(1e-9 + exp(-d/dc)*10^(pre/10) + sqrt(1-exp(-2d/dc))*10^(new/10))
         d=0 时退化为 shadow = 10*log10(1e-9 + 10^(pre/10))，即 new 项权重为 0，
         所以首帧阴影必须直接抽正态，不能走这条退化式。
    """
    D.ATTEN_OFFSET_DB = offset_db
    D.BLOCK_LOSS_DB = block_db
    ch = D.V2IChannel(n_Veh=len(D.VEH_POS), n_BS=1,
                      frequency_ranges=np.array([D.FREQ_GHZ] * D.N_RB),
                      BS_positions=[D.RSU_POS])
    ch.update_positions(D.VEH_POS)
    ch.update_pathloss()
    pl = np.asarray(ch.PathLoss)[:, 0, :]

    rng = np.random.default_rng(shadow_seed)
    sh = rng.normal(0.0, shadow_std, size=len(D.VEH_POS))

    return {nm: {"mean_db": pl[i] + sh[i] + offset_db, "std_db": np.full(D.N_RB, 3.0)}
            for i, nm in enumerate(D.LINK_NAMES)}


def check_reproducible(n=3, **kw):
    """连续构建 n 次，返回最大差异，用于自检。"""
    vals = []
    for _ in range(n):
        ch = channel_means_deterministic(**kw)
        k0 = list(ch.keys())[0]
        vals.append(np.array([ch[k]["mean_db"][0] for k in ch]))
    diffs = [float(np.max(np.abs(vals[i] - vals[0]))) for i in range(1, n)]
    return max(diffs) if diffs else 0.0


if __name__ == "__main__":
    d = check_reproducible(3)
    print("三次构建最大差 (dB):", d)
    print("可复现:", d < 1e-12)
    ch = channel_means_deterministic()
    for k in list(ch)[:3]:
        print(k, "mean_db[0]=", round(float(ch[k]["mean_db"][0]), 2))
