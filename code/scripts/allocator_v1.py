"""分配器工程版 v1：把「样本上直接优化期望效用」压到在线可用时延内。

背景（E12）：同一批预测样本下，逐格贪心拿 11.02 分，20 秒整数规划拿 16.28 分，
真实未来 + 最优是 24.36。所以分配器本身值 +5.26，占天花板差距的四成。

本脚本回答一个问题：这个增益在 1 秒内能拿到多少？
做法：用贪心解做热启动（hint）喂给 CP-SAT，限时 1 秒，看能逼近多少。

产物 code/artifacts/allocator_v1_<日期>/
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

HERMES = r"D:\hermes-yan\projects\PIJWM"
sys.path.insert(0, os.path.join(HERMES, "pipeline"))
sys.path.insert(0, os.path.join(HERMES, "module6"))
sys.path.insert(0, os.path.join(r"D:\shen\PKU\PIJWM", "code", "scripts"))

import demo_end2end_v2 as D          # noqa: E402
import sweep_v5 as S                 # noqa: E402
from oracle_2x2_gap_v1 import solve_optimal, build_packets  # noqa: E402

CONFIGS = [(0.0, 45.0, 1.1), (0.0, 45.0, 1.4)]


def compute_lam(packets):
    return float(sum(p["value"] for p in packets) / (D.TOTAL_CELLS * 4.0))


def setup(offset, block, rho_target, npl=10):
    D.ATTEN_OFFSET_DB = offset
    D.BLOCK_LOSS_DB = block
    ch = D.channel_means_from_airfogsim()
    cal = S.samples_of(ch, S.CAL_SEEDS)
    per_link_mean = cal.mean(axis=(0, 1, 3))
    supply = float(np.mean(per_link_mean)) * D.TOTAL_CELLS
    cells_total = sum((1, 2, 4)[k % 3] for k in range(len(D.LINK_NAMES) * npl))
    cell_bits = rho_target * supply / cells_total
    return ch, build_packets(cell_bits, npl=npl), compute_lam(build_packets(cell_bits, npl=npl))


def score(cap, alloc, packets):
    d, _, u = D.ledger(cap, alloc, packets)
    return sum(packets[p]["value"] for p in d if d[p]), sum(1 for v in d.values() if v) / len(packets), u


def main():
    reps = list(S.EVAL_SEEDS[:6])
    m_forecast = 8
    outdir = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "allocator_v1_" + time.strftime("%Y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for (off, blk, rho) in CONFIGS:
        ch, packets, lam = setup(off, blk, rho)
        print(f"== 配置 block={blk} rho={rho}  包数={len(packets)}", flush=True)
        for seed in reps:
            true_cap = np.asarray(D.serve_one_frame(ch, seed)["capacity_bits"])
            dec = S.samples_of(ch, [seed + 100000 + j for j in range(m_forecast)])

            # 基线：逐格贪心
            a_greedy = S.allocate("dist_prob", dec, packets, lam)
            # 工程版：限时 0.5 / 1 / 3 秒
            res = {}
            for tl in (0.5, 1.0, 3.0):
                alloc, sec, st = solve_optimal(dec, packets, time_limit=tl, workers=8,
                                               hint_alloc=a_greedy)
                u, dlv, cells = score(true_cap, alloc, packets)
                res[tl] = (u, dlv, cells, sec, st)
            # 参考上限：真值最优（限时 20 秒，用已知未来的贪心解做热启动）
            hint_d = S.oracle_allocate(true_cap, packets)
            alloc_d, sec_d, st_d = solve_optimal(true_cap[None, ...], packets, time_limit=20.0,
                                                 hint_alloc=hint_d)
            u_d, dlv_d, cells_d = score(true_cap, alloc_d, packets)
            # 贪心基线分数
            u_g, dlv_g, cells_g = score(true_cap, a_greedy, packets)

            row = dict(block_db=blk, rho=rho, seed=seed,
                       greedy=u_g, greedy_dlv=dlv_g,
                       t05=res[0.5][0], t1=res[1.0][0], t3=res[3.0][0],
                       ceiling=u_d,
                       sec05=res[0.5][3], sec1=res[1.0][3], sec3=res[3.0][3],
                       st1=res[1.0][4], st3=res[3.0][4])
            rows.append(row)
            print(f"  seed={seed} 贪心={u_g:.2f} | 0.5s={res[0.5][0]:.2f}({res[0.5][3]:.2f}s) "
                  f"1s={res[1.0][0]:.2f}({res[1.0][3]:.2f}s,{res[1.0][4]}) "
                  f"3s={res[3.0][0]:.2f}({res[3.0][3]:.2f}s) | 真值上限={u_d:.2f}", flush=True)

    p = os.path.join(outdir, "results.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 汇总
    md = ["# 分配器工程版 v1（%s）" % time.strftime("%Y-%m-%d"), "",
          "限时 CP-SAT + 8 线程，输入 8 条预测样本；天花板＝真实未来 + 20 秒最优。", "",
          "| 配置 | 贪心 | 0.5s | 1s | 3s | 真值天花板 | 1s 补上贪心差距的比例 |",
          "|---|---|---|---|---|---|---|"]
    for (blk, rho) in sorted({(r["block_db"], r["rho"]) for r in rows}):
        sub = [r for r in rows if r["block_db"] == blk and float(r["rho"]) == float(rho)]
        g = np.mean([r["greedy"] for r in sub]); a = np.mean([r["t05"] for r in sub])
        b = np.mean([r["t1"] for r in sub]); c = np.mean([r["t3"] for r in sub])
        d0 = np.mean([r["ceiling"] for r in sub])
        frac = (b - g) / (d0 - g) if d0 > g else float("nan")
        md.append(f"| 遮挡{blk}dB / ρ{rho} | {g:.2f} | {a:.2f} | {b:.2f} | {c:.2f} | {d0:.2f} | {frac:.0%} |")
    open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print("\n产物:", outdir)


if __name__ == "__main__":
    main()