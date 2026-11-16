"""分配器 v3：可复现 + 向量化全局贪心（目标：毫秒级，能进决策器）。

v2 的两个问题，本版都修：
  1. 不可复现——信道阴影走 AirFogSim 的全局随机源。改用 channel_deterministic_v1。
  2. 太慢（2 秒）——全局贪心每步在 Python 双层循环里扫 2400 个组合。
     本版把「每个包 × 每个格子」的边际增益整体向量化，用 numpy 一次算完。

产物 code/artifacts/allocator_v3_<日期>/
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

PIPE = r"D:\hermes-yan\projects\PIJWM\pipeline"
sys.path.insert(0, PIPE)
sys.path.insert(0, r"D:\hermes-yan\projects\PIJWM\module6")
sys.path.insert(0, r"D:\shen\PKU\PIJWM\code\scripts")

import demo_end2end_v2 as D                              # noqa: E402
import sweep_v5 as S                                    # noqa: E402
from channel_deterministic_v1 import channel_means_deterministic  # noqa: E402
from oracle_2x2_gap_v1 import solve_optimal, build_packets         # noqa: E402

CONFIGS = [(0.0, 45.0, 1.1), (0.0, 45.0, 1.4)]


def setup(offset, block, rho_target, npl=10):
    ch = channel_means_deterministic(offset_db=offset, block_db=block)
    cal = S.samples_of(ch, S.CAL_SEEDS)
    per_link_mean = cal.mean(axis=(0, 1, 3))
    supply = float(np.mean(per_link_mean)) * D.TOTAL_CELLS
    cells_total = sum((1, 2, 4)[k % 3] for k in range(len(D.LINK_NAMES) * npl))
    cell_bits = rho_target * supply / cells_total
    packets = build_packets(cell_bits, npl=npl)
    lam = float(sum(p["value"] for p in packets) / (D.TOTAL_CELLS * 4.0))
    return ch, packets, lam


def global_greedy_vec(dec, packets, lam, slots=None):
    """向量化全局贪心。

    dec: (M, N_SLOT, n_links, N_RB)
    每步算一遍 (包, 格子) 的边际期望价值矩阵，取最大者。
    边际价值 = 该包当前期望价值在该格加入后的增量；用「还差多少比特」的概率跃迁算。
    """
    M, NS, NL, NR = dec.shape
    slots = list(slots if slots is not None else D.USABLE_SLOTS)
    npk = len(packets)

    link_of = np.array([p["link"] for p in packets])
    bits_of = np.array([p["bits"] for p in packets], dtype=float)
    val_of = np.array([p["value"] for p in packets], dtype=float)

    # 把所有可用格子的容量一次抽出来：(n_cells, M)，按链路取值
    cell_idx = [(s, r) for s in slots for r in range(NR)]
    nc = len(cell_idx)
    caps = np.zeros((npk, nc, M))
    for c, (s, r) in enumerate(cell_idx):
        caps[:, c, :] = dec[:, s, link_of, r].T        # (M,npk) -> (npk,M)

    got = np.zeros((npk, 1, M))                        # 每包已收比特
    val_now = np.zeros(npk)
    taken = np.zeros(nc, dtype=bool)
    chosen_by_pkt = [[] for _ in range(npk)]

    for _ in range(nc):
        short = np.maximum(bits_of[:, None, None] - got, 0.0)      # (npk,1,M)
        nxt = np.minimum(got + caps, bits_of[:, None, None])
        ok_now = (got >= bits_of[:, None, None] - 1e-9)
        ok_nxt = (nxt >= bits_of[:, None, None] - 1e-9)
        delta_p = ok_nxt.mean(axis=2) - ok_now.mean(axis=2)        # (npk,nc) 概率增量
        gain = val_of[:, None] * delta_p                           # (npk,nc)
        gain[:, taken] = -np.inf
        flat = int(np.argmax(gain))
        pi, ci = divmod(flat, nc)
        if gain[pi, ci] <= lam:
            break
        taken[ci] = True
        chosen_by_pkt[pi].append(cell_idx[ci])
        got[pi, 0, :] = nxt[pi, ci, :]

    alloc = {p: chosen_by_pkt[p] for p in range(npk) if chosen_by_pkt[p]}
    return alloc


def global_greedy_ref(dec, packets, lam, slots=None):
    """v2 的参考实现（未向量化），用于对拍向量化版本是否等价。"""
    M = dec.shape[0]
    slots = list(slots if slots is not None else D.USABLE_SLOTS)
    npk = len(packets)
    got = np.zeros((npk, M), dtype=float)
    taken = {}
    val_now = np.zeros(npk)
    while True:
        best_gain, best = -np.inf, None
        for p in range(npk):
            li, bits, val = packets[p]["link"], packets[p]["bits"], packets[p]["value"]
            for s in slots:
                for r in range(D.N_RB):
                    if (s, r) in taken:
                        continue
                    cap = dec[:, s, li, r]
                    if cap.max() <= 0:
                        continue
                    new_got = np.minimum(got[p] + cap, bits)
                    new_val = val * float(np.mean(new_got >= bits - 1e-9))
                    gain = new_val - val_now[p]
                    if gain > best_gain + 1e-12:
                        best_gain, best = gain, (p, s, r, new_got, new_val)
        if best is None or best_gain <= lam:
            break
        p, s, r, new_got, new_val = best
        got[p] = new_got; val_now[p] = new_val; taken[(s, r)] = p
    alloc = {}
    for (s, r), p in taken.items():
        alloc.setdefault(p, []).append((s, r))
    return alloc


def score(cap, alloc, packets):
    d, _, u = D.ledger(cap, alloc, packets)
    return (sum(packets[p]["value"] for p in d if d[p]),
            sum(1 for v in d.values() if v) / len(packets), u)


def main():
    reps = list(S.EVAL_SEEDS[:6])
    m_forecast = 8
    outdir = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "allocator_v3_" + time.strftime("%Y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    rows = []

    # 先自检：信道可复现 + 向量化与参考实现等价
    from channel_deterministic_v1 import check_reproducible
    rep = check_reproducible(2)
    print(f"[自检] 信道三次构建最大差 {rep:.2e} dB -> 可复现 {rep < 1e-12}", flush=True)

    for (off, blk, rho) in CONFIGS:
        ch, packets, lam = setup(off, blk, rho)
        print(f"== 配置 block={blk} rho={rho} 包数={len(packets)} λ={lam:.3f}", flush=True)
        for seed in reps:
            true_cap = np.asarray(D.serve_one_frame(ch, seed)["capacity_bits"])
            dec = S.samples_of(ch, [seed + 100000 + j for j in range(m_forecast)])

            a_pp = S.allocate("dist_prob", dec, packets, lam)
            a_ref, t_ref = None, None
            t0 = time.time(); a_ref = global_greedy_ref(dec, packets, lam); t_ref = time.time() - t0
            t0 = time.time(); a_vec = global_greedy_vec(dec, packets, lam); t_vec = time.time() - t0

            u_pp, d_pp, _ = score(true_cap, a_pp, packets)
            u_ref, d_ref, _ = score(true_cap, a_ref, packets)
            u_vec, d_vec, _ = score(true_cap, a_vec, packets)

            same = (u_vec == u_ref)
            hint = S.oracle_allocate(true_cap, packets)
            a_c, _, st_c = solve_optimal(true_cap[None, ...], packets, time_limit=20.0, hint_alloc=hint)
            u_c, d_c, _ = score(true_cap, a_c, packets)

            rows.append(dict(block_db=blk, rho=rho, seed=seed,
                             perpacket=u_pp, ref=u_ref, vec=u_vec, ceiling=u_c,
                             dlv_pp=d_pp, dlv_vec=d_vec, dlv_c=d_c,
                             t_ref_ms=t_ref * 1000, t_vec_ms=t_vec * 1000,
                             match=int(same), st_c=st_c))
            print(f"  seed={seed} 逐包={u_pp:.2f} 参考={u_ref:.2f}({t_ref*1000:.0f}ms) "
                  f"向量={u_vec:.2f}({t_vec*1000:.1f}ms) 一致={same} 天花板={u_c:.2f}", flush=True)

    p = os.path.join(outdir, "results.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    md = ["# 分配器 v3：可复现 + 向量化（%s）" % time.strftime("%Y-%m-%d"), "",
          "| 配置 | 逐包贪心 | 全局贪心(参考) | 全局贪心(向量) | 真值天花板 | 补上比例 | 向量化耗时 | 与参考一致 |",
          "|---|---|---|---|---|---|---|---|"]
    for (blk, rho) in sorted({(r["block_db"], r["rho"]) for r in rows}):
        sub = [r for r in rows if r["block_db"] == blk and float(r["rho"]) == float(rho)]
        m = lambda c: float(np.mean([r[c] for r in sub]))
        a, b, c2, d0 = m("perpacket"), m("ref"), m("vec"), m("ceiling")
        frac = (c2 - a) / (d0 - a) if d0 > a else float("nan")
        md.append(f"| 遮挡{blk}dB / ρ{rho} | {a:.2f} | {b:.2f} | {c2:.2f} | {d0:.2f} | {frac:.0%} | "
                  f"{m('t_vec_ms'):.1f} ms | {int(m('match'))}/{len(sub)} |")
    open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print("\n产物:", outdir)


if __name__ == "__main__":
    main()