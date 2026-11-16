"""分配器工程版 v2：全局贪心 + 局部交换改进（毫秒级，不用整数规划）。

诊断（allocator_run2.log）：CP-SAT 限时 1 秒拿不到任何改进，全部回退到热启动解。
40 包 × 60 格的随机规划模型对在线来说太大，整数规划只能当离线上界用。

v2 的关键差别：
  sweep_v5 的 dist_prob 是「逐包贪心」——按价值密度排好包的顺序，每个包各挑自己最好的格子。
  这里改成「全局贪心」——每一步在所有 (包, 格子) 组合里挑边际期望价值最高的那一个。
逐包贪心会让先处理的包把好格子占光，后面的包只能捡剩下的；全局贪心不会。

再加一步局部交换：随机挑一个已分配的格子，试着转给另一个包，总价值涨就接受。

产物 code/artifacts/allocator_v2_<日期>/
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


def setup(offset, block, rho_target, npl=10):
    D.ATTEN_OFFSET_DB = offset
    D.BLOCK_LOSS_DB = block
    ch = D.channel_means_from_airfogsim()
    cal = S.samples_of(ch, S.CAL_SEEDS)
    per_link_mean = cal.mean(axis=(0, 1, 3))
    supply = float(np.mean(per_link_mean)) * D.TOTAL_CELLS
    cells_total = sum((1, 2, 4)[k % 3] for k in range(len(D.LINK_NAMES) * npl))
    cell_bits = rho_target * supply / cells_total
    packets = build_packets(cell_bits, npl=npl)
    lam = float(sum(p["value"] for p in packets) / (D.TOTAL_CELLS * 4.0))
    return ch, packets, lam


def expected_value(packets, p, got):
    """包 p 在已收比特向量 got（M 条样本）下的期望价值。"""
    bits = packets[p]["bits"]
    prob = float(np.mean(got >= bits - 1e-9))
    return packets[p]["value"] * prob


def global_greedy(dec, packets, lam, slots=None):
    """全局贪心：每步在所有 (包, 格子) 里挑边际期望价值最高的一对。"""
    M = dec.shape[0]
    slots = slots if slots is not None else list(D.USABLE_SLOTS)
    nr = D.N_RB
    npk = len(packets)

    got = np.zeros((npk, M), dtype=float)      # 每个包在每条样本里的已收比特
    taken = {}                                  # (s,r) -> 包
    val_now = np.zeros(npk, dtype=float)        # 每个包当前期望价值

    while True:
        best_gain, best = -np.inf, None
        for p in range(npk):
            li, bits, val = packets[p]["link"], packets[p]["bits"], packets[p]["value"]
            for s in slots:
                for r in range(nr):
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
        got[p] = new_got
        val_now[p] = new_val
        taken[(s, r)] = p

    alloc = {}
    for (s, r), p in taken.items():
        alloc.setdefault(p, []).append((s, r))
    return alloc


def local_search(dec, packets, alloc, iters=4000, rng=None):
    """局部交换：把一个格子从包 A 转给包 B（或从空位给某包），总期望价值涨就接受。"""
    rng = rng or np.random.default_rng(0)
    M = dec.shape[0]
    npk = len(packets)
    slots = list(D.USABLE_SLOTS)
    nr = D.N_RB

    cells = []          # 当前占用的格子列表
    owner = {}          # (s,r) -> 包
    for p, cs in alloc.items():
        for c in cs:
            cells.append(c)
            owner[c] = p

    def value_of(assignment):
        got = np.zeros((npk, M), dtype=float)
        for (s, r), p in assignment.items():
            got[p] += dec[:, s, packets[p]["link"], r]
        tot = 0.0
        for p in range(npk):
            tot += packets[p]["value"] * float(np.mean(got[p] >= packets[p]["bits"] - 1e-9))
        return tot, got

    cur_val, _ = value_of(owner)
    for _ in range(iters):
        # 随机挑一个格子（已占用或空）
        if rng.random() < 0.75 and cells:
            c = cells[int(rng.integers(len(cells)))]
        else:
            c = (int(rng.choice(slots)), int(rng.integers(nr)))
        p_new = int(rng.integers(npk))
        p_old = owner.get(c)
        if p_old == p_new:
            continue
        trial = dict(owner)
        if p_new == p_old:
            continue
        # 不能重复占用：新包若已占该格，跳过
        trial[c] = p_new
        tv, _ = value_of(trial)
        if tv > cur_val + 1e-9:
            owner = trial
            cur_val = tv
            if c not in cells:
                cells.append(c)
    alloc2 = {}
    for c, p in owner.items():
        alloc2.setdefault(p, []).append(c)
    return alloc2, cur_val


def score(cap, alloc, packets):
    d, _, u = D.ledger(cap, alloc, packets)
    return (sum(packets[p]["value"] for p in d if d[p]),
            sum(1 for v in d.values() if v) / len(packets), u)


def main():
    reps = list(S.EVAL_SEEDS[:6])
    m_forecast = 8
    outdir = os.path.join(r"D:\shen\PKU\PIJWM\code\artifacts", "allocator_v2_" + time.strftime("%Y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for (off, blk, rho) in CONFIGS:
        ch, packets, lam = setup(off, blk, rho)
        print(f"== 配置 block={blk} rho={rho} 包数={len(packets)} λ={lam:.3f}", flush=True)
        for seed in reps:
            true_cap = np.asarray(D.serve_one_frame(ch, seed)["capacity_bits"])
            dec = S.samples_of(ch, [seed + 100000 + j for j in range(m_forecast)])

            t0 = time.time()
            a_pp = S.allocate("dist_prob", dec, packets, lam)       # 逐包贪心（现状）
            t_pp = time.time() - t0

            t0 = time.time()
            a_gg = global_greedy(dec, packets, lam)                 # 全局贪心
            t_gg = time.time() - t0

            t0 = time.time()
            a_ls, _ = local_search(dec, packets, a_gg, iters=2000)  # + 局部交换
            t_ls = time.time() - t0

            hint = S.oracle_allocate(true_cap, packets)
            a_c, sec_c, st_c = solve_optimal(true_cap[None, ...], packets,
                                             time_limit=20.0, hint_alloc=hint)

            u_pp, d_pp, _ = score(true_cap, a_pp, packets)
            u_gg, d_gg, _ = score(true_cap, a_gg, packets)
            u_ls, d_ls, _ = score(true_cap, a_ls, packets)
            u_c, d_c, _ = score(true_cap, a_c, packets)

            rows.append(dict(block_db=blk, rho=rho, seed=seed,
                             perpacket=u_pp, globalgreedy=u_gg, localsrch=u_ls, ceiling=u_c,
                             dlv_pp=d_pp, dlv_gg=d_gg, dlv_ls=d_ls, dlv_c=d_c,
                             t_pp=t_pp, t_gg=t_gg, t_ls=t_ls, st_c=st_c))
            print(f"  seed={seed} 逐包={u_pp:.2f}({t_pp*1000:.0f}ms) 全局={u_gg:.2f}({t_gg*1000:.0f}ms) "
                  f"局部={u_ls:.2f}({t_ls*1000:.0f}ms) 天花板={u_c:.2f} [{st_c}]", flush=True)

    p = os.path.join(outdir, "results.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    md = ["# 分配器工程版 v2：全局贪心 + 局部交换（%s）" % time.strftime("%Y-%m-%d"), "",
          "| 配置 | 逐包贪心（现状） | 全局贪心 | +局部交换 | 真值天花板 | 补上比例 | 单次耗时 |",
          "|---|---|---|---|---|---|---|"]
    for (blk, rho) in sorted({(r["block_db"], r["rho"]) for r in rows}):
        sub = [r for r in rows if r["block_db"] == blk and float(r["rho"]) == float(rho)]
        m = lambda c: float(np.mean([r[c] for r in sub]))
        a, b, c2, d0 = m("perpacket"), m("globalgreedy"), m("localsrch"), m("ceiling")
        frac = (c2 - a) / (d0 - a) if d0 > a else float("nan")
        md.append(f"| 遮挡{blk}dB / ρ{rho} | {a:.2f} | {b:.2f} | {c2:.2f} | {d0:.2f} | {frac:.0%} | "
                  f"{(m('t_gg')+m('t_ls'))*1000:.0f} ms |")
    open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print("\n产物:", outdir)


if __name__ == "__main__":
    main()