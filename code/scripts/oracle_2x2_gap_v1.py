"""2x2 信息×分配 四格实验：判断「训练世界模型」与「改进分配器」各值多少。

四格定义（列＝分配器，行＝决策时掌握的信息）：
  左上 A: 分布样本 + 现有逐格贪心      = 当前系统
  右上 B: 分布样本 + 最优分配（随机规划）= 分配上界
  左下 C: 真实未来 + 现有逐格贪心      = 预测上界（分配不变）
  右下 D: 真实未来 + 最优分配          = 天花板

用法：
  python oracle_2x2_gap_v1.py --configs 4 --reps 6 --m 8

产物写到 code/artifacts/oracle_2x2_<日期>/ 下的 results.csv 与 summary.md。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERMES = r"D:\hermes-yan\projects\PIJWM"
sys.path.insert(0, os.path.join(HERMES, "pipeline"))
sys.path.insert(0, os.path.join(HERMES, "module6"))

import demo_end2end_v2 as D          # noqa: E402
import sweep_v5 as S                 # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

SCALE = 100          # 比特放大倍数（CP-SAT 要整数）
VAL_SCALE = 1000     # 价值放大倍数


def build_packets(cell_bits, npl=10):
    return S.make_packets(cell_bits, npl=npl)


def solve_optimal(dec: np.ndarray, packets, time_limit=30.0, workers=8, hint_alloc=None):
    """给定 M 条容量样本，求使期望送达价值最大的分配（随机规划）。

    dec: (M, N_SLOT, n_links, N_RB) 单位比特
    hint_alloc: 可选，{包下标: [(slot, rb), ...]}，作为热启动提示；求解失败时原样返回它。
    返回 (allocation dict, elapsed, status)
    """
    M, NS, NL, NR = dec.shape
    caps = np.maximum(0, np.rint(np.asarray(dec, dtype=float) * SCALE)).astype(np.int64)
    bits = [int(round(float(p["bits"]) * SCALE)) for p in packets]
    vals = [int(round(float(p["value"]) * VAL_SCALE)) for p in packets]
    npk = len(packets)

    model = cp_model.CpModel()
    slots = list(D.USABLE_SLOTS)

    # x[pi][s][r]: 包 pi 占用 (s, r)
    x = {}
    for pi in range(npk):
        for s in slots:
            for r in range(NR):
                x[pi, s, r] = model.NewBoolVar(f"x_{pi}_{s}_{r}")

    # 每格最多一个包
    for s in range(NS):
        for r in range(NR):
            vs = [x[pi, s, r] for pi in range(npk) if (pi, s, r) in x]
            if vs:
                model.Add(sum(vs) <= 1)

    # 每个包在每个样本里的收货量与送达指示
    z = {}
    for pi, p in enumerate(packets):
        li = p["link"]
        for m in range(M):
            terms = []
            for s in slots:
                for r in range(NR):
                    c = int(caps[m, s, li, r])
                    if c > 0:
                        terms.append(c * x[pi, s, r])
            rec = model.NewIntVar(0, 10 ** 9, f"rec_{pi}_{m}")
            model.Add(rec == (sum(terms) if terms else 0))
            zz = model.NewBoolVar(f"z_{pi}_{m}")
            model.Add(rec >= bits[pi]).OnlyEnforceIf(zz)
            model.Add(rec <= bits[pi] - 1).OnlyEnforceIf(zz.Not())
            z[pi, m] = zz

    model.Maximize(sum(vals[pi] * z[pi, m] for pi in range(npk) for m in range(M)))

    # 热启动：把贪心解作为初始提示，否则限时太短时连可行解都找不到
    if hint_alloc:
        occ = {}
        for pi, cells in hint_alloc.items():
            for (s, r) in cells:
                if (pi, s, r) in x and (s, r) not in occ:
                    occ[(s, r)] = pi
        for (pi, s, r), var in x.items():
            model.AddHint(var, 1 if occ.get((s, r)) == pi else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    t0 = time.time()
    st = solver.Solve(model)
    elapsed = time.time() - t0

    allocation = {}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for pi in range(npk):
            chosen = []
            for s in slots:
                for r in range(NR):
                    if solver.Value(x[pi, s, r]) == 1:
                        chosen.append((s, r))
            if chosen:
                allocation[pi] = chosen
        status = "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE"
    elif hint_alloc:
        # 限时内没找到解时回退到热启动解，绝不允许返回空方案
        allocation = dict(hint_alloc)
        status = "HINT_FALLBACK"
    else:
        status = "NO_SOLUTION"
    return allocation, elapsed, status


def run_one(offset, block, rho_target, rep_seeds, m_forecast, npl=10, time_limit=30.0):
    """跑一个配置下的四格。场景构造与 sweep_v5.run_config 保持一致。"""
    D.ATTEN_OFFSET_DB = offset
    D.BLOCK_LOSS_DB = block
    ch = D.channel_means_from_airfogsim()

    # 校准集定包大小（与 sweep_v5 一致）
    cal = S.samples_of(ch, S.CAL_SEEDS)
    per_link_mean = cal.mean(axis=(0, 1, 3))
    supply = float(np.mean(per_link_mean)) * D.TOTAL_CELLS
    cells_total = sum((1, 2, 4)[k % 3] for k in range(len(D.LINK_NAMES) * npl))
    cell_bits = rho_target * supply / cells_total
    packets = build_packets(cell_bits, npl=npl)

    total_value = sum(p["value"] for p in packets)
    lam = float(total_value / (D.TOTAL_CELLS * 4.0))

    rows = []
    for i, seed in enumerate(rep_seeds):
        true_cap = np.asarray(D.serve_one_frame(ch, seed)["capacity_bits"])
        dec = S.samples_of(ch, [seed + 100000 + j for j in range(m_forecast)])

        # A 左上：分布样本 + 贪心
        alloc_A = S.allocate("dist_prob", dec, packets, lam)
        # C 左下：真实未来 + 贪心
        alloc_C = S.oracle_allocate(true_cap, packets)
        # D 右下：真实未来 + 最优（M=1）
        alloc_D, tD, stD = solve_optimal(true_cap[None, ...], packets, time_limit=time_limit)
        # B 右上：分布样本 + 最优（M=m）
        alloc_B, tB, stB = solve_optimal(dec, packets, time_limit=time_limit)

        def score(alloc):
            d, _, u = D.ledger(true_cap, alloc, packets)
            util = sum(packets[p]["value"] for p in d if d[p])
            dlv = sum(1 for v in d.values() if v) / len(packets)
            return util, dlv, u

        uA, dlvA, cA = score(alloc_A)
        uB, dlvB, cB = score(alloc_B)
        uC, dlvC, cC = score(alloc_C)
        uD, dlvD, cD = score(alloc_D)

        rows.append(dict(offset_db=offset, block_db=block, rho_target=rho_target, seed=seed,
                         cell_bits_kbit=cell_bits / 1000.0,
                         util_A=uA, util_B=uB, util_C=uC, util_D=uD,
                         dlv_A=dlvA, dlv_B=dlvB, dlv_C=dlvC, dlv_D=dlvD,
                         cells_A=cA, cells_B=cB, cells_C=cC, cells_D=cD,
                         t_solve_D=tD, st_D=stD, t_solve_B=tB, st_B=stB))
        print(f"  rep {i+1}/{len(rep_seeds)} seed={seed} "
              f"A={uA:.2f} B={uB:.2f} C={uC:.2f} D={uD:.2f} ({stD}/{stB})", flush=True)
    return rows, packets


def summarize(rows):
    """按配置汇总四格均值与两个 gap。"""
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["block_db"], r["rho_target"]), []).append(r)
    for k, rs in sorted(by.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        def mean(col):
            return float(np.mean([x[col] for x in rs]))
        A, B, C, Dd = mean("util_A"), mean("util_B"), mean("util_C"), mean("util_D")
        out.append(dict(block_db=k[0], rho=k[1], reps=len(rs),
                        A=A, B=B, C=C, D=Dd,
                        # 分配 gap：信息不变（都用同一批预测样本），把贪心换成最优
                        alloc_gap=B - A,
                        # 预测 gap：分配不变（都用同一个贪心），把预测样本换成真实未来
                        pred_gap=C - A,
                        # 天花板差距
                        ceiling_gap=Dd - A,
                        # 辅助：完美信息下分配还能涨多少；最优分配下信息还能涨多少
                        alloc_gap_under_perfect=Dd - C,
                        pred_gap_under_optimal=Dd - B,
                        dlv_A=mean("dlv_A"), dlv_B=mean("dlv_B"), dlv_C=mean("dlv_C"), dlv_D=mean("dlv_D")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--m", type=int, default=8, help="右上角用的预测样本数")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(
        r"D:\shen\PKU\PIJWM\code\artifacts", "oracle_2x2_" + time.strftime("%Y%m%d"))
    os.makedirs(outdir, exist_ok=True)

    configs = [
        (0.0, 0.0, 0.6),
        (0.0, 25.0, 0.6),
        (0.0, 45.0, 1.1),
        (0.0, 45.0, 1.4),
    ]
    rep_seeds = list(S.EVAL_SEEDS[: args.reps])

    all_rows = []
    for (off, blk, rho) in configs:
        print(f"== 配置 offset={off} block={blk} rho={rho}", flush=True)
        rows, packets = run_one(off, blk, rho, rep_seeds, args.m, time_limit=args.time_limit)
        all_rows += rows

    csv_path = os.path.join(outdir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    summ = summarize(all_rows)
    md = ["# 2x2 信息×分配 四格（%s）" % time.strftime("%Y-%m-%d"), "",
          "A=分布+贪心(现状)  B=分布+最优  C=真值+贪心  D=真值+最优(天花板)", "",
          "配置 | 重复 | A | B | C | D | 分配gap(C-A) | 预测gap(B-A) | 距天花板(D-A)",
          "---|---|---|---|---|---|---|---|---"]
    for s in summ:
        md.append(f"遮挡{s['block_db']}dB/ρ{s['rho']} | {s['reps']} | {s['A']:.2f} | {s['B']:.2f} | "
                  f"{s['C']:.2f} | {s['D']:.2f} | {s['alloc_gap']:+.2f} | {s['pred_gap']:+.2f} | {s['ceiling_gap']:+.2f}")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n产物:", outdir)


if __name__ == "__main__":
    main()
