#!/usr/bin/env python3
"""Does fusing the RI-MP2 (P|mn) fill with its MO transform pay? (numpy mechanism prototype)

Production (ferric_mp2::rimp2): the raw tensor (P|mn) is filled into one in-core
(naux, nao, nao) array, then streamed row by row through the two half transforms
(C_occ^T (Q|mn) C_vir) and dressed. The fused alternative fills one aux-shell block
(np, nao, nao) at a time into a reused buffer and transforms it at once, so the
naux*nao^2 tensor is never materialized and the transform reads data still in cache.

`prefaulted` is the materialized arm with its pages touched before the clock starts: the
floor that transparent huge pages approach. The integral generation itself is the same in
all arms; here it is modelled by copying a cached source block (the write traffic of the
fill). Only the memory traffic and the transform differ, which is the mechanism under test.

Measured (2026-11-28, one thread, i7-6800K): fused/materialized 0.77-0.88, but
fused/prefaulted 0.85 (C4), 1.05 (C8), 1.03 (C12), 1.05 (C16). The whole gain is the first
touch of the calloc'd tensor, which huge-page advice on that tensor removes, so fusion is
not implemented.

    python rimp2_fused_fill_transform_proto.py [ncarbon]      (one BLAS thread)

FLOPs are the same in both arms: per aux row 2*nao^2*nocc + 2*nao*nocc*nvir.
"""

import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


def shapes(nc):
    """cc-pVDZ / cc-pVDZ-RI alkane CnH2n+2, frozen core."""
    nh = 2 * nc + 2
    nao, naux = 14 * nc + 5 * nh, 56 * nc + 14 * nh
    nocc_tot = (6 * nc + nh) // 2
    return nao, naux, nocc_tot - nc, nao - nocc_tot


def run(nc, block=10, reps=3):
    nao, naux, nocc, nvir = shapes(nc)
    rng = np.random.default_rng(1)
    src = rng.standard_normal((block, nao, nao))
    src = 0.5 * (src + src.transpose(0, 2, 1))
    c_occ = rng.standard_normal((nao, nocc))
    c_vir = rng.standard_normal((nao, nvir))

    def transform(rows, out, q0):
        for q in range(rows.shape[0]):
            half = c_occ.T @ rows[q]
            out[q0 + q] = (half @ c_vir).ravel()

    def materialized(prefault=False):
        t = np.zeros((naux, nao, nao))  # calloc: pages faulted by the fill
        if (
            prefault
        ):  # the fault-free floor of the materialized arm (what huge pages approach)
            t.fill(0.0)
        out = np.empty((naux, nocc * nvir))
        t0 = time.perf_counter()
        for p0 in range(0, naux, block):
            p1 = min(p0 + block, naux)
            t[p0:p1] = src[: p1 - p0]
        t1 = time.perf_counter()
        transform(t, out, 0)
        t2 = time.perf_counter()
        del t
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, out

    def fused():
        buf = np.zeros((block, nao, nao))
        out = np.empty((naux, nocc * nvir))
        fill = xf = 0.0
        for p0 in range(0, naux, block):
            p1 = min(p0 + block, naux)
            t0 = time.perf_counter()
            buf[: p1 - p0] = src[: p1 - p0]
            t1 = time.perf_counter()
            transform(buf[: p1 - p0], out, p0)
            t2 = time.perf_counter()
            fill += t1 - t0
            xf += t2 - t1
        return fill, xf, 0.0, out

    res = {}
    arms = (
        ("materialized", materialized),
        ("prefaulted", lambda: materialized(True)),
        ("fused", fused),
    )
    for name, fn in arms:
        best = None
        for _ in range(reps):
            r = fn()
            tot = sum(r[:3])
            if best is None or tot < sum(best[:3]):
                best = r
        res[name] = best
    assert np.array_equal(res["materialized"][3], res["fused"][3]), (
        "the two arms must agree bit for bit"
    )
    print(
        f"C{nc}: nao={nao} naux={naux} nocc={nocc} nvir={nvir} tensor={naux * nao * nao * 8 / 1e6:.0f} MB"
    )
    for name, (f, x, d, _) in res.items():
        print(
            f"  {name:12s} fill {f:7.3f}  transform {x:7.3f}  drop {d:6.3f}  total {f + x + d:7.3f} s"
        )
    m, pf, u = (sum(res[k][:3]) for k in ("materialized", "prefaulted", "fused"))
    print(f"  fused / materialized = {u / m:.3f}   fused / prefaulted = {u / pf:.3f}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for nc in [int(a) for a in sys.argv[1:]] or [4, 8, 12]:
            run(nc)
