"""ESTIMATE (no ferric run): per-(survivor, G) work of the plan.rs pair-FT kernel on a bench
cell, and how much of it bit-preserving caches remove.

Replays PairFtPlan::walk (same screen, same window formula) over ferric's segmented shells
(cell_facts.ferric_shells) and the half-G set at the facts gcut, then counts:

* surv_g   : sum over survivors of ngp            (exp + sincos evaluations today)
* exp_pp   : sum over (shell pair, prim pair) of max ngp over its images
             (exp evaluations with a per-pair magnitude cache shared across images)
* trig_ket : sum over (bra shell, ket atom, image, bra prim, ket EXPONENT) of max ngp
             (sincos evaluations with a per-task cache shared by ket shells of one atom
             that repeat the same exponent - general contraction)
* trig_all : same key with the bra side also merged over shells of one atom
             (= the sites of plan.rs's shell groups on cc-pVDZ, whose groups are exactly the
             same-exponent columns)
* trig_v2  : trig_all without the image (value-changing option V2, image split of the phase)
* cart_g   : sum over survivors of ngp * ncart_a * ncart_b (Cartesian accumulate)

Usage: python work_counts.py dryice
"""

import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc/bench")
import cell_facts as cf  # noqa: E402

THRESH = 1e-15
MARGIN = 10.0


def main(cell: str) -> None:
    sym, xyz, lat = cf.read_cell(cell)
    shells = cf.ferric_shells("cc-pvdz", sym, xyz)
    gcut = 2 * math.sqrt(math.log(1e13))
    g2 = cf.g_half_norms2(lat, gcut)
    gmax = math.sqrt(g2[-1])
    amin = min(s["e"].min() for s in shells)
    rpair = math.sqrt(2 * math.log(1e3 / THRESH) / amin) + 2.0
    images = cf.lattice_points(lat, rpair)
    atom_of = []
    for s in shells:
        atom_of.append(
            int(np.argmin([np.linalg.norm(s["c"] - np.array(c)) for c in xyz]))
        )
    surv_g = cart_g = nsurv = 0
    exp_pp = defaultdict(int)
    trig_ket = defaultdict(int)
    trig_all = defaultdict(int)
    trig_v2 = defaultdict(int)
    for ia, sa in enumerate(shells):
        for ib, sb in enumerate(shells):
            p = sa["e"][:, None] + sb["e"][None, :]
            cc = sa["k"][:, None] * sb["k"][None, :] * (math.pi / p) ** 1.5
            red = sa["e"][:, None] * sb["e"][None, :] / p
            lsum = (sa["l"] + sb["l"]) * math.log(max(gmax, 1.0))
            nc = (sa["l"] + 1) * (sa["l"] + 2) // 2 * (sb["l"] + 1) * (sb["l"] + 2) // 2
            for il, lv in enumerate(images):
                ab = sa["c"] - (sb["c"] + lv)
                mag = np.abs(cc * np.exp(-red * (ab @ ab)))
                ok = np.argwhere(mag >= THRESH)
                for ka, kb in ok:
                    g2max = (
                        4
                        * p[ka, kb]
                        * (max(math.log(mag[ka, kb] / THRESH), 0) + MARGIN + lsum)
                    )
                    ngp = int(np.searchsorted(g2, g2max, side="right"))
                    if ngp == 0:
                        continue
                    nsurv += 1
                    surv_g += ngp
                    cart_g += ngp * nc
                    k1 = (ia, ib, ka, kb)
                    exp_pp[k1] = max(exp_pp[k1], ngp)
                    ea, eb = float(sa["e"][ka]), float(sb["e"][kb])
                    k2 = (ia, atom_of[ib], il, ka, eb)
                    trig_ket[k2] = max(trig_ket[k2], ngp)
                    k3 = (atom_of[ia], ea, atom_of[ib], il, eb)
                    trig_all[k3] = max(trig_all[k3], ngp)
                    k4 = (atom_of[ia], ea, atom_of[ib], eb)
                    trig_v2[k4] = max(trig_v2[k4], ngp)
    print(
        f"{cell}: nG_half={len(g2)} gmax={gmax:.2f} images={len(images)} survivors={nsurv}"
    )
    print(f"  surv_g (exp+sincos today)       {surv_g:.3e}")
    print(
        f"  exp_pp (exp, image-shared)      {sum(exp_pp.values()):.3e}  ratio {surv_g / sum(exp_pp.values()):.2f}"
    )
    print(
        f"  trig_ket (sincos, ket-shared)   {sum(trig_ket.values()):.3e}  ratio {surv_g / sum(trig_ket.values()):.2f}"
    )
    print(
        f"  trig_all (sincos, both shared)  {sum(trig_all.values()):.3e}  ratio {surv_g / sum(trig_all.values()):.2f}"
    )
    print(
        f"  trig_v2 (sincos, + image split) {sum(trig_v2.values()):.3e}  ratio {surv_g / sum(trig_v2.values()):.2f}"
    )
    print(
        f"  cart_g (sum ngp*nca*ncb)        {cart_g:.3e}  avg ncart pair {cart_g / surv_g:.2f}"
    )


if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
