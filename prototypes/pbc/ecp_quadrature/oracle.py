"""Reference values ONLY (never imported by ecpq.py):
* pyscf_value: PySCF 2.13 ECPscalar_sph with the zero-weight 1e-3 screen-guard primitive (FINDINGS Iteration 22 (2a)).
* lebedev_value: independent radial Gauss-Legendre x Lebedev quadrature (generalises reference/pbc/ecp_accuracy/quad.py:
  semi-local projectors by the Legendre addition theorem; basis functions evaluated by PySCF's eval_gto)."""

import numpy as np
from scipy.special import eval_legendre

AUG = 1e-3


def _pyscf_ecp(e):
    out = []
    for l, ts in e["channels"]:
        pw = [[] for _ in range(7)]
        for n, z, d in ts:
            pw[n].append([z, d])
        lp = -1 if l == max(c[0] for c in e["channels"]) else l
        out.append([lp, pw])
    return {e["symbol"]: [e["ncore"], out]}


def _mol(sa, sb, e, guard=True):
    from pyscf import gto

    def bas(s):
        l, ex, c = s["raw"]
        prims = [[x, y] for x, y in zip(ex, c)]
        if guard:
            prims.append([AUG, 0.0])
        return [[l] + prims]

    Z = {"I": 53, "Au": 79}[e["symbol"]]
    spin = (2 + Z - e["ncore"]) % 2
    return gto.M(
        atom=[("H@1", sa["center"]), ("H@2", sb["center"]), (e["symbol"], e["center"])],
        basis={"H@1": bas(sa), "H@2": bas(sb), e["symbol"]: [[0, [1.0, 1.0]]]},
        ecp=_pyscf_ecp(e),
        unit="B",
        verbose=0,
        spin=spin,
        cart=False,
    )


def pyscf_value(sa, sb, e, guard=True):
    """<sa|U_e|sb> spherical block (libcint order = ferric order)."""
    return _mol(sa, sb, e, guard).intor("ECPscalar_sph", shls_slice=(0, 1, 1, 2))


def _radial(nper, edges):
    x, w = np.polynomial.legendre.leggauss(nper)
    R, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        R.append(0.5 * (b - a) * x + 0.5 * (b + a))
        W.append(0.5 * (b - a) * w)
    return np.concatenate(R), np.concatenate(W)


def lebedev_value(
    sa,
    sb,
    e,
    nang=1202,
    nper=160,
    edges=(0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6.5, 8, 10, 12, 15, 18, 24),
):
    from pyscf.dft import gen_grid

    mol = _mol(sa, sb, e, guard=False)
    na = 2 * sa["l"] + 1
    ang = gen_grid.MakeAngularGrid(nang)
    om, wo = ang[:, :3], 4 * np.pi * ang[:, 3]
    r, wr = _radial(nper, edges)
    cosg = np.clip(om @ om.T, -1, 1)
    L = max(c[0] for c in e["channels"])
    semi = [l for l, _ in e["channels"] if l != L]
    Pl = {l: (2 * l + 1) / (4 * np.pi) * eval_legendre(l, cosg) for l in semi}

    def U(l, rr):
        return sum(
            d * rr ** (n - 2) * np.exp(-z * rr * rr)
            for ll, ts in e["channels"]
            if ll == l
            for n, z, d in ts
        )

    V = np.zeros((na, 2 * sb["l"] + 1))
    C = np.asarray(e["center"])
    for ri, wi in zip(r, wr):
        ao = mol.eval_gto("GTOval_sph", C + ri * om)[:, : na + 2 * sb["l"] + 1]
        fa, fb = ao[:, :na] * wo[:, None], ao[:, na:]
        V += wi * ri * ri * U(L, ri) * (fa.T @ fb)
        for l in semi:
            V += wi * ri * ri * U(l, ri) * (fa.T @ Pl[l] @ (fb * wo[:, None]))
    return V
