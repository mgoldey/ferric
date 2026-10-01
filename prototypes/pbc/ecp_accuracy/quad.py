"""Independent radial x Lebedev quadrature of semi-local ECP integrals (Legendre addition theorem)."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
from pyscf import gto
from pyscf.dft import gen_grid
import ecp_clean as E


def radial(nper=160, edges=(0, 0.5, 1.5, 3, 5, 8, 12, 18)):
    x, w = leggauss(nper)
    R, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        R.append(0.5 * (b - a) * x + 0.5 * (b + a))
        W.append(0.5 * (b - a) * w)
    return np.concatenate(R), np.concatenate(W)


def quad_value(sa, pa, sb, pb, pc, nang=1202, nper=160):
    mol = gto.M(
        atom=[("H@1", pa), ("H@2", pb)],
        basis={
            "H@1": [[sa[0]] + [[x, y] for x, y in zip(sa[1], sa[2])]],
            "H@2": [[sb[0]] + [[x, y] for x, y in zip(sb[1], sb[2])]],
        },
        unit="B",
        verbose=0,
        spin=0,
        cart=False,
    )
    na = 2 * sa[0] + 1
    ang = gen_grid.MakeAngularGrid(nang)
    om, wo = ang[:, :3], 4 * np.pi * ang[:, 3]  # PySCF weights sum to 1
    r, wr = radial(nper)
    cosg = np.clip(om @ om.T, -1, 1)
    Pl = {l: (2 * l + 1) / (4 * np.pi) * eval_legendre(l, cosg) for l in (0, 1, 2)}

    def U(l, rr):
        out = np.zeros_like(rr)
        for ll, ts in E.ECP_I:
            if ll == l:
                for n, z, d in ts:
                    out += d * rr ** (n - 2) * np.exp(-z * rr * rr)
        return out

    V = np.zeros((na, 2 * sb[0] + 1))
    for ri, wi in zip(r, wr):
        pts = np.asarray(pc) + ri * om
        ao = mol.eval_gto("GTOval_sph", pts)  # (nang, nao)
        fa, fb = ao[:, :na] * wo[:, None], ao[:, na:] * wo[:, None]
        V += wi * ri * ri * U(3, np.array([ri]))[0] * (fa.T @ (ao[:, na:]))
        for l in (0, 1, 2):
            V += wi * ri * ri * U(l, np.array([ri]))[0] * (fa.T @ Pl[l] @ fb)
    return V


if __name__ == "__main__":
    import sys

    sa = E.I_SH[3]
    sb = E.I_SH[3]
    pa = E.I0
    pb = E.I0 + E.LV
    pc = E.I0 + np.array([1.5, -1.2, 0.8])
    for nang, nper in ((590, 120), (1202, 160), (2030, 200)):
        q = quad_value(sa, pa, sb, pb, pc, nang, nper)
        print(nang, nper, "quad", q[0, 2], flush=True)
    print(
        "pyscf",
        E.pyscf_value(sa, pa, sb, pb, pc)[0, 2],
        "libecpint",
        E.lib_value(sa, pa, sb, pb, pc)[0, 2],
    )
