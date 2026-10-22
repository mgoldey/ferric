"""Issue #490 prototype: Becke grid-weight response of the LDA XC Hessian at FIXED density.

E_xc(R) = sum_g W_g(R) f(rho(r_g(R); R)),  r_g = R_home(g) + u_g  (points ride their atom),
W_g = s_g * P_home(r_g)/sum_C P_C(r_g)  (Becke 1988, 3 iterations, Becke size adjust
a = u/(u^2-1) clamped to |a|<=1/2, Bragg radii -- ferric's becke_weights_all), D fixed.

Second derivatives are taken by second-order forward-mode AD (value, grad[3N], hess[3N,3N]
per grid point) -- an independent construction from finite differences.

Checks:
  A. FD (central, h) of the analytic moving-grid fixed-D gradient == AD moving-grid Hessian
  B. second differences of the plain (no AD) moving-grid energy == AD Hessian
  C. translation sum rule: sum_B H[(B,a),(C,b)] == 0 for the moving grid; NOT for fixed grid
  D. mutants: drop W'' ; drop W' x rho' cross terms ; drop point motion of the AOs
"""

import sys

import numpy as np
from pyscf import dft, gto
from pyscf.data.radii import BRAGG
from pyscf.dft import gen_grid, radi

XC = "LDA,VWN"
NRAD, NANG = 40, 110
ANG2BOHR = 1.8897261246257702


# ---------------------------------------------------------------- hyper-dual numbers
class HD:
    """Per-point value v (n,), gradient g (n,m), Hessian h (n,m,m)."""

    def __init__(self, v, g, h):
        self.v, self.g, self.h = v, g, h

    @staticmethod
    def const(v, m):
        n = v.shape[0]
        return HD(v, np.zeros((n, m)), np.zeros((n, m, m)))

    def __add__(s, o):
        return HD(s.v + o.v, s.g + o.g, s.h + o.h)

    def __sub__(s, o):
        return HD(s.v - o.v, s.g - o.g, s.h - o.h)

    def __mul__(s, o):
        if not isinstance(o, HD):
            o = np.asarray(o, dtype=float)
            if o.ndim == 0:
                return HD(s.v * o, s.g * o, s.h * o)
            return HD(s.v * o, s.g * o[:, None], s.h * o[:, None, None])
        return HD(
            s.v * o.v,
            s.g * o.v[:, None] + o.g * s.v[:, None],
            s.h * o.v[:, None, None]
            + o.h * s.v[:, None, None]
            + np.einsum("ni,nj->nij", s.g, o.g)
            + np.einsum("ni,nj->nij", o.g, s.g),
        )

    def apply(s, f0, f1, f2):
        return HD(
            f0,
            s.g * f1[:, None],
            s.h * f1[:, None, None]
            + np.einsum("ni,nj->nij", s.g, s.g) * f2[:, None, None],
        )

    def recip(s):
        return s.apply(1 / s.v, -1 / s.v**2, 2 / s.v**3)

    def sqrt(s):
        r = np.sqrt(s.v)
        return s.apply(r, 0.5 / r, -0.25 / r**3)

    def __truediv__(s, o):
        return s * o.recip()


def becke_size_a(z):
    rr = np.array([[BRAGG[i] / BRAGG[j] for j in z] for i in z])
    u = (rr - 1) / (rr + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = u / (u * u - 1)
    return np.clip(np.nan_to_num(a), -0.5, 0.5)


def smooth3(nu):
    """f(f(f(nu))), with first and second derivative wrt nu."""
    f, d1, d2 = nu, np.ones_like(nu), np.zeros_like(nu)
    for _ in range(3):
        p0, p1, p2 = 1.5 * f - 0.5 * f**3, 1.5 * (1 - f * f), -3 * f
        f, d1, d2 = p0, p1 * d1, p2 * d1 * d1 + p1 * d2
    return f, d1, d2


def atomic_quadrature(mol):
    ag = gen_grid.gen_atomic_grids(
        mol, atom_grid=(NRAD, NANG), radi_method=radi.treutler, prune=None
    )
    pts = []
    for ia in range(mol.natm):
        c, w = ag[mol.atom_symbol(ia)]
        pts.append((ia, c, w))
    return pts


def moving_weights(R, z, home, u, scale, m, ride=True):
    """Becke weight of each point's home atom, as HD in the 3N nuclear coordinates (or plain)."""
    nat = R.shape[0]
    n = u.shape[0]
    a = becke_size_a(z)

    def coord(atom_c):  # HD for a nuclear coordinate R_c,k (k = 0..2): grad unit vector
        out = []
        for k in range(3):
            g = np.zeros((n, m))
            g[:, 3 * atom_c + k] = 1.0
            out.append(HD(np.full(n, R[atom_c, k]), g, np.zeros((n, m, m))))
        return out

    Rc = [coord(c) for c in range(nat)]
    # point position r = R_home + u (HD per point)
    r = []
    for k in range(3):
        v = R[home, k] + u[:, k]
        g = np.zeros((n, m))
        if ride:
            g[np.arange(n), 3 * home + k] = 1.0
        r.append(HD(v, g, np.zeros((n, m, m))))
    dist = []
    for c in range(nat):
        s = None
        for k in range(3):
            d = r[k] - Rc[c][k]
            s = d * d if s is None else s + d * d
        dist.append(s.sqrt())
    P = [HD.const(np.ones(n), m) for _ in range(nat)]
    for A in range(nat):
        for B in range(nat):
            if A == B:
                continue
            s = None
            for k in range(3):
                d = Rc[A][k] - Rc[B][k]
                s = d * d if s is None else s + d * d
            RAB = s.sqrt()
            mu = (dist[A] - dist[B]) / RAB
            nu = mu + (HD.const(np.ones(n), m) - mu * mu) * a[A, B]
            f, f1, f2 = smooth3(nu.v)
            sAB = nu.apply(0.5 * (1 - f), -0.5 * f1, -0.5 * f2)
            P[A] = P[A] * sAB
    T = P[0]
    for c in range(1, nat):
        T = T + P[c]
    # select home cell function per point
    Ph = HD(
        np.choose(home, [p.v for p in P]),
        np.choose(home[:, None], [p.g for p in P]),
        np.choose(home[:, None, None], [p.h for p in P]),
    )
    return (Ph / T) * scale


def build_grid(mol):
    home, u, scale = [], [], []
    for ia, c, w in atomic_quadrature(mol):
        home += [ia] * len(w)
        u.append(c)
        scale.append(w)
    return np.array(home), np.vstack(u), np.concatenate(scale)


def xc_derivs(rho):
    exc, vxc, fxc, _ = dft.libxc.eval_xc(XC, rho, deriv=2)
    return rho * exc, vxc[0], fxc[0]


def hessian_moving(mol, D, mode="moving", mutant=None):
    """AD Hessian of E_xc at fixed AO density D. mode: moving | fixed."""
    R = mol.atom_coords()
    nat = mol.natm
    m = 3 * nat
    home, u, scale = build_grid(mol)
    pts = R[home] + u
    n = len(scale)
    if mutant == "per_point_noride":
        W = moving_weights(R, mol.atom_charges(), home, u, scale, m, ride=False)
    elif mode == "moving" and mutant != "frozen_weights":
        W = moving_weights(R, mol.atom_charges(), home, u, scale, m)
    else:
        W = HD.const(
            scale * moving_weights(R, mol.atom_charges(), home, u, np.ones(n), m).v, m
        )
    ao = mol.eval_gto("GTOval_sph_deriv2", pts)  # (10, n, nao)
    chi, dchi = ao[0], ao[1:4]
    # second derivs xx xy xz yy yz zz
    idx = {(0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 1): 7, (1, 2): 8, (2, 2): 9}
    ddchi = np.empty((3, 3) + chi.shape)
    for (i, j), k in idx.items():
        ddchi[i, j] = ddchi[j, i] = ao[k]
    aoatom = np.zeros(chi.shape[1], dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        aoatom[p0:p1] = ia
    # e[g, mu, C] = delta(C, home g) - delta(C, atom mu)
    e = np.zeros((n, chi.shape[1], nat))
    e[np.arange(n)[:, None], np.arange(chi.shape[1])[None, :], aoatom[None, :]] -= 1.0
    if mode == "moving" and mutant not in ("no_point_motion", "per_point_noride"):
        e[np.arange(n), :, home] += 1.0
    # chi^x  (n, nao, m) and chi^xy (n, nao, m, m)
    cg = np.einsum("gmC,agm->gmCa", e, dchi).reshape(n, -1, m)
    ch = np.einsum("gmC,gmD,abgm->gmCaDb", e, e, ddchi).reshape(n, -1, m, m)
    P = chi @ D
    rho = np.einsum("gm,gm->g", P, chi)
    rg = 2 * np.einsum("gm,gmx->gx", P, cg)
    rh = 2 * np.einsum("gm,gmxy->gxy", P, ch) + 2 * np.einsum(
        "gmx,mn,gny->gxy", cg, D, cg
    )
    rhoHD = HD(rho, rg, rh)
    f0, f1, f2 = xc_derivs(rho)
    F = rhoHD.apply(f0, f1, f2)
    if mutant == "drop_w2":
        W = HD(W.v, W.g, np.zeros_like(W.h))
    if mutant == "drop_w1_cross":
        E = HD(
            W.v * F.v,
            W.g * F.v[:, None] + F.g * W.v[:, None],
            W.h * F.v[:, None, None] + F.h * W.v[:, None, None],
        )
    else:
        E = W * F
    if mutant in ("per_point", "per_point_noride"):
        return E, home
    return E.v.sum(), E.g.sum(0), E.h.sum(0)


def plain_becke(R, z, home, pts):
    """Plain-numpy Becke home weight (no AD), the independent construction for check B."""
    a = becke_size_a(z)
    nat = len(R)
    r = np.linalg.norm(pts[:, None, :] - R[None, :, :], axis=2)
    P = np.ones((len(pts), nat))
    for A in range(nat):
        for B in range(nat):
            if A != B:
                mu = (r[:, A] - r[:, B]) / np.linalg.norm(R[A] - R[B])
                nu = mu + a[A, B] * (1 - mu * mu)
                P[:, A] *= 0.5 * (1 - smooth3(nu)[0])
    return P[np.arange(len(pts)), home] / P.sum(1)


def energy_plain(mol, D):
    """Moving-grid E_xc with no AD: rebuild grid + weights at mol's geometry."""
    R = mol.atom_coords()
    home, u, scale = build_grid(mol)
    W = scale * plain_becke(R, mol.atom_charges(), home, R[home] + u)
    chi = mol.eval_gto("GTOval_sph", R[home] + u)
    rho = np.einsum("gm,mn,gn->g", chi, D, chi)
    return np.dot(W, xc_derivs(rho)[0])


def displaced(mol, coord, h):
    R = mol.atom_coords().copy()
    R[coord // 3, coord % 3] += h
    return mol.set_geom_(R, unit="Bohr", inplace=False)


def main():
    mol = gto.M(
        atom="O 0 0.04 0.21; H 0.06 1.44 -0.89; H -0.04 -1.42 -0.85",
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    mf = dft.RKS(mol, xc=XC).run(conv_tol=1e-11)
    D = mf.make_rdm1()  # fixed AO density throughout
    m = 3 * mol.natm

    e0, g_mov, H_mov = hessian_moving(mol, D, "moving")
    _, g_fix, H_fix = hessian_moving(mol, D, "fixed")
    H_wr = H_mov - H_fix
    print(f"E_xc = {e0:.12f}  (plain {energy_plain(mol, D):.12f})")
    print(
        f"max|H_moving| = {abs(H_mov).max():.3e}  max|H_moving - H_fixed| (grid response) = {abs(H_wr).max():.3e}"
    )

    # A. FD of the analytic moving-grid gradient
    import os

    for h in () if os.environ.get("QUICK") else (1e-3, 5e-4):
        Hfd = np.zeros((m, m))
        for c in range(m):
            gp = hessian_moving(displaced(mol, c, h), D, "moving")[1]
            gm = hessian_moving(displaced(mol, c, -h), D, "moving")[1]
            Hfd[:, c] = (gp - gm) / (2 * h)
        print(
            f"A. h={h:.0e}: max|H_AD - FD(grad)| = {abs(H_mov - Hfd).max():.3e}   "
            f"max|H_fixedgrid - FD(grad)| = {abs(H_fix - Hfd).max():.3e}"
        )

    # B. second differences of the plain energy (Richardson h, h/2)
    def d2(h):
        H = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):

                def E(si, sj):
                    mm = displaced(displaced(mol, i, si * h), j, sj * h)
                    return energy_plain(mm, D)

                H[i, j] = H[j, i] = (E(1, 1) - E(1, -1) - E(-1, 1) + E(-1, -1)) / (
                    4 * h * h
                )
        return H

    if not os.environ.get("QUICK"):
        Hb1, Hb2 = d2(4e-3), d2(2e-3)
        Hb = (4 * Hb2 - Hb1) / 3
        print(
            f"B. max|H_AD - d2E (Richardson 4e-3/2e-3)| = {abs(H_mov - Hb).max():.3e}  (h=2e-3 alone {abs(H_mov - Hb2).max():.3e})"
        )

    # C. translation sum rule
    def tsum(H):
        return max(abs(H[a::3, :].sum(0)).max() for a in range(3))

    print(
        f"C. translation sum rule max|sum_B H[(B,a),:]|: moving {tsum(H_mov):.3e}  fixed {tsum(H_fix):.3e}"
    )
    print(
        f"   gradient sum rule: moving {abs(g_mov.reshape(-1, 3).sum(0)).max():.3e}  fixed {abs(g_fix.reshape(-1, 3).sum(0)).max():.3e}"
    )

    # D. mutants
    for mut in ("drop_w2", "drop_w1_cross", "no_point_motion", "frozen_weights"):
        Hm = hessian_moving(mol, D, "moving", mutant=mut)[2]
        print(
            f"D. mutant {mut:16s}: max|H_mut - H_AD| = {abs(Hm - H_mov).max():.3e}   sum rule {tsum(Hm):.3e}"
        )
    # E. translation reconstruction: only lab-fixed (non-home) derivatives are needed.
    #    Per home atom h, take the Hessian of its points with the point FROZEN (no point
    #    motion; weights differentiated at fixed r) over atoms B,C != h, and fill the h
    #    row/column by translation invariance.  Must equal the full moving-grid Hessian.
    E_np, home = hessian_moving(mol, D, "moving", mutant="per_point")
    E_np_frozen_pts = hessian_moving_frozen_points(mol, D)
    Hrec = np.zeros((m, m))
    for h in range(mol.natm):
        G = E_np_frozen_pts[home == h].sum(0)
        keep = [x for x in range(m) if x // 3 != h]
        T = np.zeros((m, len(keep)))
        for j, x in enumerate(keep):
            T[x, j] = 1.0
            T[3 * h + x % 3, j] = -1.0
        Hrec += T @ G[np.ix_(keep, keep)] @ T.T
    print(
        f"E. max|H_moving - sum_h T_h G_h(non-home, frozen points) T_h^T| = {abs(Hrec - H_mov).max():.3e}"
    )
    sys.stdout.flush()


def hessian_moving_frozen_points(mol, D):
    """Per-point Hessians with weights differentiated but points NOT riding (lab-fixed r)."""
    return _per_point(mol, D, ride=False)


def _per_point(mol, D, ride):
    E, _ = hessian_moving(
        mol, D, "moving", mutant="per_point_noride" if not ride else "per_point"
    )
    return E.h


if __name__ == "__main__":
    main()
