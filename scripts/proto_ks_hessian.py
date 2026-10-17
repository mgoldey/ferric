"""Numpy prototype of the fixed-grid closed-shell KS Hessian XC pieces (issue #407).

Scope: the formulas that the Rust `ks_fixed_grid` GGA path implements, checked
against PySCF (`rks.Hessian`, `grid_response=False`) BEFORE any Rust is written.

Energy:  E_xc = sum_g w_g f(rho, sigma),  sigma = |grad rho|^2,  grid FIXED in space.

Notation (electron-coordinate AO derivatives chi_{mu,a}; nuclear derivative of an AO on
atom B is minus its electron derivative):
    rho_x         = -2 sum_{mu in B} chi_{mu,a} P_mu                      (P = D chi)
    g_{gam,x}     = -2 sum_{mu in B} [chi_{mu,gam a} P_mu + chi_{mu,a} Q_{mu gam}]   (Q_gam = D chi_,gam)
    rho_xy        = 2 d_BC sum_{mu in B} chi_{mu,ab} P_mu + 2 sum_{mu in B, nu in C} D chi_{mu,a} chi_{nu,b}
    g_{gam,xy}    = 2 d_BC sum_{mu in B}[chi_{mu,gam ab} P_mu + chi_{mu,ab} Q_{mu gam}]
                    + 2 sum_{mu in B}[chi_{mu,gam a} (D chi_{,b})_{mu,C} + chi_{mu,a} (D chi_{,gam b})_{mu,C}]
Weights (fixed D):   F = w f_rho,   G_gam = 2 w f_sigma g_gam
    V_mn[F,G]         = sum_g F chi_m chi_n + G_gam (chi_m,gam chi_n + chi_m chi_n,gam)
Response of the weights to a perturbation (rho1, g1):
    dF   = w (f_rr rho1 + 2 f_rs g.g1)
    dG_gam = 2 w [ (f_rs rho1 + 2 f_ss g.g1) g_gam + f_s g1_gam ]
Then
    skeleton   H_xy = sum_g [ rho_x dF[y] + g_x . dG[y] ] + sum_g [ F rho_xy + G . g_xy ]
    V^x (fixed D) = V[dF[x], dG[x]] + AO-derivative rows/cols of V[F,G]
    kernel V1[D1] = V[dF[D1], dG[D1]]   with rho1 = sum D1 chi chi, g1 = 2 sum D1 chi_,gam chi

Run:  OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/proto_ks_hessian.py
Exactness anchors checked: (i) LDA through the GGA path == LDA path; (ii) every piece vs
FD of the energy / Vxc on the same frozen grid; (iii) the three pieces monkeypatched into
PySCF's rks.Hessian reproduce its total Hessian; (iv) the hybrid scaling a -> 0.
"""

from __future__ import annotations

import sys

import numpy as np
from pyscf import dft, gto
from pyscf.dft import libxc, numint
from pyscf.hessian import rks as rks_hess

# index of electron derivatives in pyscf eval_ao(deriv=3): [0]=1, [1:4]=x,y,z, 4:10 = xx.., 10:20 = xxx..
D1 = {0: 1, 1: 2, 2: 3}
_D2 = {(0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 1): 7, (1, 2): 8, (2, 2): 9}
_D3_LIST = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 1, 1),
    (0, 1, 2),
    (0, 2, 2),
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 2),
]
D3 = {k: 10 + i for i, k in enumerate(_D3_LIST)}


def d2(a, b):
    return _D2[tuple(sorted((a, b)))]


def d3(a, b, c):
    return D3[tuple(sorted((a, b, c)))]


class Xc:
    """Per-point f derivatives (w.r.t. rho, sigma) from libxc, unpolarized."""

    def __init__(self, name):
        self.name = name
        self.gga = numint.NumInt()._xc_type(name) != "LDA"

    def eval(self, rho, grad):
        r4 = np.vstack([rho, grad.T]) if self.gga else rho
        exc, vxc, fxc, _ = libxc.eval_xc(self.name, r4, spin=0, deriv=2)
        e = exc * rho  # per volume
        fr = vxc[0]
        fs = vxc[1] if self.gga else np.zeros_like(rho)
        frr = fxc[0]
        frs = fxc[1] if self.gga else np.zeros_like(rho)
        fss = fxc[2] if self.gga else np.zeros_like(rho)
        return e, fr, fs, frr, frs, fss


def ao_tables(mol, pts):
    return numint.eval_ao(mol, pts, deriv=3)  # (20, ng, nao) in this pyscf layout


class Pieces:
    def __init__(self, mol, pts, w, dm, xc, force_gga=False):
        self.mol, self.pts, self.w, self.dm = mol, pts, w, dm
        self.xc = xc
        self.gga = xc.gga or force_gga
        ao = ao_tables(mol, pts)  # (20, ng, nao)
        self.ao = np.ascontiguousarray(ao.transpose(0, 2, 1))  # (20, nao, ng)
        self.nao = mol.nao
        self.natm = mol.natm
        sl = mol.aoslice_by_atom()
        self.ao_atom = np.zeros(self.nao, int)
        for a, (_, _, p0, p1) in enumerate(sl):
            self.ao_atom[p0:p1] = a
        self.sl = sl
        A = self.ao
        self.P = dm @ A[0]
        self.rho = np.einsum("mg,mg->g", A[0], self.P)
        self.Q = np.stack([dm @ A[D1[c]] for c in range(3)])  # (3, nao, ng)
        self.g = 2 * np.einsum("cmg,mg->cg", A[1:4], self.P)  # (3, ng)
        e, fr, fs, frr, frs, fss = xc.eval(self.rho, self.g.T)
        if not self.gga:
            fs = frs = fss = np.zeros_like(fr)
            self.g = np.zeros_like(self.g)
        self.e, self.fr, self.fs, self.frr, self.frs, self.fss = (
            e,
            fr,
            fs,
            frr,
            frs,
            fss,
        )
        self.F = w * fr
        self.G = 2 * w * fs * self.g  # (3, ng)

    # -- weights response ------------------------------------------------
    def dweights(self, rho1, g1):
        """rho1 (..., ng), g1 (..., 3, ng) -> dF (..., ng), dG (..., 3, ng)."""
        w = self.w
        gg1 = np.einsum("cg,...cg->...g", self.g, g1)
        dF = w * (self.frr * rho1 + 2 * self.frs * gg1)
        dG = (
            2
            * w
            * (
                (self.frs * rho1 + 2 * self.fss * gg1)[..., None, :] * self.g
                + self.fs * g1
            )
        )
        return dF, dG

    def vmat(self, F, G):
        A = self.ao
        v = (A[0] * F) @ A[0].T
        if self.gga:
            for c in range(3):
                t = (A[D1[c]] * G[c]) @ A[0].T
                v += t + t.T
        return v

    # -- nuclear derivatives of rho and grad rho --------------------------
    def rho_x(self):
        A = self.ao
        out = np.zeros((self.natm, 3, len(self.w)))
        for a in range(3):
            prod = A[D1[a]] * self.P
            for mu in range(self.nao):
                out[self.ao_atom[mu], a] -= 2 * prod[mu]
        return out

    def g_x(self):
        A = self.ao
        out = np.zeros((self.natm, 3, 3, len(self.w)))  # atom, alpha, gamma, g
        for a in range(3):
            for gam in range(3):
                t = A[d2(gam, a)] * self.P + A[D1[a]] * self.Q[gam]
                for mu in range(self.nao):
                    out[self.ao_atom[mu], a, gam] -= 2 * t[mu]
        return out

    def skeleton(self):
        A, dm, nat = self.ao, self.dm, self.natm
        n3 = 3 * nat
        rx = self.rho_x().reshape(n3, -1)
        gx = self.g_x().reshape(n3, 3, -1)
        dF, dG = self.dweights(rx, gx)
        H = rx @ dF.T + np.einsum("xcg,ycg->xy", gx, dG)
        # F rho_xy + G . g_xy
        for B in range(nat):
            sB = slice(self.sl[B][2], self.sl[B][3])
            for C in range(nat):
                sC = slice(self.sl[C][2], self.sl[C][3])
                for a in range(3):
                    for b in range(3):
                        val = 0.0
                        if B == C:
                            val += 2 * np.einsum(
                                "mg,mg,g->", A[d2(a, b)][sB], self.P[sB], self.F
                            )
                        val += 2 * np.einsum(
                            "mg,mg,g->",
                            A[D1[a]][sB],
                            (dm[sB][:, sC] @ A[D1[b]][sC]),
                            self.F,
                        )
                        if self.gga:
                            for gam in range(3):
                                Gg = self.G[gam]
                                if B == C:
                                    val += 2 * np.einsum(
                                        "mg,mg,g->",
                                        A[d3(gam, a, b)][sB],
                                        self.P[sB],
                                        Gg,
                                    )
                                    val += 2 * np.einsum(
                                        "mg,mg,g->",
                                        A[d2(a, b)][sB],
                                        self.Q[gam][sB],
                                        Gg,
                                    )
                                val += 2 * np.einsum(
                                    "mg,mg,g->",
                                    A[d2(gam, a)][sB],
                                    dm[sB][:, sC] @ A[D1[b]][sC],
                                    Gg,
                                )
                                val += 2 * np.einsum(
                                    "mg,mg,g->",
                                    A[D1[a]][sB],
                                    dm[sB][:, sC] @ A[d2(gam, b)][sC],
                                    Gg,
                                )
                        H[3 * B + a, 3 * C + b] += val
        return H

    def first_order_fock(self):
        A, nat = self.ao, self.natm
        rx = self.rho_x()
        gx = self.g_x()
        out = np.zeros((nat, 3, self.nao, self.nao))
        for B in range(nat):
            for a in range(3):
                dF, dG = self.dweights(rx[B, a], gx[B, a])
                out[B, a] = self.vmat(dF, dG)
        # AO-derivative part: -T_mu rows and columns, T_mu,n = sum F chi_mu,a chi_n + G(chi_mu,gam a chi_n + chi_mu,a chi_n,gam)
        for a in range(3):
            T = (A[D1[a]] * self.F) @ A[0].T
            if self.gga:
                for gam in range(3):
                    T += (A[d2(gam, a)] * self.G[gam]) @ A[0].T + (
                        A[D1[a]] * self.G[gam]
                    ) @ A[D1[gam]].T
            for mu in range(self.nao):
                B = self.ao_atom[mu]
                out[B, a, mu, :] -= T[mu]
                out[B, a, :, mu] -= T[mu]
        return out

    def kernel(self, d1):
        A = self.ao
        rho1 = np.einsum("mg,mg->g", A[0], d1 @ A[0])
        g1 = (
            2 * np.stack([np.einsum("mg,mg->g", A[D1[c]], d1 @ A[0]) for c in range(3)])
            if self.gga
            else np.zeros((3, len(self.w)))
        )
        dF, dG = self.dweights(rho1, g1)
        return self.vmat(dF, dG)

    def energy_vxc(self):
        return (self.w * self.e).sum(), self.vmat(self.F, self.G)


def pyscf_grid(mol, level=1):
    g = dft.gen_grid.Grids(mol)
    g.level = level
    g.prune = None
    g.build()
    return g.coords.copy(), g.weights.copy()


def mol_at(atoms, basis, disp=None):
    m = gto.M(atom=atoms, basis=basis, unit="Bohr", verbose=0)
    if disp is not None:
        c = m.atom_coords().copy()
        c[disp[0], disp[1]] += disp[2]
        m = gto.M(
            atom=[(m.atom_symbol(i), tuple(c[i])) for i in range(m.natm)],
            basis=basis,
            unit="Bohr",
            verbose=0,
        )
    return m


def check(label, val, tol):
    status = "ok" if val < tol else "FAIL"
    print(f"  {label:58s} {val:10.2e}  (tol {tol:.0e}) {status}")
    return val < tol


def run(xcname, basis="sto-3g", force_gga=False):
    atoms = [("O", (0.0, 0.0, 0.2)), ("H", (0.0, 1.45, -0.9)), ("H", (0.1, -1.4, -1.0))]
    mol = mol_at(atoms, basis)
    xc = Xc(xcname)
    mf = dft.RKS(mol)
    mf.xc = xcname
    mf.conv_tol = 1e-12
    mf.grids.level = 1
    mf.grids.prune = None
    mf.kernel()
    pts, w = mf.grids.coords.copy(), mf.grids.weights.copy()
    dm = mf.make_rdm1()
    P = Pieces(mol, pts, w, dm, xc, force_gga)
    ok = True
    print(f"[{xcname} / {basis}] gga-path={P.gga}")

    # (ii) FD anchors on the frozen grid
    h = 1e-4
    nat = mol.natm
    n3 = 3 * nat

    def vxc_at(B, a, s):
        m = mol_at(atoms, basis, (B, a, s))
        return Pieces(m, pts, w, dm, xc, force_gga).energy_vxc()

    skel = P.skeleton()
    fo = P.first_order_fock()
    # gradient dE/dx = sum w f' rho_x ... from FD of E; first-order Fock from FD of V
    fd_fo = np.zeros_like(fo)
    fd_g = np.zeros((nat, 3))
    fd_skel = np.zeros((n3, n3))
    for B in range(nat):
        for a in range(3):
            ep, vp = vxc_at(B, a, h)
            em, vm = vxc_at(B, a, -h)
            fd_fo[B, a] = (vp - vm) / (2 * h)
            fd_g[B, a] = (ep - em) / (2 * h)

    # skeleton = d/dy of dE/dx at fixed D: FD the analytic gradient
    def grad_at(B, a, s):
        m = mol_at(atoms, basis, (B, a, s))
        Pm = Pieces(m, pts, w, dm, xc, force_gga)
        rx = Pm.rho_x()
        gx = Pm.g_x()
        return (
            np.einsum("xag,g->xa", rx, Pm.F) + np.einsum("xacg,cg->xa", gx, Pm.G)
        ).reshape(-1)

    gan = np.einsum("xag,g->xa", P.rho_x(), P.F) + np.einsum(
        "xacg,cg->xa", P.g_x(), P.G
    )
    ok &= check("gradient analytic vs FD(E)", abs(gan - fd_g).max(), 1e-7)
    for B in range(nat):
        for a in range(3):
            fd_skel[3 * B + a] = (grad_at(B, a, h) - grad_at(B, a, -h)) / (2 * h)
    ok &= check(
        "skeleton vs FD(analytic gradient at fixed D)", abs(skel - fd_skel).max(), 1e-5
    )  # FD stencil floor; pyscf agreement below is the authority
    ok &= check("first-order Fock vs FD(Vxc)", abs(fo - fd_fo).max(), 1e-7)
    d1 = np.random.default_rng(1).standard_normal(dm.shape) * 1e-2
    d1 = d1 + d1.T
    eps = 1e-5
    Pp = Pieces(mol, pts, w, dm + eps * d1, xc, force_gga).energy_vxc()[1]
    Pm_ = Pieces(mol, pts, w, dm - eps * d1, xc, force_gga).energy_vxc()[1]
    ok &= check(
        "kernel vs FD(Vxc along D1)",
        abs(P.kernel(d1) - (Pp - Pm_) / (2 * eps)).max(),
        1e-7,
    )

    # (iii) against PySCF analytic pieces, grid_response=False
    hobj = mf.Hessian()
    hobj.grid_response = False
    mo_c, mo_o = mf.mo_coeff, mf.mo_occ
    ref_h1 = rks_hess._get_vxc_deriv1(hobj, mo_c, mo_o, 2000)
    ok &= check("V^x vs pyscf _get_vxc_deriv1", abs(fo - ref_h1).max(), 1e-8)
    mine = skel.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3)
    dm0 = dm
    veff_diag = rks_hess._get_vxc_diag(hobj, mo_c, mo_o, 2000)
    vxc2 = rks_hess._get_vxc_deriv2(hobj, mo_c, mo_o, 2000)
    ref_xc = np.zeros((nat, nat, 3, 3))
    for i0 in range(nat):
        p0, p1 = mol.aoslice_by_atom()[i0][2:]
        ref_xc[i0, i0] += (
            np.einsum("xypq,pq->xy", veff_diag[:, :, p0:p1], dm0[p0:p1]) * 2
        )
        for j0 in range(i0 + 1):
            q0, q1 = mol.aoslice_by_atom()[j0][2:]
            ref_xc[i0, j0] += (
                np.einsum("xypq,pq->xy", vxc2[i0][:, :, q0:q1], dm0[q0:q1]) * 2
            )
        for j0 in range(i0):
            ref_xc[j0, i0] = ref_xc[i0, j0].T
    ok &= check(
        "XC skeleton vs pyscf (_get_vxc_diag+_get_vxc_deriv2)",
        abs(mine - ref_xc).max(),
        1e-8,
    )
    vind = mf.gen_response(mo_c, mo_o, hermi=1)
    hyb = mf._numint.rsh_and_hybrid_coeff(xcname, spin=0)[2]
    j, k = mf.get_jk(mol, d1)
    ref_kernel = vind(d1) - (j - 0.5 * hyb * k)
    ok &= check(
        "kernel vs pyscf gen_response (minus J,aK)",
        abs(P.kernel(d1) - ref_kernel).max(),
        1e-8,
    )

    # total Hessian: pyscf with its XC skeleton zeroed + mine, V^x swapped for mine
    ref_total = hobj.kernel()
    keep = (rks_hess._get_vxc_diag, rks_hess._get_vxc_deriv2, rks_hess._get_vxc_deriv1)
    rks_hess._get_vxc_diag = lambda *a, **k: np.zeros_like(veff_diag)
    rks_hess._get_vxc_deriv2 = lambda *a, **k: np.zeros_like(vxc2)
    rks_hess._get_vxc_deriv1 = lambda *a, **k: fo
    try:
        h2 = mf.Hessian()
        h2.grid_response = False
        patched = h2.kernel() + mine
    finally:
        rks_hess._get_vxc_diag, rks_hess._get_vxc_deriv2, rks_hess._get_vxc_deriv1 = (
            keep
        )
    ok &= check(
        f"total Hessian (hyb a={hyb:.2f}): my skeleton+Vx vs pyscf",
        abs(patched - ref_total).max(),
        1e-8,
    )
    return ok


if __name__ == "__main__":
    allok = True
    # (i) anchor: LDA through the GGA path equals the LDA path
    allok &= run("LDA_X,LDA_C_VWN")
    allok &= run("LDA_X,LDA_C_VWN", force_gga=True)
    allok &= run("HYB_LDA_XC_LDA0")
    allok &= run("PBE")
    allok &= run("PBE0")
    allok &= run("B3LYP", basis="6-31g")
    sys.exit(0 if allok else 1)
