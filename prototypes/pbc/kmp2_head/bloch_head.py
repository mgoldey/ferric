"""k-MP2 q = 0 head, candidate H5 in the b -> 0 limit: Bloch-paired (true) vs fixed-k head vectors, ANALYTIC.

The mesh ERIs approach, as ka -> ki, the limit of M_ia(k, q) = C_i(k)^H P^{k,k+q}(q) C_a(k+q) (pbc_kcorr convention,
P^{k,k'}(K) = sum_L e^{ik'.L} FT[phi_m phi_n(. - L)](K), which depends on k' only).  Its q-derivative is
    g_true_ia = C_i^H (A + dS) C_a + C_i^H S dC_a,      A = dP^{kk}/dK (the fixed-k head), dS = grad_k S(k)
and, from F C = S C eps (non-degenerate i != a):  C_i^H S dC_a = C_i^H (dF - eps_a dS) C_a / (eps_a - eps_i), so
    Delta_ia = g_true_ia - g_fixed_ia = C_i^H (dF - eps_i dS) C_a / (eps_a - eps_i)
    Delta_ai =                          C_a^H (dF - eps_a dS) C_i / (eps_i - eps_a)
dS exactly (lattice sums of the molecular overlap); dF by Fourier (Wannier) interpolation of the MESH Fock
F(k) = S C diag(eps_none) C^H S (the saved SCF, exact on the mesh), Wigner-Seitz L set, boundary split evenly.
Iteration H5 did the same with FINITE b (central difference over the mesh spacing); this is its b -> 0 limit.

Artifact hypotheses (stated before running)
  * Fourier-derivative aliasing: the same Fourier derivative applied to the mesh S(k) must reproduce the exact dS;
    its error bounds the dF error (F decays no slower than S here).  If the check is > ~10% the dF route is void.
  * Gauge/sign error in Delta: the analytic (true - fixed) head difference must agree with the finite-b rows
    (true_b_* - fixk_b_*, npz) to O(b^2) - same sign, magnitude approached as b shrinks from n = 3 to 4.  A sign or
    conj error gives the opposite sign or an n-independent mismatch.
  * The lattice (Phi) weighting of the quadratic head is done with the full cubic tensor Phi_abcd
    (Phi_aaaa = -0.171599, Phi_aabb = (1 - 3 Phi_aaaa)/6), so x/y components of g_true are weighted correctly;
    anchor: for the fixed-k g it reproduces the z-only lambda-fit number (cheap_account: quad lattice, direct part).
Physics expectation if H5 carries the residual: E3_true - E3 ~ +6.2e-5 x n^-3 (the direct-channel residual of
cheap_account), stable between n = 3 and 4, and shrinking rapidly with a (dispersion).  If not, |E3_true - E3| << that.
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc")
sys.path.insert(0, "/home/matt/qc/ferric-pbc/prototypes/pbc/kmp2_head")
from pbc_gamma import Cell  # noqa: E402
from pbc_kpts import mp_mesh  # noqa: E402
from cheap_account import kmp2_split, load, st_from  # noqa: E402
from vao_build import h2_atoms  # noqa: E402

PHI4 = -0.17159902
PHI22 = (1 - 3 * PHI4) / 6


def phi_tensor():
    T = np.zeros((3, 3, 3, 3))
    for a, b, c, d in itertools.product(range(3), repeat=4):
        idx = sorted((a, b, c, d))
        if idx[0] == idx[1] == idx[2] == idx[3]:
            T[a, b, c, d] = PHI4
        elif idx[0] == idx[1] and idx[2] == idx[3]:
            T[a, b, c, d] = PHI22
    return T


def lattice_S(cell, kpts, rcut=22.0):
    L1 = cell.translations(rcut)
    sm = cell.supermol(L1)
    nao, nb0 = cell.mol.nao, cell.mol.nbas
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(nao, len(L1), nao)
    ph = np.exp(1j * kpts @ L1.T)
    S = np.einsum("kL,mLn->kmn", ph, S_L)
    dS = np.einsum("kL,Lx,mLn->kmnx", ph, 1j * L1, S_L)
    return S, dS


def fourier_grad(Xk, cell, m):
    """grad_k of the Wigner-Seitz Fourier interpolant of mesh data Xk (Nk, ...), evaluated on the mesh."""
    ints, kpts = mp_mesh(cell, (m, m, m))
    rng = range(-(m // 2), m // 2 + 1)
    out = np.zeros(Xk.shape + (3,), complex)
    for mv in itertools.product(rng, repeat=3):
        w = np.prod([0.5 if (m % 2 == 0 and abs(x) == m // 2) else 1.0 for x in mv])
        L = np.array(mv, float) @ cell.a
        XL = np.einsum("k,k...->...", np.exp(-1j * kpts @ L), Xk) / len(kpts)
        ph = np.exp(1j * kpts @ L)
        out += w * np.einsum("k,...,x->k...x", ph, XL, 1j * L)
    return out


def head_vectors(z, cell, m, dF_override=None):
    C, eps, A = z["C"], z["eps"], z["A"]
    ints, kpts = mp_mesh(cell, (m, m, m))
    S, dS = lattice_S(cell, kpts)
    Fk = np.einsum("kmp,kp,knp->kmn", np.einsum("kmn,knp->kmp", S, C), eps, np.einsum("kmn,knp->kmp", S, C).conj())
    dF = fourier_grad(Fk, cell, m) if dF_override is None else dF_override
    dS_f = fourier_grad(S, cell, m)
    ci, ca = C[:, :, 0], C[:, :, 1]
    gfix_ov = np.einsum("km,kmnx,kn->kx", ci.conj(), A, ca)
    gfix_vo = np.einsum("km,kmnx,kn->kx", ca.conj(), A, ci)
    ei, ea = eps[:, 0], eps[:, 1]
    Xi = dF - ei[:, None, None, None] * dS
    Xa = dF - ea[:, None, None, None] * dS
    d_ov = np.einsum("km,kmnx,kn->kx", ci.conj(), Xi, ca) / (ea - ei)[:, None]
    d_vo = np.einsum("km,kmnx,kn->kx", ca.conj(), Xa, ci) / (ei - ea)[:, None]
    info = dict(S_err=abs(S - z["P0"]).max(), dS_alias=abs(dS_f - dS).max() / abs(dS).max(),
                d_rel=np.sqrt((abs(d_ov) ** 2).sum() / (abs(gfix_ov) ** 2).sum()))
    return gfix_ov, gfix_vo, gfix_ov + d_ov, gfix_vo + d_vo, info


def head_energies(st, gov, gvo, eo, ev, vol, PHI):
    """(L_cub, Q_cub_direct, Q_lat_direct) per cell from MO head vectors (1 occ, 1 vir per k)."""
    Nk = st["Nk"]
    c = 4 * np.pi / vol
    eo_, ev_ = np.array(eo)[:, 0], np.array(ev)[:, 0]
    D = eo_[:, None] + eo_[None, :] - ev_[:, None] - ev_[None, :]  # (ki, kj), ka = ki, kb = kj
    G = c * np.einsum("ia,jb->ijab", gov, gvo.conj())  # H(u) = u_a u_b G_ab
    Hc = np.einsum("ijaa->ij", G) / 3
    Qc = 2 * np.sum(abs(Hc) ** 2 / D) / Nk**3
    Ql = 2 * np.sum(np.einsum("ijab,ijcd,abcd->ij", G, G.conj(), PHI).real / D) / Nk**3
    # linear part via the exact quadratic of the cubic head (all channels)
    def e_with(lam):
        new = dict(st)
        V = st["V"].copy()
        for x in range(Nk):
            V[x, :, x, 0, 0, 0, 0] += lam * Hc[x]
        new["V"] = V
        return sum(kmp2_split(new, eo, ev))
    s0 = sum(kmp2_split(st, eo, ev))
    e1, e2 = e_with(1.0) - s0, e_with(2.0) - s0
    Qall = (e2 - 2 * e1) / 2
    return e1 - Qall, Qc, Ql, Qall


def main():
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    orient = sys.argv[2] if len(sys.argv) > 2 else "z"
    ms = [int(x) for x in sys.argv[3:]] or [3, 4]
    suf = "" if orient == "z" else f"_{orient}"
    cell = Cell(np.eye(3) * a, h2_atoms(orient), "sto-3g")
    PHI = phi_tensor()
    for m in ms:
        z, row = load(a, m, suf)
        st = st_from(z, cell, m)
        n3 = m**3
        eps, vm = z["eps"], float(z["vm"])
        eo = [e[:1] - vm for e in eps]
        ev = [e[1:] for e in eps]
        S0 = row["sh"][0]
        F = row["fock"][0] - S0
        gfo, gfv, gto_, gtv, info = head_vectors(z, cell, m)
        print(f"== a={a:g} {orient} n={m}: |S_lat - P0| {info['S_err']:.1e}; Fourier-dS aliasing (rel) "
              f"{info['dS_alias']:.2e}; |Delta_ov|/|g_fix_ov| {info['d_rel']:.4f}")
        print(f"   g_fix_ov k-avg |x,y,z|^2 {np.mean(abs(gfo) ** 2, 0)}; g_true {np.mean(abs(gto_) ** 2, 0)}")
        out = {}
        for tag, (go, gv) in (("fixed", (gfo, gfv)), ("true", (gto_, gtv))):
            L, Qc, Ql, Qall = head_energies(st, go, gv, eo, ev, cell.vol, PHI)
            E3d = S0 + L + Qall + (Ql - Qc) + F  # cubic head (all channels) + lattice re-weighting of the direct quad
            out[tag] = (L, Qc, Ql, Qall, E3d)
            print(f"   {tag:5s} (x n^3): L {L * n3:+.5e} Qcub(direct) {Qc * n3:+.5e} Qcub(all) {Qall * n3:+.5e} "
                  f"Qlat(direct) {Ql * n3:+.5e} | E1 {S0 + L + Qall:.10e} E3d {E3d:.10e}")
        if "fixed" in out:
            print(f"   anchor: fixed-k E1 from MO vectors - row cub1 = {S0 + out['fixed'][0] + out['fixed'][3] - row['cub1'][0]:.1e}")
        dE = out["true"][4] - out["fixed"][4]
        print(f"   H5 analytic: E3d(true) - E3d(fixed) = {dE:+.3e} (x n^3 {dE * n3:+.4e}); split L {(out['true'][0] - out['fixed'][0]) * n3:+.3e}"
              f" Qlat {(out['true'][2] - out['fixed'][2]) * n3:+.3e} Qall-Qc {((out['true'][3] - out['true'][1]) - (out['fixed'][3] - out['fixed'][1])) * n3:+.3e}")
        if "true_b_cub" in row:
            fb = (row["true_b_cub"][0] - row["fixk_b_cub"][0]) * n3
            print(f"   finite-b rows (b = 2pi/(n a)): true_b_cub - fixk_b_cub = {fb:+.4e} x n^-3; analytic cubic-head "
                  f"difference {(out['true'][0] + out['true'][3] - out['fixed'][0] - out['fixed'][3]) * n3:+.4e}")


if __name__ == "__main__":
    main()
