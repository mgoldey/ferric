"""Candidate (e)/(c): orbital relaxation of the mesh SCF under the missing q = 0 Fock head.

The mesh (exxdiv='ewald') Fock differs from the converged one by F_mesh - F_true = +dK/2 (dK = the q = 0 exchange
head beyond v_M S dm S; truefock.py).  E3 corrects ONLY the eigenvalues (first order, frozen orbitals).  But dK has
an ov block at every k with kz != 0 (cheap_account: |dK_ia| ~ 2e-2 x n^-3), so the mesh ORBITALS carry an O(1/Nk)
rotation and the MP2 numerators an O(1/Nk) error that no eigenvalue correction sees.

Construction (all from V_AO, vao_build.py; no second integral pass):
  kernels Jker = V_AO[k, j, k], Kker = V_AO[k, k', k'];  h = T + V_ne;  pbc_kpts.krhf with a jk hook
  SCF_W: the same SCF with the FIXED one-body term dK_AO(k) = S C dK_mo C^H S added to K (true orbitals to O(1/Nk),
         J/K response included; dK itself frozen at the mesh density: its own variation is O(1/Nk^2))
  MP2 from V_AO for any orbitals (pbc_kcorr._accumulate algebra).
Anchors (must pass before any interpretation):
  A1 SCF from V_AO kernels == saved mesh eps (kshift 0) and E_HF;  A2 MP2(saved C, eps) == row 'sh';
  A3 MP2(C, eps - diag(dK)/2) with the A/B dK == row 'fock';  A4 SCF_W eigenvalues - mesh = -diag(dK)/2 + O(Nk^-2)
  A5 lambda-linearity: the relaxation effect with dK scaled by 1/2 is half (first order), so it is an O(1/Nk) term.
Artifact hypothesis: if the ov block were a phase/gauge artefact of the MO basis, SCF_W would give a relaxed
density equal to the mesh one (orbital effect 0 to O(Nk^-2)) — the uncoupled and coupled rotations would both vanish.

usage: relax.py a m [orient]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc")
sys.path.insert(0, "/home/matt/qc/ferric-pbc/prototypes/pbc/kmp2_head")
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, ewald_nn  # noqa: E402
from pbc_kpts import mesh_index, mp_mesh  # noqa: E402
from cheap_account import kmp2_split, load  # noqa: E402
from vao_build import h2_atoms  # noqa: E402

HERE = "/home/matt/qc/ferric-pbc/prototypes/pbc/kmp2_head"


def load_vao(a, m, suf, Nk):
    files = sorted(glob.glob(f"{HERE}/vao_data/vao_a{a:g}_n{m}{suf}_q*.npz"))
    V, Vk, done = None, None, set()
    for f in files:
        z = np.load(f)
        rng = set(range(int(z["iq0"]), min(int(z["iq1"]), Nk)))
        assert not (rng & done), f
        done |= rng
        V = z["V"] if V is None else V + z["V"]
        if 0 in rng:
            Vk = z["Vk"]
    assert done == set(range(Nk)), sorted(set(range(Nk)) - done)
    return V, Vk


def kb_from(cell, m, V, Vk, rcut_1e=22.0):
    n = (m, m, m)
    ints, kpts = mp_mesh(cell, n)
    Nk, nao = len(kpts), cell.mol.nao
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    nb0 = cell.mol.nbas
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(nao, len(L1), nao)
    T_L = sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(nao, len(L1), nao)
    ph1 = np.exp(1j * kpts @ L1.T)
    S = np.einsum("kL,mLn->kmn", ph1, S_L)
    T = np.einsum("kL,mLn->kmn", ph1, T_L)
    ii = np.arange(Nk)
    Jker = V[ii[:, None], ii[None, :], ii[:, None]]
    Kker = V[ii[:, None], ii[None, :], ii[None, :]]
    return dict(S=S, T=T, h=T + Vk, Jker=Jker, Kker=Kker, enn=ewald_nn(cell, 1.0),
                madelung=PK.kmesh_madelung(cell, n), kpts=kpts, ints=ints, n=n, Nk=Nk)


def mo_store(V, C, n, ints):
    """pbc_kcorr store (1 occ, 1 vir per k) for orbitals C from V_AO."""
    Nk = len(ints)
    ii = np.arange(Nk)
    KB = np.array([[[mesh_index(n, ints[a] + ints[b] - ints[c]) for c in ii] for b in ii] for a in ii])
    ci, ca = C[:, :, 0], C[:, :, 1]
    Vmo = np.einsum("xm,zn,xyzl,ys,xyzmnls->xyz", ci.conj(), ca, ca[KB], ci.conj(), V, optimize=True)
    return dict(V=Vmo[..., None, None, None, None], n=n, ints=ints, Nk=Nk)


def scf(kb, extraK=None, kshift=None):
    jk = None
    if extraK is not None:
        def jk(dm):
            J, K = PK.jk_k(kb, dm)
            return J, K + extraK
    e, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-13, kshift=kshift, return_mo=True, jk=jk)
    assert all(int(o.sum()) == 1 and o[0] for o in occ)
    return e, np.array(eps), np.array(C)


def align(C, Cref, S):
    """Phase-align columns of C to Cref (MP2 is gauge invariant; this only makes differences readable)."""
    out = C.copy()
    for k in range(len(C)):
        for p in range(C.shape[2]):
            ov = Cref[k][:, p].conj() @ S[k] @ C[k][:, p]
            out[k][:, p] *= np.conj(ov) / abs(ov)
    return out


def mp2(V, C, eps_sh, kb):
    st = mo_store(V, C, kb["n"], kb["ints"])
    eo = [e[:1] for e in eps_sh]
    ev = [e[1:] for e in eps_sh]
    return sum(kmp2_split(st, eo, ev))


def main():
    a, m = float(sys.argv[1]), int(sys.argv[2])
    orient = sys.argv[3] if len(sys.argv) > 3 else "z"
    suf = "" if orient == "z" else f"_{orient}"
    cell = Cell(np.eye(3) * a, h2_atoms(orient), "sto-3g")
    z, row = load(a, m, suf)
    Nk, n3 = m**3, m**3
    V, Vk = load_vao(a, m, suf, Nk)
    kb = kb_from(cell, m, V, Vk)
    vm = kb["madelung"]
    # A1
    e0, eps0, C0 = scf(kb, kshift=0.0)
    print(f"A1: |E_HF - saved| {abs(e0 - float(z['e_hf'])):.1e}  max|eps - saved| {abs(eps0 - z['eps']).max():.1e}  v_M {vm:.8f} "
          f"(saved {float(z['vm']):.8f})")
    C = z["C"]
    eps_sh = z["eps"].copy()
    eps_sh[:, 0] -= vm
    # A2
    S0 = mp2(V, C, eps_sh, kb)
    print(f"A2: MP2(saved C) - row sh {S0 - row['sh'][0]:.1e}")
    dKs = np.load(f"{HERE}/dK_a{a:g}_n{m}{suf}.npz")
    for tag in ("ana", "true_ewald"):
        dK = dKs[tag]
        epsF = eps_sh - np.einsum("kpp->kp", dK).real / 2
        SF = mp2(V, C, epsF, kb)
        if tag == "ana":
            print(f"A3: MP2(C, eps - dK/2) [A/B] - row fock {SF - row['fock'][0]:.1e}")
        res = {}
        for lam in (1.0, 0.5):
            SC = np.einsum("kmp,kpq->kmq", kb["S"], C)
            dK_ao = lam * np.einsum("kmp,kpq,knq->kmn", SC, dK, SC.conj())
            eW, epsW, CW = scf(kb, extraK=dK_ao)  # kshift = v_M: eps already 'shifted'
            CW = align(CW, C, kb["S"])
            epsF_l = eps_sh - lam * np.einsum("kpp->kp", dK).real / 2
            E_full = mp2(V, CW, epsW, kb)       # relaxed orbitals + SCF eigenvalues
            E_orbfix = mp2(V, C, epsW, kb)      # mesh orbitals + SCF eigenvalues
            E_first = mp2(V, C, epsF_l, kb)     # E3's Fock head (first order, frozen orbitals)
            # uncoupled first-order rotation (no J/K response): kappa_ai = -W_ai / (e_a - e_i), W = -dK/2 (per k)
            CU = C.copy()
            W = -lam * dK / 2
            CU[:, :, 0] = C[:, :, 0] + (W[:, 1, 0] / (eps_sh[:, 0] - eps_sh[:, 1]))[:, None] * C[:, :, 1]
            CU[:, :, 1] = C[:, :, 1] + (W[:, 0, 1] / (eps_sh[:, 1] - eps_sh[:, 0]))[:, None] * C[:, :, 0]
            E_unc = mp2(V, CU, epsW, kb)
            deig = epsW - eps_sh
            a4 = abs(deig + lam * np.einsum("kpp->kp", dK).real / 2).max()
            # overlap of the relaxed occupied orbital with the mesh virtual (the rotation actually made)
            rot = np.array([abs(C[k][:, 1].conj() @ kb["S"][k] @ CW[k][:, 0]) for k in range(Nk)])
            res[lam] = (E_full - E_orbfix, E_unc - E_orbfix, E_orbfix - E_first, E_full - SF if lam == 1 else None)
            print(f"[{tag}] lam {lam}: A4 max|d eps_SCF + lam dK/2| {a4:.1e} (x n^3 {a4 * n3:.2e}); |<a|i'>| max {rot.max() * n3:.4f} x n^-3 "
                  f"| x n^3: orbital relaxation (coupled) {(E_full - E_orbfix) * n3:+.4e}  (uncoupled) {(E_unc - E_orbfix) * n3:+.4e}  "
                  f"eig beyond 1st order {(E_orbfix - E_first) * n3:+.4e}")
        r1, rh = res[1.0][0], res[0.5][0]
        print(f"[{tag}] A5 linearity: relax(lam=1)/relax(lam=1/2) = {r1 / rh:.4f} (first order: 2)")
        print(f"[{tag}] SUMMARY x n^3: F (1st-order eig, E3's) {(SF - S0) * n3:+.4e}; full SCF-relaxed Fock head "
              f"{(res[1.0][3] + SF - S0) * n3:+.4e}; difference (what E3 misses) {res[1.0][3] * n3:+.4e}")


if __name__ == "__main__":
    main()
