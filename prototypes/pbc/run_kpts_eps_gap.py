"""Triclinic k-point eigenvalue gap vs PySCF KRHF/AFTDF (investigation, 2026-09-27).

Hypotheses (stated before the run):
 (a) near-degenerate levels: individual eps depend on the mixing gauge -> the largest |d eps| sit in
     near-degenerate pairs and per-cluster sums agree.  Predicts: gap concentrated where min spacing < ~1e-6.
 (b) SCF convergence: eps are linear in the density error, E is quadratic.  The oracle ran ours to
     |[F,DS]| < 1e-7 (krhf's hard-coded emax) and PySCF to conv_tol 1e-11 (conv_tol_grad = sqrt -> 3e-6).
     Predicts: at the SAME density both builds give eps equal to ~1e-11, and the gap shrinks when both SCFs
     are tightened (commutator 1e-10); gap ~ O(residual).
 (c) exxdiv shift applied differently to occ vs virt: predicts a same-density gap in the ewald run only,
     of the form v_M * (S D S) projected onto one class of orbitals.
 (d) PySCF Fock at finite precision/mesh: predicts a same-density Fock difference ~1e-9 that moves with the
     AFTDF mesh / cell.precision; (e) complex eigensolver: predicts a gap at identical Fock (checked by
     diagonalising the SAME F with both our X-orthogonalised eigh and scipy.linalg.eigh(F, S)).
Artifact hypothesis: if (b), comparing a loose SCF on one side to a tight one on the other shows the
loose side's residual; tightening only one side must NOT close the gap.

Usage: python3 run_kpts_eps_gap.py n1n2n3 [pyscf_mesh]    (caches the dense build in $EPSGAP_DIR)
"""

import os
import sys
import time

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbc_gamma import Cell  # noqa: E402
from pbc_kpts import build_k, jk_k  # noqa: E402
from run_kpts_anchor import SP, TRI_A, TRI_ATOMS  # noqa: E402
from pyscf.pbc import df as pdf  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import scf as pscf  # noqa: E402

CACHE = os.environ.get("EPSGAP_DIR", ".")
NELEC = 4


def fock(kb, dm, vm):
    J, K = jk_k(kb, dm)
    if vm:
        K = K + vm * np.einsum("kab,kbc,kcd->kad", kb["S"], dm, kb["S"])
    return kb["h"] + J - 0.5 * K


def energy(kb, dm, F):
    return 0.5 * np.einsum("kmn,knm->", kb["h"] + F, dm).real / kb["Nk"] + kb["enn"]


def diag(F, S, lindep=1e-8):
    eps, C = [], []
    for k in range(len(F)):
        s, U = np.linalg.eigh(S[k])
        X = U[:, s > lindep] / np.sqrt(s[s > lindep])
        e, c = np.linalg.eigh(X.conj().T @ F[k] @ X)
        eps.append(e)
        C.append(X @ c)
    return eps, C


def density(eps, C, Nk, nocc):
    allE = sorted((e, k, i) for k, ek in enumerate(eps) for i, e in enumerate(ek))[
        : nocc * Nk
    ]
    occ = [np.zeros(len(e), bool) for e in eps]
    for _, k, i in allE:
        occ[k][i] = True
    return np.array(
        [2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(Nk)]
    ), occ


def resid(F, dm, S):
    return max(
        abs(F[k] @ dm[k] @ S[k] - S[k] @ dm[k] @ F[k]).max() for k in range(len(F))
    )


def scf(kb, vm, gtol, dm=None, maxiter=300):
    """DIIS k-RHF to max|FDS-SDF| < gtol; returns dm, F(dm), eps of F(dm) (un-extrapolated), residual."""
    S, Nk = kb["S"], kb["Nk"]
    if dm is None:
        eps, C = diag(kb["h"], S)
        dm, _ = density(eps, C, Nk, NELEC // 2)
    fs, es = [], []
    for it in range(maxiter):
        F = fock(kb, dm, vm)
        r = resid(F, dm, S)
        if r < gtol:
            eps, C = diag(F, S)
            return dm, F, eps, r, it
        err = [F[k] @ dm[k] @ S[k] - S[k] @ dm[k] @ F[k] for k in range(Nk)]
        fs, es = (fs + [F])[-8:], (es + [err])[-8:]
        m = len(fs)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[-1, -1] = 0
            for i in range(m):
                for j in range(m):
                    B[i, j] = sum(np.vdot(a, b).real for a, b in zip(es[i], es[j]))
            rhs = np.zeros(m + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            F = sum(ci * fi for ci, fi in zip(c, fs))
        eps, C = diag(F, S)
        dm, _ = density(eps, C, Nk, NELEC // 2)
    raise RuntimeError("no convergence")


def maxd(a, b):
    return max(abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, b))


if __name__ == "__main__":
    n = tuple(int(c) for c in sys.argv[1])
    pymesh = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    prec = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
    path = os.path.join(CACHE, f"kb_tri_{''.join(map(str, n))}.npz")
    cell = Cell(TRI_A, TRI_ATOMS, SP)
    if os.path.exists(path):
        kb = dict(np.load(path))
        kb["Nk"] = int(kb["Nk"])
        kb["enn"] = float(kb["enn"])
        kb["madelung"] = float(kb["madelung"])
    else:
        kb = build_k(cell, n, exxdiv="ewald")
        np.savez(path, **{k: v for k, v in kb.items()})
    Nk = kb["Nk"]
    pc = pgto.Cell(a=TRI_A, atom=TRI_ATOMS, basis=SP, unit="B", cart=True, verbose=0)
    pc.precision = prec
    pc.max_memory = 1200
    pc.build()
    kp = pc.make_kpts(list(n))
    print(f"tri {n}: PySCF mesh {pymesh}^3 precision {prec:g}", flush=True)
    for ex in (None, "ewald"):
        vm = kb["madelung"] if ex else 0.0
        # --- ours, loose (the oracle's level) and tight
        runs = {}
        for tag, g in (("loose1e-7", 1e-7), ("tight1e-11", 1e-11)):
            dm, F, eps, r, it = scf(kb, vm, g)
            runs[tag] = (dm, F, eps, r)
            print(
                f"[{ex}] ours {tag}: E {energy(kb, dm, F):.13f} resid {r:.1e} it {it}",
                flush=True,
            )
        dmT, FT, epsT, _ = runs["tight1e-11"]
        print(
            f"[{ex}] ours loose-vs-tight: max|d eps| {maxd(runs['loose1e-7'][2], epsT):.1e}",
            flush=True,
        )
        # (a) spacing and per-level table at tight
        for k in range(Nk):
            sp = np.diff(epsT[k])
            print(
                f"[{ex}] k{k} eps {np.array2string(epsT[k], precision=6, max_line_width=250)} min spacing {sp.min():.2e}"
            )
        # --- PySCF at OUR tight density (same-density Fock)
        mf = pscf.KRHF(pc, kp, exxdiv=ex)
        mf.with_df = pdf.AFTDF(pc, kp)
        mf.with_df.mesh = [pymesh] * 3
        t = time.time()
        Sp = np.asarray(mf.get_ovlp())
        hp = np.asarray(mf.get_hcore())
        vj, vk = mf.get_jk(dm_kpts=dmT)
        Fp = hp + np.asarray(vj) - 0.5 * np.asarray(vk)
        print(
            f"[{ex}] same-density builds ({time.time() - t:.0f}s): |dS| {abs(Sp - kb['S']).max():.1e} "
            f"|dh| {abs(hp - kb['h']).max():.1e} |dT| {abs(np.asarray(mf.get_hcore()) - hp).max():.0e} "
            f"|dF| {abs(Fp - FT).max():.1e}",
            flush=True,
        )
        J, K = jk_k(kb, dmT)
        if vm:
            K = K + vm * np.einsum("kab,kbc,kcd->kad", kb["S"], dmT, kb["S"])
        print(
            f"[{ex}]   |dJ| {abs(np.asarray(vj) - J).max():.1e} |dK| {abs(np.asarray(vk) - K).max():.1e}",
            flush=True,
        )
        eP, _ = diag(Fp, Sp)
        print(
            f"[{ex}]   same-density max|d eps| (PySCF F vs ours F) {maxd(eP, epsT):.1e}",
            flush=True,
        )
        eS = [
            scipy.linalg.eigh(FT[k], kb["S"][k], eigvals_only=True) for k in range(Nk)
        ]
        print(
            f"[{ex}]   (e) same F, scipy eigh(F,S) vs X-orth eigh: {maxd(eS, epsT):.1e}",
            flush=True,
        )
        # decompose the same-density eps gap by term (first-order: C^H dF C diagonal)
        _, CT = diag(FT, kb["S"])
        for lab, dX in (
            ("h", hp - kb["h"]),
            ("J", np.asarray(vj) - J),
            ("-K/2", -0.5 * (np.asarray(vk) - K)),
        ):
            d1 = max(
                abs(np.einsum("mi,mn,ni->i", CT[k].conj(), dX[k], CT[k]).real).max()
                for k in range(Nk)
            )
            print(f"[{ex}]   first-order eps shift from d{lab}: {d1:.1e}", flush=True)
        # --- PySCF SCF from our tight dm: default (oracle) tolerances, then tight
        for tag, ct, cg in (
            ("oracle conv_tol 1e-11 (grad sqrt)", 1e-11, None),
            ("tight 1e-14 / grad 1e-10", 1e-14, 1e-10),
        ):
            mf2 = pscf.KRHF(pc, kp, exxdiv=ex)
            mf2.with_df = mf.with_df
            mf2.conv_tol = ct
            if cg:
                mf2.conv_tol_grad = cg
            for start, d0 in (
                ("our loose dm", runs["loose1e-7"][0]),
                ("our tight dm", dmT),
            ):
                t = time.time()
                ep = mf2.kernel(dm0=d0)
                dmp = np.asarray(mf2.make_rdm1())
                Fpp = np.asarray(mf2.get_fock(dm=dmp))
                rp = resid(Fpp, dmp, Sp)
                print(
                    f"[{ex}] PySCF {tag} from {start}: E {ep:.13f} dE(ours tight) {energy(kb, dmT, FT) - ep:.1e} "
                    f"cycles {mf2.cycles} resid(own) {rp:.1e} max|d eps| {maxd(mf2.mo_energy, epsT):.1e} "
                    f"|dD| {abs(dmp - dmT).max():.1e} ({time.time() - t:.0f}s)",
                    flush=True,
                )
