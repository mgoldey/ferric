#!/usr/bin/env python3
"""Regenerate the exact-ERI oracle references asserted by
crates/ferric-mp2/tests/rccd_family.rs.

INDEPENDENT CONSTRUCTION, on purpose: this shares no code with ferric. It
builds the closed-shell spatial drCCD/SOSEX/rCCD quantities from PySCF's
EXACT (non-RI) MO integrals, using the formulation verified against a full
spin-orbital antisymmetrized Riccati oracle in
wiki/notebooks/15-sosex-rccd.ipynb.

Geometries are ferric's OWN testdata/molecules/*.xyz, NOT the notebook's --
the notebook uses H2 at 0.74 A where ferric's testdata has 0.7414 A, so its
published table cannot be asserted against ferric directly. That mismatch is
the reason this script exists rather than hand-copied literals.

The remaining ferric-vs-oracle gap is the RI floor (ferric fits (ia|jb) in an
auxiliary basis; this does not), MEASURED at 1.2e-6..3.6e-5 for the systems
below. It is a basis-incompleteness difference, not solver error.

Usage:  OPENBLAS_NUM_THREADS=4 python3 scripts/gen_rccd_oracle_refs.py
"""
import os
import sys

import numpy as np
from pyscf import ao2mo, gto, scf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_xyz(name):
    """ferric testdata xyz -> PySCF atom string (both in Angstrom)."""
    path = os.path.join(REPO, "testdata", "molecules", name)
    with open(path) as fh:
        lines = fh.read().split("\n")
    natom = int(lines[0])
    return "; ".join(" ".join(l.split()) for l in lines[2 : 2 + natom])


def spatial_pieces(atom, basis, fc):
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, "reference SCF did not converge"
    nmo = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    occ_idx = list(range(fc, nocc))
    vir_idx = list(range(nocc, nmo))
    # Recomputed Fock, not mf.mo_energy: they differ by ~1e-7 at this SCF
    # tolerance, which is enough to move a non-variational energy off a
    # tight bar (notebook 12 s.2 lesson).
    fmo = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    e = np.diag(fmo).copy()
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    return dict(
        K=eri[np.ix_(occ_idx, vir_idx, occ_idx, vir_idx)],  # (ia|jb)
        eo=e[occ_idx],
        ev=e[vir_idx],
        no=len(occ_idx),
        nv=len(vir_idx),
        e_rhf=mf.e_tot,
    )


def fock_super(T, eo, ev):
    return (
        ev[None, :, None, None] * T
        + T * ev[None, None, None, :]
        - eo[:, None, None, None] * T
        - T * eo[None, None, :, None]
    )


def solve_riccati(B, eo, ev, damp=0.7, rtol=1e-13, maxiter=20000):
    """Damped Riccati fixed point R = B + F(T) + BT + TB + TBT."""
    no, nv = len(eo), len(ev)
    D = (
        ev[None, :, None, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, None, :, None]
    )
    bn = np.linalg.norm(B)
    if bn == 0.0:  # zero kernel has T=0 as its exact fixed point
        return np.zeros_like(B), 0, 0.0
    Bm = B.reshape(no * nv, no * nv)
    T = -B / D
    for it in range(1, maxiter + 1):
        Tm = T.reshape(no * nv, no * nv)
        BT, TB = Bm @ Tm, Tm @ Bm
        R = B + fock_super(T, eo, ev) + (BT + TB + Tm @ BT).reshape(B.shape)
        relres = np.linalg.norm(R) / bn
        if relres < rtol:
            return T, it, relres
        T = T - damp * (R / D)
    raise RuntimeError(f"Riccati did not converge: relres={relres:.3e}")


SYSTEMS = [
    ("H2_STO3G", "h2.xyz", "sto-3g", 0),
    ("WATER_631G_FC1", "water.xyz", "6-31g", 1),
]


def main():
    print("// exact-ERI oracle references at ferric's testdata geometries")
    print("// regenerate: scripts/gen_rccd_oracle_refs.py")
    for const, xyz, basis, fc in SYSTEMS:
        p = spatial_pieces(load_xyz(xyz), basis, fc)
        K, eo, ev = p["K"], p["eo"], p["ev"]
        Kex = K.transpose(0, 3, 2, 1)  # (ib|ja)

        T, it, _ = solve_riccati(2.0 * K, eo, ev)
        e_drccd = 0.5 * np.sum(2.0 * K * T)
        e_sosex = np.sum((K - 0.5 * Kex) * T)

        T_s, _, _ = solve_riccati(2.0 * K - Kex, eo, ev)
        T_t, _, _ = solve_riccati(-Kex, eo, ev)
        e_s = 0.5 * np.sum((2.0 * K - Kex) * T_s)
        e_t = 0.5 * np.sum((-Kex) * T_t)

        print(
            f"\n// {xyz} / {basis} / frozen_core={fc}  "
            f"(no={p['no']} nv={p['nv']} E_RHF={p['e_rhf']:.10f}, drCCD {it} iters)"
        )
        print(f"pub const {const}: (f64, f64, f64, f64) = (")
        for v in (e_drccd, e_sosex, e_s, e_t):
            print(f"    {v:.10f},")
        print(");")
        print(f"// E_rCCD = E_S + E_T = {e_s + e_t:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
