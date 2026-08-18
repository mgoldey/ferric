#!/usr/bin/env python3
"""Regenerate the exact-ERI LCCD/CEPA(0) references asserted by
crates/ferric-mp2/tests/lccd_cepa0.rs.

INDEPENDENT CONSTRUCTION: shares no code with ferric. Closed-shell spatial
LCCD residual + energy from PySCF's EXACT (non-RI) MO integrals, in the form
verified element-by-element against a full spin-orbital antisymmetrized LCCD
oracle in wiki/notebooks/16-lccd-cepa0.ipynb.

Geometries come from ferric's OWN testdata/molecules/*.xyz so the reference
cannot silently drift from what the tests actually run.

NOTE this script uses a damped Jacobi iteration, NOT the GMRES the Rust port
uses -- deliberately, so that agreement is between two DIFFERENT solvers on
the same equations rather than a reimplementation of the same algorithm.

Usage:  OPENBLAS_NUM_THREADS=4 python3 scripts/gen_lccd_oracle_refs.py
"""
import os
import sys

import numpy as np
from pyscf import ao2mo, gto, scf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_xyz(name):
    path = os.path.join(REPO, "testdata", "molecules", name)
    with open(path) as fh:
        lines = fh.read().split("\n")
    natom = int(lines[0])
    return "; ".join(" ".join(l.split()) for l in lines[2 : 2 + natom])


def solve_lccd(atom, basis, fc, damp=1.0, rtol=1e-12, maxiter=5000):
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    nmo = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    occ_idx = list(range(fc, nocc))
    vir_idx = list(range(nocc, nmo))
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    J = eri[np.ix_(occ_idx, vir_idx, occ_idx, vir_idx)]  # (ia|jb)
    OO = eri[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]  # (ik|jl)
    VV = eri[np.ix_(vir_idx, vir_idx, vir_idx, vir_idx)]  # (ac|bd)
    V = eri[np.ix_(occ_idx, occ_idx, vir_idx, vir_idx)]  # (ij|ab)
    F = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    e = np.diag(F).copy()
    eo, ev = e[occ_idx], e[vir_idx]
    D = (
        ev[None, :, None, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, None, :, None]
    )

    def ring(T):
        tA = np.einsum("kcjb,iakc->iajb", J, T, optimize=True)
        tC = np.einsum("kcjb,icka->iajb", J, T, optimize=True)
        tE = np.einsum("kjcb,iakc->iajb", V, T, optimize=True)
        t1 = 2 * tA - tC - tE
        t2 = np.einsum("kibc,jcka->iajb", V, T, optimize=True)
        t3 = np.einsum("kjac,ickb->iajb", V, T, optimize=True)
        return t1 - t2 - t3 + t1.transpose(2, 3, 0, 1)

    def energy(T):
        return np.sum(J * (2 * T - T.transpose(0, 3, 2, 1)))

    T = -J / D
    bn = np.linalg.norm(J)
    for it in range(1, maxiter + 1):
        R = (
            J
            + np.einsum("ikjl,kalb->iajb", OO, T, optimize=True)
            + np.einsum("acbd,icjd->iajb", VV, T, optimize=True)
            + ring(T)
        )
        relres = np.linalg.norm((-D) * T - R) / bn
        if relres < rtol:
            return energy(T), it, relres, mf.e_tot
        T = damp * (R / (-D)) + (1 - damp) * T
    raise RuntimeError(f"LCCD did not converge: relres={relres:.3e}")


SYSTEMS = [
    ("h2.xyz", "sto-3g", 0),
    ("water.xyz", "6-31g", 1),
]


def main():
    print("// exact-ERI LCCD references at ferric's testdata geometries")
    print("// regenerate: scripts/gen_lccd_oracle_refs.py")
    for xyz, basis, fc in SYSTEMS:
        e, it, relres, e_rhf = solve_lccd(load_xyz(xyz), basis, fc)
        print(
            f"// {xyz} / {basis} / frozen_core={fc}  "
            f"(E_RHF={e_rhf:.10f}, {it} iters, relres {relres:.1e})"
        )
        print(f"//   E_LCCD = {e:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
