#!/usr/bin/env python3
"""Numpy demo of domain-truncated RI fitting with robust (Dunlap) assembly,
built entirely on ferric's pyo3 bindings — the prototyping pattern for this
lane (see wiki/local-ri-robust-domain-fitting.md; Rust is for when the answer
IS the wall-clock).

Bindings used: run_rhf (+ mo_coefficients), compute_eri3_mo (blocked AO->MO
3-center transform with ARBITRARY coefficient matrices — never materializes
the naux*nbas^2 AO tensor), compute_metric_2c, run_rimp2 (exact anchor).

Numpy stand-ins for bindings that don't exist yet:
  - Localized occupieds: pivoted Cholesky of the AO density D = C_occ C_occ^T
    (Boys isn't exposed; Cholesky orbitals localize around pivot AOs and span
    the same space). They are NOT orthonormal, so the loc->canonical rotation
    U (from lstsq, no overlap matrix needed) is inverted, not transposed.
  - Domains: per-orbital, data-driven — keep the top fraction f of aux
    functions ranked by that orbital's fitting-row weight ||(P|ia)||_a.
    (The production Rust probes use geometric balls around Boys centers;
    magnitude selection is the geometry-free demo equivalent.)

What it shows (same structure as the Rust probes):
  exact:  c = V^{-1} A,          G_ex  = A^T c
  naive:  G_1  = A^T c~          (FIRST order in dc = c~ - c)
  robust: G_rob = G_1 + c~^T (A - V c~)   (SECOND order: G_rob - G_ex = -dc^T V dc)
and the MP2 energy error of each, vs the kept-fraction f.
Anchors: f = 1.0 must give machine-eps errors, and E_corr(exact fit) must
match run_rimp2. Expected: robust consistently below naive, with
g_rob/c_err^2 approaching a constant as c_err shrinks. NOTE the demo's
stand-ins (Cholesky orbitals, magnitude-ranked domains) leave c_err at
O(0.2-0.7) — the regime ENTERING second order, not the clean asymptotic law;
the Boys+geometric Rust probes reach c_err ~1e-2 where the quadratic
constant is flat (wiki doc §§3-5).

Usage: python scripts/py_domain_fit_demo.py [xyz] [basis] [auxbasis]
       (defaults: testdata/molecules/alkane_4.xyz cc-pvdz cc-pvdz-ri)
"""
import sys

import numpy as np

import ferric


def cholesky_orbitals(c_occ):
    """Pivoted Cholesky of D = C C^T: localized, same-span, NOT orthonormal."""
    d = c_occ @ c_occ.T
    n, nocc = d.shape[0], c_occ.shape[1]
    loc = np.zeros((n, nocc))
    for k in range(nocc):
        p = int(np.argmax(np.diag(d)))
        loc[:, k] = d[:, p] / np.sqrt(d[p, p])
        d -= np.outer(loc[:, k], loc[:, k])
    return loc


def mp2_energy(g4, eps_occ, eps_vir):
    """Closed-shell MP2 from canonical (ia|jb), shape (nocc,nvir,nocc,nvir)."""
    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    return float(np.sum(g4 * (2.0 * g4 - g4.transpose(2, 1, 0, 3)) / denom))


def main():
    xyz = sys.argv[1] if len(sys.argv) > 1 else "testdata/molecules/alkane_4.xyz"
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"
    auxbasis = sys.argv[3] if len(sys.argv) > 3 else "cc-pvdz-ri"

    mol = ferric.Molecule.from_xyz(xyz)
    obs = ferric.BasisSet.bundled(basis)
    aux = ferric.BasisSet.bundled(auxbasis)

    rhf = ferric.run_rhf(mol, basis_set=obs)
    c = rhf.mo_coefficients()
    eps = np.asarray(rhf.orbital_energies())
    nbas = c.shape[0]
    nocc = mol.nelec() // 2
    nvir = nbas - nocc
    eps_occ, eps_vir = eps[:nocc], eps[nocc:]
    c_occ = np.ascontiguousarray(c[:, :nocc])
    c_vir = np.ascontiguousarray(c[:, nocc:])

    # Localize + the exact (non-orthogonal-safe) back-rotation C_loc = C_can U.
    c_loc = cholesky_orbitals(c_occ)
    u, *_ = np.linalg.lstsq(c_occ, c_loc, rcond=None)
    w = np.linalg.inv(u)
    span_err = np.abs(c_occ @ u - c_loc).max()

    a3 = ferric.compute_eri3_mo(mol, obs, aux, c_loc, c_vir)  # (naux, nocc, nvir)
    v = ferric.compute_metric_2c(mol, obs, aux)
    naux = a3.shape[0]
    nov = nocc * nvir
    a2 = a3.reshape(naux, nov)

    c_glob = np.linalg.solve(v, a2)
    g_ex = a2.T @ c_glob
    g_ex_norm = np.linalg.norm(g_ex)

    def to_canonical(g_flat):
        g4 = g_flat.reshape(nocc, nvir, nocc, nvir)
        return np.einsum("pi,paqb,qj->iajb", w, g4, w, optimize=True)

    e_exact = mp2_energy(to_canonical(g_ex), eps_occ, eps_vir)
    rimp2 = ferric.run_rimp2(mol, basis_set=obs, auxbasis=aux)
    print(f"# {xyz}  {basis}/{auxbasis}  nocc={nocc} nvir={nvir} naux={naux}")
    print(f"# localization span check |C U - C_loc|_max = {span_err:.2e}")
    print(f"# E_corr anchor: demo exact fit {e_exact:.9f} vs run_rimp2 {rimp2.mp2_corr:.9f}  (d = {e_exact - rimp2.mp2_corr:.2e})\n")

    # Rank aux functions per orbital by the EXACT fit's coefficient weight —
    # the best data-driven importance measure available without geometry.
    weights = np.linalg.norm(c_glob.reshape(naux, nocc, nvir), axis=2)  # (naux, nocc)

    print(f"{'f':>5} {'avg|P_i|':>9} {'c_err':>10} {'g_naive':>10} {'g_rob':>10} {'rob/cerr^2':>11} {'dE_naive':>11} {'dE_rob':>11}")
    for f in [0.2, 0.3, 0.5, 0.7, 1.0]:
        k = max(1, int(round(f * naux)))
        c_t = np.zeros((naux, nocc, nvir))
        for i in range(nocc):
            idx = np.sort(np.argsort(weights[:, i])[::-1][:k])
            c_t[idx, i, :] = np.linalg.solve(v[np.ix_(idx, idx)], a3[idx, i, :])
        c2 = c_t.reshape(naux, nov)

        c_err = np.linalg.norm(c2 - c_glob) / np.linalg.norm(c_glob)
        g_1 = a2.T @ c2
        g_rob = g_1 + c2.T @ (a2 - v @ c2)
        g_naive_err = np.linalg.norm(g_1 - g_ex) / g_ex_norm
        g_rob_err = np.linalg.norm(g_rob - g_ex) / g_ex_norm

        de_naive = mp2_energy(to_canonical(g_1), eps_occ, eps_vir) - e_exact
        de_rob = mp2_energy(to_canonical(g_rob), eps_occ, eps_vir) - e_exact
        ratio = g_rob_err / c_err**2 if c_err > 1e-12 else float("nan")
        print(f"{f:>5.2f} {k:>9} {c_err:>10.3e} {g_naive_err:>10.3e} {g_rob_err:>10.3e} {ratio:>11.3e} {de_naive:>11.3e} {de_rob:>11.3e}")


if __name__ == "__main__":
    main()
