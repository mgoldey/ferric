#!/usr/bin/env python3
"""Numpy demo of domain-truncated RI fitting with robust (Dunlap) assembly,
built entirely on ferric's pyo3 bindings — the prototyping pattern for this
lane (see wiki/local-ri-robust-domain-fitting.md; Rust is for when the answer
IS the wall-clock).

Bindings used: run_rhf (+ mo_coefficients, orbital_energies), boys_localize
(same Jacobi-sweep Rust implementation the probes use), shell_info (aux shell
centers -> geometric domains), compute_eri3_mo (blocked AO->MO 3-center
transform with ARBITRARY coefficient matrices — never materializes the
naux*nbas^2 AO tensor), compute_metric_2c, run_rimp2 (exact anchor).

Construction (identical to benchmarks/harness/examples/local_ri_scaling_bench.rs):
Boys-localized occupieds, per-orbital aux domains = whole shells within r_cut
of each Boys center, sub-block V^{-1} fits, then
  exact:  c = V^{-1} A,           G_ex  = A^T c
  naive:  G_1  = A^T c~           (FIRST order in dc = c~ - c)
  robust: G_rob = G_1 + c~^T (A - V c~)   (SECOND order: -dc^T V dc)
and the MP2 energy of each via the loc->canonical rotation U (orthonormal
here, but computed by lstsq so the demo stays overlap-free and would survive
a non-orthogonal localizer too).

Anchors: a huge r_cut gives machine-eps errors in every column; E_corr from
the exact fit matches run_rimp2. Cross-language check: on the default system
(butane), c_err / g_naive / g_rob at each r_cut should reproduce the Rust
probe's alkane_4 rows (wiki doc §3-5 series) — same orbitals, same domains,
same algebra, different language.

Usage: python scripts/py_domain_fit_demo.py [xyz] [basis] [auxbasis]
       (defaults: testdata/molecules/alkane_4.xyz cc-pvdz cc-pvdz-ri)
"""
import sys

import numpy as np

import ferric

R_CUTS_BOHR = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 50.0]


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

    boys = ferric.boys_localize(mol, obs, c_occ)
    assert boys.converged, f"Boys not converged in {boys.iterations} sweeps"
    c_loc = boys.c_loc()
    centers = boys.centers()  # (nocc, 3), Bohr

    # loc -> canonical rotation, overlap-free (exact: same span).
    u, *_ = np.linalg.lstsq(c_occ, c_loc, rcond=None)
    w = np.linalg.inv(u)

    a3 = ferric.compute_eri3_mo(mol, obs, aux, c_loc, c_vir)  # (naux, nocc, nvir)
    v = ferric.compute_metric_2c(mol, obs, aux)
    aux_centers, aux_offs, aux_dims = ferric.shell_info(mol, aux)
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
    print(f"# Boys converged in {boys.iterations} sweeps")
    print(f"# E_corr anchor: demo exact fit {e_exact:.9f} vs run_rimp2 {rimp2.mp2_corr:.9f}  (d = {e_exact - rimp2.mp2_corr:.2e})\n")

    print(f"{'r_cut':>6} {'avg|P_i|':>9} {'c_err':>10} {'g_naive':>10} {'g_rob':>10} {'rob/cerr^2':>11} {'dE_naive':>11} {'dE_rob':>11}")
    for r_cut in R_CUTS_BOHR:
        c_t = np.zeros((naux, nocc, nvir))
        sizes = []
        for i in range(nocc):
            d = np.linalg.norm(aux_centers - centers[i], axis=1)
            sh = np.nonzero(d <= r_cut)[0]
            idx = np.concatenate([np.arange(aux_offs[s], aux_offs[s] + aux_dims[s]) for s in sh])
            sizes.append(len(idx))
            c_t[idx, i, :] = np.linalg.solve(v[np.ix_(idx, idx)], a3[idx, i, :])
        c2 = c_t.reshape(naux, nov)

        c_err = np.linalg.norm(c2 - c_glob) / np.linalg.norm(c_glob)
        g_1 = a2.T @ c2
        g_rob = g_1 + c2.T @ (a2 - v @ c2)
        g_naive_err = np.linalg.norm(g_1 - g_ex) / g_ex_norm
        g_rob_err = np.linalg.norm(g_rob - g_ex) / g_ex_norm

        de_naive = mp2_energy(to_canonical(g_1), eps_occ, eps_vir) - e_exact
        de_rob = mp2_energy(to_canonical(g_rob), eps_occ, eps_vir) - e_exact
        ratio = g_rob_err / c_err**2 if c_err > 1e-10 else float("nan")
        print(
            f"{r_cut:>6.1f} {np.mean(sizes):>9.1f} {c_err:>10.3e} {g_naive_err:>10.3e} "
            f"{g_rob_err:>10.3e} {ratio:>11.3e} {de_naive:>11.3e} {de_rob:>11.3e}"
        )


if __name__ == "__main__":
    main()
