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
and the MP2 energy of each in the canonical basis.

Two evaluation tiers, selected by nov = nocc*nvir (or DEMO_FORCE_STREAM=1):
  dense  (small):  full (nov x nov) G matrices, loc->canonical rotation of G.
  stream (large):  Frobenius errors via exact aux-space trace identities
                   (never forms an nov^2 object), energies via a j-chunked
                   per-occupied assembly in the canonical basis (coefficients
                   rotated, not G). This is what runs at C32/cc-pVDZ
                   (nov ~ 84k; a dense G would be ~55 GB).

Anchors: a huge r_cut gives machine-eps errors in every column; E_corr from
the exact fit matches run_rimp2 (dense tier); the stream tier reproduces the
dense tier digit-for-digit on butane (DEMO_FORCE_STREAM=1). The dense-tier
butane table also reproduces the Rust probe's alkane_4 rows exactly.

Usage: python scripts/py_domain_fit_demo.py [xyz] [basis] [auxbasis]
       (defaults: testdata/molecules/alkane_4.xyz cc-pvdz cc-pvdz-ri)
Env:   DEMO_FORCE_STREAM=1  force the streaming tier (validation)
"""
import os
import sys

import numpy as np
import scipy.linalg as sla

import ferric

DENSE_MAX_NOV = 25000
J_CHUNK = 8


def mp2_energy_dense(g4, eps_occ, eps_vir):
    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    return float(np.sum(g4 * (2.0 * g4 - g4.transpose(0, 3, 2, 1)) / denom))


def domains_and_fit(a3, v, centers, aux_centers, aux_offs, aux_dims, r_cut):
    """Per-orbital geometric domain fits; returns c~ (naux, nocc, nvir) + sizes."""
    naux, nocc, nvir = a3.shape
    c_t = np.zeros((naux, nocc, nvir))
    sizes = []
    for i in range(nocc):
        d = np.linalg.norm(aux_centers - centers[i], axis=1)
        sh = np.nonzero(d <= r_cut)[0]
        idx = np.concatenate([np.arange(aux_offs[s], aux_offs[s] + aux_dims[s]) for s in sh])
        sizes.append(len(idx))
        c_t[idx, i, :] = np.linalg.solve(v[np.ix_(idx, idx)], a3[idx, i, :])
    return c_t, sizes


def stream_energies(a2c, dc, v_dc, lu, eps_occ, eps_vir):
    """(e_exact, e_naive, e_robust) by j-chunked per-i assembly, canonical basis.

    exact_i = A_i^T c_glob ; naive_i = exact_i + A_i^T dc ;
    robust_i = exact_i - dc_i^T (V dc). Never forms an nov^2 matrix.
    """
    nocc, nvir = len(eps_occ), len(eps_vir)
    e_ex = e_na = e_rob = 0.0
    for j0 in range(0, nocc, J_CHUNK):
        j1 = min(nocc, j0 + J_CHUNK)
        cols = slice(j0 * nvir, j1 * nvir)
        cg_c = sla.lu_solve(lu, a2c[:, cols])
        dc_c = dc[:, cols]
        vdc_c = v_dc[:, cols]
        d_j = (
            eps_occ[None, j0:j1, None]
            - eps_vir[:, None, None]
            - eps_vir[None, None, :]
        )  # (nvir, jchunk, nvir), missing eps_occ[i]
        for i in range(nocc):
            icols = slice(i * nvir, (i + 1) * nvir)
            a_i = a2c[:, icols]
            g_ex = a_i.T @ cg_c
            g_na = g_ex + a_i.T @ dc_c
            g_rb = g_ex - dc[:, icols].T @ vdc_c
            denom = eps_occ[i] + d_j
            for acc, g in ((0, g_ex), (1, g_na), (2, g_rb)):
                g4 = g.reshape(nvir, j1 - j0, nvir)
                e = float(np.sum(g4 * (2.0 * g4 - g4.transpose(2, 1, 0)) / denom))
                if acc == 0:
                    e_ex += e
                elif acc == 1:
                    e_na += e
                else:
                    e_rob += e
    return e_ex, e_na, e_rob


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
    nov = nocc * nvir
    eps_occ, eps_vir = eps[:nocc], eps[nocc:]
    c_occ = np.ascontiguousarray(c[:, :nocc])
    c_vir = np.ascontiguousarray(c[:, nocc:])

    boys = ferric.boys_localize(mol, obs, c_occ)
    assert boys.converged, f"Boys not converged in {boys.iterations} sweeps"
    c_loc = boys.c_loc()
    centers = boys.centers()
    u, *_ = np.linalg.lstsq(c_occ, c_loc, rcond=None)

    a3 = ferric.compute_eri3_mo(mol, obs, aux, c_loc, c_vir)
    v = ferric.compute_metric_2c(mol, obs, aux)
    aux_centers, aux_offs, aux_dims = ferric.shell_info(mol, aux)
    naux = a3.shape[0]
    a2 = a3.reshape(naux, nov)
    lu = sla.lu_factor(v)

    stream = nov > DENSE_MAX_NOV or os.environ.get("DEMO_FORCE_STREAM") == "1"
    tier = "stream" if stream else "dense"
    print(f"# {xyz}  {basis}/{auxbasis}  nocc={nocc} nvir={nvir} naux={naux}  tier={tier}")
    print(f"# Boys converged in {boys.iterations} sweeps")

    # Trace-identity ingredients (aux-space; used by both tiers for errors).
    c_glob = sla.lu_solve(lu, a2)
    s_a = a2 @ a2.T
    s_c = c_glob @ c_glob.T
    g_ex_norm = np.sqrt(max(np.sum(s_a * s_c.T), 0.0))
    c_glob_norm = np.linalg.norm(c_glob)

    if not stream:
        g_ex = a2.T @ c_glob
        w = np.linalg.inv(u)

        def to_canonical(g_flat):
            g4 = g_flat.reshape(nocc, nvir, nocc, nvir)
            return np.einsum("pi,paqb,qj->iajb", w, g4, w, optimize=True)

        e_exact = mp2_energy_dense(to_canonical(g_ex), eps_occ, eps_vir)
        rimp2 = ferric.run_rimp2(mol, basis_set=obs, auxbasis=aux)
        print(f"# E_corr anchor: demo exact fit {e_exact:.9f} vs run_rimp2 {rimp2.mp2_corr:.9f}  (d = {e_exact - rimp2.mp2_corr:.2e})\n")
    else:
        print("# stream tier: errors via aux-space trace identities; energies j-chunked\n")

    a2c = None
    if stream:
        a2c = ferric.compute_eri3_mo(mol, obs, aux, c_occ, c_vir).reshape(naux, nov)

    r_cuts = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 50.0] if not stream else [6.0, 10.0]

    rows = []
    for r_cut in r_cuts:
        c_t, sizes = domains_and_fit(a3, v, centers, aux_centers, aux_offs, aux_dims, r_cut)
        c2 = c_t.reshape(naux, nov)
        del c_t
        dc_loc = c2 - c_glob
        c_err = np.linalg.norm(dc_loc) / c_glob_norm

        # Frobenius errors via exact trace identities (both tiers, so the
        # dense tier continuously cross-checks them against its dense G).
        s_dc = dc_loc @ dc_loc.T
        g_naive_err = np.sqrt(max(np.sum(s_a * s_dc.T), 0.0)) / g_ex_norm
        m = v @ s_dc
        g_rob_err = np.sqrt(max(np.sum(m * m.T), 0.0)) / g_ex_norm
        del s_dc, m
        rows.append((r_cut, np.mean(sizes), c_err, g_naive_err, g_rob_err))

        if not stream:
            g_1 = a2.T @ c2
            g_rob = g_1 + c2.T @ (a2 - v @ c2)
            dense_naive = np.linalg.norm(g_1 - g_ex) / g_ex_norm
            dense_rob = np.linalg.norm(g_rob - g_ex) / g_ex_norm
            for t_val, d_val, lab in ((g_naive_err, dense_naive, "naive"), (g_rob_err, dense_rob, "robust")):
                if d_val > 1e-8 and abs(t_val - d_val) / d_val > 1e-6:
                    print(f"!! trace-vs-dense mismatch ({lab}) at r={r_cut}: {t_val:.6e} vs {d_val:.6e}")
            de_na = mp2_energy_dense(to_canonical(g_1), eps_occ, eps_vir) - e_exact
            de_rb = mp2_energy_dense(to_canonical(g_rob), eps_occ, eps_vir) - e_exact
            del g_1, g_rob
        else:
            # Rotate COEFFICIENTS to canonical (cheap), then stream energies.
            dc_can = np.einsum("ij,pja->pia", u, dc_loc.reshape(naux, nocc, nvir)).reshape(naux, nov)
            del dc_loc, c2
            v_dc = v @ dc_can
            e_ex_s, e_na_s, e_rb_s = stream_energies(a2c, dc_can, v_dc, lu, eps_occ, eps_vir)
            del dc_can, v_dc
            de_na, de_rb = e_na_s - e_ex_s, e_rb_s - e_ex_s
            if r_cut == r_cuts[0]:
                print(f"# E_corr (streamed exact fit) = {e_ex_s:.9f}")

        r, avg, ce, gn, gr = rows[-1]
        ratio = gr / ce**2 if ce > 1e-10 else float("nan")
        if len(rows) == 1:
            print(f"{'r_cut':>6} {'avg|P_i|':>9} {'c_err':>10} {'g_naive':>10} {'g_rob':>10} {'rob/cerr^2':>11} {'dE_naive':>11} {'dE_rob':>11}")
        print(
            f"{r:>6.1f} {avg:>9.1f} {ce:>10.3e} {gn:>10.3e} {gr:>10.3e} {ratio:>11.3e} {de_na:>11.3e} {de_rb:>11.3e}"
        )


if __name__ == "__main__":
    main()
