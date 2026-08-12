#!/usr/bin/env python3
"""Pure-numpy RI-MP2 on top of ferric's pyo3 integral bindings, raced against
the Rust ri_mp2 driver.

The Python side leans on ferric for everything integral-shaped (compute_eri3,
compute_metric_2c, run_rhf) and does the MO transform + dressing + energy
assembly in numpy, mirroring crates/ferric-mp2/src/rimp2.rs:

  1. V = (P|Q), Cholesky L (any F with F F^T = V^{-1} gives the same fitted
     (ia|jb), so L^{-1} here vs V^{-1/2} in Rust changes nothing at ~1e-10)
  2. B[P,ia] = sum_{mu nu} (P|mu nu) C_occ[mu,i] C_vir[nu,a]  (two GEMMs)
  3. b = L^{-1} B  (triangular solve, BLAS3)
  4. E_os/E_ss from G_i = b_i^T b_tail per occupied i, unique pairs j >= i
     with the fac=2/fac=1 symmetry weights — same loop structure as
     spin_components_from_b_ov.

Usage: python scripts/py_rimp2_bench.py <xyz> <basis> <auxbasis> [--check]
  --check  also run the Rust run_rimp2 and assert |dE_corr| < 1e-8
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

import ferric


def py_rimp2(mol, obs, aux, rhf, frozen_core=0):
    """Returns (e_os, e_ss, timings dict). Mirrors ri_mp2_spin_components."""
    t = {}

    t0 = time.perf_counter()
    eri3 = ferric.compute_eri3(mol, obs, aux)  # (naux, nbf, nbf)
    t["eri3"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    v2c = ferric.compute_metric_2c(mol, obs, aux)
    L = sla.cholesky(v2c, lower=True)
    t["metric"] = time.perf_counter() - t0

    eps = rhf.orbital_energies()
    c = rhf.mo_coefficients()
    naux, nbf, _ = eri3.shape
    nocc_total = mol.nelec() // 2
    nocc = nocc_total - frozen_core
    nvir = nbf - nocc_total
    c_occ = c[:, frozen_core:nocc_total]
    c_vir = c[:, nocc_total:]

    t0 = time.perf_counter()
    # (P|mu nu) -> (P|i a): virtual side first (one wide GEMM), then occ side.
    half = eri3.reshape(naux * nbf, nbf) @ c_vir            # (naux*nbf, nvir)
    half = half.reshape(naux, nbf, nvir)
    b = np.tensordot(c_occ, half, axes=([0], [1]))          # (nocc, naux, nvir)
    b = np.ascontiguousarray(b.transpose(1, 0, 2)).reshape(naux, nocc * nvir)
    t["transform"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = sla.solve_triangular(L, b, lower=True)              # dressed B (naux, ia)
    t["dress"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_o = eps[frozen_core:nocc_total]
    e_v = eps[nocc_total:]
    e_os = 0.0
    e_ss = 0.0
    for i in range(nocc):
        b_i = b[:, i * nvir:(i + 1) * nvir]                 # (naux, nvir)
        g = b_i.T @ b[:, i * nvir:]                         # (nvir, (nocc-i)*nvir)
        g = g.reshape(nvir, nocc - i, nvir).transpose(1, 0, 2)  # (j, a, b)
        denom = (e_o[i] + e_o[i:])[:, None, None] - e_v[:, None] - e_v[None, :]
        tamp = g / denom
        fac = np.full(nocc - i, 2.0)
        fac[0] = 1.0                                        # j == i diagonal
        e_os += np.einsum("j,jab,jab->", fac, tamp, g, optimize=True)
        e_ss += np.einsum("j,jab,jab->", fac, tamp, g - g.transpose(0, 2, 1),
                          optimize=True)
    t["energy"] = time.perf_counter() - t0
    t["mp2_total"] = t["metric"] + t["transform"] + t["dress"] + t["energy"]
    t["mp2_total_with_eri3"] = t["mp2_total"] + t["eri3"]
    return e_os, e_ss, t


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    dfjk = "--dfjk" in sys.argv        # fast DF-JK reference (timing-only runs)
    no_rust = "--no-rust" in sys.argv  # skip the Rust comparison leg
    xyz, basis_name, aux_name = args
    mol = ferric.Molecule.from_xyz(xyz)
    obs = ferric.BasisSet.bundled(basis_name)
    aux = ferric.BasisSet.bundled(aux_name)

    rhf_kwargs = (
        dict(df_j_aux="def2-universal-jkfit", df_k_aux="def2-universal-jkfit")
        if dfjk else {}
    )
    t0 = time.perf_counter()
    # No kwargs: identical config to run_rimp2's internal RHF, so the reference
    # orbitals are deterministic-identical and the anchor isolates the MP2 stage.
    rhf = ferric.run_rhf(mol, obs, **rhf_kwargs)
    t_rhf = time.perf_counter() - t0
    assert rhf.converged

    e_os, e_ss, t = py_rimp2(mol, obs, aux, rhf)
    e_corr_py = e_os + e_ss

    print(f"system: {xyz}  basis={basis_name} aux={aux_name}")
    print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    print(f"RHF energy {rhf.energy:.10f}  ({t_rhf:.3f} s)")
    print(f"python E_corr = {e_corr_py:.10f} (os {e_os:.10f}, ss {e_ss:.10f})")
    for k, v in t.items():
        print(f"  py {k:20s} {v:8.3f} s")

    # Rust driver (includes its own RHF; subtract the measured RHF time for the
    # per-stage comparison — same config, so same work).
    if no_rust:
        return

    # Second RHF timing so the run_rimp2-minus-RHF subtraction uses a warm,
    # same-process measurement rather than the cold first call.
    t0 = time.perf_counter()
    ferric.run_rhf(mol, obs)
    t_rhf2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = ferric.run_rimp2(mol, obs, aux)
    t_rust_total = time.perf_counter() - t0
    t_rust_mp2 = t_rust_total - t_rhf2
    print(f"rust  E_corr = {r.mp2_corr:.10f}  "
          f"(total {t_rust_total:.3f} s, RHF warm {t_rhf2:.3f} s, "
          f"MP2 stage ~{t_rust_mp2:.3f} s)")
    d = e_corr_py - r.mp2_corr
    print(f"dE_corr(py - rust) = {d:.3e}")
    if check:
        assert abs(d) < 1e-8, f"exactness anchor FAILED: {d:.3e}"
        print("exactness anchor PASSED (<1e-8)")


if __name__ == "__main__":
    main()
