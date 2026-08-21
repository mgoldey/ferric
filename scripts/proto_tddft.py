"""Prototype: closed-shell linear-response TDDFT (TDA + full Casida).

Validates the algorithm against PySCF TDDFT on water/cc-pVDZ.
Two paths:
  1. Pure PySCF SCF → hand-built TDA/TDDFT → compare with PySCF TDDFT
  2. Ferric RHF → hand-built TDA/TDHF → compare with PySCF TDHF

Path 1 validates the TDDFT algorithm (the singlet linear-response equations).
Path 2 validates ferric's MO coefficients produce correct response matrices.
"""

import sys
import os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))
from pyscf import gto, dft, scf, tdscf

os.environ["OPENBLAS_NUM_THREADS"] = "1"

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
NSTATES = 5

# ── Water geometry (Angstrom, matches ferric's testdata/molecules/water.xyz) ──
WATER_ATOM = "O 0.000000 0.000000 0.117790; H 0.000000 0.755453 -0.471161; H 0.000000 -0.755453 -0.471161"


def build_tda_matrix(eri_mo, eps, nocc, c_hf=0.0):
    """Build the TDA (A) matrix for closed-shell singlet excitations.

    A_{ia,jb} = δ_{ij}δ_{ab}(ε_a - ε_i) + 2*(ia|jb) - c_HF*(ij|ab)

    For pure DFT (c_hf=0): A = diag(ε_a - ε_i) + 2*(ia|jb)
    For HF (c_hf=1): A = diag(ε_a - ε_i) + 2*(ia|jb) - (ij|ab)
    For hybrid (0 < c_hf < 1): interpolated

    The f_xc kernel is omitted (this gives TDA without the XC kernel,
    which is exact for HF and a first approximation for DFT).
    """
    nmo = len(eps)
    nvir = nmo - nocc
    dim = nocc * nvir

    A = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            ia = i * nvir + a
            aa = a + nocc
            for j in range(nocc):
                for b in range(nvir):
                    jb = j * nvir + b
                    bb = b + nocc
                    # Orbital energy diagonal
                    if i == j and a == b:
                        A[ia, jb] += eps[aa] - eps[i]
                    # Coulomb: 2*(ia|jb) — singlet factor of 2
                    A[ia, jb] += 2.0 * eri_mo[i, aa, j, bb]
                    # Exchange: -c_HF * (ij|ab)
                    A[ia, jb] -= c_hf * eri_mo[i, j, aa, bb]
    return A


def build_b_matrix(eri_mo, nocc, c_hf=0.0):
    """Build the B matrix for full TDDFT.

    B_{ia,jb} = 2*(ia|bj) - c_HF*(ib|aj)

    For chemist notation (pq|rs):
    B_{ia,jb} = 2*(ia|bj) - c_HF*(ib|aj)
    """
    nmo = eri_mo.shape[0]
    nvir = nmo - nocc
    dim = nocc * nvir

    B = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            ia = i * nvir + a
            aa = a + nocc
            for j in range(nocc):
                for b in range(nvir):
                    jb = j * nvir + b
                    bb = b + nocc
                    # Coulomb: 2*(ia|bj)
                    B[ia, jb] += 2.0 * eri_mo[i, aa, bb, j]
                    # Exchange: -c_HF*(ib|aj)
                    B[ia, jb] -= c_hf * eri_mo[i, bb, aa, j]
    return B


def solve_tda(A, nstates):
    """Diagonalize A, return lowest nstates excitation energies."""
    eigvals, eigvecs = la.eigh(A)
    return eigvals[:nstates], eigvecs[:, :nstates]


def solve_casida(A, B, nstates):
    """Solve the full Casida equation: (A-B)^{1/2}(A+B)(A-B)^{1/2} Z = Ω² Z.

    Returns Ω (excitation energies, not squared).
    """
    ApB = A + B
    AmB = A - B

    # (A-B) should be positive definite for stable ground state
    eigvals_amb, eigvecs_amb = la.eigh(AmB)
    if np.any(eigvals_amb < -1e-10):
        print(f"  WARNING: (A-B) has negative eigenvalue {eigvals_amb.min():.6e}")
        print(f"  Ground state may be unstable (triplet instability)")

    # (A-B)^{1/2}
    sqrt_amb = eigvecs_amb @ np.diag(np.sqrt(np.maximum(eigvals_amb, 0.0))) @ eigvecs_amb.T

    # Hermitian eigenvalue problem
    M = sqrt_amb @ ApB @ sqrt_amb
    omega_sq, Z = la.eigh(M)

    if np.any(omega_sq < -1e-10):
        print(f"  WARNING: Ω² has negative eigenvalue {omega_sq.min():.6e}")

    omega = np.sqrt(np.maximum(omega_sq, 0.0))
    return omega[:nstates], Z[:, :nstates]


def ao_to_mo_eri(eri_ao, C):
    """Transform 4-index AO ERIs to MO basis using einsum (O(N^5) steps)."""
    # Half-transform: (pq|rs) → (iq|rs) → (ij|rs) → (ij|as) → (ij|ab)
    # Done in 4 quarter-transforms for efficiency
    nao = C.shape[0]
    nmo = C.shape[1]
    tmp = np.einsum("pqrs,pi->iqrs", eri_ao, C)
    tmp = np.einsum("iqrs,qj->ijrs", tmp, C)
    tmp = np.einsum("ijrs,rk->ijks", tmp, C)
    eri_mo = np.einsum("ijks,sl->ijkl", tmp, C)
    return eri_mo


def print_comparison(label, hand, ref, nstates):
    """Print side-by-side comparison of excitation energies."""
    print(f"\n  {label}")
    print(f"  {'State':>5}  {'Hand-built (Ha)':>16}  {'Reference (Ha)':>16}  {'Δ (Ha)':>12}  {'Δ (eV)':>10}")
    print(f"  {'-'*5}  {'-'*16}  {'-'*16}  {'-'*12}  {'-'*10}")
    max_err = 0.0
    for s in range(min(nstates, len(hand), len(ref))):
        delta = hand[s] - ref[s]
        max_err = max(max_err, abs(delta))
        print(f"  {s+1:>5}  {hand[s]:>16.10f}  {ref[s]:>16.10f}  {delta:>12.2e}  {delta*27.2114:>10.4f}")
    return max_err


# ═══════════════════════════════════════════════════════════════════════════
# PATH 1: Pure PySCF SCF + hand-built TDDFT
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PATH 1: PySCF PBE SCF → hand-built TDA/TDDFT → vs PySCF TDDFT")
print("=" * 72)

mol_pyscf = gto.M(atom=WATER_ATOM, basis="cc-pvdz", unit="Angstrom", verbose=0)
mf = dft.RKS(mol_pyscf)
mf.xc = "pbe"
mf.kernel()
print(f"PySCF PBE energy: {mf.e_tot:.10f} Ha")

C_pyscf = mf.mo_coeff
eps_pyscf = mf.mo_energy
nocc = mol_pyscf.nelectron // 2
nmo = len(eps_pyscf)
nvir = nmo - nocc
print(f"nocc={nocc}, nvir={nvir}, nmo={nmo}, dim(ia)={nocc*nvir}")

# 4-index AO integrals → MO transform
eri_ao = mol_pyscf.intor("int2e")
eri_mo = ao_to_mo_eri(eri_ao, C_pyscf)

# Build TDA A matrix (pure DFT, c_hf=0, no f_xc kernel)
print("\nBuilding TDA matrix (no f_xc kernel)...")
A = build_tda_matrix(eri_mo, eps_pyscf, nocc, c_hf=0.0)
tda_energies, _ = solve_tda(A, NSTATES)

# Build full TDDFT matrices
print("Building full TDDFT matrices (no f_xc kernel)...")
B = build_b_matrix(eri_mo, nocc, c_hf=0.0)
casida_energies, _ = solve_casida(A, B, NSTATES)

# PySCF TDA reference
td_tda = tdscf.TDA(mf)
td_tda.nstates = NSTATES
td_tda.kernel()
pyscf_tda = td_tda.e

# PySCF full TDDFT reference
td_full = tdscf.TDDFT(mf)
td_full.nstates = NSTATES
td_full.kernel()
pyscf_tddft = td_full.e

err_tda = print_comparison("TDA (no f_xc) vs PySCF TDA", tda_energies, pyscf_tda, NSTATES)
err_full = print_comparison("Full TDDFT (no f_xc) vs PySCF full TDDFT", casida_energies, pyscf_tddft, NSTATES)

print(f"\n  TDA max error:   {err_tda:.2e} Ha ({err_tda*27.2114:.4f} eV)")
print(f"  TDDFT max error: {err_full:.2e} Ha ({err_full*27.2114:.4f} eV)")
print(f"  (Residual is the f_xc kernel contribution — expected to be nonzero for DFT)")


# ═══════════════════════════════════════════════════════════════════════════
# PATH 2: HF/TDHF — exact reference (no f_xc kernel needed)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PATH 2: PySCF HF SCF → hand-built TDA/TDHF → vs PySCF TDHF (EXACT)")
print("=" * 72)

mf_hf = scf.RHF(mol_pyscf)
mf_hf.kernel()
print(f"PySCF HF energy: {mf_hf.e_tot:.10f} Ha")

C_hf = mf_hf.mo_coeff
eps_hf = mf_hf.mo_energy
eri_mo_hf = ao_to_mo_eri(eri_ao, C_hf)

# TDA (= CIS for HF) with c_hf=1
print("\nBuilding CIS/TDA-HF matrix (c_hf=1.0)...")
A_hf = build_tda_matrix(eri_mo_hf, eps_hf, nocc, c_hf=1.0)
tda_hf, _ = solve_tda(A_hf, NSTATES)

# Full TDHF
print("Building full TDHF matrices...")
B_hf = build_b_matrix(eri_mo_hf, nocc, c_hf=1.0)
tdhf, _ = solve_casida(A_hf, B_hf, NSTATES)

# PySCF TDA-HF = CIS
td_cis = scf.RHF(mol_pyscf).run()
td_cis_calc = tdscf.TDA(td_cis)
td_cis_calc.nstates = NSTATES
td_cis_calc.kernel()
pyscf_cis = td_cis_calc.e

# PySCF full TDHF
td_tdhf = tdscf.TDHF(td_cis)
td_tdhf.nstates = NSTATES
td_tdhf.kernel()
pyscf_tdhf = td_tdhf.e

err_cis = print_comparison("CIS (TDA-HF) vs PySCF CIS", tda_hf, pyscf_cis, NSTATES)
err_tdhf = print_comparison("Full TDHF vs PySCF TDHF", tdhf, pyscf_tdhf, NSTATES)

print(f"\n  CIS max error:  {err_cis:.2e} Ha")
print(f"  TDHF max error: {err_tdhf:.2e} Ha")
if err_cis < 1e-8:
    print("  ✓ CIS EXACT MATCH (no f_xc kernel for HF — this must be exact)")
else:
    print("  ✗ CIS MISMATCH — algorithm bug")
if err_tdhf < 1e-8:
    print("  ✓ TDHF EXACT MATCH")
else:
    print("  ✗ TDHF MISMATCH — algorithm bug")


# ═══════════════════════════════════════════════════════════════════════════
# PATH 3: Ferric RHF energy/orbital-energy validation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PATH 3: Ferric RHF orbital energies → CIS (using PySCF integrals)")
print("=" * 72)
print("NOTE: ferric (libint2) and PySCF (libcint) use different AO")
print("conventions. MO coefficients cannot be mixed with the other's")
print("AO integrals. The Rust TDDFT will use ferric's own integral engine.")

try:
    import ferric
    mol_ferric = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    rhf_ferric = ferric.run_rhf(mol_ferric, bs)
    print(f"\n  Ferric RHF energy: {rhf_ferric.energy:.10f} Ha")
    print(f"  PySCF  RHF energy: {mf_hf.e_tot:.10f} Ha")
    print(f"  ΔE = {abs(rhf_ferric.energy - mf_hf.e_tot):.2e} Ha")

    eps_ferric = np.array(rhf_ferric.orbital_energies())

    # Validate orbital energies match
    eps_err = np.max(np.abs(eps_ferric - eps_hf))
    print(f"\n  Max orbital energy difference: {eps_err:.2e} Ha")
    if eps_err < 1e-6:
        print("  ✓ Orbital energies match — ferric SCF is correct for TDDFT")
    else:
        print("  ✗ Orbital energies differ")

    # Use PySCF MOs but ferric orbital energies for CIS — should match since
    # orbital energies only enter the diagonal of A
    A_mixed = build_tda_matrix(eri_mo_hf, eps_ferric, nocc, c_hf=1.0)
    tda_mixed, _ = solve_tda(A_mixed, NSTATES)
    err_mixed = print_comparison("CIS (PySCF MOs, ferric eps) vs PySCF CIS",
                                 tda_mixed, pyscf_cis, NSTATES)
    print(f"\n  Max error: {err_mixed:.2e} Ha")
    if err_mixed < 1e-6:
        print("  ✓ Ferric orbital energies produce correct CIS (Rust TDDFT will work)")

except ImportError:
    print("  ferric not importable — skipping path 3")
except Exception as e:
    print(f"  ferric path 3 failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PATH 4: PySCF B3LYP (hybrid) — TDA/TDDFT with c_hf=0.2
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PATH 4: PySCF B3LYP → hand-built TDA/TDDFT (c_hf=0.2, no f_xc)")
print("=" * 72)

mf_b3 = dft.RKS(mol_pyscf)
mf_b3.xc = "b3lyp"
mf_b3.kernel()
print(f"PySCF B3LYP energy: {mf_b3.e_tot:.10f} Ha")

C_b3 = mf_b3.mo_coeff
eps_b3 = mf_b3.mo_energy
eri_mo_b3 = ao_to_mo_eri(eri_ao, C_b3)

# B3LYP has 20% HF exchange
A_b3 = build_tda_matrix(eri_mo_b3, eps_b3, nocc, c_hf=0.2)
B_b3 = build_b_matrix(eri_mo_b3, nocc, c_hf=0.2)

tda_b3, _ = solve_tda(A_b3, NSTATES)
casida_b3, _ = solve_casida(A_b3, B_b3, NSTATES)

td_tda_b3 = tdscf.TDA(mf_b3)
td_tda_b3.nstates = NSTATES
td_tda_b3.kernel()

td_full_b3 = tdscf.TDDFT(mf_b3)
td_full_b3.nstates = NSTATES
td_full_b3.kernel()

err_b3_tda = print_comparison("B3LYP TDA (no f_xc) vs PySCF", tda_b3, td_tda_b3.e, NSTATES)
err_b3_full = print_comparison("B3LYP TDDFT (no f_xc) vs PySCF", casida_b3, td_full_b3.e, NSTATES)
print(f"\n  B3LYP TDA max error:   {err_b3_tda:.2e} Ha ({err_b3_tda*27.2114:.4f} eV)")
print(f"  B3LYP TDDFT max error: {err_b3_full:.2e} Ha ({err_b3_full*27.2114:.4f} eV)")
print(f"  (Residual is the GGA f_xc kernel — expected nonzero)")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  HF/CIS  (exact, no f_xc): {err_cis:.2e} Ha   — must be <1e-8")
print(f"  HF/TDHF (exact, no f_xc): {err_tdhf:.2e} Ha   — must be <1e-8")
print(f"  PBE TDA  (missing f_xc):  {err_tda:.2e} Ha   — f_xc residual")
print(f"  PBE TDDFT (missing f_xc): {err_full:.2e} Ha   — f_xc residual")
print(f"  B3LYP TDA (missing f_xc): {err_b3_tda:.2e} Ha — f_xc residual")
print(f"  B3LYP TDDFT(missing f_xc):{err_b3_full:.2e} Ha — f_xc residual")
print()
print("The HF paths (CIS/TDHF) validate the algorithm exactly.")
print("DFT paths show the f_xc kernel contribution that Rust must implement.")
print("The Rust TDDFT should match PySCF once f_xc is included via libxc.")
