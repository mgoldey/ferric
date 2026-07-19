"""Independent PySCF-integral CIS-TDA oscillator-strength reference, DF kernel
+ EXACT (non-DF) RHF reference, matching ferric's run_cis_tda EXACTLY: the
RHF orbitals come from ferric's `RhfConfig::default()` (df_j_aux/df_k_aux
both None -> exact 4-index SCF, NOT density-fitted), while the CIS-TDA
Coulomb+exchange kernel itself uses the RI/cc-pvdz-ri auxiliary basis (via
`mo_b::build_full_b`) -- ferric mixes an exact-SCF reference with a
DF two-electron kernel for the excited-state matrix, so the reference here
must do the same to isolate the oscillator-strength FORMULA (the thing this
task adds), not a Coulomb-kernel or RHF-reference mismatch.

(An earlier version of this script used `scf.RHF(mol).density_fit(...)` for
the RHF step and got excitation energies off by ~1.4e-2 eV from ferric's
run_cis_tda -- that mismatch was the DF-vs-exact RHF reference, not a bug in
either codebase's CIS-TDA kernel; fixed by switching to exact RHF below.)

Mirrors ferric's bse.rs run_cis_tda kernel:
  A_{ia,jb} = (eps_a - eps_i) d_ij d_ab + 2(ia|jb) - (ab|ij)     [bare Coulomb, DF]
and the new tda_oscillator_strengths():
  <0|r|n> = sqrt(2) * sum_ia X_n(i,a) <i|r|a>   (X normalized: sum X^2 = 1)
  f_n = (2/3) * Omega_n * |<0|r|n>|^2

Geometry + basis byte-identical to h2o_bse_tda.rs / bse_oscillator_strength.rs:
  O 0.0 0.0 0.117790 ; H 0.0 0.755453 -0.471161 ; H 0.0 -0.755453 -0.471161 (Angstrom)
  cc-pVDZ orbital, cc-pvdz-ri auxiliary.

Cited by crates/ferric-gw/tests/bse_oscillator_strength.rs
(cis_tda_oscillator_strengths_match_pyscf_df_kernel) -- that test hardcodes
this script's printed output as the pass/fail reference. Re-run this script
and update the test's `pyscf_ref` array if the DF integral generation here
ever changes.

Run: OMP_NUM_THREADS=2 python3 scripts/pyscf_cis_osc_ref.py

Measured 2026-07-19 with pyscf 2.13.0 (pip), numpy per pyscf's pinned dep.
"""
import numpy as np
from pyscf import gto, scf, df

HARTREE2EV = 27.211386245988

mol = gto.M(
    atom="O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
    basis="cc-pvdz", unit="Angstrom", verbose=0,
)
mf = scf.RHF(mol)  # EXACT (non-DF) RHF -- matches ferric's RhfConfig::default()
mf.kernel()
print(f"# exact RHF E = {mf.e_tot:.10f}")

mo_energy = mf.mo_energy
mo_coeff = mf.mo_coeff
nmo = mo_energy.size
nocc = mol.nelectron // 2
nvir = nmo - nocc
n = nocc * nvir
print(f"# nmo={nmo} nocc={nocc} nvir={nvir} n={n}")

# ---- RI 3-index B_pq^P in MO basis (same convention as ferric's mo_b.rs) ----
auxmol = df.addons.make_auxmol(mol, auxbasis="cc-pvdz-ri")
ints_3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")  # (nao,nao,naux)
ints_2c = auxmol.intor("int2c2e", aosym="s1")                        # (naux,naux)
w2, U2 = np.linalg.eigh(ints_2c)
pos = w2 > 1e-10
w2inv_sqrt = np.zeros_like(w2)
w2inv_sqrt[pos] = 1.0 / np.sqrt(w2[pos])
V_inv_sqrt = (U2 * w2inv_sqrt) @ U2.T
B_ao = np.einsum("pqQ,QP->pqP", ints_3c, V_inv_sqrt)
B_mo = np.einsum("pi,pqP,qj->ijP", mo_coeff, B_ao, mo_coeff)  # (nmo,nmo,naux)

def bare(p, q, r, s):
    return np.dot(B_mo[p, q, :], B_mo[r, s, :])

# ---- Assemble A_{ia,jb} = (eps_a-eps_i) d + 2(ia|jb) - (ab|ij) ----
A = np.zeros((n, n))
occ = range(nocc)
virt = range(nocc, nmo)
for i in occ:
    for a_ in virt:
        ia = i * nvir + (a_ - nocc)
        for j in occ:
            for b_ in virt:
                jb = j * nvir + (b_ - nocc)
                coul = bare(i, a_, j, b_)
                exch = bare(a_, b_, i, j)
                A[ia, jb] = 2.0 * coul - exch
        A[ia, ia] += mo_energy[a_] - mo_energy[i]

evals, evecs = np.linalg.eigh(A)
print("# lowest 6 CIS-TDA (DF kernel, exact RHF) excitation energies (eV):")
for k in range(6):
    print(f"#   {k+1}  {evals[k]*HARTREE2EV:.6f}")

# ---- Oscillator strengths: f_n = 2/3 * Omega_n * |<0|r|n>|^2 ----
with mol.with_common_orig((0.0, 0.0, 0.0)):
    dip_ao = mol.intor_symmetric("int1e_r", comp=3)
orbo = mo_coeff[:, :nocc]
orbv = mo_coeff[:, nocc:]
dip_ia = np.einsum("xpq,pi,qa->xia", dip_ao, orbo, orbv)  # (3, nocc, nvir)

print("# oscillator strengths (length gauge, sqrt(2) convention):")
for k in range(6):
    X = evecs[:, k].reshape(nocc, nvir)
    mu = np.sqrt(2.0) * np.einsum("ia,xia->x", X, dip_ia)
    f = (2.0 / 3.0) * evals[k] * np.dot(mu, mu)
    print(f"#   {k+1}  E={evals[k]*HARTREE2EV:.6f} eV   f={f:.6e}")

# Also cross-check with a shifted dipole origin -> must be identical (origin
# independence of occ-virt transition dipoles).
with mol.with_common_orig((1.7, -0.4, 0.9)):
    dip_ao2 = mol.intor_symmetric("int1e_r", comp=3)
dip_ia2 = np.einsum("xpq,pi,qa->xia", dip_ao2, orbo, orbv)
print("# origin-shift cross-check (must match above f values):")
for k in range(6):
    X = evecs[:, k].reshape(nocc, nvir)
    mu = np.sqrt(2.0) * np.einsum("ia,xia->x", X, dip_ia2)
    f = (2.0 / 3.0) * evals[k] * np.dot(mu, mu)
    print(f"#   {k+1}  f={f:.6e}")
