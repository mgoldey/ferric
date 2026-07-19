"""Independent PySCF-integral CIS-TDA oscillator-strength reference for
ETHYLENE / cc-pVDZ -- direct adaptation of scripts/pyscf_cis_osc_ref.py
(the water cross-check) to isolate whether ferric's run_cis_tda kernel
itself is correct on ethylene, independent of any GW/BSE screening.

Same geometry as testdata/molecules/c2h4.xyz / examples/c2h4-bse-tda.toml
(D2h, rCC=1.3395 Ang, rCH=1.086 Ang, HCH=117.4 deg).

Run: OMP_NUM_THREADS=2 python3 scripts/pyscf_c2h4_osc_ref.py
"""
import numpy as np
from pyscf import gto, scf, df

HARTREE2EV = 27.211386245988

mol = gto.M(
    atom="""
    C 0.000000 0.000000 0.669500
    C 0.000000 0.000000 -0.669500
    H 0.000000 0.922832 1.237695
    H 0.000000 -0.922832 1.237695
    H 0.000000 0.922832 -1.237695
    H 0.000000 -0.922832 -1.237695
    """,
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

auxmol = df.addons.make_auxmol(mol, auxbasis="cc-pvdz-ri")
ints_3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
ints_2c = auxmol.intor("int2c2e", aosym="s1")
w2, U2 = np.linalg.eigh(ints_2c)
pos = w2 > 1e-10
w2inv_sqrt = np.zeros_like(w2)
w2inv_sqrt[pos] = 1.0 / np.sqrt(w2[pos])
V_inv_sqrt = (U2 * w2inv_sqrt) @ U2.T
B_ao = np.einsum("pqQ,QP->pqP", ints_3c, V_inv_sqrt)
B_mo = np.einsum("pi,pqP,qj->ijP", mo_coeff, B_ao, mo_coeff)

def bare(p, q, r, s):
    return np.dot(B_mo[p, q, :], B_mo[r, s, :])

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

with mol.with_common_orig((0.0, 0.0, 0.0)):
    dip_ao = mol.intor_symmetric("int1e_r", comp=3)
orbo = mo_coeff[:, :nocc]
orbv = mo_coeff[:, nocc:]
dip_ia = np.einsum("xpq,pi,qa->xia", dip_ao, orbo, orbv)

print("# oscillator strengths (length gauge, sqrt(2) convention):")
for k in range(6):
    X = evecs[:, k].reshape(nocc, nvir)
    mu = np.sqrt(2.0) * np.einsum("ia,xia->x", X, dip_ia)
    f = (2.0 / 3.0) * evals[k] * np.dot(mu, mu)
    print(f"#   {k+1}  E={evals[k]*HARTREE2EV:.6f} eV   f={f:.6e}")

# Also try pyscf's own built-in TDA for a second independent cross-check.
from pyscf import tdscf
td = tdscf.TDA(mf)
td.nstates = 6
td.kernel()
print("# pyscf tdscf.TDA (built-in, exact RHF, exact 4-index ERIs, no DF):")
osc = td.oscillator_strength()
for k in range(6):
    print(f"#   {k+1}  E={td.e[k]*HARTREE2EV:.6f} eV   f={osc[k]:.6e}")
