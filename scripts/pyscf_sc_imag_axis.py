"""Dump PySCF Σc(HOMO) on the imaginary axis (ef + iω), pre-analytic-continuation,
for H2O/cc-pVDZ @PBE — to compare against ferric's [sc-trace] raw Σc and split
'self-energy wrong' from 'AC wrong'."""
import numpy as np
from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC, get_sigma, _get_scaled_legendre_roots
HA = 27.211386245988
mol = gto.M(atom="O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
            basis="cc-pvdz", unit="Angstrom", verbose=0)
mf = dft.RKS(mol); mf.xc = "pbe"; mf.run()
nocc = mol.nelectron // 2; homo = nocc - 1
gw = GWAC(mf)
gw.orbs = list(range(max(0, homo-1), nocc+2))
gw.initialize_df()
Lpq = gw.ao2mo(mf.mo_coeff)
ef = gw.get_ef(mo_energy=mf.mo_energy)
quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
eval_freqs = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)
sigmaI, omega = get_sigma(gw, gw.orbs, Lpq, quad_freqs, quad_wts, ef,
                          mf.mo_energy, iw_cutoff=gw.ac_iw_cutoff,
                          eval_freqs=eval_freqs, fullsigma=gw.fullsigma)
# sigmaI shape: (n_orbs, n_eval) complex. Find HOMO's row.
homo_in_orbs = gw.orbs.index(homo)
print(f"ef = {ef:.6f} Ha,  nw={gw.nw},  n_eval={len(omega)}")
print(f"{'omega(Ha)':>12} {'ReSc(eV)':>12} {'ImSc(eV)':>12}")
for j, w in enumerate(omega):
    sc = sigmaI[homo_in_orbs, j]
    print(f"{w:12.5f} {sc.real*HA:12.6f} {sc.imag*HA:12.6f}")
