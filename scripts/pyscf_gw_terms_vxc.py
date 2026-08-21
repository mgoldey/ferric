"""Decompose PySCF G0W0@PBE HOMO into terms to localize ferric's 0.762 eV offset.
ferric: Σx=-27.064, vxc=-19.749, ε_KS=-6.121, IP=11.933 (PySCF IP=11.171)."""
import numpy as np
from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC
HA = 27.211386245988
mol = gto.M(atom="O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
            basis="cc-pvdz", unit="Angstrom", verbose=0)
mf = dft.RKS(mol); mf.xc = "pbe"; mf.run()
nocc = mol.nelectron // 2; homo = nocc - 1

gw = GWAC(mf); gw.orbs = list(range(max(0, homo-1), nocc+2)); gw.kernel()

# Σx (exchange self-energy, MO diagonal)
vk = gw.get_sigma_exchange(mf.mo_coeff) if hasattr(gw, "get_sigma_exchange") else None
# vxc in MO basis
dm = mf.make_rdm1()
vxc_ao = mf.get_veff(mol, dm) - mf.get_j(mol, dm)  # v_xc = veff - J (for pure GGA, no HF exch)
vxc_mo = np.einsum("pi,pq,qi->i", mf.mo_coeff, vxc_ao, mf.mo_coeff)

eps_ks = mf.mo_energy[homo] * HA
ip_qp = -gw.mo_energy[homo] * HA
print(f"ε_KS(HOMO)   = {eps_ks:8.4f} eV   (ferric -6.1213)")
print(f"vxc(HOMO)    = {vxc_mo[homo]*HA:8.4f} eV   (ferric -19.7494)")
if vk is not None:
    try:
        print(f"Σx(HOMO)     = {vk[homo]*HA:8.4f} eV   (ferric -27.0635)")
    except Exception as e:
        print("Σx via get_sigma_exchange shape:", np.shape(vk), e)
print(f"IP_qp        = {ip_qp:8.4f} eV   (ferric 11.933)")
# implied Σc = -IP - ε_KS - (Σx - vxc)
print(f"implied (Σx-vxc) consistency: ε_KS + Σx - vxc = {(eps_ks + (vk[homo]*HA if vk is not None else 0) - vxc_mo[homo]*HA):8.4f}")
