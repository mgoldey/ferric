import numpy as np
from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC
HA = 27.211386245988
mol = gto.M(atom="O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
            basis="cc-pvdz", unit="Angstrom", verbose=0)
mf = dft.RKS(mol); mf.xc = "pbe"; mf.run()
nocc = mol.nelectron // 2; homo = nocc - 1
gw = GWAC(mf); gw.orbs = list(range(max(0, homo-1), nocc+2)); gw.kernel()
vk = gw.get_sigma_exchange(mf.mo_coeff)
vk = np.asarray(vk)
sx_homo = (vk[homo, homo] if vk.ndim == 2 else vk[homo]) * HA
eps_ks = mf.mo_energy[homo]*HA
ip = -gw.mo_energy[homo]*HA
# IP = -(eps_ks + Sigma_c + (Sigma_x - vxc));  vxc = -19.749
vxc = -19.7493
sigma_c = -ip - eps_ks - (sx_homo - vxc)
print(f"Σx(HOMO)  PySCF = {sx_homo:8.4f} eV   ferric -27.0635   Δ={sx_homo-(-27.0635):+.4f}")
print(f"Σc(HOMO)  PySCF implied = {sigma_c:8.4f} eV")
print(f"  (ferric Σc implied = -ip_ferric - eps_ks - (Σx_f - vxc) = {-11.933 - eps_ks - (-27.0635 - vxc):8.4f})")
