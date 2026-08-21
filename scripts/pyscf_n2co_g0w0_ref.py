"""PySCF G0W0@HF reference for N2 (and CO, H2O control) at the EXACT ferric
geometry/basis, to discriminate: is ferric's N2 HOMO QP (17.12 eV) a BUG or a
genuine GW@HF error? PySCF gw_ac is the same imaginary-axis + analytic-
continuation method ferric mirrors, and is what the Sigma_c fix validated on H2O.

Ferric numbers to compare (HOMO IP = -eps_qp, eV):
  N2:   ferric G0W0 = 17.12   (exp 15.58)  -- suspected wrong-sign correction
  CO:   ferric G0W0 = 14.76   (exp 14.01)
  H2O:  ferric G0W0 = 12.89   (exp 12.62)  -- known-good control (fix validated here)
"""
import numpy as np
from pyscf import gto, scf
from pyscf.gw.gw_ac import GWAC

CASES = {
    # name: (atom string in Angstrom, ferric G0W0 HOMO IP eV, exp eV)
    "N2":  ("N 0 0 -0.5488; N 0 0 0.5488", 17.12, 15.58),
    "CO":  ("C 0 0 -0.6442; O 0 0 0.4828", 14.76, 14.01),
    "H2O": ("O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
            12.89, 12.62),
}
HA = 27.211386245988

for name, (atom, ferric_ip, exp_ip) in CASES.items():
    mol = gto.M(atom=atom, basis="aug-cc-pvtz", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    homo = nocc - 1
    # G0W0@HF, analytic continuation (Pade is the default AC in gw_ac).
    gw = GWAC(mf)
    gw.orbs = list(range(max(0, homo - 1), nocc + 2))
    gw.kernel()
    eps_qp = gw.mo_energy  # full array, QP-corrected where computed
    ip_pyscf = -eps_qp[homo] * HA
    ehomo_hf = -mf.mo_energy[homo] * HA
    print(f"{name:4}  HF/Koop={ehomo_hf:7.3f}  PySCF_G0W0={ip_pyscf:7.3f}  "
          f"ferric_G0W0={ferric_ip:7.3f}  exp={exp_ip:6.2f}  "
          f"| ferric-PySCF={ferric_ip - ip_pyscf:+.3f}")
