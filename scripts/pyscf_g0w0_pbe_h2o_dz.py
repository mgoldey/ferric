"""PySCF G0W0@PBE HOMO IP for H2O / cc-pVDZ — the TDD target for ferric's
closed-shell GW@PBE. Same geometry ferric uses in gw100_full.rs."""
from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC
HA = 27.211386245988
mol = gto.M(atom="O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
            basis="cc-pvdz", unit="Angstrom", verbose=0)
mf = dft.RKS(mol); mf.xc = "pbe"; mf.run()
nocc = mol.nelectron // 2; homo = nocc - 1
gw = GWAC(mf); gw.orbs = list(range(max(0, homo-1), nocc+2)); gw.kernel()
print(f"H2O G0W0@PBE HOMO IP (cc-pVDZ) = {-gw.mo_energy[homo]*HA:.4f} eV")
print(f"PBE/Koopmans HOMO IP          = {-mf.mo_energy[homo]*HA:.4f} eV")
