"""Hypothesis test: does G0W0@PBE fix the N2/CO over-shoot that G0W0@HF has?
If yes (N2 @PBE lands near exp 15.58 vs @HF 17.12), the 'over-correction' is a
HF-starting-point problem and the resolution is GW@PBE, not a code fix.
Compares @HF vs @PBE G0W0 HOMO IP for N2, CO, H2O at the ferric geometry/basis.
"""
import numpy as np
from pyscf import gto, scf, dft
from pyscf.gw.gw_ac import GWAC

CASES = {
    "N2":  ("N 0 0 -0.5488; N 0 0 0.5488", 15.58),
    "CO":  ("C 0 0 -0.6442; O 0 0 0.4828", 14.01),
    "H2O": ("O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161", 12.62),
}
HA = 27.211386245988

def g0w0_ip(mf, mol):
    nocc = mol.nelectron // 2
    homo = nocc - 1
    gw = GWAC(mf)
    gw.orbs = list(range(max(0, homo - 1), nocc + 2))
    gw.kernel()
    return -gw.mo_energy[homo] * HA, -mf.mo_energy[homo] * HA

print(f"{'mol':4} {'exp':>6} | {'HF/Koop':>8} {'G0W0@HF':>8} | {'PBE/Koop':>8} {'G0W0@PBE':>8}")
for name, (atom, exp_ip) in CASES.items():
    mol = gto.M(atom=atom, basis="aug-cc-pvtz", unit="Angstrom", verbose=0)
    mf_hf = scf.RHF(mol).run()
    ip_hf, koop_hf = g0w0_ip(mf_hf, mol)
    mf_pbe = dft.RKS(mol); mf_pbe.xc = "pbe"; mf_pbe.run()
    ip_pbe, koop_pbe = g0w0_ip(mf_pbe, mol)
    print(f"{name:4} {exp_ip:6.2f} | {koop_hf:8.3f} {ip_hf:8.3f} | {koop_pbe:8.3f} {ip_pbe:8.3f}")
