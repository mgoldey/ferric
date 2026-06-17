#!/usr/bin/env python3
"""PySCF G0W0@HF on a given xyz + basis — the reference half of the
same-geometry same-basis cross-check against ferric (examples/gw_xcheck.rs).

Both codes: identical geometry (GW100 canonical xyz), identical basis
(def2-TZVP), G0W0 starting from RHF/HF, density-fitted. ferric uses PDEP-as-W;
PySCF uses analytic continuation (gw_ac). If they agree per-molecule, ferric's
G0W0 implementation is proven on the IDENTICAL setup — no geometry/basis caveat.

Prints one parseable line:  PYSCF <ip_g0w0_ev> <ip_koopmans_ev>

Usage: pyscf_g0w0.py <file.xyz> [basis]   (basis default def2-tzvp)
"""
import sys

import numpy as np
from pyscf import dft, gto, scf
from pyscf.gw.gw_ac import GWAC

HARTREE2EV = 27.211386245988


def main():
    xyz_path = sys.argv[1]
    basis = sys.argv[2] if len(sys.argv) > 2 else "def2-tzvp"

    lines = open(xyz_path).read().splitlines()
    nat = int(lines[0].split()[0])
    atom = "\n".join(lines[2 : 2 + nat])

    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    nocc = mol.nelectron // 2
    homo = nocc - 1
    ip_koop = -mf.mo_energy[homo] * HARTREE2EV

    # G0W0@HF on all orbitals (we only need HOMO). gw_ac = analytic continuation.
    # PySCF 2.13: orbs is an attribute, not a kernel kwarg.
    gw = GWAC(mf)
    gw.orbs = range(mol.nao_nr())
    gw.kernel()
    ip_g0w0 = -gw.mo_energy[homo] * HARTREE2EV

    print(f"PYSCF {ip_g0w0:.4f} {ip_koop:.4f}")


if __name__ == "__main__":
    main()
