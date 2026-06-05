#!/usr/bin/env python3
"""PySCF RI-MP2/cc-pVDZ on the same ACONF conformers, for a per-conformer
cross-check against ferric. Uses the identical geometries (TURBOMOLE -> Bohr,
fed to PySCF in Bohr directly) so the only difference is the program.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_aconf as R  # reuse REACTIONS + element map

from pyscf import gto, scf
from pyscf import df

HERE = os.path.dirname(os.path.abspath(__file__))
STRUC = os.path.join(HERE, "ACONFstructures")
HARTREE_TO_KCAL = 627.509474


def read_turbomole_bohr(path):
    """Parse TURBOMOLE $coord -> (atom_str, unit='Bohr') for PySCF."""
    atoms = []
    with open(path) as fh:
        in_coord = False
        for line in fh:
            s = line.strip()
            if s.startswith("$coord"):
                in_coord = True
                continue
            if s.startswith("$") and in_coord:
                break
            if in_coord and s:
                x, y, z, sym = s.split()
                el = R.ELEMENTS.get(sym.lower(), sym.capitalize())
                atoms.append(f"{el} {x} {y} {z}")
    return "; ".join(atoms)


def energy(name, basis="cc-pvdz", auxbasis="cc-pvdz-ri"):
    atom = read_turbomole_bohr(os.path.join(STRUC, name))
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit").run()
    # RI-MP2 with the same correlation aux as ferric (cc-pVDZ-RI)
    from pyscf.mp.dfmp2_native import DFMP2
    pt = DFMP2(mf, auxbasis=auxbasis).run()
    return mf.e_tot + pt.e_corr, mf.e_tot, pt.e_corr


def main():
    names = sorted({n for r in R.REACTIONS for n in r[:2]})
    print(f"# ACONF / PySCF RI-MP2 / cc-pVDZ  ({len(names)} conformers)")
    e = {}
    t0 = time.time()
    for n in names:
        try:
            tot, rhf, corr = energy(n)
            e[n] = tot
            print(f"  {n:12s} E = {tot:.8f} Ha  (RHF {rhf:.8f}, MP2corr {corr:.8f})", flush=True)
        except Exception as ex:
            print(f"  {n:12s} FAILED: {ex}", flush=True)
            e[n] = None

    print(f"\n# reactions (kcal/mol): PySCF vs CCSD(T)/CBS ref")
    print(f"{'rxn':>3} {'reactant':>10} {'product':>10} {'pyscf':>9} {'ref':>9} {'err':>8}")
    errs = []
    for i, (a, b, ref) in enumerate(R.REACTIONS, 1):
        if e.get(a) is None or e.get(b) is None:
            print(f"{i:3d} {a:>10} {b:>10} {'--':>9} {ref:9.3f}  skipped")
            continue
        de = (e[b] - e[a]) * HARTREE_TO_KCAL
        err = de - ref
        errs.append(err)
        print(f"{i:3d} {a:>10} {b:>10} {de:9.3f} {ref:9.3f} {err:8.3f}")
    if errs:
        mae = sum(abs(x) for x in errs) / len(errs)
        mx = max(abs(x) for x in errs)
        rmsd = (sum(x * x for x in errs) / len(errs)) ** 0.5
        md = sum(errs) / len(errs)
        print(f"\n# {len(errs)}/{len(R.REACTIONS)} reactions | "
              f"MAE={mae:.3f}  MD={md:+.3f}  RMSD={rmsd:.3f}  MAX={mx:.3f} kcal/mol "
              f"| {time.time()-t0:.1f}s")
    # dump per-conformer for the ferric diff
    import json
    with open(os.path.join(HERE, "aconf_pyscf_energies.json"), "w") as fh:
        json.dump(e, fh, indent=2)


if __name__ == "__main__":
    main()
