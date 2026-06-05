#!/usr/bin/env python3
"""GMTKN30 / ACONF benchmark for ferric.

ACONF: 18 alkane conformers (butane/pentane/hexane), 15 conformer-energy
reactions with W1h-val CCSD(T)/CBS references (Gruzman, Karton, Martin,
JPCA 2009, 113, 11974). Each reaction is -1*(reactant) + 1*(product) in
kcal/mol.

Runs ferric RHF + RI-MP2 (+ optional CCSD(T)) on each conformer, forms the
reaction energies, and reports MAE / MAX vs the CCSD(T)/CBS reference.

Geometries are TURBOMOLE $coord (Bohr) -> converted to Angstrom for ferric.
"""
import os
import re
import sys
import time

import ferric

HERE = os.path.dirname(os.path.abspath(__file__))
STRUC = os.path.join(HERE, "ACONFstructures")
BOHR_TO_ANG = 0.52917721092

# reaction table from ACONFref.html: (reactant, product, ref_kcal_mol)
REACTIONS = [
    ("B_T", "B_G", 0.598), ("P_TT", "P_TG", 0.614), ("P_TT", "P_GG", 0.961),
    ("P_TT", "P_GX", 2.813), ("H_ttt", "H_gtt", 0.595), ("H_ttt", "H_tgt", 0.604),
    ("H_ttt", "H_tgg", 0.934), ("H_ttt", "H_gtg", 1.178), ("H_ttt", "H_g+t+g-", 1.302),
    ("H_ttt", "H_ggg", 1.250), ("H_ttt", "H_g+x-t+", 2.632), ("H_ttt", "H_t+g+x-", 2.740),
    ("H_ttt", "H_g+x-g-", 3.283), ("H_ttt", "H_x+g-g-", 3.083), ("H_ttt", "H_x+g-x+", 4.925),
]
HARTREE_TO_KCAL = 627.509474

ELEMENTS = {"c": "C", "h": "H", "o": "O", "n": "N"}


def read_turbomole(path):
    """Parse a TURBOMOLE $coord file (Bohr) -> xyz string (Angstrom) for ferric."""
    atoms = []
    with open(path) as fh:
        in_coord = False
        for line in fh:
            s = line.strip()
            if s.startswith("$coord"):
                in_coord = True
                continue
            if s.startswith("$end") or (s.startswith("$") and in_coord):
                break
            if in_coord and s:
                x, y, z, sym = s.split()
                el = ELEMENTS.get(sym.lower(), sym.capitalize())
                atoms.append((el, float(x) * BOHR_TO_ANG,
                              float(y) * BOHR_TO_ANG, float(z) * BOHR_TO_ANG))
    lines = [str(len(atoms)), ""]
    for el, x, y, z in atoms:
        lines.append(f"{el} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def energies_for(name, obs, aux, method):
    xyz = read_turbomole(os.path.join(STRUC, name))
    mol = ferric.Molecule.from_xyz_string(xyz, 0, 1)
    if method == "rhf":
        return ferric.run_rhf(mol, obs).energy
    if method == "rimp2":
        return ferric.run_rimp2(mol, obs, aux).total_energy
    if method == "ccsd_t":
        r = ferric.run_ccsd_t(mol, obs, aux)
        return r.correlation_energy + r.t_correction  # corr only; add RHF below
    raise ValueError(method)


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "rimp2"
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"
    aux_name = "cc-pvdz-ri"
    obs = ferric.BasisSet.bundled(basis)
    aux = ferric.BasisSet.bundled(aux_name)

    # unique conformers referenced by the reactions
    names = sorted({n for r in REACTIONS for n in r[:2]})
    print(f"# ACONF / ferric  method={method}  basis={basis}  ({len(names)} conformers)")
    e = {}
    t0 = time.time()
    for n in names:
        try:
            if method == "ccsd_t":
                xyz = read_turbomole(os.path.join(STRUC, n))
                mol = ferric.Molecule.from_xyz_string(xyz, 0, 1)
                rhf = ferric.run_rhf(mol, obs).energy
                r = ferric.run_ccsd_t(mol, obs, aux)
                e[n] = rhf + r.correlation_energy + r.t_correction
            else:
                e[n] = energies_for(n, obs, aux, method)
            print(f"  {n:12s} E = {e[n]:.8f} Ha", flush=True)
        except Exception as ex:
            print(f"  {n:12s} FAILED: {ex}", flush=True)
            e[n] = None

    print(f"\n# reactions (kcal/mol):  ferric vs CCSD(T)/CBS W1h-val ref")
    print(f"{'rxn':>3} {'reactant':>10} {'product':>10} {'ferric':>9} {'ref':>9} {'err':>8}")
    errs = []
    for i, (a, b, ref) in enumerate(REACTIONS, 1):
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
        print(f"\n# {len(errs)}/{len(REACTIONS)} reactions | "
              f"MAE={mae:.3f}  MD={md:+.3f}  RMSD={rmsd:.3f}  MAX={mx:.3f} kcal/mol "
              f"| {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
