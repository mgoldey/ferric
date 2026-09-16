#!/usr/bin/env python3
"""Generate PySCF UHF/def2-SVP reference energies for He, He+, Ne, Ne+.

Feeds crates/ferric-scf/tests/hene_atomic_ip_anchor.rs. The four atomic energies
and their dIP combination are the R -> infinity limit of the HeNe+ cDFT-ET
diabatic gap, and the only externally-referenced fact in that lane.

Usage: OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_atomic_ip_refs.py
"""
import numpy as np
from pyscf import gto, scf

def run(sym, charge, spin):
    m = gto.M(atom=f"{sym} 0 0 0", basis="def2-svp", charge=charge, spin=spin,
              verbose=0, unit="Angstrom")
    mf = scf.UHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    e = mf.kernel()
    assert mf.converged, f"{sym} q={charge} s={spin} NOT converged"
    ss = mf.spin_square()
    return e, ss, m.nao

res = {}
for label, sym, q, s in [("He",   "He", 0, 0),
                         ("He+",  "He", 1, 1),
                         ("Ne",   "Ne", 0, 0),
                         ("Ne+",  "Ne", 1, 1)]:
    e, ss, nao = run(sym, q, s)
    res[label] = e
    print(f"{label:4s} nao={nao:3d}  E = {e:.12f}  <S^2>={ss[0]:.6f} 2S+1={ss[1]:.4f}")

HA2EV = 27.211386245988
dip = (res["He+"] + res["Ne"]) - (res["He"] + res["Ne+"])
print()
print(f"IP(He) = {res['He+'] - res['He']:.12f} Ha = {(res['He+']-res['He'])*HA2EV:.6f} eV")
print(f"IP(Ne) = {res['Ne+'] - res['Ne']:.12f} Ha = {(res['Ne+']-res['Ne'])*HA2EV:.6f} eV")
print(f"dIP    = {dip:.12f} Ha = {dip*HA2EV:.6f} eV")
