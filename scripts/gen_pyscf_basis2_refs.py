#!/usr/bin/env python3
"""PySCF reference for ferric's secondary-basis (BASIS2) guess projection.

Writes testdata/reference/water_631g_ccpvdz_basis2_projection.json for
crates/ferric-scf/tests/basis2_guess.rs.

What is compared: the projection itself, not an SCF. This script converges
water/6-31G RHF in PySCF and stores the small-basis density and occupied MOs.
The Rust test feeds THOSE matrices to ferric's projector and compares with

  * project_dm_nr2nr(mol_s, dm, mol_b)    -> S_bb^-1 S_bs D S_sb S_bb^-1
  * project_mo_nr2nr(mol_s, C_occ, mol_b) -> S_bb^-1 S_bs C_occ

so the only thing under test is the cross overlap and the S_bb solve. Both
overlaps are stored too, so a basis-data or AO-ordering mismatch fails on the
overlap check before any projected number is trusted. For 6-31G and cc-pVDZ
on H/O, PySCF's AO order equals ferric's (s, then p as x, y, z, then d as
m = -2..2), so no permutation is needed.

Run: OMP_NUM_THREADS=1 uv run --no-sync python scripts/gen_pyscf_basis2_refs.py
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.scf import addons

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "testdata/reference/water_631g_ccpvdz_basis2_projection.json"


def main():
    xyz = (ROOT / "testdata/molecules/water.xyz").read_text().splitlines()[2:]
    atom = "; ".join(line.strip() for line in xyz if line.strip())
    mol_s = gto.M(atom=atom, basis="6-31g", verbose=0)
    mol_b = gto.M(atom=atom, basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol_s)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    nocc = mol_s.nelectron // 2
    c_occ = mf.mo_coeff[:, :nocc]
    dm = mf.make_rdm1()
    out = {
        "pyscf_version": pyscf.__version__,
        "small_basis": "6-31g",
        "target_basis": "cc-pvdz",
        "nocc": nocc,
        "s_small": mol_s.intor_symmetric("int1e_ovlp").tolist(),
        "s_target": mol_b.intor_symmetric("int1e_ovlp").tolist(),
        "dm_small": dm.tolist(),
        "c_occ_small": c_occ.tolist(),
        "dm_projected": addons.project_dm_nr2nr(mol_s, dm, mol_b).tolist(),
        "c_occ_projected": addons.project_mo_nr2nr(mol_s, c_occ, mol_b).tolist(),
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} (PySCF {pyscf.__version__}, E_small = {mf.e_tot:.12f})")
    print("max|dm_proj| =", np.abs(out["dm_projected"]).max())


if __name__ == "__main__":
    main()
