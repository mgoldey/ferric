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
overlap check before any projected number is trusted.

The PySCF molecules are built from ferric's bundled BSE JSON (one segmented
shell per contraction column), so the basis data is identical. PySCF sorts an
atom's shells by angular momentum (s, s, s, p, p) while ferric keeps the file
order (s, sp -> s p, sp -> s p), so every matrix is permuted into ferric's AO
order before it is written. Within a shell both codes order p as x, y, z and d
as m = -2..2.

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


def ferric_basis(name, elements):
    """PySCF basis dict from ferric's bundled JSON, in ferric's shell order."""
    path = ROOT / f"crates/ferric-core/src/basis/bundled/{name}.json"
    data = json.loads(path.read_text())["elements"]
    out = {}
    for sym, z in elements.items():
        shells = []
        for sh in data[str(z)]["electron_shells"]:
            exps = [float(e) for e in sh["exponents"]]
            ams = sh["angular_momentum"]
            for k, col in enumerate(sh["coefficients"]):
                ang = ams[k] if len(ams) > 1 else ams[0]
                shells.append([ang, *[[e, float(c)] for e, c in zip(exps, col)]])
        out[sym] = shells
    return out


def ferric_ao_order(mol, basis):
    """perm[i] = PySCF AO index of ferric AO i (PySCF stable-sorts shells by l)."""
    perm, off = [], 0
    for ia in range(mol.natm):
        ls = [sh[0] for sh in basis[mol.atom_pure_symbol(ia)]]
        sorted_idx = sorted(range(len(ls)), key=lambda k: ls[k])
        start, pos = {}, off
        for k in sorted_idx:
            start[k] = pos
            pos += 2 * ls[k] + 1 if ls[k] > 1 else (1 if ls[k] == 0 else 3)
        for k in range(len(ls)):
            n = 2 * ls[k] + 1 if ls[k] > 1 else (1 if ls[k] == 0 else 3)
            perm.extend(range(start[k], start[k] + n))
        off = pos
    assert sorted(perm) == list(range(mol.nao_nr()))
    return perm


def main():
    xyz = (ROOT / "testdata/molecules/water.xyz").read_text().splitlines()[2:]
    atom = "; ".join(line.strip() for line in xyz if line.strip())
    elements = {"O": 8, "H": 1}
    bas_s = ferric_basis("6-31g", elements)
    bas_b = ferric_basis("cc-pvdz", elements)
    mol_s = gto.M(atom=atom, basis=bas_s, verbose=0)
    mol_b = gto.M(atom=atom, basis=bas_b, verbose=0)
    ps, pb = ferric_ao_order(mol_s, bas_s), ferric_ao_order(mol_b, bas_b)
    mf = scf.RHF(mol_s)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    nocc = mol_s.nelectron // 2
    c_occ = mf.mo_coeff[:, :nocc]
    dm = mf.make_rdm1()
    s_s = mol_s.intor_symmetric("int1e_ovlp")
    s_b = mol_b.intor_symmetric("int1e_ovlp")
    dm_p = addons.project_dm_nr2nr(mol_s, dm, mol_b)
    c_p = addons.project_mo_nr2nr(mol_s, c_occ, mol_b)
    out = {
        "pyscf_version": pyscf.__version__,
        "small_basis": "6-31g",
        "target_basis": "cc-pvdz",
        "nocc": nocc,
        "s_small": s_s[np.ix_(ps, ps)].tolist(),
        "s_target": s_b[np.ix_(pb, pb)].tolist(),
        "dm_small": dm[np.ix_(ps, ps)].tolist(),
        "c_occ_small": c_occ[ps].tolist(),
        "dm_projected": dm_p[np.ix_(pb, pb)].tolist(),
        "c_occ_projected": c_p[pb].tolist(),
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} (PySCF {pyscf.__version__}, E_small = {mf.e_tot:.12f})")
    print("max|dm_proj| =", np.abs(out["dm_projected"]).max())


if __name__ == "__main__":
    main()
