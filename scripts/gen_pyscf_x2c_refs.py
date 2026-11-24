#!/usr/bin/env python3
"""PySCF spin-free X2C-1e references for crates/ferric-scf/tests/x2c.rs and
crates/ferric-python/tests/test_x2c_hcore.py.

Settings: ``scf.RHF(mol).sfx2c1e()`` with ``approx = '1e'``,
``xuncontract = True``, ``basis = None``, c = ``lib.param.LIGHT_SPEED``, the
same settings as scripts/prototypes/x2c_proto.py. The PySCF molecule is built
from ferric's bundled cc-pVDZ JSON (same exponents and coefficients),
spherical d. Per system the file stores:

* ``hcore_eigenvalues``: generalized eigenvalues of (get_hcore(), S);
* ``hcore_eig_bar``: their first-order rounding bound (the prototype's
  ``gen_eig_bar``: n * ROUND * (max|h| + |lambda| max|S|) / lambda_min(S),
  ROUND = 3e-13), per eigenvalue;
* ``e_nr``, ``e_x2c``: RHF energies without and with X2C at ``conv_tol``.

Usage: OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_x2c_refs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import gto, lib, scf

REPO = Path(__file__).resolve().parents[1]
BUNDLED = REPO / "crates/ferric-core/src/basis/bundled"
OUT = REPO / "testdata/reference/x2c_sfx2c1e_cc-pvdz.json"
SYMBOLS = {1: "H", 10: "Ne", 18: "Ar", 35: "Br", 36: "Kr"}
SYSTEMS = {
    "Ne": "Ne 0 0 0",
    "Ar": "Ar 0 0 0",
    "Kr": "Kr 0 0 0",
    "HBr": "H 0 0 0\nBr 0 0 1.414",
}
ROUND = 3e-13
CONV_TOL = 1e-10


def pyscf_basis(zs: list[int]) -> dict:
    data = json.loads((BUNDLED / "cc-pvdz.json").read_text())["elements"]
    out = {}
    for z in zs:
        entries = []
        for sh in data[str(z)]["electron_shells"]:
            assert len(sh["angular_momentum"]) == 1, "sp shells not handled"
            ang = sh["angular_momentum"][0]
            exps = [float(e) for e in sh["exponents"]]
            for col in sh["coefficients"]:
                entries.append([ang] + [[e, float(c)] for e, c in zip(exps, col)])
        out[SYMBOLS[z]] = entries
    return out


def gen_eig_bar(a, s):
    n = s.shape[0]
    lam_min = np.linalg.eigvalsh(s).min()
    lam = scipy.linalg.eigh(a, s, eigvals_only=True)
    return lam, n * ROUND * (np.abs(a).max() + np.abs(lam) * np.abs(s).max()) / lam_min


def main() -> None:
    c = float(lib.param.LIGHT_SPEED)
    systems = {}
    for label, xyz in SYSTEMS.items():
        zs = sorted(
            {
                next(z for z, s in SYMBOLS.items() if s == ln.split()[0])
                for ln in xyz.splitlines()
            }
        )
        mol = gto.M(
            atom=xyz.replace("\n", "; "),
            basis=pyscf_basis(zs),
            unit="Angstrom",
            cart=False,
            verbose=0,
        )
        rel = scf.RHF(mol).sfx2c1e()
        x2c = rel.with_x2c
        assert x2c.approx.upper() == "1E" and x2c.xuncontract and x2c.basis is None
        assert abs(lib.param.LIGHT_SPEED - c) == 0
        h = rel.get_hcore()
        s = mol.intor_symmetric("int1e_ovlp")
        lam, bar = gen_eig_bar(h, s)
        rel.conv_tol = CONV_TOL
        e_x2c = rel.kernel()
        nr = scf.RHF(mol)
        nr.conv_tol = CONV_TOL
        e_nr = nr.kernel()
        assert rel.converged and nr.converged
        systems[label] = {
            "xyz_angstrom": xyz,
            "nao": int(mol.nao),
            "hcore_eigenvalues": lam.tolist(),
            "hcore_eig_bar": bar.tolist(),
            "e_nr": float(e_nr),
            "e_x2c": float(e_x2c),
        }
        print(
            f"{label}: nao {mol.nao}, lowest eig {lam[0]:.10f}, dE {e_x2c - e_nr:.10f}"
        )
    OUT.write_text(
        json.dumps(
            {
                "generator": "scripts/gen_pyscf_x2c_refs.py",
                "pyscf_version": __import__("pyscf").__version__,
                "basis": "cc-pvdz (ferric bundled JSON)",
                "light_speed": c,
                "conv_tol": CONV_TOL,
                "settings": "scf.RHF(mol).sfx2c1e(), approx='1e', xuncontract=True, cart=False",
                "systems": systems,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
