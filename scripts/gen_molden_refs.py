#!/usr/bin/env python
"""PySCF's reading of Molden AOs, for ferric's Molden writer tests.

Feeds ``crates/ferric-python/tests/test_molden.py`` (the tests that run
without PySCF). Writes two files to ``testdata/reference``:

* ``molden_cart_spdfg_basis.json``: a BSE-format basis whose O and H shells
  are ferric's bundled cc-pVQZ shells (exponents and coefficients as
  ``BasisSet.shells`` returns them) marked ``gto_cartesian``, so the d, f and
  g shells are Cartesian (6D 10F 15G). No bundled basis has Cartesian f or g.
* ``molden_ao_values_pyscf.json``: for an O-H pair (fixed geometry in Bohr)
  in that Cartesian basis and in bundled cc-pVQZ (spherical d, f, g), the
  values at fixed points of the AOs a Molden file describes, as
  ``pyscf.tools.molden`` defines them: Molden AO j is PySCF AO
  ``order_ao_index(mol)[j]``, divided by its norm sqrt(S_jj) for Cartesian
  shells (the loader multiplies Molden coefficients by 1/sqrt(S_jj)).

PySCF builds each molecule from the same numbers ferric uses (shells copied
from ``BasisSet.shells``), and the script checks PySCF keeps ferric's shell
order, since the Molden AO order is the ``[GTO]`` shell order.

Usage (one heavy op at a time on the shared box):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 \\
    /home/matt/qc/ferric/.venv/bin/python scripts/gen_molden_refs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto
from pyscf.tools import molden

import ferric

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "reference"
BASIS_JSON = OUT / "molden_cart_spdfg_basis.json"
VALUES_JSON = OUT / "molden_ao_values_pyscf.json"

# O and H off every symmetry element, in Bohr.
ATOMS = [("O", 8, (0.10, -0.20, 0.30)), ("H", 1, (1.05, 1.21, -0.47))]
N_POINTS = 24
SEED = 20261112


def cart_basis_json(source: ferric.BasisSet) -> dict:
    elements = {}
    for _, z, _ in ATOMS:
        shells = []
        for ang, _pure, exps, coefs in source.shells(z):
            shells.append(
                {
                    "function_type": "gto_cartesian",
                    "angular_momentum": [ang],
                    "exponents": [repr(float(e)) for e in exps],
                    "coefficients": [[repr(float(c)) for c in coefs]],
                }
            )
        elements[str(z)] = {"electron_shells": shells}
    return {"name": "cc-pvqz-cartesian (molden test)", "elements": elements}


def pyscf_mol(bs: ferric.BasisSet, cart: bool) -> gto.Mole:
    basis = {
        sym: [
            [ang, *zip(exps, coefs, strict=True)]
            for ang, _, exps, coefs in bs.shells(z)
        ]
        for sym, z, _ in ATOMS
    }
    mol = gto.M(
        atom=[(sym, xyz) for sym, _, xyz in ATOMS],
        unit="Bohr",
        basis=basis,
        cart=cart,
        spin=1,
        verbose=0,
    )
    want = [ang for _, z, _ in ATOMS for ang, *_ in bs.shells(z)]
    got = [int(mol.bas_angular(i)) for i in range(mol.nbas)]
    if got != want:
        raise SystemExit(f"PySCF reordered the shells: {got} vs ferric {want}")
    return mol


def molden_ao_values(mol: gto.Mole, points: np.ndarray) -> np.ndarray:
    ao = mol.eval_gto("GTOval_cart" if mol.cart else "GTOval_sph", points)
    idx = molden.order_ao_index(mol)
    ao = ao[:, idx]
    if mol.cart:
        norm = np.sqrt(mol.intor("int1e_ovlp").diagonal())
        ao = ao / norm[idx]
    return ao


def main() -> None:
    rng = np.random.default_rng(SEED)
    centres = np.array([xyz for *_, xyz in ATOMS])
    points = centres[np.arange(N_POINTS) % len(ATOMS)] + rng.normal(
        scale=0.9, size=(N_POINTS, 3)
    )

    bundled = ferric.BasisSet.bundled("cc-pvqz")
    BASIS_JSON.write_text(json.dumps(cart_basis_json(bundled), indent=1) + "\n")
    cart = ferric.BasisSet.from_bse_json(str(BASIS_JSON))

    cases = {}
    for name, bs, is_cart in (("cc-pvqz", bundled, False), ("cart-spdfg", cart, True)):
        mol = pyscf_mol(bs, is_cart)
        cases[name] = {
            "cart": is_cart,
            "nao": int(mol.nao),
            "ao_values": molden_ao_values(mol, points).tolist(),
        }
    VALUES_JSON.write_text(
        json.dumps(
            {
                "generator": "scripts/gen_molden_refs.py",
                "pyscf_version": pyscf.__version__,
                "atoms": [[sym, z, list(xyz)] for sym, z, xyz in ATOMS],
                "units": "bohr",
                "basis_file": BASIS_JSON.name,
                "points": points.tolist(),
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {BASIS_JSON} and {VALUES_JSON}")


if __name__ == "__main__":
    main()
