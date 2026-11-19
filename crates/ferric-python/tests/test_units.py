"""`ferric.units`: constants equal the Rust ones and what Molecule applies."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

import ferric.units as u


def test_bohr_to_angstrom_is_the_ratio_molecule_applies():
    # 1/0.52917721092 is how XYZ Angstrom becomes Bohr; coords() undoes it.
    mol = ferric.Molecule.from_arrays(
        ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], units="bohr"
    )
    ang = mol.coords()[1][2]
    bohr = mol.coords_bohr()[1][2]
    assert bohr == 0.74
    assert ang / bohr == pytest.approx(u.BOHR_TO_ANGSTROM, rel=1e-15)
    mol2 = ferric.Molecule.from_arrays(["H", "H"], [[0, 0, 0], [0, 0, 1.0]])
    assert mol2.coords_bohr()[1][2] == pytest.approx(u.ANGSTROM_TO_BOHR, rel=1e-15)


def test_values_match_rust_constants():
    assert u.BOHR_TO_ANGSTROM == 0.52917721092
    assert u.BOHR_TO_ANGSTROM * u.ANGSTROM_TO_BOHR == pytest.approx(1.0, abs=4e-16)
    assert u.HARTREE_TO_EV == 27.211386245988
    assert u.HARTREE_TO_KCAL_MOL == 627.5094740631
    assert u.HARTREE_TO_KJ_MOL == pytest.approx(2625.49963948, rel=1e-10)
    assert u.HARTREE_TO_KJ_MOL == u.HARTREE_TO_KCAL_MOL * 4.184


def test_viz_uses_the_units_constant():
    viz = pytest.importorskip("ferric.viz")
    assert viz.BOHR_TO_ANGSTROM is u.BOHR_TO_ANGSTROM
