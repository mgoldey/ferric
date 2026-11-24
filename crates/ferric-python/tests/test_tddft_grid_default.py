"""run_tddft's main-grid default: meta-GGA gets 150x434, the rest 75x110.

The reference SCF and the f_xc kernel share the grid TddftResult.grid
reports, so asserting it pins the grid both used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
PRODUCTION = (75, 110, "nwchem-radial", 1e-14)
META_GGA = (150, 434, "nwchem-radial", 1e-14)


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz(str(WATER))
    return mol, ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")


def test_meta_gga_defaults_to_the_finer_grid(water):
    r = ferric.run_tddft(*water, functional="r2SCAN", n_roots=2)
    assert r.grid == META_GGA


def test_explicit_grid_kwarg_overrides_the_meta_gga_default(water):
    r = ferric.run_tddft(*water, functional="r2SCAN", n_roots=2, grid_angular=194)
    assert r.grid == (75, 194, "nwchem-radial", 1e-14)


def test_non_meta_gga_keeps_the_production_grid(water):
    r = ferric.run_tddft(*water, functional="PBE", n_roots=2)
    assert r.grid == PRODUCTION
    assert ferric.run_tddft(*water, n_roots=2).grid is None


def test_grid_kwarg_on_an_hf_reference_is_refused(water):
    with pytest.raises(ValueError, match="HF reference"):
        ferric.run_tddft(*water, n_roots=2, grid_radial=99)
