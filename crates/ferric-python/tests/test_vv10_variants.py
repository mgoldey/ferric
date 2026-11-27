"""Coverage for ``run_vv10_variants`` (the MP2-V research binding)."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def _water():
    return ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    )


def test_run_vv10_variants_returns_one_energy_per_spec():
    mol = _water()
    bs = ferric.BasisSet.bundled("sto-3g")
    specs = [("none", 1.0, None, 0), ("eq11", 2.0, 0.5, 0)]
    out = ferric.run_vv10_variants(mol, bs, specs)
    assert isinstance(out, list)
    assert len(out) == len(specs)
    assert all(isinstance(x, float) for x in out)


def test_run_vv10_variants_rejects_unknown_kind():
    mol = _water()
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match="unknown kind"):
        ferric.run_vv10_variants(mol, bs, [("bogus", 1.0, None, 0)])
