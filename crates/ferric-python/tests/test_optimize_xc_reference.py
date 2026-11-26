"""Geometry optimization from Python with an XC functional (run_optimize)."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def _water():
    return ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    )


def test_run_optimize_with_xc_and_reference():
    mol = _water()
    res = ferric.run_optimize(
        mol, "sto-3g", xc="PBE", reference="rhf", max_steps=3, e_conv=1e-4
    )
    assert res.energy is not None
    assert res.steps >= 1
    assert isinstance(res.converged, bool)


def test_run_optimize_rejects_unknown_reference():
    mol = _water()
    with pytest.raises(ValueError, match="unknown reference"):
        ferric.run_optimize(mol, "sto-3g", reference="rks", max_steps=1)
