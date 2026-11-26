"""relativistic="x2c1e" on the correlated (MP2/CC) Python drivers."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def _mol():
    return ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nBr 0 0 1.414\n")


def _bs():
    return ferric.BasisSet.bundled("cc-pvdz")


def test_rimp2_x2c_shifts_energy():
    mol = _mol()
    bs = _bs()
    nr = ferric.run_rimp2(mol, bs, bs)
    rel = ferric.run_rimp2(mol, bs, bs, relativistic="x2c1e")
    assert rel.total_energy != nr.total_energy
    assert abs(rel.total_energy - nr.total_energy) > 1.0


def test_rimp2_x2c_refuses_gradient():
    mol = _mol()
    bs = _bs()
    with pytest.raises(ValueError, match="with_gradient"):
        ferric.run_rimp2(mol, bs, bs, relativistic="x2c1e", with_gradient=True)


def test_scs_mp2_x2c_runs():
    mol = _mol()
    bs = _bs()
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_scs_mp2(mol, bs, aux, relativistic="x2c1e")
    assert r.total_energy is not None


def test_ccsd_x2c_runs():
    mol = _mol()
    bs = _bs()
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_ccsd(mol, bs, aux, relativistic="x2c1e")
    assert r.correlation_energy is not None


def test_ccsd_t_x2c_runs():
    mol = _mol()
    bs = _bs()
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_ccsd_t(mol, bs, aux, relativistic="x2c1e")
    assert r.correlation_energy is not None
    assert r.t_correction is not None


def test_unknown_spelling_raises():
    mol = _mol()
    bs = _bs()
    with pytest.raises(ValueError, match="relativistic"):
        ferric.run_rimp2(mol, bs, bs, relativistic="X2C1E")
