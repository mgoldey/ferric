"""`ferric.set_blas_threads` / `ferric.blas_threads`: the serial-section
OpenBLAS thread count (Fock diagonalization, DIIS)."""

import os

import pytest

import ferric


def test_explicit_setting_is_honoured_and_none_restores_the_default():
    default = ferric.blas_threads()
    assert default >= 1
    try:
        ferric.set_blas_threads(1)
        assert ferric.blas_threads() == 1
        ferric.set_blas_threads(3)
        assert ferric.blas_threads() == 3
    finally:
        ferric.set_blas_threads(None)
    assert ferric.blas_threads() == default


def test_zero_is_refused():
    with pytest.raises(ValueError):
        ferric.set_blas_threads(0)


def test_scf_energy_is_the_same_at_one_and_several_blas_threads():
    testdata = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata")
    mol = ferric.Molecule.from_xyz(os.path.join(testdata, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    try:
        ferric.set_blas_threads(1)
        e1 = ferric.run_rhf(mol, bs).energy
        ferric.set_blas_threads(4)
        e4 = ferric.run_rhf(mol, bs).energy
    finally:
        ferric.set_blas_threads(None)
    assert abs(e1 - e4) < 1e-10
