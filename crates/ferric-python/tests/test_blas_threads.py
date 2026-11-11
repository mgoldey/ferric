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


def _loaded_openblas():
    """ctypes handle on the OpenBLAS this process loaded (Linux only)."""
    import ctypes

    try:
        with open("/proc/self/maps") as fh:
            paths = {ln.split()[-1] for ln in fh if "openblas" in ln and "/" in ln}
    except OSError:
        return None
    if len(paths) != 1:
        return None
    lib = ctypes.CDLL(paths.pop())
    lib.openblas_get_config.restype = ctypes.c_char_p
    return lib


def test_reported_count_is_what_openblas_runs():
    """A request above the OpenBLAS build's MAX_THREADS is reported as the
    count OpenBLAS actually runs. The 0.1.0rc11 wheel reported 12 here while
    its OpenBLAS (MAX_THREADS=4) ran 4."""
    import re

    lib = _loaded_openblas()
    if lib is None:
        pytest.skip("cannot locate the loaded OpenBLAS (needs /proc/self/maps)")
    config = lib.openblas_get_config().decode()
    m = re.search(r"MAX_THREADS=(\d+)", config)
    assert m, config
    cap = int(m.group(1))
    huge = 100_000
    try:
        ferric.set_blas_threads(huge)
        reported = ferric.blas_threads()
    finally:
        ferric.set_blas_threads(None)
    assert reported == cap, (
        f"blas_threads() = {reported}, OpenBLAS cap {cap} ({config})"
    )
    before = lib.openblas_get_num_threads()
    lib.openblas_set_num_threads(reported)
    runs = lib.openblas_get_num_threads()
    lib.openblas_set_num_threads(before)
    assert runs == reported
