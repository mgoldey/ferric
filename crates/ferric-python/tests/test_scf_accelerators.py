"""AURORA and TRAH are settable from Python (#418), opt-in, same fixed point as DIIS.

Both are OFF unless asked. The checks: the keyword reaches the solver (the iteration
count changes), the energy agrees with DIIS to the SCF tolerance, bad values are typed
errors, and the default call is unchanged by the new keywords' existence.
"""

from __future__ import annotations

import ferric
import pytest

TOL = 1e-8
KW = dict(energy_conv=1e-10, density_conv=1e-8, max_iter=200)


def _mol(name: str, mult: int = 1):
    return ferric.Molecule.from_xyz(f"testdata/molecules/{name}.xyz", 0, mult)


def _bs(name: str = "sto-3g"):
    return ferric.BasisSet.bundled(name)


@pytest.fixture(scope="module")
def diis():
    return ferric.run_rhf(_mol("water"), _bs("6-31g"), **KW)


def test_aurora_reaches_the_diis_fixed_point(diis):
    r = ferric.run_rhf(_mol("water"), _bs("6-31g"), aurora=True, **KW)
    assert r.converged
    assert abs(r.energy - diis.energy) < TOL
    assert r.iterations != diis.iterations, "aurora=True did not reach the solver"


def test_trah_reaches_the_diis_fixed_point(diis):
    r = ferric.run_rhf(_mol("water"), _bs("6-31g"), trah_trigger=1e-1, **KW)
    assert r.converged
    assert abs(r.energy - diis.energy) < TOL
    assert r.iterations != diis.iterations, "trah_trigger did not reach the solver"


def test_uhf_trah_reaches_the_diis_fixed_point():
    base = ferric.run_uhf(_mol("water"), _bs("6-31g"), **KW)
    r = ferric.run_uhf(_mol("water"), _bs("6-31g"), trah_trigger=1e-1, **KW)
    assert r.converged and abs(r.energy - base.energy) < TOL
    assert r.iterations != base.iterations


def test_defaults_are_off(diis):
    r = ferric.run_rhf(
        _mol("water"), _bs("6-31g"), aurora=False, trah_trigger=None, **KW
    )
    assert r.iterations == diis.iterations
    assert r.energy == pytest.approx(diis.energy, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_trah_trigger_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="trah_trigger"):
        ferric.run_rhf(_mol("water"), _bs(), trah_trigger=bad)
    with pytest.raises(ValueError, match="trah_trigger"):
        ferric.run_uhf(_mol("water"), _bs(), trah_trigger=bad)
