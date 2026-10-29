"""Secondary-basis (Q-Chem BASIS2) guess through the Python bindings.

The physics is pinned in crates/ferric-scf/tests/basis2_guess.rs; this file
checks the binding surface: the kwargs reach the solver, the report is echoed,
the converged energy is the default-guess energy, and bad input raises.
"""

import ferric
import pytest

WATER = "3\n\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n"
NH2 = "3\n\nN 0 0 0.1404\nH 0 0.8019 -0.4914\nH 0 -0.8019 -0.4914\n"
TIGHT = {"density_conv": 1e-8}


@pytest.fixture
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture
def nh2():
    return ferric.Molecule.from_xyz_string(NH2, 0, 2)


@pytest.fixture
def dz():
    return ferric.BasisSet.bundled("cc-pvdz")


@pytest.mark.parametrize("projection", ["fock", "overlap"])
def test_run_rhf_echoes_report_and_keeps_energy(water, dz, projection):
    ref = ferric.run_rhf(water, dz, **TIGHT)
    r = ferric.run_rhf(
        water, dz, guess_basis="6-31g", guess_projection=projection, **TIGHT
    )
    assert ref.guess_basis is None
    rep = r.guess_basis
    assert rep["basis"].lower() == "6-31g"
    assert rep["projection"] == projection
    assert rep["nbasis_guess"] == 13 and rep["nbasis_target"] == 24
    assert rep["iterations"] > 0 and rep["energy"] > r.energy
    assert rep["converged"] is True and rep["used_as_guess"] is True
    assert rep["unconverged_reason"] is None
    assert abs(r.energy - ref.energy) < 1e-8


@pytest.mark.parametrize("runner", ["run_uhf", "run_rohf"])
def test_open_shell_runners_take_the_guess(nh2, dz, runner):
    run = getattr(ferric, runner)
    ref = run(nh2, dz, **TIGHT)
    r = run(nh2, dz, guess_basis="6-31g", **TIGHT)
    assert r.guess_basis["projection"] == "fock"
    assert abs(r.energy - ref.energy) < 1e-8


def test_run_dft_takes_the_guess(water, dz):
    ref = ferric.run_dft(water, dz, functional="PBE", **TIGHT)
    r = ferric.run_dft(water, dz, functional="PBE", guess_basis="6-31g", **TIGHT)
    assert r.guess_basis["basis"].lower() == "6-31g"
    assert abs(r.total_energy - ref.total_energy) < 1e-8


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"guess_basis": "6-31g", "guess_projection": "fop"}, "guess_projection"),
        ({"guess_basis": "not-a-basis"}, "guess_basis"),
        ({"guess_projection": "overlap"}, "guess_basis is not"),
    ],
)
def test_bad_kwargs_raise(water, dz, kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ferric.run_rhf(water, dz, **kwargs)


def test_unconverged_secondary_is_reported_not_raised():
    # STO-3G UHF CN oscillates for 200 iterations; the target starts from the
    # default guess instead and converges to the default-guess state.
    cn = ferric.Molecule.from_xyz_string("2\n\nC 0 0 0\nN 0 0 1.172\n", 0, 2)
    adz = ferric.BasisSet.bundled("aug-cc-pvdz")
    ref = ferric.run_uhf(cn, adz, **TIGHT)
    r = ferric.run_uhf(cn, adz, guess_basis="sto-3g", **TIGHT)
    rep = r.guess_basis
    assert rep["converged"] is False and rep["used_as_guess"] is False
    assert "iterations" in rep["unconverged_reason"]
    assert abs(r.energy - ref.energy) < 1e-8
