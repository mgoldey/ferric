"""IEF-PCM implicit solvation, reachable from Python.

Without it a ligand and its pocket each leave water for free, so burying an
ionized group costs nothing. MEASURED on danuglipron/GLP-1R: the vacuum
electrostatic interaction overshoots the experimental binding free energy
(~-10 kcal/mol at 80 nM) by 4.5x.
"""

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = str(
    Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
)


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz(WATER)
    return mol, ferric.BasisSet.bundled("sto-3g")


def test_solvent_none_is_BIT_IDENTICAL_to_vacuum(water):
    """The trivial-limit anchor: no solvent must change nothing at all."""
    mol, bs = water
    assert (
        ferric.run_rhf(mol, bs, solvent=None).energy == ferric.run_rhf(mol, bs).energy
    )


def test_water_stabilises_by_the_expected_magnitude(water):
    """Validated elsewhere at -3.813 vs PySCF IEF-PCM -3.8228 kcal/mol."""
    mol, bs = water
    vac = ferric.run_rhf(mol, bs).energy
    aq = ferric.run_rhf(mol, bs, solvent="water").energy
    dG = (aq - vac) * 627.5095
    assert aq < vac, "solvation must stabilise a polar molecule"
    assert -6.0 < dG < -1.0, f"water/STO-3G solvation {dG:.3f}, expected ~-3.6"


def test_a_name_and_its_dielectric_agree(water):
    mol, bs = water
    assert (
        abs(
            ferric.run_rhf(mol, bs, solvent="water").energy
            - ferric.run_rhf(mol, bs, solvent=78.4).energy
        )
        < 1e-12
    )


def test_an_unknown_solvent_ERRORS_rather_than_running_in_vacuum(water):
    """Silently falling back to vacuum looks like a successful solvated run."""
    mol, bs = water
    with pytest.raises(ValueError, match="not recognised"):
        ferric.run_rhf(mol, bs, solvent="watr")


def test_a_dielectric_below_vacuum_is_REFUSED(water):
    mol, bs = water
    with pytest.raises(ValueError, match="must be > 1.0"):
        ferric.run_rhf(mol, bs, solvent=0.5)


@pytest.mark.parametrize("eps", [float("nan"), float("inf")])
def test_a_NON_FINITE_dielectric_is_REFUSED(water, eps):
    """`nan <= 1.0` and `inf <= 1.0` are both False, so a bare `<=` let them in."""
    mol, bs = water
    with pytest.raises(ValueError, match="finite"):
        ferric.run_rhf(mol, bs, solvent=eps)


@pytest.mark.parametrize("order", [0, 7, 194])
def test_an_unsupported_lebedev_order_is_a_ValueError(water, order):
    """Refused at the kwarg, not as a RuntimeError from RHF setup."""
    mol, bs = water
    with pytest.raises(ValueError, match="pcm_lebedev_order"):
        ferric.run_rhf(mol, bs, solvent="water", pcm_lebedev_order=order)


def test_a_higher_dielectric_stabilises_more(water):
    """Monotonicity: the ordering is physics, not a fitted constant."""
    mol, bs = water
    e = [ferric.run_rhf(mol, bs, solvent=eps).energy for eps in (2.38, 20.7, 78.4)]
    assert e[0] > e[1] > e[2], f"not monotone in dielectric: {e}"
