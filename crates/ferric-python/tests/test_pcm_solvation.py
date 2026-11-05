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


def test_optimize_with_solvent_follows_the_pcm_gradient(water):
    """run_optimize(solvent=...) walks the solvated surface: its minimum is
    below the vacuum geometry's solvated energy, differs from the vacuum
    minimum, and the certificate names the scf pcm approximation."""
    mol, _ = water
    vac = ferric.run_optimize(mol, "sto-3g", scf_energy_conv=1e-9, scf_density_conv=1e-8)
    aq = ferric.run_optimize(
        mol, "sto-3g", solvent="water", scf_energy_conv=1e-9, scf_density_conv=1e-8
    )
    assert vac.converged and aq.converged
    # Stationary on the solvated surface: re-solvating the vacuum geometry
    # cannot beat the solvated optimum.
    e_vac_geom_solvated = ferric.run_rhf(
        vac.mol(), ferric.BasisSet.bundled("sto-3g"), solvent="water"
    ).energy
    assert aq.energy <= e_vac_geom_solvated + 1e-8
    assert aq.energy < vac.energy, "solvation must stabilise the optimum"
    # The geometry actually moved relative to the vacuum optimum.
    assert abs(aq.energy - vac.energy) > 1e-4
    names = [a["name"] for a in aq.certificate["approximations"]]
    assert "scf pcm" in names
    assert "scf pcm" not in [a["name"] for a in vac.certificate["approximations"]]


def test_optimize_solvent_none_is_bit_identical_to_vacuum(water):
    """Trivial-limit anchor: no solvent changes nothing."""
    mol, _ = water
    a = ferric.run_optimize(mol, "sto-3g", max_steps=3)
    b = ferric.run_optimize(mol, "sto-3g", max_steps=3, solvent=None)
    assert a.energy == b.energy
    assert a.energy_trace == b.energy_trace


def test_optimize_uhf_with_solvent(water):
    mol = ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n", 0, 1
    )
    aq = ferric.run_optimize_uhf(
        mol, "sto-3g", solvent=78.4, pcm_lebedev_order=110, max_steps=60
    )
    assert aq.converged
    assert "scf pcm" in [a["name"] for a in aq.certificate["approximations"]]
    # Closed-shell UHF equals RHF on the same solvated surface.
    rhf = ferric.run_optimize(
        mol, "sto-3g", solvent=78.4, pcm_lebedev_order=110, max_steps=60
    )
    assert abs(aq.energy - rhf.energy) < 1e-6


def test_optimize_refuses_a_bad_solvent_like_run_rhf(water):
    mol, _ = water
    with pytest.raises(ValueError, match="not recognised"):
        ferric.run_optimize(mol, "sto-3g", solvent="watr")
    with pytest.raises(ValueError, match="pcm_lebedev_order"):
        ferric.run_optimize(mol, "sto-3g", solvent="water", pcm_lebedev_order=7)
