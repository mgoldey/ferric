"""`run_dft` must let a caller turn density fitting OFF.

`run_rhf` has always exposed `df_j_aux`/`df_k_aux`; `run_dft` hardcoded RI-J
with no override. A caller comparing ferric against an exact-Coulomb reference
(ORCA `NORI`, PySCF without `density_fit()`) therefore measured the RI-J
FITTING ERROR and read it as a ferric defect.

MEASURED at PBE/STO-3G against conventional J: water 0.28, benzene 1.16, and a
71-atom drug molecule 9.5 kcal/mol -- the error grows with system size, so the
bigger the system the more convincing the false "defect" looks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = str(
    Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
)
HARTREE_TO_KCAL = 627.5095


@pytest.fixture(scope="module")
def setup():
    return ferric.Molecule.from_xyz(WATER), ferric.BasisSet.bundled("sto-3g")


def test_the_default_is_unchanged(setup):
    """Anchor: omitting the kwarg must behave exactly as before."""
    mol, bs = setup
    assert (
        ferric.run_dft(mol, bs, "PBE").total_energy
        == ferric.run_dft(mol, bs, "PBE", df_j_aux=None).total_energy
    )


def test_turning_density_fitting_OFF_changes_the_energy(setup):
    """The bug: the kwarg existed but the auto-default overrode it.

    `None` meant "unset, so auto-default" in the SCF layer, leaving no way to
    say "do not density-fit". An accepted-but-ignored kwarg is worse than none.
    """
    mol, bs = setup
    rij = ferric.run_dft(mol, bs, "PBE").total_energy
    exact = ferric.run_dft(mol, bs, "PBE", df_j_aux="", df_k_aux="").total_energy
    err = (rij - exact) * HARTREE_TO_KCAL
    assert abs(err) > 0.05, (
        f"RI-J and exact J differ by only {err:.4f} kcal/mol -- the opt-out "
        "did not reach the SCF"
    )
    # Water's RI-J error is ~0.28 kcal/mol; a wildly different value means
    # something other than density fitting changed.
    assert abs(err) < 2.0, f"implausible RI-J error {err:.3f} kcal/mol"


@pytest.mark.parametrize("spelling", ["", "none", "off", "exact", "conventional"])
def test_every_opt_out_spelling_agrees(setup, spelling):
    mol, bs = setup
    ref = ferric.run_dft(mol, bs, "PBE", df_j_aux="").total_energy
    assert ferric.run_dft(mol, bs, "PBE", df_j_aux=spelling).total_energy == ref


def test_an_explicit_basis_name_is_honoured(setup):
    """A named aux basis must not be swallowed by the default."""
    mol, bs = setup
    named = ferric.run_dft(mol, bs, "PBE", df_j_aux="def2-universal-jkfit").total_energy
    assert named == ferric.run_dft(mol, bs, "PBE").total_energy


def test_exact_J_moves_TOWARD_an_independent_reference(setup):
    """The point of the switch: it should improve agreement, not just differ.

    PySCF conventional-J PBE/STO-3G on this geometry is -75.225640 Ha. Exact J
    must land closer to that than RI-J does.
    """
    mol, bs = setup
    reference = -75.225640
    rij = ferric.run_dft(mol, bs, "PBE").total_energy
    exact = ferric.run_dft(mol, bs, "PBE", df_j_aux="", df_k_aux="").total_energy
    assert abs(exact - reference) < abs(rij - reference), (
        f"exact J ({exact:.6f}) is not closer to PySCF ({reference}) than "
        f"RI-J ({rij:.6f})"
    )


def test_RSH_refuses_a_DF_K_opt_out_by_name(setup):
    """RSH exchange exists only as RI-K, so `df_k_aux="off"` cannot be honoured.

    It used to reach `basis::bundled("")` and fail with an unrelated basis
    error. It must instead say what was refused and why.
    """
    mol, bs = setup
    with pytest.raises(RuntimeError, match="range-separated"):
        ferric.run_dft(mol, bs, "wB97X-V", df_k_aux="off")


def test_RSH_still_honours_a_DF_J_opt_out(setup):
    """Only the K opt-out is unsupported for RSH; exact J must still work."""
    mol, bs = setup
    rij = ferric.run_dft(mol, bs, "wB97X-V").total_energy
    exact = ferric.run_dft(mol, bs, "wB97X-V", df_j_aux="off").total_energy
    assert rij != exact
