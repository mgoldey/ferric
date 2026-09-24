"""`run_rhf` / `run_uhf` / `run_rohf` must read `df_j_aux` like `run_dft` does.

`run_dft` resolves "" / "exact" / "none" / "off" / "conventional" to
conventional four-centre integrals. The HF bindings passed the string through
raw, so the same spelling was a basis-name lookup there. MEASURED before the
fix (water, STO-3G):

    run_rhf(df_j_aux="")      -> -74.96314680003904
    run_rhf(df_j_aux="exact") -> RuntimeError: basis error: unknown bundled basis: exact

All surfaces (the three HF bindings, run_dft, and the CLI `[scf]` keys) now
share `ferric_scf::rhf::normalize_df_aux`.

HOW THESE FAIL IF THE FIX IS REVERTED: the parametrized spellings raise the
"unknown bundled basis" RuntimeError instead of returning the exact-J energy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

MOLECULES = Path(__file__).resolve().parents[3] / "testdata" / "molecules"


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz(str(MOLECULES / "water.xyz"))


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def rhf_exact(water, sto3g):
    return ferric.run_rhf(water, sto3g, df_j_aux="", df_k_aux="").energy


@pytest.mark.parametrize("spelling", ["exact", "none", "off", "conventional", "NONE"])
def test_run_rhf_reads_every_opt_out_spelling_as_exact(
    water, sto3g, rhf_exact, spelling
):
    e = ferric.run_rhf(water, sto3g, df_j_aux=spelling, df_k_aux=spelling).energy
    assert e == rhf_exact


def test_open_shell_bindings_share_the_parser(sto3g):
    oh = ferric.Molecule.from_xyz(str(MOLECULES / "oh.xyz"), 0, 2)
    for run in (ferric.run_uhf, ferric.run_rohf):
        ref = run(oh, sto3g, df_j_aux="").energy
        assert run(oh, sto3g, df_j_aux="exact").energy == ref, run.__name__


def test_a_named_basis_and_the_default_are_unchanged(water, sto3g, rhf_exact):
    """Reachability anchor: a real aux name must still density-fit.

    If every string were mapped to "", the spelling tests above would pass
    trivially. A named JK-fit basis must give a DIFFERENT (RI) energy, and an
    omitted kwarg must stay the plain-HF exact energy.
    """
    ri = ferric.run_rhf(
        water,
        sto3g,
        df_j_aux="def2-universal-jkfit",
        df_k_aux="def2-universal-jkfit",
    ).energy
    assert ri != rhf_exact
    assert abs(ri - rhf_exact) < 1e-3
    assert ferric.run_rhf(water, sto3g).energy == rhf_exact
