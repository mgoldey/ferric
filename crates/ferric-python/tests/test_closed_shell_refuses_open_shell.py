"""Closed-shell entry points must REFUSE an open-shell molecule.

`solve_rhf` occupied `nelec / 2` orbitals and never read the multiplicity.
With an EVEN electron count and multiplicity > 1 it converged the closed-shell
SINGLET and every closed-shell binding returned that singlet as the answer.
MEASURED before the fix (water, STO-3G, cc-pvdz-ri aux):

    run_rimp2(singlet).total_energy = -74.99874957952717
    run_rimp2(triplet).total_energy = -74.99874957952717   # identical
    run_rhf(triplet).energy         = -74.96314680003904   # the singlet RHF
    run_dft(triplet, "PBE")         = -75.22616747570409   # the singlet RKS

With an ODD count the failure was loud but misleading: `run_dft` on the OH
doublet raised "SCF did not converge after 0 iterations".

The guard lives in `ferric_scf::rhf::require_closed_shell`, called first thing
in `solve_rhf`, so it covers every binding that builds its reference there
(run_rhf, run_dft/run_ksdft, the whole RI-MP2 family, CC, RPA, GW, BSE,
TDDFT, run_optimize, run_frequencies(reference="rhf"), ...). This file checks
a representative, cheap subset.

HOW THESE FAIL IF THE FIX IS REVERTED: the `pytest.raises` blocks do not raise
(the triplet quietly returns the singlet energy), and the OH case raises a
message about SCF convergence instead of one naming the multiplicity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

MOLECULES = Path(__file__).resolve().parents[3] / "testdata" / "molecules"
WATER = str(MOLECULES / "water.xyz")
OH = str(MOLECULES / "oh.xyz")


@pytest.fixture(scope="module")
def bases():
    return ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")


@pytest.fixture(scope="module")
def triplet_water():
    return ferric.Molecule.from_xyz(WATER, 0, 3)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m, bs, aux: ferric.run_rhf(m, bs), id="run_rhf"),
        pytest.param(lambda m, bs, aux: ferric.run_dft(m, bs, "PBE"), id="run_dft"),
        pytest.param(lambda m, bs, aux: ferric.run_rimp2(m, bs, aux), id="run_rimp2"),
        pytest.param(lambda m, bs, aux: ferric.run_mp3(m, bs, aux), id="run_mp3"),
        pytest.param(
            lambda m, bs, aux: ferric.run_scs_mp2(m, bs, aux), id="run_scs_mp2"
        ),
        pytest.param(lambda m, bs, aux: ferric.run_ccsd(m, bs, aux), id="run_ccsd"),
    ],
)
def test_triplet_water_is_refused_not_answered_as_the_singlet(
    call, triplet_water, bases
):
    bs, aux = bases
    with pytest.raises(RuntimeError, match="multiplicity 3"):
        call(triplet_water, bs, aux)


def test_odd_electron_KS_names_the_multiplicity_not_convergence(bases):
    """The OH doublet used to fail as "SCF did not converge after 0 iterations"."""
    bs, _ = bases
    oh = ferric.Molecule.from_xyz(OH, 0, 2)
    with pytest.raises(RuntimeError) as exc:
        ferric.run_dft(oh, bs, "PBE")
    msg = str(exc.value)
    assert "multiplicity 2" in msg, msg
    assert "did not converge" not in msg, msg


def test_the_singlet_and_the_open_shell_paths_still_run(triplet_water, bases):
    """Reachability anchor: the guard must refuse ONLY the open-shell case.

    A guard that refused everything would pass every test above. The singlet
    must still run through the closed-shell path, and the triplet must still
    run through the open-shell one (and land on a DIFFERENT state from the
    singlet, which is the whole point of refusing to substitute one for the
    other).
    """
    bs, aux = bases
    singlet = ferric.Molecule.from_xyz(WATER)
    e_rhf = ferric.run_rhf(singlet, bs).energy
    assert ferric.run_rimp2(singlet, bs, aux).total_energy < e_rhf
    e_uhf_triplet = ferric.run_uhf(triplet_water, bs).energy
    assert abs(e_uhf_triplet - e_rhf) > 1e-2, (e_uhf_triplet, e_rhf)
