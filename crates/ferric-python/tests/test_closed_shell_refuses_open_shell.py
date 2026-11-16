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
(run_rhf, run_dft/run_ksdft, the RI-MP2 family bar run_rimp2, CC, RPA, GW, BSE,
TDDFT, run_optimize, run_frequencies(reference="rhf"), ...). This file checks
a representative, cheap subset.

HOW THESE FAIL IF THE FIX IS REVERTED: the `pytest.raises` blocks do not raise
(the triplet quietly returns the singlet energy), and the OH case raises a
message about SCF convergence instead of one naming the multiplicity.

`run_rimp2` is NOT refused: an open-shell molecule gets a UHF reference and
unrestricted RI-MP2 (UMP2), checked against PySCF at the bottom of this file.
Nor are run_attenuated_rimp2, run_terfc_rimp2, run_scs_mp2,
run_scs_mp2_2terfc and run_rs_mp2_rpa, which take the same UHF route
(checked against run_rimp2's UMP2 below).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

TESTDATA = Path(__file__).resolve().parents[3] / "testdata"
MOLECULES = TESTDATA / "molecules"
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
        pytest.param(lambda m, bs, aux: ferric.run_mp3(m, bs, aux), id="run_mp3"),
        pytest.param(lambda m, bs, aux: ferric.run_ccd(m, bs, aux), id="run_ccd"),
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


# The singlet RI-MP2 energy the triplet used to come back as (water, STO-3G,
# cc-pvdz-ri aux), quoted in the module docstring.
SINGLET_RIMP2 = -74.99874957952717


def test_run_rimp2_on_triplet_water_is_unrestricted_not_the_singlet(
    triplet_water, bases
):
    """Triplet water gets UHF + UMP2, not the singlet and not a refusal.

    Fails if the open-shell route is removed (the call raises "multiplicity
    3"), if it ever reverts to the closed-shell kernel (the total equals
    SINGLET_RIMP2), or if the reference is not the plain UHF `run_uhf` solves
    (the `rhf_energy` check).
    """
    bs, aux = bases
    r = ferric.run_rimp2(triplet_water, bs, aux)
    assert r.reference == "UHF"
    assert abs(r.total_energy - SINGLET_RIMP2) > 1e-2, r.total_energy
    assert r.mp2_corr < 0.0
    e_uhf = ferric.run_uhf(triplet_water, bs).energy
    assert abs(r.rhf_energy - e_uhf) < 1e-6, (r.rhf_energy, e_uhf)
    assert abs(r.total_energy - (r.rhf_energy + r.mp2_corr)) < 1e-10
    # The singlet keeps the closed-shell route and says so.
    singlet = ferric.run_rimp2(ferric.Molecule.from_xyz(WATER), bs, aux)
    assert singlet.reference == "RHF"


def test_run_rimp2_oh_doublet_matches_pyscf_ump2(tmp_path):
    """OH/cc-pVDZ (0.97 A), cc-pvdz-ri: UHF and DF-UMP2 vs the PySCF harness.

    Reference: testdata/reference/oh_cc-pvdz_u-oomp2-fd.json (e_hf, e_corr;
    scripts/gen_pyscf_u_oo_rimp2_ref.py, same aux basis). The ROHF-based
    variant of the same kernel is ~6e-3 Ha off on this system
    (ferric-cc/tests/semicanonical_mp2.rs), so the 1e-4 bar distinguishes the
    reference as well as the method.
    """
    ref = json.loads(
        (TESTDATA / "reference" / "oh_cc-pvdz_u-oomp2-fd.json").read_text()
    )
    xyz = tmp_path / "oh_097.xyz"
    xyz.write_text("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n")
    oh = ferric.Molecule.from_xyz(str(xyz), 0, 2)
    r = ferric.run_rimp2(
        oh, ferric.BasisSet.bundled("cc-pvdz"), ferric.BasisSet.bundled("cc-pvdz-ri")
    )
    assert r.reference == "UHF"
    assert abs(r.rhf_energy - ref["e_hf"]) < 1e-6, (r.rhf_energy, ref["e_hf"])
    assert abs(r.mp2_corr - ref["e_corr"]) < 1e-4, (r.mp2_corr, ref["e_corr"])


def test_run_rimp2_open_shell_refuses_kappa(triplet_water, bases):
    """kappa-MP2 is closed-shell only; dropping it silently would print UMP2."""
    bs, aux = bases
    with pytest.raises(ValueError, match="kappa"):
        ferric.run_rimp2(triplet_water, bs, aux, kappa=1.0)


def test_attenuated_and_scs_family_take_the_uhf_route(triplet_water, bases):
    """Triplet water: the attenuated / SCS / RS-MP2-RPA bindings run UHF.

    Independent route: run_rimp2's UMP2 (PySCF-checked above). SCS reads
    E_OS = E_ab, E_SS = E_aa + E_bb, so e_os + e_ss is the UMP2 correlation
    (rounding only, 1e-10). As omega -> 0 erfc(omega r)/r -> 1/r, so the
    attenuated energy at omega = 1e-4 A^-1 approaches UMP2 (the gap is O(omega),
    measured < 1e-7 Ha on OH/cc-pVDZ in ferric-mp2 open_shell_attenuated.rs;
    bar 1e-6). RS-MP2-RPA (delta-lr) at omega = 0.02 A^-1 adds a long-range
    ring term of third order in erf(omega r)/r ~ 2 omega / sqrt(pi), far below
    1e-6 Ha. Fails if any binding refuses the triplet, reports "RHF", or
    returns the singlet.
    """
    bs, aux = bases
    ump2 = ferric.run_rimp2(triplet_water, bs, aux)
    assert ump2.reference == "UHF"
    scs = ferric.run_scs_mp2(triplet_water, bs, aux, c_os=1.2, c_ss=1.0 / 3.0)
    assert scs.reference == "UHF"
    assert abs(scs.e_os + scs.e_ss - ump2.mp2_corr) < 1e-10, (scs.e_os, scs.e_ss)
    assert abs(scs.scs_corr - (1.2 * scs.e_os + scs.e_ss / 3.0)) < 1e-12
    assert abs(scs.rhf_energy - ump2.rhf_energy) < 1e-8
    att = ferric.run_attenuated_rimp2(triplet_water, bs, aux, omega=1e-4)
    assert att.reference == "UHF"
    assert abs(att.mp2_corr - ump2.mp2_corr) < 1e-6, (att.mp2_corr, ump2.mp2_corr)
    att0 = ferric.run_attenuated_rimp2(triplet_water, bs, aux)
    assert abs(att0.mp2_corr - ump2.mp2_corr) > 1e-3  # the attenuation is real
    # run_rs_mp2_rpa defaults to an RI-JK reference; "exact" gives the same
    # UHF run_rimp2 builds.
    rs = ferric.run_rs_mp2_rpa(triplet_water, bs, aux, omega=0.02, scf_jk="exact")
    assert rs.reference == "UHF"
    assert abs(rs.e_mp2_full - ump2.mp2_corr) < 1e-10
    assert abs(rs.e_corr - ump2.mp2_corr) < 1e-6, (rs.e_corr, ump2.mp2_corr)
    singlet = ferric.Molecule.from_xyz(WATER)
    assert ferric.run_scs_mp2(singlet, bs, aux).reference == "RHF"
    assert ferric.run_rs_mp2_rpa(singlet, bs, aux).reference == "RHF"
