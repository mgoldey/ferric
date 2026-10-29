"""`run_mp2_v_and_scs_2terfc` (one SCF) must equal the per-method bindings.

EXACTNESS ANCHOR for the S66 shared-SCF entry point (mgoldey/ferric#278).
`run_mp2_v`, `run_scs_mp2_2terfc` and `run_rimp2` each run their own SCF with
the library-default config; the shared binding runs one. Every energy must
match BITWISE, and `==` (not a tolerance) is safe because every parallel
reduction on these paths has a fixed order by construction, independent of
thread count and work stealing: the J/K builders fold group partials in
ascending group order (ferric_scf::reduce), RI-MP2 sums its per-i partials
from an index-ordered collect (rimp2.rs), and the VV10 pair sum folds its
rows serially. Same build, same process, same inputs => same bits. (Across
machines/builds it would not be: see the golden-energy memory; this test never
compares across those.)

Artifact hypothesis: a wrong frozen_core, a swapped r0, a units slip (A vs
Bohr) or a different SCF config in the shared binding moves the energies by
>= 1e-6 Ha and fails the `==` here. Water dimer is used so the same-spin
space is real (nocc > 1) and frozen core (2) is non-trivial.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

ROOT = Path(__file__).resolve().parents[3]
TERF = ROOT / "terf-tables"

pytestmark = pytest.mark.skipif(
    not hasattr(ferric, "run_mp2_v_and_scs_2terfc"),
    reason="loaded ferric build predates run_mp2_v_and_scs_2terfc",
)


@pytest.fixture(scope="module", autouse=True)
def _terf_tables():
    if (
        not (TERF / "16_4_2.bin").is_file()
        and "FERRIC_TERF_TABLE_DIR" not in os.environ
    ):
        pytest.skip("terfc interpolation tables unavailable")
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF))


@pytest.fixture(scope="module")
def system():
    mol = ferric.Molecule.from_xyz_string(
        (ROOT / "testdata/molecules/s22/water_dimer.xyz").read_text()
    )
    return (
        mol,
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def test_shared_scf_matches_the_three_per_method_bindings_bitwise(system):
    mol, bs, aux = system
    fc = 2
    shared = ferric.run_mp2_v_and_scs_2terfc(mol, bs, aux, frozen_core=fc)
    mp2v = ferric.run_mp2_v(mol, bs, aux, frozen_core=fc)
    scs = ferric.run_scs_mp2_2terfc(mol, bs, aux, frozen_core=fc)
    mp2 = ferric.run_rimp2(mol, bs, aux, frozen_core=fc)
    pairs = {
        "rhf_energy": (shared["rhf_energy"], mp2v.rhf_energy),
        "rhf_energy(scs)": (shared["rhf_energy"], scs.rhf_energy),
        "rhf_energy(mp2)": (shared["rhf_energy"], mp2.rhf_energy),
        "mp2v_total": (shared["mp2v_total"], mp2v.total_energy),
        "mp2v_att_mp2_corr": (shared["mp2v_att_mp2_corr"], mp2v.att_mp2_corr),
        "mp2v_vv10_e_nl": (shared["mp2v_vv10_e_nl"], mp2v.vv10_e_nl),
        "mp2v_e_os": (shared["mp2v_e_os"], mp2v.e_os),
        "mp2v_e_ss": (shared["mp2v_e_ss"], mp2v.e_ss),
        "scs_total": (shared["scs_total"], scs.total_energy),
        "scs_corr": (shared["scs_corr"], scs.scs_corr),
        "scs_e_os": (shared["scs_e_os"], scs.e_os),
        "scs_e_ss": (shared["scs_e_ss"], scs.e_ss),
        "mp2_total": (shared["mp2_total"], mp2.total_energy),
        "mp2_corr": (shared["mp2_corr"], mp2.mp2_corr),
    }
    bad = {k: (a, b, a - b) for k, (a, b) in pairs.items() if a != b}
    assert not bad, f"shared != separate: {bad}"
    assert shared["n_nlc_points"] == mp2v.n_nlc_points
    # Non-vacuity: three genuinely different methods.
    assert abs(shared["mp2v_total"] - shared["scs_total"]) > 1e-4
    assert abs(shared["mp2_total"] - shared["scs_total"]) > 1e-4
    t = shared["timings"]
    assert set(t) == {"setup_s", "scf_s", "mp2v_s", "scs_s", "mp2_s"}
    assert all(v >= 0.0 for v in t.values()) and t["scf_s"] > 0.0


def test_density_conv_is_threaded_and_validated(system):
    mol, bs, aux = system
    loose = ferric.run_mp2_v_and_scs_2terfc(
        mol, bs, aux, frozen_core=2, include_coulomb_mp2=False
    )
    tight = ferric.run_mp2_v_and_scs_2terfc(
        mol, bs, aux, frozen_core=2, include_coulomb_mp2=False, density_conv=1e-9
    )
    assert loose["density_conv"] == 1e-6 and tight["density_conv"] == 1e-9
    assert loose["mp2_total"] is None and tight["timings"]["mp2_s"] == 0.0
    assert tight["scf_iterations"] > loose["scf_iterations"]
    assert abs(tight["scs_total"] - loose["scs_total"]) < 1e-6
    with pytest.raises(ValueError, match="density_conv"):
        ferric.run_mp2_v_and_scs_2terfc(mol, bs, aux, density_conv=0.0)


def test_run_rimp2_density_conv_kwarg(system):
    mol, bs, aux = system
    loose = ferric.run_rimp2(mol, bs, aux, frozen_core=2)
    tight = ferric.run_rimp2(mol, bs, aux, frozen_core=2, density_conv=1e-9)
    assert loose.rhf_energy != tight.rhf_energy  # the kwarg reached the SCF
    assert abs(tight.total_energy - loose.total_energy) < 1e-6
    with pytest.raises(ValueError, match="density_conv"):
        ferric.run_rimp2(mol, bs, aux, density_conv=-1.0)
    with pytest.raises(ValueError, match="density_conv"):
        ferric.run_rimp2(
            mol, bs, aux, local="amplitude-threshold", eps=1e-4, density_conv=1e-9
        )
