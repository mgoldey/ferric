"""`scf_jk` / `jk_auxbasis` on the MP2-family entry points.

One shared rule (`ferric_scf::scf_jk`, via `mp2_reference_config` in the
binding) maps `scf_jk="exact"|"rijk"|"rijcosx"` onto the reference SCF's
`df_j_aux` / `df_k_aux` / `k_builder`.

HOW THESE FAIL IF BROKEN: mapping "rijk" to "exact" makes the run_rhf match
and the PySCF anchor miss by the RI-JK fitting error (~3e-5 Ha); dropping
`jk_auxbasis` makes the cc-pvtz-jkfit run equal the default-aux run; putting
the JK aux on the K side of "rijcosx" is refused by the SCF (two exchange
builders); a lax parser stops raising.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
WATER = str(ROOT / "testdata" / "molecules" / "water.xyz")
REF = (
    ROOT / "testdata" / "reference" / "h2o_cc-pvdz_rijk-def2-universal-jkfit_dfmp2.json"
)
JK = "def2-universal-jkfit"

ALL_ENTRY_POINTS = [
    "run_rimp2",
    "run_oo_rimp2",
    "run_laplace_mp2",
    "run_laplace_sos_mp2",
    "run_attenuated_rimp2",
    "run_terfc_rimp2",
    "run_scs_mp2",
    "run_scs_mp2_2terfc",
    "run_mp2_v",
    "run_rs_mp2_rpa",
]
# Entry points that need no terfc tables and report the reference SCF energy.
WITH_RHF_ENERGY = [
    "run_rimp2",
    "run_scs_mp2",
    "run_attenuated_rimp2",
    "run_laplace_sos_mp2",
]


@pytest.fixture(scope="module")
def water():
    return (
        ferric.Molecule.from_xyz(WATER),
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def _rhf(mol, bs, **kw):
    """run_rhf with the MP2 entry points' own SCF defaults (RhfConfig::default)."""
    return ferric.run_rhf(mol, bs, energy_conv=1e-3, density_conv=1e-6, **kw).energy


@pytest.mark.parametrize("name", ["run_rimp2", "run_scs_mp2"])
def test_default_is_exact_bitwise(water, name):
    fn = getattr(ferric, name)
    default = fn(*water)
    exact = fn(*water, scf_jk="exact")
    assert default.total_energy == exact.total_energy
    assert default.rhf_energy == exact.rhf_energy
    assert (default.scf_jk, default.jk_auxbasis) == ("exact", None)


def test_rs_mp2_rpa_default_is_rijk_and_says_so(water):
    default = ferric.run_rs_mp2_rpa(*water)
    rijk = ferric.run_rs_mp2_rpa(*water, scf_jk="rijk")
    assert default.total_energy == rijk.total_energy
    assert (default.scf_jk, default.jk_auxbasis) == ("rijk", JK)
    exact = ferric.run_rs_mp2_rpa(*water, scf_jk="exact")
    assert exact.scf_jk == "exact"
    assert abs(exact.rhf_energy - default.rhf_energy) > 1e-7


@pytest.mark.parametrize("name", WITH_RHF_ENERGY)
def test_rijk_reference_is_run_rhf_with_both_fits(water, name):
    mol, bs, aux = water
    r = getattr(ferric, name)(mol, bs, aux, scf_jk="rijk")
    assert (r.scf_jk, r.jk_auxbasis) == ("rijk", JK)
    ref = _rhf(mol, bs, df_j_aux=JK, df_k_aux=JK)
    assert abs(r.rhf_energy - ref) <= 1e-12, (r.rhf_energy, ref)
    exact = _rhf(mol, bs)
    assert abs(r.rhf_energy - exact) > 1e-7, "rijk ran the four-centre SCF"


def test_rijk_rimp2_matches_pyscf_density_fit_dfmp2(water):
    ref = json.loads(REF.read_text())
    r = ferric.run_rimp2(*water, scf_jk="rijk")
    assert abs(r.rhf_energy - ref["rhf_energy"]) <= 1e-8, (r.rhf_energy, ref)
    assert abs(r.mp2_corr - ref["mp2_corr"]) <= 1e-8, (r.mp2_corr, ref)
    assert abs(r.total_energy - ref["total_energy"]) <= 1e-8


def test_named_jk_auxbasis_reaches_both_fits(water):
    mol, bs, aux = water
    tz = ferric.run_rimp2(mol, bs, aux, scf_jk="rijk", jk_auxbasis="cc-pvtz-jkfit")
    assert tz.jk_auxbasis == "cc-pvtz-jkfit"
    ref = _rhf(mol, bs, df_j_aux="cc-pvtz-jkfit", df_k_aux="cc-pvtz-jkfit")
    assert abs(tz.rhf_energy - ref) <= 1e-12
    default_aux = ferric.run_rimp2(mol, bs, aux, scf_jk="rijk")
    assert abs(tz.rhf_energy - default_aux.rhf_energy) > 1e-8


def test_rijcosx_reference_is_run_rhf_ri_j_plus_cosx(water):
    mol, bs, aux = water
    r = ferric.run_rimp2(mol, bs, aux, scf_jk="rijcosx")
    assert (r.scf_jk, r.jk_auxbasis) == ("rijcosx", JK)
    ref = _rhf(mol, bs, df_j_aux=JK, k_builder="cosx")
    assert abs(r.rhf_energy - ref) <= 1e-12, (r.rhf_energy, ref)
    rijk = _rhf(mol, bs, df_j_aux=JK, df_k_aux=JK)
    exact_j_cosx = _rhf(mol, bs, k_builder="cosx")
    assert abs(r.rhf_energy - rijk) > 1e-8, "COSX exchange not used"
    assert abs(r.rhf_energy - exact_j_cosx) > 1e-8, "RI-J not used"


@pytest.mark.parametrize("name", ALL_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"scf_jk": "RIJK"}, "scf_jk: unknown value"),
        ({"scf_jk": "ri-jk"}, "scf_jk: unknown value"),
        ({"scf_jk": "exact", "jk_auxbasis": JK}, "only read with scf_jk"),
        ({"scf_jk": "rijk", "k_builder": "link"}, "conflicts with scf_jk"),
        ({"scf_jk": "rijk", "k_builder": "direct"}, "conflicts with scf_jk"),
        ({"scf_jk": "rijcosx", "k_builder": "link"}, "conflicts with scf_jk"),
        ({"scf_jk": "rijk", "jk_auxbasis": "exact"}, "turns density fitting off"),
        ({"scf_jk": "rijk", "jk_auxbasis": "no-such-jkfit"}, "no-such-jkfit"),
    ],
)
def test_strict_parse_and_conflicts_raise_value_error(water, name, kwargs, needle):
    with pytest.raises(ValueError, match=needle) as err:
        getattr(ferric, name)(*water, **kwargs)
    assert name in str(err.value)


def test_jk_auxbasis_without_scf_jk_raises_on_exact_default(water):
    with pytest.raises(ValueError, match="only read with scf_jk"):
        ferric.run_rimp2(*water, jk_auxbasis=JK)


def _terf_tables_present() -> bool:
    d = os.environ.get("FERRIC_TERF_TABLE_DIR", "")
    return bool(d) and (Path(d) / "16_4_2.bin").is_file()


needs_terf = pytest.mark.skipif(
    not _terf_tables_present(),
    reason="terfc interpolation tables absent (set FERRIC_TERF_TABLE_DIR to a "
    "directory holding 16_4_2.bin)",
)


def _hf_energy(r) -> float:
    """The reference SCF energy a result carries (LaplaceMp2Result has none,
    so it is total minus correlation)."""
    if hasattr(r, "rhf_energy"):
        return r.rhf_energy
    return r.total_energy - r.mp2_corr


# Every entry point that honours scf_jk, with the kwargs that make it run on
# water/cc-pVDZ. run_oo_rimp2 refuses the fitted modes (tested below).
HONOURING = [
    pytest.param("run_rimp2", {}, id="rimp2"),
    pytest.param(
        "run_rimp2", {"local": "amplitude-threshold", "eps": 1e-6}, id="rimp2-local"
    ),
    pytest.param("run_laplace_mp2", {}, id="laplace_mp2"),
    pytest.param("run_laplace_sos_mp2", {}, id="laplace_sos_mp2"),
    pytest.param("run_attenuated_rimp2", {}, id="attenuated_rimp2"),
    pytest.param("run_scs_mp2", {}, id="scs_mp2"),
    pytest.param("run_rs_mp2_rpa", {}, id="rs_mp2_rpa"),
    pytest.param("run_terfc_rimp2", {}, id="terfc_rimp2", marks=needs_terf),
    pytest.param("run_scs_mp2_2terfc", {}, id="scs_mp2_2terfc", marks=needs_terf),
    pytest.param("run_mp2_v", {}, id="mp2_v", marks=needs_terf),
]


@pytest.fixture(scope="module")
def rhf_refs(water):
    mol, bs, _ = water
    return {
        "exact": _rhf(mol, bs),
        "rijk": _rhf(mol, bs, df_j_aux=JK, df_k_aux=JK),
    }


@pytest.mark.parametrize(("name", "kwargs"), HONOURING)
def test_every_entry_point_runs_the_rijk_reference_it_echoes(
    water, rhf_refs, name, kwargs
):
    """HOW THIS FAILS IF BROKEN: an entry point that builds its SCF config
    without the scf_jk rule (e.g. back on rhf_config_budgeted) returns the
    exact energy while echoing "rijk": the >1e-7 shift and the run_rhf
    match both fail."""
    fn = getattr(ferric, name)
    rijk = fn(*water, scf_jk="rijk", **kwargs)
    exact = fn(*water, scf_jk="exact", **kwargs)
    assert (rijk.scf_jk, rijk.jk_auxbasis) == ("rijk", JK)
    assert (exact.scf_jk, exact.jk_auxbasis) == ("exact", None)
    e_rijk, e_exact = _hf_energy(rijk), _hf_energy(exact)
    shift = abs(e_rijk - e_exact)
    assert 1e-7 < shift < 1e-3, (name, e_rijk, e_exact)
    assert abs(e_rijk - rhf_refs["rijk"]) <= 1e-10, (name, e_rijk, rhf_refs["rijk"])
    assert abs(e_exact - rhf_refs["exact"]) <= 1e-10, (name, e_exact)
    assert abs(rijk.total_energy - exact.total_energy) > 1e-7


@pytest.mark.parametrize("mode", ["rijk", "rijcosx"])
def test_oo_rimp2_refuses_a_fitted_reference(water, mode):
    with pytest.raises(ValueError, match="rebuilds exact four-centre J/K") as err:
        ferric.run_oo_rimp2(*water, scf_jk=mode)
    assert "run_oo_rimp2" in str(err.value)
    assert f'scf_jk = "{mode}" is refused' in str(err.value)


def test_exact_echo_names_a_non_default_k_builder(water):
    r = ferric.run_rimp2(*water, k_builder="link")
    assert "SCF J/K: exact, k_builder=link" in str(r)
    assert "SCF J/K: exact)" in str(ferric.run_rimp2(*water))
