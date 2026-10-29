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
