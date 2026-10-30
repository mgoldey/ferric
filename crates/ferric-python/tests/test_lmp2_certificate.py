"""The amplitude-threshold LMP2 model certificate through the Python
surface: every certificate field (ferric_mp2::lmp2_certificate, the same
list the CLI writes to its run-log JSON) is present in ``result.local`` with
the value the run implies, plus the build stamp.
"""

import pytest

ferric = pytest.importorskip("ferric")

CERT_KEYS = [
    "scheme",
    "eps",
    "keep_fraction",
    "pair_fraction",
    "n_pairs_gated",
    "gate_cal",
    "integral_direct",
    "maps",
    "precision",
    "cg_relres",
    "cg_converged",
    "frozen_core",
    "operator",
    "solve_plan_bytes",
    "solve_resident_bytes",
    "matvec_path",
    "validation_grade",
    "eps_trust",
    "warnings",
]
MAP_KEYS = [
    "aux_radius",
    "virt_radius",
    "ao_tail",
    "schwarz_skip",
    "batch_merge",
    "virt_schwarz_kappa",
]


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
    return mol, ferric.BasisSet.bundled("6-31g"), ferric.BasisSet.bundled("cc-pvdz-ri")


def _build_stamp():
    info = ferric.build_info()
    return info["commit"], info["dirty"]


def test_certificate_fields_round_trip_direct(water):
    mol, obs, aux = water
    r = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=1e-4,
        integral_direct=True,
        gate_cal=0.7,
        batch_merge=2,
    )
    loc = r.local
    for k in CERT_KEYS:
        assert k in loc, f"certificate field {k} missing"
    assert loc["scheme"] == "amplitude-threshold"
    assert loc["eps"] == 1e-4
    assert loc["integral_direct"] is True
    assert loc["gate_cal"] == 0.7
    assert loc["precision"] == "f64"
    assert loc["frozen_core"] == 1
    assert loc["cg_converged"] is True and 0.0 <= loc["cg_relres"] < 1e-9
    assert 0.0 < loc["keep_fraction"] <= 1.0 and 0.0 < loc["pair_fraction"] <= 1.0
    assert isinstance(loc["n_pairs_gated"], int)
    assert sorted(loc["maps"]) == sorted(MAP_KEYS)
    assert loc["maps"]["batch_merge"] == 2
    assert loc["maps"]["virt_schwarz_kappa"] is None
    assert "not validated" in loc["validation_grade"]
    assert "99.671% at 1e-4" in loc["eps_trust"]
    assert "slope ~1.2" in loc["eps_trust"] and "93.81% at 1e-3" in loc["eps_trust"]
    assert loc["operator"] == "coulomb" and "CAVEAT" not in loc["eps_trust"]
    assert loc["solve_plan_bytes"] > 0 and loc["matvec_path"] == "planned"
    assert loc["cg_lanczos_kappa_lower"] >= 1.0
    assert loc["t_eri3_fill_thread_s"] > 0.0
    # eps 1e-4 > 1e-5 and the gate is on: both out-of-range warnings fire
    assert any("1.0e-4" in w for w in loc["warnings"])
    assert any("gate" in w for w in loc["warnings"])
    commit, dirty = _build_stamp()
    assert loc["build_commit"] == commit and loc["build_dirty"] == dirty
    assert loc["gate_on"] is True and loc["n_pairs_unique"] > 0


def test_certificate_in_core_exact_limit(water):
    mol, obs, aux = water
    r = ferric.run_rimp2(
        mol, obs, aux, frozen_core=1, local="amplitude-threshold", eps=0.0
    )
    loc = r.local
    for k in CERT_KEYS:
        assert k in loc, f"certificate field {k} missing"
    assert loc["integral_direct"] is False
    assert loc["maps"] is None and loc["gate_cal"] is None
    assert loc["validation_grade"].startswith("Proven")
    assert "exact" in loc["eps_trust"]
    assert not any("eps" in w for w in loc["warnings"])
    assert loc["t_spaces_s"] > 0.0


def test_certificate_flags_a_non_coulomb_operator(water):
    """The measured eps record is Coulomb: an erfc (att-rimp2) run gets the
    caveat in its trust line and a warning."""
    mol, obs, aux = water
    r = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=1e-5,
        att_operator="erfc",
        omega=0.42,
    )
    loc = r.local
    assert loc["operator"].startswith("erfc(")
    assert "CAVEAT" in loc["eps_trust"]
    assert any("non-Coulomb" in w for w in loc["warnings"])
