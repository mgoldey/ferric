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
    "fock_screen",
    "integral_direct",
    "maps",
    "pair_fit",
    "precision",
    "cg_rtol",
    "cg_rtol_source",
    "cg_tightened",
    "cg_energy_error_target",
    "cg_energy_target_met",
    "cg_relres",
    "cg_true_relres",
    "cg_energy_error_bound",
    "cg_converged",
    "frozen_core",
    "operator",
    "solve_plan_bytes",
    "solve_resident_bytes",
    "matvec_path",
    "validation_grade",
    "eps_trust",
    "warnings",
    "solver_warnings",
    "mask_certificate",
]
MASK_CERT_KEYS = [
    "bounds",
    "error_estimate",
    "lower_bound",
    "upper_bound",
    "e_corr_interval_lo",
    "e_corr_interval_hi",
    "e_corr_corrected",
    "near_upper",
    "far_upper",
    "g_far_sq",
    "h_far_m_sq",
    "lambda_min",
    "tau",
    "n_kept",
    "n_near",
    "n_far",
    "t_s",
]
MAP_KEYS = [
    "aux_radius",
    "virt_radius",
    "ao_tail",
    "schwarz_skip",
    "batch_merge",
    "virt_schwarz_kappa",
]
PAIR_FIT_KEYS = ["n_pair_groups", "n_distinct_occ_per_group", "n_trsm", "n_w_reused"]


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
    assert loc["fock_screen"] == 1.0  # on by default on both paths
    assert loc["precision"] == "f64"
    assert loc["frozen_core"] == 1
    assert loc["cg_converged"] is True and 0.0 <= loc["cg_relres"] < 1e-6
    assert loc["cg_rtol"] == 1e-6 and loc["cg_relres"] < loc["cg_rtol"]
    assert 0.0 < loc["cg_true_relres"] < 1e-5
    assert loc["solver_warnings"] == []
    assert loc["cg_rtol_source"] == "adaptive-default"
    assert loc["cg_tightened"] is False
    assert loc["cg_energy_error_target"] == 1e-10
    assert loc["cg_energy_target_met"] is True
    assert 0.0 <= loc["cg_energy_error_bound"] <= 1e-10
    assert 0.0 < loc["keep_fraction"] <= 1.0 and 0.0 < loc["pair_fraction"] <= 1.0
    assert isinstance(loc["n_pairs_gated"], int)
    assert sorted(loc["maps"]) == sorted(MAP_KEYS)
    assert loc["maps"]["batch_merge"] == 2
    assert loc["maps"]["virt_schwarz_kappa"] is None
    # unset maps take the CLI [local] defaults (shared parser)
    assert loc["maps"]["virt_radius"] == 20.0 and loc["maps"]["ao_tail"] == 1e-4
    assert sorted(loc["pair_fit"]) == sorted(PAIR_FIT_KEYS)
    assert loc["pair_fit"]["n_pair_groups"] > 0 and loc["pair_fit"]["n_trsm"] > 0
    assert loc["pair_fit"]["n_w_reused"] == 0  # no kappa screen, nothing to reuse
    assert "not validated" in loc["validation_grade"]
    assert "99.8930% at 1e-4" in loc["eps_trust"]
    assert "slope ~1.8" in loc["eps_trust"] and "95.57% at 1e-3" in loc["eps_trust"]
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
    assert loc["mask_certificate"] is None  # not requested


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
    assert loc["fock_screen"] == 1.0  # on by default on the in-core path too
    assert loc["pair_fit"] is None
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


@pytest.mark.parametrize("integral_direct", [False, True])
def test_fock_screen_kwarg_off_on_both_paths(water, integral_direct):
    """`fock_screen=False` turns the default-on screen off on the in-core and
    the integral-direct path (it only adds amplitudes, so keep can only drop);
    on the exact method the kwarg is a ValueError, as in the CLI."""
    mol, obs, aux = water
    kw = dict(
        frozen_core=1,
        local="amplitude-threshold",
        eps=1e-4,
        integral_direct=integral_direct,
    )
    on = ferric.run_rimp2(mol, obs, aux, **kw).local
    off = ferric.run_rimp2(mol, obs, aux, fock_screen=False, **kw).local
    assert on["fock_screen"] == 1.0 and off["fock_screen"] is None
    assert off["keep_fraction"] <= on["keep_fraction"]
    with pytest.raises(ValueError, match="fock_screen"):
        ferric.run_rimp2(mol, obs, aux, frozen_core=1, fock_screen=False)


def test_mask_certificate_encloses_the_canonical_energy(water):
    """`mask_certificate=True` (in core) fills `result.local["mask_certificate"]`;
    its certified interval contains the canonical RI-MP2 correlation energy,
    the E* it bounds on this path. Refused integral-direct and on the exact
    method (ValueError, as the CLI)."""
    mol, obs, aux = water
    kw = dict(frozen_core=1, local="amplitude-threshold", eps=1e-4)
    r = ferric.run_rimp2(
        mol, obs, aux, compute_reference=True, mask_certificate=True, **kw
    )
    m = r.local["mask_certificate"]
    assert sorted(m) == sorted(MASK_CERT_KEYS)
    e_ref = r.local["e_corr_canonical_ri"]
    assert m["e_corr_interval_lo"] <= e_ref <= m["e_corr_interval_hi"] + 1e-10
    assert 0.0 <= m["lower_bound"] <= m["upper_bound"]
    assert m["e_corr_corrected"] == r.mp2_corr - m["error_estimate"]
    assert "RI" in m["bounds"]
    with pytest.raises(ValueError, match="in-core"):
        ferric.run_rimp2(
            mol, obs, aux, integral_direct=True, mask_certificate=True, **kw
        )
    with pytest.raises(ValueError, match="mask_certificate"):
        ferric.run_rimp2(mol, obs, aux, frozen_core=1, mask_certificate=True)
