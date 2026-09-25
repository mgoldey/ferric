"""`run_rhf_gamma`: Gamma-point periodic RHF through the Python surface.

Stage-1 PBC design step 11 (reference/pbc/stage1-design.md): the step-5
H2/STO-3G numbers, reached from Python, plus the hard errors.

References (PySCF 2.13 AFTDF, mesh 61^3, pinned in
reference/pbc/test_prototype.py and the Rust suite
crates/ferric-pbc/tests/pbc_dense_aft_scf.rs):
    H2/STO-3G, cubic a = 4 Bohr: E = -1.658327061049 (ewald),
    -0.949002691179 (none).

Units. The binding takes Angstrom (coordinates via Molecule, lattice rows,
omega in 1/Angstrom). The references are defined in Bohr, so this file
converts Bohr -> Angstrom with the reciprocal of ferric's own factor
(1 / 0.52917721092); the binding multiplies back, which round-trips to ~1 ulp.

Basis. The pins were computed with PySCF's STO-3G digits, which are shorter
than ferric's bundled BSE copy; the difference is visible at 1e-8, so the
basis is written out here as a BSE JSON file with PySCF's digits (the same
fixture the Rust suite builds in tests/common/mod.rs::pyscf_sto3g_h).

Independent checks beyond the pins: E_ewald - E_none = -v_M * N_e / 2 (the
Madelung shift acts as -v_M on the occupied space without changing D), and
omega-independence of the energy (omega is a numerical Ewald split).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR

H2_ATOMS_BOHR = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5)]
A_BOHR = 4.0
E_EWALD = -1.658327061049
E_NONE = -0.949002691179
TOL = 1e-8

PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.1688554"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}


def _xyz(atoms_bohr, symbol="H"):
    lines = [str(len(atoms_bohr)), "pbc gamma test (Angstrom)"]
    for x, y, z in atoms_bohr:
        lines.append(
            f"{symbol} {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return "\n".join(lines) + "\n"


def _cubic_angstrom(a_bohr):
    a = a_bohr * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR))


@pytest.fixture(scope="module")
def runs(h2, basis):
    lat = _cubic_angstrom(A_BOHR)
    return {
        exx: ferric.run_rhf_gamma(h2, lat, basis, exxdiv=exx)
        for exx in ("ewald", "none")
    }


def test_molecule_coordinates_round_trip_to_bohr(h2):
    got = np.array(h2.coords_bohr())
    assert np.abs(got - np.array(H2_ATOMS_BOHR)).max() < 1e-13


@pytest.mark.parametrize("exx,e_ref", [("ewald", E_EWALD), ("none", E_NONE)])
def test_h2_sto3g_a4_matches_pinned_pyscf_aftdf(runs, exx, e_ref):
    r = runs[exx]
    assert r.converged
    assert r.exxdiv == exx
    assert abs(r.energy - e_ref) < TOL, f"{exx}: E {r.energy!r} vs {e_ref!r}"


def test_madelung_shift_identity(runs):
    # N_e = 2, so E_ewald - E_none = -v_M.
    shift = runs["ewald"].energy - runs["none"].energy
    assert abs(shift + runs["ewald"].madelung) < 1e-9
    assert runs["ewald"].madelung == runs["none"].madelung


def test_result_arrays_are_consistent(runs):
    r = runs["ewald"]
    assert r.nao == 2
    d, s, c = r.density(), r.overlap(), r.mo_coeff()
    assert d.shape == s.shape == c.shape == (2, 2)
    assert abs(np.trace(d @ s) - 2.0) < 1e-10
    assert np.abs(c.T @ s @ c - np.eye(2)).max() < 1e-10
    eps = r.mo_energy()
    assert eps.shape == (2,) and eps[0] < eps[1]
    assert np.isfinite(r.e_nuc) and r.n_g_half > 0


def test_energy_is_omega_independent(h2, basis, runs):
    # 0.8 Bohr^-1 (the Rust suite's split) expressed in 1/Angstrom.
    omega = 0.8 / BOHR_IN_ANGSTROM
    r = ferric.run_rhf_gamma(
        h2, _cubic_angstrom(A_BOHR), basis, exxdiv="ewald", omega=omega
    )
    assert abs(r.omega - omega) < 1e-12
    assert abs(r.omega - runs["ewald"].omega) > 0.1  # really a different split
    assert abs(r.energy - runs["ewald"].energy) < 1e-9


def test_charged_cell_is_a_hard_error(basis):
    ion = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR), charge=-2)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_rhf_gamma(ion, _cubic_angstrom(A_BOHR), basis)


def test_odd_electron_count_is_a_hard_error(basis):
    # Molecule already refuses a parity-inconsistent (H, multiplicity 1), so an
    # odd count reaches the binding as an open shell: RHF-only must refuse it.
    h = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR[:1]), multiplicity=2)
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rhf_gamma(h, _cubic_angstrom(A_BOHR), basis)


@pytest.mark.parametrize("bad", ["vcut_sph", "", "ewald ", "madelung"])
def test_bad_exxdiv_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="exxdiv"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, exxdiv=bad)


def test_oversize_cell_is_refused_before_any_work(h2, basis):
    # nao = 2 needs 8 * 2**4 = 128 bytes; 1e-8 GiB is ~10 bytes.
    with pytest.raises(ValueError, match="128 bytes"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, max_eri_gb=1e-8)


@pytest.mark.parametrize(
    "lattice",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # not 3x3
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # singular
    ],
)
def test_bad_lattice_is_a_hard_error(h2, basis, lattice):
    with pytest.raises(ValueError):
        ferric.run_rhf_gamma(h2, lattice, basis)


@pytest.mark.parametrize("kw", [{"omega": 0.0}, {"max_eri_gb": -1.0}])
def test_nonpositive_knobs_are_hard_errors(h2, basis, kw):
    with pytest.raises(ValueError):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, **kw)


# ── jk="rsgdf": range-separated Gaussian density fitting ──
#
# References: E_rsgdf - E_dense for H2/STO-3G (PySCF digits), a = 4 Bohr,
# RS-GDF omega = 1 Bohr^-1, nuclear-attraction omega = 0.8 Bohr^-1 -- the
# exact setup of crates/ferric-pbc/tests/pbc_rsgdf.rs
# (h2_sto3g_fitting_error_matches_prototype), whose measured dE these are.
# They agree with the Python prototype (reference/pbc/pbc_gdf.py:
# -1.974228505e-6 / -4.472349375e-6) to ~3e-10, inside that suite's 1e-9
# bar. The fit is exxdiv-independent at Gamma (the Madelung term v_M S D S
# does not touch the fit), so one dE holds for both settings.

RSGDF_DE = {
    "cc-pvdz-ri": (-1.974228205803e-6, 28),
    "def2-universal-jkfit": (-4.472349094398e-6, 36),
}
RSGDF_TOL = 1e-10
HCORE_OMEGA_ANGSTROM = 0.8 / BOHR_IN_ANGSTROM


@pytest.fixture(scope="module")
def dense_at_rust_omega(h2, basis):
    lat = _cubic_angstrom(A_BOHR)
    return {
        exx: ferric.run_rhf_gamma(
            h2, lat, basis, exxdiv=exx, omega=HCORE_OMEGA_ANGSTROM
        )
        for exx in ("ewald", "none")
    }


@pytest.mark.parametrize("aux", sorted(RSGDF_DE))
@pytest.mark.parametrize("exx", ["ewald", "none"])
def test_rsgdf_fitting_error_matches_rust_suite(
    h2, basis, dense_at_rust_omega, aux, exx
):
    de_ref, naux = RSGDF_DE[aux]
    r = ferric.run_rhf_gamma(
        h2,
        _cubic_angstrom(A_BOHR),
        basis,
        exxdiv=exx,
        omega=HCORE_OMEGA_ANGSTROM,
        jk="rsgdf",
        auxbasis=aux,
    )
    assert r.converged
    assert r.jk == "rsgdf" and r.auxbasis == aux and r.exxdiv == exx
    assert r.naux == naux
    assert r.naux_kept + r.n_dropped == r.naux
    assert r.n_g_half is None
    de = r.energy - dense_at_rust_omega[exx].energy
    assert abs(de - de_ref) < RSGDF_TOL, f"{aux}/{exx}: dE {de!r} vs {de_ref!r}"


def test_dense_result_reports_no_fitting_fields(runs):
    r = runs["ewald"]
    assert r.jk == "dense"
    assert r.auxbasis is None
    assert r.naux is None and r.naux_kept is None and r.n_dropped is None
    assert r.n_g_half is not None and r.n_g_half > 0


def test_rsgdf_accepts_a_basis_set_object(h2, basis):
    lat = _cubic_angstrom(A_BOHR)
    by_name = ferric.run_rhf_gamma(h2, lat, basis, jk="rsgdf", auxbasis="cc-pvdz-ri")
    by_obj = ferric.run_rhf_gamma(
        h2, lat, basis, jk="rsgdf", auxbasis=ferric.BasisSet.bundled("cc-pvdz-ri")
    )
    assert by_obj.auxbasis == "cc-pvdz-ri"
    # Same inputs; rayon reductions need not be bit-identical run to run.
    assert abs(by_obj.energy - by_name.energy) < 1e-12


def test_stage_timings_are_reported(h2, basis, runs):
    # Observation only (ferric_pbc::timing): stages are disjoint leaves, so
    # their sum stays within the total; counters mirror the build's own.
    r = ferric.run_rhf_gamma(
        h2, _cubic_angstrom(A_BOHR), basis, jk="rsgdf", auxbasis="cc-pvdz-ri"
    )
    for res in (r, runs["ewald"]):
        t = res.timings
        stages = t["stages"]
        assert t["wall_s"] > 0.0 and stages
        assert sum(s["wall_s"] for s in stages.values()) <= t["wall_s"] + 1e-3
        assert all(s["wall_s"] >= 0.0 and s["calls"] >= 0 for s in stages.values())
        assert "hcore SR attraction" in stages
    assert r.timings["stages"]["scf K (rsgdf)"]["calls"] >= 1
    assert r.timings["counters"]["rsgdf aux dropped"] == r.n_dropped
    assert r.timings["counters"]["rsgdf SR3 triplets"] > 0
    assert runs["ewald"].timings["stages"]["scf J (dense AFT)"]["calls"] >= 1


def _s_aux(tmp_path, exps, name):
    shells = [
        {
            "function_type": "gto",
            "angular_momentum": [0],
            "exponents": [repr(e)],
            "coefficients": [["1.0"]],
        }
        for e in exps
    ]
    p = tmp_path / f"{name}.json"
    p.write_text(
        json.dumps({"name": name, "elements": {"1": {"electron_shells": shells}}})
    )
    return ferric.BasisSet.from_bse_json(str(p))


def test_rsgdf_reports_lindep_drops(h2, basis, tmp_path):
    # A duplicated aux shell makes the metric exactly singular: the extra
    # function must be dropped AND counted, and the fit (hence E) unchanged.
    lat = _cubic_angstrom(A_BOHR)
    base_exps = [4.0, 1.0, 0.3]
    base = ferric.run_rhf_gamma(
        h2, lat, basis, jk="rsgdf", auxbasis=_s_aux(tmp_path, base_exps, "s3")
    )
    dup = ferric.run_rhf_gamma(
        h2, lat, basis, jk="rsgdf", auxbasis=_s_aux(tmp_path, base_exps + [1.0], "s4")
    )
    assert base.naux == 6 and dup.naux == 8  # 2 atoms x shells
    assert dup.n_dropped == base.n_dropped + 2  # one duplicate per atom
    assert dup.naux_kept == base.naux_kept
    assert abs(dup.energy - base.energy) < 1e-9


@pytest.mark.parametrize("bad", ["", "gdf", "rs-gdf", "dense ", "aft"])
def test_bad_jk_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="jk"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, jk=bad)


def test_rsgdf_without_auxbasis_is_a_hard_error(h2, basis):
    with pytest.raises(ValueError, match="requires auxbasis"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, jk="rsgdf")


def test_auxbasis_with_dense_is_a_hard_error(h2, basis):
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, auxbasis="cc-pvdz-ri")


def test_memory_budget_with_dense_is_a_hard_error(h2, basis):
    with pytest.raises(ValueError, match="memory_budget_gb"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, memory_budget_gb=1.0)


def test_max_eri_gb_with_rsgdf_is_a_hard_error(h2, basis):
    with pytest.raises(ValueError, match="max_eri_gb"):
        ferric.run_rhf_gamma(
            h2,
            _cubic_angstrom(A_BOHR),
            basis,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            max_eri_gb=1.0,
        )


def test_unknown_bundled_auxbasis_is_a_hard_error(h2, basis):
    with pytest.raises(ValueError, match="not-a-basis"):
        ferric.run_rhf_gamma(
            h2, _cubic_angstrom(A_BOHR), basis, jk="rsgdf", auxbasis="not-a-basis"
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_nonpositive_rsgdf_budget_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="memory_budget_gb"):
        ferric.run_rhf_gamma(
            h2,
            _cubic_angstrom(A_BOHR),
            basis,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            memory_budget_gb=bad,
        )


def test_tiny_rsgdf_budget_is_refused(h2, basis):
    # ~11 bytes cannot hold the n x n matrices the builder reserves first.
    with pytest.raises(RuntimeError, match="RsGdf"):
        ferric.run_rhf_gamma(
            h2,
            _cubic_angstrom(A_BOHR),
            basis,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            memory_budget_gb=1e-8,
        )
