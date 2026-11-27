"""The shared result getters read the right source.

`certificate`, `timings` and the `energy`/`total_energy` alias of the result
classes are one-line getters over shared helpers
(`crates/ferric-python/src/method_result.rs`, roadmap item 9b). These cases
pin that each getter is wired to its class's own value: the alias equals the
class's total-energy attribute, the certificate is the class's (its capability
names the method, and a DFT dispersion correction appears in it exactly when
one was requested), and a periodic result's `timings` has the documented
shape.
"""

from __future__ import annotations

import json

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"
OH = "2\n\nO 0 0 0\nH 0 0 0.97"
BOHR_IN_ANGSTROM = 0.529177210903
PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.16885540"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}


def _water():
    return ferric.Molecule.from_xyz_string(WATER, 0, 1)


def _oh():
    return ferric.Molecule.from_xyz_string(OH, 0, 2)


def _sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def _approximations(cert):
    return [a["name"] for a in cert["approximations"]]


# name -> (run, capability, attribute holding the class's own total energy)
_CASES = {
    "RhfResult": (lambda: ferric.run_rhf(_water(), _sto3g()), "rhf", "energy"),
    "SymmetricRhfResult": (
        lambda: ferric.run_rhf(_water(), _sto3g(), symmetry=True),
        None,
        "energy",
    ),
    "UhfResult": (lambda: ferric.run_uhf(_oh(), _sto3g()), "uhf", "energy"),
    "DftResult": (
        lambda: ferric.run_dft(_water(), _sto3g(), "LDA"),
        "ksdft",
        "total_energy",
    ),
    "RiMp2Result": (
        lambda: ferric.run_rimp2(
            _water(), _sto3g(), ferric.BasisSet.bundled("cc-pvdz-ri")
        ),
        "rimp2",
        "total_energy",
    ),
    "CasCiResult": (
        lambda: ferric.run_casci(_water(), _sto3g(), 4, (2, 2)),
        "casci",
        "e_total",
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_generated_getters_read_their_own_class(name):
    run, capability, total_attr = _CASES[name]
    r = run()
    assert type(r).__name__ == name
    total = getattr(r, total_attr)
    assert isinstance(total, float) and total < 0.0
    # Both spellings, where the class has both, are the class's own total.
    for alias in ("energy", "total_energy"):
        if hasattr(r, alias):
            assert getattr(r, alias) == total, (name, alias)
    if capability is None:
        assert not hasattr(r, "certificate")
        return
    cert = r.certificate
    assert isinstance(cert, dict)
    assert cert["capability"] == capability
    # A fresh dict per read, equal to the last, as the hand-written getter gave.
    assert r.certificate == cert and r.certificate is not cert
    json.dumps(cert)


def test_dft_certificate_names_dispersion_only_when_requested():
    plain = ferric.run_dft(_water(), _sto3g(), "B3LYP")
    d3 = ferric.run_dft(_water(), _sto3g(), "B3LYP", dispersion="d3bj")
    assert "dispersion" not in _approximations(plain.certificate)
    disp = [a for a in d3.certificate["approximations"] if a["name"] == "dispersion"]
    assert disp == [{"name": "dispersion", "detail": "D3(BJ)"}]
    # The alias follows the dispersion-corrected total, not the bare SCF one.
    assert d3.energy == d3.total_energy != plain.total_energy
    assert plain.energy == plain.total_energy


def test_periodic_gamma_rhf_getters(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    basis = ferric.BasisSet.from_bse_json(str(p))
    b = BOHR_IN_ANGSTROM
    h2 = ferric.Molecule.from_xyz_string(
        f"2\nH2\nH {0.3 * b} {0.2 * b} {0.1 * b}\nH {0.3 * b} {0.2 * b} {1.5 * b}\n"
    )
    a = 4.0 * b
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    r = ferric.run_rhf_gamma(h2, lattice, basis, omega=0.8 / b)
    assert type(r).__name__ == "GammaRhfResult"
    assert r.total_energy == r.energy
    cert = r.certificate
    assert cert["capability"] == "rhf-gamma"
    assert _approximations(cert) == ["jk", "exxdiv", "kpoints"]
    t = r.timings
    assert set(t) == {
        "wall_s",
        "cpu_s",
        "unattributed_wall_s",
        "stages",
        "substages",
        "counters",
    }
    assert t["wall_s"] > 0.0
    assert {"hcore S/T", "scf J (dense AFT)", "scf K (dense AFT)"} <= set(t["stages"])
    for stage in t["stages"].values():
        assert set(stage) == {"wall_s", "cpu_s", "calls"}


def test_dft_result_exposes_fock_hooked():
    mol = ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"
    )
    bs = ferric.BasisSet.bundled("sto-3g")
    plain = ferric.run_dft(mol, bs, "LDA")
    hooked = ferric.run_dft(mol, bs, "LDA", fock_hook=lambda d: (0.0, d * 0.0))
    assert plain.fock_hooked is False
    assert hooked.fock_hooked is True


def test_frequency_result_carries_the_geometry_it_was_computed_at():
    h2 = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0.1 -0.2 0.74\n")
    res = ferric.run_frequencies(h2, "sto-3g")
    assert isinstance(res.molecule, ferric.Molecule)
    assert res.molecule.symbols() == h2.symbols()
    assert res.molecule.coords_bohr() == h2.coords_bohr()


def test_mp2_family_jk_label_echoes_the_reference_scf_jk():
    water = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled("sto-3g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    exact = ferric.run_rimp2(water, bs, aux)
    assert (exact.scf_jk, exact.jk_label, exact.jk_auxbasis) == ("exact", "exact", None)
    # The label names a non-default exchange builder; scf_jk stays the route.
    link = ferric.run_rimp2(water, bs, aux, k_builder="link")
    assert link.scf_jk == "exact"
    assert link.jk_label != "exact" and "link" in link.jk_label.lower()
    rijk = ferric.run_scs_mp2(water, bs, aux, scf_jk="rijk")
    assert rijk.scf_jk == "rijk"
    assert rijk.jk_label.startswith("rijk")
    assert rijk.jk_auxbasis is not None


@pytest.fixture(scope="module")
def pdep_runs():
    water = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled("sto-3g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    return (
        ferric.run_pdep_rpa(water, bs, aux),
        ferric.run_pdep_rpa(water, bs, aux, with_inv_dielectric_freq=True),
        aux,
    )


def test_pdep_rpa_result_exposes_the_route_and_dressed_eigenvectors(pdep_runs):
    np = pytest.importorskip("numpy")
    r, _, _ = pdep_runs
    assert r.route == "static-eigenbasis"
    u = np.asarray(r.dressed_eigenvectors)
    m = r.n_eigenpotentials
    assert u.shape == (np.asarray(r.eigenpotentials).shape[0], m)
    # Eigenvectors of the symmetric dressed chi0: orthonormal columns.
    np.testing.assert_allclose(u.T @ u, np.eye(m), atol=1e-8)


def test_pdep_rpa_inverse_dielectric_stack_is_opt_in_and_consistent(pdep_runs):
    np = pytest.importorskip("numpy")
    plain, full, _ = pdep_runs
    assert plain.inv_dielectric_freq is None
    stack = full.inv_dielectric_freq
    m = full.n_eigenpotentials
    assert len(stack) == len(full.quad_freqs)
    assert full.e_rpa == pytest.approx(plain.e_rpa, abs=1e-12)
    lam = np.asarray(full.eigenvalues_freq)
    for k, w in enumerate(stack):
        w = np.asarray(w)
        assert w.shape == (m, m)
        np.testing.assert_allclose(w, w.T, atol=1e-10)
        # eigenvalues_freq holds the eigenvalues eps_a >= 1 of the projected
        # dielectric matrix at this frequency, and W = eps^-1 - 1 shares them.
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvalsh(w)), np.sort(1.0 / lam[k] - 1.0), atol=1e-8
        )
