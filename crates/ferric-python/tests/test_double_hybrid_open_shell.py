"""run_double_hybrid on a UKS reference and with D3(BJ).

PySCF comparisons at matched settings are in
crates/ferric-mp2/tests/validation_udh_pyscf.rs. These pin the binding:
the route, the assembly from the returned pieces, the singlet anchor, the
dispersion term, and the refusals.
"""

import pytest

import ferric

_WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
_NH2 = "3\n\nN 0 0 0\nH 0 0.803611 -0.634654\nH 0 -0.803611 -0.634654\n"
_SETS = {"b2plyp": (0.27, 0.27), "dsd-pbep86": (0.51, 0.36)}

# Tight SCF so the singlet comparison measures the route, not the stopping
# point. Measured loose-vs-tight terms of the closed-shell double hybrids are
# about 2e-11 Ha (validation_dsd_pbep86_pyscf.rs); 1e-9 keeps a 40x margin and
# stays far below the 1e-4 to 1e-3 Ha a lost spin channel or swapped scaling
# produces.
_CONV = {"scf_energy_conv": 1e-11, "scf_density_conv": 1e-9, "scf_max_iter": 200}
_BAR = 1e-9


def _basis():
    return ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")


@pytest.mark.parametrize("kind", sorted(_SETS))
def test_open_shell_total_is_its_pieces(kind):
    mol = ferric.Molecule.from_xyz_string(_NH2, multiplicity=2)
    bs, aux = _basis()
    r = ferric.run_double_hybrid(mol, bs, aux, kind=kind)
    c_os, c_ss = _SETS[kind]
    assert r.reference == "unrestricted"
    assert (r.c_os, r.c_ss) == (c_os, c_ss)
    assert r.e_os < -1e-4 and r.e_ss < -1e-5
    assert abs(r.e_corr_scaled - (c_os * r.e_os + c_ss * r.e_ss)) < 1e-14
    assert abs(r.total_energy - (r.e_ks + r.e_corr_scaled)) < 1e-14
    assert r.e_dispersion is None


@pytest.mark.parametrize("kind", sorted(_SETS))
def test_singlet_unrestricted_route_equals_restricted_route(kind):
    mol = ferric.Molecule.from_xyz_string(_WATER)
    bs, aux = _basis()
    rks = ferric.run_double_hybrid(mol, bs, aux, kind=kind, **_CONV)
    uks = ferric.run_double_hybrid(mol, bs, aux, kind=kind, unrestricted=True, **_CONV)
    assert rks.reference == "restricted" and uks.reference == "unrestricted"
    for name in ("e_ks", "e_os", "e_ss", "total_energy"):
        assert abs(getattr(rks, name) - getattr(uks, name)) < _BAR, name
    assert abs(uks.e_os) > 1e-3 and abs(uks.e_ss) > 1e-4


def test_b2plyp_d3bj_adds_the_b2plyp_row_only():
    mol = ferric.Molecule.from_xyz_string(_NH2, multiplicity=2)
    bs, aux = _basis()
    plain = ferric.run_double_hybrid(mol, bs, aux, kind="b2plyp", **_CONV)
    d3 = ferric.run_double_hybrid(
        mol, bs, aux, kind="b2plyp", dispersion="D3(BJ)", **_CONV
    )
    want = ferric.d3bj_energy(mol, "b2plyp")
    assert d3.e_dispersion == pytest.approx(want, abs=1e-14)
    assert d3.e_dispersion < -1e-6
    assert d3.e_ks == plain.e_ks and d3.e_os == plain.e_os
    assert abs(d3.total_energy - plain.total_energy - want) < 1e-14
    names = [a["name"] for a in d3.certificate["approximations"]]
    assert "scf dispersion" in names
    # The anchor's evidence runs no dispersion: it is not attached here.
    assert d3.certificate.get("measured_error") is None


def test_dsd_pbep86_d3bj_is_the_refit_set_as_a_whole():
    mol = ferric.Molecule.from_xyz_string(_NH2, multiplicity=2)
    bs, aux = _basis()
    plain = ferric.run_double_hybrid(mol, bs, aux, kind="dsd-pbep86", **_CONV)
    d3 = ferric.run_double_hybrid(
        mol, bs, aux, kind="dsd-pbep86", dispersion="d3bj", **_CONV
    )
    assert (d3.c_os, d3.c_ss) == (0.52, 0.22)
    assert (plain.c_os, plain.c_ss) == (0.51, 0.36)
    # 0.69 HF against 0.72 HF: the SCF itself differs.
    assert abs(d3.e_ks - plain.e_ks) > 1e-4
    assert d3.e_dispersion == pytest.approx(ferric.d3bj_energy(mol, "dsdpbep86"), abs=1e-14)
    assert abs(d3.total_energy - (d3.e_ks + d3.e_corr_scaled + d3.e_dispersion)) < 1e-14


@pytest.mark.parametrize("spec", ["mbd", "d3bj(pbe)", "d2"])
def test_dispersion_other_than_the_own_d3bj_set_is_refused(spec):
    mol = ferric.Molecule.from_xyz_string(_WATER)
    bs, aux = _basis()
    with pytest.raises(ValueError, match="double hybrid"):
        ferric.run_double_hybrid(mol, bs, aux, kind="b2plyp", dispersion=spec)
