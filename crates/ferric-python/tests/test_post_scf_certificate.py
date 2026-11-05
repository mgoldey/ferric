"""The `certificate` dict on post-SCF results: RiMp2Result, CcResult,
PdepRpaResult, GwResult, BseResult, and the MP2-family, dRPA, LinLCCD,
open-shell GW and RPAx results."""

import pytest

import ferric

_WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"


def _setup():
    mol = ferric.Molecule.from_xyz_string(_WATER)
    return mol, ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")


def _names(c):
    return [a["name"] for a in c["approximations"]]


def test_rimp2_certificate():
    mol, bs, aux = _setup()
    c = ferric.run_rimp2(mol, bs, aux, frozen_core=1).certificate
    assert c["capability"] == "rimp2"
    assert c["grade"] == "proven"
    assert _names(c) == ["ri", "frozen-core"]
    assert c["approximations"][0]["detail"] == "aux cc-pvdz-ri"
    assert c["build"]["commit"] == ferric.build_info()["commit"]


def test_local_rimp2_certificate_is_capped():
    mol, bs, aux = _setup()
    c = ferric.run_rimp2(mol, bs, aux, local="amplitude-threshold", eps=1e-4).certificate
    assert c["capability"] == "rimp2"
    assert c["grade"] == "ungraded"
    assert "local" in _names(c)


def test_cc_certificates_carry_table_grades():
    mol, bs, aux = _setup()
    assert ferric.run_ccsd(mol, bs, aux).certificate["grade"] == "proven"
    t = ferric.run_ccsd_t(mol, bs, aux).certificate
    assert t["capability"] == "ccsd(t)"
    assert t["grade"] == "proven_narrow"
    assert ferric.run_ccd(mol, bs, aux).certificate["capability"] == "ccd"


def test_pdep_gw_bse_certificates():
    mol, bs, aux = _setup()
    p = ferric.run_pdep_rpa(mol, bs, aux).certificate
    assert p["capability"] == "pdep-rpa"
    assert p["grade"] == "proven"
    assert "frequency-quadrature" in _names(p)
    g = ferric.run_gw(mol, bs, aux).certificate
    assert g["capability"] == "gw"
    assert g["grade"] == "smoke"
    assert any("this run uses n_quad = 20" in w for w in g["warnings"]), g
    b = ferric.run_bse_tda(mol, bs, aux).certificate
    assert b["capability"] == "bse-tda"
    assert b["grade"] == "smoke"


def test_periodic_certificate_is_ungraded():
    a = 4.0 * 0.529177210903  # 4 Bohr cubic cell, in Angstrom
    h2 = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.74\n")
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    r = ferric.run_rhf_gamma(h2, lattice, ferric.BasisSet.bundled("sto-3g"))
    c = r.certificate
    assert c["capability"] == "rhf-gamma"
    assert c["grade"] == "ungraded"
    assert c["grade_source"] is None
    assert [a["name"] for a in c["approximations"]] == ["jk", "exxdiv", "kpoints"]
    assert c["approximations"][0]["detail"] == "dense"


def _row(c, cap, grade, approx):
    assert c["capability"] == cap, c
    assert c["grade"] == grade, c
    assert c["grade_source"] == "site/src/reference/validation.md"
    for a in approx:
        assert a in _names(c), (a, c)
    # Stamped with this build.
    assert c["build"]["commit"] == ferric.build_info()["commit"]


def test_mp2_family_certificates_carry_table_grades():
    mol, bs, aux = _setup()
    _row(ferric.run_mp3(mol, bs, aux).certificate, "mp3", "proven", ["ri"])
    _row(ferric.run_oo_rimp2(mol, bs, aux).certificate, "oo-rimp2", "proven_narrow", ["ri"])
    lap = ferric.run_laplace_mp2(mol, bs, aux, n_quad=3).certificate
    _row(lap, "laplace-mp2", "proven", ["ri", "laplace"])
    assert [a["detail"] for a in lap["approximations"] if a["name"] == "laplace"] == [
        "3-point minimax quadrature"
    ]
    sos = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.1, n_quad=3).certificate
    _row(sos, "laplace-sos-mp2", "ungraded", ["sos", "laplace"])
    assert [a["detail"] for a in sos["approximations"] if a["name"] == "sos"] == [
        "c_os = 1.1, c_ss = 0"
    ]
    att = ferric.run_attenuated_rimp2(mol, bs, aux).certificate
    _row(att, "att-rimp2", "proven", ["ri", "operator"])
    scs = ferric.run_scs_mp2(mol, bs, aux, c_os=1.1, c_ss=0.4).certificate
    _row(scs, "scs-mp2", "proven", ["ri", "scs"])
    assert [a["detail"] for a in scs["approximations"] if a["name"] == "scs"] == [
        "c_os = 1.1, c_ss = 0.4"
    ]


def test_oo_rimp2_unconverged_is_not_graded():
    mol, bs, aux = _setup()
    r = ferric.run_oo_rimp2(mol, bs, aux, max_iter=1)
    assert not r.converged
    c = r.certificate
    assert c["grade"] == "ungraded"
    assert any("orbital optimization did not converge" in w for w in c["warnings"]), c


def test_drpa_and_linlccd_dicts_carry_certificates():
    mol, bs, aux = _setup()
    d = ferric.run_drpa(mol, bs, aux)
    _row(d["certificate"], "drpa", "proven_narrow", ["ri"])
    assert "local" not in _names(d["certificate"])
    dl = ferric.run_drpa(mol, bs, aux, local="amplitude-threshold", eps=1e-4)
    assert dl["certificate"]["grade"] == "ungraded"
    assert "local" in _names(dl["certificate"])
    l = ferric.run_linlccd(mol, bs, aux)
    _row(l["certificate"], "linlccd", "proven_narrow", ["ri", "variant"])
    ll = ferric.run_linlccd(mol, bs, aux, local="amplitude-threshold", eps=1e-4)
    assert ll["certificate"]["grade"] == "ungraded"
    assert "local" in _names(ll["certificate"])


def test_rs_mp2_rpa_certificate_names_its_split():
    mol, bs, aux = _setup()
    c = ferric.run_rs_mp2_rpa(mol, bs, aux, omega=0.42).certificate
    _row(c, "rs-mp2-rpa", "smoke", ["ri", "range-separation", "frequency-quadrature"])
    detail = [a["detail"] for a in c["approximations"] if a["name"] == "range-separation"]
    assert detail and detail[0].startswith("erf/erfc split"), detail


def test_u_gw_certificate_is_graded_on_the_open_shell_reference():
    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", multiplicity=2)
    bs, aux = ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    c = ferric.run_u_gw(oh, bs, aux, qp_mos=(3, 6), n_quad=8).certificate
    assert c["capability"] == "gw"
    assert c["grade"] == "smoke"
    assert any("n_quad" in w for w in c["warnings"]), c
    assert c["build"]["commit"] == ferric.build_info()["commit"]
    # A ROHF reference is accepted and carries the same method row.
    r = ferric.run_u_gw(oh, bs, aux, reference="rohf", qp_mos=(3, 6), n_quad=8).certificate
    assert r["capability"] == "gw"


def test_tdhf_static_polarizability_certificate():
    mol, bs, aux = _setup()
    r = ferric.run_tdhf_static_polarizability(mol, bs, aux, "pbe", scissor=0.36, n_quad=8)
    c = r.certificate
    assert c["capability"] == "tdhf-static-polarizability"
    assert c["grade"] == "smoke"
    names = _names(c)
    assert "kernel" in names and "scissor" in names, names
    assert c["build"]["commit"] == ferric.build_info()["commit"]


def test_mp2_v_certificate_records_operator_and_vv10():
    h2 = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.74\n")
    bs, aux = ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    c = ferric.run_mp2_v(h2, bs, aux, attenuator="erfc").certificate
    _row(c, "mp2-v", "proven_narrow", ["ri", "operator", "vv10"])
