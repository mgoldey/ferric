"""The `certificate` dict on post-SCF results: RiMp2Result, CcResult,
PdepRpaResult, GwResult and BseResult."""

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
