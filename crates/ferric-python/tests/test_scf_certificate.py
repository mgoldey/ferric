"""The `certificate` dict on RhfResult / UhfResult / DftResult."""

import warnings

import ferric

_WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"


def _water():
    return ferric.Molecule.from_xyz_string(_WATER)


def test_rhf_certificate_is_proven_and_stamped():
    c = ferric.run_rhf(_water(), ferric.BasisSet.bundled("sto-3g")).certificate
    assert c["capability"] == "rhf"
    assert c["grade"] == "proven"
    assert c["grade_source"] == "site/src/reference/validation.md"
    assert c["approximations"] == []
    assert c["warnings"] == []
    assert c["build"]["commit"] == ferric.build_info()["commit"]


def test_unconverged_rhf_is_ungraded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = ferric.run_rhf(_water(), ferric.BasisSet.bundled("sto-3g"), max_iter=2)
    assert not r.converged
    c = r.certificate
    assert c["grade"] == "ungraded"
    assert any("SCF did not converge" in w for w in c["warnings"])


def test_uhf_and_dft_certificates():
    bs = ferric.BasisSet.bundled("sto-3g")
    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2)
    u = ferric.run_uhf(oh, bs).certificate
    assert u["capability"] == "uhf"
    assert any(w.startswith("internal stability not checked") for w in u["warnings"])
    d = ferric.run_dft(_water(), bs, "PBE").certificate
    assert d["capability"] == "ksdft"
    assert d["grade"] == "proven"
    assert "xc-grid" in [a["name"] for a in d["approximations"]]
