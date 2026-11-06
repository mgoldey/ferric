"""`.certificate` on run_optimize / run_optimize_uhf / run_frequencies results
and per scan point on run_drpa_scan / run_mp2_v_scan."""

import os
from pathlib import Path

import pytest

import ferric

_H2 = "2\n\nH 0 0 0\nH 0 0 0.80\n"
_WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"


def _names(c):
    return [a["name"] for a in c["approximations"]]


def _detail(c, name):
    return next(a["detail"] for a in c["approximations"] if a["name"] == name)


def test_optimize_certificate_inherits_the_final_point_scf():
    mol = ferric.Molecule.from_xyz_string(_H2)
    r = ferric.run_optimize(mol, "sto-3g")
    assert r.converged
    c = r.certificate
    assert c["capability"] == "optimize:rhf"
    assert c["grade"] == "proven"
    assert c["grade_source"] == "site/src/reference/validation.md"
    assert c["build"]["commit"] == ferric.build_info()["commit"]
    assert _detail(c, "gradient") == "analytic"
    assert "BFGS" in _detail(c, "optimizer")
    assert c["warnings"] == []


def test_unconverged_optimization_is_ungraded():
    mol = ferric.Molecule.from_xyz_string(_H2)
    r = ferric.run_optimize(mol, "sto-3g", max_steps=1)
    assert not r.converged
    c = r.certificate
    assert c["grade"] == "ungraded"
    assert any("optimizer did not converge" in w for w in c["warnings"]), c


def test_open_shell_optimize_is_graded_under_uhf():
    # OH radical; the doublet needs the UHF driver.
    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 1.0\n", 0, 2)
    c = ferric.run_optimize_uhf(oh, "sto-3g", max_steps=3).certificate
    assert c["capability"] == "optimize:uhf"
    assert any(
        w.startswith("scf reference: internal stability not checked")
        for w in c["warnings"]
    )


def test_frequencies_certificate_records_the_hessian():
    mol = ferric.Molecule.from_xyz_string(_WATER)
    r = ferric.run_frequencies(mol, "sto-3g")
    c = r.certificate
    assert c["capability"] == "frequencies:rhf"
    assert c["grade"] == "proven"
    assert _detail(c, "hessian") == r.hessian_source == "analytic"
    assert "hessian-asymmetry" in _names(c)
    fd = ferric.run_frequencies(
        ferric.Molecule.from_xyz_string(_H2), "sto-3g", hessian="fd"
    ).certificate
    assert _detail(fd, "hessian").startswith("finite-difference")
    assert "12 gradient evaluation(s)" in _detail(fd, "hessian")


def test_imaginary_frequency_is_a_warning():
    # Bent H2O at a linear geometry has an imaginary bend.
    lin = ferric.Molecule.from_xyz_string("3\n\nO 0 0 0\nH 0 0 0.96\nH 0 0 -0.96\n")
    r = ferric.run_frequencies(lin, "sto-3g")
    assert any(f < 0 for f in r.frequencies)
    assert any("imaginary frequenc" in w for w in r.certificate["warnings"])


def test_drpa_scan_has_a_certificate_per_point_equal_to_run_drpa():
    mol = ferric.Molecule.from_xyz_string(_WATER)
    bs, aux = ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    eps = [1e-4, 1e-3]
    scan = ferric.run_drpa_scan(mol, bs, aux, eps, frozen_core=1)
    assert len(scan) == 2
    for point, e in zip(scan, eps):
        single = ferric.run_drpa(
            mol, bs, aux, local="amplitude-threshold", eps=e, frozen_core=1
        )["certificate"]
        c = point["certificate"]
        assert c["capability"] == "drpa"
        assert c == single
        assert "local" in _names(c)
        assert c["grade"] == "ungraded"  # finite eps is not graded


@pytest.mark.skipif(
    not (
        os.environ.get("FERRIC_TERF_TABLE_DIR")
        and (Path(os.environ["FERRIC_TERF_TABLE_DIR"]) / "16_4_2.bin").is_file()
    ),
    reason="terfc interpolation tables absent (set FERRIC_TERF_TABLE_DIR)",
)
def test_mp2_v_scan_has_a_certificate_per_arm_equal_to_run_mp2_v():
    h2 = ferric.Molecule.from_xyz_string(_H2)
    bs, aux = ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    arms = [(1.0, None), (1.5, None)]
    scan = ferric.run_mp2_v_scan(h2, bs, aux, arms, [10.0, 12.0])
    for arm, r0 in zip(scan["arms"], (1.0, 1.5)):
        c = arm["certificate"]
        assert c["capability"] == "mp2-v"
        assert c["grade"] == "proven_narrow"
        single = ferric.run_mp2_v(h2, bs, aux, r0=r0).certificate

        # The scan's SCF is density-fitted by default (run_mp2_v's is exact), so
        # compare the correlation-side records only.
        def own(cert):
            return [
                a
                for a in cert["approximations"]
                if not a["name"].startswith("scf ") and a["name"] != "vv10-b-scan"
            ]

        assert own(c) == own(single)
        assert _detail(c, "vv10-b-scan").startswith("b in [10.0, 12.0]")
    # The two arms differ in r0, so their operator records differ.
    ops = [_detail(a["certificate"], "operator") for a in scan["arms"]]
    assert ops[0] != ops[1]
