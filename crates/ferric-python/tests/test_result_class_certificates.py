"""`certificate` on the wrapper result classes: QmmmResult, QmmmOptimizeResult,
CdftResult, CdftCouplingResult and SaddleResult.

None of these entry points has a row in the method.kind matrix of
site/src/reference/validation.md (the page lists them under "Python entry
points", "not graded individually"), so every certificate is `ungraded` with
no grade source. What the tests pin is the approximation record each wrapper
adds on top of its SCF reference: the embedding for QM/MM, the constraints for
cDFT, the search and Hessian for the saddle. Each test fails on an assertion
if that record is dropped (mutation-checked, see the commit message).
BoysResult has no certificate on purpose: a localization is a unitary rotation
of orbitals whose own result carries the certificate.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

# Water plus two MM sites (+1 and -1 e), the geometry of the PySCF mm_charge
# reference water_sto-3g_qmmm_two_charges (Angstrom).
_SYMBOLS = ["O", "H", "H", "X", "X"]
_COORDS = [
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
    [2.5, 0.0, 0.0],
    [-2.5, 1.0, 0.5],
]
_CHARGES = [0.0, 0.0, 0.0, 1.0, -1.0]


def _system():
    return ferric.QmmmSystem(_SYMBOLS, _COORDS, _CHARGES, qm_indices=[0, 1, 2])


def _names(c):
    return [a["name"] for a in c["approximations"]]


def _detail(c, name):
    hits = [a["detail"] for a in c["approximations"] if a["name"] == name]
    assert len(hits) == 1, f"expected one {name!r} record, got {_names(c)}"
    return hits[0]


def _ungraded_without_source(c):
    assert c["grade"] == "ungraded"
    assert c.get("grade_source") is None
    assert c.get("measured_error") is None
    assert c["build"]["commit"] == ferric.build_info()["commit"]


def test_qmmm_certificate_records_the_embedding():
    r = ferric.run_qmmm(_system(), "sto-3g")
    c = r.certificate
    assert c["capability"] == "qmmm:rhf"
    _ungraded_without_source(c)
    assert _detail(c, "qmmm-partition") == (
        "3 QM atom(s), 2 MM atom(s), QM charge 0, multiplicity 1"
    )
    assert _detail(c, "qmmm-embedding") == (
        "electrostatic embedding: 2 point and 0 Gaussian-smeared MM charge(s), "
        "0 Thole polarizable site(s), boundary scheme Keep "
        "(0 off-atom boundary charge(s)), 0 link atom(s)"
    )
    # The SCF reference carries the same potential as its own approximation.
    assert "scf external-potential" in _names(c)
    assert "mm-force-field" not in _names(c)
    assert c["warnings"] == [
        w for w in c["warnings"] if w.startswith("scf reference: ")
    ]


def test_qmmm_certificate_records_the_boundary_scheme_and_force_field():
    top = ferric.MmTopology.from_amber_units(
        [0.0, 0.0, 0.0, 1.0, -1.0],
        [1.0] * 5,
        [0.0] * 5,
        [],
        [],
        [],
    )
    r = ferric.run_qmmm(_system(), "sto-3g", mm_topology=top)
    c = r.certificate
    assert _detail(c, "mm-force-field").endswith("over 5 atom(s)")


def test_qmmm_optimize_certificate_records_the_moved_atoms_and_optimizer():
    r = ferric.run_optimize_qmmm(_system(), "sto-3g", max_steps=2)
    c = r.certificate
    assert c["capability"] == "qmmm-optimize:rhf"
    _ungraded_without_source(c)
    assert _detail(c, "qmmm-move-mm") == "None"
    assert "BFGS, at most 2 steps" in _detail(c, "optimizer")
    assert "qmmm-embedding" in _names(c)
    assert "scf external-potential" in _names(c)
    assert r.converged or any("did not converge" in w for w in c["warnings"])


def _lih():
    return ferric.Molecule.from_xyz_string("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", 0, 1)


def _cdft_state(target=2.2, **kw):
    kw.setdefault("lambda_tol", 1e-5)
    return ferric.run_cdft(
        _lih(),
        ferric.BasisSet.bundled("sto-3g"),
        [ferric.CdftConstraint([0], target, kind="charge")],
        max_outer=40,
        **kw,
    )


def test_cdft_certificate_records_the_constraint_and_weight_grid():
    r = _cdft_state()
    c = r.certificate
    assert c["capability"] == "cdft:uhf"
    _ungraded_without_source(c)
    d = _detail(c, "cdft-constraint")
    assert d.startswith("Total population of atoms [0], target 2.2, lambda ")
    assert "(99, 302)" in _detail(c, "cdft-weight-grid")
    assert "tolerance 1e-5 e" in _detail(c, "cdft-outer-loop")
    assert r.converged
    assert not any("did not converge" in w for w in c["warnings"])
    # The descent is the stability check on HF; no "not checked" caveat remains.
    assert "stability descent on" in _detail(c, "cdft-outer-loop")
    assert not any("stability not checked" in w for w in c["warnings"])
    ks = _cdft_state(2.4, functional="PBE", lambda_tol=1e-4).certificate
    assert ks["capability"] == "cdft:uks"
    assert "skipped on a KS reference" in _detail(ks, "cdft-outer-loop")
    assert any("descent is skipped" in w for w in ks["warnings"])


def test_cdft_coupling_certificate_folds_in_both_states():
    mol = ferric.Molecule.from_xyz_string("2\nHe2+\nHe 0 0 0\nHe 0 0 2.5\n", 1, 2)
    kw = dict(lambda_tol=1e-2, max_outer=40, level_shift=0.2)
    bs = ferric.BasisSet.bundled("sto-3g")
    a = ferric.run_cdft(mol, bs, [ferric.CdftConstraint([0], 1.0)], **kw)
    b = ferric.run_cdft(mol, bs, [ferric.CdftConstraint([1], 1.0)], **kw)
    c = ferric.cdft_coupling(a, b).certificate
    assert c["capability"] == "cdft-coupling"
    _ungraded_without_source(c)
    assert _detail(c, "coupling-formula").startswith("Wu-Van Voorhis orthogonalized")
    assert _detail(c, "scf cdft-constraint").startswith("Total population of atoms [0]")
    assert _detail(c, "state B cdft-constraint").startswith(
        "Total population of atoms [1]"
    )


def test_saddle_certificate_records_the_search_and_hessian():
    nh3 = ferric.Molecule.from_xyz_string(
        "4\nnh3 near planar\n"
        "N  0.0000  0.0000  0.0000\n"
        "H  0.0000  1.0100  0.1000\n"
        "H  0.8747 -0.5050  0.1000\n"
        "H -0.8747 -0.5050  0.1000\n"
    )
    r = ferric.run_saddle(nh3, "sto-3g", max_steps=40, delta=0.01)
    c = r.certificate
    assert c["capability"] == "saddle:rhf"
    _ungraded_without_source(c)
    assert r.is_transition_state()
    assert _detail(c, "saddle-search").startswith("P-RFO following mode 0")
    assert _detail(c, "hessian") == "finite-difference, delta 0.01 Bohr"
    assert not any("imaginary mode" in w for w in c["warnings"])
