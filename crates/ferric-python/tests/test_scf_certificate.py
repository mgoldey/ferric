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


def _anchor_warnings(c):
    return [w for w in c["warnings"] if "energy anchor" in w]


def test_scf_outside_its_anchor_warns_without_changing_the_grade():
    # The anchor sets are crates/ferric-core/src/certificate/anchors.rs
    # ENERGY_ANCHORS. RKS covers H, Si, S, Cl / def2-SVP, def2-TZVP: water /
    # STO-3G is outside on element and basis. RHF covers H, C, O, Ne, Cl, Br,
    # Ag, I with STO-3G: NH3 is outside on element only. UHF covers 6-31G and
    # def2-SVP: OH / STO-3G is outside on basis only.
    sto = ferric.BasisSet.bundled("sto-3g")
    d = ferric.run_dft(_water(), sto, "PBE").certificate
    ev = d["measured_error"]["evidence"]
    assert d["grade"] == "proven"
    assert d["measured_error"]["value"] == 4.2e-12
    assert _anchor_warnings(d) == [
        f"element O (covered: H, Si, S, Cl) is outside the RKS energy anchor ({ev})",
        f"basis sto-3g is outside the RKS energy anchor ({ev})",
        # The run_dft defaults are not the evidence's settings.
        f"RI-J tensor storage f32 (evidence: f64) is outside the RKS energy anchor ({ev})",
        "XC grid (75, 110), nwchemradial, weight cut 1e-14 (evidence: (75, 110), flat, "
        f"no weight cut) is outside the RKS energy anchor ({ev})",
    ]

    nh3 = ferric.Molecule.from_xyz_string(
        "4\n\nN 0 0 0.1173\nH 0 0.9377 -0.2737\nH 0.8121 -0.4689 -0.2737\n"
        "H -0.8121 -0.4689 -0.2737\n"
    )
    r = ferric.run_rhf(nh3, sto).certificate
    ev = r["measured_error"]["evidence"]
    assert r["grade"] == "proven"
    assert r["warnings"] == [
        (
            "element N (covered: H, C, O, Ne, Cl, Br, Ag, I) is outside the RHF "
            f"energy anchor ({ev})"
        )
    ]

    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2)
    u = ferric.run_uhf(oh, sto).certificate
    ev = u["measured_error"]["evidence"]
    assert u["grade"] == "proven"
    assert _anchor_warnings(u) == [
        f"basis sto-3g is outside the UHF energy anchor ({ev})"
    ]


def test_scf_inside_its_anchor_carries_no_envelope_warning():
    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2)
    b = ferric.BasisSet.bundled("6-31g")
    for c in (ferric.run_uhf(oh, b).certificate, ferric.run_rohf(oh, b).certificate):
        assert c["measured_error"] is not None, c
        assert _anchor_warnings(c) == [], c


def test_scf_with_an_external_field_or_fock_hook_warns_outside_the_vacuum_anchor():
    # The SCF anchors' evidence runs are in vacuum: an external field or a
    # Fock hook changes the Hamiltonian, so the run is warned as outside the
    # anchor (value and grade kept). Water / STO-3G is otherwise inside the
    # RHF anchor, so these are the only anchor warnings.
    import numpy as np

    sto = ferric.BasisSet.bundled("sto-3g")
    c = ferric.run_rhf(_water(), sto, external_field=(0.0, 0.0, 1e-3)).certificate
    ev = c["measured_error"]["evidence"]
    assert c["grade"] == "proven"
    assert _anchor_warnings(c) == [
        "external potential (evidence: vacuum Hamiltonian) is outside the RHF "
        f"energy anchor ({ev})"
    ]

    def zero_hook(d):
        return 0.0, np.zeros_like(d)

    h = ferric.run_rhf(_water(), sto, fock_hook=zero_hook).certificate
    assert h["grade"] == "proven"
    assert _anchor_warnings(h) == [
        f"Fock hook (evidence: vacuum Hamiltonian) is outside the RHF energy anchor ({ev})"
    ]
