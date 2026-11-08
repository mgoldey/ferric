"""`ri_j_precision` / `scf_ri_j_precision`: the RI-J tensor storage precision.

The default is f32 for single points and f64 for the geometry and frequency
drivers (run_optimize*, run_frequencies, run_saddle, run_irc: the f32 surface
jitters between nearby geometries); "f32" and "f64" are honoured as given on
every surface and reported by the result (`ri_j_precision`, the certificate's
`ri-j` entry, which also says why); "f64" is the pre-2026-11-07 default;
anything else raises ValueError before any integral.
"""

import ferric
import pytest

WATER = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"


def _water():
    return ferric.Molecule.from_xyz_string(WATER)


def _ri_j_detail(r, name="ri-j"):
    (a,) = [a for a in r.certificate["approximations"] if a["name"] == name]
    return a["detail"]


def test_run_dft_default_is_f32_and_explicit_values_are_honoured():
    bs = ferric.BasisSet.bundled("sto-3g")
    d = ferric.run_dft(_water(), bs, "PBE")
    f32 = ferric.run_dft(_water(), bs, "PBE", ri_j_precision="f32")
    f64 = ferric.run_dft(_water(), bs, "PBE", ri_j_precision="f64")
    assert d.ri_j_precision == "f32"
    assert f32.ri_j_precision == "f32"
    assert f64.ri_j_precision == "f64"
    assert _ri_j_detail(d).endswith("f32 tensor storage (requested f32: default)")
    assert _ri_j_detail(f64).endswith("f64 tensor storage (requested f64: explicit)")
    assert d.e_scf == f32.e_scf
    assert d.e_scf != f64.e_scf
    # The f32 shift is the µEh-scale storage error, not a different method.
    assert abs(d.e_scf - f64.e_scf) < 1e-5


def test_run_rhf_honours_the_kwarg_and_reports_no_precision_without_ri_j():
    bs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_rhf(
        _water(), bs, df_j_aux="def2-universal-jkfit", ri_j_precision="f64"
    )
    assert r.ri_j_precision == "f64"
    r = ferric.run_rhf(_water(), bs, df_j_aux="def2-universal-jkfit")
    assert r.ri_j_precision == "f32"
    # Exact Coulomb: no RI-J tensor, nothing to report.
    assert ferric.run_rhf(_water(), bs).ri_j_precision is None


@pytest.mark.parametrize("bad", ["F32", "single", "fp64", ""])
def test_an_unknown_value_raises(bad):
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match="ri_j_precision"):
        ferric.run_dft(_water(), bs, "PBE", ri_j_precision=bad)
    with pytest.raises(ValueError, match="ri_j_precision"):
        ferric.run_rimp2(
            _water(),
            bs,
            ferric.BasisSet.bundled("cc-pvdz-ri"),
            scf_jk="rijk",
            scf_ri_j_precision=bad,
        )


def test_frequencies_default_to_f64_and_honour_an_explicit_f32():
    # PBE takes RI-J by default; "auto" runs the analytic moving-grid Hessian,
    # whose reference SCF the certificate describes.
    d = ferric.run_frequencies(_water(), "sto-3g", xc="PBE")
    f32 = ferric.run_frequencies(_water(), "sto-3g", xc="PBE", scf_ri_j_precision="f32")
    f64 = ferric.run_frequencies(_water(), "sto-3g", xc="PBE", scf_ri_j_precision="f64")
    # A task certificate carries its reference SCF's entries as "scf <name>".
    assert _ri_j_detail(d, "scf ri-j").endswith(
        "f64 tensor storage (requested f64: geometry-task default)"
    )
    assert _ri_j_detail(f32, "scf ri-j").endswith(
        "f32 tensor storage (requested f32: explicit)"
    )
    assert _ri_j_detail(f64, "scf ri-j").endswith(
        "f64 tensor storage (requested f64: explicit)"
    )
    assert d.energy == f64.energy
    assert d.energy != f32.energy
    assert d.frequencies == f64.frequencies
    # The same molecule as a single point keeps the f32 default.
    sp = ferric.run_dft(_water(), ferric.BasisSet.bundled("sto-3g"), "PBE")
    assert sp.ri_j_precision == "f32"


def test_qmmm_optimize_defaults_to_f64_and_honours_an_explicit_f32():
    sys = ferric.QmmmSystem(
        ["O", "H", "H"],
        [(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)],
        [0.0, 0.0, 0.0],
        qm_indices=[0, 1, 2],
    )

    def first_energy(**kw):
        r = ferric.run_optimize_qmmm(
            sys, "sto-3g", method="rks", xc="PBE", max_steps=1, **kw
        )
        return r.energies()[0]

    d = first_energy()
    assert d == first_energy(scf_ri_j_precision="f64")
    assert d != first_energy(scf_ri_j_precision="f32")


@pytest.mark.parametrize("driver", ["run_saddle", "run_irc"])
def test_saddle_and_irc_take_a_strict_scf_ri_j_precision(driver):
    args = (_water(), "sto-3g") + (([0.0] * 9,) if driver == "run_irc" else ())
    with pytest.raises(ValueError, match="ri_j_precision"):
        getattr(ferric, driver)(*args, xc="PBE", scf_ri_j_precision="half")
