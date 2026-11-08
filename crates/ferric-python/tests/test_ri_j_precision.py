"""`ri_j_precision` / `scf_ri_j_precision`: the RI-J tensor storage precision.

The default is f32; "f32" and "f64" are honoured as given and reported by the
result (`ri_j_precision`, the certificate's `ri-j` entry); "f64" is the
pre-2026-11-07 default; anything else raises ValueError before any integral.
"""

import ferric
import pytest

WATER = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"


def _water():
    return ferric.Molecule.from_xyz_string(WATER)


def _ri_j_detail(r):
    (a,) = [a for a in r.certificate["approximations"] if a["name"] == "ri-j"]
    return a["detail"]


def test_run_dft_default_is_f32_and_explicit_values_are_honoured():
    bs = ferric.BasisSet.bundled("sto-3g")
    d = ferric.run_dft(_water(), bs, "PBE")
    f32 = ferric.run_dft(_water(), bs, "PBE", ri_j_precision="f32")
    f64 = ferric.run_dft(_water(), bs, "PBE", ri_j_precision="f64")
    assert d.ri_j_precision == "f32"
    assert f32.ri_j_precision == "f32"
    assert f64.ri_j_precision == "f64"
    assert _ri_j_detail(d).endswith("f32 tensor storage")
    assert _ri_j_detail(f64).endswith("f64 tensor storage")
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
