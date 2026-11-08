"""`ri_k_precision` / `scf_ri_k_precision`: the RI-K dressed-tensor storage.

The default is f64 (the packed tensor, bit-identical to the former square
one); "f32" stores it in f32 and moves the energy at the 1e-7 Ha scale; the
result (`ri_k_precision`, the certificate's `ri-k` entry) says which was
used; anything else raises ValueError before any integral.
"""

import ferric
import pytest

WATER = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"


def _water():
    return ferric.Molecule.from_xyz_string(WATER)


def _ri_k_detail(r):
    (a,) = [a for a in r.certificate["approximations"] if a["name"] == "ri-k"]
    return a["detail"]


def test_run_dft_default_is_f64_and_explicit_values_are_honoured():
    bs = ferric.BasisSet.bundled("sto-3g")
    d = ferric.run_dft(_water(), bs, "B3LYP")
    f64 = ferric.run_dft(_water(), bs, "B3LYP", ri_k_precision="f64")
    f32 = ferric.run_dft(_water(), bs, "B3LYP", ri_k_precision="f32")
    assert d.ri_k_precision == "f64"
    assert f64.ri_k_precision == "f64"
    assert f32.ri_k_precision == "f32"
    assert _ri_k_detail(d).endswith("f64 tensor storage")
    assert _ri_k_detail(f32).endswith("f32 tensor storage")
    assert d.e_scf == f64.e_scf
    assert d.e_scf != f32.e_scf
    # The f32 shift is the storage rounding, not a different method.
    assert abs(d.e_scf - f32.e_scf) < 1e-6


def test_no_ri_k_reports_no_precision():
    bs = ferric.BasisSet.bundled("sto-3g")
    assert ferric.run_dft(_water(), bs, "PBE").ri_k_precision is None
    assert ferric.run_rhf(_water(), bs).ri_k_precision is None


@pytest.mark.parametrize("bad", ["F32", "single", "fp64", ""])
def test_an_unknown_value_raises(bad):
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match="ri_k_precision"):
        ferric.run_dft(_water(), bs, "B3LYP", ri_k_precision=bad)
    with pytest.raises(ValueError, match="ri_k_precision"):
        ferric.run_rhf(
            _water(), bs, df_k_aux="def2-universal-jkfit", ri_k_precision=bad
        )
