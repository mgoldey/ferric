"""`run_frequencies(df_j_aux=, df_k_aux=)`: the analytic RI Hessian from Python.

The anchor is an independent construction: the analytic Hessian of the
fitted energy (ferric_scf::hessian_df / hessian_df_k) must equal the central
finite difference of the fitted ANALYTIC gradient (`hessian="fd"`, same
kwargs), which shares no Hessian code with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
NH2 = "3\n\nN 0 0 0\nH 0 0.802 0.597\nH 0 -0.802 0.597\n"
AUX = "cc-pvdz-ri"
TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}


def _freq(xyz, multiplicity=1, **kw):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    return ferric.run_frequencies(mol, "sto-3g", **kw, **TIGHT)


@pytest.mark.parametrize(
    "fits",
    [{"df_j_aux": AUX}, {"df_k_aux": AUX}, {"df_j_aux": AUX, "df_k_aux": AUX}],
    ids=["ri-j", "ri-k", "ri-jk"],
)
@pytest.mark.parametrize(
    "xyz, reference, mult",
    [(WATER, "rhf", 1), (NH2, "uhf", 2), (NH2, "rohf", 2)],
    ids=["rhf", "uhf", "rohf"],
)
def test_analytic_ri_hessian_matches_finite_difference(fits, xyz, reference, mult):
    a = _freq(xyz, mult, reference=reference, hessian="analytic", **fits)
    f = _freq(xyz, mult, reference=reference, hessian="fd", **fits)
    assert a.hessian_source == "analytic"
    assert f.hessian_source == "finite-difference"
    ha, hf = np.array(a.cartesian_hessian), np.array(f.cartesian_hessian)
    np.testing.assert_allclose(ha, hf, atol=2e-5)
    # The fit is in use: the exact analytic Hessian differs.
    exact = np.array(
        _freq(xyz, mult, reference=reference, hessian="analytic").cartesian_hessian
    )
    assert np.abs(ha - exact).max() > 1e-5


def test_auto_picks_analytic_for_ri_and_fd_where_unsupported():
    fits = {"df_j_aux": AUX, "df_k_aux": AUX}
    assert _freq(WATER, **fits).hessian_source == "analytic"
    assert _freq(NH2, 2, reference="uhf", **fits).hessian_source == "analytic"
    assert _freq(NH2, 2, reference="rohf", **fits).hessian_source == "analytic"
    charges = {"point_charges": [(0.3, 0.0, 0.0, 6.0)]}
    assert _freq(WATER, **fits, **charges).hessian_source == "finite-difference"
    with pytest.raises((ValueError, RuntimeError)):
        _freq(WATER, hessian="analytic", **fits, **charges)


def test_exact_is_unchanged_without_the_kwargs():
    a = _freq(WATER, hessian="analytic")
    b = _freq(WATER, hessian="analytic", df_j_aux="none", df_k_aux="none")
    np.testing.assert_array_equal(
        np.array(a.cartesian_hessian), np.array(b.cartesian_hessian)
    )
