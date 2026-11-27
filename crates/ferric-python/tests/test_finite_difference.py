"""Finite-difference gradient helpers (ferric.finite_difference).

* A central difference is exact for a quadratic and has error
  h^2 f'''/6 for a cubic: for f = sum x^3 that is exactly h^2 per component,
  which a forward difference (error 3 x h) or a wrong denominator misses.
* On a real surface: the central difference of the H2 RHF energy against the
  analytic gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")
fd = pytest.importorskip("ferric.finite_difference")

X = np.array([[0.3, -1.1, 2.0], [1.7, 0.4, -0.6]])


def test_central_difference_is_exact_for_a_quadratic():
    g = fd.finite_difference_gradient(
        lambda x: float(np.sum((x - 1.2) ** 2)), X, h=1e-3
    )
    np.testing.assert_allclose(g, 2.0 * (X - 1.2), rtol=0.0, atol=1e-10)


def test_central_difference_error_on_a_cubic_is_h_squared():
    h = 1e-2
    g = fd.finite_difference_gradient(lambda x: float(np.sum(x**3)), X, h=h)
    np.testing.assert_allclose(g - 3.0 * X**2, np.full(X.shape, h * h), rtol=1e-6)


def test_the_input_coordinates_are_not_modified():
    x = X.copy()
    fd.finite_difference_gradient(lambda y: float(np.sum(y)), x)
    np.testing.assert_array_equal(x, X)


def test_gradient_check_measures_and_enforces_atol():
    a = np.zeros((2, 3))
    n = a.copy()
    n[1, 2] = 3e-7
    assert fd.gradient_check(a, n, atol=1e-6) == pytest.approx(3e-7)
    with pytest.raises(AssertionError, match=r"3\.000e-07 at index \(1, 2\)"):
        fd.gradient_check(a, n, atol=1e-7)
    assert fd.gradient_check(a, n, atol=None) == pytest.approx(3e-7)
    with pytest.raises(ValueError, match="shape mismatch"):
        fd.gradient_check(a, n[:1])


def test_gradient_check_refuses_a_nan_rather_than_passing_it():
    # max() of an array holding a NaN is NaN, and NaN > atol is False.
    a = np.zeros((1, 3))
    n = np.array([[np.nan, 1.0, 0.0]])
    with pytest.raises(AssertionError, match="non-finite"):
        fd.gradient_check(a, n, atol=1e-6)
    with pytest.raises(AssertionError, match="non-finite"):
        fd.gradient_check(n, a, atol=1e-6)
    assert np.isnan(fd.gradient_check(a, n, atol=None))


def test_bad_arguments_raise_value_error():
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        fd.finite_difference_gradient(lambda x: 0.0, np.zeros(3))
    with pytest.raises(ValueError, match="h must be positive"):
        fd.finite_difference_gradient(lambda x: 0.0, X, h=0.0)


def test_h2_rhf_central_difference_matches_the_analytic_gradient():
    basis = ferric.BasisSet.bundled("sto-3g")
    conv = dict(energy_conv=1e-12, density_conv=1e-10)
    coords = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 1.5]])

    def mol(x):
        return ferric.Molecule.from_arrays(["H", "H"], x, units="bohr")

    analytic = np.asarray(
        ferric.run_rhf(mol(coords), basis, with_gradient=True, **conv).gradient()
    )
    h = 1e-3
    numerical = fd.finite_difference_gradient(
        lambda x: ferric.run_rhf(mol(x), basis, **conv).energy, coords, h=h
    )
    # Truncation h^2 E'''/6 with |E'''| of order 1 Ha/Bohr^3, plus the energy
    # noise 1e-12 / h: 1e-6 Ha/Bohr covers both, and the gradient itself is
    # four orders larger (asserted, so a zero gradient cannot pass).
    assert np.abs(analytic).max() > 1e-2
    assert fd.gradient_check(analytic, numerical, atol=1e-6) < 1e-6
