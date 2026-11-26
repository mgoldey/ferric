"""Finite-difference gradient helpers."""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")


def test_fd_gradient_of_quadratic():
    import ferric.finite_difference as fd

    def energy(x):
        return float(np.sum((x - 1.2) ** 2))

    x = np.zeros((2, 3))
    g = fd.finite_difference_gradient(energy, x, h=1e-6)
    assert np.allclose(g, -2.4)


def test_gradient_check():
    import ferric.finite_difference as fd

    a = np.array([[1.0, 2.0, 3.0]])
    n = np.array([[1.0 + 1e-8, 2.0, 3.0]])
    assert fd.gradient_check(a, n) < 1e-7
