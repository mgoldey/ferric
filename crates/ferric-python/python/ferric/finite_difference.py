"""Finite-difference helpers for validating analytic gradients.

These are the reference comparisons the Python bindings use internally
(``test_python_gradients.py``); exposing them lets a ported method check its
own gradient against a numerical one without reimplementing the stencil.
"""

from __future__ import annotations

import numpy as np


def finite_difference_gradient(energy, coords, h=1e-5):
    """Central-difference gradient of ``energy(coords)``.

    ``coords`` is an (N, 3) array (Angstrom or Bohr, matching ``energy``);
    ``energy`` returns a scalar for a given (N, 3) array. Returns an
    (N, 3) array.
    """
    coords = np.asarray(coords, dtype=float).copy()
    grad = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for j in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, j] += h
            minus[i, j] -= h
            grad[i, j] = (energy(plus) - energy(minus)) / (2.0 * h)
    return grad


def gradient_check(analytic, numerical, atol=1e-6):
    """Max absolute difference between two gradient arrays."""
    a = np.asarray(analytic, dtype=float)
    n = np.asarray(numerical, dtype=float)
    if a.shape != n.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {n.shape}")
    return float(np.max(np.abs(a - n)))
