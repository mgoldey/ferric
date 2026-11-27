"""Finite-difference helpers for validating analytic gradients.

A ported or prototyped method can check its gradient against a numerical one
without reimplementing the stencil::

    def energy(x):  # x: (N, 3) Bohr
        mol = ferric.Molecule.from_arrays(symbols, x, units="bohr")
        return ferric.run_rhf(mol, basis, ri_j_precision="f64").energy

    numerical = finite_difference_gradient(energy, mol.coords_bohr())
    gradient_check(analytic, numerical, atol=1e-6)

The step and the units are the caller's: the gradient comes back in the
energy unit per the unit of ``coords``. ferric's analytic gradients are in
Hartree/Bohr, so pass Bohr coordinates (``Molecule.coords_bohr()``) to compare
against them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["finite_difference_gradient", "gradient_check"]


def finite_difference_gradient(energy, coords, h=1e-3):
    """Central-difference gradient of ``energy(coords)``.

    ``coords`` is an (N, 3) array and ``energy`` returns a scalar for such an
    array; ``h`` is the displacement in the unit of ``coords``. Costs 6 N
    energy evaluations and returns an (N, 3) array. The truncation error is
    O(h^2) and the round-off error is the noise of ``energy`` divided by
    ``h``: the default 1e-3 suits an SCF energy converged to about 1e-10 with
    coordinates in Bohr or Angstrom.
    """
    coords = np.array(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
    if not h > 0.0:
        raise ValueError(f"h must be positive, got {h}")
    grad = np.zeros_like(coords)
    for index in np.ndindex(*coords.shape):
        x0 = coords[index]
        coords[index] = x0 + h
        e_plus = energy(coords.copy())
        coords[index] = x0 - h
        e_minus = energy(coords.copy())
        coords[index] = x0
        grad[index] = (e_plus - e_minus) / (2.0 * h)
    return grad


def gradient_check(analytic, numerical, atol=1e-6):
    """Largest absolute difference between two gradient arrays.

    Raises ``AssertionError`` naming the worst element when it exceeds
    ``atol``; ``atol=None`` only measures. Shapes must match (``ValueError``).
    """
    a = np.asarray(analytic, dtype=float)
    n = np.asarray(numerical, dtype=float)
    if a.shape != n.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {n.shape}")
    diff = np.abs(a - n)
    worst = float(diff.max()) if diff.size else 0.0
    if atol is not None and worst > atol:
        index = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            f"gradients differ by {worst:.3e} at index {tuple(int(i) for i in index)} "
            f"(analytic {a[index]:.10e}, numerical {n[index]:.10e}), above atol={atol:g}"
        )
    return worst
