"""Time-correlation functions and transport properties for MD trajectories.

All arrays are in consistent units: positions/velocities as (n_frames, n_atoms, 3)
in Angstrom (or any length) and Angstrom/fs respectively, time in fs. The
diffusion coefficient comes back in length^2/time.
"""

from __future__ import annotations

import numpy as np


def _as_3d(name, arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name} must have shape (n_frames, n_atoms, 3), got {a.shape}")
    return a


def velocity_autocorrelation(velocities, normalize=True):
    """Velocity autocorrelation function C(tau) = <v(0).v(tau)>.

    Averaged over all starting frames and atoms. ``normalize=True`` divides by
    C(0) so C(0) == 1.
    """
    v = _as_3d("velocities", velocities)
    n_frames = v.shape[0]
    out = np.zeros(n_frames, dtype=float)
    for lag in range(n_frames):
        a = v[: n_frames - lag]
        b = v[lag:]
        out[lag] = float(np.mean(np.sum(a * b, axis=2)))
    if normalize and out[0] != 0.0:
        out = out / out[0]
    return out


def mean_squared_displacement(positions):
    """Mean squared displacement < |r(t)-r(0)|^2 > averaged over origins."""
    r = _as_3d("positions", positions)
    n_frames = r.shape[0]
    out = np.zeros(n_frames, dtype=float)
    for lag in range(n_frames):
        dr = r[lag:] - r[: n_frames - lag]
        out[lag] = float(np.mean(np.sum(dr * dr, axis=2)))
    return out


def diffusion_coefficient(velocities, dt, method="green-kubo"):
    """Self-diffusion coefficient from an MD trajectory.

    ``method="green-kubo"`` integrates the velocity autocorrelation function:
    D = (1/3) integral_0^inf C(t) dt.

    ``method="einstein"`` is not supported from velocities alone; pass positions
    to ``diffusion_from_msd``.
    """
    if method != "green-kubo":
        raise ValueError('method must be "green-kubo"')
    vacf = velocity_autocorrelation(velocities, normalize=False)
    integral = float(np.trapezoid(vacf, dx=float(dt)))
    return integral / 3.0


def diffusion_from_msd(positions, dt):
    """Einstein diffusion coefficient D = slope(MSD) / 6 for 3-D."""
    msd = mean_squared_displacement(positions)
    # Linear fit of MSD vs time for lags beyond the ballistic regime.
    t = np.arange(msd.shape[0]) * float(dt)
    if msd.shape[0] < 3:
        raise ValueError("need at least 3 frames")
    slope, _ = np.polyfit(t[1:], msd[1:], 1)
    return float(slope) / 6.0
