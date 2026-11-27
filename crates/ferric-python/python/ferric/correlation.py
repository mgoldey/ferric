"""Time-correlation functions and transport properties for MD trajectories.

Positions and velocities are (n_frames, n_atoms, 3) arrays at a fixed frame
spacing ``dt``, as ``MdResult.positions_angstrom`` / ``MdResult.velocities``
return them (Angstrom, Angstrom/fs; ``dt`` is then ``dt_fs`` times the
recording stride). Nothing here converts units: a diffusion coefficient
comes back in length^2/time of the inputs (Angstrom^2/fs for those).
Positions must be unwrapped. Every average runs over all atoms and all
time origins, so the late lags are averages over few origins.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "diffusion_coefficient",
    "diffusion_from_msd",
    "mean_squared_displacement",
    "velocity_autocorrelation",
]


def _as_3d(name, arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (n_frames, n_atoms, 3), got {a.shape}"
        )
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} needs at least one frame and one atom, got {a.shape}")
    return a


def _spacing(dt):
    dt = float(dt)
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be a positive finite frame spacing, got {dt}")
    return dt


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


def diffusion_coefficient(velocities, dt):
    """Green-Kubo self-diffusion coefficient D = (1/3) integral C(t) dt,
    C the unnormalized velocity autocorrelation function, by the trapezoid
    rule over every recorded lag (the integral is cut at the trajectory
    length). ``diffusion_from_msd`` is the Einstein route from positions.
    """
    dt = _spacing(dt)
    vacf = velocity_autocorrelation(velocities, normalize=False)
    integral = float(np.trapezoid(vacf, dx=dt))
    return integral / 3.0


def diffusion_from_msd(positions, dt, fit_fraction=(0.1, 0.5)):
    """Einstein diffusion coefficient in three dimensions, D = slope / 6 of
    a least-squares line through MSD(t).

    The line is fitted over the lags between ``fit_fraction[0]`` and
    ``fit_fraction[1]`` of the trajectory length: the early lags are
    ballistic (MSD ~ t^2) and the late ones average over few origins.
    Needs at least two lags inside the window (``ValueError``).
    """
    dt = _spacing(dt)
    msd = mean_squared_displacement(positions)
    lo, hi = fit_fraction
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(
            f"fit_fraction must satisfy 0 <= lo < hi <= 1, got {fit_fraction}"
        )
    n = msd.shape[0]
    first = max(1, int(np.ceil(lo * (n - 1))))
    last = int(np.floor(hi * (n - 1)))
    if last - first < 1:
        raise ValueError(
            f"{n} frames leave fewer than two lags in fit_fraction={fit_fraction}"
        )
    t = np.arange(first, last + 1) * dt
    slope, _ = np.polyfit(t, msd[first : last + 1], 1)
    return float(slope) / 6.0
