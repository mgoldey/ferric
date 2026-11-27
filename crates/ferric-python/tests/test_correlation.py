"""Time-correlation and transport helpers (ferric.correlation) against
trajectories whose correlation functions are known in closed form.

* Free flight r_i(t) = v_i t: C(tau) = <|v_i|^2> at every lag,
  MSD(tau) = <|v_i|^2> tau^2, and the trapezoid Green-Kubo integral over the
  n - 1 recorded intervals is <|v_i|^2> (n - 1) dt / 3. All exact to round-off.
* A Gaussian random walk with per-component step variance s^2 per frame:
  MSD(tau) = 3 s^2 tau / dt, so D = s^2 / (2 dt). Statistical, see the bar.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ferric")
c = pytest.importorskip("ferric.correlation")

DT = 0.5
N_FRAMES = 12
V = np.array([[1.0, -2.0, 0.5], [0.0, 0.25, 3.0], [-1.5, 0.0, 0.0]])
V2 = float(np.mean(np.sum(V * V, axis=1)))


def _free_flight():
    t = np.arange(N_FRAMES)[:, None, None] * DT
    return V[None] * t, np.broadcast_to(V, (N_FRAMES, *V.shape))


def test_free_flight_vacf_is_the_mean_square_speed_at_every_lag():
    _, v = _free_flight()
    np.testing.assert_allclose(
        c.velocity_autocorrelation(v, normalize=False),
        np.full(N_FRAMES, V2),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        c.velocity_autocorrelation(v), np.ones(N_FRAMES), rtol=1e-13
    )


def test_free_flight_msd_is_quadratic_in_the_lag():
    r, _ = _free_flight()
    lags = np.arange(N_FRAMES) * DT
    np.testing.assert_allclose(
        c.mean_squared_displacement(r), V2 * lags**2, rtol=1e-12, atol=0.0
    )


def test_green_kubo_integrates_the_unnormalized_vacf_over_the_recorded_lags():
    _, v = _free_flight()
    assert c.diffusion_coefficient(v, DT) == pytest.approx(
        V2 * (N_FRAMES - 1) * DT / 3.0, rel=1e-13
    )


def test_einstein_slope_recovers_the_random_walk_diffusion_coefficient():
    n_frames, n_atoms, sigma = 400, 500, 0.3
    rng = np.random.default_rng(20261127)
    steps = rng.normal(scale=sigma, size=(n_frames, n_atoms, 3))
    steps[0] = 0.0
    d = c.diffusion_from_msd(np.cumsum(steps, axis=0), DT)
    expected = sigma**2 / (2.0 * DT)
    # MSD(tau) is a chi-square mean over 3 n_atoms independent components per
    # origin, and the origins inside the fit window are strongly correlated, so
    # the slope's relative error is about sqrt(2 / (3 n_atoms)) = 3.7%: 4 sigma.
    assert d == pytest.approx(expected, rel=4.0 * np.sqrt(2.0 / (3 * n_atoms)))


def test_einstein_fit_skips_the_ballistic_lags():
    """Free flight has no diffusive regime: MSD = V2 tau^2, whose least-squares
    slope over lags [first, last] is V2 (t_first + t_last). The default window
    of an 11-interval trajectory is lags 2..5."""
    r, _ = _free_flight()
    assert c.diffusion_from_msd(r, DT) == pytest.approx(
        V2 * (2 + 5) * DT / 6.0, rel=1e-12
    )


@pytest.mark.parametrize(
    "call, needle",
    [
        (
            lambda: c.velocity_autocorrelation(np.zeros((4, 3))),
            "velocities must have shape",
        ),
        (
            lambda: c.mean_squared_displacement(np.zeros((4, 2, 2))),
            "positions must have shape",
        ),
        (lambda: c.diffusion_from_msd(np.zeros((3, 1, 3)), DT), "fewer than two lags"),
        (
            lambda: c.diffusion_from_msd(
                np.zeros((40, 1, 3)), DT, fit_fraction=(0.6, 0.5)
            ),
            "fit_fraction must satisfy",
        ),
        (
            lambda: c.velocity_autocorrelation(np.zeros((0, 1, 3))),
            "at least one frame",
        ),
        (
            lambda: c.mean_squared_displacement(np.zeros((4, 0, 3))),
            "at least one frame",
        ),
        (lambda: c.diffusion_coefficient(np.ones((12, 1, 3)), -1.0), "dt must be"),
        (lambda: c.diffusion_from_msd(np.zeros((40, 1, 3)), 0.0), "dt must be"),
        (
            lambda: c.diffusion_coefficient(np.ones((12, 1, 3)), float("nan")),
            "dt must be",
        ),
    ],
)
def test_bad_input_raises_value_error(call, needle):
    with pytest.raises(ValueError, match=needle):
        call()
