"""Time-correlation / transport helpers (ferric.correlation)."""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")


def test_vacf_normalised_at_zero():
    import ferric.correlation as c

    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 4, 3))
    vacf = c.velocity_autocorrelation(v)
    assert vacf.shape == (20,)
    assert abs(vacf[0] - 1.0) < 1e-12


def test_msd_zero_at_zero_lag():
    import ferric.correlation as c

    rng = np.random.default_rng(1)
    r = rng.normal(size=(10, 3, 3))
    msd = c.mean_squared_displacement(r)
    assert msd[0] == 0.0
    assert np.all(msd >= 0)


def test_diffusion_green_kubo_runs():
    import ferric.correlation as c

    rng = np.random.default_rng(2)
    v = rng.normal(size=(15, 2, 3))
    d = c.diffusion_coefficient(v, 1.0)
    assert isinstance(d, float)
