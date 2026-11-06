"""GFN-xTB geometry optimization / frequencies through the Python bindings.

The functions always exist; a build without the `xtb` cargo feature raises
RuntimeError naming it, and these tests skip in that case. The Rust-level
validation (CLI energies, FD gradient, optimizer, Hessian) is in
`crates/ferric-xtb/tests/`; these pin the Python surface.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

WATER = "3\nd\nO 0.0 0.0 0.15\nH 0.0 0.90 -0.45\nH 0.0 -0.80 -0.50\n"
CO2 = "3\nd\nC 0.0 0.0 0.0\nO 0.0 0.0 1.20\nO 0.0 0.0 -1.10\n"


def _mol(xyz):
    return ferric.Molecule.from_xyz_string(xyz, 0, 1)


@pytest.fixture(scope="module", autouse=True)
def _need_xtb():
    try:
        ferric.run_xtb_energy_gradient(_mol(WATER))
    except RuntimeError as e:
        if "xtb` feature" in str(e):
            pytest.skip("ferric built without the xtb feature")
        raise


def test_optimize_converges_and_gradient_is_small():
    r = ferric.run_xtb_optimize(
        _mol(WATER), g_max_thresh=1e-4, g_rms_thresh=5e-5, e_conv=1e-8
    )
    assert r.converged
    e, g = ferric.run_xtb_energy_gradient(r.mol())
    assert e == pytest.approx(r.energy, abs=1e-9)
    assert max(abs(x) for row in g for x in row) < 1e-4
    assert r.energy_trace[1] < r.energy_trace[0]
    assert r.certificate["capability"] == "optimize:xtb"


def test_frequencies_water_and_co2_zero_mode_counts():
    w = ferric.run_xtb_optimize(
        _mol(WATER), g_max_thresh=1e-4, g_rms_thresh=5e-5, e_conv=1e-8
    )
    f = ferric.run_xtb_frequencies(w.mol())
    assert (len(f.trans_rot_frequencies), len(f.frequencies), f.is_linear) == (
        6,
        3,
        False,
    )
    assert all(abs(t) < 30 for t in f.trans_rot_frequencies)
    assert f.hessian_source == "finite-difference"
    c = ferric.run_xtb_optimize(
        _mol(CO2), g_max_thresh=1e-4, g_rms_thresh=5e-5, e_conv=1e-8
    )
    fc = ferric.run_xtb_frequencies(c.mol())
    assert (len(fc.trans_rot_frequencies), len(fc.frequencies), fc.is_linear) == (
        5,
        4,
        True,
    )


def test_unconverged_scc_raises_in_optimize_and_frequencies():
    with pytest.raises(RuntimeError, match="xtb"):
        ferric.run_xtb_optimize(_mol(WATER), scc_max_iter=1)
    with pytest.raises(RuntimeError, match="xtb"):
        ferric.run_xtb_frequencies(_mol(WATER), scc_max_iter=1)


def test_unknown_method_is_a_value_error():
    with pytest.raises(ValueError, match="unknown xtb method"):
        ferric.run_xtb_optimize(_mol(WATER), method="gfn3")
