"""`ferric.esp_on_surface` binds `ferric_scf::properties::esp_on_surface`.

Reference numbers are the Rust integration test's own output (water, cc-pVDZ,
RHF, vdw_scale=1.4, 110 Lebedev points), printed by
`cargo test -p ferric-scf --test esp_surface -- --nocapture`
(`surface_counts_buried_points_for_polyatomic`). The binding must reproduce them
to 1e-10 so the Lebedev grid, vdW scaling and buried-point rule cannot drift
from the Rust one.
"""

import numpy as np
import pytest

import ferric

WATER_XYZ = (
    "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"
)

# From the Rust test (see module docstring).
RUST_N_POINTS = 154
RUST_N_BURIED = 176
RUST_SUM = 9.085327916437809e-1
RUST_SUM_ABS = 3.987656600183048e0
RUST_FIRST = -2.250396583511804e-2
RUST_LAST = 3.530571792071813e-2


@pytest.fixture(scope="module")
def water(tmp_path_factory):
    p = tmp_path_factory.mktemp("esp") / "water.xyz"
    p.write_text(WATER_XYZ)
    mol = ferric.Molecule.from_xyz(str(p))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    return mol, bs, ferric.run_rhf(mol, bs)


def test_matches_the_rust_surface_esp(water):
    mol, bs, rhf = water
    pts, esp, n_buried = ferric.esp_on_surface(
        mol, bs, rhf, vdw_scale=1.4, n_angular=110
    )
    esp = np.asarray(esp)
    assert pts.shape == (RUST_N_POINTS, 3)
    assert esp.shape == (RUST_N_POINTS,)
    assert n_buried == RUST_N_BURIED
    assert esp.sum() == pytest.approx(RUST_SUM, abs=1e-10)
    assert np.abs(esp).sum() == pytest.approx(RUST_SUM_ABS, abs=1e-10)
    assert esp[0] == pytest.approx(RUST_FIRST, abs=1e-10)
    assert esp[-1] == pytest.approx(RUST_LAST, abs=1e-10)


def test_defaults_are_the_rust_test_settings(water):
    mol, bs, rhf = water
    pts, esp, n_buried = ferric.esp_on_surface(mol, bs, rhf)
    assert (len(esp), n_buried) == (RUST_N_POINTS, RUST_N_BURIED)


def test_points_are_consistent_with_esp_at_points(water):
    """The returned potential is the exact ESP at the returned points."""
    mol, bs, rhf = water
    pts, esp, _ = ferric.esp_on_surface(mol, bs, rhf)
    again = np.asarray(ferric.esp_at_points(mol, bs, rhf, pts))
    assert np.max(np.abs(again - np.asarray(esp))) < 1e-12


def test_buried_count_nonzero_for_polyatomic_zero_for_atom(water, tmp_path):
    mol, bs, rhf = water
    pts, _esp, n_buried = ferric.esp_on_surface(mol, bs, rhf, n_angular=110)
    assert n_buried > 0
    assert len(pts) + n_buried == 3 * 110

    p = tmp_path / "ne.xyz"
    p.write_text("1\nne\nNe 0.0 0.0 0.0\n")
    ne = ferric.Molecule.from_xyz(str(p))
    ne_bs = ferric.BasisSet.bundled("sto-3g")
    ne_r = ferric.run_rhf(ne, ne_bs)
    ne_pts, _e, ne_buried = ferric.esp_on_surface(ne, ne_bs, ne_r, n_angular=110)
    assert ne_buried == 0
    assert ne_pts.shape == (110, 3)


def test_vdw_scale_is_honoured(water):
    """Changing the scale changes which points are buried."""
    mol, bs, rhf = water
    _, _, b_small = ferric.esp_on_surface(mol, bs, rhf, vdw_scale=1.0)
    _, _, b_big = ferric.esp_on_surface(mol, bs, rhf, vdw_scale=2.0)
    assert b_small != b_big


def test_rejects_nonpositive_scale(water):
    mol, bs, rhf = water
    with pytest.raises(Exception, match="vdw_scale"):
        ferric.esp_on_surface(mol, bs, rhf, vdw_scale=0.0)
