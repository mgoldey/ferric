"""Main XC grid kwargs on `run_dft` / `run_ksdft`: size and NWChem-style pruning.

`PruneScheme::NwchemLike` existed in ferric-dft (and the CLI's `[dft]
grid_prune`) but no Python caller could reach it. These pin:

  1. The trivial limit: spelling out the default grid (75, 110, "none") is
     BYTE-identical to passing nothing.
  2. Pruning really reaches the grid (fewer points) and the SCF (a different,
     but close, energy).
  3. Every unsupported value raises ValueError instead of being defaulted.
"""

from pathlib import Path

import pytest

ferric = pytest.importorskip(
    "ferric",
    reason="the compiled ferric extension is not importable; build it with "
    "`cargo build --release -p ferric-python` and make sure the .so is on the "
    "site-packages path (see CLAUDE.md).",
)

REPO = Path(__file__).resolve().parents[3]
WATER = REPO / "testdata" / "molecules" / "water.xyz"

# Pruned-minus-flat total energy, MEASURED with PySCF on this water geometry,
# (75, 110) Treutler/Becke grid, with ferric's snapped region table
# [26, 110, 110, 110, 110] supplied as a custom `grids.prune` (conv_tol 1e-11,
# RI-JK def2-universal-jkfit): PBE/STO-3G 1.1e-10, PBE/6-31G 2.3e-10,
# B3LYP/STO-3G 2.2e-10, B3LYP/6-31G 4.9e-10 Ha. A BROKEN table measured
# 7e-6 Ha on H2O E_xc and 4.7e-5 Ha on CH4/PBE (outermost region snapped down,
# see ferric-dft/src/prune.rs). 1e-7 sits ~200x above the measured delta and
# ~70x below the smallest broken-table value.
PRUNE_TOL = 1e-7

SCF = dict(energy_conv=1e-10, density_conv=1e-8)


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz(str(WATER))


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def test_explicit_default_grid_is_byte_identical_to_no_kwargs(water, sto3g):
    """Exactness anchor. Fails if the explicit-config path differs from the
    implicit default in anything (sizes, prune, or a grid rebuilt differently)."""
    e_none = ferric.run_dft(water, sto3g, "PBE", **SCF).total_energy
    e_explicit = ferric.run_dft(
        water, sto3g, "PBE", grid_radial=75, grid_angular=110, grid_prune="none", **SCF
    ).total_energy
    assert e_none == e_explicit, f"{e_none!r} != {e_explicit!r}"


def test_pruned_grid_has_fewer_points(water):
    """Measured 23.4% fewer on H2O at 75x110 (ferric-dft/tests/grid_prune.rs).
    Fails if `grid_prune` is parsed but never reaches the grid builder."""
    flat = ferric.dft_grid_point_count(water)
    assert flat == ferric.dft_grid_point_count(water, grid_prune="none")
    assert flat == 3 * 75 * 110  # no weight screening between build and use
    pruned = ferric.dft_grid_point_count(water, grid_prune="nwchem")
    saved = 1.0 - pruned / flat
    assert saved > 0.15, f"pruning removed only {100 * saved:.1f}% of {flat} points"


@pytest.mark.parametrize("run", ["run_dft", "run_ksdft"])
def test_pruned_energy_is_close_but_really_pruned(water, sto3g, run):
    """Fails on the `!=` if run_dft drops the kwarg (the SCF is deterministic,
    so the same grid gives the same bits), and on the tolerance if the pruned
    table is wrong."""
    fn = getattr(ferric, run)
    e_flat = fn(water, sto3g, "PBE", **SCF).total_energy
    e_pruned = fn(water, sto3g, "PBE", grid_prune="nwchem", **SCF).total_energy
    d = abs(e_pruned - e_flat)
    assert e_pruned != e_flat, "grid_prune='nwchem' did not change the SCF grid"
    assert d < PRUNE_TOL, f"pruned vs flat |dE| = {d:.3e} Ha > {PRUNE_TOL:.0e}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_prune": "sg1"},  # unknown scheme
        {"grid_prune": ""},  # empty is not "none"
        {"grid_angular": 100},  # not a Lebedev order (lebedev() would panic)
        {"grid_radial": 0},
        {"grid_angular": 50, "grid_prune": "nwchem"},  # no pruned table at 50
    ],
)
def test_bad_grid_kwargs_raise_value_error(water, sto3g, kwargs):
    with pytest.raises(ValueError):
        ferric.run_dft(water, sto3g, "PBE", **kwargs)
    with pytest.raises(ValueError):
        ferric.dft_grid_point_count(water, **kwargs)


def test_grid_kwargs_refuse_a_gradient(water, sto3g):
    """The analytic KS gradient is built on the default grid; pairing it with a
    different-grid energy would describe two surfaces."""
    with pytest.raises(ValueError, match="with_gradient"):
        ferric.run_dft(water, sto3g, "PBE", with_gradient=True, grid_prune="nwchem")
