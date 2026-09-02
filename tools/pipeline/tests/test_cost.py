"""The DFT cost model, and the counter-intuitive fact it exists to pin."""
from __future__ import annotations

from tools.pipeline.cost import (
    GRID_POINTS_PER_ATOM, DftSize, fits_in_budget,
)

# The two systems the campaign actually compared (RESULTS.md M10 + correction).
ALKANE_SVP = DftSize(n_atoms=32, n_basis_functions=330)
DRUG_STO3G = DftSize(n_atoms=70, n_basis_functions=234)


def test_grid_scales_with_atoms_not_basis():
    """The whole point: basis set does not move the grid at all."""
    small_basis = DftSize(n_atoms=70, n_basis_functions=100)
    large_basis = DftSize(n_atoms=70, n_basis_functions=900)
    assert small_basis.grid_points == large_basis.grid_points
    assert small_basis.grid_points == 70 * GRID_POINTS_PER_ATOM


def test_smaller_basis_can_still_cost_more_memory():
    """Guards the exact reasoning error that produced the wrong M10 claim.

    The drug runs a SMALLER basis than the alkane (234 vs 330 functions) and
    still needs a LARGER resident cache, because it has 2.19x the grid points.
    Anyone tempted to conclude "fewer basis functions, therefore cheaper" is
    contradicted here.
    """
    assert DRUG_STO3G.n_basis_functions < ALKANE_SVP.n_basis_functions
    assert DRUG_STO3G.ao_cache_gb > ALKANE_SVP.ao_cache_gb


def test_measured_cache_sizes_match_the_recorded_table():
    """Pins the numbers quoted in RESULTS.md so prose and code cannot drift."""
    assert ALKANE_SVP.ao_cache_gb == round(ALKANE_SVP.ao_cache_gb, 10)
    assert abs(ALKANE_SVP.ao_cache_gb - 2.79) < 0.01
    assert abs(DRUG_STO3G.ao_cache_gb - 4.32) < 0.01


def test_xc_work_ratio_is_modest_compared_to_the_runtime_blowup():
    """The anomaly, stated honestly: 1.55x the work, >35x the runtime.

    If this ratio is ever found to be large, the 'tier 4 is anomalous'
    conclusion would need revisiting -- the blowup would just be size.
    """
    ratio = DRUG_STO3G.xc_work / ALKANE_SVP.xc_work
    assert 1.5 < ratio < 1.6


def test_budget_gate_detects_the_batching_cliff():
    assert fits_in_budget(DRUG_STO3G, budget_gb=9.6)      # 12G cap -> 0.8x
    assert not fits_in_budget(DRUG_STO3G, budget_gb=2.0)  # would batch


def test_ample_budget_still_fits():
    """An over-estimating guard is also a bug: ample budgets must pass."""
    assert fits_in_budget(DftSize(1, 10), budget_gb=64.0)


def test_atom_count_alone_hides_composition():
    """Two molecules of similar SIZE can differ ~2x in XC work.

    alkane_20 (62 atoms) is hydrogen-padded: 20 C + 42 H = 142 STO-3G
    functions. Danuglipron (70 atoms) is heavy-atom rich: 41 heavy + 29 H =
    234. So a 1.13x atom ratio conceals a 1.86x XC-work ratio. Comparing
    "same atom count" molecules without checking composition is a trap.
    """
    from tools.pipeline.cost import sto3g_basis_functions

    alkane = ["C"] * 20 + ["H"] * 42
    drug = ["C"] * 31 + ["N"] * 5 + ["O"] * 4 + ["F"] + ["H"] * 29

    assert sto3g_basis_functions(alkane) == 142
    assert sto3g_basis_functions(drug) == 234

    a = DftSize(len(alkane), sto3g_basis_functions(alkane))
    d = DftSize(len(drug), sto3g_basis_functions(drug))
    atom_ratio = d.n_atoms / a.n_atoms
    work_ratio = d.xc_work / a.xc_work
    assert abs(atom_ratio - 1.13) < 0.02
    assert abs(work_ratio - 1.86) < 0.02
    assert work_ratio > 1.6 * atom_ratio      # composition dominates
