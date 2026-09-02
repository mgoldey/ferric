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


def test_xc_fock_work_predicts_measured_alkane_runtimes():
    """The cost model is calibrated against real runs, not asserted.

    STO-3G/PBE, 10 SCF iterations each, quiet box, budget pinned. Predicting
    from alkane_5 alone must land within 15% at both larger sizes -- a span of
    54x in cost. If this drifts, the O(nbf^2 x npts) attribution is wrong.
    """
    a5 = DftSize(n_atoms=17, n_basis_functions=37)
    a10 = DftSize(n_atoms=32, n_basis_functions=72)
    a20 = DftSize(n_atoms=62, n_basis_functions=142)

    assert abs(a10.predicted_seconds(a5, 2.5) - 19.6) / 19.6 < 0.15
    assert abs(a20.predicted_seconds(a5, 2.5) - 130.2) / 130.2 < 0.15


def test_atom_count_alone_mispredicts_badly():
    """Guards against reverting to the axis that produced the wrong story.

    Scaling alkane_5 -> alkane_20 by ATOM COUNT (cubed, the naive reading of
    the observed exponent) misses the measured 130.2 s substantially, because
    it ignores that nbf grows too. The nbf^2 x npts model does not.
    """
    a5 = DftSize(n_atoms=17, n_basis_functions=37)
    a20 = DftSize(n_atoms=62, n_basis_functions=142)

    by_atoms = 2.5 * (62 / 17) ** 3
    by_model = a20.predicted_seconds(a5, 2.5)
    assert abs(by_model - 130.2) < abs(by_atoms - 130.2)


def test_xc_fock_work_is_quadratic_in_basis_not_linear():
    """nbf^2, not nbf: doubling the basis quadruples the Fock assembly."""
    small = DftSize(n_atoms=10, n_basis_functions=50)
    big = DftSize(n_atoms=10, n_basis_functions=100)
    assert big.xc_fock_work == 4 * small.xc_fock_work
    assert big.xc_work == 2 * small.xc_work        # the one-pass term is linear


def test_model_is_documented_as_per_iteration_not_total():
    """`predicted_seconds` assumes a comparable ITERATION COUNT, and that
    assumption fails across chemistries.

    Measured 2026-09-02: every alkane converged in exactly 10 iterations,
    danuglipron in 18. Scaling by work alone predicts 6.8 min against an
    actual 10.2 min. This pins the DOCUMENTATION of that limit, because the
    failure mode is quoting a predicted wall time as a promise.
    """
    from tools.pipeline.cost import DftSize

    doc = DftSize.xc_fock_work.__doc__ or ""
    assert "LOWER BOUND" in doc
    assert "PER ITERATION" in doc


def test_correcting_for_iteration_count_recovers_the_measurement():
    """Work-scaling plus the real iteration ratio predicts danuglipron.

    408 s (work-scaled from alkane_20, implicitly 10 iterations) x 18/10
    = 734 s against a measured 612.4 s -- within 20%, erring conservative.
    Scaling by work ALONE is off by 50%, which is the point of the test.
    """
    a20 = DftSize(n_atoms=62, n_basis_functions=142)
    danu = DftSize(n_atoms=71, n_basis_functions=235)

    work_only = danu.predicted_seconds(a20, 130.2)
    assert abs(work_only - 612.4) / 612.4 > 0.3          # work alone: way off

    with_iters = work_only * (18 / 10)
    assert abs(with_iters - 612.4) / 612.4 < 0.25        # with iterations: close


def test_ri_tensor_is_a_first_class_memory_term():
    """The RI (P|mn) tensor is ~a sixth of a drug-scale run's peak.

    Measured on danuglipron: naux=3,635 (counted from the bundled
    def2-universal-jkfit JSON), nbf=235 -> 1.61 GB. An earlier guess of
    naux~700-950 was low by ~4x and left 1.2 GB filed as unexplained, which is
    why this is computed rather than estimated.
    """
    from tools.pipeline.cost import ri_tensor_gb

    assert abs(ri_tensor_gb(235, 3635) - 1.61) < 0.02


def test_aux_basis_dwarfs_the_orbital_basis_at_sto3g():
    """def2-universal-jkfit is sized for LARGE orbital bases.

    At STO-3G it is 15x the orbital basis, so the fitting tensor costs more
    than the accuracy it can deliver. Pinned because the fix (a right-sized
    JK-fitting basis) is worth ~1.6 GB and is easy to forget.
    """
    from tools.pipeline.cost import ri_tensor_gb

    nbf, naux = 235, 3635
    assert naux / nbf > 10
    # The tensor is larger than the entire SCF matrix working set.
    assert ri_tensor_gb(nbf, naux) > 100 * (nbf * nbf * 8 / 1e9)
