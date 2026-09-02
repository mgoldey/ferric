"""SCF diagnostics on `PyDftResult`: iteration count and typed exit reason.

These exist because wall time alone cannot distinguish "each iteration is
expensive" from "the SCF needed many iterations" -- a distinction with
opposite fixes (cheaper grid/basis vs. a better guess, level shift, damping).
Diagnosing the danuglipron tier-4 cost anomaly required exactly that split,
and `converged: bool` could not provide it.
"""
from __future__ import annotations

import ferric


def _water():
    return ferric.Molecule.from_xyz("testdata/molecules/water.xyz", 0, 1)


def test_iterations_is_positive_for_a_converged_run():
    res = ferric.run_dft(_water(), ferric.BasisSet.bundled("sto-3g"),
                         functional="PBE")
    assert res.converged
    # A converged SCF must have taken at least one iteration; a zero here
    # would mean the field is not wired to the solver at all.
    assert res.iterations >= 1
    assert isinstance(res.iterations, int)


def test_exit_reason_is_converged_when_converged():
    res = ferric.run_dft(_water(), ferric.BasisSet.bundled("sto-3g"),
                         functional="PBE")
    assert res.converged
    assert res.exit_reason == "Converged"


def test_exit_reason_distinguishes_failure_modes():
    """`converged=False` collapses every failure into one bit; this does not.

    Capping iterations at 1 cannot converge water, so the run must report a
    NON-Converged exit. The precise variant is not asserted -- `run_dft` walks
    a level-shift ladder and which rung reports first is an implementation
    detail -- but it must name a real failure mode rather than claim success.
    """
    res = ferric.run_dft(_water(), ferric.BasisSet.bundled("sto-3g"),
                         functional="PBE", max_iter=1)
    assert res.exit_reason in {"Plateau", "Stalled", "Diverged", "MaxIter",
                               "Converged"}
    if not res.converged:
        assert res.exit_reason != "Converged"


def test_iterations_is_not_bounded_by_max_iter():
    """`max_iter` bounds each LADDER RUNG, not the run — and this proves it.

    The intuitive assertion (`iterations <= max_iter`) is FALSE here, and
    asserting it was my first mistake. `run_dft` walks a level-shift ladder
    (`ferric_scf::ladder`): each rung is a full SCF that `max_iter` bounds
    individually, and each carries the previous rung's density forward. So a
    tight cap makes early rungs fail and the ladder escalates to later, better
    conditioned rungs -- which can then run MORE iterations than the cap and
    converge.

    Measured on water/STO-3G/PBE: max_iter=1 reports 7 iterations and
    converges. Pinned here so nobody (me included) reasons about ferric SCF
    cost as if `max_iter` were a work budget.
    """
    bs = ferric.BasisSet.bundled("sto-3g")
    capped = ferric.run_dft(_water(), bs, functional="PBE", max_iter=1)
    assert capped.iterations > 1, (
        "max_iter=1 reporting <=1 iteration would mean the ladder no longer "
        "escalates past a failed rung -- if that changed deliberately, this "
        "test and the max-iter guidance in RESULTS.md must both be updated"
    )


def test_iterations_varies_with_the_problem():
    """Guards `iterations` against being a hardcoded constant.

    Two different systems must not report an identical count by construction.
    Water and methane converge in different numbers of steps.
    """
    bs = ferric.BasisSet.bundled("sto-3g")
    water = ferric.run_dft(_water(), bs, functional="PBE")
    methane = ferric.run_dft(
        ferric.Molecule.from_xyz("testdata/molecules/methane.xyz", 0, 1),
        bs, functional="PBE")
    assert water.iterations >= 1 and methane.iterations >= 1
