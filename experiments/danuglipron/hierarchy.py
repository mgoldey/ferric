"""Danuglipron's cost hierarchy: the measured cost of each tier on THIS system.

The generic types and rules live in `tools/campaign/hierarchy.py`. The numbers
here are campaign-specific -- measured on a 70-atom anion in a 6458-charge
pocket on this machine, 2026-08-29 -- and are not transferable as-is.
"""
from __future__ import annotations

from tools.campaign.hierarchy import Tier, TierSpec

# Costs MEASURED on danuglipron (70-atom anion, 6458-charge pocket), 2026-08-29,
# rather than quoted from literature -- they are what this box actually does.
DANUGLIPRON_HIERARCHY: tuple[TierSpec, ...] = (
    TierSpec(Tier.SEARCH, "AutoDock Vina 1.2.7", 1e-5, "10^5-10^6",
             "search 6 rigid-body DOF + 9 torsions",
             validated_by="redocked 7LCJ to 0.95 A (top-ranked pose), 20/20 "
                          "poses within 5 A of the site"),
    TierSpec(Tier.FORCE_FIELD, "MMFF94 (RDKit)", 1e-3, "10^2-10^3",
             "relax embedded geometry, drop clashes",
             validated_by="GFN2 moves an MMFF geometry 12-14 kcal/mol and "
                          "0.13-0.39 A, so MMFF is adequate for declashing "
                          "and NOT for ranking"),
    TierSpec(Tier.SEMIEMPIRICAL, "GFN2-xTB", 0.5, "10-10^2",
             "rank survivors; polarization in a point-charge field",
             validated_by="anion-vs-neutral separation of 143 kcal/mol at a "
                          "fixed geometry, i.e. it resolves formal charge"),
    # MEASURED 2026-09-02 on a verified-quiet box: 612.4 s (10.2 min),
    # 18 iterations, CONVERGED, for the 71-atom neutral acid at STO-3G/PBE.
    #
    # This SUPERSEDES the 2026-08-30 figure of ">57 min, did not finish", which
    # was NOT a measurement of DFT cost. That run auto-resolved a 7.26 GB
    # memory budget from live MemAvailable while needing ~9.5-9.9 GB, and spent
    # its time paging until the kernel OOM-killed it. It measured contention.
    # Pin the budget (tier4_dft's mem_budget_gb) and give it headroom.
    #
    # Cost driver, for estimating a new candidate: vxc.rs assembles V_xc with
    # an O(nbf^2 x npts) GEMM EVERY iteration, and ferric's grid is 75x110 per
    # ATOM, so nbf and atom count both matter and the basis alone does not
    # predict cost. See tools/pipeline/cost.py. The iteration count is a
    # SEPARATE factor and is not transferable: alkanes converge in 10, this
    # molecule in 18.
    #
    # validated_by is now set: the tier produced a converged number at this
    # system's real size, which is what it had never done before.
    TierSpec(Tier.QUANTUM, "ferric DFT + dispersion", 612.4, "1-10",
             "final energetics on the handful that survive",
             validated_by="612.4 s / 18 iterations / converged on the 71-atom "
                          "neutral acid at STO-3G/PBE, on a box verified free "
                          "of memory pressure for the whole run"),
)
