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
    TierSpec(Tier.QUANTUM, "ferric DFT + dispersion", 600.0, "1-10",
             "final energetics on the handful that survive",
             validated_by=None),   # NOT YET USED IN THIS CAMPAIGN
)
