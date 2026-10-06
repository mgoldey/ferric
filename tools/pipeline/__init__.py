"""Ordered tier execution: the funnel that turns many candidates into few.

Generation lives in `tools/isomers`; the individual methods live in
`tools/docking`, `tools/morph` and `tools/campaign`. This package only sequences
them and records what happened at each step.
"""

from .funnel import (
    FunnelReport,
    IncomparableError,
    Stage,
    paired_delta,
    require_same_formula,
    run_funnel,
)
from .tiers import TierResult, tier1_dock, tier2_forcefield, tier3_gfn2, tier4_dft

__all__ = [
    "FunnelReport",
    "IncomparableError",
    "paired_delta",
    "require_same_formula",
    "Stage",
    "run_funnel",
    "TierResult",
    "tier1_dock",
    "tier2_forcefield",
    "tier3_gfn2",
    "tier4_dft",
]
