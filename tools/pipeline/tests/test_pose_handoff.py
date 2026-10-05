"""The pose the docking tier picked must be the pose tiers 3 and 4 score.

`funnel._harvest_geometry` carries each stage's geometry into
`context["geometry"]`, which `tiers._embedded` reads. The production funnel
(experiments/danuglipron/run_isomer_pipeline.py) is

    dock -> mmff (tier2_forcefield) -> gfn2 -> dft

and `tier2_forcefield` re-embeds from SMILES in free solution and returns THAT
geometry in its payload. If harvesting lets a later stage replace an earlier
one, tier 2 silently swaps the docked pose for a conformer that never saw the
receptor -- the defect `_harvest_geometry` was written to close, reopened by
the stage that sits between the producer and the consumers.

`test_funnel.py::test_a_produced_geometry_reaches_the_next_tier` covers the
adjacent case (dock -> consumer). This file covers the production ladder.

Salvaged from the unmerged fix/funnel-pose-handoff branch (its D1 tests),
rewritten against main's harvest-into-context design.
"""

from __future__ import annotations

from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline import tiers as tiers_mod
from tools.pipeline.funnel import Stage, run_funnel
from tools.pipeline.tiers import TierResult

P = "OC(=O)c1ccccc1"
CANDS = [
    Isomer(s, "substitutional", "t", P) for s in ["OC(=O)c1ccccc1", "OC(=O)c1ccc(F)cc1"]
]

# The sentinel is the candidate's OWN complete geometry (so `_embedded`'s
# formula guard accepts it), translated far from anywhere an embedding would
# place it. Distinguishable without being a different molecule.
SENTINEL_OFFSET = 500.0


def _is_sentinel(coords) -> bool:
    return all(c > SENTINEL_OFFSET / 2 for xyz in coords for c in xyz)


def _dock_stand_in(iso, ctx):
    """Stands in for tier 1: a successful result carrying a recognisable pose.

    Module level so a fan-out stage could pickle it.
    """
    r = tiers_mod.tier2_forcefield(iso, {"seed": 11})
    assert r.ok, r.error
    coords = [
        (x + SENTINEL_OFFSET, y + SENTINEL_OFFSET, z + SENTINEL_OFFSET)
        for x, y, z in r.payload["coords"]
    ]
    return TierResult(
        iso.canonical, -5.0, payload={"symbols": r.payload["symbols"], "coords": coords}
    )


def _geometry_echo(iso, ctx):
    """Stands in for tier 3/4: reports the geometry `_embedded` hands it."""
    syms, coords = tiers_mod._embedded(iso, ctx)
    assert syms is not None, coords
    return TierResult(iso.canonical, -1.0, payload={"symbols": syms, "coords": coords})


def test_the_docked_pose_survives_an_intervening_force_field_stage():
    """THE production ladder: dock -> real tier 2 -> quantum tier."""
    rep = run_funnel(
        CANDS,
        [
            Stage(Tier.SEARCH, _dock_stand_in, keep=2, name="dock"),
            Stage(Tier.FORCE_FIELD, tiers_mod.tier2_forcefield, keep=2, name="mmff"),
            Stage(Tier.SEMIEMPIRICAL, _geometry_echo, keep=2, name="gfn2"),
        ],
        {},
    )
    assert len(rep.results["gfn2"]) == len(CANDS), "the quantum stand-in never ran"
    for r in rep.results["gfn2"]:
        assert r.ok, r.error
        assert _is_sentinel(r.payload["coords"]), (
            f"{r.candidate_id}: the quantum tier scored tier 2's free-solution "
            f"MMFF re-embedding, not the docked pose -- the force-field stage "
            f"overwrote context['geometry']"
        )


def test_without_a_docking_stage_the_quantum_tier_scores_tier2s_geometry():
    """Reachability and the trivial limit: with no upstream pose, the quantum
    tier must still get tier 2's geometry, and it must NOT be a sentinel --
    otherwise the test above could pass with `_is_sentinel` always True."""
    ctx: dict = {}
    rep = run_funnel(
        CANDS,
        [
            Stage(Tier.FORCE_FIELD, tiers_mod.tier2_forcefield, keep=2, name="mmff"),
            Stage(Tier.SEMIEMPIRICAL, _geometry_echo, keep=2, name="gfn2"),
        ],
        ctx,
    )
    by_id = {r.candidate_id: r for r in rep.results["mmff"]}
    for r in rep.results["gfn2"]:
        assert r.ok, r.error
        assert not _is_sentinel(r.payload["coords"])
        assert r.payload["coords"] == by_id[r.candidate_id].payload["coords"]
