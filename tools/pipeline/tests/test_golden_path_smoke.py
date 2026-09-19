"""The golden path, composed from REAL tiers — the test nothing else does.

`test_funnel.py` verifies the funnel's plumbing with stub tiers: narrowing,
ranking direction, failure handling. That is the right shape for those
questions, and it means **nothing in the suite has ever run the real tiers
together.** Each stage is exercised alone, and the composition only by hand.

The composition is where the interesting failures live, because they happen
BETWEEN stages rather than inside one:

* a tier reads `context["geometry"]` that no earlier tier wrote — the golden
  path's headline defect (#93)
* a tier passes Angstrom where the next expects Bohr — a silent 1.89x error,
  not a crash
* a docked pose is UNITED-ATOM and the next tier builds a QM molecule from it.
  MEASURED (M14): 263 electrons against the real molecule's 292, because PDBQT
  merges nonpolar hydrogens. `pose_fit` accepted it silently;
  `embed_ligand_from_coords` refused.

## Cost, and why tiers 3-4 are absent

Tier 1 DOCKS (~30 s/ligand at `exhaustiveness=4`), so this is not fast-tier
work. It runs two tiny candidates so the bound is ~a minute rather than the
hours a real funnel takes.

Tiers 3-4 are deliberately excluded: GFN2 needs the xtb binary and DFT is
~600 s/candidate. Both have their own tests, and adding them here would make
this unrunnable without buying any more coverage of the composition question.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.isomers.model import Isomer
from tools.pipeline import Stage, run_funnel
from tools.campaign.hierarchy import Tier
from tools.pipeline.tiers import tier1_dock, tier2_forcefield

REPO = Path(__file__).resolve().parents[3]
POCKET_PDB = REPO / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"

# Two small, unambiguous molecules. The chemistry is irrelevant -- what is under
# test is that two REAL tiers hand off to each other -- so these are chosen to
# embed and dock fast rather than to mean anything.
CANDIDATES = [
    Isomer("CCO", "ethanol", "none", "CCO"),
    Isomer("CC(=O)O", "acetic-acid", "none", "CC(=O)O"),
]


def _docking_available() -> bool:
    try:
        import meeko  # noqa: F401
        import vina  # noqa: F401
    except ImportError:
        return False
    return True


def test_real_force_field_tier_composes_through_the_funnel():
    """Tier 2 with the REAL implementation, not a stub.

    Cheap enough for the fast tier (MMFF is ~1 ms/pose), and it already covers
    the handoff the stub tests cannot: a real `TierResult` flowing through
    `run_funnel`'s ranking, narrowing and bookkeeping.

    Asserts the composition, not the chemistry.
    """
    stages = [Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=2, name="ff")]
    rep = run_funnel(list(CANDIDATES), stages, {"seed": 0xF00D})

    assert rep.survivors, "the force-field tier dropped every candidate"
    assert len(rep.survivors) <= 2

    # Every survivor must carry a real number. `None` would mean the funnel
    # promoted a failure, and 0.0 in an ascending energy ranking is the best
    # possible score -- both are the failure mode TierResult exists to prevent.
    for iso in rep.survivors:
        v = rep.value("ff", iso.canonical)
        assert v is not None, f"{iso.canonical} survived with no value"
        assert isinstance(v, float)

    # Per-tier timing is recorded. A funnel that cannot say which tier cost the
    # run cannot be tuned, and the answer is routinely not what the cost table
    # predicts (M11: tier 1, not tier 4).
    assert rep.outcomes, "run_funnel must record a TierOutcome per stage"
    assert all(o.seconds >= 0.0 for o in rep.outcomes)


@pytest.mark.skipif(
    not _docking_available(), reason="needs the 'docking' extra (vina + meeko)"
)
@pytest.mark.skipif(not POCKET_PDB.is_file(), reason=f"no pocket at {POCKET_PDB}")
def test_tier1_dock_returns_a_number_or_an_explained_failure(tmp_path):
    """Tier 1 end to end from SMILES against a real receptor.

    SLOW (~30 s): it actually docks. Kept because the contract it pins is the
    one the whole funnel rests on -- a tier returns a NUMBER or an EXPLAINED
    FAILURE, never both, and never 0.0 as a stand-in for either.

    0.0 matters specifically: the funnel ranks ASCENDING, so a placeholder zero
    is the best possible score and would carry a broken candidate to the top.
    """
    from tools.active_site.pocket_charges import derive_pocket_charges
    from tools.docking.vina_dock import prepare_receptor

    receptor = tmp_path / "receptor.pdbqt"
    try:
        prepare_receptor(POCKET_PDB, receptor)
    except Exception as exc:  # noqa: BLE001 -- environment, not a code defect
        pytest.skip(f"receptor preparation unavailable: {type(exc).__name__}: {exc}")

    # Box centre from the pocket's OWN charges rather than a hardcoded triple,
    # so the box cannot silently drift from the structure it should cover.
    # PointCharge is (q, x, y, z) with coordinates in BOHR; the docking box is
    # in ANGSTROM -- mixing them is the 1.89x error this module's docs warn
    # about, so the conversion is explicit.
    bohr_to_angstrom = 0.529177210903
    qs = derive_pocket_charges(str(POCKET_PDB)).charges
    if not qs:
        pytest.skip("no pocket charges derived")
    centre = tuple(
        sum(c[i] for c in qs) / len(qs) * bohr_to_angstrom for i in (1, 2, 3)
    )

    context = {
        "seed": 0xF00D,
        "receptor_pdbqt": str(receptor),
        "box_center": centre,
        # M11: ex=4 matches ex=32's accuracy at a quarter the cost, and 32 had
        # the WORST mean of the four levels tried. Do not raise this.
        "exhaustiveness": 4,
        "n_seeds": 1,
    }
    res = tier1_dock(CANDIDATES[0], context)

    if res.error is not None:
        assert res.value is None, (
            f"tier1_dock returned BOTH an error ({res.error!r}) and a value "
            f"({res.value}) -- a failed tier must not also report a score"
        )
        assert res.error.strip(), "a failure must say why"
        return

    assert res.value is not None
    assert res.value != 0.0, (
        "0.0 is the best possible score in an ascending ranking; a real Vina "
        "score is negative, and a placeholder zero would top the funnel"
    )
    assert res.value < 0.0, f"a Vina score should be negative, got {res.value}"


def test_a_rejected_candidate_carries_no_value_at_all():
    """A tier that REFUSES a candidate must return `None`, never a number.

    Exercises the rejection branches directly, because the happy-path test
    above cannot: a valid SMILES never reaches them. MEASURED by mutation --
    changing `TierResult(iso.canonical, None, "unparseable SMILES")` to return
    `0.0` instead left the docking test PASSING, so that contract had no
    coverage at all.

    0.0 is the specific hazard: `run_funnel` ranks ASCENDING, so a rejected
    candidate scored 0.0 sorts ABOVE every real (negative) Vina score and is
    carried to the top of the funnel by the failure that should have dropped it.
    """
    # NOT tested here: tier1_dock's unparseable-SMILES branch. `Isomer.__init__`
    # already raises on one (`tools/isomers/model.py:39`), so that branch is
    # defence in depth and unreachable through the funnel -- constructing the
    # input to reach it fails first. Left in the code; asserting on it would
    # mean asserting on a state the pipeline cannot produce.
    #
    # Multi-fragment IS reachable: a transform can sever a ring rather than
    # shrinking it, and the resulting salt/fragment pair parses fine while
    # being undockable.
    frags = Isomer("CCO.CCO", "two-fragments", "none", "CCO.CCO")
    res = tier1_dock(frags, {"seed": 0xF00D})
    assert res.error is not None, "a multi-fragment input must be rejected"
    assert res.value is None, (
        f"a rejected candidate came back with value={res.value}; in an ascending "
        "ranking any number -- 0.0 especially -- promotes it over real scores"
    )
    assert not res.ok
    assert "fragment" in res.error.lower(), (
        f"the failure should name the reason, got {res.error!r}"
    )


def test_the_full_funnel_reaches_tier_4_and_produces_a_survivor():
    """FF -> xtb -> DFT, end to end, with a real energy at the end.

    M10 ran the four-tier stack on danuglipron and tier 4 returned **0 of 5**.
    Two bugs, both since fixed: the driver declared `net_charge=-1` on NEUTRAL
    structures (asking for an electron that does not exist), and ring
    contractions that SEVER a ring produced undockable fragment pairs.

    Both have unit tests. What did NOT exist until now is a run of the whole
    stack SINCE the fixes -- so "tier 4 works" rested on per-bug coverage
    rather than on a completed funnel. This is that run, on molecules small
    enough (~1.7 s total) to belong in the fast tier.

    Asserts the COMPOSITION reaches the last tier with something in hand:

        FORCE_FIELD    3 -> 3   failed 0
        SEMIEMPIRICAL  3 -> 2   failed 0
        QUANTUM        2 -> 1   failed 0     <- the step that used to be 0

    The chemistry is irrelevant; a survivor carrying a real DFT energy is the
    claim.
    """
    from tools.campaign.hierarchy import Tier
    from tools.pipeline.tiers import tier3_gfn2, tier4_dft

    try:
        from tools.campaign.xtb_engine import verify_xtb_build
    except ImportError:  # pragma: no cover -- environment
        pytest.skip("xtb engine unavailable")
    ok, err = verify_xtb_build()
    if not ok:
        pytest.skip(f"xtb unusable: {err}")

    cands = [Isomer(s, s, "none", s) for s in ("CCO", "CC(=O)O", "CCN")]
    stages = [
        Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=3, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=2, name="xtb"),
        Stage(Tier.QUANTUM, tier4_dft, keep=1, name="dft"),
    ]
    rep = run_funnel(cands, stages, {"seed": 0xF00D, "basis": "sto-3g"})

    by_tier = {o.tier.name: o for o in rep.outcomes}
    assert "QUANTUM" in by_tier, (
        "the funnel never reached tier 4; it stopped at "
        f"{[o.tier.name for o in rep.outcomes]}"
    )
    q = by_tier["QUANTUM"]
    assert q.n_in > 0, "tier 4 received nothing -- an earlier tier emptied the funnel"
    assert q.n_out > 0, (
        f"tier 4 produced NO survivors ({q.n_failed} failed) -- this is the exact "
        "M10 failure mode, and it is what this test exists to catch"
    )

    assert rep.survivors, "the funnel produced no survivors"
    for iso in rep.survivors:
        e = rep.value("dft", iso.canonical)
        assert e is not None, f"{iso.canonical} survived tier 4 with no energy"
        assert e < 0.0, f"an electronic energy must be negative, got {e}"
