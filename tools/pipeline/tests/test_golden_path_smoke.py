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

import os
import subprocess
import sys
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


def test_the_quickstart_block_actually_RUNS(tmp_path):
    """Extract section 0b and execute it. Not compile it -- run it.

    Two guards already cover this block and both stop short by design:
    `test_the_quickstart_binds_every_name_it_uses` parses it and checks every
    name is bound, and `test_the_quickstart_names_functions_that_return_what
    _it_claims` checks one specific return type. Each documents why it does
    not execute -- a DFT call does not belong in the fast tier.

    But "it compiles" is not "it runs", and the two defects those guards exist
    for were both found by RUNNING the block: `read_structure` vs `read`
    (both names exist, the wrong one reads perfectly) and a
    `relative_descriptors` value printed as exactly zero when it is -1.42e-14.

    SLOW (~20 s): it runs two real SCFs. The one edit made here is the
    PLACEHOLDER path the block itself flags -- "a PLACEHOLDER path, substitute
    your own file" -- pointed at a generated fixture. Everything else executes
    verbatim, so a rename or a changed return type in any function the
    quickstart names fails here rather than in a user's paste.
    """
    import re

    pytest.importorskip("rdkit")
    pytest.importorskip("ferric")

    golden = (
        Path(__file__).resolve().parents[3]
        / "site/src/reference/pipeline-golden-path.md"
    )
    if not golden.is_file():  # pragma: no cover - source checkout only
        pytest.skip(f"no {golden}")

    text = golden.read_text()
    start = text.index("## 0b.")
    block = text[start : text.index("\n## ", start + 5)]
    code = "\n".join(re.findall(r"```python\n(.*?)```", block, re.S))
    assert code.strip(), "section 0b has no python blocks; the guard needs re-deriving"

    # The placeholder the block tells you to replace.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    fixture = tmp_path / "ligand_with_hydrogens.pdb"
    Chem.MolToPDBFile(m, str(fixture))
    assert "ligand_with_hydrogens.pdb" in code, (
        "the quickstart no longer names the placeholder this test substitutes; "
        "re-derive the substitution rather than deleting the test"
    )
    # repr(), not an f-string. A path interpolated raw becomes part of a
    # Python string LITERAL in the generated child script, so a backslash in
    # it is an escape: a Windows `...\Users\...` yields `\U`, which is a
    # syntax error, and `\t`/`\n` would corrupt the path silently. repr()
    # produces a valid literal for any path. (Linux-only CI here, so this is
    # a correctness-by-construction fix rather than an observed failure.)
    code = code.replace('"ligand_with_hydrogens.pdb"', repr(str(fixture)))

    # A SUBPROCESS, not `exec`. Two reasons, and bandit flagging B102 is the
    # lesser one: running the block in-process would leak its imports, its
    # matplotlib state and its working directory into the rest of the session,
    # and a `sys.exit` or a stray global in the doc would take the test run
    # with it. A child process also gives the real "paste it into a fresh
    # interpreter" semantics the quickstart promises.
    repo = Path(__file__).resolve().parents[3]
    script = tmp_path / "quickstart_block.py"
    script.write_text(code)
    env = {**os.environ, "PYTHONPATH": str(repo), "OPENBLAS_NUM_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "the quickstart block does not RUN. This is what a reader gets when "
        f"they paste it:\n--- stderr ---\n{proc.stderr[-2500:]}"
    )


@pytest.mark.skipif(
    not _docking_available(), reason="needs the 'docking' extra (vina + meeko)"
)
@pytest.mark.skipif(not POCKET_PDB.is_file(), reason=f"no pocket at {POCKET_PDB}")
def test_ALL_FOUR_tiers_compose_from_substitutions_to_a_survivor(tmp_path):
    """The whole golden path in one run: substitutions -> dock -> FF -> xtb -> DFT.

    `test_the_full_funnel_reaches_tier_4_and_produces_a_survivor` starts AFTER
    docking, because tier 1 needs a prepared receptor. That left the first hop
    -- the one that turns a SMILES list into poses, and the one that costs the
    most -- outside any composition test. Nothing exercised all four together.

    SLOW (~45 s): it docks 10 candidates and runs two real SCFs.

    MEASURED 2026-09-20 on this stack, benzoic acid + {F, Cl, Me} against 7LCJ,
    keeping 6/4/2/1:

        STO-3G     45.0 s   dock 72.7%  FF 0.1%  xtb 0.3%  DFT 27.0%
        def2-svp   82.1 s   dock 39.8%  FF 0.1%  xtb 0.1%  DFT 60.0%

    The test asserts COMPOSITION, not those timings: a survivor comes out, no
    tier fails, and every stage passes something to the next. Timings vary with
    the box; what must not vary is that the four tiers still fit together.
    """
    import numpy as np

    from tools.active_site.pocket_charges import derive_pocket_charges
    from tools.campaign.hierarchy import Tier
    from tools.docking.vina_dock import prepare_receptor
    from tools.isomers.model import Isomer
    from tools.pipeline import Stage, run_funnel
    from tools.pipeline.substitution import propose_substitutions
    from tools.pipeline.tiers import (
        tier1_dock,
        tier2_forcefield,
        tier3_gfn2,
        tier4_dft,
    )

    receptor = tmp_path / "receptor.pdbqt"
    try:
        prepare_receptor(POCKET_PDB, receptor)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"receptor prep unavailable: {exc}")

    pocket = derive_pocket_charges(str(POCKET_PDB))
    bohr = 0.52917721092
    centre = tuple(
        float(v)
        for v in np.array(
            [[q[1] * bohr, q[2] * bohr, q[3] * bohr] for q in pocket.charges]
        ).mean(axis=0)
    )

    parent = "c1ccccc1C(=O)O"
    props = propose_substitutions(parent, {"F": "F", "Cl": "Cl", "Me": "C"})
    cands = [
        Isomer(
            smiles=p.smiles,
            kind="substitutional",
            transform="sub",
            parent_smiles=parent,
        )
        for p in props
    ]
    assert len(cands) >= 4, f"expected several candidates, got {len(cands)}"

    ctx = {
        "receptor_pdbqt": str(receptor),
        "box_center": centre,
        "basis": "sto-3g",  # the FAST basis: this is a composition test
    }
    rep = run_funnel(
        cands,
        [
            Stage(Tier.SEARCH, tier1_dock, keep=4, name="dock"),
            Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=3, name="ff"),
            Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=2, name="xtb"),
            Stage(Tier.QUANTUM, tier4_dft, keep=1, name="dft"),
        ],
        ctx,
    )

    assert len(rep.outcomes) == 4, (
        f"all four stages must RUN; the funnel stops early on an empty "
        f"population, so fewer outcomes means a tier emptied it: "
        f"{[(o.note, o.n_in, o.n_out) for o in rep.outcomes]}"
    )
    for o in rep.outcomes:
        assert o.n_out > 0, f"{o.note}: passed nothing to the next tier"
        assert o.n_failed == 0, f"{o.note}: {o.n_failed} failed -- {o.errors}"
    assert len(rep.survivors) == 1, f"expected one survivor, got {rep.survivors}"

    # The last tier must have produced a real energy, not merely survived.
    dft = rep.results["dft"][0]
    assert dft.ok and dft.value is not None
    assert dft.value < -100.0, (
        f"a benzoic-acid-sized DFT total energy should be well below -100 Ha, "
        f"got {dft.value} -- a placeholder would pass a bare `is not None`"
    )


@pytest.mark.skipif(
    not _docking_available(), reason="needs the 'docking' extra (vina + meeko)"
)
@pytest.mark.skipif(not POCKET_PDB.is_file(), reason=f"no pocket at {POCKET_PDB}")
def test_the_harvested_pose_carries_ITS_HYDROGENS(tmp_path):
    """A PDBQT pose is UNITED-ATOM. The harvested geometry must not be.

    Vina merges nonpolar hydrogens into their carbons, so `DockedPose.symbols`
    is not the molecule that was docked. MEASURED on aspirin: 14 atoms out
    where 21 went in. `funnel._harvest_geometry` puts that straight into
    `context["geometry"]`, so tiers 3 and 4 score the stripped fragment.

    The severity is uneven, which is why this went unnoticed:

        tier 4  FAILS -- an odd electron count trips ferric's
                charge/multiplicity parity check. Protected by accident.
        tier 3  DOES NOT. GFN2 has no such check and returned -35.492226
                against -39.621219 for the real molecule -- both plausible
                GFN2 numbers, neither an error, 2591 kcal/mol apart.

    `restore_hydrogens` already existed for exactly this and had NO production
    caller. This pins the wiring.
    """
    from rdkit import Chem

    from tools.active_site.pocket_charges import derive_pocket_charges
    from tools.docking.vina_dock import prepare_receptor
    from tools.isomers.model import Isomer
    from tools.pipeline.tiers import tier1_dock

    import numpy as np

    receptor = tmp_path / "receptor.pdbqt"
    try:
        prepare_receptor(POCKET_PDB, receptor)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"receptor prep unavailable: {exc}")

    pocket = derive_pocket_charges(str(POCKET_PDB))
    bohr = 0.52917721092
    centre = tuple(
        float(v)
        for v in np.array(
            [[q[1] * bohr, q[2] * bohr, q[3] * bohr] for q in pocket.charges]
        ).mean(axis=0)
    )

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin: 21 atoms with H, 13 heavy
    iso = Isomer(smiles=smiles, kind="parent", transform="t", parent_smiles=smiles)
    res = tier1_dock(iso, {"receptor_pdbqt": str(receptor), "box_center": centre})
    assert res.ok, f"docking failed: {res.error}"

    expected = Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()
    got = len(res.payload["symbols"])
    assert got == expected, (
        f"the harvested pose has {got} atoms but the molecule has {expected} -- "
        f"a united-atom pose reached context['geometry'], and tier 3 will score "
        f"it without complaint"
    )
    assert len(res.payload["coords"]) == expected

    # The hydrogen COUNT specifically, since that is what PDBQT drops.
    n_h = sum(1 for s in res.payload["symbols"] if s == "H")
    n_h_expected = sum(
        1
        for a in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms()
        if a.GetSymbol() == "H"
    )
    assert n_h == n_h_expected, (
        f"{n_h} hydrogens in the harvested pose, {n_h_expected} in the molecule"
    )
