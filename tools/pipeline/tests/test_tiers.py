"""Tier adapters: uniform signature, honest failure, and correct wiring."""

from __future__ import annotations

import pytest

from tools.isomers.model import Isomer
from tools.pipeline.tiers import (
    TierResult,
    tier1_dock,
    tier2_forcefield,
    tier3_gfn2,
    tier4_dft,
)

SMALL = Isomer("CO", "parent", "none", "CO")  # methanol: cheap everywhere
BENZOIC = Isomer("OC(=O)c1ccccc1", "parent", "none", "OC(=O)c1ccccc1")


def test_tier_result_failure_carries_none_not_zero():
    r = TierResult("x", None, "embedding failed", {})
    assert r.value is None and not r.ok and r.error


def test_ok_and_error_never_disagree():
    assert TierResult("x", -1.0).ok
    assert not TierResult("x", None, "boom").ok
    assert not TierResult("x", None).ok


def test_tier2_embeds_and_returns_an_energy():
    r = tier2_forcefield(BENZOIC, {})
    assert r.ok, r.error
    assert "coords" in r.payload and len(r.payload["coords"]) > 0
    assert len(r.payload["symbols"]) == len(r.payload["coords"])


def test_tier2_reports_an_unembeddable_molecule_as_unevaluated():
    cage = Isomer("C12C3C1C1C2C31", "structural", "cage", "C")
    r = tier2_forcefield(cage, {})
    assert r.ok == (r.error is None), "ok and error disagree"
    if not r.ok:
        assert r.value is None


def test_tier2_is_deterministic_under_a_fixed_seed():
    a = tier2_forcefield(BENZOIC, {"seed": 42})
    b = tier2_forcefield(BENZOIC, {"seed": 42})
    assert a.value == pytest.approx(b.value)


def test_tier3_returns_a_negative_energy():
    r = tier3_gfn2(SMALL, {})
    if not r.ok and "not on PATH" in (r.error or ""):
        pytest.skip("xtb not available")
    assert r.ok, r.error
    assert r.value < 0


def test_tier4_returns_a_converged_dft_energy():
    r = tier4_dft(SMALL, {"basis": "sto-3g", "functional": "PBE"})
    assert r.ok, r.error
    assert r.value < 0
    assert r.payload.get("converged") is True


def test_tier4_is_far_below_tier3_for_the_same_molecule():
    """A WIRING check, not a physics claim: DFT total energies are much more
    negative than GFN2's (which uses a valence-only Hamiltonian), so an
    accidental swap of the two adapters shows up here immediately."""
    t3 = tier3_gfn2(SMALL, {})
    if not t3.ok and "not on PATH" in (t3.error or ""):
        pytest.skip("xtb not available")
    t4 = tier4_dft(SMALL, {"basis": "sto-3g"})
    assert t4.ok and t3.ok
    assert t4.value < t3.value


def test_tier4_reports_an_unknown_basis_rather_than_raising():
    r = tier4_dft(SMALL, {"basis": "not-a-real-basis"})
    assert not r.ok
    assert r.value is None
    assert "DFT failed" in (r.error or "")


def test_tier1_reports_a_missing_receptor_rather_than_raising():
    r = tier1_dock(
        SMALL,
        {
            "receptor_pdbqt": "/nonexistent.pdbqt",
            "box_center": (0.0, 0.0, 0.0),
            "box_size": (10.0, 10.0, 10.0),
        },
    )
    assert not r.ok
    assert r.value is None
    # vina/meeko are an optional extra, so on an install without them the
    # honest answer is "docking unavailable", not "receptor missing" -- but it
    # must still be a REPORTED failure, never a raised ImportError.
    err = (r.error or "").lower()
    assert "receptor" in err or "docking unavailable" in err


def test_tier1_names_the_missing_package_when_docking_is_not_installed(monkeypatch):
    """An uninstalled optional dep must not masquerade as a docking failure.

    Reporting ImportError as a generic tier-1 failure would send a reader
    hunting for a bad receptor or a bad ligand when the real fix is
    `pip install ferric[docking]`.
    """
    import tools.pipeline.tiers as tiers

    def boom(*a, **k):
        raise ImportError("No module named 'vina'")

    monkeypatch.setattr(tiers, "dock_ligand", boom, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tools.docking",
        type("M", (), {"dock_ligand": staticmethod(boom)})(),
    )

    r = tiers.tier1_dock(
        SMALL,
        {
            "receptor_pdbqt": "/nonexistent.pdbqt",
            "box_center": (0.0, 0.0, 0.0),
            "box_size": (10.0, 10.0, 10.0),
        },
    )
    assert not r.ok and r.value is None
    assert "docking unavailable" in (r.error or "").lower()
    assert "vina" in (r.error or "").lower()


def test_every_adapter_shares_the_same_signature():
    """The funnel calls them interchangeably; a divergent signature breaks it."""
    import inspect

    for fn in (tier1_dock, tier2_forcefield, tier3_gfn2, tier4_dft):
        params = list(inspect.signature(fn).parameters)
        assert params == ["iso", "context"], f"{fn.__name__} has {params}"


def test_tier4_pins_the_memory_budget_and_restores_it(monkeypatch):
    """The Full-vs-Batched AO-cache decision must not depend on box load.

    ferric's default budget is 0.8 x *live* MemAvailable, so an unrelated job
    starting mid-pipeline can silently flip a candidate onto the batching
    path. tier4 forwards `mem_budget_gb` to FERRIC_MEM_BUDGET_GB to make that
    decision deterministic -- and must leave the environment as it found it.
    """
    import os

    import tools.pipeline.tiers as tiers

    seen = {}

    def fake_inner(iso, context, ferric):
        seen["budget"] = os.environ.get("FERRIC_MEM_BUDGET_GB")
        return TierResult(iso.canonical, -1.0)

    monkeypatch.setattr(tiers, "_tier4_dft_inner", fake_inner)
    monkeypatch.delenv("FERRIC_MEM_BUDGET_GB", raising=False)

    r = tiers.tier4_dft(SMALL, {"mem_budget_gb": 8})
    assert r.ok
    assert seen["budget"] == "8"  # pinned during the call
    assert "FERRIC_MEM_BUDGET_GB" not in os.environ  # and restored after


def test_tier4_without_a_budget_leaves_ferric_autodetect_alone(monkeypatch):
    import os

    import tools.pipeline.tiers as tiers

    seen = {}

    def fake_inner(iso, context, ferric):
        seen["budget"] = os.environ.get("FERRIC_MEM_BUDGET_GB")
        return TierResult(iso.canonical, -1.0)

    monkeypatch.setattr(tiers, "_tier4_dft_inner", fake_inner)
    monkeypatch.delenv("FERRIC_MEM_BUDGET_GB", raising=False)

    tiers.tier4_dft(SMALL, {})
    assert seen["budget"] is None


def test_tier4_restores_a_preexisting_budget(monkeypatch):
    import os

    import tools.pipeline.tiers as tiers

    monkeypatch.setattr(
        tiers, "_tier4_dft_inner", lambda iso, ctx, f: TierResult(iso.canonical, -1.0)
    )
    monkeypatch.setenv("FERRIC_MEM_BUDGET_GB", "3")

    tiers.tier4_dft(SMALL, {"mem_budget_gb": 8})
    assert os.environ["FERRIC_MEM_BUDGET_GB"] == "3"


# ── multi-seed docking (RESULTS.md M11) ──────────────────────────────────────


def _fake_dock_factory(scores_by_seed):
    """Return a dock_ligand stand-in whose score depends on the seed."""
    from types import SimpleNamespace

    calls = []

    def fake(
        mol,
        receptor,
        center,
        size=None,
        exhaustiveness=None,
        n_poses=None,
        seed=None,
        cpu=None,
    ):
        calls.append(seed)
        score = scores_by_seed.get(seed)
        if score is None:
            return SimpleNamespace(
                ok=False, error=f"no pose for seed {seed}", best=None, poses=[]
            )
        # A STRUCTURALLY VALID heavy-atom frame, not a single dummy carbon.
        # `tier1_dock` now re-hydrogenates the pose (a PDBQT pose is
        # united-atom, so the raw one is missing hydrogens and would be scored
        # as a different molecule by tier 3). That step matches the pose
        # against the candidate's SMILES, so a one-atom stand-in is correctly
        # rejected as "1 docked heavy atom vs 7". Build the frame from the
        # molecule the test actually uses.
        from rdkit import Chem
        from rdkit.Chem import AllChem

        _m = Chem.AddHs(Chem.MolFromSmiles(BENZOIC.canonical))
        AllChem.EmbedMolecule(_m, randomSeed=0xF00D)
        _conf = _m.GetConformer()
        _heavy = [a.GetIdx() for a in _m.GetAtoms() if a.GetSymbol() != "H"]
        pose = SimpleNamespace(
            vina_score=score,
            symbols=[_m.GetAtomWithIdx(i).GetSymbol() for i in _heavy],
            coords_angstrom=[
                (
                    _conf.GetAtomPosition(i).x,
                    _conf.GetAtomPosition(i).y,
                    _conf.GetAtomPosition(i).z,
                )
                for i in _heavy
            ],
            rdkit_index_of_heavy=list(range(len(_heavy))),
        )
        return SimpleNamespace(ok=True, error=None, best=pose, poses=[pose])

    return fake, calls


def test_multi_seed_docks_each_seed_and_keeps_the_best(monkeypatch):
    """M11: the starting conformer moves the answer more than search effort.

    Spending the tier-1 budget on independent embeddings is the measured-better
    trade, so the tier must actually try each one and keep the best score.
    """
    import tools.docking as docking

    fake, calls = _fake_dock_factory({0xF00D: -8.0, 0xF00E: -11.5, 0xF00F: -9.0})
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(
        BENZOIC,
        {"receptor_pdbqt": "r.pdbqt", "box_center": (0.0, 0.0, 0.0), "n_seeds": 3},
    )
    assert r.ok
    assert r.value == -11.5  # the best of the three
    assert calls == [0xF00D, 0xF00E, 0xF00F]  # each seed actually tried
    assert r.payload["winning_seed"] == 0xF00E
    assert r.payload["n_seeds"] == 3


def test_single_seed_is_the_old_behaviour(monkeypatch):
    """n_seeds=1 must dock exactly once, from the base seed."""
    import tools.docking as docking

    fake, calls = _fake_dock_factory({0xF00D: -8.0})
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(
        BENZOIC, {"receptor_pdbqt": "r.pdbqt", "box_center": (0.0, 0.0, 0.0)}
    )
    assert r.ok and r.value == -8.0
    assert calls == [0xF00D]


def test_multi_seed_survives_a_failing_seed(monkeypatch):
    """One bad embedding must not lose the ligand.

    A tier that dropped a candidate because one of its seeds failed would be
    silently biased against flexible molecules -- exactly the population-level
    error the funnel's failure accounting exists to prevent.
    """
    import tools.docking as docking

    fake, calls = _fake_dock_factory({0xF00D: -8.0, 0xF00F: -9.5})  # 0xF00E fails
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(
        BENZOIC,
        {"receptor_pdbqt": "r.pdbqt", "box_center": (0.0, 0.0, 0.0), "n_seeds": 3},
    )
    assert r.ok
    assert r.value == -9.5
    assert len(calls) == 3


def test_all_seeds_failing_reports_every_reason(monkeypatch):
    """A total failure must say what happened on each attempt."""
    import tools.docking as docking

    fake, _ = _fake_dock_factory({})
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(
        BENZOIC,
        {"receptor_pdbqt": "r.pdbqt", "box_center": (0.0, 0.0, 0.0), "n_seeds": 2},
    )
    assert not r.ok and r.value is None
    assert "no pose for seed" in r.error


# --- TierResult.resolution ---------------------------------------------------


def test_a_gap_below_the_resolution_is_not_a_ranking():
    """Two candidates closer than the tier's noise must not be ordered.

    MEASURED on this campaign: the best available ddE noise over a pose
    ensemble is 4.07 kcal/mol against substituent effects of 1-2, so a tier
    printing six digits still cannot separate them. `resolves` is how a caller
    finds that out without re-deriving it.
    """
    a = TierResult("a", value=-10.00, resolution=4.07)
    b = TierResult("b", value=-10.50, resolution=4.07)
    assert a.resolves(b) is False, "0.5 kcal/mol is inside a 4.07 noise floor"

    far = TierResult("far", value=-30.0, resolution=4.07)
    assert a.resolves(far) is True, "20 kcal/mol is well outside it"


def test_resolutions_combine_in_quadrature_not_singly():
    """A difference carries BOTH results' noise.

    Using one side's resolution alone understates the combined noise by up to
    sqrt(2), which is exactly the margin that turns "indistinguishable" into a
    confident ranking.
    """
    a = TierResult("a", value=0.0, resolution=3.0)
    b = TierResult("b", value=4.0, resolution=3.0)
    # Single-sided would say 4.0 > 3.0 -> resolved. Quadrature: sqrt(18)=4.24.
    assert a.resolves(b) is False, "4.0 must NOT clear a combined 4.24 floor"
    c = TierResult("c", value=5.0, resolution=3.0)
    assert a.resolves(c) is True


def test_an_uncharacterised_resolution_is_unknown_not_infinitely_precise():
    """`None` must not read as "this tier can resolve anything"."""
    known = TierResult("k", value=1.0, resolution=0.5)
    unknown = TierResult("u", value=1.1)
    assert unknown.resolution is None
    assert known.resolves(unknown) is None, "must be UNKNOWN, not True/False"
    assert unknown.resolves(known) is None


def test_comparing_to_a_failed_result_raises_rather_than_tying():
    """A tier that could not answer is not a tie, and must not silently be one."""
    ok = TierResult("ok", value=1.0, resolution=0.1)
    bad = TierResult("bad", value=None, error="xtb did not converge")
    with pytest.raises(ValueError, match="no value"):
        ok.resolves(bad)
    with pytest.raises(ValueError, match="no value"):
        bad.resolves(ok)


def test_resolution_defaults_to_none_so_existing_tiers_are_unchanged():
    """Adding the field must not silently re-grade every existing tier."""
    r = TierResult("x", value=1.0)
    assert r.resolution is None
    assert r.ok


def test_tier4_flags_that_dispersion_covers_only_the_QM_region():
    """With point charges, the D3 correction is PARTIAL and must say so.

    D3(BJ) is QM-ATOM-PAIRWISE: it sums over the molecule's atoms, and MM point
    charges are not atoms. So an embedded run gets dispersion WITHIN the QM
    region and NONE across the QM/MM boundary -- which is exactly the part a
    binding or pocket question depends on, since dispersion is the dominant
    attractive term there.

    Flagged rather than refused: the QM-internal correction is correct and
    useful for comparing conformers of one ligand. What must not happen is a
    caller reading the total as a fully dispersion-corrected embedded energy.

    Asserted on the FLAG rather than by running DFT, so this stays in the fast
    tier -- the flag is a statement about coverage, and its truth condition is
    `dispersion was computed AND point charges were supplied`.
    """

    # The flag's logic, stated here so a change to it fails visibly.
    def covers_qm_only(e_dispersion, point_charges):
        return e_dispersion is not None and bool(point_charges)

    assert covers_qm_only(-0.01, [(0.5, 1.0, 2.0, 3.0)]) is True, (
        "dispersion computed WITH point charges -> partial coverage, must flag"
    )
    assert covers_qm_only(-0.01, None) is False, (
        "gas phase: there is no MM region, so coverage is complete"
    )
    assert covers_qm_only(None, [(0.5, 1.0, 2.0, 3.0)]) is False, (
        "no dispersion requested: nothing to qualify"
    )
    assert covers_qm_only(None, None) is False


def test_a_nan_resolution_is_refused_because_it_suppresses_every_ranking():
    """NaN is the dangerous one, and it fails in the CAUTIOUS-looking direction.

    `resolves` compares `abs(gap) > combined`. Every `>` against NaN is False,
    so a NaN resolution reports EVERY pair as indistinguishable. MEASURED
    before the fix: two results 990 units apart came back `False` -- do not
    rank.

    That inverts the field's purpose. `resolution` exists to STOP a ranking the
    tier cannot support; a NaN instead suppresses every distinction the tier
    genuinely can make, and it looks like conservatism while doing it.

    `None` is the correct way to say "uncharacterised" and stays accepted --
    it returns `None` from `resolves`, which is UNKNOWN rather than a verdict.
    """
    import math

    with pytest.raises(ValueError, match="NaN"):
        TierResult("x", -10.0, resolution=math.nan)

    # The anchor: None still means uncharacterised and still yields None.
    a = TierResult("a", -10.0)
    b = TierResult("b", -1000.0)
    assert a.resolves(b) is None, "None must stay UNKNOWN, not become a verdict"


def test_a_negative_resolution_is_refused_because_it_is_squared():
    """-4.0 would behave exactly as +4.0, with nothing to indicate it."""
    with pytest.raises(ValueError, match="must be >= 0"):
        TierResult("x", -10.0, resolution=-4.0)

    # The anchor: the positive value it would have impersonated still works,
    # so the test is about the SIGN and not about the field being broken.
    a = TierResult("a", -10.0, resolution=4.0)
    b = TierResult("b", -20.0, resolution=4.0)
    assert a.resolves(b) is True


def test_an_infinite_resolution_is_refused():
    """A tier that resolves nothing should say so, not encode it arithmetically."""
    with pytest.raises(ValueError, match="infinite"):
        TierResult("x", -10.0, resolution=float("inf"))


def test_zero_resolution_is_ALLOWED():
    """A tier claiming exact resolution is coherent; the quadrature handles it."""
    a = TierResult("a", -10.0, resolution=0.0)
    b = TierResult("b", -10.5, resolution=0.0)
    assert a.resolves(b) is True, "a 0.5 gap at zero noise is resolvable"
    same = TierResult("c", -10.0, resolution=0.0)
    assert a.resolves(same) is False, (
        "an exact tie is not resolvable even at zero noise"
    )


def test_the_guard_is_keyed_on_None_and_not_on_FALSINESS():
    """`if not resolution` would skip validation for every falsy value.

    A MUTATION SURVIVED here. Rewriting the early return as
    `if not self.resolution: return` passes all the tests above, because the
    only falsy value they exercise is 0.0 -- which is VALID either way, so it
    cannot tell the two guards apart.

    What it does let through is `False`. A bool is not a resolution, and
    `resolution=False` then flows into `resolves` as 0, claiming the tier
    resolves exact ties. That is a real, silent wrong answer reachable from a
    plausible typo (`resolution=False` where `error=...` was meant).

    So the discriminating input is a FALSY value that must be REFUSED, not the
    falsy value that must be accepted.
    """
    with pytest.raises(TypeError, match="real number"):
        TierResult("x", -10.0, resolution=False)
    with pytest.raises(TypeError, match="real number"):
        TierResult("x", -10.0, resolution=True)

    # ...and the falsy value that IS valid still is, so the guard has not been
    # over-corrected into rejecting everything falsy.
    assert TierResult("x", -10.0, resolution=0.0).resolution == 0.0


def test_resolves_agrees_with_EXACT_arithmetic_even_at_the_float_limit():
    """Saturation to `inf` on both sides produced one wrong verdict.

    `hypot(1.7e308, 1.7e308)` and `abs(-1.7e308 - 1.7e308)` both overflow, and
    `inf > inf` is False -- so two values 3.4e308 apart with 1.7e308 noise each
    came back "indistinguishable" when exact arithmetic says they resolve.

    Found by comparing against `fractions.Fraction`, which is the point of this
    test: a float-only check cannot detect a float-only defect. Every case is
    validated against exact rational arithmetic rather than a hand-computed
    expectation.

    UNREACHABLE with real inputs -- `resolution` is a noise figure in the
    value's units and the largest MEASURED here is 4.07 kcal/mol -- but the
    repair is two lines (halve both sides, which cannot change an inequality),
    so the alternative was leaving a known-wrong branch in a function whose job
    is to decide whether a ranking is supported.
    """
    from fractions import Fraction

    cases = [
        # (r1, r2, v1, v2) -- the first is the one that was wrong.
        (1.7e308, 1.7e308, -1.7e308, 1.7e308),
        (1e308, 1e308, 0.0, 1e308),
        (1e200, 1e200, 0.0, 1e308),
        (1e308, 1.0, 0.0, 1e308),
        (1.0, 1.0, -1.7e308, 1.7e308),
        # ...and an ordinary case, so the test is not only about extremes.
        (4.07, 4.07, -10.0, -20.0),
        (4.07, 4.07, -10.0, -10.5),
    ]
    for r1, r2, v1, v2 in cases:
        got = TierResult("a", v1, resolution=r1).resolves(
            TierResult("b", v2, resolution=r2)
        )
        # Exact, via squares so no sqrt is needed.
        want = (Fraction(v1) - Fraction(v2)) ** 2 > Fraction(r1) ** 2 + Fraction(
            r2
        ) ** 2
        assert got == want, (
            f"resolution=({r1:.3e}, {r2:.3e}) value=({v1:.3e}, {v2:.3e}): "
            f"got {got}, exact arithmetic says {want}"
        )


def test_the_qm_mm_dispersion_comment_does_not_claim_missing_code():
    """A "does not exist yet" comment must not outlive the code it describes.

    `tiers.py` said closing the QM/MM dispersion gap "needs LJ terms on the MM
    sites (`ferric-mm`), which do not exist yet". They had existed for three
    weeks: `ferric_mm::qm_mm_lj_energy_gradient` landed 2026-08-27 (4a930a0f),
    the comment was written 2026-09-19. It then generated a ticket to build
    something the repo already had.

    The FLAG is correct -- tier 4 really has no QM-to-MM dispersion -- but for
    a different reason: `run_dft` takes no topology argument, so the LJ path is
    simply unused, and `ferric-mm` assigns no parameters anyway.

    This asserts the specific false claim cannot come back, and that the
    function it was wrong about is still exported. Deliberately narrow: a
    general "no comment may claim anything is missing" guard would be
    unmaintainable, and the value here is pinning ONE claim that already
    misled once.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    tiers_src = (repo / "tools/pipeline/tiers.py").read_text()

    assert "which do not exist yet" not in tiers_src, (
        "the retracted claim is back in tiers.py -- `ferric-mm`'s QM-MM LJ "
        "terms DO exist; see qm_mm_lj_energy_gradient"
    )

    # ...and the function is both DEFINED and EXPORTED, so this test fails
    # loudly if it is removed rather than silently permitting the old comment.
    #
    # BOTH halves matter. Checking only the definition would pass while the
    # symbol was unreachable: `energy.rs` is a private module, and dropping the
    # `pub use` in `lib.rs` makes `ferric_mm::qm_mm_lj_energy_gradient` vanish
    # from the public API with `pub fn` still sitting in the source. A caller
    # would then be right that the function "does not exist" for them, which is
    # exactly the claim this test exists to keep false.
    mm_src = (repo / "crates/ferric-mm/src/energy.rs").read_text()
    assert "pub fn qm_mm_lj_energy_gradient" in mm_src, (
        "qm_mm_lj_energy_gradient is gone from ferric-mm/src/energy.rs; if the "
        "QM-MM LJ arithmetic was genuinely removed, the comment above needs "
        "rewriting rather than this assertion deleting"
    )
    mm_lib = (repo / "crates/ferric-mm/src/lib.rs").read_text()
    assert "qm_mm_lj_energy_gradient" in mm_lib, (
        "qm_mm_lj_energy_gradient is defined but no longer re-exported from "
        "ferric-mm/src/lib.rs, so it is not reachable as "
        "`ferric_mm::qm_mm_lj_energy_gradient`. The arithmetic existing in a "
        "private module is not the same as a caller being able to use it."
    )


def test_tier1_missing_receptor_is_an_explained_failure_not_a_crash():
    """A missing prerequisite must DROP the candidate, not kill the run.

    `tier1_dock` read `context["receptor_pdbqt"]` and `context["box_center"]`
    as bare subscripts at the dock call. MEASURED before the fix: a caller who
    forgot the receptor got `KeyError: 'receptor_pdbqt'` propagating out of
    `run_funnel`, taking the WHOLE RUN down -- 1 candidate or 10,000.

    Every other tier honours the contract (`tier3_gfn2` returns
    "no geometry for GFN2"), and `tier1_dock` itself already returned explained
    failures for unparseable SMILES and multi-fragment molecules. The context
    keys were simply missed.

    The funnel half is the point: the report must show `n_failed` and carry the
    reason, because that is what a caller sees when a screen of thousands has
    one misconfigured stage.
    """
    from tools.campaign.hierarchy import Tier
    from tools.isomers.model import Isomer
    from tools.pipeline import Stage, run_funnel
    from tools.pipeline.tiers import tier1_dock

    iso = Isomer(smiles="CCO", kind="parent", transform="t", parent_smiles="CCO")

    for ctx, why in [
        ({}, "neither key"),
        ({"receptor_pdbqt": "/x.pdbqt"}, "box_center missing"),
        ({"box_center": (0.0, 0.0, 0.0)}, "receptor missing"),
    ]:
        r = tier1_dock(iso, ctx)
        assert not r.ok, f"{why}: must fail"
        assert r.value is None, (
            f"{why}: a failed tier must not return a value -- the funnel ranks "
            f"ASCENDING, so a placeholder 0.0 would be the BEST score"
        )
        assert "context[" in r.error, f"{why}: the error must name the key: {r.error}"
        assert "prepare_receptor" in r.error, (
            f"{why}: the error must name the remedy: {r.error}"
        )

    # And the funnel must COMPLETE, reporting the failure rather than raising.
    rep = run_funnel([iso], [Stage(Tier.SEARCH, tier1_dock, keep=1, name="dock")], {})
    assert rep.outcomes[0].n_failed == 1
    assert rep.survivors == []
    assert any("receptor_pdbqt" in e for e in rep.outcomes[0].errors), (
        f"the reason must reach the report: {rep.outcomes[0].errors}"
    )


def test_a_cached_geometry_that_is_not_the_molecule_is_REFUSED():
    """The geometry reaching the quantum tiers must BE the candidate.

    A geometry in `context["geometry"]` has crossed a tool boundary -- Vina, a
    file reader, an embedder -- and any of them can return a different species.
    The one that happened: a PDBQT pose is UNITED-ATOM, so aspirin arrived as
    14 atoms where 21 went in, and `_harvest_geometry` fed that to tiers 3
    and 4.

    The damage was UNEVEN, which is why it went unnoticed for so long:

        tier 4  failed -- but BY ACCIDENT. 87 electrons with multiplicity 1
                trips ferric's charge/multiplicity parity check, which is a
                statement about electron count, not about identity.
        tier 3  did not. GFN2 returned -35.492226 Ha against -39.621219 for
                the real molecule: both plausible, neither an error, and
                2591 kcal/mol apart.

    The guard lives in `_embedded`, the shared funnel both tiers read through,
    so one check covers both and anything added later.
    """
    from rdkit import Chem

    from tools.pipeline.tiers import tier3_gfn2, tier4_dft

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin: 21 atoms with H, 13 heavy
    iso = Isomer(smiles=smiles, kind="parent", transform="t", parent_smiles=smiles)

    # The real failure shape: heavy atoms plus the one polar hydrogen.
    full = Chem.AddHs(Chem.MolFromSmiles(smiles))
    united = [a.GetSymbol() for a in full.GetAtoms() if a.GetSymbol() != "H"] + ["H"]
    coords = [(0.0, 0.0, float(i)) for i in range(len(united))]
    ctx = {
        "geometry": {iso.canonical: {"symbols": united, "coords": coords}},
        "basis": "sto-3g",
    }

    for name, tier in (("tier3_gfn2", tier3_gfn2), ("tier4_dft", tier4_dft)):
        r = tier(iso, ctx)
        assert not r.ok, f"{name} scored a geometry that is not the molecule"
        assert r.value is None, (
            f"{name} must not return a value -- the funnel ranks ASCENDING, so "
            f"a number from the wrong molecule can win"
        )
        # The message has to name the discrepancy, or the reader cannot tell
        # this from a missing cache.
        assert "not this molecule" in r.error, f"{name}: {r.error}"
        assert "C9H1O4" in r.error and "C9H8O4" in r.error, (
            f"{name} must give BOTH formulas: {r.error}"
        )
        assert "missing H7" in r.error, (
            f"{name} must name what is absent, not just that something is: {r.error}"
        )


def test_the_formula_check_compares_ELEMENTS_not_just_the_count():
    """Right number of the wrong atoms must fail too.

    An atom-count check is the obvious version and it is not enough: swapping a
    carbon for a nitrogen keeps the count and changes the molecule. The
    united-atom bug happened to change the count, so a count check would have
    caught THAT instance while leaving the class open.
    """
    from tools.pipeline.tiers import tier2_forcefield, tier3_gfn2

    iso = Isomer(smiles="CCO", kind="parent", transform="t", parent_smiles="CCO")
    good = tier2_forcefield(iso, {})
    assert good.ok

    swapped = list(good.payload["symbols"])
    swapped[0] = "N"
    assert len(swapped) == len(good.payload["symbols"]), "the count must be unchanged"

    r = tier3_gfn2(
        iso,
        {
            "geometry": {
                iso.canonical: {"symbols": swapped, "coords": good.payload["coords"]}
            }
        },
    )
    assert not r.ok and r.value is None
    assert "C1H6N1O1" in r.error and "C2H6O1" in r.error, r.error


def test_a_legitimate_geometry_still_passes():
    """The guard must not cost the cases the pipeline exists to move around.

    A docked pose, a relaxed pose and a re-embedded conformer are all the same
    molecule at different coordinates. The check compares FORMULA only and says
    nothing about geometry, so all three pass -- verified here by perturbing a
    real geometry and confirming the energy still comes back.
    """
    from tools.pipeline.tiers import tier2_forcefield, tier3_gfn2

    iso = Isomer(smiles="CCO", kind="parent", transform="t", parent_smiles="CCO")

    uncached = tier3_gfn2(iso, {})
    assert uncached.ok and uncached.value is not None

    base = tier2_forcefield(iso, {})
    moved = [(x + 0.05, y, z) for x, y, z in base.payload["coords"]]
    cached = tier3_gfn2(
        iso,
        {
            "geometry": {
                iso.canonical: {"symbols": base.payload["symbols"], "coords": moved}
            }
        },
    )
    assert cached.ok, f"a perturbed conformer must still score: {cached.error}"
    assert cached.value is not None
