"""Tier adapters: uniform signature, honest failure, and correct wiring."""
from __future__ import annotations

import pytest

from tools.isomers.model import Isomer
from tools.pipeline.tiers import (
    TierResult, tier1_dock, tier2_forcefield, tier3_gfn2, tier4_dft,
)

SMALL = Isomer("CO", "parent", "none", "CO")          # methanol: cheap everywhere
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
    r = tier1_dock(SMALL, {"receptor_pdbqt": "/nonexistent.pdbqt",
                           "box_center": (0.0, 0.0, 0.0),
                           "box_size": (10.0, 10.0, 10.0)})
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
        __import__("sys").modules, "tools.docking",
        type("M", (), {"dock_ligand": staticmethod(boom)})())

    r = tiers.tier1_dock(SMALL, {"receptor_pdbqt": "/nonexistent.pdbqt",
                                 "box_center": (0.0, 0.0, 0.0),
                                 "box_size": (10.0, 10.0, 10.0)})
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
    assert seen["budget"] == "8"                      # pinned during the call
    assert "FERRIC_MEM_BUDGET_GB" not in os.environ   # and restored after


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

    monkeypatch.setattr(tiers, "_tier4_dft_inner",
                        lambda iso, ctx, f: TierResult(iso.canonical, -1.0))
    monkeypatch.setenv("FERRIC_MEM_BUDGET_GB", "3")

    tiers.tier4_dft(SMALL, {"mem_budget_gb": 8})
    assert os.environ["FERRIC_MEM_BUDGET_GB"] == "3"


# ── multi-seed docking (RESULTS.md M11) ──────────────────────────────────────

def _fake_dock_factory(scores_by_seed):
    """Return a dock_ligand stand-in whose score depends on the seed."""
    from types import SimpleNamespace

    calls = []

    def fake(mol, receptor, center, size=None, exhaustiveness=None,
             n_poses=None, seed=None, cpu=None):
        calls.append(seed)
        score = scores_by_seed.get(seed)
        if score is None:
            return SimpleNamespace(ok=False, error=f"no pose for seed {seed}",
                                   best=None, poses=[])
        pose = SimpleNamespace(vina_score=score, symbols=["C"],
                               coords_angstrom=[(0.0, 0.0, 0.0)])
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

    r = tier1_dock(BENZOIC, {"receptor_pdbqt": "r.pdbqt",
                             "box_center": (0.0, 0.0, 0.0),
                             "n_seeds": 3})
    assert r.ok
    assert r.value == -11.5                       # the best of the three
    assert calls == [0xF00D, 0xF00E, 0xF00F]      # each seed actually tried
    assert r.payload["winning_seed"] == 0xF00E
    assert r.payload["n_seeds"] == 3


def test_single_seed_is_the_old_behaviour(monkeypatch):
    """n_seeds=1 must dock exactly once, from the base seed."""
    import tools.docking as docking

    fake, calls = _fake_dock_factory({0xF00D: -8.0})
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(BENZOIC, {"receptor_pdbqt": "r.pdbqt",
                             "box_center": (0.0, 0.0, 0.0)})
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

    r = tier1_dock(BENZOIC, {"receptor_pdbqt": "r.pdbqt",
                             "box_center": (0.0, 0.0, 0.0),
                             "n_seeds": 3})
    assert r.ok
    assert r.value == -9.5
    assert len(calls) == 3


def test_all_seeds_failing_reports_every_reason(monkeypatch):
    """A total failure must say what happened on each attempt."""
    import tools.docking as docking

    fake, _ = _fake_dock_factory({})
    monkeypatch.setattr(docking, "dock_ligand", fake)

    r = tier1_dock(BENZOIC, {"receptor_pdbqt": "r.pdbqt",
                             "box_center": (0.0, 0.0, 0.0),
                             "n_seeds": 2})
    assert not r.ok and r.value is None
    assert "no pose for seed" in r.error
