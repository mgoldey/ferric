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
    assert "receptor" in (r.error or "").lower()


def test_every_adapter_shares_the_same_signature():
    """The funnel calls them interchangeably; a divergent signature breaks it."""
    import inspect

    for fn in (tier1_dock, tier2_forcefield, tier3_gfn2, tier4_dft):
        params = list(inspect.signature(fn).parameters)
        assert params == ["iso", "context"], f"{fn.__name__} has {params}"
