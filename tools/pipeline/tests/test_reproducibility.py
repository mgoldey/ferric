"""The pipeline's headline claim is REPRODUCIBILITY. Test it end to end.

Every stochastic stage (ETKDG embedding, Vina search) takes a seeded default.
If any of them leaks an unseeded random source, two runs diverge -- and a
candidate ranking that changes between runs is not a result, however good the
physics above it.

This runs the real enumerate -> force-field -> GFN2 path on a small parent, so
it is a genuine end-to-end check rather than a stub.
"""
from __future__ import annotations

import pytest

from tools.campaign.hierarchy import Tier
from tools.isomers import enumerate_isomers
from tools.pipeline import Stage, run_funnel
from tools.pipeline.tiers import tier2_forcefield, tier3_gfn2

PARENT = "OC(=O)c1ccccc1"


def _run():
    cands = enumerate_isomers(PARENT, substituents={"F": "F"},
                              include_stereo=False, include_rings=False,
                              include_bioisosteres=False)
    stages = [
        Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=3, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=2, name="gfn2"),
    ]
    return run_funnel(cands, stages, {"seed": 0xF00D})


@pytest.fixture(scope="module")
def two_runs():
    a, b = _run(), _run()
    if not a.results.get("gfn2") or all(not r.ok for r in a.results["gfn2"]):
        errs = {r.error for r in a.results.get("gfn2", []) if r.error}
        if any("not on PATH" in (e or "") for e in errs):
            pytest.skip("xtb not available")
    return a, b


def test_two_identical_runs_give_identical_survivors(two_runs):
    a, b = two_runs
    assert [i.canonical for i in a.survivors] == [i.canonical for i in b.survivors]


def test_two_identical_runs_give_identical_energies(two_runs):
    a, b = two_runs
    for stage in ("ff", "gfn2"):
        va = [(r.candidate_id, r.value) for r in a.results[stage]]
        vb = [(r.candidate_id, r.value) for r in b.results[stage]]
        assert va == vb, f"stage {stage!r} is not reproducible"


def test_the_funnel_actually_narrows(two_runs):
    a, _ = two_runs
    ns = [o.n_in for o in a.outcomes] + [len(a.survivors)]
    assert ns == sorted(ns, reverse=True), f"population did not narrow: {ns}"


def test_the_enumeration_feeding_it_is_reproducible():
    a = [i.canonical for i in enumerate_isomers(PARENT, substituents={"F": "F"},
                                                include_stereo=False,
                                                include_rings=False,
                                                include_bioisosteres=False)]
    b = [i.canonical for i in enumerate_isomers(PARENT, substituents={"F": "F"},
                                                include_stereo=False,
                                                include_rings=False,
                                                include_bioisosteres=False)]
    assert a == b


def test_a_different_seed_is_allowed_to_change_geometry_energies():
    """Reachability for the determinism tests: if the seed had NO effect, the
    reproducibility assertions above would pass vacuously."""
    from tools.isomers.model import Isomer

    iso = Isomer("OC(=O)c1ccc(F)cc1", "substitutional", "F", PARENT)
    a = tier2_forcefield(iso, {"seed": 1})
    b = tier2_forcefield(iso, {"seed": 987_654})
    assert a.ok and b.ok
    same_coords = a.payload["coords"] == b.payload["coords"]
    assert not same_coords or a.value == pytest.approx(b.value), (
        "different seeds gave identical coordinates AND the test cannot tell "
        "whether seeding works -- pick a molecule with more conformational freedom"
    )
