"""#324: with a receptor configured, tiers 3 and 4 score IN the pocket field.

The danuglipron driver built a context with a receptor and no
`point_charges`, so both quantum tiers silently scored the docked pose in
vacuum. Three things are pinned here:

1. A receptor without a field is REFUSED by the tiers (explicit
   `point_charges=None` still means vacuum).
2. The real 7LCJ pocket field CHANGES the energy by far more than SCF noise
   -- the negative control: a vacuum run would pass every plumbing test, so
   only a measured difference proves the field is applied -- while an EMPTY
   field reproduces vacuum exactly (the anchor).
3. The danuglipron driver's context carries that field, and the quantum
   tiers receive it through `run_funnel`.

Probe: acetate (charge -1) placed at the pocket's charge centroid, so it sits
in the field's frame. Coordinates in Angstrom, charges in Bohr (the convention
pinned in test_pocket_field_units.py).
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest

from tools.isomers.model import Isomer
from tools.pipeline import tiers as T

REPO = Path(__file__).resolve().parents[3]
POCKET_PDB = REPO / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
BOHR = 0.52917721092
ACETATE = Isomer("CC(=O)[O-]", "parent", "none", "CC(=O)[O-]", net_charge=-1)

# MEASURED (see the PR): field minus vacuum for this probe is O(0.1) Ha at
# both tiers, and a repeated vacuum run differs from itself by 0. The bar is
# set orders of magnitude below the effect and far above any SCF noise.
MIN_FIELD_SHIFT_HA = 1e-3


@pytest.fixture(scope="module")
def pocket():
    if shutil.which("pdb2pqr") is None and shutil.which("pdb2pqr30") is None:
        pytest.skip("pdb2pqr not available")
    from tools.active_site.pocket_charges import derive_pocket_charges

    return derive_pocket_charges(str(POCKET_PDB)).charges


@pytest.fixture(scope="module")
def geometry(pocket):
    r = T.tier2_forcefield(ACETATE, {"seed": 1})
    assert r.ok, r.error
    xs = r.payload["coords"]
    mid = [sum(c[i] for c in xs) / len(xs) for i in range(3)]
    cen = [sum(q[i] for q in pocket) / len(pocket) * BOHR for i in (1, 2, 3)]
    coords = [tuple(c[i] - mid[i] + cen[i] for i in range(3)) for c in xs]
    return {ACETATE.canonical: {"symbols": r.payload["symbols"], "coords": coords}}


def _ok(r):
    if not r.ok and "not on PATH" in (r.error or ""):
        pytest.skip("xtb not available")
    assert r.ok, r.error
    return r


TIERS = [
    pytest.param(T.tier3_gfn2, {}, id="tier3"),
    pytest.param(T.tier4_dft, {"basis": "sto-3g"}, id="tier4"),
]


# ── 1. refusal ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fn,extra", TIERS)
def test_a_receptor_without_a_field_is_refused_not_scored_in_vacuum(
    fn, extra, geometry
):
    ctx = {"geometry": geometry, "receptor_pdbqt": "r.pdbqt", **extra}
    r = fn(ACETATE, ctx)
    assert not r.ok and r.value is None
    assert "VACUUM" in r.error, r.error


@pytest.mark.parametrize("fn,extra", TIERS)
def test_an_explicit_None_field_still_means_vacuum(fn, extra, geometry):
    ctx = {"geometry": geometry, "receptor_pdbqt": "r.pdbqt", **extra}
    explicit = _ok(fn(ACETATE, {**ctx, "point_charges": None}))
    plain = _ok(fn(ACETATE, {"geometry": geometry, **extra}))
    assert explicit.value == plain.value


# ── 2. the field is applied: negative control + exact anchor ────────────────


@pytest.mark.parametrize("fn,extra", TIERS)
def test_the_pocket_field_changes_the_energy_and_an_empty_one_does_not(
    fn, extra, geometry, pocket
):
    vac = _ok(fn(ACETATE, {"geometry": geometry, **extra}))
    vac2 = _ok(fn(ACETATE, {"geometry": geometry, **extra}))
    empty = _ok(fn(ACETATE, {"geometry": geometry, "point_charges": [], **extra}))
    fld = _ok(fn(ACETATE, {"geometry": geometry, "point_charges": pocket, **extra}))

    noise = abs(vac2.value - vac.value)
    shift = abs(fld.value - vac.value)
    # ANCHOR: no charges, no change -- bit for bit.
    assert empty.value == vac.value
    # NEGATIVE CONTROL: a vacuum calculation could not produce this.
    assert shift > MIN_FIELD_SHIFT_HA, f"field shift {shift:.3e} Ha"
    assert shift > 1e6 * max(noise, 1e-15), (shift, noise)


@pytest.mark.parametrize("fn,extra", TIERS)
def test_the_interaction_score_is_field_minus_vacuum(fn, extra, geometry, pocket):
    base = {"geometry": geometry, **extra}
    vac = _ok(fn(ACETATE, base))
    fld = _ok(fn(ACETATE, {**base, "point_charges": pocket}))
    inter = _ok(fn(ACETATE, {**base, "point_charges": pocket, "score": "interaction"}))
    assert not inter.formula_bound and fld.formula_bound
    assert inter.value == inter.payload["e_total"] - inter.payload["e_vacuum"]
    assert inter.value == pytest.approx(fld.value - vac.value, abs=1e-8)


@pytest.mark.parametrize("fn,extra", TIERS)
def test_an_interaction_score_without_a_field_is_refused(fn, extra, geometry):
    r = fn(ACETATE, {"geometry": geometry, "score": "interaction", **extra})
    assert not r.ok and "non-empty" in r.error


# ── 3. the danuglipron driver supplies the field, and it arrives ───────────


def _driver():
    return importlib.import_module("experiments.danuglipron.run_isomer_pipeline")


def test_the_driver_context_carries_the_pocket_field_in_bohr(pocket):
    ctx = _driver().build_context("r.pdbqt", (0.0, 0.0, 0.0), (20.0,) * 3, pocket)
    assert ctx["point_charges"] == list(pocket) and len(ctx["point_charges"]) > 0
    assert ctx["score"] == "interaction"


def test_the_driver_refuses_an_empty_pocket():
    with pytest.raises(ValueError, match="no pocket charges"):
        _driver().build_context("r.pdbqt", (0.0, 0.0, 0.0), (20.0,) * 3, [])


def test_the_quantum_tiers_RECEIVE_the_field_through_the_funnel(pocket, geometry):
    """Spy on what tiers 3 and 4 are handed by run_funnel under the driver's
    context: the full pocket field, not None and not empty."""
    from tools.campaign.hierarchy import Tier
    from tools.pipeline import Stage, TierResult, run_funnel

    seen = {}

    def spy(name):
        def fn(iso, ctx):
            seen[name] = ctx.get("point_charges")
            return TierResult(iso.canonical, -1.0)

        return fn

    ctx = _driver().build_context("r.pdbqt", (0.0, 0.0, 0.0), (20.0,) * 3, pocket)
    ctx["geometry"] = geometry
    run_funnel(
        [ACETATE],
        [
            Stage(Tier.SEMIEMPIRICAL, spy("gfn2"), 1, "gfn2"),
            Stage(Tier.QUANTUM, spy("dft"), 1, "dft"),
        ],
        ctx,
    )
    assert seen["gfn2"] == seen["dft"] == list(pocket)
