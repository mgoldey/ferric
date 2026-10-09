"""The pocket field reaches BOTH quantum tiers, in Bohr, and is applied.

`context["point_charges"]` is forwarded unchanged to two engines:

    tier3_gfn2 -> xtb_engine.singlepoint -> xtb's `pcharge` file
    tier4_dft  -> ferric.run_dft(point_charges=...)

Both must read the coordinates as BOHR, which is what
`tools.active_site.pocket_charges` produces. A unit slip here does not crash:
the engine converges, a ranking comes out, and every candidate is wrong by the
same factor -- a deterministic artifact that reproduces perfectly and so looks
like physics.

The unit is pinned against an INDEPENDENT construction, Coulomb's law, rather
than against the code under test. NH4+ (charge +1) with a +1 point charge
20 Bohr from the nitrogen: the leading interaction is q1 q2 / R = 1/20 Ha, and
polarisation is O(R^-3) or smaller.

MEASURED 2026-10-05 (xtb 6.x, GFN2, `pcharge` file), dE versus 1/R in Bohr and
in Angstrom:

    R    dE (xtb)   1/R_bohr   1/R_angstrom
    10   0.096579   0.100000   0.052918
    20   0.049574   0.050000   0.026459
    40   0.024943   0.025000   0.013229

So xtb reads BOHR -- the residual against 1/R_bohr is 4.3e-4 Ha at R=20 and
falls ~8x per doubling, as a polarisation tail should. An unsalvaged branch
(fix/funnel-pose-handoff) asserted the opposite (Angstrom) and converted at
tier 3; this test would reject that conversion, which puts the charge at
10.6 Bohr and gives dE ~ 0.094 Ha.
"""

from __future__ import annotations

import pytest

from tools.isomers.model import Isomer
from tools.pipeline import tiers as tiers_mod

ANGSTROM_PER_BOHR = 0.529_177_210_92

# The probe: NH4+ centred on N, a +1 charge R_BOHR away along z.
R_BOHR = 20.0
EXPECTED_DE = 1.0 / R_BOHR  # Hartree, Coulomb's law with R in Bohr
# Measured residual against EXPECTED_DE at R=20 is 4.3e-4 Ha for GFN2 (the
# polarisation tail). The bar is ~10x that; the Angstrom reading is 0.0235 Ha
# away, ~5x the bar.
TOL_HA = 5e-3

NH4 = Isomer("[NH4+]", "parent", "none", "[NH4+]", net_charge=1)


def _nh4_context(**extra):
    """NH4+ with N at the origin (Angstrom), as a cached geometry."""
    r = tiers_mod.tier2_forcefield(NH4, {"seed": 1})
    assert r.ok, r.error
    n = r.payload["symbols"].index("N")
    ox, oy, oz = r.payload["coords"][n]
    coords = [(x - ox, y - oy, z - oz) for x, y, z in r.payload["coords"]]
    ctx = {
        "geometry": {NH4.canonical: {"symbols": r.payload["symbols"], "coords": coords}}
    }
    ctx.update(extra)
    return ctx


def _xtb_or_skip(result):
    if not result.ok and "not on PATH" in (result.error or ""):
        pytest.skip("xtb not available")
    assert result.ok, result.error


def test_the_reachability_premise_bohr_and_angstrom_are_distinguishable():
    """The two readings must sit further apart than the tolerance, or the
    tests below could not tell them apart."""
    angstrom_reading = 1.0 / (R_BOHR / ANGSTROM_PER_BOHR)
    assert abs(EXPECTED_DE - angstrom_reading) > 4 * TOL_HA


# ── exactness anchors: a zero charge is not a charge ─────────────────────────


def test_a_zero_charge_reproduces_the_vacuum_energy_in_tier3():
    vac = tiers_mod.tier3_gfn2(NH4, _nh4_context())
    _xtb_or_skip(vac)
    zero = tiers_mod.tier3_gfn2(
        NH4, _nh4_context(point_charges=[(0.0, 0.0, 0.0, R_BOHR)])
    )
    assert zero.ok, zero.error
    assert zero.value == pytest.approx(vac.value, abs=1e-8)


def test_a_zero_charge_reproduces_the_vacuum_energy_in_tier4():
    vac = tiers_mod.tier4_dft(NH4, _nh4_context(basis="sto-3g"))
    assert vac.ok, vac.error
    zero = tiers_mod.tier4_dft(
        NH4, _nh4_context(basis="sto-3g", point_charges=[(0.0, 0.0, 0.0, R_BOHR)])
    )
    assert zero.ok, zero.error
    assert zero.value == pytest.approx(vac.value, abs=1e-8)


# ── the unit, against Coulomb's law ──────────────────────────────────────────


def test_tier3_reads_the_pocket_field_in_bohr():
    """Fails if tier 3 drops the field (dE = 0) or converts it to Angstrom
    before xtb (dE ~ 0.094 Ha), or if xtb's own convention changes."""
    vac = tiers_mod.tier3_gfn2(NH4, _nh4_context())
    _xtb_or_skip(vac)
    fld = tiers_mod.tier3_gfn2(
        NH4, _nh4_context(point_charges=[(1.0, 0.0, 0.0, R_BOHR)])
    )
    assert fld.ok, fld.error
    de = fld.value - vac.value
    assert de == pytest.approx(EXPECTED_DE, abs=TOL_HA), (
        f"tier 3 shift {de:.6f} Ha for a +1 charge at {R_BOHR} Bohr; Coulomb "
        f"says {EXPECTED_DE:.6f}. 0 means the field was dropped; ~0.0265 means "
        f"it was read as Angstrom; ~0.094 means it was converted to Angstrom "
        f"and then read as Bohr"
    )


def test_tier4_reads_the_pocket_field_in_bohr():
    """The same probe through ferric: the two tiers must agree on the unit,
    since they are handed the SAME context key."""
    vac = tiers_mod.tier4_dft(NH4, _nh4_context(basis="sto-3g"))
    assert vac.ok, vac.error
    fld = tiers_mod.tier4_dft(
        NH4,
        _nh4_context(basis="sto-3g", point_charges=[(1.0, 0.0, 0.0, R_BOHR)]),
    )
    assert fld.ok, fld.error
    de = fld.value - vac.value
    assert de == pytest.approx(EXPECTED_DE, abs=TOL_HA), (
        f"tier 4 shift {de:.6f} Ha for a +1 charge at {R_BOHR} Bohr; Coulomb "
        f"says {EXPECTED_DE:.6f}"
    )
