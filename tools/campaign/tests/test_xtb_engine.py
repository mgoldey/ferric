"""Tests for the GFN2-xTB subprocess engine.

Two failure modes here are silent and would corrupt every downstream number, so
they get dedicated tests:

1.  **A miscompiled xtb.** gfortran 13.3 at `-O3` makes xtb 6.7.1's GFN1/GFN2
    analytic gradients ~20x wrong while leaving energies BYTE-IDENTICAL. A
    single-point comparison cannot detect it; only an optimization can. So
    `verify_xtb_build` is the gate, and `relax` must refuse to run without it.

2.  **Silently ignored point charges.** xtb takes external point charges from a
    bare `pcharge` file in the working directory. `xtb --help` documents no flag
    for this, and the obvious guess (`--input pcharge`) is xtb's DETAILED-INPUT
    flag with a totally different namelist format. If the file is not picked up,
    every "in field" energy silently equals its vacuum value -- a wrong answer
    with no error anywhere. `test_point_charges_actually_change_the_energy`
    pins that the field is really being applied.
"""
from __future__ import annotations

import math
import shutil

import pytest

from tools.campaign.xtb_engine import (
    HARTREE_TO_KCAL_MOL,
    anneal,
    relax,
    singlepoint,
    verify_xtb_build,
    xtb_available,
)

pytestmark = pytest.mark.skipif(
    not xtb_available(),
    reason="the `xtb` binary is not on PATH; see tools/campaign/xtb_engine.py "
           "for the LD_LIBRARY_PATH/XTBPATH setup this box needs",
)

WATER_SYMBOLS = ["O", "H", "H"]
WATER = [(0.0, 0.0, 0.0), (0.9578, 0.0, 0.0), (-0.24, 0.927, 0.0)]
# GFN2 water single-point at this geometry, measured 2026-08-29 on the verified
# -O2 build. Written as a literal so a library change moves this test.
WATER_GFN2_E = -5.070374101618


def test_singlepoint_reproduces_the_known_gfn2_water_energy():
    r = singlepoint(WATER_SYMBOLS, WATER)
    assert r.ok, r.error
    assert r.energy == pytest.approx(WATER_GFN2_E, abs=1e-6)
    assert r.symbols == WATER_SYMBOLS


def test_build_verification_passes_on_this_box():
    """If this fails, every optimized geometry in the campaign is suspect."""
    ok, err = verify_xtb_build(force=True)
    assert ok, f"xtb build check failed: {err}"


def test_relax_moves_a_distorted_geometry_onto_the_gfn2_minimum():
    distorted = [(0.0, 0.0, 0.0), (1.20, 0.0, 0.0), (0.0, 1.20, 0.0)]
    r = relax(WATER_SYMBOLS, distorted)
    assert r.ok, r.error
    assert r.coords_angstrom is not None
    o, h1, h2 = r.coords_angstrom
    r1, r2 = math.dist(o, h1), math.dist(o, h2)
    assert r1 == pytest.approx(0.9589, abs=0.02)
    assert r2 == pytest.approx(0.9589, abs=0.02)
    # Relaxing must LOWER the energy -- the -O3 miscompile drove H2 uphill.
    sp = singlepoint(WATER_SYMBOLS, distorted)
    assert r.energy < sp.energy


def test_relax_of_an_already_relaxed_geometry_is_idempotent():
    """The trivial-limit anchor for the relaxation step: re-relaxing a minimum
    must not move it. A non-idempotent relaxation would make a strain energy
    depend on how many times it had been run."""
    first = relax(WATER_SYMBOLS, WATER)
    assert first.ok, first.error
    second = relax(WATER_SYMBOLS, first.coords_angstrom)
    assert second.ok, second.error
    assert second.energy == pytest.approx(first.energy, abs=1e-6)
    for a, b in zip(first.coords_angstrom, second.coords_angstrom):
        assert math.dist(a, b) < 1e-3


# ── the point-charge invariant ──

def test_point_charges_actually_change_the_energy():
    """THE test for silently-ignored point charges.

    A +2 charge 3 Bohr from the oxygen must shift the GFN2 energy by tens of
    kcal/mol. If the `pcharge` file were being ignored, in-field and vacuum
    energies would be exactly equal and every binding/strain number computed in
    a field would silently be a vacuum number.
    """
    vac = singlepoint(WATER_SYMBOLS, WATER)
    field = singlepoint(WATER_SYMBOLS, WATER, point_charges=[(2.0, 0.0, 0.0, 3.0)])
    assert vac.ok and field.ok, (vac.error, field.error)
    shift_kcal = (field.energy - vac.energy) * HARTREE_TO_KCAL_MOL
    assert abs(shift_kcal) > 5.0, (
        f"a +2 point charge 3 Bohr away moved the energy by only "
        f"{shift_kcal:.3f} kcal/mol -- the pcharge file is almost certainly "
        "being ignored, which would make every in-field number a vacuum number"
    )


def test_zero_point_charges_reproduce_the_vacuum_energy_exactly():
    """The trivial limit of the embedding: a zero-magnitude charge must be a
    no-op. Non-zero here would mean the field machinery perturbs the vacuum
    reference, invalidating every DIFFERENCE computed against it."""
    vac = singlepoint(WATER_SYMBOLS, WATER)
    zero = singlepoint(WATER_SYMBOLS, WATER, point_charges=[(0.0, 0.0, 0.0, 3.0)])
    assert vac.ok and zero.ok
    assert zero.energy == pytest.approx(vac.energy, abs=1e-8)


def test_point_charge_sign_flips_the_shift():
    """Reachability/polarity: a -2 charge must shift the opposite way from +2.
    A magnitude-only bug (e.g. dropping the sign when writing the file) would
    pass the 'charges do something' test above but be physically wrong."""
    vac = singlepoint(WATER_SYMBOLS, WATER).energy
    plus = singlepoint(WATER_SYMBOLS, WATER, point_charges=[(2.0, 0.0, 0.0, 3.0)]).energy
    minus = singlepoint(WATER_SYMBOLS, WATER, point_charges=[(-2.0, 0.0, 0.0, 3.0)]).energy
    assert (plus - vac) * (minus - vac) < 0, (
        f"+2 shift {plus - vac:.6f} and -2 shift {minus - vac:.6f} have the "
        "same sign -- the charge sign is being dropped"
    )


# ── honest failure ──

def test_bad_geometry_reports_an_error_and_no_energy():
    """Two atoms on top of each other must fail with an error, not return 0.0."""
    r = singlepoint(["O", "O"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    assert r.energy is None or not r.ok, (
        "coincident atoms produced a usable energy"
    )
    if not r.ok:
        assert r.error is not None
        assert r.energy is None, "a failed run must report energy=None, never 0.0"


def test_missing_binary_is_reported_not_raised(monkeypatch):
    """If xtb disappears, callers must get an error object, not an exception --
    a batch over 20 analogues must not abort on it."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    r = singlepoint(WATER_SYMBOLS, WATER)
    assert not r.ok
    assert r.energy is None
    assert "not on PATH" in (r.error or "")


def test_relax_refuses_to_run_when_the_build_check_fails(monkeypatch):
    """The safety interlock: a miscompiled build must block optimizations
    rather than silently produce uphill geometries."""
    import tools.campaign.xtb_engine as eng

    monkeypatch.setattr(eng, "verify_xtb_build", lambda: (False, "simulated bad build"))
    r = eng.relax(WATER_SYMBOLS, WATER)
    assert not r.ok
    assert r.energy is None
    assert "refusing to optimize" in (r.error or "")
    assert "simulated bad build" in (r.error or "")


# ── annealing (MD pose search) ──
#
# `anneal` exists because geometry OPTIMIZATION cannot fix a torsional error.
# Measured on danuglipron: relaxing a generated pose improved its RMSD to the
# bound pose by only 0.04-0.16 A, against a 0.3-2.1 A gap to the success bar,
# because relaxation settles bonds and angles and stays in its starting basin.
# MD at elevated temperature crosses torsional barriers, so it can reach a
# different basin -- which is the only mechanism here that can.

def test_anneal_returns_multiple_distinct_frames():
    frames, run = anneal(WATER_SYMBOLS, WATER, picoseconds=0.3,
                         dump_every_fs=50.0, skip_build_check=True)
    assert run.ok, run.error
    assert run.n_frames == len(frames) >= 2
    assert all(len(f) == len(WATER_SYMBOLS) for f in frames)
    # Frames must actually differ -- identical frames would mean the dynamics
    # never ran and we are just re-reading the input geometry.
    assert frames[0] != frames[-1], "trajectory frames are identical; MD did not move"


def test_anneal_frames_are_physical():
    """Hot MD frames are distorted, but not broken: bonds must stay bonds."""
    frames, run = anneal(WATER_SYMBOLS, WATER, picoseconds=0.3,
                         dump_every_fs=50.0, skip_build_check=True)
    assert run.ok, run.error
    for f in frames:
        oh = math.dist(f[0], f[1])
        assert 0.7 < oh < 1.4, f"O-H {oh:.3f} A is not a bond -- frame is broken"


def test_anneal_respects_the_pocket_field():
    """THE reason to anneal in-pocket rather than in vacuum: the receptor must
    bias which torsions get populated. If the field were ignored during MD (as
    opposed to during single points), in-pocket annealing would be no better
    than free-solution conformer generation -- the exact failure it exists to
    fix.
    """
    vac, rv = anneal(WATER_SYMBOLS, WATER, picoseconds=0.3, dump_every_fs=50.0,
                     skip_build_check=True)
    fld, rf = anneal(WATER_SYMBOLS, WATER, picoseconds=0.3, dump_every_fs=50.0,
                     point_charges=[(-2.0, 0.0, 0.0, 4.0)], skip_build_check=True)
    assert rv.ok and rf.ok, (rv.error, rf.error)
    assert vac[-1] != fld[-1], (
        "a -2 point charge 4 Bohr away did not change the trajectory -- the "
        "pcharge file is being ignored during MD, so in-pocket annealing would "
        "silently be vacuum annealing"
    )


def test_anneal_refuses_without_a_build_check(monkeypatch):
    """Same interlock as `relax`: MD uses GRADIENTS, which are what the
    gfortran -O3 miscompile breaks while leaving energies intact."""
    import tools.campaign.xtb_engine as eng

    monkeypatch.setattr(eng, "verify_xtb_build", lambda: (False, "simulated bad build"))
    frames, run = eng.anneal(WATER_SYMBOLS, WATER, picoseconds=0.1)
    assert frames == []
    assert not run.ok
    assert "refusing to run MD" in (run.error or "")


def test_anneal_reports_a_missing_binary_rather_than_raising(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    frames, run = anneal(WATER_SYMBOLS, WATER, picoseconds=0.1,
                         skip_build_check=True)
    assert frames == [] and not run.ok
    assert "not on PATH" in (run.error or "")


def test_anneal_run_never_reports_ok_without_frames():
    """The contract `XtbRun` could not express, and which the first version of
    `anneal` violated: ok=False with error=None. AnnealRun must always pair a
    failure with a reason."""
    from tools.campaign.xtb_engine import AnnealRun

    bad = AnnealRun(0, False, None)
    assert not bad.ok
    # An ok run must have frames and no error.
    good = AnnealRun(5, True, None)
    assert good.ok and good.n_frames == 5 and good.error is None


def test_trajectory_parser_handles_a_truncated_final_frame(tmp_path):
    """xtb can be killed mid-write. A half-frame must be dropped, not returned
    with missing atoms -- a short coordinate list would silently misalign every
    downstream per-atom operation."""
    from tools.campaign.xtb_engine import _read_trajectory

    trj = tmp_path / "t.trj"
    frame = "3\n energy: -5.0\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
    trj.write_text(frame + frame + "3\n energy: -5.0\nO 0.0 0.0 0.0\n")
    frames = _read_trajectory(trj, 3)
    assert len(frames) == 2, f"expected the truncated frame to be dropped, got {len(frames)}"
    assert all(len(f) == 3 for f in frames)
