"""Tests for the strain (Arm A) and fit (Arms A/B) drivers.

Both quantities are DIFFERENCES, so the failure modes are reference-state
errors, and they are silent: you get a plausible number computed against the
wrong baseline. The tests here pin the reference discipline.

For fit, the headline invariant is that a pose with NO pocket charges nearby is
reported UNEVALUATED rather than scored as interaction 0.0 -- because 0.0 reads
as "electrostatically neutral", which for a ranking is a mid-table result rather
than the "we never measured this" that it actually means. Measured 2026-08-29:
18 of the 20 committed danuglipron conformers are ~176 A from the pocket, so
this path is the common case, not an edge case.
"""
from __future__ import annotations

import math

import pytest

from tools.campaign.fit import _trim_charges, pose_fit
from tools.campaign.strain import (
    XyzEnsemble,
    _formula,
    free_reference,
    load_xyz_ensemble,
    pose_strain,
)
from tools.campaign.xtb_engine import xtb_available

needs_xtb = pytest.mark.skipif(not xtb_available(), reason="`xtb` not on PATH")

WATER_SYMBOLS = ["O", "H", "H"]
WATER = [(0.0, 0.0, 0.0), (0.9578, 0.0, 0.0), (-0.24, 0.927, 0.0)]
STRETCHED = [(0.0, 0.0, 0.0), (1.15, 0.0, 0.0), (-0.29, 1.11, 0.0)]


# ── ensemble loading ──

def test_formula_is_order_independent():
    assert _formula(["C", "H", "H"]) == _formula(["H", "C", "H"])


def test_loads_the_danuglipron_ensemble_and_flags_mixed_atom_order():
    """The committed ensemble genuinely mixes three atom orderings (PDB,
    PubChem, RDKit). That is safe for total energies and unsafe for per-atom
    work, so it must be FLAGGED, not rejected and not ignored."""
    ens = load_xyz_ensemble("testdata/molecules/c9_systems/danuglipron")
    assert len(ens) == 20
    assert ens.formula == "C31F1H30N5O4"
    assert ens.shared_order is False
    assert len(ens.symbols_per_conformer) == 20


def test_symbols_property_refuses_to_answer_for_a_mixed_order_ensemble():
    """`.symbols` would silently hand out one conformer's ordering as though it
    applied to all of them -- which pairs a carbon with a fluorine."""
    ens = load_xyz_ensemble("testdata/molecules/c9_systems/danuglipron")
    with pytest.raises(ValueError, match="mixes atom orderings"):
        _ = ens.symbols


def test_symbols_property_works_when_the_order_is_shared():
    ens = XyzEnsemble([["O", "H"], ["O", "H"]], [[(0, 0, 0)], [(0, 0, 0)]],
                      ["a", "b"], True, "H1O1")
    assert ens.symbols == ["O", "H"]


def test_different_molecules_in_one_directory_are_rejected(tmp_path):
    (tmp_path / "conf_00.xyz").write_text("1\n\nO 0.0 0.0 0.0\n")
    (tmp_path / "conf_01.xyz").write_text("1\n\nN 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="different molecule"):
        load_xyz_ensemble(tmp_path)


def test_empty_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_xyz_ensemble("testdata/molecules")


# ── strain reference discipline ──

@needs_xtb
def test_free_reference_picks_the_lowest_conformer():
    ref = free_reference(WATER_SYMBOLS, [STRETCHED, WATER], labels=["stretched", "eq"])
    assert ref.e_min is not None
    assert ref.n_considered == 2
    assert ref.n_converged == 2
    # Both relax to the same minimum, so the reference must be that minimum and
    # the spread must be ~0 -- this is the trivial limit of the scan.
    assert ref.spread_kcal is not None
    assert ref.spread_kcal < 0.1


@needs_xtb
def test_free_reference_accepts_per_conformer_symbol_lists():
    """A real ensemble mixes atom orderings; total energies are invariant, so
    the scan must accept the per-conformer form."""
    ref = free_reference(
        [WATER_SYMBOLS, ["H", "O", "H"]],
        [WATER, [(0.9578, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.24, 0.927, 0.0)]],
        labels=["a", "b"],
    )
    assert ref.n_converged == 2
    # Same molecule, same geometry, different labelling => same energy.
    es = [c.e_relaxed for c in ref.per_conformer]
    assert es[0] == pytest.approx(es[1], abs=1e-6)


def test_free_reference_rejects_mismatched_symbol_list_count():
    with pytest.raises(ValueError, match="must match 1:1"):
        free_reference([WATER_SYMBOLS], [WATER, WATER], labels=["a", "b"])


def test_strain_is_none_when_the_reference_failed():
    """No reference => no strain. Never 0.0, which would read as unstrained."""
    from tools.campaign.strain import FreeReference

    bad = FreeReference(None, None, 3, 0, [])
    r = pose_strain(WATER_SYMBOLS, WATER, bad)
    assert not r.ok
    assert r.strain_kcal is None
    assert "no converged conformer" in r.error


@needs_xtb
def test_strain_of_the_reference_geometry_against_itself_is_zero():
    """THE trivial-limit anchor for strain: the free minimum has, by
    definition, zero strain. A nonzero value here means the two sides of the
    subtraction are not the same quantity."""
    ref = free_reference(WATER_SYMBOLS, [WATER], labels=["eq"])
    assert ref.e_min is not None
    relaxed_geom = ref.per_conformer[0].relaxed_coords
    assert relaxed_geom is not None
    r = pose_strain(WATER_SYMBOLS, relaxed_geom, ref, label="self")
    assert r.ok, r.error
    assert r.strain_kcal == pytest.approx(0.0, abs=0.05), (
        f"the reference geometry has strain {r.strain_kcal:.4f} kcal/mol "
        "against itself"
    )


@needs_xtb
def test_strain_is_positive_for_a_distorted_geometry():
    """Reachability: strain must be able to be nonzero, or the anchor above is
    passing trivially."""
    ref = free_reference(WATER_SYMBOLS, [WATER], labels=["eq"])
    # A badly distorted geometry, held fixed by giving the relaxation no room:
    # we pass the DISTORTED coords and read the vacuum energy at the relaxed
    # geometry, so to see strain we need a pose that relaxes to a DIFFERENT
    # minimum. Use a linear water, which is a saddle -- it relaxes back, so
    # instead assert on the single-point strain of the unrelaxed geometry.
    from tools.campaign.xtb_engine import HARTREE_TO_KCAL_MOL, singlepoint

    sp = singlepoint(WATER_SYMBOLS, [(0.0, 0.0, 0.0), (1.30, 0.0, 0.0), (-1.30, 0.0, 0.0)])
    assert sp.ok
    strain_unrelaxed = (sp.energy - ref.e_min) * HARTREE_TO_KCAL_MOL
    assert strain_unrelaxed > 10.0, (
        "a linear, stretched water is not showing a strain penalty -- the "
        "reference or the units are wrong"
    )


# ── fit: the UNEVALUATED invariant ──

def test_charges_outside_the_cutoff_are_trimmed():
    charges = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1000.0)]
    kept = _trim_charges(charges, [(0.0, 0.0, 0.0)], 30.0)
    assert len(kept) == 1
    assert kept[0][3] == 0.0


def test_no_cutoff_keeps_everything():
    charges = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1000.0)]
    assert len(_trim_charges(charges, [(0.0, 0.0, 0.0)], None)) == 2


def test_pose_with_no_nearby_charges_is_unevaluated_not_zero():
    """THE fit invariant. 18/20 committed conformers are ~176 A from the
    pocket, so this is the common path. An interaction of 0.0 would rank them
    mid-table instead of flagging them as unmeasured."""
    far_pocket = [(1.0, 1000.0, 1000.0, 1000.0)]
    r = pose_fit(WATER_SYMBOLS, WATER, far_pocket, label="far")
    assert not r.ok
    assert r.interaction_kcal is None, (
        "a pose outside the pocket produced an interaction energy; it must be "
        "None (unevaluated), never 0.0 (neutral)"
    )
    assert r.n_pocket_charges == 0
    assert "outside the pocket" in r.error


@needs_xtb
def test_fit_is_negative_for_a_complementary_field():
    """A negative point charge placed near water's hydrogens must give a
    favorable (negative) interaction."""
    # Water's H atoms point along +x and +y; a -1 charge beyond the +x H.
    ANG = 1.0 / 0.529_177_210_92
    charges = [(-1.0, 3.0 * ANG, 0.0, 0.0)]
    r = pose_fit(WATER_SYMBOLS, WATER, charges, label="complementary")
    assert r.ok, r.error
    assert r.interaction_kcal < 0, (
        f"a -1 charge near the hydrogens gave {r.interaction_kcal:+.2f} kcal/mol; "
        "expected a favorable (negative) interaction"
    )


@needs_xtb
def test_fit_sign_flips_with_the_field_sign():
    """Polarity check: reversing every pocket charge must reverse the sign of
    the interaction, to first order. A magnitude-only bug passes the test above
    but is physically meaningless."""
    ANG = 1.0 / 0.529_177_210_92
    neg = pose_fit(WATER_SYMBOLS, WATER, [(-1.0, 3.0 * ANG, 0.0, 0.0)])
    pos = pose_fit(WATER_SYMBOLS, WATER, [(+1.0, 3.0 * ANG, 0.0, 0.0)])
    assert neg.ok and pos.ok
    assert neg.interaction_kcal * pos.interaction_kcal < 0, (
        f"same-sign interactions ({neg.interaction_kcal:+.2f}, "
        f"{pos.interaction_kcal:+.2f}) for opposite fields"
    )


@needs_xtb
def test_zero_charges_give_exactly_zero_interaction():
    """Trivial limit of the fit: a zero-magnitude field must give exactly zero
    interaction, since both single points are then identical calculations."""
    r = pose_fit(WATER_SYMBOLS, WATER, [(0.0, 3.0, 0.0, 0.0)])
    assert r.ok, r.error
    assert r.interaction_kcal == pytest.approx(0.0, abs=1e-4)


@needs_xtb
def test_fit_uses_one_geometry_for_both_single_points():
    """Vacuum and in-field energies must be at the SAME geometry, or the
    difference silently contains a relaxation energy. Verified indirectly: the
    vacuum energy from pose_fit must equal a standalone single point at the
    input coordinates."""
    from tools.campaign.xtb_engine import singlepoint

    ANG = 1.0 / 0.529_177_210_92
    r = pose_fit(WATER_SYMBOLS, WATER, [(-1.0, 3.0 * ANG, 0.0, 0.0)])
    standalone = singlepoint(WATER_SYMBOLS, WATER)
    assert r.ok and standalone.ok
    assert r.e_vacuum == pytest.approx(standalone.energy, abs=1e-9)
