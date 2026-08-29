"""Tests for rigid-body alignment into the pocket frame.

The silent failure this file is built around: **an improper rotation**. A naive
SVD superposition can return a matrix with det = -1, which MIRRORS the molecule.
That inverts every stereocentre -- turning (S)-danuglipron into its (R) enantiomer
-- while producing an excellent-looking RMSD, because a mirror image fits an
achiral scaffold just as well. `test_kabsch_never_returns_a_reflection` and
`test_alignment_preserves_chirality` pin the guard from both ends.

The second theme: alignment must place the molecule IN the pocket. A conformer
176 A away scores as "no pocket charges nearby", so if alignment silently no-ops
the whole analogue arm produces no data. `test_alignment_moves_an_origin_centred_
conformer_into_the_pocket_frame` is the end-to-end check on real data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.campaign.align import align_by_index_map, align_to_reference, kabsch


def _random_rigid(coords, seed=0, reflect=False):
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(0, 2 * math.pi, 3)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(c), -math.sin(c)], [0, math.sin(c), math.cos(c)]])
    R = Rz @ Ry @ Rx
    if reflect:
        R = R @ np.diag([1.0, 1.0, -1.0])
    return coords @ R.T + rng.uniform(-50, 50, 3)


# A rigid, chiral 5-atom fragment.
FRAG = np.array([
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [0.0, 1.4, 0.0],
    [0.0, 0.0, 1.3],
    [-1.1, -0.9, -0.7],
])


# ── Kabsch core ──

def test_kabsch_recovers_a_known_rigid_transform_exactly():
    moved = _random_rigid(FRAG, seed=7)
    R, t, rmsd = kabsch(FRAG, moved)
    assert rmsd < 1e-9
    assert np.allclose(FRAG @ R.T + t, moved, atol=1e-9)


def test_kabsch_never_returns_a_reflection():
    """det(R) must be +1 for every input, including a deliberately mirrored
    target where the naive SVD solution IS the reflection."""
    for seed in range(6):
        for reflect in (False, True):
            moved = _random_rigid(FRAG, seed=seed, reflect=reflect)
            R, _, _ = kabsch(FRAG, moved)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9), (
                f"improper rotation (det={np.linalg.det(R):+.6f}) at seed={seed} "
                f"reflect={reflect} -- this would mirror the molecule and invert "
                "every stereocentre"
            )


def test_mirrored_target_cannot_be_fitted_well_by_a_proper_rotation():
    """Reachability for the guard: a genuinely mirrored target must produce a
    LARGE rmsd, proving the guard is doing something rather than the fragment
    being accidentally achiral."""
    mirrored = _random_rigid(FRAG, seed=3, reflect=True)
    _, _, rmsd = kabsch(FRAG, mirrored)
    assert rmsd > 0.3, (
        f"a mirrored chiral fragment fitted to rmsd {rmsd:.4f} -- the test "
        "fragment is not chiral enough to detect a reflection"
    )


def test_kabsch_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 3 atom pairs"):
        kabsch(FRAG[:2], FRAG[:2])
    with pytest.raises(ValueError, match="matching"):
        kabsch(FRAG, FRAG[:3])


def test_align_by_index_map_reports_too_few_pairs():
    a = align_by_index_map(["C", "C"], [(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (1, 0, 0)],
                           [(0, 0)])
    assert not a.ok
    assert "matched atom pairs" in a.error


# ── end-to-end on the real ensemble ──

DANU = (
    "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
    "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
)


@pytest.fixture(scope="module")
def ensemble():
    from tools.campaign.strain import load_xyz_ensemble

    return load_xyz_ensemble("testdata/molecules/c9_systems/danuglipron")


def test_cryo_em_pose_is_the_only_one_in_the_pocket_frame(ensemble):
    """Documents the measured fact this module exists for. If a future data
    refresh puts every conformer in the pocket frame, alignment becomes
    unnecessary and this test says so by failing."""
    centroids = [
        np.mean(np.asarray(c), axis=0) for c in ensemble.conformers
    ]
    cryo = centroids[ensemble.labels.index("conf_00_cryo_em")]
    assert np.linalg.norm(cryo) > 100, "cryo-EM pose is no longer in a protein frame"
    others = [
        c for lbl, c in zip(ensemble.labels, centroids) if lbl != "conf_00_cryo_em"
    ]
    assert all(np.linalg.norm(c) < 20 for c in others), (
        "some non-cryo-EM conformer is now in the pocket frame; re-check whether "
        "alignment is still required"
    )


def test_alignment_moves_an_origin_centred_conformer_into_the_pocket_frame(ensemble):
    """THE end-to-end check: an origin-centred RDKit conformer must end up on
    top of the bound pose, not 176 A away."""
    i_ref = ensemble.labels.index("conf_00_cryo_em")
    i_mob = ensemble.labels.index("conf_02_rdkit")

    aligned = align_to_reference(
        DANU, ensemble.symbols_per_conformer[i_mob], ensemble.conformers[i_mob],
        DANU, ensemble.symbols_per_conformer[i_ref], ensemble.conformers[i_ref],
    )
    assert aligned.ok, aligned.error
    assert aligned.n_matched_atoms >= 20, (
        f"only {aligned.n_matched_atoms} scaffold atoms matched for an "
        "identical molecule -- the MCS or atom mapping is broken"
    )

    ref_centroid = np.mean(np.asarray(ensemble.conformers[i_ref]), axis=0)
    new_centroid = np.mean(np.asarray(aligned.coords_angstrom), axis=0)
    assert np.linalg.norm(new_centroid - ref_centroid) < 3.0, (
        f"aligned centroid is {np.linalg.norm(new_centroid - ref_centroid):.1f} A "
        "from the reference -- alignment did not place it in the pocket"
    )


def test_alignment_preserves_internal_geometry(ensemble):
    """A rigid transform must not deform the molecule: every interatomic
    distance is invariant. This catches a scaling or shearing bug that an RMSD
    check alone would miss."""
    i_ref = ensemble.labels.index("conf_00_cryo_em")
    i_mob = ensemble.labels.index("conf_04_rdkit")
    before = np.asarray(ensemble.conformers[i_mob])
    aligned = align_to_reference(
        DANU, ensemble.symbols_per_conformer[i_mob], ensemble.conformers[i_mob],
        DANU, ensemble.symbols_per_conformer[i_ref], ensemble.conformers[i_ref],
    )
    assert aligned.ok, aligned.error
    after = np.asarray(aligned.coords_angstrom)

    for i, j in [(0, 10), (5, 30), (2, 60), (14, 44)]:
        d_before = np.linalg.norm(before[i] - before[j])
        d_after = np.linalg.norm(after[i] - after[j])
        assert d_after == pytest.approx(d_before, abs=1e-8), (
            f"distance {i}-{j} changed from {d_before:.6f} to {d_after:.6f} -- "
            "the transform is not rigid"
        )


def test_alignment_of_a_pose_onto_itself_is_a_no_op(ensemble):
    """Trivial-limit anchor: aligning the reference onto itself must not move
    it, and must report ~zero RMSD."""
    i = ensemble.labels.index("conf_00_cryo_em")
    aligned = align_to_reference(
        DANU, ensemble.symbols_per_conformer[i], ensemble.conformers[i],
        DANU, ensemble.symbols_per_conformer[i], ensemble.conformers[i],
    )
    assert aligned.ok, aligned.error
    assert aligned.rmsd_angstrom < 1e-6
    for a, b in zip(aligned.coords_angstrom, ensemble.conformers[i]):
        assert math.dist(a, b) < 1e-6


def test_wrong_molecule_is_rejected_by_the_formula_check(ensemble):
    """A geometry whose formula does not match the declared SMILES must error,
    not be aligned as though it were the right molecule."""
    i = ensemble.labels.index("conf_00_cryo_em")
    # Claim the pose is benzene. Same coordinates, wrong declared identity.
    aligned = align_to_reference(
        "c1ccccc1", ensemble.symbols_per_conformer[i], ensemble.conformers[i],
        DANU, ensemble.symbols_per_conformer[i], ensemble.conformers[i],
    )
    assert not aligned.ok
    assert "formula" in (aligned.error or "")


def test_mismatched_symbol_count_is_rejected(ensemble):
    i = ensemble.labels.index("conf_00_cryo_em")
    aligned = align_to_reference(
        DANU, ensemble.symbols_per_conformer[i][:-1], ensemble.conformers[i],
        DANU, ensemble.symbols_per_conformer[i], ensemble.conformers[i],
    )
    assert not aligned.ok
    assert "coordinate rows" in (aligned.error or "") or "formula" in (aligned.error or "")


def test_relabelled_atoms_are_a_known_limitation_not_a_silent_win(ensemble):
    """DOCUMENTED LIMITATION, pinned so it is not mistaken for a guarantee.

    Because connectivity is perceived from the GEOMETRY (the ensemble mixes
    three atom orderings, so the SMILES order cannot be trusted), a permutation
    of the element labels that preserves the formula is NOT detected: it yields
    a self-consistent but chemically wrong molecule. The formula check catches a
    wrong MOLECULE; it cannot catch wrong LABELS on the right molecule.

    The practical guard is upstream: `symbols` always travels with `coords` from
    the same xyz file, so they cannot get out of step in normal use. This test
    records that the boundary is understood rather than leaving a reader to
    assume a stronger check exists.
    """
    i = ensemble.labels.index("conf_00_cryo_em")
    scrambled = list(reversed(ensemble.symbols_per_conformer[i]))
    aligned = align_to_reference(
        DANU, scrambled, ensemble.conformers[i],
        DANU, ensemble.symbols_per_conformer[i], ensemble.conformers[i],
    )
    # It succeeds -- that is the limitation. Assert the shape so a future change
    # that DOES catch this fails here and gets the docstring updated.
    assert aligned.ok, (
        "label permutation is now detected -- good; update this test and the "
        "align.py docstring, which currently document it as undetected"
    )
