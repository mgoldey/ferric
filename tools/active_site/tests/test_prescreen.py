from pathlib import Path

import pytest

from tools.active_site.ligand_embedding import embed_ligand_from_coords
from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR, PocketCharges
from tools.active_site.prescreen import prescreen_pose

WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = [(0.0, 0.0, 0.117790), (0.0, 0.755453, -0.471161), (0.0, -0.755453, -0.471161)]


def test_prescreen_pose_no_pocket_raises():
    embedded = embed_ligand_from_coords(WATER_SYMBOLS, WATER_COORDS, basis="sto-3g")
    with pytest.raises(ValueError, match="point_charges"):
        prescreen_pose(embedded, atom_charges=[0.0, 0.0, 0.0])


def test_prescreen_pose_charge_count_mismatch_raises():
    pocket = PocketCharges(
        charges=[(1.0, 10.0, 10.0, 10.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(WATER_SYMBOLS, WATER_COORDS, pocket=pocket, basis="sto-3g")
    with pytest.raises(ValueError, match="atom_charges has"):
        prescreen_pose(embedded, atom_charges=[0.0, 0.0])  # only 2, need 3


def test_prescreen_pose_zero_charges_gives_zero_score():
    # Formal charges all zero -> score (a linear functional of them) is
    # exactly zero regardless of the pocket field, independent of geometry.
    pocket = PocketCharges(
        charges=[(1.0, 10.0, 10.0, 10.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(WATER_SYMBOLS, WATER_COORDS, pocket=pocket, basis="sto-3g")
    result = prescreen_pose(embedded, atom_charges=[0.0, 0.0, 0.0])
    assert result.score == pytest.approx(0.0, abs=1e-12)
    assert result.field_at_atoms.shape == (3, 4)
    assert result.n_pocket_charges == 1


def test_prescreen_pose_score_matches_hand_calc_single_charge():
    # +1 pocket charge far along +x from a single-atom "ligand" at the
    # origin with formal charge -1: score = q_ligand * q_pocket / r (a.u.),
    # i.e. the classical Coulomb interaction energy, sign included.
    r_angstrom = 5.0
    pocket = PocketCharges(
        charges=[(1.0, r_angstrom * ANGSTROM_TO_BOHR, 0.0, 0.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(["H"], [(0.0, 0.0, 0.0)], pocket=pocket, basis="sto-3g")
    result = prescreen_pose(embedded, atom_charges=[-1.0])
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    expected = -1.0 * (1.0 / r_bohr)
    assert result.score == pytest.approx(expected, rel=1e-10)


def test_prescreen_pose_uses_filtered_not_raw_pocket_charges():
    # A pocket charge that overlaps the ligand (and is therefore filtered
    # out of embedded.point_charges by embed_ligand_from_coords) must NOT
    # contribute to the prescreen score -- the whole point is to match what
    # the QM field embedding would actually see.
    overlapping = (5.0, 0.0, 0.0, 0.117790 * ANGSTROM_TO_BOHR)  # sits on the O atom
    far_away = (1.0, 100.0, 100.0, 100.0)
    pocket = PocketCharges(
        charges=[overlapping, far_away], source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(
        WATER_SYMBOLS, WATER_COORDS, pocket=pocket, basis="sto-3g", overlap_cutoff_angstrom=0.1,
    )
    assert len(embedded.point_charges) == 1  # only far_away survives
    result = prescreen_pose(embedded, atom_charges=[-0.8, 0.4, 0.4])
    assert result.n_pocket_charges == 1
