from pathlib import Path

import pytest

from tools.active_site.ligand_embedding import embed_ligand_from_coords
from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR, PocketCharges
from tools.active_site.prescreen import batch_prescreen, prescreen_pose

WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = [(0.0, 0.0, 0.117790), (0.0, 0.755453, -0.471161), (0.0, -0.755453, -0.471161)]
WATER_XYZ = "testdata/molecules/water.xyz"
METHANE_XYZ = "testdata/molecules/methane.xyz"


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
    # H with a -1 formal charge = hydride, 2 electrons, singlet. The charge
    # must be passed to the embedding too, not just to `atom_charges`: a
    # neutral H is 1 electron and CANNOT be a singlet, so the SCF setup rejects
    # it before the prescreen is ever reached.
    embedded = embed_ligand_from_coords(
        ["H"], [(0.0, 0.0, 0.0)], pocket=pocket, basis="sto-3g", charge=-1
    )
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


def _formal_charges_by_natoms(embedded) -> list[float]:
    # Toy charge source keyed only on atom count, standing in for a real
    # force-field/QM charge lookup in these tests.
    if embedded.mol.natoms() == 3:
        return [-0.8, 0.4, 0.4]  # water-shaped
    if embedded.mol.natoms() == 5:
        return [-0.4, 0.1, 0.1, 0.1, 0.1]  # methane-shaped
    raise ValueError(f"no toy charges for {embedded.mol.natoms()}-atom ligand")


def test_batch_prescreen_ranks_ascending_by_score():
    # Two pockets at different distances from the origin -> two poses (same
    # geometry, water and methane both centered near the origin) get
    # different scores purely from which xyz is which; what matters here is
    # that batch_prescreen returns them SORTED by score ascending, ranked
    # 1-based, not in input order.
    pocket = PocketCharges(
        charges=[(-1.0, 5.0 * ANGSTROM_TO_BOHR, 0.0, 0.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    ranked = batch_prescreen(
        pocket, [METHANE_XYZ, WATER_XYZ], charge_source=_formal_charges_by_natoms, basis="sto-3g",
    )
    assert len(ranked) == 2
    assert all(e.error is None for e in ranked)
    assert [e.rank for e in ranked] == [1, 2]
    assert ranked[0].result.score <= ranked[1].result.score


def test_batch_prescreen_isolates_per_pose_failures():
    # A charge_source that raises for one specific pose must not abort the
    # whole batch -- the other pose still gets scored and ranked, and the
    # failed one is reported with .error set, sorted to the end.
    pocket = PocketCharges(
        charges=[(1.0, 10.0, 10.0, 10.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
    )

    def flaky_charge_source(embedded):
        if embedded.mol.natoms() == 5:
            raise RuntimeError("simulated charge-derivation failure for methane")
        return _formal_charges_by_natoms(embedded)

    ranked = batch_prescreen(
        pocket, [WATER_XYZ, METHANE_XYZ], charge_source=flaky_charge_source, basis="sto-3g",
    )
    assert len(ranked) == 2
    ok = [e for e in ranked if e.error is None]
    failed = [e for e in ranked if e.error is not None]
    assert len(ok) == 1 and ok[0].ligand_xyz == Path(WATER_XYZ)
    assert len(failed) == 1 and failed[0].ligand_xyz == Path(METHANE_XYZ)
    assert "simulated charge-derivation failure" in failed[0].error
    assert failed[0].rank is None
    # failed entries sort after all successful ones
    assert ranked[-1] is failed[0]


def test_batch_prescreen_missing_file_is_a_per_pose_failure_not_a_crash():
    pocket = PocketCharges(
        charges=[(1.0, 10.0, 10.0, 10.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    ranked = batch_prescreen(
        pocket, [WATER_XYZ, "testdata/molecules/does_not_exist.xyz"],
        charge_source=_formal_charges_by_natoms, basis="sto-3g",
    )
    assert len(ranked) == 2
    assert sum(1 for e in ranked if e.error is not None) == 1
