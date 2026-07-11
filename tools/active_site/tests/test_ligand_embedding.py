from pathlib import Path

from tools.active_site.ligand_embedding import embed_ligand, embed_ligand_from_coords
from tools.active_site.pocket_charges import PocketCharges

WATER_XYZ = "testdata/molecules/water.xyz"
WATER_COORDS = [(0.0, 0.0, 0.117790), (0.0, 0.755453, -0.471161), (0.0, -0.755453, -0.471161)]


def test_embed_ligand_no_pocket():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    assert embedded.point_charges is None
    assert embedded.pocket is None
    assert embedded.mol.natoms() == 3


def test_embed_ligand_with_pocket_filters_overlap():
    # A pocket charge sitting exactly at a water atom's own coordinate must
    # be dropped by the ligand-overlap filter.
    pocket = PocketCharges(
        charges=[(1.0, 0.0, 0.0, 0.117790 * 1.8897261245650618), (0.5, 100.0, 100.0, 100.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand(WATER_XYZ, pocket=pocket, basis="sto-3g", overlap_cutoff_angstrom=0.1)
    assert embedded.point_charges is not None
    # the overlapping charge is dropped, the far-away one survives
    assert len(embedded.point_charges) == 1


def test_embed_ligand_from_coords_matches_file():
    from_file = embed_ligand(WATER_XYZ, basis="sto-3g")
    from_coords = embed_ligand_from_coords(["O", "H", "H"], WATER_COORDS, basis="sto-3g")
    assert from_file.mol.natoms() == from_coords.mol.natoms()
    assert from_file.mol.nuclear_repulsion() == from_coords.mol.nuclear_repulsion()
