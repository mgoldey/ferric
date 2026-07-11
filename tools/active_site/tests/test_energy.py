from pathlib import Path

from tools.active_site.energy import compute_energy
from tools.active_site.ligand_embedding import embed_ligand
from tools.active_site.pocket_charges import PocketCharges

WATER_XYZ = "testdata/molecules/water.xyz"


def test_compute_energy_vacuum_no_pocket():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    result = compute_energy(embedded, method="rhf")
    assert result.converged
    assert result.field is False


def test_compute_energy_use_field_false_forces_vacuum():
    pocket = PocketCharges(charges=[(1.0, 0.0, 0.0, 10.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    embedded = embed_ligand(WATER_XYZ, pocket=pocket, basis="sto-3g")
    vac = compute_energy(embedded, use_field=False)
    field = compute_energy(embedded, use_field=True)
    assert vac.field is False
    assert field.field is True
    assert vac.energy != field.energy


def test_compute_energy_dft_requires_xc():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    try:
        compute_energy(embedded, method="dft")
        assert False, "should have raised"
    except ValueError:
        pass


def test_compute_energy_unknown_method_raises():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    try:
        compute_energy(embedded, method="bogus")
        assert False, "should have raised"
    except ValueError:
        pass
