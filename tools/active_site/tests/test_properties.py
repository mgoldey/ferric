from tools.active_site.energy import compute_energy
from tools.active_site.ligand_embedding import embed_ligand
from tools.active_site.properties import compute_charges

WATER_XYZ = "testdata/molecules/water.xyz"


def test_compute_charges_rhf():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    result = compute_energy(embedded, method="rhf")
    charges = compute_charges(embedded, result)
    assert len(charges["hirshfeld"]) == 3
    assert len(charges["lowdin"]) == 3


def test_compute_charges_dft():
    embedded = embed_ligand(WATER_XYZ, basis="sto-3g")
    result = compute_energy(embedded, method="dft", xc="LDA")
    charges = compute_charges(embedded, result)
    assert len(charges["hirshfeld"]) == 3
    assert len(charges["lowdin"]) == 3
