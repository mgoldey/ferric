from pathlib import Path

import pytest

from tools.active_site.ligand_embedding import embed_ligand_from_coords
from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR, PocketCharges
from tools.active_site.pose_relaxation import relax_pose_in_pocket_field

H2_SYMBOLS = ["H", "H"]
H2_COORDS = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.8)]  # slightly off equilibrium (~0.74 A)
WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = [(0.0, 0.0, 0.117790), (0.0, 0.755453, -0.471161), (0.0, -0.755453, -0.471161)]


def test_relax_pose_no_pocket_raises():
    embedded = embed_ligand_from_coords(H2_SYMBOLS, H2_COORDS, basis="sto-3g")
    with pytest.raises(ValueError, match="point_charges"):
        relax_pose_in_pocket_field(embedded)


def test_relax_pose_pocket_with_no_surviving_charges_raises():
    # A pocket charge that overlaps every ligand atom is entirely filtered
    # out by embed_ligand_from_coords -- embedded.point_charges ends up an
    # empty list (pocket is not None, but nothing survives), which must be
    # treated the same as "no pocket" by relax_pose_in_pocket_field.
    pocket = PocketCharges(
        charges=[(1.0, 0.0, 0.0, 0.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(
        H2_SYMBOLS, H2_COORDS, pocket=pocket, basis="sto-3g", overlap_cutoff_angstrom=5.0,
    )
    assert embedded.point_charges == []
    with pytest.raises(ValueError, match="point_charges"):
        relax_pose_in_pocket_field(embedded)


def test_relax_pose_h2_in_synthetic_pocket_field_real_optimization():
    # Real end-to-end call into ferric.run_optimize: a slightly-stretched H2
    # (0.8 A vs ~0.74 A equilibrium) next to a synthetic +1 point charge far
    # enough away to be a weak perturbation, not a dissociating one. Small
    # basis (sto-3g) and a 2-atom system keep this fast.
    r_angstrom = 6.0
    pocket = PocketCharges(
        charges=[(1.0, r_angstrom * ANGSTROM_TO_BOHR, 0.0, 0.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(H2_SYMBOLS, H2_COORDS, pocket=pocket, basis="sto-3g")
    assert embedded.point_charges == pocket.charges  # far away -> nothing filtered

    relaxed = relax_pose_in_pocket_field(embedded, max_steps=50)

    assert relaxed.converged is True
    assert relaxed.steps >= 1
    assert relaxed.symbols == H2_SYMBOLS
    assert relaxed.n_pocket_charges == 1
    # The relaxed geometry comes back in Angstrom: the deliberately stretched
    # 0.8 A bond must have shortened toward the STO-3G equilibrium (~0.71 A).
    assert len(relaxed.coords_angstrom) == 2
    import math

    d = math.dist(relaxed.coords_angstrom[0], relaxed.coords_angstrom[1])
    assert 0.70 < d < 0.75, d
    # Sanity: the in-field relaxed H2 energy should land close to isolated
    # H2/sto-3g's equilibrium energy (~-1.1175 Ha) -- the point charge here
    # is a weak, distant perturbation, not a dissociating one.
    assert relaxed.energy == pytest.approx(-1.1175, abs=5e-3)


def test_relax_pose_energy_lower_than_unrelaxed_start():
    # The optimizer must actually improve on the (deliberately stretched)
    # starting geometry's energy -- otherwise this isn't testing that
    # ferric.run_optimize is doing real work, just that it returns *a*
    # number. Compare against a single-point in-field energy at the
    # unrelaxed starting geometry.
    from tools.active_site.energy import compute_energy

    pocket = PocketCharges(
        charges=[(1.0, 6.0 * ANGSTROM_TO_BOHR, 0.0, 0.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    embedded = embed_ligand_from_coords(H2_SYMBOLS, H2_COORDS, pocket=pocket, basis="sto-3g")

    unrelaxed = compute_energy(embedded, method="rhf", use_field=True)
    relaxed = relax_pose_in_pocket_field(embedded, max_steps=50)

    assert relaxed.converged is True
    assert relaxed.energy < unrelaxed.energy
