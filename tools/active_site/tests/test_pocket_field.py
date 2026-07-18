from pathlib import Path

import numpy as np
import pytest

from tools.active_site.pocket_charges import PocketCharges, ANGSTROM_TO_BOHR
from tools.active_site.pocket_field import pocket_field_at_atoms


def test_single_charge_potential_matches_coulomb_law():
    # +1 charge at origin (Bohr); probe site 1 Angstrom away along x.
    pocket = PocketCharges(charges=[(1.0, 0.0, 0.0, 0.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    site_angstrom = [(1.0, 0.0, 0.0)]
    out = pocket_field_at_atoms(pocket, site_angstrom)
    r_bohr = 1.0 * ANGSTROM_TO_BOHR
    assert out.shape == (1, 4)
    assert out[0, 0] == pytest.approx(1.0 / r_bohr, rel=1e-10)
    assert out[0, 1] == pytest.approx(1.0 / r_bohr**2, rel=1e-10)  # Ex points toward +x
    assert out[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert out[0, 3] == pytest.approx(0.0, abs=1e-12)


def test_zero_net_charge_pair_cancels_at_midpoint():
    pocket = PocketCharges(
        charges=[(1.0, -1.0, 0.0, 0.0), (-1.0, 1.0, 0.0, 0.0)],
        source_pdb=Path("fake.pdb"), ff="AMBER",
    )
    out = pocket_field_at_atoms(pocket, [(0.0, 0.0, 0.0)])
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_multiple_sites_shape():
    pocket = PocketCharges(charges=[(0.3, 5.0, 5.0, 5.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    out = pocket_field_at_atoms(pocket, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 0.0, 0.0)])
    assert out.shape == (3, 4)


def test_coincident_site_raises():
    pocket = PocketCharges(charges=[(1.0, 0.0, 0.0, 0.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    with pytest.raises(ValueError, match="coincides"):
        pocket_field_at_atoms(pocket, [(0.0, 0.0, 0.0)])
