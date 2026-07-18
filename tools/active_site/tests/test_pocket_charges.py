import pickle
from pathlib import Path

from tools.active_site.pocket_charges import PocketCharges


def test_pocket_charges_n_charges_derived():
    pc = PocketCharges(charges=[(0.1, 0.0, 0.0, 0.0), (-0.1, 1.0, 1.0, 1.0)],
                        source_pdb=Path("fake.pdb"), ff="AMBER")
    assert pc.n_charges == 2


def test_pocket_charges_picklable():
    pc = PocketCharges(charges=[(0.5, 1.0, 2.0, 3.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    pc2 = pickle.loads(pickle.dumps(pc))
    assert pc2.charges == pc.charges
    assert pc2.n_charges == pc.n_charges
    assert pc2.ff == pc.ff
