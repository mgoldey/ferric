import pickle
import shutil
from pathlib import Path

from tools.active_site.pocket_charges import PocketCharges, derive_pocket_charges

FIXTURE = Path(__file__).parent / "fixture.pqr"


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


def test_pocket_charges_residue_fields_default_to_none():
    # Exactness anchor: constructing PocketCharges the OLD way (no residue
    # kwargs) must be unaffected — all three new fields default to None.
    pc = PocketCharges(charges=[(0.1, 0.0, 0.0, 0.0)], source_pdb=Path("fake.pdb"), ff="AMBER")
    assert pc.residue_ids is None
    assert pc.atom_names is None
    assert pc.res_names is None


def test_pocket_charges_residue_fields_are_settable():
    pc = PocketCharges(
        charges=[(0.1, 0.0, 0.0, 0.0)], source_pdb=Path("fake.pdb"), ff="AMBER",
        residue_ids=[0], atom_names=["N"], res_names=["THR"],
    )
    assert pc.residue_ids == [0]
    assert pc.atom_names == ["N"]
    assert pc.res_names == ["THR"]


def test_derive_pocket_charges_populates_residue_fields(monkeypatch, tmp_path):
    # Stub out the external pdb2pqr30 call: copy the checked-in fixture PQR
    # instead of actually running the tool.
    def fake_run_pdb2pqr(pdb_path, pqr_path, ff="AMBER"):
        shutil.copy(FIXTURE, pqr_path)
        return Path(pqr_path)

    monkeypatch.setattr("tools.active_site.pocket_charges.run_pdb2pqr", fake_run_pdb2pqr)
    fake_pdb = tmp_path / "fake.pdb"
    fake_pdb.write_text("")

    pc = derive_pocket_charges(fake_pdb, ff="AMBER")
    assert pc.n_charges == 5
    assert pc.residue_ids is not None
    assert pc.atom_names == ["N", "CA", "C", "O", "HG21"]
    assert pc.res_names == ["THR", "THR", "THR", "THR", "THR"]
    # Single-residue fixture: every atom shares one residue id.
    assert len(set(pc.residue_ids)) == 1
    assert len(pc.residue_ids) == 5
