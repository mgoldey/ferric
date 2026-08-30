"""The Isomer record: canonical identity and honest failure."""
from __future__ import annotations

import pytest

from tools.isomers.model import Isomer


def test_canonical_smiles_is_order_independent():
    a = Isomer("OC(=O)c1ccccc1", "substitutional", "none", "OC(=O)c1ccccc1")
    b = Isomer("c1ccccc1C(O)=O", "substitutional", "none", "OC(=O)c1ccccc1")
    assert a.canonical == b.canonical


def test_parent_is_detected_by_canonical_form_not_string_equality():
    """The same molecule written differently must still register as the parent,
    or a dedup pass would keep two copies of it."""
    i = Isomer("c1ccccc1C(O)=O", "substitutional", "none", "OC(=O)c1ccccc1")
    assert i.is_parent is True


def test_a_real_variant_is_not_the_parent():
    i = Isomer("OC(=O)c1ccc(F)cc1", "substitutional", "F", "OC(=O)c1ccccc1")
    assert i.is_parent is False


def test_unparseable_smiles_raises_rather_than_returning_none():
    with pytest.raises(ValueError, match="unparseable"):
        _ = Isomer("C1CC(((", "substitutional", "x", "C").canonical


def test_transform_and_parent_travel_with_the_candidate():
    i = Isomer("OC(=O)c1ccc(F)cc1", "substitutional", "[cH:1] -> F",
               "OC(=O)c1ccccc1")
    assert i.transform == "[cH:1] -> F"
    assert i.parent_smiles == "OC(=O)c1ccccc1"
    assert i.kind == "substitutional"


# ── deprotonation ──
#
# This is the bug that killed a 55-minute pipeline run: every tier-4 candidate
# failed with "325 electrons with multiplicity 1 implies n_alpha = 325/2, not
# an integer". The cause was declaring net_charge=-1 on a NEUTRAL structure.
# Removing H+ takes a proton and leaves its electrons behind, so an anion has
# the SAME electron count as its acid -- asking for one more electron is asking
# for something that does not exist.

def _electron_count(smiles: str) -> int:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return (sum(a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms())
            - Chem.GetFormalCharge(mol))


def test_deprotonation_conserves_electron_count():
    """THE invariant. H+ is a bare proton: the electrons stay behind."""
    acid = Isomer("OC(=O)c1ccccc1", "parent", "none", "OC(=O)c1ccccc1")
    anion = acid.deprotonated()
    assert anion is not None
    assert _electron_count(anion.canonical) == _electron_count(acid.canonical)


def test_deprotonated_electron_count_is_even_for_a_closed_shell_singlet():
    """An odd electron count with multiplicity 1 is unrepresentable, which is
    exactly the error ferric raised on the broken run."""
    acid = Isomer("OC(=O)c1ccccc1", "parent", "none", "OC(=O)c1ccccc1")
    assert _electron_count(acid.deprotonated().canonical) % 2 == 0


def test_deprotonation_changes_the_structure_not_just_the_charge():
    anion = Isomer("OC(=O)c1ccccc1", "p", "none", "OC(=O)c1ccccc1").deprotonated()
    assert "[O-]" in anion.canonical
    assert anion.net_charge == -1


def test_deprotonation_handles_every_supported_acid_type():
    from rdkit import Chem

    for smi in ("OC(=O)c1ccccc1",               # carboxylic acid
                "c1ccccc1c1nn[nH]n1",           # tetrazole
                "CC(=O)NS(=O)(=O)C"):           # acylsulfonamide
        iso = Isomer(smi, "p", "none", smi)
        anion = iso.deprotonated()
        assert anion is not None, f"no deprotonation found for {smi}"
        assert Chem.GetFormalCharge(Chem.MolFromSmiles(anion.canonical)) == -1
        assert _electron_count(anion.canonical) == _electron_count(smi)


def test_a_molecule_with_no_acidic_proton_returns_none():
    """None, not a fabricated anion: a neutral molecule scored at charge -1
    would be a different chemical system entirely."""
    assert Isomer("c1ccccc1", "p", "none", "c1ccccc1").deprotonated() is None


def test_deprotonation_records_its_provenance():
    anion = Isomer("OC(=O)c1ccccc1", "p", "orig", "OC(=O)c1ccccc1").deprotonated()
    assert "deprotonated" in anion.transform
    assert "orig" in anion.transform
    assert "pH-7.4 anion" in anion.notes
