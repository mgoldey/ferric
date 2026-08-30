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
