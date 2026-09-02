"""Structural isomers: scaffold changes, and their honest no-ops."""
from __future__ import annotations

import pytest

from tools.isomers.structural import (
    ACID_BIOISOSTERES,
    bioisostere_swaps,
    ring_contractions,
    stereoisomers,
)


def test_stereoisomers_enumerates_unassigned_centres():
    out = stereoisomers("CC(N)C(=O)O")
    assert len(out) == 2
    assert {i.canonical for i in out} == {"C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"}
    assert all(i.kind == "structural" for i in out)


def test_stereoisomers_respects_the_cap():
    """Several unassigned centres would be 2^n; the cap must bind."""
    out = stereoisomers("CC(N)C(O)C(N)C(O)C(=O)O", max_isomers=4)
    assert len(out) <= 4


def test_molecule_with_no_unassigned_centres_gives_itself():
    out = stereoisomers("c1ccccc1")
    assert len(out) == 1


def test_ring_contraction_shrinks_a_saturated_ring():
    out = ring_contractions("C1CCNCC1")          # piperidine
    assert out, "no contraction produced from piperidine"
    canon = {i.canonical for i in out}
    assert any(c in ("C1CNC1", "C1CCNC1") for c in canon), canon


def test_ring_contraction_is_a_no_op_without_a_matching_ring():
    assert ring_contractions("CCO") == []


def test_bioisostere_swap_replaces_a_carboxylic_acid():
    out = bioisostere_swaps("OC(=O)c1ccccc1")
    labels = {i.transform for i in out}
    assert "tetrazole" in labels, labels
    assert all(i.kind == "structural" for i in out)


def test_bioisostere_products_no_longer_contain_the_original_acid():
    """Reachability: a swap that returned the parent would pass a count check."""
    from rdkit import Chem

    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    for i in bioisostere_swaps("OC(=O)c1ccccc1"):
        mol = Chem.MolFromSmiles(i.canonical)
        assert not mol.HasSubstructMatch(acid), i.canonical


def test_bioisostere_swap_is_a_no_op_without_the_group():
    assert bioisostere_swaps("c1ccccc1") == []


def test_a_malformed_transform_raises_rather_than_skipping():
    with pytest.raises(ValueError, match="bad transform"):
        bioisostere_swaps("OC(=O)c1ccccc1", {"broken": ("[CX3](=O)[OX2H1]", "C1CC(((")})


def test_structural_enumeration_is_deterministic():
    a = [i.canonical for i in bioisostere_swaps("OC(=O)c1ccccc1")]
    b = [i.canonical for i in bioisostere_swaps("OC(=O)c1ccccc1")]
    assert a == b
