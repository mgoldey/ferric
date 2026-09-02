"""Substituent scans: symmetry dedup, provenance, and determinism."""
from __future__ import annotations

import pytest

from tools.isomers.substitutional import COMMON_SUBSTITUENTS, substituent_scan

BENZOIC = "OC(=O)c1ccccc1"


def test_fluorine_scan_finds_the_three_distinct_positions():
    out = substituent_scan(BENZOIC, {"F": "F"})
    assert len(out) == 3, [i.smiles for i in out]
    assert len({i.canonical for i in out}) == 3


def test_products_are_deduplicated_by_symmetry():
    """Benzoic acid has 5 aryl H but only 3 distinct products (ortho/meta/para).
    Returning 5 would mean symmetry-equivalent duplicates leaked through and
    every downstream population count is inflated."""
    assert len(substituent_scan(BENZOIC, {"F": "F"})) == 3


def test_each_isomer_records_the_transform_that_made_it():
    out = substituent_scan(BENZOIC, {"CN": "C#N"})
    assert out
    assert all(i.kind == "substitutional" for i in out)
    assert all("CN" in i.transform for i in out)
    assert all(i.parent_smiles == BENZOIC for i in out)


def test_multiple_substituents_compose():
    out = substituent_scan(BENZOIC, {"F": "F", "Cl": "Cl"})
    assert len(out) == 6


def test_products_really_contain_the_substituent():
    """Reachability: a scan that returned the parent N times would pass a count
    check. Assert the atom actually appears."""
    for i in substituent_scan(BENZOIC, {"F": "F"}):
        assert "F" in i.canonical


def test_unparseable_parent_raises():
    with pytest.raises(ValueError, match="unparseable parent"):
        substituent_scan("C1CC(((", {"F": "F"})


def test_enumeration_is_deterministic():
    """RunReactants does not guarantee product order; the sort must."""
    a = [i.canonical for i in substituent_scan(BENZOIC, COMMON_SUBSTITUENTS)]
    b = [i.canonical for i in substituent_scan(BENZOIC, COMMON_SUBSTITUENTS)]
    assert a == b, "same input gave a different candidate ORDER"


def test_a_site_with_no_match_yields_nothing():
    assert substituent_scan("CCO", {"F": "F"}, site_smarts="[cH:1]") == []
