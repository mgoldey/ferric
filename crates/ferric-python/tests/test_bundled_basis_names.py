"""`BasisSet.bundled_names()`, aliases and element coverage from Python.

The numbers behind every set are checked in Rust against PySCF
(crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs); this file pins the
Python surface: the name list, alias resolution, `elements()`, and that an
unknown name or an uncovered element is a clear error rather than a silent
omission.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def test_bundled_names_lists_every_set_and_each_loads():
    names = ferric.BasisSet.bundled_names()
    assert len(names) == len(set(names)) >= 60
    for want in ("6-31g*", "6-311++g**", "def2-tzvppd", "cc-pvqz", "pcseg-1"):
        assert want in names
    for name in names:
        assert ferric.BasisSet.bundled(name).elements(), name


@pytest.mark.parametrize(
    ("alias", "primary"),
    [
        ("6-31G(d,p)", "6-31g**"),
        ("6-31g(d)", "6-31g*"),
        ("cc-pvdz-ri", "cc-pvdz-rifit"),
        ("def2-TZVP/C", "def2-tzvp-rifit"),
        ("def2/jk", "def2-universal-jkfit"),
        ("def2-svp-jkfit", "def2-universal-jkfit"),
    ],
)
def test_aliases_resolve_to_the_same_set(alias, primary):
    a, b = ferric.BasisSet.bundled(alias), ferric.BasisSet.bundled(primary)
    assert a.elements() == b.elements()


def test_unknown_name_is_a_value_error_listing_known_sets():
    with pytest.raises(ValueError, match="unknown bundled basis.*def2-tzvp"):
        ferric.BasisSet.bundled("6-31g***")


def test_coverage_grew_and_gaps_error_with_the_covered_range():
    assert 35 in ferric.BasisSet.bundled("cc-pvdz").elements()  # Br, new
    assert 86 in ferric.BasisSet.bundled("def2-svp").elements()  # Rn, new
    jk = ferric.BasisSet.bundled("cc-pvtz-jkfit")
    assert 3 not in jk.elements()
    lih = ferric.Molecule.from_xyz_string("2\nLiH\nLi 0 0 0\nH 0 0 1.6\n", 0, 1)
    with pytest.raises(Exception, match=r"Li \(Z=3\).*H, B-F, Al-Cl, Ga-Br"):
        ferric.run_rhf(lih, jk)
