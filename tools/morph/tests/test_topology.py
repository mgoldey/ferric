"""Coordinates must never be allowed to rewrite the bond graph.

MEASURED 2026-09-21: `rdDetermineBonds.DetermineConnectivity` on 8 docked
danuglipron poses produced 8/8 chemically impossible graphs (carbons of degree
5-6, hydrogens with two bonds). The MCS against the same molecule then matched
19 of 41 heavy atoms -- 27% -- and nothing downstream noticed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="topology handling is RDKit-based")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from tools.morph.topology import (  # noqa: E402
    GraphSanityError,
    assert_graph_is_sane,
    mol_with_coords,
)

TOLUENE = "Cc1ccccc1"


def _toluene_ref():
    mol = Chem.AddHs(Chem.MolFromSmiles(TOLUENE))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in row) for row in pos]


def test_the_topology_comes_from_smiles_and_the_geometry_from_coords():
    syms, coords = _toluene_ref()
    mol = mol_with_coords(TOLUENE, syms, coords)
    assert mol.GetNumAtoms() == len(syms)
    got = mol.GetConformer().GetPositions()
    for want, row in zip(coords, got):
        assert tuple(round(float(v), 6) for v in row) == tuple(
            round(v, 6) for v in want
        )


def test_a_squashed_geometry_does_NOT_corrupt_the_graph():
    """The regression this module exists for.

    Distance-based perception invents bonds when atoms are pushed together.
    Building from the SMILES cannot, however bad the geometry is.
    """
    syms, coords = _toluene_ref()
    squashed = [(x * 0.45, y * 0.45, z * 0.45) for x, y, z in coords]

    mol = mol_with_coords(TOLUENE, syms, squashed)
    assert_graph_is_sane(mol, expect_heavy=7)
    assert all(a.GetDegree() <= 1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
    assert max(a.GetDegree() for a in mol.GetAtoms() if a.GetSymbol() == "C") <= 4

    # And the negative control: perception on the SAME squashed coordinates
    # must actually break, or this test proves nothing about the fix.
    from rdkit.Chem import rdDetermineBonds

    xyz = f"{len(syms)}\n\n" + "".join(
        f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n" for s, c in zip(syms, squashed)
    )
    perceived = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(perceived)
    worst_h = max(
        (a.GetDegree() for a in perceived.GetAtoms() if a.GetSymbol() == "H"),
        default=0,
    )
    worst_c = max(
        (a.GetDegree() for a in perceived.GetAtoms() if a.GetSymbol() == "C"),
        default=0,
    )
    assert worst_h > 1 or worst_c > 4, (
        "perception did NOT break on this geometry, so it is not a negative "
        "control -- squash it further or the comparison is vacuous"
    )


def test_an_impossible_graph_is_REFUSED():
    syms, coords = _toluene_ref()
    squashed = [(x * 0.3, y * 0.3, z * 0.3) for x, y, z in coords]
    from rdkit.Chem import rdDetermineBonds

    xyz = f"{len(syms)}\n\n" + "".join(
        f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n" for s, c in zip(syms, squashed)
    )
    bad = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(bad)
    with pytest.raises(GraphSanityError, match="not chemistry"):
        assert_graph_is_sane(bad)


def test_a_heavy_atom_count_mismatch_is_REFUSED():
    syms, coords = _toluene_ref()
    mol = mol_with_coords(TOLUENE, syms, coords)
    with pytest.raises(GraphSanityError, match="different molecules"):
        assert_graph_is_sane(mol, expect_heavy=99)


def test_symbol_order_must_match_the_topology():
    """Coordinates landing on the wrong atoms is silent and catastrophic."""
    syms, coords = _toluene_ref()
    shuffled = list(syms)
    i = next(k for k, s in enumerate(shuffled) if s == "H")
    j = next(k for k, s in enumerate(shuffled) if s == "C")
    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    with pytest.raises(ValueError, match="symbol order"):
        mol_with_coords(TOLUENE, shuffled, coords)


def test_per_atom_lengths_must_match():
    syms, coords = _toluene_ref()
    with pytest.raises(ValueError, match="per-atom"):
        mol_with_coords(TOLUENE, syms, coords[:-1])


def test_a_different_molecule_is_REFUSED():
    syms, coords = _toluene_ref()
    with pytest.raises(ValueError, match="different molecules"):
        mol_with_coords("CCO", syms, coords)
