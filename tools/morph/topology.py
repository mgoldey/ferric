"""Build a molecule from a KNOWN topology and write coordinates onto it.

## Why this exists

Perceiving bonds from coordinates (`rdDetermineBonds.DetermineConnectivity`)
uses INTERATOMIC DISTANCE ALONE and enforces no valence rule. On a docked pose
-- strained and close-packed by construction -- it bridges atoms that are near
but not bonded.

MEASURED 2026-09-21 on 8 Vina poses of danuglipron in 7LCJ, each already
re-hydrogenated to the correct 71-atom molecule: **8/8 came back chemically
impossible**, with carbons of degree 5-6 and hydrogens carrying two bonds.

The damage is silent and downstream. `Chem.RemoveHs(mol, sanitize=False)`
cannot drop a 2-bonded "hydrogen", so the perceived graph carried 43 heavy
atoms where the molecule has 41, and `rdFMCS.FindMCS` matched only **19 of 41
(27%)** in a molecule paired with ITSELF -- where the MCS is by definition the
whole molecule. The unmatched 73% then re-embedded freely, ~6 A away.

When the topology is already known -- and it always is, because the SMILES is
what produced the pose -- there is nothing to perceive. Build from SMILES,
write the coordinates on, and assert the graph.

    mol = mol_with_coords(smiles, symbols, coords)   # 100% MCS coverage

## The assertions are the point

`assert_graph_is_sane` is cheap and catches exactly the failure above:
no hydrogen with more than one bond, no carbon above degree four, and a
heavy-atom count that matches the declared topology. All three fire on the
distance-perceived poses; all three pass here.
"""

from __future__ import annotations

from collections.abc import Sequence

Coords = Sequence[tuple[float, float, float]]

__all__ = ["mol_with_coords", "assert_graph_is_sane", "GraphSanityError"]

# Maximum bond count per element for the organic subset these pipelines use.
# Exceeding one of these means the graph is not chemistry.
_MAX_DEGREE = {"H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1, "O": 2, "N": 4, "C": 4}


class GraphSanityError(ValueError):
    """A molecular graph that cannot be chemistry."""


def assert_graph_is_sane(mol, *, expect_heavy: int | None = None) -> None:
    """Raise `GraphSanityError` if `mol`'s connectivity is impossible.

    Checks, in the order they were observed to fail on real docked poses:
      1. no atom exceeds its element's maximum bond count (a 2-bonded hydrogen
         and a 5- or 6-valent carbon were both produced by distance-only
         perception);
      2. the heavy-atom count matches `expect_heavy`, when given.
    """
    offenders = []
    for atom in mol.GetAtoms():
        cap = _MAX_DEGREE.get(atom.GetSymbol())
        if cap is not None and atom.GetDegree() > cap:
            offenders.append(
                f"{atom.GetSymbol()}{atom.GetIdx()} has {atom.GetDegree()} "
                f"bonds (max {cap})"
            )
    if offenders:
        raise GraphSanityError(
            "impossible connectivity -- this graph is not chemistry, and any "
            "MCS/scaffold match computed on it is meaningless: "
            + "; ".join(offenders[:6])
        )

    if expect_heavy is not None:
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if heavy != expect_heavy:
            raise GraphSanityError(
                f"{heavy} heavy atoms but the topology declares {expect_heavy} "
                "-- these are different molecules"
            )


def mol_with_coords(
    smiles: str,
    symbols: Sequence[str],
    coords: Coords,
    *,
    random_seed: int = 0xF00D,
    check: bool = True,
):
    """An RDKit molecule with `smiles`' topology and `coords`' geometry.

    The bond graph comes from `smiles`, which is authoritative; the
    coordinates are written onto it in order. No bond is ever inferred from a
    distance, so a strained or close-packed pose cannot corrupt the graph.

    `symbols` is checked against the topology element-by-element -- a mismatch
    means the coordinates belong to a different molecule and is raised, never
    zipped away.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    if len(symbols) != len(coords):
        raise ValueError(
            f"mol_with_coords: {len(symbols)} symbols but {len(coords)} "
            "coordinate rows -- these are per-atom and must match"
        )

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"mol_with_coords: RDKit could not parse {smiles!r}")
    mol = Chem.AddHs(parsed)

    if mol.GetNumAtoms() != len(symbols):
        raise ValueError(
            f"mol_with_coords: the topology has {mol.GetNumAtoms()} atoms but "
            f"{len(symbols)} coordinates were supplied -- these are different "
            "molecules, not a hydrogen-count difference"
        )

    topo_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if topo_symbols != list(symbols):
        first = next(
            (i for i, (x, y) in enumerate(zip(topo_symbols, symbols)) if x != y),
            None,
        )
        raise ValueError(
            "mol_with_coords: symbol order does not match the topology "
            f"(index {first}: topology {topo_symbols[first]!r} vs supplied "
            f"{symbols[first]!r}) -- the coordinates would land on the wrong atoms"
        )

    if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
        raise ValueError(
            "mol_with_coords: RDKit could not embed the topology, so there is "
            "no conformer to write coordinates onto"
        )
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    if check:
        assert_graph_is_sane(mol, expect_heavy=parsed.GetNumAtoms())
    return mol
