"""A docked pose is not the molecule that was docked until its H's are back.

MEASURED 2026-09-21, danuglipron into 7LCJ: 71 atoms in, **42 out**. The
missing 29 are nonpolar hydrogens Vina merges into their carbons. Feeding that
42-atom object to a scorer returns a plausible number for a different molecule.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="re-hydrogenation is RDKit-based")

from rdkit import Chem  # noqa: E402

from tools.docking.united_atom import restore_hydrogens  # noqa: E402

# Small, unambiguous, and has nonpolar hydrogens to lose: toluene.
TOLUENE = "Cc1ccccc1"
# Heavy atoms only, in SMILES heavy-atom order (a united-atom pose's shape).
TOLUENE_HEAVY = ["C", "C", "C", "C", "C", "C", "C"]
TOLUENE_COORDS = [
    (0.000, 0.000, 0.000),
    (1.500, 0.000, 0.000),
    (2.200, 1.212, 0.000),
    (3.600, 1.212, 0.000),
    (4.300, 0.000, 0.000),
    (3.600, -1.212, 0.000),
    (2.200, -1.212, 0.000),
]


def test_hydrogens_come_back_and_the_formula_is_the_molecule():
    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    assert len(syms) == len(coords)
    # C7H8 -- the united-atom input had only the 7 carbons.
    assert syms.count("C") == 7
    assert syms.count("H") == 8
    assert len(syms) == 15 > len(TOLUENE_HEAVY)


def test_the_docked_heavy_atoms_DO_NOT_MOVE():
    """The heavy atoms are the docking result.

    If relaxation moved them, the scored pose would not be the pose that was
    docked -- a silent substitution of the answer.
    """
    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    heavy_out = [c for s, c in zip(syms, coords) if s != "H"]
    assert len(heavy_out) == len(TOLUENE_COORDS)
    worst = max(
        max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(TOLUENE_COORDS, heavy_out)
    )
    assert worst == pytest.approx(0.0, abs=1e-9), (
        f"heavy atoms moved by {worst} A -- they are FIXED during H placement"
    )


def test_a_heavy_atom_COUNT_MISMATCH_is_refused_not_truncated():
    """The failure that motivated this module.

    Zipping to the shorter list would score a fragment of the molecule and
    return a number rather than an error.
    """
    with pytest.raises(ValueError, match="different molecules"):
        restore_hydrogens(TOLUENE, TOLUENE_HEAVY[:-1], TOLUENE_COORDS[:-1])


def test_symbols_and_coords_must_be_per_atom():
    with pytest.raises(ValueError, match="per-atom"):
        restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS[:-1])


def test_an_unparseable_smiles_is_refused():
    with pytest.raises(ValueError, match="could not parse"):
        restore_hydrogens("not-a-smiles((", TOLUENE_HEAVY, TOLUENE_COORDS)


def test_hydrogens_are_placed_at_chemically_sane_distances():
    """A vacuity guard: the H's must be BONDED, not dumped at the origin.

    Without this, a function that appended 8 hydrogens at (0,0,0) would pass
    every count-based assertion above.
    """
    import math

    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    heavy = [c for s, c in zip(syms, coords) if s != "H"]
    for s, c in zip(syms, coords):
        if s != "H":
            continue
        nearest = min(math.dist(c, h) for h in heavy)
        assert 0.8 < nearest < 1.3, (
            f"hydrogen {nearest:.2f} A from its nearest heavy atom -- a C-H "
            "bond is ~1.09 A, so this H was not actually placed on the molecule"
        )


# --- atom ORDER: meeko does not preserve it ---------------------------------
# MEASURED 2026-09-21 on danuglipron: only 12 of 41 heavy positions matched in
# input order after a meeko PDBQT round trip, and the declared (S) stereocentre
# came back (R) -- the mirror image of the drug, with no error raised.

ALANINE = "C[C@@H](N)C(=O)O"  # one defined stereocentre, small enough to check


def _alanine_ref():
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(ALANINE))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in row) for row in pos]


def test_a_SCRAMBLED_atom_order_is_refused_on_stereochemistry():
    """The silent failure: coordinates landing on the wrong atoms.

    Swapping two heavy atoms' coordinates makes a different isomer. Counts,
    formula and bond graph all still check out -- only the stereocentre shows
    it, which is why the stereo guard exists.
    """
    from tools.docking.united_atom import StereochemistryError

    syms, coords = _alanine_ref()
    heavy = [i for i, s in enumerate(syms) if s != "H"]
    scrambled = list(coords)
    a, b = heavy[0], heavy[1]
    scrambled[a], scrambled[b] = scrambled[b], scrambled[a]

    with pytest.raises(StereochemistryError, match="different isomer"):
        restore_hydrogens(ALANINE, syms, scrambled)


def test_an_explicit_order_map_is_validated_against_the_topology():
    syms, coords = _alanine_ref()
    heavy_n = sum(1 for s in syms if s != "H")
    with pytest.raises(ValueError, match="not a permutation"):
        restore_hydrogens(ALANINE, syms, coords, rdkit_index_of_heavy=[999] * heavy_n)


def test_pose_to_rdkit_order_reads_meekos_own_map():
    from tools.docking.united_atom import pose_to_rdkit_order

    pdbqt = (
        "REMARK SMILES CC(N)C(=O)O\n"
        "REMARK SMILES IDX 3 1 2 2 1 3\n"
        "ATOM      1  N   UNL     1       0.000   0.000   0.000\n"
    )
    smiles, order = pose_to_rdkit_order(pdbqt)
    assert smiles == "CC(N)C(=O)O"
    # sorted by pdbqt serial 1,2,3 -> smiles idx 3,2,1 -> 0-based 2,1,0
    assert order == [2, 1, 0]


def test_a_pdbqt_without_the_remarks_is_REFUSED():
    """Falling back to positional order is the bug, so absence must raise."""
    from tools.docking.united_atom import pose_to_rdkit_order

    with pytest.raises(ValueError, match="no `REMARK SMILES`"):
        pose_to_rdkit_order("ATOM      1  N   UNL     1       0.0   0.0   0.0\n")
