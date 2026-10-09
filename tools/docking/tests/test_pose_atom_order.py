"""A docked pose's heavy atoms must land on the RIGHT atoms of the molecule.

Meeko writes a PDBQT in torsion-tree order (ROOT, then each BRANCH), not in
RDKit's order, and Vina keeps that order in its output. A pose that is mapped
back positionally puts coordinates on the wrong atoms while every count and
formula still checks out -- the quietest wrong answer the pipeline can give.

Salvaged from the unmerged fix/funnel-pose-handoff branch, whose positional
mapping was verified on phenol (where the two orders coincide by luck) and then
failed on the first real candidate. Rewritten against main's mechanism
(`REMARK SMILES IDX` -> `heavy_atom_mapping` -> `restore_hydrogens` with
Meeko's own SMILES), and driven through `tier1_dock` with ONLY Vina's search
replaced, so the PDBQT, its remarks and the mapping are all production.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from tools.docking.united_atom import restore_hydrogens  # noqa: E402

# The candidate that killed the 2026-09-12 run on the branch: 42 heavy atoms,
# one declared stereocentre, and a Meeko order that differs from RDKit's.
KILLER = (
    "N#Cc1ccc(COc2cc(F)cc(C3CCN(Cc4nc5ccc(C(=O)[O-])cc5n4C[C@@H]4CCO4)CC3)n2)c(F)c1"
)

# Where the fake "docked" pose is put: far from any embedding, so a
# re-embedding cannot be mistaken for the pose.
POSE_SHIFT = 50.0


def _skeleton_smiles(mol) -> str:
    """Element + connectivity only: no bond orders, charges or stereo.

    Lets a molecule perceived from bare coordinates be compared with one
    built from SMILES.
    """
    rw = Chem.RWMol()
    for a in mol.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetNoImplicit(True)
        rw.AddAtom(na)
    for b in mol.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(m)


def _perceived_skeleton(symbols, coords) -> str:
    """The molecule these coordinates describe, from geometry ALONE.

    An independent construction: it never sees the SMILES, the mapping or the
    PDBQT, so it cannot share a mistake with them.
    """
    from rdkit.Chem import rdDetermineBonds

    xyz = "\n".join(
        [str(len(symbols)), ""]
        + [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords)]
    )
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    return _skeleton_smiles(mol)


def _expected_skeleton(smiles) -> str:
    return _skeleton_smiles(Chem.AddHs(Chem.MolFromSmiles(smiles)))


# ── an explicit map that names the wrong element must be refused ─────────────

# Achiral, mixed elements: the stereo guard cannot fire, so only an element
# check can catch a mis-mapping.
FLUOROPHENOL = "Oc1ccc(F)cc1"


def test_a_map_that_puts_atoms_on_the_wrong_element_is_refused():
    """A REVERSED map is a valid permutation, so the permutation check passes
    it; only comparing each docked atom's element with its target's catches
    it. Without that, O's coordinates land on a carbon and F's on another,
    and a full-hydrogen molecule comes back with no error."""
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(FLUOROPHENOL))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    pos = mol.GetConformer().GetPositions()
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]
    coords = [tuple(float(v) for v in pos[i]) for i in heavy]

    reversed_map = list(reversed(heavy))
    # Premise: the reversed map really is a permutation (so the existing
    # permutation check cannot be what catches it) and really does misassign
    # elements (so there is something to catch).
    assert sorted(reversed_map) == sorted(heavy)
    assert any(
        mol.GetAtomWithIdx(t).GetSymbol() != s for t, s in zip(reversed_map, syms)
    )

    with pytest.raises(ValueError, match="element"):
        restore_hydrogens(FLUOROPHENOL, syms, coords, rdkit_index_of_heavy=reversed_map)


def test_the_correct_map_still_passes():
    """Reachability: the guard must not refuse the identity case."""
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(FLUOROPHENOL))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    pos = mol.GetConformer().GetPositions()
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]
    coords = [tuple(float(v) for v in pos[i]) for i in heavy]
    out_syms, _ = restore_hydrogens(
        FLUOROPHENOL, syms, coords, rdkit_index_of_heavy=list(heavy)
    )
    assert len(out_syms) == mol.GetNumAtoms()


# ── the production Meeko path, end to end through tier 1 ─────────────────────


class _EchoVina:
    """Vina with the search removed: returns the input ligand as pose 1.

    Real Vina moves coordinates but keeps the input PDBQT's atom lines in
    order, so echoing the input (shifted) is the shape a real pose has.
    """

    def __init__(self, **kwargs):
        self._ligand = ""

    def set_receptor(self, *a, **k):
        pass

    def set_ligand_from_string(self, text, *a, **k):
        self._ligand = text

    def compute_vina_maps(self, *a, **k):
        pass

    def dock(self, *a, **k):
        pass

    def poses(self, n_poses=1):
        out = ["MODEL 1", "REMARK VINA RESULT:    -7.5      0.000      0.000"]
        for line in self._ligand.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                line = (
                    f"{line[:30]}{x + POSE_SHIFT:8.3f}{y + POSE_SHIFT:8.3f}"
                    f"{z + POSE_SHIFT:8.3f}{line[54:]}"
                )
            out.append(line)
        out.append("ENDMDL")
        return "\n".join(out) + "\n"


@pytest.fixture
def echo_vina(monkeypatch, tmp_path):
    pytest.importorskip("meeko")
    mod = types.ModuleType("vina")
    mod.Vina = _EchoVina
    monkeypatch.setitem(sys.modules, "vina", mod)
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("ATOM\n")
    return receptor


def _meeko_order_differs_from_rdkit(smiles) -> bool:
    """Does Meeko's PDBQT list this molecule's heavy atoms in a different
    order from RDKit? If not, positional mapping would pass by luck."""
    from rdkit.Chem import AllChem

    from tools.docking.united_atom import parse_smiles_idx_remark
    from tools.docking.vina_dock import _ligand_pdbqt_from_rdkit

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    pdbqt = _ligand_pdbqt_from_rdkit(mol)
    serial_to_idx = parse_smiles_idx_remark(pdbqt)
    order = [serial_to_idx[k] for k in sorted(serial_to_idx)]
    return order != sorted(order)


def test_the_killer_molecules_orders_really_differ(echo_vina):
    """Reachability for the test below. Phenol alone gave false confidence on
    the branch because its two orders coincide."""
    assert _meeko_order_differs_from_rdkit(KILLER), (
        "Meeko's order coincides with RDKit's for this molecule; pick a harder one"
    )


def test_tier1_returns_the_docked_molecule_with_its_own_connectivity(echo_vina):
    """THE property: the pose tier 1 hands on is THIS molecule, at the docked
    coordinates, with every hydrogen back.

    Checked against the connectivity perceived from the output coordinates
    alone. A mapping error (positional fallback, the map paired with the wrong
    SMILES, a mis-read remark) puts heavy atoms on the wrong atoms, and the
    perceived bond graph then differs from the SMILES graph.
    """
    from tools.isomers.model import Isomer
    from tools.pipeline.tiers import tier1_dock

    iso = Isomer(KILLER, "substitutional", "killer", KILLER, net_charge=-1)
    res = tier1_dock(
        iso, {"receptor_pdbqt": str(echo_vina), "box_center": (0.0, 0.0, 0.0)}
    )
    assert res.ok, res.error
    syms, coords = res.payload["symbols"], res.payload["coords"]

    expected_n = Chem.AddHs(Chem.MolFromSmiles(KILLER)).GetNumAtoms()
    assert len(syms) == len(coords) == expected_n == 70

    assert _perceived_skeleton(syms, coords) == _expected_skeleton(KILLER), (
        "the coordinates tier 1 returned do not describe this molecule's bond "
        "graph -- heavy atoms were placed on the wrong atoms"
    )

    # The docked heavy atoms are DATA: every one must sit in the shifted frame,
    # i.e. it is the pose and not a re-embedding.
    heavy = [c for s, c in zip(syms, coords) if s != "H"]
    assert all(min(c) > POSE_SHIFT / 2 for c in heavy), (
        "heavy atoms are not at the docked coordinates"
    )
