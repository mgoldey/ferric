"""RDKit interoperability: ``ferric.Molecule`` from and to ``rdkit.Chem.Mol``.

RDKit is an optional dependency, imported inside each function::

    pip install rdkit            # or: pip install "ferric[rdkit]"

    import ferric.rdkit as fr

    mol = fr.from_smiles("CCO")          # AddHs + ETKDG (fixed seed) + MMFF
    rd = fr.to_rdkit(mol)                # RDKit Mol with a 3-D conformer

Coordinates travel as Angstrom floats and are handed to
``Molecule.from_arrays`` as float64 (no text round trip), so they are
preserved exactly; ferric converts to Bohr once, as for any Angstrom input.
Charge and multiplicity are read from the ``ferric.Molecule`` by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import ferric

__all__ = ["from_rdkit", "from_smiles", "to_rdkit"]

_INSTALL_HINT = (
    "RDKit is required for ferric.rdkit: pip install rdkit "
    '(or pip install "ferric[rdkit]")'
)


def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    return Chem


def from_rdkit(
    mol: Any,
    charge: int | None = None,
    multiplicity: int = 1,
    conf_id: int = -1,
) -> ferric.Molecule:
    """A ``ferric.Molecule`` from an RDKit ``Mol`` with a 3-D conformer.

    ``charge=None`` uses ``Chem.GetFormalCharge(mol)``. ``conf_id`` selects the
    conformer (-1: the default one). Explicit hydrogens are NOT added; call
    ``Chem.AddHs`` first (``from_smiles`` does). ``ValueError`` if the molecule
    has no conformer, the conformer is 2-D, or ``conf_id`` does not exist.
    """
    Chem = _rdkit()
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "the RDKit molecule has no conformer; embed one first "
            "(AllChem.EmbedMolecule on Chem.AddHs(mol)) or use ferric.rdkit.from_smiles"
        )
    try:
        conf = mol.GetConformer(conf_id)
    except ValueError as err:
        raise ValueError(f"the RDKit molecule has no conformer {conf_id}") from err
    if not conf.Is3D():
        raise ValueError(
            "the RDKit conformer is 2-D (flagged non-3D); embed 3-D coordinates first"
        )
    coords = np.asarray(conf.GetPositions(), dtype=float)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if charge is None:
        charge = Chem.GetFormalCharge(mol)
    return ferric.Molecule.from_arrays(
        symbols,
        coords,
        units="angstrom",
        charge=int(charge),
        multiplicity=int(multiplicity),
    )


def from_smiles(
    smiles: str,
    charge: int | None = None,
    multiplicity: int = 1,
    seed: int = 0xF00D,
    optimize: bool = True,
) -> ferric.Molecule:
    """Embed a SMILES string in 3-D and return a ``ferric.Molecule``.

    Adds hydrogens, embeds with ETKDG (``randomSeed=seed``, so the result is
    reproducible for a given RDKit version) and, when ``optimize``, relaxes
    with MMFF (UFF if MMFF has no parameters). ``charge=None`` is the SMILES'
    formal charge. ``ValueError`` for an unparsable SMILES or a failed embed.
    """
    Chem = _rdkit()
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"RDKit could not embed {smiles!r} in 3-D")
    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return from_rdkit(mol, charge=charge, multiplicity=multiplicity)


def to_rdkit(
    mol: ferric.Molecule, charge: int | None = None, perceive_bonds: bool = True
) -> Any:
    """An RDKit ``Mol`` (one 3-D conformer, Angstrom) from a ``ferric.Molecule``.

    With ``perceive_bonds`` the connectivity, bond orders and formal charges
    come from ``rdDetermineBonds.DetermineBonds`` (total ``charge``, default ``mol.charge``;
    needs a closed-shell-like valence assignment). If that fails, or RDKit
    lacks it, connectivity alone is tried (``DetermineConnectivity``), and
    failing that the molecule has no bonds. Ghost atoms are rejected:
    ``ValueError``.
    """
    Chem = _rdkit()
    if charge is None:
        charge = mol.charge
    if any(mol.is_ghost()):
        raise ValueError("to_rdkit: ghost atoms have no RDKit equivalent")
    symbols = mol.symbols()
    coords = np.asarray(mol.coords(), dtype=float)
    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(sym))
    conf = Chem.Conformer(len(symbols))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    conf.Set3D(True)
    rw.AddConformer(conf, assignId=True)
    out = rw.GetMol()
    if perceive_bonds:
        try:
            from rdkit.Chem import rdDetermineBonds
        except ImportError:
            return out
        try:
            trial = Chem.Mol(out)
            rdDetermineBonds.DetermineBonds(trial, charge=int(charge))
            return trial
        except (ValueError, RuntimeError):
            pass
        try:
            trial = Chem.Mol(out)
            rdDetermineBonds.DetermineConnectivity(trial)
            return trial
        except (ValueError, RuntimeError):
            pass
    return out
