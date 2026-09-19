"""2-D molecule depictions, including substitution-aware highlighting.

Rendering goes through RDKit, which is already a declared dependency of the
`docking` extra and is what every other structure-handling module in `tools/`
uses. No 3-D viewer is added here: py3Dmol and nglview are not installed, they
need a browser or a notebook to show anything, and the question these
depictions answer -- "which atoms changed, and where is the alert?" -- is a
2-D question.

## Why highlighting is computed, not passed in

`highlight_difference` finds the changed atoms itself, via a maximum common
substructure. A caller that passes an index list is asserting a mapping between
two molecules, and that assertion is exactly the thing most likely to be wrong
after an enumeration step reorders atoms. Computing it from the structures
means the picture cannot disagree with the molecules it depicts.

## What a depiction does not tell you

A 2-D drawing carries no conformer, no pose and no energy. `grid_with_scores`
prints whatever score it is handed underneath each structure, and labels an
absent score as "n/a" rather than omitting it -- so a candidate that was never
evaluated cannot be misread as one that scored badly, or vice versa.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "depict",
    "highlight_difference",
    "grid_with_scores",
    "contact_map",
    "contacting_atom_indices",
]

# A palette that survives greyscale printing and the common colour-vision
# deficiencies: added = blue, removed = orange. Deliberately NOT red/green.
_ADDED_RGB = (0.20, 0.45, 0.80)
_REMOVED_RGB = (0.90, 0.55, 0.15)


def _rdkit():
    """Import RDKit with an actionable message when it is absent.

    RDKit lives in the `docking` extra rather than the core dependencies (see
    pyproject.toml), so a bare install will not have it and the failure should
    say what to install rather than surfacing as a bare ImportError.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError as exc:  # pragma: no cover -- environment-dependent
        raise ImportError(
            "tools.viz.molecules needs RDKit, which is in the 'docking' extra: "
            "pip install 'ferric[docking]' (or uv sync --extra docking)."
        ) from exc
    return Chem, AllChem, Draw, rdMolDraw2D


def _mol_from(smiles: str):
    Chem, _, _, _ = _rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        # RDKit returns None rather than raising, which turns a typo'd SMILES
        # into a blank image several steps later.
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return m


def depict(
    smiles: str, *, width: int = 350, height: int = 300, legend: str = ""
) -> bytes:
    """One molecule as PNG bytes.

    Returns bytes rather than writing a file so the caller decides where it
    goes -- a test can assert on them without touching the filesystem.
    """
    Chem, _, _, rdMolDraw2D = _rdkit()
    m = _mol_from(smiles)
    Chem.rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()


def highlight_difference(
    parent_smiles: str,
    analogue_smiles: str,
    *,
    width: int = 350,
    height: int = 300,
    legend: str = "",
) -> bytes:
    """Depict `analogue`, highlighting the atoms that are NOT in `parent`.

    The shared scaffold is found with a maximum common substructure search, so
    the highlight is derived from the two structures rather than asserted by
    the caller. Atoms outside the MCS are drawn in blue.

    When the MCS covers the whole analogue -- the two are the same molecule, or
    the analogue is a substructure of the parent -- nothing is highlighted and
    the plain depiction is returned. That is the honest output: there is no
    difference to point at.
    """
    Chem, _, _, rdMolDraw2D = _rdkit()
    from rdkit.Chem import rdFMCS

    parent = _mol_from(parent_smiles)
    analogue = _mol_from(analogue_smiles)

    # ringMatchesRingOnly stops an open chain from matching a ring fragment,
    # which otherwise reports a "shared scaffold" that no chemist would accept.
    res = rdFMCS.FindMCS(
        [parent, analogue],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    changed: list[int] = []
    if res.numAtoms:
        patt = Chem.MolFromSmarts(res.smartsString)
        match = analogue.GetSubstructMatch(patt) if patt is not None else ()
        shared = set(match)
        changed = [a.GetIdx() for a in analogue.GetAtoms() if a.GetIdx() not in shared]

    Chem.rdDepictor.Compute2DCoords(analogue)
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    colors = {i: _ADDED_RGB for i in changed}
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        analogue,
        legend=legend,
        highlightAtoms=changed,
        highlightAtomColors=colors,
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def grid_with_scores(
    smiles: Sequence[str],
    scores: Sequence[float | None],
    *,
    labels: Sequence[str] | None = None,
    unit: str = "kcal/mol",
    per_row: int = 4,
    sub_size: tuple[int, int] = (300, 260),
) -> bytes:
    """A grid of structures with their scores underneath, as PNG bytes.

    A `None` score is captioned "n/a" and NOT sorted or styled as a bad score.
    A candidate that was never evaluated and a candidate that scored badly
    demand opposite responses, so they must not look alike -- the same rule
    `tools/campaign/hierarchy.py` states for tier results.
    """
    Chem, _, Draw, _ = _rdkit()
    if len(smiles) != len(scores):
        raise ValueError(
            f"{len(smiles)} structures but {len(scores)} scores -- these must "
            "correspond one-to-one"
        )
    if labels is not None and len(labels) != len(smiles):
        raise ValueError(f"{len(labels)} labels for {len(smiles)} structures")
    if not smiles:
        raise ValueError("nothing to draw")

    mols = [_mol_from(s) for s in smiles]
    for m in mols:
        Chem.rdDepictor.Compute2DCoords(m)

    captions = []
    for i, sc in enumerate(scores):
        name = labels[i] if labels is not None else f"#{i}"
        captions.append(f"{name}\nn/a" if sc is None else f"{name}\n{sc:.2f} {unit}")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(per_row, len(mols)),
        subImgSize=sub_size,
        legends=captions,
        useSVG=False,
        returnPNG=True,
    )
    # MolsToGridImage returns bytes with returnPNG=True, but older RDKit builds
    # hand back a PIL image; normalise so callers always get bytes.
    if isinstance(img, bytes):
        return img
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


#: Contacts, drawn in a third colour distinct from added/removed.
_CONTACT_RGB = (0.35, 0.65, 0.35)


def contacting_atom_indices(
    ligand_coords, pocket_coords, cutoff_angstrom: float
) -> list[int]:
    """Indices of ligand atoms within `cutoff_angstrom` of any pocket atom.

    Split out of `contact_map` so the CONTACT LOGIC can be asserted directly.
    Comparing two rendered PNGs cannot do it: an implementation that highlights
    every atom regardless of distance renders any two cutoffs identically to
    each other, and that mutation SURVIVED a byte-comparison test. A list of
    indices is checkable; an image is not.
    """
    import numpy as np

    lig = np.asarray(ligand_coords, dtype=float)
    pocket = np.asarray(pocket_coords, dtype=float)
    # Squared distances, no sqrt: the comparison is monotone in it.
    d2 = ((lig[:, None, :] - pocket[None, :, :]) ** 2).sum(axis=2)
    return [int(i) for i in np.where(d2.min(axis=1) <= cutoff_angstrom**2)[0]]


def contact_map(
    smiles: str,
    coords_angstrom,
    pocket_coords_angstrom,
    *,
    cutoff_angstrom: float = 4.0,
    width: int = 400,
    height: int = 340,
    legend: str = "",
) -> bytes:
    """Depict a ligand with its POCKET-CONTACTING atoms highlighted.

    A docked pose IS a 3-D geometry, so a 2-D drawing of one discards the thing
    that makes it a pose -- which is why `depict` takes SMILES and there is no
    "draw this pose" function. The question that DOES survive flattening is
    *which atoms touch the pocket*, and that is the one a medicinal chemist
    asks before choosing where to substitute: a buried atom has no room for a
    CF3, and a solvent-exposed one is where the scan should go.

    `coords_angstrom` is the ligand's 3-D geometry (e.g.
    `DockedPose.coords_angstrom`); `pocket_coords_angstrom` any iterable of
    (x, y, z). `cutoff_angstrom` is the contact radius -- 4.0 A is the usual
    van-der-Waals-contact convention, and it is a PARAMETER because the right
    value depends on what you mean by contact.

    **This does not say a contact is favourable.** Proximity is geometry; an
    unfavourable clash is also a contact. It shows where the ligand touches,
    not whether touching there is good.

    Atom ORDER must match between `smiles` (with explicit hydrogens, as RDKit
    builds it) and `coords_angstrom`. A mismatch raises rather than
    highlighting the wrong atoms -- which would be a confident, wrong picture.
    """
    Chem, AllChem, _, rdMolDraw2D = _rdkit()
    import numpy as np

    lig = np.asarray([tuple(float(v) for v in c) for c in coords_angstrom], dtype=float)
    if lig.ndim != 2 or lig.shape[1] != 3:
        raise ValueError(f"coords_angstrom must be (N, 3), got shape {lig.shape}")
    pocket = np.asarray(
        [tuple(float(v) for v in c) for c in pocket_coords_angstrom], dtype=float
    )
    if pocket.size == 0:
        raise ValueError(
            "no pocket coordinates -- with nothing to contact, every atom would "
            "render as non-contacting, which is a claim rather than an absence"
        )
    if pocket.ndim != 2 or pocket.shape[1] != 3:
        raise ValueError(f"pocket coords must be (M, 3), got shape {pocket.shape}")
    if not (cutoff_angstrom > 0):
        raise ValueError(f"cutoff_angstrom must be > 0, got {cutoff_angstrom}")

    mol = Chem.AddHs(_mol_from(smiles))
    if mol.GetNumAtoms() != len(lig):
        raise ValueError(
            f"{smiles!r} has {mol.GetNumAtoms()} atoms (with explicit H) but "
            f"{len(lig)} coordinates were given. Highlighting under a mismatched "
            "atom order would mark the WRONG atoms -- check whether the pose is "
            "united-atom (PDBQT merges nonpolar hydrogens: a 71-atom ligand "
            "comes back with 42)."
        )

    contacting = contacting_atom_indices(lig, pocket, cutoff_angstrom)

    Chem.rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    heavy_contacts = sum(
        1 for i in contacting if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    )
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        legend=legend
        or f"{heavy_contacts}/{n_heavy} heavy atoms within {cutoff_angstrom} A",
        highlightAtoms=contacting,
        highlightAtomColors={i: _CONTACT_RGB for i in contacting},
    )
    d.FinishDrawing()
    return d.GetDrawingText()
