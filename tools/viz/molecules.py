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
