"""Analogue ("morphology") enumeration for a lead compound.

Generates *designed* structural variants of a parent molecule, each one
addressing a specific, named liability hypothesis, and each one carrying the
pharmacophore constraint it is required not to break. The point is not to
enumerate chemical space — it is to make a small set of falsifiable design
proposals whose fit and liability can both be measured.

Modules:
- `design.py`   — the analogue definitions and the `Analogue` record.
- `embed.py`    — SMILES -> 3D conformers (RDKit ETKDGv3 + MMFF), with the
                  pharmacophore check applied to the generated geometry.
"""

from .design import Analogue, DANUGLIPRON_SMILES, danuglipron_analogues, PharmacophoreSpec
from .embed import embed_analogue, EmbeddedAnalogue

__all__ = [
    "Analogue",
    "PharmacophoreSpec",
    "DANUGLIPRON_SMILES",
    "danuglipron_analogues",
    "embed_analogue",
    "EmbeddedAnalogue",
]
