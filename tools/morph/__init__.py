"""Designed structural analogues of a lead compound: types and 3D embedding.

GENERIC LIBRARY -- contains no molecule. A campaign's hypothesis set lives with
that campaign (e.g. `experiments/danuglipron/design.py`) and is passed in.

- `design.py` -- the `Analogue` / `PharmacophoreSpec` types.
- `embed.py`  -- SMILES -> 3D conformers (RDKit ETKDGv3 + MMFF), with honest
                 failure and the pharmacophore check applied to the geometry.
"""

from .design import Analogue, PharmacophoreSpec
from .embed import EmbeddedAnalogue, embed_analogue

__all__ = [
    "Analogue",
    "PharmacophoreSpec",
    "EmbeddedAnalogue",
    "embed_analogue",
]
