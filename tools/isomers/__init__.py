"""Isomer enumeration: generate candidates from a parent, reproducibly.

Substitutional isomers decorate a fixed scaffold; structural isomers change it.
Both are deduplicated on canonical SMILES and carry the transform that produced
them, so a candidate list is auditable rather than merely asserted.

GENERIC LIBRARY -- no molecule lives here. A campaign supplies its own parent
(see `experiments/<name>/design.py`).
"""

from .enumerate import EnumerationReport, enumerate_isomers, enumerate_with_report
from .model import Isomer
from .structural import bioisostere_swaps, ring_contractions, stereoisomers
from .substitutional import COMMON_SUBSTITUENTS, substituent_scan

__all__ = [
    "Isomer",
    "EnumerationReport",
    "enumerate_isomers",
    "enumerate_with_report",
    "substituent_scan",
    "COMMON_SUBSTITUENTS",
    "bioisostere_swaps",
    "ring_contractions",
    "stereoisomers",
]
