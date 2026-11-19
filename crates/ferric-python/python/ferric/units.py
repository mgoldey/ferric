"""Unit-conversion constants, the same numbers the Rust crates use.

Each constant is a multiplier: ``x_angstrom = x_bohr * BOHR_TO_ANGSTROM``.

* ``BOHR_TO_ANGSTROM`` / ``ANGSTROM_TO_BOHR``: ``crates/ferric-core/src/units.rs``
  (``BOHR_RADIUS_ANGSTROM`` = 0.52917721092, CODATA 2010, PySCF ``BOHR``;
  ``ANGSTROM_TO_BOHR`` is its reciprocal, computed the same way).
* ``HARTREE_TO_EV``: ``crates/ferric-cli/src/lib.rs`` (``ha_to_ev``,
  27.211386245988, CODATA 2018).
* ``HARTREE_TO_KCAL_MOL``: ``crates/ferric-cli/src/thermo_ir.rs``
  (``HARTREE_TO_KCAL_PER_MOL``, 627.5094740631).
* ``HARTREE_TO_KJ_MOL``: ferric has no Rust constant for kJ/mol; it is the
  kcal/mol value times the thermochemical calorie, 4.184 kJ/kcal (exact).
"""

from __future__ import annotations

__all__ = [
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_MOL",
    "HARTREE_TO_KJ_MOL",
]

BOHR_TO_ANGSTROM = 0.529_177_210_92
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV = 27.211_386_245_988
HARTREE_TO_KCAL_MOL = 627.509_474_063_1
HARTREE_TO_KJ_MOL = HARTREE_TO_KCAL_MOL * 4.184
