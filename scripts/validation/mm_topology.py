"""Parameter ASSIGNMENT for ferric-mm from an OpenMM ForceField.

The implementation is the packaged module `ferric.openmm`
(`crates/ferric-python/python/ferric/openmm.py`); this file keeps the import
name the validation generators use (`from mm_topology import ...`).
"""

from ferric.openmm import (  # noqa: F401
    Angle,
    Bond,
    Torsion,
    topology_from_openmm,
    topology_from_system,
)
