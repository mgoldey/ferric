"""Parameter ASSIGNMENT for ferric-mm: read explicit AMBER-form parameters
out of a real OpenMM ForceField applied to a real structure.

`ferric-mm` (crates/ferric-mm) assigns no parameters of its own — it is
arithmetic over caller-supplied explicit numbers. This module is the
assignment step: build an `openmm.System` from a PDB file and a named force
field (default `amber14-all.xml`), then walk `system.getForces()` and undo
OpenMM's own conventions to hand back plain AMBER-convention data that
`ferric.MmTopology.from_amber_units` (or the Rust `MmTopology::from_amber_units`)
can consume directly, with no further conversion.

Unit/convention bridge (the reverse of `scripts/gen_openmm_mm_refs.py`'s
forward direction):

    k_amber(bond)  = k_openmm(bond)  / 2      [kcal/mol/A^2 <- kJ/mol/nm^2]
    k_amber(angle) = k_openmm(angle) / 2      [kcal/mol/rad^2 <- kJ/mol/rad^2]
    PeriodicTorsionForce needs no factor-of-2 undo (OpenMM already uses
    E = k(1 + cos(n*phi - delta)), same as ferric-mm).

NonbondedForce's per-particle charge/sigma/epsilon are read directly (no
1-2/1-3/1-4 exceptions are read back out — `ferric.MmTopology.from_amber_units`
derives its OWN exclusions/1-4 pairs from the bond list via a BFS, so the
force field's `createExceptionsFromBonds`-derived exceptions are redundant
with, not a second source of truth alongside, what ferric-mm computes).
"""
from __future__ import annotations

from pathlib import Path

Bond = tuple[int, int, float, float]
Angle = tuple[int, int, int, float, float]
Torsion = tuple[int, int, int, int, int, float, float]


def topology_from_openmm(pdb_path: str | Path, forcefield: tuple[str, ...] = ("amber14-all.xml",)) -> dict:
    """Build an OpenMM System from `pdb_path` and `forcefield`, and return
    its parameters in plain AMBER-convention units as a dict:

        n_atoms: int
        charges: list[float]              (e)
        sigmas_angstrom: list[float]
        epsilons_kcal: list[float]
        bonds: list[(i, j, k, r0)]         k kcal/mol/A^2, r0 Angstrom
        angles: list[(i, j, k, k_theta, theta0)]   k_theta kcal/mol/rad^2, theta0 DEGREES
        torsions: list[(i, j, k, l, periodicity, k_phi, phase)]  k_phi kcal/mol, phase DEGREES

    Directly usable as the keyword arguments of
    `ferric.MmTopology.from_amber_units(**result)` after dropping `n_atoms`
    (or `ferric_mm::MmTopology::from_amber_units` on the Rust side, same
    argument order).
    """
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(*forcefield)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

    n_atoms = system.getNumParticles()
    charges: list[float] = [0.0] * n_atoms
    sigmas_angstrom: list[float] = [0.0] * n_atoms
    epsilons_kcal: list[float] = [0.0] * n_atoms
    bonds: list[Bond] = []
    angles: list[Angle] = []
    torsions: list[Torsion] = []

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(b)
                r0_ang = r0.value_in_unit(unit.angstrom)
                k_amber = 0.5 * k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2)
                bonds.append((i, j, k_amber, r0_ang))
        elif isinstance(force, openmm.HarmonicAngleForce):
            for a in range(force.getNumAngles()):
                i, j, k, theta0, k_theta = force.getAngleParameters(a)
                theta0_deg = theta0.value_in_unit(unit.degree)
                k_theta_amber = 0.5 * k_theta.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2)
                angles.append((i, j, k, k_theta_amber, theta0_deg))
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for t in range(force.getNumTorsions()):
                i, j, k, l, periodicity, phase, k_phi = force.getTorsionParameters(t)
                phase_deg = phase.value_in_unit(unit.degree)
                k_phi_kcal = k_phi.value_in_unit(unit.kilocalorie_per_mole)
                torsions.append((i, j, k, l, periodicity, k_phi_kcal, phase_deg))
        elif isinstance(force, openmm.NonbondedForce):
            for p in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(p)
                charges[p] = q.value_in_unit(unit.elementary_charge)
                sigmas_angstrom[p] = sigma.value_in_unit(unit.angstrom)
                epsilons_kcal[p] = epsilon.value_in_unit(unit.kilocalorie_per_mole)
        # CMMotionRemover and any other force (CMAP, GBSA, ...) carry no
        # AMBER-form bonded/nonbonded parameters ferric-mm models; skipped.

    return dict(
        n_atoms=n_atoms,
        charges=charges,
        sigmas_angstrom=sigmas_angstrom,
        epsilons_kcal=epsilons_kcal,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )
