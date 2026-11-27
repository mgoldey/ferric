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

NonbondedForce's per-particle charge/sigma/epsilon are read directly.
`ferric.MmTopology.from_amber_units` derives its OWN exclusions (1-2, 1-3)
and scaled 1-4 pairs (Coulomb 1/1.2, Lennard-Jones 1/2) from the bond list,
so the System's exceptions are not carried over: they are CHECKED against
that derivation, and a System whose exceptions differ (a hand-added
exception, other 1-4 scale factors) is refused with `ValueError`, as is a
System with a force ferric-mm has no term for (CMAP, GBSA, custom forces)
or with NonbondedForce parameter offsets (the parameters would depend on a
global parameter's value).

The NonbondedForce's method (cutoff, reaction field, PME, switching, the
dispersion correction) is NOT read or checked: it is how one program sums
the pairs, not a force-field parameter, and ferric-mm has its own setting
for it (all pairs by default).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["topology_from_openmm", "topology_from_system"]

Bond = tuple[int, int, float, float]
Angle = tuple[int, int, int, float, float]
Torsion = tuple[int, int, int, int, int, float, float]

_SCALE_COUL_14 = 1.0 / 1.2  # ferric_mm::DEFAULT_SCALE_COUL_14
_SCALE_LJ_14 = 0.5  # ferric_mm::DEFAULT_SCALE_LJ_14
# What ferric-mm evaluates for a pair, by bond separation, as fractions
# (Coulomb, Lennard-Jones) of the full interaction; other pairs get (1, 1).
_PAIR_SCALES = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (_SCALE_COUL_14, _SCALE_LJ_14)}
# amber14-all.xml writes coulomb14scale as 0.8333333 (4e-8 from 1/1.2).
_EXCEPTION_RTOL = 1.0e-6


def _bond_separations(n_atoms: int, bonds: list[Bond]) -> dict[tuple[int, int], int]:
    """Shortest bond-path length (1, 2 or 3) of every pair ``i < j`` within
    three bonds, as ferric-mm's `MmTopology::new` classifies them."""
    neighbours: list[set[int]] = [set() for _ in range(n_atoms)]
    for i, j, _, _ in bonds:
        neighbours[i].add(j)
        neighbours[j].add(i)
    separation: dict[tuple[int, int], int] = {}
    for start in range(n_atoms):
        seen = {start}
        frontier = {start}
        for depth in (1, 2, 3):
            frontier = {b for a in frontier for b in neighbours[a]} - seen
            seen.update(frontier)
            for other in frontier:
                if start < other:
                    separation[(start, other)] = depth
    return separation


def _close(got: float, want: float, scale: float) -> bool:
    return abs(got - want) <= _EXCEPTION_RTOL * max(abs(want), abs(scale))


def _check_exceptions(force, bonds, charges, sigmas_angstrom, epsilons_kcal) -> None:
    """Raise `ValueError` unless the NonbondedForce's exceptions are exactly
    what ferric-mm derives from ``bonds``: 1-2/1-3 pairs excluded, 1-4 pairs
    scaled by the AMBER factors with Lorentz-Berthelot mixing, no others."""
    from openmm import unit

    separation = _bond_separations(len(charges), bonds)
    seen = set()
    for index in range(force.getNumExceptions()):
        i, j, qq, sigma, epsilon = force.getExceptionParameters(index)
        pair = (min(i, j), max(i, j))
        seen.add(pair)
        qq = qq.value_in_unit(unit.elementary_charge**2)
        sigma = sigma.value_in_unit(unit.angstrom)
        epsilon = epsilon.value_in_unit(unit.kilocalorie_per_mole)
        depth = separation.get(pair)
        s_coul, s_lj = _PAIR_SCALES.get(depth, (1.0, 1.0))
        # Tolerances are relative to this pair's own full interaction, so a
        # weakly charged pair is held as tightly as a strongly charged one.
        full_qq = charges[i] * charges[j]
        full_eps = (epsilons_kcal[i] * epsilons_kcal[j]) ** 0.5
        want_qq = s_coul * full_qq
        want_eps = s_lj * full_eps
        want_sigma = 0.5 * (sigmas_angstrom[i] + sigmas_angstrom[j])
        ok = _close(qq, want_qq, full_qq) and _close(epsilon, want_eps, full_eps)
        if ok and want_eps > 0.0:
            ok = _close(sigma, want_sigma, want_sigma)
        if not ok:
            where = f"{depth} bonds apart" if depth else "more than 3 bonds apart"
            raise ValueError(
                f"NonbondedForce exception {index} on atoms ({i}, {j}), {where}, has "
                f"chargeProd={qq:.6g} e^2, sigma={sigma:.6g} A, epsilon={epsilon:.6g} "
                f"kcal/mol; ferric-mm would evaluate chargeProd={want_qq:.6g}, "
                f"sigma={want_sigma:.6g}, epsilon={want_eps:.6g} for that pair "
                "(1-2/1-3 excluded, 1-4 scaled by 1/1.2 and 1/2), so the converted "
                "topology would have a different potential."
            )
    missing = sorted(set(separation) - seen)
    if missing:
        i, j = missing[0]
        raise ValueError(
            f"{len(missing)} atom pairs within three bonds have no NonbondedForce "
            f"exception (first: ({i}, {j}), {separation[(i, j)]} bonds apart); "
            "ferric-mm would exclude or scale them, so the converted topology "
            "would have a different potential."
        )


def topology_from_openmm(
    pdb_path: str | Path, forcefield: tuple[str, ...] = ("amber14-all.xml",)
) -> dict:
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
    from openmm import app

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(*forcefield)
    # rigidWater=False and constraints=None: a constrained bond is LEFT OUT of
    # HarmonicBondForce, and ferric-mm derives its exclusions from the bond
    # list, so a rigid water would reach ferric with no O-H bonds and its
    # intramolecular O-H pairs would get full Coulomb + LJ.
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    return topology_from_system(system)


def topology_from_system(system) -> dict:
    """The extraction half of `topology_from_openmm`: read the AMBER-form
    parameters out of an already-built `openmm.System` (same return value).

    Refuses (`ValueError`) a System the returned parameters would not
    reproduce: one with constraints (a constrained bond has no
    HarmonicBondForce entry, so ferric-mm would derive the wrong exclusions),
    one with a force other than harmonic bonds and angles, periodic torsions
    and a NonbondedForce (CMMotionRemover and barostats aside), one whose
    nonbonded exceptions are not the bond-derived AMBER set, or one with
    NonbondedForce parameter offsets. The nonbonded method (cutoff, PME) is
    not read: see the module docstring.
    """
    import openmm
    from openmm import unit

    if system.getNumConstraints() > 0:
        raise ValueError(
            f"System has {system.getNumConstraints()} constraints; constrained "
            "bonds are absent from HarmonicBondForce, so ferric-mm would derive "
            "wrong exclusions. Build it with constraints=None, rigidWater=False."
        )

    n_atoms = system.getNumParticles()
    charges: list[float] = [0.0] * n_atoms
    sigmas_angstrom: list[float] = [0.0] * n_atoms
    epsilons_kcal: list[float] = [0.0] * n_atoms
    bonds: list[Bond] = []
    angles: list[Angle] = []
    torsions: list[Torsion] = []
    nonbonded = []

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(b)
                r0_ang = r0.value_in_unit(unit.angstrom)
                k_amber = 0.5 * k.value_in_unit(
                    unit.kilocalorie_per_mole / unit.angstrom**2
                )
                bonds.append((i, j, k_amber, r0_ang))
        elif isinstance(force, openmm.HarmonicAngleForce):
            for a in range(force.getNumAngles()):
                i, j, k, theta0, k_theta = force.getAngleParameters(a)
                theta0_deg = theta0.value_in_unit(unit.degree)
                k_theta_amber = 0.5 * k_theta.value_in_unit(
                    unit.kilocalorie_per_mole / unit.radian**2
                )
                angles.append((i, j, k, k_theta_amber, theta0_deg))
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for t in range(force.getNumTorsions()):
                i, j, k, l, periodicity, phase, k_phi = force.getTorsionParameters(t)
                phase_deg = phase.value_in_unit(unit.degree)
                k_phi_kcal = k_phi.value_in_unit(unit.kilocalorie_per_mole)
                torsions.append((i, j, k, l, periodicity, k_phi_kcal, phase_deg))
        elif isinstance(force, openmm.NonbondedForce):
            offsets = (
                force.getNumParticleParameterOffsets()
                + force.getNumExceptionParameterOffsets()
            )
            if offsets > 0:
                raise ValueError(
                    f"NonbondedForce has {offsets} parameter offsets: its charges "
                    "and Lennard-Jones parameters depend on a global parameter, "
                    "so the base values read here are not the System's potential."
                )
            nonbonded.append(force)
            for p in range(force.getNumParticles()):
                q, sigma, epsilon = force.getParticleParameters(p)
                charges[p] = q.value_in_unit(unit.elementary_charge)
                sigmas_angstrom[p] = sigma.value_in_unit(unit.angstrom)
                epsilons_kcal[p] = epsilon.value_in_unit(unit.kilocalorie_per_mole)
        else:
            kind = type(force).__name__
            # These two kinds carry no potential energy; anything else does.
            if kind != "CMMotionRemover" and "Barostat" not in kind:
                raise ValueError(
                    f"System has a {kind}, which ferric-mm has no term for; the "
                    "converted topology would leave its energy out."
                )
    if len(nonbonded) > 1:
        raise ValueError(f"System has {len(nonbonded)} NonbondedForces; expected one")
    for force in nonbonded:
        _check_exceptions(force, bonds, charges, sigmas_angstrom, epsilons_kcal)

    return dict(
        n_atoms=n_atoms,
        charges=charges,
        sigmas_angstrom=sigmas_angstrom,
        epsilons_kcal=epsilons_kcal,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )
