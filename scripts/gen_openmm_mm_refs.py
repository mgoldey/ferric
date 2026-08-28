#!/usr/bin/env python3
"""Generate OpenMM references for `ferric-mm`'s AMBER-form force field.

Cross-code anchor for `crates/ferric-mm/tests/vs_openmm.rs`: builds two toy
topologies (ethane, ethanol) with EXPLICIT, made-up-but-realistic AMBER-ish
bonded/nonbonded parameters, feeds the SAME explicit parameters into an
OpenMM `System` (HarmonicBondForce / HarmonicAngleForce /
PeriodicTorsionForce / NonbondedForce), and records per-force-group energies
plus per-atom forces on the Reference (double precision) platform.

Unit/convention bridge (`ferric-mm` uses the AMBER convention with NO
leading 1/2 on the harmonic terms; OpenMM uses 1/2):

    k_openmm(bond)  = 2 * k_amber(bond)      [kJ/mol/nm^2 <- kcal/mol/A^2]
    k_openmm(angle) = 2 * k_amber(angle)     [kJ/mol/rad^2 <- kcal/mol/rad^2]
    PeriodicTorsionForce already uses E = k(1 + cos(n*phi - delta)), i.e. NO
    factor-of-2 conversion needed there (only the kcal->kJ unit conversion,
    handled by OpenMM's Quantity machinery).

NonbondedForce with NoCutoff carries LJ + Coulomb for every pair NOT listed
as an exception; `addException` zeroes 1-2/1-3 pairs and scales 1-4 pairs
(charge product / scale_coul_14, LJ epsilon * scale_lj_14, sigma
Lorentz-Berthelot-mixed as usual — OpenMM's own createExceptionsFromBonds
would do this for us, but we call addException explicitly per pair so the
scale factors are visible and match ferric-mm's DEFAULT_SCALE_LJ_14 (0.5) /
DEFAULT_SCALE_COUL_14 (1/1.2) exactly).

Records, in kcal/mol / kcal/mol/Angstrom / Angstrom, PER CASE:
  coordinates_angstrom, charges, lj (sigma_angstrom, epsilon_kcal),
  bonds/angles/torsions (explicit AMBER-unit parameters, matching
  `MmTopology::from_amber_units`'s tuple shapes), scale_lj_14/scale_coul_14,
  energy components (bond/angle/torsion/lj/coulomb/total),
  forces (per-atom, per-component: bond/angle/torsion/nonbonded/total) — LJ
  and Coulomb are reported together as "nonbonded" per OpenMM's
  NonbondedForce force-group convention (both live in the one Force object),
  matching against ferric's lj+coulomb sum rather than the two separately.

Usage:
    OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python \
        scripts/gen_openmm_mm_refs.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import openmm
from openmm import unit

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1

SCALE_LJ_14 = 0.5
SCALE_COUL_14 = 1.0 / 1.2

# Force groups, so per-term energies can be read back independently.
FG_BOND = 0
FG_ANGLE = 1
FG_TORSION = 2
FG_NONBONDED = 3


def ethane_topology():
    """Staggered ethane, matching the geometry construction convention in
    `crates/ferric-scf/tests/qmmm.rs::ethane_atoms` (but standalone here —
    ferric-mm does not depend on ferric-scf). Atom order: C0, C1, then H's
    2,4,6 on C0 and 3,5,7 on C1 (interleaved, matching the Rust fixture).
    """
    cc = 1.53
    ch = 1.09
    theta = math.radians(109.5)
    s, c = math.sin(theta), math.cos(theta)
    coords = [(0.0, 0.0, 0.0), (0.0, 0.0, cc)]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        # H on C0 pointing toward -z.
        coords.append((ch * s * math.cos(phi), ch * s * math.sin(phi), ch * c))
        # H on C1 pointing toward +z (mirrored).
        coords.append((ch * s * math.cos(phi), ch * s * math.sin(phi), cc - ch * c))

    charges = [-0.18, -0.18, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
    # (sigma_angstrom, epsilon_kcal) — generic aliphatic C/H (GAFF-like order
    # of magnitude, not sourced from a real force field file).
    lj = [(3.4, 0.109), (3.4, 0.109)] + [(2.6, 0.0157)] * 6

    bonds = [
        (0, 1, 310.0, cc),  # C-C
        (0, 2, 340.0, ch), (0, 4, 340.0, ch), (0, 6, 340.0, ch),
        (1, 3, 340.0, ch), (1, 5, 340.0, ch), (1, 7, 340.0, ch),
    ]
    # Angles: H-C-C (6) and H-C-H (3 per methyl x 2 = 6).
    angles = []
    tetra = 109.5
    for h in (2, 4, 6):
        angles.append((h, 0, 1, 50.0, tetra))  # H-C0-C1
    for h in (3, 5, 7):
        angles.append((h, 1, 0, 50.0, tetra))  # H-C1-C0
    hc0 = [2, 4, 6]
    for a in range(3):
        for b in range(a + 1, 3):
            angles.append((hc0[a], 0, hc0[b], 35.0, tetra))
    hc1 = [3, 5, 7]
    for a in range(3):
        for b in range(a + 1, 3):
            angles.append((hc1[a], 1, hc1[b], 35.0, tetra))

    # Torsions: H-C0-C1-H, one per (H on C0, H on C1) pair = 9, periodicity 3
    # (methyl rotor), k=0.16 kcal/mol (a realistic HC-CT-CT-HC-like torsion).
    torsions = []
    for hi in (2, 4, 6):
        for hj in (3, 5, 7):
            torsions.append((hi, 0, 1, hj, 3, 0.16, 0.0))

    return dict(
        name="ethane",
        coords=coords,
        charges=charges,
        lj=lj,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _vscale(v, s):
    return tuple(c * s for c in v)


def _vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _two_more_tetrahedral_dirs(d1, d2):
    """Given two unit vectors at the tetrahedral angle (109.47 deg) from a
    common vertex, return the other two directions completing the sp3 set
    (all six pairwise angles exactly 109.47 deg). Used to place the two CH2
    hydrogens on ethanol's C1 given its two heavy-atom bond directions
    (to C0 and to O) — verified numerically (see scripts/README or the
    commit message) rather than assumed.
    """
    bis = _normalize(_vadd(d1, d2))
    half = math.acos(_vdot(d1, bis))
    perp2 = _normalize(_vsub(d1, _vscale(bis, _vdot(d1, bis))))
    e2 = _vcross(bis, perp2)
    target = math.radians(109.4712206)
    ca = -math.cos(target) / math.cos(half)
    sa = math.sqrt(max(0.0, 1.0 - ca * ca))
    d3 = _normalize(_vadd(_vscale(bis, -ca), _vscale(e2, sa)))
    d4 = _normalize(_vadd(_vscale(bis, -ca), _vscale(e2, -sa)))
    return d3, d4


def ethanol_topology():
    """Ethanol: C0(methyl)-C1(CH2)-O-H, adds a polar O-H and 1-4 pairs
    across the C-O bond (e.g. H(methyl)-C0-C1-O and C0-C1-O-H torsions,
    and the H(methyl)...O / H(methyl)...H(hydroxyl) 1-4 nonbonded pairs).
    Atom order: 0=C0(methyl C), 1=C1(CH2), 2=O, 3=H(hydroxyl),
    4,5,6 = H on C0, 7,8 = H on C1.

    Every heavy-atom-adjacent bond/angle here is built from an exact
    tetrahedral (109.5 deg) construction and its distances/angles are
    verified at generation time (see `main()`'s geometry sanity check) —
    an earlier ad hoc placement of the two CH2 hydrogens put them at the
    wrong C-H distance (1.00 A instead of 1.09 A), which is exactly the
    kind of silent construction bug the exactness-first convention exists
    to catch; this replacement is solved analytically instead of guessed.
    """
    cc = 1.52
    co = 1.43
    oh = 0.96
    ch = 1.09
    theta = math.radians(109.5)
    s, c = math.sin(theta), math.cos(theta)

    c0 = (0.0, 0.0, 0.0)
    c1 = (0.0, 0.0, cc)
    o_dir = (math.sin(theta), 0.0, -math.cos(theta))
    o = _vadd(c1, _vscale(o_dir, co))

    # Hydroxyl H: rotate the O->C1 direction by 108 deg within the xz-plane
    # (both vectors already lie in that plane), giving an exact O-H bond
    # length and C1-O-H angle with a generic (non-planar-with-everything)
    # torsion.
    o_to_c1 = _normalize(_vsub(c1, o))
    ang = math.radians(108.0)
    ox, oz = o_to_c1[0], o_to_c1[2]
    rx = ox * math.cos(ang) - oz * math.sin(ang)
    rz = ox * math.sin(ang) + oz * math.cos(ang)
    h_dir = _normalize((rx, 0.0, rz))
    h_oh = _vadd(o, _vscale(h_dir, oh))

    coords = [c0, c1, o, h_oh]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        coords.append((ch * s * math.cos(phi), ch * s * math.sin(phi), ch * c))  # H on C0

    # Two H's on C1: the remaining two tetrahedral directions given C1's
    # bonds to C0 and O (exact 109.47 deg to both, and to each other).
    c1_to_c0 = _normalize(_vsub(c0, c1))
    h_dir_a, h_dir_b = _two_more_tetrahedral_dirs(c1_to_c0, o_dir)
    coords.append(_vadd(c1, _vscale(h_dir_a, ch)))
    coords.append(_vadd(c1, _vscale(h_dir_b, ch)))

    # indices: 0 C0, 1 C1, 2 O, 3 H(OH), 4,5,6 H(C0), 7,8 H(C1)
    charges = [-0.09, 0.14, -0.60, 0.40, 0.03, 0.03, 0.03, 0.02, 0.02]
    lj = [
        (3.4, 0.109),  # C0
        (3.4, 0.109),  # C1
        (3.0, 0.21),   # O
        (0.0, 0.0),    # H(OH) — zero LJ (typical AMBER polar H)
        (2.6, 0.0157), (2.6, 0.0157), (2.6, 0.0157),  # H on C0
        (2.5, 0.0157), (2.5, 0.0157),  # H on C1
    ]

    bonds = [
        (0, 1, 310.0, cc),   # C0-C1
        (1, 2, 320.0, co),   # C1-O
        (2, 3, 553.0, oh),   # O-H
        (0, 4, 340.0, ch), (0, 5, 340.0, ch), (0, 6, 340.0, ch),
        (1, 7, 340.0, ch), (1, 8, 340.0, ch),
    ]

    angles = [
        (1, 2, 3, 55.0, 108.5),   # C1-O-H
        (0, 1, 2, 50.0, 109.5),   # C0-C1-O
        (7, 1, 8, 33.0, 109.5),   # H-C1-H
        (0, 1, 7, 50.0, 109.5), (0, 1, 8, 50.0, 109.5),  # C0-C1-H
        (2, 1, 7, 50.0, 109.5), (2, 1, 8, 50.0, 109.5),  # O-C1-H
        (1, 0, 4, 50.0, 109.5), (1, 0, 5, 50.0, 109.5), (1, 0, 6, 50.0, 109.5),  # C1-C0-H
        (4, 0, 5, 35.0, 109.5), (4, 0, 6, 35.0, 109.5), (5, 0, 6, 35.0, 109.5),  # H-C0-H
    ]

    torsions = [
        # H-C0-C1-O (methyl rotor about C0-C1)
        (4, 0, 1, 2, 3, 0.16, 0.0), (5, 0, 1, 2, 3, 0.16, 0.0), (6, 0, 1, 2, 3, 0.16, 0.0),
        # H-C0-C1-H
        (4, 0, 1, 7, 3, 0.16, 0.0), (4, 0, 1, 8, 3, 0.16, 0.0),
        (5, 0, 1, 7, 3, 0.16, 0.0), (5, 0, 1, 8, 3, 0.16, 0.0),
        (6, 0, 1, 7, 3, 0.16, 0.0), (6, 0, 1, 8, 3, 0.16, 0.0),
        # C0-C1-O-H (hydroxyl rotor)
        (0, 1, 2, 3, 3, 0.25, 0.0),
        # H-C1-O-H
        (7, 1, 2, 3, 1, 0.30, 0.0), (8, 1, 2, 3, 1, 0.30, 0.0),
    ]

    return dict(
        name="ethanol",
        coords=coords,
        charges=charges,
        lj=lj,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )


def _bond_pairs(bonds):
    return [(b[0], b[1]) for b in bonds]


def _adjacency(n, bond_pairs):
    adj = {i: [] for i in range(n)}
    for a, b in bond_pairs:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _classify_pairs(n, bond_pairs):
    """Mirror ferric-mm's BFS classification exactly: returns
    (exclusions: set[(i,j) i<j], pairs14: set[(i,j) i<j])."""
    adj = _adjacency(n, bond_pairs)
    exclusions = set()
    pairs14 = set()
    for start in range(n):
        depth = {start: 0}
        frontier = [start]
        for d in range(1, 4):
            nxt = []
            for node in frontier:
                for nb in adj[node]:
                    if nb not in depth:
                        depth[nb] = d
                        nxt.append(nb)
            frontier = nxt
        for other, d in depth.items():
            if other == start:
                continue
            pair = (min(start, other), max(start, other))
            if d in (1, 2):
                exclusions.add(pair)
            elif d == 3:
                pairs14.add(pair)
    pairs14 -= exclusions
    return exclusions, pairs14


def build_system(topo):
    n = len(topo["coords"])
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)  # mass is irrelevant for a single-point energy/force eval

    bond_force = openmm.HarmonicBondForce()
    bond_force.setForceGroup(FG_BOND)
    for i, j, k_amber, r0 in topo["bonds"]:
        k_openmm = 2.0 * k_amber * KCAL_TO_KJ / (ANGSTROM_TO_NM ** 2)
        bond_force.addBond(i, j, r0 * ANGSTROM_TO_NM * unit.nanometer, k_openmm * unit.kilojoule_per_mole / unit.nanometer ** 2)
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    angle_force.setForceGroup(FG_ANGLE)
    for i, j, k, k_amber, theta0_deg in topo["angles"]:
        k_openmm = 2.0 * k_amber * KCAL_TO_KJ
        angle_force.addAngle(i, j, k, math.radians(theta0_deg) * unit.radian, k_openmm * unit.kilojoule_per_mole / unit.radian ** 2)
    system.addForce(angle_force)

    torsion_force = openmm.PeriodicTorsionForce()
    torsion_force.setForceGroup(FG_TORSION)
    for i, j, k, l, periodicity, k_amber, phase_deg in topo["torsions"]:
        k_openmm = k_amber * KCAL_TO_KJ
        torsion_force.addTorsion(i, j, k, l, periodicity, math.radians(phase_deg) * unit.radian, k_openmm * unit.kilojoule_per_mole)
    system.addForce(torsion_force)

    nb_force = openmm.NonbondedForce()
    nb_force.setForceGroup(FG_NONBONDED)
    nb_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for q, (sigma_a, eps_kcal) in zip(topo["charges"], topo["lj"]):
        sigma_nm = sigma_a * ANGSTROM_TO_NM
        eps_kj = eps_kcal * KCAL_TO_KJ
        nb_force.addParticle(q * unit.elementary_charge, sigma_nm * unit.nanometer, eps_kj * unit.kilojoule_per_mole)

    exclusions, pairs14 = _classify_pairs(n, _bond_pairs(topo["bonds"]))
    for (i, j) in sorted(exclusions):
        nb_force.addException(i, j, 0.0 * unit.elementary_charge ** 2, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    for (i, j) in sorted(pairs14):
        qi, qj = topo["charges"][i], topo["charges"][j]
        sigma_i, eps_i = topo["lj"][i]
        sigma_j, eps_j = topo["lj"][j]
        sigma_mix_nm = 0.5 * (sigma_i + sigma_j) * ANGSTROM_TO_NM
        eps_mix_kj = math.sqrt(eps_i * eps_j) * KCAL_TO_KJ if eps_i > 0 and eps_j > 0 else 0.0
        q_scaled = qi * qj * SCALE_COUL_14 * unit.elementary_charge ** 2
        eps_scaled = eps_mix_kj * SCALE_LJ_14 * unit.kilojoule_per_mole
        nb_force.addException(i, j, q_scaled, sigma_mix_nm * unit.nanometer, eps_scaled)
    system.addForce(nb_force)

    return system, exclusions, pairs14


def run_case(topo):
    system, exclusions, pairs14 = build_system(topo)
    n = len(topo["coords"])
    positions = [(x * ANGSTROM_TO_NM, y * ANGSTROM_TO_NM, z * ANGSTROM_TO_NM) for x, y, z in topo["coords"]]

    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    components = {}
    forces = {}
    for label, fg in (("bond", FG_BOND), ("angle", FG_ANGLE), ("torsion", FG_TORSION), ("nonbonded", FG_NONBONDED)):
        state = context.getState(getEnergy=True, getForces=True, groups={fg})
        e_kcal = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        components[label] = e_kcal
        f_kj_nm = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        # kJ/mol/nm -> kcal/mol/Angstrom
        f_kcal_ang = (f_kj_nm / KCAL_TO_KJ) * ANGSTROM_TO_NM
        forces[label] = f_kcal_ang.tolist()

    state_all = context.getState(getEnergy=True, getForces=True)
    total_e = state_all.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    total_f = (state_all.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) / KCAL_TO_KJ) * ANGSTROM_TO_NM

    return dict(
        name=topo["name"],
        n_atoms=n,
        coordinates_angstrom=topo["coords"],
        charges=topo["charges"],
        lj_sigma_angstrom=[p[0] for p in topo["lj"]],
        lj_epsilon_kcal=[p[1] for p in topo["lj"]],
        bonds=topo["bonds"],
        angles=topo["angles"],
        torsions=topo["torsions"],
        scale_lj_14=SCALE_LJ_14,
        scale_coul_14=SCALE_COUL_14,
        exclusions=sorted([list(p) for p in exclusions]),
        pairs14=sorted([list(p) for p in pairs14]),
        energy_kcal=dict(
            bond=components["bond"],
            angle=components["angle"],
            torsion=components["torsion"],
            nonbonded=components["nonbonded"],
            total=total_e,
        ),
        forces_kcal_per_angstrom=dict(
            bond=forces["bond"],
            angle=forces["angle"],
            torsion=forces["torsion"],
            nonbonded=forces["nonbonded"],
            total=total_f.tolist(),
        ),
    )


def _sanity_check_geometry(topo):
    """Every bond's built coordinates must reproduce its own r0 to high
    precision -- a construction bug here (e.g. the ethanol CH2-hydrogen
    placement bug found and fixed during generation) silently produces a
    physically wrong but still-computable reference, which is exactly the
    kind of error the repo's exactness-first convention exists to catch
    before it contaminates a downstream Rust test tolerance.
    """
    coords = topo["coords"]
    for i, j, _k, r0 in topo["bonds"]:
        r = math.dist(coords[i], coords[j])
        if abs(r - r0) > 1e-6:
            raise AssertionError(f"{topo['name']}: bond ({i},{j}) built at r={r:.6f}, expected r0={r0:.6f}")


def main():
    REFDIR.mkdir(parents=True, exist_ok=True)
    for topo in (ethane_topology(), ethanol_topology()):
        _sanity_check_geometry(topo)
        result = run_case(topo)
        out_path = REFDIR / f"mm_{topo['name']}_openmm.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path} (total energy {result['energy_kcal']['total']:.6f} kcal/mol)")


if __name__ == "__main__":
    main()
