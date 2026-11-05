#!/usr/bin/env python3
"""Generate OpenMM CutoffNonPeriodic references for `ferric-mm`'s opt-in
nonbonded cutoff (`crates/ferric-mm/src/cutoff.rs`).

Cross-code anchor for `crates/ferric-mm/tests/cutoff_vs_openmm.rs`. The
system is the ethanol topology of `gen_openmm_mm_refs.py` solvated by 12
flexible TIP3P-charged waters (seeded random placement). Each case builds the
same explicit-parameter OpenMM `System` with `NonbondedForce` set to
`CutoffNonPeriodic` and records energies and forces on the Reference
(double precision) platform:

    rf        cutoff 5.0 A, eps_rf 78.3, no switching function
    rf_switch cutoff 5.0 A, eps_rf 78.3, LJ switching from 4.0 A
    rf_eps1   cutoff 5.0 A, eps_rf 1.0,  no switching function
    short     cutoff 2.9 A, eps_rf 78.3, LJ switching from 2.2 A
              (several 1-4 pairs lie beyond the cutoff)

Before writing, every case is re-evaluated by an independent numpy model of
the form ferric implements (theory.md section 8): non-exception pairs with
r < r_c get reaction-field Coulomb q_i q_j (1/r + k_rf r^2 - c_rf) and LJ
times the switching function; 1-2/1-3 exceptions contribute nothing; 1-4
exceptions are plain scaled Coulomb + LJ at every distance (no cutoff, no
reaction field). The script exits nonzero if that model misses OpenMM by
more than 1e-9 relative, and also checks that the alternative reading (1-4
exceptions cut off at r_c) does NOT match the `short` case, so the reference
discriminates the two.

Usage:
    flock -o /tmp/ferric-compile.lock \
      /home/matt/qc/ferric/.venv/bin/python scripts/gen_openmm_mm_cutoff_refs.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import openmm
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_openmm_mm_refs import (
    ANGSTROM_TO_NM,
    KCAL_TO_KJ,
    SCALE_COUL_14,
    SCALE_LJ_14,
    _bond_pairs,
    _classify_pairs,
    ethanol_topology,
)

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"
ONE_4PI_EPS0 = 138.93545764438198  # OpenMM 8 Coulomb constant (measured: E of two unit charges 1 nm apart), kJ/mol nm / e^2

CASES = [
    dict(name="rf", cutoff=5.0, eps_rf=78.3, switch=None),
    dict(name="rf_switch", cutoff=5.0, eps_rf=78.3, switch=4.0),
    dict(name="rf_eps1", cutoff=5.0, eps_rf=1.0, switch=None),
    dict(name="short", cutoff=2.9, eps_rf=78.3, switch=2.2),
]


def solvated_ethanol(n_water=12, seed=20261105):
    topo = ethanol_topology()
    coords = [tuple(c) for c in topo["coords"]]
    charges = list(topo["charges"])
    lj = list(topo["lj"])
    bonds = list(topo["bonds"])
    angles = list(topo["angles"])
    torsions = list(topo["torsions"])
    center = np.mean(np.array(coords), axis=0)
    rng = np.random.default_rng(seed)
    r_oh, theta = 0.9572, math.radians(104.52)
    heavy = [np.array(coords[i]) for i in (0, 1, 2)]
    placed = 0
    while placed < n_water:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        o = center + d * rng.uniform(3.2, 7.0)
        if any(np.linalg.norm(o - h) < 2.8 for h in heavy):
            continue
        # random orthonormal frame for the two O-H directions
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        u, v = q[:, 0], q[:, 1]
        h1 = o + r_oh * (math.cos(theta / 2) * u + math.sin(theta / 2) * v)
        h2 = o + r_oh * (math.cos(theta / 2) * u - math.sin(theta / 2) * v)
        allc = np.array(coords)
        if np.min(np.linalg.norm(allc - o, axis=1)) < 2.4:
            continue
        if min(np.min(np.linalg.norm(allc - h, axis=1)) for h in (h1, h2)) < 1.6:
            continue
        base = len(coords)
        coords += [tuple(o), tuple(h1), tuple(h2)]
        charges += [-0.834, 0.417, 0.417]
        lj += [(3.15061, 0.1521), (0.0, 0.0), (0.0, 0.0)]
        bonds += [(base, base + 1, 553.0, r_oh), (base, base + 2, 553.0, r_oh)]
        angles += [(base + 1, base, base + 2, 100.0, 104.52)]
        heavy.append(o)
        placed += 1
    return dict(
        name="ethanol_12w",
        coords=coords,
        charges=charges,
        lj=lj,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )


def build_nonbonded_system(topo, case):
    """OpenMM System holding only the NonbondedForce (bonded terms are
    covered by vs_openmm.rs and are unchanged by the cutoff)."""
    n = len(topo["coords"])
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(case["cutoff"] * ANGSTROM_TO_NM)
    nb.setReactionFieldDielectric(case["eps_rf"])
    if case["switch"] is not None:
        nb.setUseSwitchingFunction(True)
        nb.setSwitchingDistance(case["switch"] * ANGSTROM_TO_NM)
    for q, (sigma_a, eps_kcal) in zip(topo["charges"], topo["lj"]):
        nb.addParticle(q, sigma_a * ANGSTROM_TO_NM, eps_kcal * KCAL_TO_KJ)
    exclusions, pairs14 = _classify_pairs(n, _bond_pairs(topo["bonds"]))
    for i, j in sorted(exclusions):
        nb.addException(i, j, 0.0, 0.1, 0.0)
    for i, j in sorted(pairs14):
        si, ei = topo["lj"][i]
        sj, ej = topo["lj"][j]
        nb.addException(
            i,
            j,
            topo["charges"][i] * topo["charges"][j] * SCALE_COUL_14,
            0.5 * (si + sj) * ANGSTROM_TO_NM,
            math.sqrt(ei * ej) * KCAL_TO_KJ * SCALE_LJ_14,
        )
    system.addForce(nb)
    return system, exclusions, pairs14


def openmm_eval(topo, case):
    system, exclusions, pairs14 = build_nonbonded_system(topo, case)
    ctx = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx.setPositions([tuple(c * ANGSTROM_TO_NM for c in xyz) for xyz in topo["coords"]])
    st = ctx.getState(getEnergy=True, getForces=True)
    e_kj = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_kj_nm = st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return e_kj, np.array(f_kj_nm), exclusions, pairs14


def model_eval(topo, case, exclusions, pairs14, cut_14=False):
    """Independent numpy model in OpenMM units (kJ/mol, nm)."""
    x = np.array(topo["coords"]) * ANGSTROM_TO_NM
    q = np.array(topo["charges"])
    sig = np.array([p[0] for p in topo["lj"]]) * ANGSTROM_TO_NM
    eps = np.array([p[1] for p in topo["lj"]]) * KCAL_TO_KJ
    rc = case["cutoff"] * ANGSTROM_TO_NM
    erf = case["eps_rf"]
    krf = (erf - 1.0) / ((2.0 * erf + 1.0) * rc**3)
    crf = 3.0 * erf / ((2.0 * erf + 1.0) * rc)
    rsw = None if case["switch"] is None else case["switch"] * ANGSTROM_TO_NM
    n = len(q)
    e = 0.0
    f = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in exclusions:
                continue
            d = x[i] - x[j]
            r = float(np.linalg.norm(d))
            s = 0.5 * (sig[i] + sig[j])
            ep = math.sqrt(eps[i] * eps[j])
            sr6 = (s / r) ** 6
            elj = 4 * ep * (sr6 * sr6 - sr6)
            dlj = 4 * ep * (-12 * sr6 * sr6 + 6 * sr6) / r
            qq = ONE_4PI_EPS0 * q[i] * q[j]
            if (i, j) in pairs14:
                if cut_14 and r >= rc:
                    continue
                epair = SCALE_LJ_14 * elj + SCALE_COUL_14 * qq / r
                dedr = SCALE_LJ_14 * dlj - SCALE_COUL_14 * qq / r**2
            else:
                if r >= rc:
                    continue
                sw, dsw = 1.0, 0.0
                if rsw is not None and r > rsw:
                    t = (r - rsw) / (rc - rsw)
                    sw = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
                    dsw = (-30 * t**2 + 60 * t**3 - 30 * t**4) / (rc - rsw)
                epair = elj * sw + qq * (1 / r + krf * r * r - crf)
                dedr = dlj * sw + elj * dsw + qq * (-1 / r**2 + 2 * krf * r)
            e += epair
            f[i] -= dedr * d / r
            f[j] += dedr * d / r
    return e, f


def main():
    topo = solvated_ethanol()
    x = np.array(topo["coords"])
    diam = max(np.linalg.norm(a - b) for a in x for b in x)
    out_cases = []
    for case in CASES:
        e_omm, f_omm, excl, p14 = openmm_eval(topo, case)
        e_mod, f_mod = model_eval(topo, case, excl, p14)
        rel_e = abs(e_mod - e_omm) / max(1.0, abs(e_omm))
        rel_f = np.max(np.abs(f_mod - f_omm)) / np.max(np.abs(f_omm))
        print(
            f"{case['name']}: E_omm {e_omm:.10f} kJ/mol  model rel_e {rel_e:.2e} rel_f {rel_f:.2e}"
        )
        if rel_e > 1e-9 or rel_f > 1e-9:
            raise SystemExit(f"{case['name']}: numpy model disagrees with OpenMM")
        # pairs near the cutoff/switch edge would make the reference fragile
        rc = case["cutoff"]
        n = len(x)
        edges = [rc] + ([case["switch"]] if case["switch"] is not None else [])
        gap = min(
            abs(np.linalg.norm(x[i] - x[j]) - edge)
            for i in range(n)
            for j in range(i + 1, n)
            for edge in edges
        )
        if gap < 1e-3:
            raise SystemExit(
                f"{case['name']}: a pair sits {gap:.1e} A from a cutoff edge"
            )
        if case["name"] == "short":
            n14_beyond = sum(1 for (i, j) in p14 if np.linalg.norm(x[i] - x[j]) >= rc)
            e_alt, _ = model_eval(topo, case, excl, p14, cut_14=True)
            print(
                f"  1-4 pairs beyond cutoff: {n14_beyond}; cut-1-4 model off by {abs(e_alt - e_omm):.3e} kJ/mol"
            )
            if n14_beyond == 0 or abs(e_alt - e_omm) < 1e-3:
                raise SystemExit("short case does not discriminate 1-4 cutoff handling")
        out_cases.append(
            dict(
                name=case["name"],
                cutoff_angstrom=case["cutoff"],
                reaction_field_dielectric=case["eps_rf"],
                switch_angstrom=case["switch"],
                nonbonded_energy_kcal=e_omm / KCAL_TO_KJ,
                nonbonded_forces_kcal_per_angstrom=(
                    f_omm / KCAL_TO_KJ * ANGSTROM_TO_NM
                ).tolist(),
            )
        )
    print(f"system diameter {diam:.3f} A, {len(x)} atoms")
    result = dict(
        name=topo["name"],
        openmm_version=openmm.__version__,
        n_atoms=len(x),
        coordinates_angstrom=[list(c) for c in topo["coords"]],
        charges=topo["charges"],
        lj_sigma_angstrom=[p[0] for p in topo["lj"]],
        lj_epsilon_kcal=[p[1] for p in topo["lj"]],
        bonds=topo["bonds"],
        angles=topo["angles"],
        torsions=topo["torsions"],
        scale_lj_14=SCALE_LJ_14,
        scale_coul_14=SCALE_COUL_14,
        coulomb_constant_kj_nm=ONE_4PI_EPS0,
        cases=out_cases,
    )
    out = REFDIR / "mm_cutoff_ethanol_12w_openmm.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
