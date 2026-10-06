#!/usr/bin/env python3
"""Generate OpenMM references for the QM-MM Lennard-Jones term across a
covalent QM/MM cut (`crates/ferric-scf/tests/qmmm_lj_vs_openmm.rs`).

The question is whether `ferric_scf::qmmm::qmmm_mm_terms` applies the force
field's own nonbonded rules to QM-MM pairs that are bonded across the cut:
1-2 and 1-3 pairs excluded, 1-4 pairs scaled by the 1-4 LJ factor.

Independent construction. The exclusion and 1-4 sets here come from OpenMM's
own `NonbondedForce.createExceptionsFromBonds`, NOT from a port of ferric-mm's
BFS (which `gen_openmm_mm_refs.py` mirrors). Charges are all zero, so the
NonbondedForce energy is pure LJ. The QM-MM part is isolated by switching LJ
off per region, BEFORE the exceptions are created (so the 1-4 exception
epsilons see the zero too):

    E_QMMM = E(all) - E(QM epsilon = 0) - E(MM epsilon = 0)
    E_MMMM = E(QM epsilon = 0)

and the same for the forces. E(QM epsilon = 0) holds the MM-MM pairs only and
E(MM epsilon = 0) the QM-QM pairs only, so the difference is exactly the
cross term.

System: ethanol (geometry and bonds from `gen_openmm_mm_refs.ethanol_topology`)
with realistic LJ sizes (C 3.4 A). The hydroxyl H is given a small nonzero LJ
(1.0 A, 0.046 kcal/mol) so it carries a 1-5 pair across the C-C cut. Cutting
C0-C1 gives every pair class across the cut: 1-2 (C0-C1), 1-3 (C0-O, C0-H7/8,
C1-H4/5/6), 1-4 (O-H4/5/6, H7/8-H4/5/6, C0-HO) and 1-5 (HO-H4/5/6).

Cases: QM = CH2OH (scale_lj_14 0.5), QM = CH3 (0.5), QM = CH2OH (0.75 — a
non-AMBER LJ 1-4 factor, so a hardcoded 0.5 or the Coulomb factor 1/1.2 in its
place would show).

Usage:
    OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python \
        scripts/gen_openmm_qmmm_lj_refs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openmm
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_openmm_mm_refs import (  # noqa: E402
    ANGSTROM_TO_NM,
    KCAL_TO_KJ,
    ethanol_topology,
)

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

# The Coulomb 1-4 factor is irrelevant (all charges zero) but is passed to
# OpenMM anyway, set to AMBER's 1/1.2, distinct from every LJ factor used.
SCALE_COUL_14 = 1.0 / 1.2


def realistic_ethanol():
    topo = ethanol_topology()
    lj = list(topo["lj"])
    lj[3] = (1.0, 0.046)  # hydroxyl H: small but nonzero, carries a 1-5 pair
    topo["lj"] = lj
    return topo


def lj_energy_forces(topo, eps_zero: set[int], scale_lj_14: float):
    """Pure-LJ NonbondedForce energy (kcal/mol) and forces (kcal/mol/A),
    with epsilon zeroed on `eps_zero` before OpenMM derives the exceptions."""
    n = len(topo["coords"])
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for i, (sigma_a, eps_kcal) in enumerate(topo["lj"]):
        eps = 0.0 if i in eps_zero else eps_kcal * KCAL_TO_KJ
        nb.addParticle(
            0.0 * unit.elementary_charge,
            sigma_a * ANGSTROM_TO_NM * unit.nanometer,
            eps * unit.kilojoule_per_mole,
        )
    nb.createExceptionsFromBonds(
        [(b[0], b[1]) for b in topo["bonds"]], SCALE_COUL_14, scale_lj_14
    )
    system.addForce(nb)
    ctx = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0 * unit.femtosecond),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx.setPositions(
        [
            (x * ANGSTROM_TO_NM, y * ANGSTROM_TO_NM, z * ANGSTROM_TO_NM)
            for x, y, z in topo["coords"]
        ]
    )
    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    f = (
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        / KCAL_TO_KJ
        * ANGSTROM_TO_NM
    )
    return e, f


def run_case(name, topo, qm, scale_lj_14):
    n = len(topo["coords"])
    mm = [i for i in range(n) if i not in qm]
    e_all, f_all = lj_energy_forces(topo, set(), scale_lj_14)
    e_qm0, f_qm0 = lj_energy_forces(topo, set(qm), scale_lj_14)
    e_mm0, f_mm0 = lj_energy_forces(topo, set(mm), scale_lj_14)
    e_unexcl, _ = lj_energy_forces(
        {**topo, "bonds": []}, set(), scale_lj_14
    )  # no exceptions at all: the pre-fix (every-pair) construction's total
    e_unexcl_qm0, _ = lj_energy_forces({**topo, "bonds": []}, set(qm), scale_lj_14)
    e_unexcl_mm0, _ = lj_energy_forces({**topo, "bonds": []}, set(mm), scale_lj_14)
    return dict(
        name=name,
        n_atoms=n,
        qm_indices=sorted(qm),
        coordinates_angstrom=topo["coords"],
        lj_sigma_angstrom=[p[0] for p in topo["lj"]],
        lj_epsilon_kcal=[p[1] for p in topo["lj"]],
        bonds=[[b[0], b[1]] for b in topo["bonds"]],
        scale_lj_14=scale_lj_14,
        scale_coul_14=SCALE_COUL_14,
        qm_mm_lj_kcal=e_all - e_qm0 - e_mm0,
        mm_mm_lj_kcal=e_qm0,
        qm_mm_lj_no_exclusions_kcal=e_unexcl - e_unexcl_qm0 - e_unexcl_mm0,
        qm_mm_lj_forces_kcal_per_angstrom=(f_all - f_qm0 - f_mm0).tolist(),
        mm_mm_lj_forces_kcal_per_angstrom=f_qm0.tolist(),
    )


def main():
    topo = realistic_ethanol()
    cases = [
        run_case("ethanol_qm_ch2oh", topo, [1, 2, 3, 7, 8], 0.5),
        run_case("ethanol_qm_ch3", topo, [0, 4, 5, 6], 0.5),
        run_case("ethanol_qm_ch2oh_lj14_075", topo, [1, 2, 3, 7, 8], 0.75),
    ]
    out = REFDIR / "qmmm_lj_ethanol_openmm.json"
    out.write_text(
        json.dumps({"openmm_version": openmm.__version__, "cases": cases}, indent=2)
    )
    for c in cases:
        print(
            f"{c['name']}: QM-MM LJ {c['qm_mm_lj_kcal']:.10f} kcal/mol "
            f"(no exclusions: {c['qm_mm_lj_no_exclusions_kcal']:.4f}), "
            f"MM-MM LJ {c['mm_mm_lj_kcal']:.10f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
