"""OpenMM references for the validation row "ferric-mm vs OpenMM".

Consumer: crates/ferric-mm/tests/validation_mm.rs
Output:   testdata/reference/validation/mm/<system>.json
Inputs:   testdata/molecules/validation/mm/*.pdb (committed; built by the
          `build-inputs` group below) and tools/active_site/tests/ala_ala.pdb

WHAT IS COMPARED
    ferric-mm (crates/ferric-mm) evaluates AMBER-form bonded + nonbonded
    energies and analytic gradients from explicit parameters. The parameters
    here come from a REAL force field (amber14-all.xml = ff14SB, plus
    amber14/tip3p.xml) applied by OpenMM, and are read out of the OpenMM
    System by the production extraction code,
    tools/active_site/mm_topology.py::topology_from_system (the half of
    topology_from_openmm that does not build the System). The Rust test feeds
    that dict to MmTopology::from_amber_units, i.e. the same path the Python
    bindings use. ferric derives its OWN 1-2/1-3 exclusions and 1-4 pairs from
    the bond list and applies global 1-4 scales (0.5 LJ, 1/1.2 Coulomb);
    OpenMM uses the force field's per-pair exceptions. The exception list is
    exported so the Rust side can compare the two pair sets directly.

    ferric-mm supports: harmonic bonds, harmonic angles, periodic torsions
    (propers and impropers alike), Lennard-Jones with Lorentz-Berthelot
    mixing, bare Coulomb, bond-graph exclusions, global 1-4 scaling, NO
    cutoff and NO periodicity. So every OpenMM System here is NoCutoff,
    non-periodic, unconstrained (constraints=None, rigidWater=False: a
    constrained bond is absent from HarmonicBondForce, so ferric would miss
    its exclusion), and has no CMMotionRemover. A System containing any other
    Force (CMAP, GB, custom forces) is REFUSED.

REFERENCE PLATFORM
    OpenMM 'Reference' platform: double precision throughout, deterministic.

PER-TERM ENERGIES AND FORCES
    Each Force gets its own force group (bond 0, angle 1, torsion 2,
    nonbonded 3) and is evaluated with getState(groups={g}). NonbondedForce
    holds LJ and Coulomb together; they are separated by evaluating two
    CLONES of the NonbondedForce (XmlSerializer round trip) in single-force
    Systems: "coulomb" = every particle epsilon and every exception epsilon
    set to 0 (charges and chargeProds kept); "lj" = every particle charge and
    every exception chargeProd set to 0 (sigma/epsilon kept). Both terms are
    pairwise-additive and linear in their own parameters, so coulomb + lj
    must reproduce the unsplit group-3 energy and forces; the generator
    records that residual and REFUSES above SPLIT_TOL.

SYSTEMS
    ala_ala          zwitterionic ALA-ALA (23 atoms), the committed
                     tools/active_site/tests/ala_ala.pdb fixture.
    ace_phe_nme      capped dipeptide ACE-PHE-NME (6-ring: para pairs are
                     1-4, meta pairs excluded). Heavy atoms from 7LCJ chain R
                     60-62 (LEU60 C/O/CA -> ACE C/O/CH3, PHE61, CYS62 N/CA ->
                     NME N/C).
    trp_pro_asp      zwitterionic TRP-PRO-ASP from 7LCJ chain R 72-74 (fused
                     indole bicycle, proline 5-ring, NH3+, COO-, ASP-; net
                     charge -1). OXT placed in the CA-C-O plane.
    ace_phe_nme_3wat ace_phe_nme + the 3 TIP3P waters nearest the peptide
                     from Modeller.addSolvent (flexible TIP3P).
    ace_phe_nme_phase37  SYNTHETIC parameters: ace_phe_nme with every torsion
                     phase shifted by +37 degrees in the OpenMM System. ff14SB
                     phases are all 0 or 180 degrees, where cos(n*phi - delta)
                     is even in phi, so a dihedral SIGN-convention error is
                     invisible on the real force field; this case makes it
                     visible.
    Hydrogens are added by Modeller.addHydrogens(ff14SB) at pH 7, and each
    built structure is relaxed by LocalEnergyMinimizer (Reference platform,
    tolerance 10 kJ/mol/nm, 500 iterations) before being written, so the
    committed PDB has no crystal-hydrogen clashes. The 3-decimal PDB
    coordinates ARE the reference geometry "pdb".

GEOMETRIES (per system)
    pdb             the committed PDB coordinates
    perturbed_0p03  + N(0, 0.03 A) per Cartesian component
    perturbed_0p10  + N(0, 0.10 A) per Cartesian component
    numpy default_rng(PERTURB_SEED + system index); coordinates written at
    full double precision in Angstrom. Perturbation keeps forces away from
    the near-minimum the relaxed PDB sits at.

UNITS
    Native OpenMM: kJ/mol, kJ/mol/nm (forces F = -dE/dR).
    ferric: Hartree, Hartree/Bohr, gradients dE/dR. Conversion uses ferric's
    own constants (crates/ferric-mm/src/units.rs):
        1 kcal/mol = 4.184 kJ/mol (exact, thermochemical calorie)
        1 Hartree  = 627.509474 kcal/mol
        1 Bohr     = 0.52917721092 Angstrom (CODATA 2010)
    so dE/dR [Ha/Bohr] = -F [kJ/mol/nm] / 4.184 / 627.509474 * 0.052917721092.
    Topology parameters are exported in the AMBER units from_amber_units
    takes (e, Angstrom, kcal/mol, degrees; bond/angle k WITHOUT the 1/2).

Run:
    scripts/validation/run_slot.sh --light -- \\
        python scripts/validation/gen_mm.py [build-inputs]
    (no argument: write references from the committed inputs; `build-inputs`
    builds any MISSING input PDB from 7LCJ and never overwrites one:
    Modeller.addHydrogens is not deterministic, so the committed PDBs, pinned
    by input_pdb_sha256 in each reference, are the inputs of record)
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path.insert(0, str(common.ROOT))
from tools.active_site.mm_topology import (  # noqa: E402
    topology_from_openmm,
    topology_from_system,
)

ROW = "mm"
ROW_NAME = "ferric-mm vs OpenMM"
GEN = "scripts/validation/gen_mm.py"
INPUT_DIR = common.MOL_DIR / "mm"
SOURCE_PDB = common.ROOT / "testdata/molecules/c9_systems/danuglipron/7LCJ.pdb"
ALA_ALA_PDB = common.ROOT / "tools/active_site/tests/ala_ala.pdb"

FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")
PERTURB_SEED = 20260925
PERTURB_SIGMAS = (("perturbed_0p03", 0.03), ("perturbed_0p10", 0.10))
PHASE_SHIFT_DEG = 37.0
# coulomb + lj clones must reproduce the unsplit NonbondedForce group to this
# relative accuracy (both are exact pair sums; measured residual is recorded).
SPLIT_TOL = 1e-12

KJ_PER_KCAL = 4.184
KCAL_PER_HARTREE = 627.509474
BOHR_IN_NM = common.BOHR_IN_ANGSTROM / 10.0
KJ_TO_HARTREE = 1.0 / (KJ_PER_KCAL * KCAL_PER_HARTREE)
# dE/dR [Ha/Bohr] = -F [kJ/mol/nm] * FORCE_KJNM_TO_GRAD_HA_BOHR
FORCE_KJNM_TO_GRAD_HA_BOHR = KJ_TO_HARTREE * BOHR_IN_NM

GROUPS = {"bond": 0, "angle": 1, "torsion": 2, "nonbonded": 3}
FORCE_GROUP = {
    "HarmonicBondForce": "bond",
    "HarmonicAngleForce": "angle",
    "PeriodicTorsionForce": "torsion",
    "NonbondedForce": "nonbonded",
}

# (system, input pdb, torsion phase shift, description)
SYSTEMS = [
    ("ala_ala", ALA_ALA_PDB, 0.0, "zwitterionic ALA-ALA (committed tools fixture)"),
    ("ace_phe_nme", INPUT_DIR / "ace_phe_nme.pdb", 0.0, "capped dipeptide ACE-PHE-NME"),
    (
        "trp_pro_asp",
        INPUT_DIR / "trp_pro_asp.pdb",
        0.0,
        "zwitterionic tripeptide TRP-PRO-ASP (net charge -1)",
    ),
    (
        "ace_phe_nme_3wat",
        INPUT_DIR / "ace_phe_nme_3wat.pdb",
        0.0,
        "ACE-PHE-NME + 3 flexible TIP3P waters",
    ),
    (
        "ace_phe_nme_phase37",
        INPUT_DIR / "ace_phe_nme.pdb",
        PHASE_SHIFT_DEG,
        "SYNTHETIC: ACE-PHE-NME with every torsion phase + 37 degrees",
    ),
]


# ---------------------------------------------------------------------------
# Input construction (group `build-inputs`)
# ---------------------------------------------------------------------------


def _source_atoms():
    """{resSeq: [(name, resname, xyz)]} for 7LCJ chain R ATOM records."""
    out: dict[int, list] = {}
    for line in SOURCE_PDB.read_text().splitlines():
        if not line.startswith("ATOM") or line[21] != "R":
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip()
        seq = int(line[22:26])
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        out.setdefault(seq, []).append((name, resname, np.array(xyz)))
    return out


def _pdb_text(residues):
    """residues: [(resname, [(atom name, element, xyz)])] -> heavy-atom PDB."""
    lines = []
    serial = 1
    for rnum, (resname, atoms) in enumerate(residues, start=1):
        for name, elem, xyz in atoms:
            nm = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {nm} {resname:>3} A{rnum:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2}"
            )
            serial += 1
    lines += ["TER", "END"]
    return "\n".join(lines) + "\n"


def _elem(name):
    return name[0]


def _heavy_ace_phe_nme(src):
    leu = {n: x for n, _, x in src[60]}
    cys = {n: x for n, _, x in src[62]}
    phe = [(n, _elem(n), x) for n, _, x in src[61]]
    return [
        ("ACE", [("CH3", "C", leu["CA"]), ("C", "C", leu["C"]), ("O", "O", leu["O"])]),
        ("PHE", phe),
        ("NME", [("N", "N", cys["N"]), ("C", "C", cys["CA"])]),
    ]


def _heavy_trp_pro_asp(src):
    res = []
    for seq in (72, 73, 74):
        atoms = [(n, _elem(n), x) for n, _, x in src[seq]]
        res.append((src[seq][0][1], atoms))
    asp = {n: x for n, _, x in res[-1][1]}
    # OXT in the CA-C-O plane, 1.25 A from C, bisecting the exterior of CA-C-O.
    c, ca, o = asp["C"], asp["CA"], asp["O"]
    u1 = (ca - c) / np.linalg.norm(ca - c)
    u2 = (o - c) / np.linalg.norm(o - c)
    d = -(u1 + u2)
    oxt = c + 1.25 * d / np.linalg.norm(d)
    res[-1][1].append(("OXT", "O", oxt))
    return res


def _modeller_from_heavy(residues, ff):
    from openmm import app

    pdb = app.PDBFile(io.StringIO(_pdb_text(residues)))
    m = app.Modeller(pdb.topology, pdb.positions)
    m.addHydrogens(ff, pH=7.0)
    return m


def _keep_nearest_waters(m, ff, n_keep):
    from openmm import unit

    m.addSolvent(ff, model="tip3p", padding=0.6 * unit.nanometer, neutralize=False)
    pos = np.array(m.positions.value_in_unit(unit.angstrom))
    solute = [a.index for a in m.topology.atoms() if a.residue.name != "HOH"]
    waters = [r for r in m.topology.residues() if r.name == "HOH"]

    def dmin(r):
        o = next(a for a in r.atoms() if a.element.symbol == "O")
        return float(np.min(np.linalg.norm(pos[solute] - pos[o.index], axis=1)))

    ranked = sorted(waters, key=lambda r: (dmin(r), r.index))
    m.delete(ranked[n_keep:])
    m.topology.setPeriodicBoxVectors(None)
    return m


def _relax(m, ff):
    import openmm
    from openmm import app

    system = ff.createSystem(
        m.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(m.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 500)
    pos = ctx.getState(getPositions=True).getPositions()
    buf = io.StringIO()
    app.PDBFile.writeFile(m.topology, pos, buf, keepIds=False)
    return buf.getvalue()


def build_inputs():
    from openmm import app

    ff = app.ForceField(*FORCEFIELD)
    src = _source_atoms()
    builds = {}
    m = _modeller_from_heavy(_heavy_ace_phe_nme(src), ff)
    builds["ace_phe_nme.pdb"] = _relax(m, ff)
    m = _modeller_from_heavy(_heavy_trp_pro_asp(src), ff)
    builds["trp_pro_asp.pdb"] = _relax(m, ff)
    m = _modeller_from_heavy(_heavy_ace_phe_nme(src), ff)
    m = _keep_nearest_waters(m, ff, 3)
    builds["ace_phe_nme_3wat.pdb"] = _relax(m, ff)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in builds.items():
        path = INPUT_DIR / name
        if path.exists():
            # Modeller.addHydrogens is NOT deterministic (a fresh build moves
            # hydrogens by up to ~0.16 A), so a rebuild can never be checked
            # against the committed file; the committed PDB, pinned by the
            # sha256 in every reference's provenance, is the input of record.
            print(
                f"{name}: exists, NOT overwritten (delete it deliberately to rebuild)"
            )
            continue
        path.write_text(text)
        print(f"{name}: written")


# ---------------------------------------------------------------------------
# Reference evaluation
# ---------------------------------------------------------------------------


def _build_system(pdb_path, phase_shift_deg):
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(*FORCEFIELD)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    if system.usesPeriodicBoundaryConditions():
        raise RuntimeError(f"{pdb_path}: periodic System; ferric-mm has no PBC")
    for f in system.getForces():
        name = type(f).__name__
        if name not in FORCE_GROUP:
            raise RuntimeError(f"{pdb_path}: Force {name} has no ferric-mm counterpart")
        f.setForceGroup(GROUPS[FORCE_GROUP[name]])
        if name == "NonbondedForce":
            assert f.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff
        if name == "PeriodicTorsionForce" and phase_shift_deg:
            for t in range(f.getNumTorsions()):
                i, j, k, m, n, phase, kphi = f.getTorsionParameters(t)
                f.setTorsionParameters(
                    t, i, j, k, m, n, phase + phase_shift_deg * unit.degree, kphi
                )
    pos = np.array(pdb.positions.value_in_unit(unit.angstrom))
    return pdb, system, pos


def _nonbonded(system):
    import openmm

    return next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))


def _split_systems(system):
    """Single-force Systems holding a Coulomb-only and an LJ-only clone."""
    import openmm

    nb = _nonbonded(system)
    out = {}
    for label in ("coulomb", "lj"):
        s = openmm.System()
        for p in range(system.getNumParticles()):
            s.addParticle(system.getParticleMass(p))
        f = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(nb))
        f.setForceGroup(0)
        for p in range(f.getNumParticles()):
            q, sig, eps = f.getParticleParameters(p)
            if label == "coulomb":
                f.setParticleParameters(p, q, sig, 0.0)
            else:
                f.setParticleParameters(p, 0.0, sig, eps)
        for x in range(f.getNumExceptions()):
            i, j, qq, sig, eps = f.getExceptionParameters(x)
            if label == "coulomb":
                f.setExceptionParameters(x, i, j, qq, sig, 0.0)
            else:
                f.setExceptionParameters(x, i, j, 0.0, sig, eps)
        s.addForce(f)
        out[label] = s
    return out


def _context(system):
    import openmm

    integ = openmm.VerletIntegrator(0.001)
    plat = openmm.Platform.getPlatformByName("Reference")
    return openmm.Context(system, integ, plat)


def _eval(ctx, pos_ang, groups=None):
    from openmm import unit

    ctx.setPositions(pos_ang * 0.1)  # nm
    kw = {} if groups is None else {"groups": groups}
    st = ctx.getState(getEnergy=True, getForces=True, **kw)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.array(
        st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
    )
    return e, f


def _coulomb_constant():
    """OpenMM's e^2/(4 pi eps0) in kJ/mol*nm, MEASURED (two unit charges 1 nm
    apart), and the same number in ferric's units, where it is exactly 1 by
    construction (Hartree*Bohr). The difference from 1 is a CONSTANTS
    mismatch (OpenMM uses CODATA 2018; ferric's 627.509474 kcal/mol/Ha and
    0.52917721092 A/Bohr are older), so every Coulomb energy and force in
    the reference is larger than ferric's by exactly this ratio."""
    import openmm

    s = openmm.System()
    f = openmm.NonbondedForce()
    for _ in range(2):
        s.addParticle(1.0)
        f.addParticle(1.0, 1.0, 0.0)
    s.addForce(f)
    e, _ = _eval(_context(s), np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    return e, e * KJ_TO_HARTREE / BOHR_IN_NM


def _exceptions(system, top):
    """Classify NonbondedForce exceptions as excluded / scaled 1-4 / ambiguous
    (both the excluded and the scaled value would be exactly zero), and
    measure the 1-4 scale factors the force field actually used."""
    from openmm import unit

    nb = _nonbonded(system)
    q = np.array(top["charges"])
    eps = np.array(top["epsilons_kcal"])
    sig = np.array(top["sigmas_angstrom"])
    excluded, scaled, ambiguous = [], [], []
    coul_ratios, lj_ratios, sig_err = [], [], 0.0
    for x in range(nb.getNumExceptions()):
        i, j, qq, s, e = nb.getExceptionParameters(x)
        qq = qq.value_in_unit(unit.elementary_charge**2)
        e = e.value_in_unit(unit.kilocalorie_per_mole)
        s = s.value_in_unit(unit.angstrom)
        pair = sorted((int(i), int(j)))
        zero_if_14 = q[i] * q[j] == 0.0 and eps[i] * eps[j] == 0.0
        if qq == 0.0 and e == 0.0:
            (ambiguous if zero_if_14 else excluded).append(pair)
            continue
        scaled.append(pair)
        if q[i] * q[j] != 0.0:
            coul_ratios.append(qq / (q[i] * q[j]))
        if eps[i] * eps[j] != 0.0:
            lj_ratios.append(e / np.sqrt(eps[i] * eps[j]))
            sig_err = max(sig_err, abs(s - 0.5 * (sig[i] + sig[j])))
    coul_ratios = np.array(coul_ratios)
    lj_ratios = np.array(lj_ratios)
    scales = {
        "scale_coul_14": float(np.mean(coul_ratios)),
        "scale_coul_14_spread": float(np.ptp(coul_ratios)),
        "scale_lj_14": float(np.mean(lj_ratios)),
        "scale_lj_14_spread": float(np.ptp(lj_ratios)),
        "sigma_14_max_dev_from_lorentz_angstrom": float(sig_err),
    }
    # ferric applies ONE global pair of 1-4 scales; a per-pair force field
    # would not be representable, so refuse it here rather than fail obscurely.
    if scales["scale_coul_14_spread"] > 1e-12 or scales["scale_lj_14_spread"] > 1e-12:
        raise RuntimeError(f"non-uniform 1-4 scales: {scales}")
    if sig_err > 1e-12:
        raise RuntimeError(f"1-4 sigma is not the Lorentz mean: {sig_err}")
    return {
        "excluded": sorted(excluded),
        "scaled14": sorted(scaled),
        "ambiguous": sorted(ambiguous),
    }, scales


def _min_pair_distance(pos, exc):
    skip = {tuple(p) for p in exc["excluded"]}
    n = len(pos)
    best = np.inf
    for i in range(n):
        d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        for k, dist in enumerate(d):
            j = i + 1 + k
            if (i, j) not in skip and dist < best:
                best = dist
    return float(best)


def _geometry(label, sigma, pos, ctx_full, ctx_split, exc):
    energy_kj, forces = {}, {}
    for term, g in GROUPS.items():
        e, f = _eval(ctx_full, pos, groups={g})
        energy_kj[term], forces[term] = e, f
    for term, ctx in ctx_split.items():
        e, f = _eval(ctx, pos)
        energy_kj[term], forces[term] = e, f
    e_tot, f_tot = _eval(ctx_full, pos)
    energy_kj["total"], forces["total"] = e_tot, f_tot

    split_e = abs(energy_kj["coulomb"] + energy_kj["lj"] - energy_kj["nonbonded"])
    split_f = float(
        np.abs(forces["coulomb"] + forces["lj"] - forces["nonbonded"]).max()
    )
    rel_e = split_e / max(abs(energy_kj["nonbonded"]), 1.0)
    rel_f = split_f / max(float(np.abs(forces["nonbonded"]).max()), 1.0)
    if rel_e > SPLIT_TOL or rel_f > SPLIT_TOL:
        raise RuntimeError(
            f"{label}: coulomb+lj split residual {rel_e:.2e}/{rel_f:.2e}"
        )
    sum_terms = sum(energy_kj[t] for t in ("bond", "angle", "torsion", "nonbonded"))
    if abs(sum_terms - e_tot) > SPLIT_TOL * max(abs(e_tot), 1.0):
        raise RuntimeError(f"{label}: groups do not sum to the total")

    terms = ("bond", "angle", "torsion", "coulomb", "lj", "nonbonded", "total")
    return {
        "label": label,
        "perturbation_sigma_angstrom": sigma,
        "coords_angstrom": pos.tolist(),
        "min_nonexcluded_pair_distance_angstrom": _min_pair_distance(pos, exc),
        "coulomb_lj_split_residual": {"energy_rel": rel_e, "force_rel": rel_f},
        "energy_kj_mol": {t: energy_kj[t] for t in terms},
        "energy_hartree": {t: energy_kj[t] * KJ_TO_HARTREE for t in terms},
        "forces_kj_mol_nm": {"total": forces["total"].tolist()},
        "gradient_hartree_bohr": {
            t: (-forces[t] * FORCE_KJNM_TO_GRAD_HA_BOHR).tolist() for t in terms
        },
    }


def write_references():
    import openmm

    written = []
    ke_kj_nm, ke_ferric = _coulomb_constant()
    print(
        f"OpenMM e^2/(4 pi eps0) = {ke_kj_nm!r} kJ/mol nm = {ke_ferric!r} Ha Bohr "
        f"(rel. offset {ke_ferric - 1.0:+.3e} vs ferric's 1)"
    )
    for idx, (system_name, pdb_path, phase_shift, desc) in enumerate(SYSTEMS):
        if not pdb_path.exists():
            raise SystemExit(
                f"missing committed input {pdb_path}; run the build-inputs group"
            )
        t0 = time.perf_counter()
        pdb, system, pos0 = _build_system(pdb_path, phase_shift)
        top = topology_from_system(system)
        if not phase_shift:
            # The production entry point (PDB -> System -> dict) must give the
            # SAME dict as the System evaluated here.
            prod = topology_from_openmm(pdb_path, forcefield=FORCEFIELD)
            if prod != top:
                raise RuntimeError(
                    f"{system_name}: topology_from_openmm != topology_from_system"
                )
        exc, scales = _exceptions(system, top)
        ctx_full = _context(system)
        ctx_split = {k: _context(s) for k, s in _split_systems(system).items()}

        rng = np.random.default_rng(PERTURB_SEED + idx)
        geoms = [_geometry("pdb", 0.0, pos0, ctx_full, ctx_split, exc)]
        for label, sigma in PERTURB_SIGMAS:
            pos = pos0 + rng.normal(0.0, sigma, size=pos0.shape)
            geoms.append(_geometry(label, sigma, pos, ctx_full, ctx_split, exc))

        n_atoms = system.getNumParticles()
        phases = [t[6] for t in top["torsions"]]
        payload = {
            "row": ROW_NAME,
            "system": system_name,
            "description": desc,
            "n_atoms": n_atoms,
            "net_charge": float(sum(top["charges"])),
            "units": "topology: e, Angstrom, kcal/mol, degrees (AMBER, harmonic k "
            "without 1/2); energy_kj_mol / forces_kj_mol_nm are OpenMM-native "
            "(F = -dE/dR); energy_hartree / gradient_hartree_bohr are dE/dR in "
            "ferric's units",
            "topology": {k: v for k, v in top.items() if k != "n_atoms"},
            "all_torsion_phases_0_or_180": all(
                min(abs(p % 180.0), 180.0 - abs(p % 180.0)) < 1e-9 for p in phases
            ),
            "exceptions": exc,
            "exception_scales": scales,
            "geometries": geoms,
            "provenance": {
                "code": "OpenMM",
                "version": openmm.__version__,
                "platform": "Reference (double precision)",
                "forcefield": list(FORCEFIELD),
                "create_system": "nonbondedMethod=NoCutoff, constraints=None, "
                "rigidWater=False, removeCMMotion=False",
                "torsion_phase_shift_deg": phase_shift,
                "extraction": "tools/active_site/mm_topology.py::topology_from_system",
                "coulomb_lj_split": "XmlSerializer clones of the NonbondedForce: "
                "coulomb = particle and exception epsilons zeroed; lj = charges and "
                "chargeProds zeroed; each in its own System",
                "force_groups": GROUPS,
                "input_pdb": str(pdb_path.relative_to(common.ROOT)),
                "input_pdb_sha256": common.sha256_file(pdb_path),
                "perturbation": {
                    "seed": PERTURB_SEED + idx,
                    "rng": "numpy.random.default_rng(seed).normal, applied in order "
                    + ", ".join(f"{lab} sigma={s} A" for lab, s in PERTURB_SIGMAS),
                },
                "coulomb_constant": {
                    "openmm_kj_mol_nm": ke_kj_nm,
                    "openmm_in_hartree_bohr": ke_ferric,
                    "note": "ferric's Coulomb constant is exactly 1 Ha*Bohr; OpenMM's, "
                    "converted with ferric's constants, is openmm_in_hartree_bohr. "
                    "Reference Coulomb energies/forces exceed ferric's by this ratio.",
                },
                "conversions": {
                    "kj_per_kcal": KJ_PER_KCAL,
                    "kcal_per_hartree": KCAL_PER_HARTREE,
                    "bohr_in_angstrom": common.BOHR_IN_ANGSTROM,
                    "kj_mol_to_hartree": KJ_TO_HARTREE,
                    "force_kj_mol_nm_to_gradient_hartree_bohr": -FORCE_KJNM_TO_GRAD_HA_BOHR,
                },
                "generator": GEN,
                "git_head": common.git_head(),
                "generated_utc": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(timespec="seconds"),
            },
        }
        path = common.REF_DIR / ROW / f"{system_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        written.append(path)
        dt = time.perf_counter() - t0
        g0 = geoms[0]["energy_kj_mol"]
        print(
            f"{path.name}: n={n_atoms} q={payload['net_charge']:+.3f} "
            f"bonds={len(top['bonds'])} angles={len(top['angles'])} "
            f"torsions={len(top['torsions'])} excl={len(exc['excluded'])} "
            f"14={len(exc['scaled14'])} amb={len(exc['ambiguous'])} "
            f"scales lj={scales['scale_lj_14']:.15f} coul={scales['scale_coul_14']:.15f} "
            f"E_pdb(kJ) " + " ".join(f"{k}={v:.6f}" for k, v in g0.items()) + f" "
            f"min_d={[round(g['min_nonexcluded_pair_distance_angstrom'], 3) for g in geoms]} "
            f"split={max(g['coulomb_lj_split_residual']['energy_rel'] for g in geoms):.1e} "
            f"({dt:.2f}s)"
        )
    return written


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    t0 = time.perf_counter()
    if args == ["build-inputs"]:
        build_inputs()
        print(f"GEN_MM_INPUTS_DONE ({time.perf_counter() - t0:.1f}s)")
        return 0
    if args:
        raise SystemExit("usage: gen_mm.py [build-inputs]")
    written = write_references()
    print(f"GEN_MM_DONE written={len(written)} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
