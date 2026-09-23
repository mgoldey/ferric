# QM/MM

Run a quantum region inside a classical environment: a ligand in a protein
pocket, a reacting site in an enzyme, a solute in explicit solvent. The QM
atoms get a real wavefunction; everything else becomes point charges that
polarize it.

**What you get:** QM region selection (by index, by radius, or by whole
residue), link atoms across covalent cuts with four boundary-charge schemes,
Gaussian-smeared charges, Thole polarizable embedding, an AMBER-form MM crate,
analytic QM and MM forces, a full-structure gradient across the cut, geometry
optimization, and a TIP3P solvation droplet. Structures come from PDB, mmCIF,
PQR, SDF, mol2, GROMACS `.gro`, XYZ or SMILES.

**What is missing:** no AMBER `prmtop` reader (go through OpenMM), and no
periodic boundary conditions — the solvation droplet is finite, with a vacuum
boundary.

---

## Before you start: relax the geometry with xtb

**The single highest-value habit in this whole workflow.** DFT on an unrelaxed
structure wastes hours and often simply fails.

Measured on this project, a 27-atom carbocation built by hand:

```text
hand-built    C7-H24 = 0.902 Å      SCF: Stalled@37, never converged
xtb GFN2      C7-H24 = 1.084 Å      SCF: converged
                                    545 kcal/mol lower
```

A 0.9 Å C–H bond is not a slightly-off geometry; it is a structure whose SCF has
no reason to converge. xtb finds that in **seconds** where DFT spends hours
failing.

```bash
xtb input.xyz --opt --gfn 2 --chrg 1        # seconds
ferric single_point.toml                     # then the expensive step
```

Use it for the initial geometry of anything you did not get from a crystal
structure or a previous optimization — hand-built ligands, docked poses,
edited residues, anything from a SMILES string. See
[Applications](applications.md) for the full docking → xtb → DFT funnel.

> **Build note:** xtb built with `-O3` miscompiles gradients on this platform.
> Build with `-Doptimization=2`.

---

## The smallest working example

```python
import ferric

# The FULL structure: every atom, QM and MM alike.
symbols = ["O", "H", "H",   "O", "H", "H"]     # two waters
coords  = [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0],
           [0.0, 0.0, 3.0], [0.76, 0.59, 3.0], [-0.76, 0.59, 3.0]]
charges = [-0.834, 0.417, 0.417,  -0.834, 0.417, 0.417]   # MM partial charges (e)

# QM = the first water; the second becomes point charges.
sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 1, 2])

res = ferric.run_qmmm(sys, "cc-pvdz", method="rhf")
print(res.energy)
```

`charges` are ignored for atoms that end up in the QM region, so you can pass a
full force-field charge set and let the selection decide.

## Selecting the QM region

Three ways, and the choice matters more than it looks:

```python
# 1. Explicit indices — full control.
ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 1, 2])

# 2. Everything within a radius of some seed atoms.
ferric.QmmmSystem(symbols, coords, charges,
                  qm_seeds=[0], qm_radius_angstrom=5.0)

# 3. Whole residues — a residue joins if ANY of its atoms is in range.
#    Prefer this for proteins: a radius cut alone will slice through a
#    residue and leave you with a chemically meaningless fragment.
ferric.QmmmSystem(symbols, coords, charges,
                  qm_seeds=[0], qm_radius_angstrom=5.0,
                  residue_ids=residue_ids)
```

`residue_ids` together with `qm_indices` is an **error**, not a silent
no-op — an explicit index list and a residue expansion are contradictory
instructions.

## Cutting a covalent bond

If the QM/MM boundary crosses a bond, you need a link atom and a scheme for
what to do with the host's charge:

```python
sys = (ferric.QmmmSystem(symbols, coords, charges, qm_indices=qm)
       .with_link_atoms(bonds)                        # bonds crossing the cut
       .with_boundary_charges(bonds, "rcd"))          # keep | delete-host | rc | rcd
```

| scheme | what it does |
|---|---|
| `keep` | leave the host charge in place |
| `delete-host` | Z1 — delete it |
| `rc` | Lin–Truhlar redistributed charge |
| `rcd` | redistributed charge **and dipole** |

The parser is strict: an unknown string is an error, never a silent default.

## Energies, forces, and optimization

```python
res = ferric.run_qmmm(sys, "cc-pvdz", method="uhf")
# KS-DFT: method="rks" or "uks" AND xc, e.g. method="rks", xc="PBE".
# xc without rks/uks (or rks/uks without xc) raises ValueError.
res.energy
res.qm_gradient()      # dE/dR on the QM atoms
res.mm_forces()        # forces on the MM sites
res.full_gradient()    # across the cut, link rows folded back onto the frontier

opt = ferric.run_optimize_qmmm(sys, "cc-pvdz", method="rhf", move_mm="none")
```

`move_mm` chooses which MM atoms relax: `"none"` (the default, where only QM
atoms move), `("within", radius_angstrom)`, `("residues", [ids])` (the system
must have been built with `residue_ids=`), or `"all"`. Any value other than
`"none"` requires `mm_topology=`, because moving MM atoms without a force field
to hold their own geometry together is meaningless.

## Polarizable embedding

Thole polarizable sites, if fixed charges are not enough:

```python
sys = ferric.QmmmSystem(symbols, coords, charges,
                        qm_indices=qm,
                        polarizabilities_angstrom3=alphas)
res = ferric.run_qmmm(sys, "cc-pvdz", method="rhf", thole_a=2.1304)
res.e_pol                  # the polarization energy
res.induced_dipoles()
```

## What is validated, and how far

Checked against `pyscf.qmmm.mm_charge`: **energy shift agrees to <1e-8 Ha, MM
forces to 3e-10**. An empty or all-zero MM region is **bit-identical** to a gas
phase calculation — the trivial limit is a genuine no-op, not an approximation
that happens to be small.

Not validated: periodic boundary conditions (absent), and the solvation
droplet, which is a hard-sphere packing at roughly bulk density rather than an
equilibrated box.

## Reading a structure

Every format lands in the same place, so a PDB and an XYZ of one molecule give
a bit-identical `Molecule`:

```python
from tools.structure import read, from_smiles

mol = read("ligand.pdb")        # or .cif .pqr .sdf .mol2 .gro .xyz
mol = from_smiles("CCO")        # ETKDG geometry -- tier-2 grade, NOT optimized
```

A PQR carries MM charges as well as coordinates, which is why the CLI's
`[qmmm]` section reads one. Note that a docked pose from Vina is **united-atom**
— nonpolar hydrogens are merged into their carbons — so it is not a QM geometry
until those are restored; `tier1_dock` does that for you.

## From the CLI

A `[qmmm]` section runs QM/MM from a TOML file, with no Python needed. The QM
region becomes the molecule that is solved, and the MM region becomes the
external potential it is solved in. A complete, runnable input is
`examples/water-qmmm.toml` (repository, not the wheel):

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"   # required by the parser, NOT read with [qmmm]
charge = 0                             # apply to the QM region
multiplicity = 1

[basis]
name = "sto-3g"

[method]
kind = "rhf"
task = "energy"

[qmmm]
pqr = "testdata/molecules/water_na.pqr"
qm_indices = [0, 1, 2]            # zero-based; or: qm_seeds = [0], qm_radius_angstrom = 1.5
# link_bonds = [[0, 3]]           # required when the cut crosses a covalent bond
# boundary_scheme = "delete-host" # default; also "keep", "rc", "rcd"
```

MEASURED on that file (water QM, one Na⁺ 4 Å away as MM, STO-3G): vacuum
−74.9629466809, embedded **−74.9653197421** Ha (−1.489 kcal/mol). That matches
`ferric.run_rhf(point_charges=...)` to all ten printed digits.

**The geometry and the MM charges both come from the PQR.** An xyz has no
partial charges, and an MM region without charges is just a set of ignored
coordinates. `[molecule] xyz` must still be present, because the parser
requires it, but it is not read when `[qmmm]` is present. That matters when
you compute a vacuum reference: delete `[qmmm]` and ferric falls back to the
xyz. If the xyz holds a different geometry, you are comparing two different
molecules. In the example above, the xyz is an optimized water that gives
−74.9631468000, which is 0.13 kcal/mol of error in the difference. Compare at
one geometry.

**`boundary_scheme` defaults to `"delete-host"`, not `"keep"`.** Keeping the
host charge across a covalent cut puts a bare point charge inside the link
atom's bond length (MEASURED 0.443 Å on an ethane C–C cut). A geometry
optimization in that field diverges instead of failing with an error. `"keep"`
is still available and is the right choice when the cut isn't covalent. An
unknown scheme name is an error.

**`[qmmm]` and `[external_potential]` together are refused.** The MM region
*is* an external potential, so combining them would silently count a
contribution twice.

**Scope of the CLI section.** It reads only a PQR, because it needs charges
and geometry together. Use Python (`tools.structure`, below) for the other
formats. `tools.active_site.solvate` can write a solvated system straight to a
PQR for this section. The CLI section does electrostatic embedding only.
Polarizable sites, smeared charges, the MM force field and MM relaxation are
Python-only.

## Solvating a solute

```python
from tools.active_site.solvate import solvate, write_pqr

drop = solvate(symbols, coords_angstrom, radius_angstrom=12.0)
write_pqr("solvated.pqr", symbols, coords_angstrom, charges, drop)
```

The solute is written **first**, so its indices are `0 .. n-1` and can go
straight into `[qmmm] qm_indices`. Waters are TIP3P at roughly bulk density;
this is a starting structure, not an equilibrated one, and a droplet has a
vacuum boundary. Repeat over several `seed=` values before trusting a
difference — `dE_statistics` does that and refuses fewer than two seeds.

## Known limits, stated plainly

- **No AMBER `prmtop` reader.** Go through OpenMM
  (`active_site.mm_topology.topology_from_openmm`), or build the arrays.
- **No periodic boundary conditions.** `solvate()` gives a finite droplet with
  a vacuum boundary — adequate for a local environment, not for bulk.
- **The pocket field is fixed unless you ask otherwise.** `move_mm="none"` is
  the default in `run_optimize_qmmm`; the MM sites do not relax with the QM
  region until you widen it.
- **No QM/MM dispersion.** D3/D4/XDM/VV10 are all QM-atom-pairwise, so
  dispersion between the QM region and the MM charges is absent. The MM crate
  supplies Lennard-Jones terms for the MM-MM part only.
- The MM crate is AMBER-form (harmonic bonds and angles, periodic torsions,
  Lennard-Jones, Coulomb), validated against OpenMM.
