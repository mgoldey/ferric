# QM/MM

ferric has a validated QM/MM layer that is easy to miss: until this page it was
mentioned **nowhere** in the README or this site, despite being ~2,500 lines
with a Python API and nine test files. An external reviewer reading the repo
concluded ferric had "no QM/MM environment builder". The physics was there; the
documentation was not.

**What exists:** QM region selection (by index, by radius, by whole residue),
link atoms, three boundary-charge schemes, Gaussian-smeared charges, Thole
polarizable embedding, an AMBER-form MM crate, analytic QM and MM forces, a
full-structure gradient across the cut, and geometry optimization.

**What does not:** there is **no PDB / prmtop / GRO reader, no solvation, and
no PBC**. You supply coordinate arrays. That is the real gap — the physics is
complete, the input plumbing is not.

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
res = ferric.run_qmmm(sys, "cc-pvdz", method="uhf")   # or xc="PBE" for KS-DFT
res.energy
res.qm_gradient()      # dE/dR on the QM atoms
res.mm_forces()        # forces on the MM sites
res.full_gradient()    # across the cut, link rows folded back onto the frontier

opt = ferric.run_optimize_qmmm(sys, "cc-pvdz", method="rhf", move_mm="none")
```

`move_mm` chooses what is allowed to relax: `none`, a radius, whole residues, or
everything.

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

Not validated: periodic boundary conditions (absent), solvation (absent), and
anything requiring a structure-file reader (absent).

## Known limits, stated plainly

- **No PDB / prmtop / GRO parsing.** Build the arrays yourself, or via RDKit /
  MDAnalysis / ASE.
- **No PBC, no solvation box.**
- Not wired into the CLI TOML — QM/MM is Rust and Python API only.
- The MM crate is AMBER-form (harmonic bonds and angles, periodic torsions,
  Lennard-Jones, Coulomb) and was validated against OpenMM.
