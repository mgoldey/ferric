# ferric recipes

Copy-paste workflows that end in a number. Each gives its **expected output**
and **rough runtime**, so you can tell "still running" from "silently wrong".
That confusion is the most common failure when you drive a QC code you haven't
used before.

Recipes 0, 1, 3 and 4 were executed as written for this page, on 2026-09-23.
Recipe 2's output is quoted from an earlier run. Recipe 5 is a **sketch**,
marked as such where it appears. Numbers labelled MEASURED came out of the
program.

**Prerequisite:** a working `ferric`. The fast path is the prebuilt wheel
(about a minute). A source build takes ~30 minutes, most of it libint2:

```bash
pip install ferric
```

See [Installation](installation.md). Build from source only if you are changing
ferric itself or need MPI.

**Which recipes need a git clone.** The wheel contains the compiled library and
the `ferric` command, nothing else. `examples/`, `testdata/`, `scripts/` and the
`tools/` Python package live in the repository
([what the wheel does not contain](installation.md#what-the-wheel-does-not-contain)).
Run those recipes from the repository root:

| Recipe | Needs a clone? | Extra dependencies |
|---|---|---|
| 0. SMILES → energy | yes (`tools.structure`) | RDKit |
| 1. Single point | only for `examples/water-rhf.toml`; your own `.xyz` + TOML works anywhere | — |
| 2. Ions and radicals | no | — |
| 3. Optimize | only for `examples/h2-lda-opt.toml` | — |
| 4. Ligand funnel | yes (`tools.pipeline`) | RDKit, `xtb` on `PATH` |
| 5. Residue ranking | yes (`tools.active_site`) | pdb2pqr |

**On a shared machine, run anything real under a memory cap.** ferric's
memory budget charges only the large, size-dependent tensors; basis-sized
matrices, integral engines, BLAS scratch and allocator overhead are not
charged, so a job's resident memory can exceed the budget. Only a cgroup puts
a hard ceiling on the process, and without one an overshoot can trigger the
system-wide OOM killer, which may kill unrelated processes. The details are in
[Sharp bits](sharp-bits.md#memory-budget_gb-does-not-cap-the-whole-process)
and [For agents](agents.md#memory-the-budget-predicts-the-cgroup-enforces).
`scripts/ferric-limited` (repository, Linux with systemd) wraps a command in a
`systemd-run --user` scope:

```bash
scripts/ferric-limited --max=8G --high=7G -- ferric input.toml
```

---

## 0. From a SMILES to an energy

```python
import ferric
from tools.structure import from_smiles        # needs a clone + RDKit

mol = from_smiles("CCO")                        # ethanol
res = ferric.run_dft(mol, ferric.BasisSet.bundled("def2-svp"),
                     functional="PBE", dispersion="d3bj")
assert res.converged                            # always check this first
print(res.total_energy, res.e_scf, res.e_dispersion)
```

MEASURED (2026-09-23, 4 threads on a loaded machine, about 4 s):

```text
-154.72507297  -154.72071374  -0.00435923
```

- `from_smiles` returns an RDKit ETKDG embedding plus an MMFF cleanup, seeded so
  it is reproducible. That is a **starting structure, not a minimum**. For
  anything you will report, relax it (xtb, then a ferric optimization) first.
  See recipe 3 and [Golden paths](applications.md) Step 1b.
- `total_energy` is `e_scf + e_dispersion`. `dispersion=None` (the default)
  leaves `e_dispersion` as `None`, meaning "not evaluated", never `0.0`.
- `run_dft` density-fits the Coulomb term by default. That matters when you
  compare against a code using exact Coulomb (recipe 6).
- `from_smiles` reads the charge from the SMILES. Spin is never inferred:
  pass `multiplicity=2` for a radical.

For a file instead of a SMILES, `tools.structure.read("x.sdf")` returns the same
kind of `Molecule` (PDB, mmCIF, PQR, SDF, mol2, `.gro` and XYZ are supported).
A PDB must already carry explicit hydrogens.

---

## 1. Single-point energy from a TOML file

The first success that confirms your install works. From the repository root:

```bash
ferric examples/water-rhf.toml
```

MEASURED (about 2 s including start-up):

```text
RHF/sto-3g on testdata/molecules/water.xyz
  nbasis     = 7
  iterations = 9
  converged  = true
  energy     = -74.9631468000 Hartree
```

From a source checkout, the same run is
`cargo run --release --bin ferric -- examples/water-rhf.toml`.

For your own molecule you need an `.xyz` (Å) and a TOML file:

```toml
[molecule]
xyz = "mymol.xyz"        # relative to the directory you run ferric from

[basis]
name = "def2-svp"

[method]
kind = "rhf"
```

For every `method.kind`, see [Capabilities](../reference/capabilities.md). For
every TOML key, see the [input reference](../reference/input.md).

---

## 2. Charged and open-shell species

**Read this before you run any ion, radical or metal center.** `charge` and
`multiplicity` are `[molecule]` keys. Most files in `examples/` leave them at
their defaults (0 and 1). The open-shell exceptions are `examples/h_uhf.toml`
and `examples/oh-ugw.toml`.

```toml
[molecule]
xyz = "tBu.xyz"
charge = 1          # cation
multiplicity = 1    # closed-shell singlet

[basis]
name = "def2-svp"

[method]
kind = "ksdft"

[dft]
functional = "B3LYP"
```

**Relax the geometry with xtb first.** A correct formula doesn't guarantee a
physical structure, and the parity check below can't tell the difference.
MEASURED: a hand-built C10H17+ with a 0.902 Å C–H contact stalled the SCF.
After `xtb mol.xyz --opt --chrg 1 --uhf 0 --gfn 2` it converged, 545 kcal/mol
lower. See [Golden paths](applications.md) Step 1b.

MEASURED on the tert-butyl cation C4H9+ (13 atoms), B3LYP/def2-SVP:

```
converged  = true
energy     = -157.4301362681 Hartree
```

Expect seconds to about a minute at this size.

### The error you will hit first

If the electron count and the multiplicity disagree, you get this error. It's
worth learning to recognise:

```
error: inconsistent charge/multiplicity: 35 electrons with multiplicity 1
implies n_alpha = (35 + 1 - 1) / 2 = 35/2, which is not a non-negative
integer... An odd electron count needs an even multiplicity (2, 4, ...)
and vice versa
```

It means **your geometry, your charge or your multiplicity is wrong**. ferric
can handle ions and open shells. The message prints the arithmetic and the
rule, so check the atom count in your `.xyz` against the first line of the
file, then the `charge` and `multiplicity` you set.

A radical (open-shell doublet) uses the same keys:

```toml
[molecule]
xyz = "radical.xyz"
charge = 0
multiplicity = 2    # one unpaired electron -> UHF/UKS
```

From Python, multiplicity belongs to the molecule, not to the `run_*` call:
`ferric.Molecule.from_xyz("x.xyz", charge=0, multiplicity=2)`.

---

## 3. Optimize a geometry

```bash
ferric examples/h2-lda-opt.toml
```

MEASURED (about 2 s): `converged = true`, `steps = 2`,
`final E = -1.1212649781 Hartree`.

The pattern is `task = "optimize"` next to any `kind` that has analytic
gradients:

```toml
[method]
kind = "ksdft"
task = "optimize"

[dft]
functional = "B3LYP"

[optimize]
max_steps = 30
```

[Capabilities](../reference/capabilities.md) lists which methods have analytic
gradients. Harmonic frequencies use `task = "frequencies"` (finite differences
of the analytic gradient; see `examples/water-frequencies.toml`).

**Runtime depends on the system, so measure before you plan.** A whole
optimization at drug-like size has not been timed on this page. For scale,
two single-point measurements:

- 32 atoms, PBE/def2-SVP: 96 s (the figure recorded in the
  `tools.pipeline.tiers.tier4_dft` docstring).
- A 27-atom delocalised cation, PBE/6-31G: 4.9 min, because it needed 173 SCF
  iterations. At B3LYP/def2-SVP the same molecule was killed for memory after
  22 min ([Golden paths](applications.md), Step 2).

An optimization multiplies the single-point cost by the number of steps.
Measure one species before you queue many.

---

## 4. Ligand screening: force field → xtb → DFT

`tools.pipeline.run_funnel` runs a **tiered funnel**: cheap scoring on many
candidates, and expensive QM only on the few that survive. Use it when you have
N molecules and want DFT numbers on the good ones. Don't write your own loop.

This was run exactly as shown, from the repository root:

```python
from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline import Stage, run_funnel
from tools.pipeline.tiers import tier2_forcefield, tier3_gfn2, tier4_dft

# Three isomers of C3H6O2, so comparing absolute energies is meaningful.
smiles = ["CCC(=O)O", "COC(C)=O", "CCOC=O"]
candidates = [
    Isomer(smiles=s, kind="structural", transform=s, parent_smiles=smiles[0])
    for s in smiles
]
stages = [
    Stage(Tier.FORCE_FIELD,   tier2_forcefield, keep=3, name="ff"),
    Stage(Tier.SEMIEMPIRICAL, tier3_gfn2,       keep=2, name="xtb"),
    Stage(Tier.QUANTUM,       tier4_dft,        keep=1, name="dft"),
]
rep = run_funnel(candidates, stages, {"seed": 0xF00D, "basis": "sto-3g"})
print(rep.table())
for iso in rep.survivors:
    print(iso.canonical, rep.value("dft", iso.canonical))
```

MEASURED (2026-09-23, one process):

```text
tier  stage           in   out  failed     secs   s/cand  note
   2  ff               3     3       0      0.4     0.13  ff: kept 3 of 3 scored
   3  xtb              3     2       0      0.1     0.04  xtb: kept 2 of 3 scored
   4  dft              2     1       0      4.8     2.38  dft: kept 1 of 2 scored
      TOTAL                                 5.3
      dominant tier 4 (dft) = 90% of wall
COC(C)=O -264.5628932123858
```

**STO-3G is a smoke-test basis, so don't read chemistry into which isomer
survived.** The run shows the plumbing works. Change `"basis"` in the context
for real work.

What the funnel does that a hand-written loop usually doesn't:

* **Ranks ascending at every tier.** Lower is better, because every tier
  reports an energy or an energy-like score. For the same reason, **rank only
  candidates that share a molecular formula**. An absolute energy of a larger
  molecule is lower because it has more electrons, not because it is better. For
  substituent series, rank against the parent (`parent_smiles`). The
  [pipeline notes](../reference/pipeline-golden-path.md) §0b cover that.
* **Drops failures instead of ranking them.** A candidate that a tier failed on
  is counted as failed. It is never treated as having scored well. That's easy
  to get wrong by hand, and it silently corrupts a screen.
* **Times each tier.** The tier that actually costs the run is often not the one
  the cost table predicts.
* **Stops early** when the population is empty, instead of running an expensive
  tier on nothing.

`tier4_dft` adds D3(BJ) dispersion by default. Pass
`context["dispersion"] = None` for the bare SCF energy. Docking
(`tier1_dock`) needs the `ferric[docking]` extra, a receptor
(`context["receptor_pdbqt"]`) and a box centre (`context["box_center"]`). The
[pipeline notes](../reference/pipeline-golden-path.md) §0b show the substituent
version, with liability flags and parent-relative gating.

**Before you rank anything by the DFT tier, read the noise measurement.**
MEASURED on a real campaign: the best available ΔΔE noise over a pose ensemble
was 4.07 kcal/mol, against substituent effects of 1–2 kcal/mol. See the
[pharma coverage notes](../reference/pharma-use-case-coverage.md).

Related modules in `tools/active_site/`: `ligand_embedding`, `pose_relaxation`,
`binding_energy`, `prescreen`, `pocket_charges`, `pocket_field`.

---

## 5. Rank residues for mutation (electrostatic pre-screen)

ferric does **not** design mutations. It can rank which active-site residues
most influence a reactive center, which gives you a short list worth testing
instead of a guess.

> **Sketch, not executed for this page.** The two library calls are real.
> The per-residue split is ordinary Python written for this page, and
> `derive_pocket_charges` needs `pdb2pqr` and a pocket PDB.

```python
from collections import defaultdict
from tools.active_site.pocket_charges import PocketCharges, derive_pocket_charges
from tools.active_site.pocket_field import pocket_field_at_atoms

pocket = derive_pocket_charges("pocket.pdb")   # runs pdb2pqr; fills residue_ids
site_xyz = [(x, y, z)]                         # reactive-center atom(s), Angstrom

# pocket_field_at_atoms returns the TOTAL field. To rank residues, split the
# charges by residue and evaluate each group on its own.
by_res = defaultdict(list)
for q, rid in zip(pocket.charges, pocket.residue_ids):
    by_res[rid].append(q)
contrib = {
    rid: pocket_field_at_atoms(PocketCharges(qs, pocket.source_pdb, pocket.ff), site_xyz)
    for rid, qs in by_res.items()
}   # each value: (N_sites, 4) array of [phi, Ex, Ey, Ez], atomic units
```

The method:

1. Compute the field each residue produces at the reactive center.
2. Rank residues by their contribution. That's your candidate list.
3. **Check the top few with QM/MM** ([QM/MM](qmmm.md); the embedding matches
   `pyscf.qmmm.mm_charge` to <1e-8 Ha). This step is what turns the ranking
   into physics rather than electrostatic hand-waving.

**Limits. Include these in any write-up that uses this method.** It's a
classical point-charge pre-screen, so it has no polarization response of the
protein, no sterics, no conformational change on mutation, and no ΔΔG. It
ranks *hypotheses*. A residue it flags is a candidate for QM/MM, not a designed
mutation.

---

## 6. Comparing against another code

Three causes account for most spurious "ferric disagrees" reports:

**Density fitting.** ferric's KS-DFT (`kind = "ksdft"`, `run_dft`) fits the
Coulomb term by default, and the fitting error grows with system size. MEASURED
at PBE/STO-3G against exact Coulomb: water 0.28, benzene 1.16, and a 71-atom
drug molecule 9.5 kcal/mol. Compare like with like. `run_dft(...,
df_j_aux="exact")` turns fitting off, or you can turn it on in the other code.

**Grid.** ferric's KS-DFT default grid is `(75, 110)` (radial, angular), flat,
with no pruning. PySCF's default `grid.level=3` is roughly `(75, 302)`. The
difference is worth ~1e-5 Ha on water and grows with **atom count**, because
grid error scales with the number of atoms, not the basis size. A small basis
on a big molecule can be worse than a big basis on a small one. Match grids
before you conclude anything.

**Convergence criteria.** Setting `energy_conv` alone can leave the density
loosely converged. Variational quantities (E_HF) don't mind. Anything that
depends linearly on the MO coefficients (correlation energies, properties) does.
`density_conv` is the criterion that actually converges the density, so
tighten it when you care about those quantities. Keep `energy_conv` loose: with
density fitting, a very tight `energy_conv` may be unreachable
([Golden paths](applications.md), Step 2).

```toml
[scf]
density_conv = 1e-9
```
