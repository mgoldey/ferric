# ferric recipes

Copy-paste workflows that end in a number. Each gives its **expected output**
and **rough runtime**, so you can tell "still running" from "silently wrong".
That confusion is the most common failure when you drive a QC code you haven't
used before.

Recipes 1 and 3 were executed as written for this page, on 2026-09-23.
Recipe 2's output is quoted from an earlier run. Numbers labelled MEASURED came
out of the program.

**Prerequisite:** a working `ferric`. The fast path is the prebuilt wheel
(about a minute). A source build takes ~30 minutes, most of it libint2:

```bash
pip install ferric
```

See [Installation](installation.md). Build from source only if you are changing
ferric itself or need MPI.

**Which recipes need a git clone.** The wheel contains the compiled library and
the `ferric` command, nothing else. `examples/`, `testdata/` and `scripts/`
live in the repository
([what the wheel does not contain](installation.md#what-the-wheel-does-not-contain)).
Run those recipes from the repository root:

| Recipe | Needs a clone? | Extra dependencies |
|---|---|---|
| 1. Single point | only for `examples/water-rhf.toml`; your own `.xyz` + TOML works anywhere | — |
| 2. Ions and radicals | no | — |
| 3. Optimize | only for `examples/h2-lda-opt.toml` | — |

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

For every `method.kind`, see [Capabilities and validation](../reference/validation.md). For
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

[Capabilities and validation](../reference/validation.md) lists which methods have analytic
gradients. Harmonic frequencies use `task = "frequencies"` (finite differences
of the analytic gradient; see `examples/water-frequencies.toml`).

**Runtime depends on the system, so measure before you plan.** A whole
optimization at drug-like size has not been timed on this page. For scale,
two single-point measurements:

- 32 atoms, PBE/def2-SVP: 96 s.
- A 27-atom delocalised cation, PBE/6-31G: 4.9 min, because it needed 173 SCF
  iterations. At B3LYP/def2-SVP the same molecule was killed for memory after
  22 min ([Golden paths](applications.md), Step 2).

An optimization multiplies the single-point cost by the number of steps.
Measure one species before you queue many.

---

## 4. Comparing against another code

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
