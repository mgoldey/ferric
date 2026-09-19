# ferric recipes

Copy-paste workflows that end in a number. Each states its **expected output**
and **rough runtime**, so you can tell "still running" from "silently wrong" —
the most common failure mode when driving a QC code you haven't used before.

Every recipe here has been executed, not inferred. Where a number is quoted, it
came out of the binary.

**Prerequisite:** a working `ferric`. The fast path is the prebuilt wheel --
seconds, not the ~45 minutes a source build costs:

```bash
pip install ferric          # or ferric-mpi, which needs system OpenMPI 4.x
ferric examples/water-rhf.toml
```

Build from source only if you are changing ferric itself
([Installation](installation.md)).

**Run anything real under the memory guard.** ferric's budget predicts; only a
cgroup enforces. Measured: a 27-atom job overshot its 4.72 GiB budget to
6.04 GiB and was SIGKILLed, taking unrelated processes with it. Full
explanation in [For agents](agents.md#failure-modes-that-cost-the-most-time).

```bash
scripts/ferric-limited --max=8G --high=7G -- ferric input.toml
```

---

## 1. Single-point energy on a molecule you have

The 30-second first success. Confirms your build works before you trust it with
anything real.

```bash
cargo run --release -- examples/water-rhf.toml
```

Expected: `converged = true` and an `energy = ...` line. Seconds.

Your own molecule needs a `.xyz` (Å) and a TOML:

```toml
[molecule]
xyz = "mymol.xyz"

[basis]
name = "def2-svp"

[method]
kind = "rhf"        # see docs/METHODS.md for the full list
```

---

## 2. Charged and open-shell species

**Read this before running any ion, radical, or metal center.** `charge` and
`multiplicity` are `[molecule]` keys, and nothing in `examples/` demonstrates
them.

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

**Relax the geometry with xtb first.** A correct formula does not mean a
physical structure, and the parity check cannot see the difference. A
hand-built C10H17+ with a 0.902 A C-H contact stalled the SCF; after
`xtb mol.xyz --opt --chrg 1 --uhf 0 --gfn 2` it converged, 545 kcal/mol lower.
See [Golden paths](applications.md) Step 1b.

MEASURED, tert-butyl cation C4H9+ (13 atoms), B3LYP/def2-SVP:

```
converged  = true
energy     = -157.4301362681 Hartree
```

Seconds to ~a minute at this size.

### The error you will hit first

Get the atom count wrong and you get this, which is worth recognising on sight:

```
error: inconsistent charge/multiplicity: 35 electrons with multiplicity 1
implies n_alpha = (35 + 1 - 1) / 2 = 35/2, which is not a non-negative
integer... An odd electron count needs an even multiplicity (2, 4, ...)
and vice versa
```

That means **your geometry is wrong**, not that ferric cannot do ions. It
prints the arithmetic and the rule; check your `.xyz` atom count against the
first line of the file.

Open-shell doublet (a radical) is the same shape:

```toml
charge = 0
multiplicity = 2    # one unpaired electron -> UHF/UKS
```

---

## 3. Optimize a geometry

```bash
cargo run --release -- examples/h2-lda-opt.toml
```

The pattern is `task = "optimize"` alongside any supported `kind`:

```toml
[method]
kind = "ksdft"
task = "optimize"

[dft]
functional = "B3LYP"

[optimize]
max_steps = 30
```

Analytical gradients are used where available (RHF/UHF/ROHF, KS-DFT including
meta-GGA closed-shell). Runtime scales with the number of steps — budget
~10-40 min for a ~30-atom system at def2-SVP on 8 cores.

---

## 4. Ligand screening: dock → xtb → DFT

`tools/pipeline/funnel.py` runs a **tiered funnel**: cheap scoring on many
candidates, expensive QM on the few that survive. This is the right entry point
for "I have N ligands and want DFT numbers on the good ones" — do not rebuild
it.

```python
from tools.pipeline import Stage, run_funnel

report = run_funnel(
    candidates=isomers,        # list[Isomer]
    stages=[
        Stage(name="dock",  tier=0, keep=50,  fn=dock_fn),
        Stage(name="xtb",   tier=1, keep=10,  fn=xtb_fn),
        Stage(name="dft",   tier=2, keep=3,   fn=ferric_dft_fn),
    ],
    context={...},
)
```

What it gives you, which is why it is worth using over a hand-rolled loop:

* **Ascending rank at every tier** — lower is better, because every tier
  reports an energy or energy-like score.
* **Failures are dropped, never ranked.** A candidate a tier failed on is
  counted as failed, not treated as having scored well. That distinction is
  easy to get wrong by hand and silently poisons a screen.
* **Per-tier wall times**, because the tier that actually costs the run is
  routinely not the one the cost table predicts.
* **Early stop** on an empty population rather than running an expensive tier
  on nothing.

Related, in `tools/active_site/`: `ligand_embedding`, `pose_relaxation`,
`binding_energy`, `prescreen`, `pocket_charges`, `pocket_field`.

---

## 5. Rank residues for mutation (electrostatic pre-screen)

ferric does **not** design mutations. What it can do honestly is rank which
active-site residues most influence a reactive center, so you have a short list
worth testing rather than a guess.

```python
from tools.active_site.pqr_parser import parse_pqr
from tools.active_site.pocket_charges import derive_pocket_charges
from tools.active_site.pocket_field import pocket_field_at_atoms

pocket = derive_pocket_charges(...)              # per-residue point charges
field  = pocket_field_at_atoms(pocket, site_xyz) # (N, 4): [phi, Ex, Ey, Ez] a.u.
```

The method:

1. Compute the field each residue exerts at the reactive center.
2. Rank residues by contribution — that is your candidate list.
3. **Validate the top few with QM/MM** (`ferric_scf::qmmm`, validated against
   `pyscf.qmmm.mm_charge` to <1e-8). This step is what makes it physics rather
   than electrostatic hand-waving.

**Limits, which belong in any write-up that uses this.** It is a classical
point-charge pre-screen: no polarization response of the protein, no sterics,
no conformational change on mutation, and no ΔΔG. It ranks *hypotheses*. A
residue this flags is a candidate for QM/MM, not a designed mutation.

---

## 6. Comparing against another code

Two things cause most spurious "ferric disagrees" reports:

**Grid.** ferric's KS-DFT default is `(75, 110)` (radial, angular), flat, no
pruning. PySCF's default `grid.level=3` is roughly `(75, 302)`. That difference
is worth ~1e-5 Ha on water and grows with **atom count** — grid error scales
with the number of atoms, not the basis size, so a small molecule with more
atoms can be worse. Match grids before concluding anything.

**Convergence criteria.** Setting `energy_conv` alone can leave the density
loosely converged. Variational quantities (E_HF) are insensitive to that;
anything depending linearly on the MO coefficients (correlation energies,
properties) is not. Set `density_conv` too when you care about the latter.

```toml
[scf]
energy_conv = 1e-10
density_conv = 1e-9
```
