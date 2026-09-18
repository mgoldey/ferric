# Golden paths: ferric in applications

End-to-end workflows for real questions, written so an agent can execute them
without reading the rest of the repo. Each one states **what it answers**,
**what it cannot**, and **how to tell it went wrong**.

Read [For agents](agents.md) first if you are automated — it has the failure
modes that cost the most time. Individual steps live in [Recipes](recipes.md).

---

## Golden path A — Reaction energetics of a cation cascade

**Answers:** which intermediate is the bottleneck, and which branch point
controls product selectivity.

**Worked example:** terpene synthase carbocation cascades (geranyl → linalyl →
α-terpinyl → product). The same shape applies to any closed-shell cation
mechanism.

### Step 0 — gate before you spend hours

```bash
pip install ferric && ferric examples/water-rhf.toml
```

Do not skip this, and **do not build from source first** -- the wheel is
seconds and a source build is ~45 minutes. If the install is broken, every
downstream failure will look like a chemistry problem.

Run the study itself under `scripts/ferric-limited --max=8G --high=7G --`:
these are multi-hour jobs and an OOM kills them with a truncated log and no
error message.

### Step 1 — build the species, and check parity

Generate starting geometries however you like (RDKit ETKDG + MMFF is fine —
**force fields are for geometry seeds only, never for a reported energy**).

Then, before any QM, verify electron parity for every species:

```python
n_elec = sum(ATOMIC_NUMBER[sym] for sym in symbols) - charge
assert (n_elec % 2 == 0) == (multiplicity % 2 == 1), (
    f"{name}: {n_elec} electrons cannot have multiplicity {multiplicity}"
)
```

A cation is `charge=1, multiplicity=1`. Getting this wrong is the single most
common first failure; ferric catches it with a clear message, but checking in
your own loop fails faster and names the species.

**The parity gate is NOT a geometry check, and you need both.** A structure can
have a perfectly correct formula and still be physically impossible. MEASURED:
a hand-built C10H17+ passed parity (76 electrons, singlet) while carrying a
0.902 A C-H contact -- shorter than a real bond -- and a hydrogen sitting
1.030 A from a *second* carbon. The SCF stalled at 37 iterations. That is the
correct behaviour for overlapping nuclei, and it cost a run to discover.

```python
# formula check and geometry check fail on DIFFERENT mistakes -- keep both
assert (n_elec % 2 == 0) == (multiplicity % 2 == 1)   # bad stoichiometry
assert min_interatomic_distance(xyz) > 0.9            # bad geometry
```

### Step 1b -- relax with xtb before ANY DFT

This is the middle tier of the funnel and skipping it is expensive. GFN2-xTB
fixes a bad structure in seconds:

```bash
LD_LIBRARY_PATH=~/.local/lib/x86_64-linux-gnu \
  xtb mol.xyz --opt --chrg 1 --uhf 0 --gfn 2
# -> xtbopt.xyz
```

`LD_LIBRARY_PATH` is not optional on a local install: bare `xtb` fails with
`libxtb.so.6: cannot open shared object file`, which reads like a broken
install and is not one.

MEASURED on the same molecule, same method, same basis:

    hand-built geometry : closest contact 0.902 A -> SCF Stalled @37, E = -389.5110690
    xtb-relaxed (54 its): closest contact 1.084 A -> SCF converged,   E = -390.3794234

**0.868 Ha = 545 kcal/mol** lower, and the difference between a result and a
failure. Seconds of xtb bought that.

### Step 2 — optimize each intermediate

```toml
[molecule]
xyz = "alpha_terpinyl.xyz"
charge = 1
multiplicity = 1

[basis]
name = "def2-svp"

[method]
kind = "ksdft"
task = "optimize"

[dft]
functional = "B3LYP"

[scf]
energy_conv = 1e-10
density_conv = 1e-9      # set BOTH; see below

[optimize]
max_steps = 60
```

**Open-shell cations need iterations, and a reachable `energy_conv`.**
MEASURED, the same converged system, changing only these two knobs:

    max_iter 100, energy_conv 1e-8  ->  MaxIter @100,  29 min, 3.30 GB, no result
    max_iter 400, energy_conv 1e-7  ->  CONVERGED @173, 4.9 min, 1.81 GB

Two things to take from that. **173 iterations is normal** for a delocalised
carbocation -- the default cap is not sized for this. And **`energy_conv: 1e-8`
was unreachable**: under density fitting the energy change floors with `naux`,
so a tighter dE than the RI noise floor can never be met and the run burns its
whole iteration budget chasing it. `density_conv` is the criterion that does
the real work in this codebase; leave the dE bound loose enough to be
satisfiable.

Note the failure MODE is diagnostic. `Stalled` means the SCF could not make
progress -- suspect the geometry. `MaxIter` means it was progressing and ran
out of room -- raise the cap. They are different problems and the exit reason
names which one you have.

**Set `density_conv`, not just `energy_conv`.** E_HF is variational and
tolerates a loose density; anything depending linearly on the MO coefficients
does not. This bit a real measurement in this repo: a pinned correlation energy
was reproducible only until something upstream moved, because the density had
never actually converged.

**Checkpoint every species as it finishes.** These are minutes-to-tens-of-
minutes each; a crash at species 6 of 8 should cost one species, not the run.

**Basis choice decides whether the study is possible at all.** MEASURED on the
same 27-atom cation:

    def2-SVP / B3LYP : >22 min, 6.04 GB, SIGKILLed before converging
    6-31G   / PBE    :  4.9 min, 1.81 GB, CONVERGED

A third of the memory and it finishes. At ~5 min per single point a geometry
optimisation lands back inside the "10-40 min" a study plan would assume --
but only at 6-31G, and only from a relaxed starting structure. Do not plan a
def2-SVP cascade on a workstation without measuring one species first.

**A 27-atom cation at def2-SVP needs ~6 GB.** Budget the study accordingly:
eight species at that size run concurrently is not a plan, it is an outage.
Always wrap the run in `scripts/ferric-limited --max=8G --high=7G --`; why the
in-process budget alone is not enough is explained in
[For agents](agents.md#failure-modes-that-cost-the-most-time).

**The release binary is not the whole answer either.** MEASURED here: the
alpha-terpinyl cation (C10H17+, 27 atoms, def2-SVP, B3LYP) ran **>34 minutes
without converging a SINGLE POINT** on a debug binary sharing a box with one
other job. An optimization is 30-60 single-point equivalents, so that is
**17-34 hours per species** -- against a plan that budgeted "10-40 min per
optimization". The gap is not tuning; it is the binary. Build with `--release`
or install the wheel, and serialize.

**Size it before you start.** MEASURED on this system: tert-butyl cation
(C4H9+, ~101 basis functions at def2-SVP) is a seconds-scale single-point.
A C10 cascade cation (C10H17+) is ~225 basis functions -- about **16x** the
single-point cost at N^3.5 scaling. That is still cheap for an energy, but an
optimization multiplies it by the step count, so a 30-60 step optimization on a
C10 species is the dominant cost of the whole study. Run one species end to end
and time it before queueing eight.

### Step 3 — confirm the ordering survives the functional

Single-points on the optimized geometries with **PBE** and **wB97X-V**
(both available; `wB97X-V` resolves to `HYB_GGA_XC_WB97X_V`).

If the three functionals disagree on the *ordering* of intermediates, **report
the disagreement**. Do not average them and do not pick the one that matches
your hypothesis. A cascade whose ordering is functional-dependent is a finding
about the system, not a number to be cleaned up.

### Step 4 — properties at the key intermediate

Hirshfeld/Löwdin charges, ESP at nuclei, and static polarizability for the
π-stabilized cation.

**Cost warning:** `pdep_polarizability_static` lives in `ferric-rpa` and is an
RPA-level calculation, not a cheap add-on to the DFT run. Budget it separately;
it can exceed the optimization it follows.

### What this cannot tell you

State these in any write-up; they are not hedging, they bound the claim.

* **Gas-phase cluster models.** No enzyme environment unless you add QM/MM
  (golden path B).
* **No transition states.** libint2 as built here has no second derivatives, so
  there are no analytic Hessians and no TS characterization. You get
  intermediate energies, not barriers. *kcat depends on barriers.*
* **No entropy, no ZPE.** Electronic energies only.
* **Relative energies only.** Absolute totals carry basis-set and functional
  error far larger than the differences you are interpreting.

---

## Golden path B — Which residue should I mutate?

**Answers:** which active-site residues most influence a reactive center, so
you have a ranked short list.

**Does not answer:** what mutation to make. ferric ranks hypotheses. It does
not design, and it does not predict ΔΔG.

### Step 1 — classical pre-screen (cheap, all residues)

```python
from tools.active_site.pqr_parser import parse_pqr
from tools.active_site.pocket_charges import derive_pocket_charges
from tools.active_site.pocket_field import pocket_field_at_atoms

pocket = derive_pocket_charges(...)
field  = pocket_field_at_atoms(pocket, reactive_center_xyz)  # (N,4) a.u.
```

Rank residues by their contribution to the field at the reactive center.
`pqr_parser` carries `res_name` and `res_seq`, so the ranking is
residue-resolved and directly reportable.

### Step 2 — QM/MM the top few (expensive, short list only)

`ferric_scf::qmmm` — validated against `pyscf.qmmm.mm_charge` to <1e-8.
This step is what makes the answer physics rather than electrostatics.

The pre-screen exists to make this step affordable. Running QM/MM on every
residue is the thing the funnel pattern is designed to avoid.

### Limits

Classical point charges: no protein polarization response, no sterics, no
conformational change on mutation, no ΔΔG. A flagged residue is a **candidate
for QM/MM**, not a designed mutation.

---

## Golden path C — Screen many ligands down to DFT

**Answers:** of N candidates, which few deserve expensive QM.

Use `tools/pipeline/funnel.py`. Do not hand-roll this loop.

```python
from tools.pipeline import Stage, run_funnel

report = run_funnel(
    candidates=isomers,
    stages=[
        Stage(name="dock", tier=0, keep=50, fn=dock_fn),
        Stage(name="xtb",  tier=1, keep=10, fn=xtb_fn),
        Stage(name="dft",  tier=2, keep=3,  fn=ferric_dft_fn),
    ],
    context={...},
)
```

Why this over a loop you write yourself:

* **Failed candidates are dropped and counted — never ranked.** A hand-rolled
  screen that sorts ascending on a sentinel value silently promotes its
  failures to the top. This is the bug the funnel exists to prevent.
* **Per-tier wall times**, because the tier that actually dominates is
  routinely not the one the cost table predicts.
* **Early stop** on an empty population.
* **Ascending rank** at every tier, since each reports an energy-like score.

---

## Reading the output

| Signal | Meaning |
|---|---|
| `converged = true` | Trust the energy. Absence of this line is not success. |
| `converged = false` | An energy is still printed. It is NOT a result. |
| `exit Some(Stalled)` | The SCF could not progress. Suspect the GEOMETRY -- check the minimum interatomic distance and relax with xtb. |
| `exit Some(MaxIter)` | It WAS progressing and ran out of iterations. Raise `max_iter` (173 is normal for a delocalised cation) and check `energy_conv` is reachable under DF. |
| charge/multiplicity error | Your `.xyz` atom count is wrong. Read the arithmetic it prints. |
| Disagreement with another code ~1e-5 Ha | Check the grid: ferric `(75,110)` vs PySCF ~`(75,302)`. Scales with **atom count**. |
| A correlation energy that moves when nothing physical changed | `density_conv` was never set. |

## Before publishing any number

Check `wiki/VALIDATION.md` for the capability's maturity grade. A method being
CLI-wired does not mean its numbers are production-grade — the matrix
distinguishes proven, smoke-tested, and stub, and it is the authority.
