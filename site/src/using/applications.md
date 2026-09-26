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
pip install ferric
git clone https://github.com/mgoldey/ferric && cd ferric   # examples/, testdata/, tools/, scripts/
ferric examples/water-rhf.toml    # expect converged = true, energy = -74.9631468000
```

Don't skip this, and **don't build from source first**. The wheel installs in
about a minute, and a source build takes ~30 minutes. If the install is broken,
every later failure will look like a chemistry problem. The clone is needed
because the wheel doesn't ship `examples/`, `testdata/`, `scripts/` or `tools/`.

Run the study itself under `scripts/ferric-limited --max=8G --high=7G --`.
These can be multi-hour jobs, and an OOM kill leaves a truncated log with no
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
assert min_interatomic_distance(xyz) > 0.9            # gross overlap only (your own helper, Angstrom)
# The failure above PASSES that screen (closest contact 0.902 A): its H sat
# ~1 A from two carbons at once. Also reject any H bonded to two heavy atoms:
assert max_heavy_neighbours_of_hydrogen(xyz, cutoff=1.3) <= 1
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
name = "6-31g"           # def2-SVP did not fit on a workstation here; see below

[method]
kind = "ksdft"
task = "optimize"

[dft]
functional = "PBE"

[scf]
max_iter = 400           # 173 iterations was needed; the default is 100
energy_conv = 1e-7       # a loose, reachable bound; see below
# density_conv defaults to 1e-6 and is the criterion that does the real work.
# Tighten it if you need more than the energy; see below.

[optimize]
max_steps = 60
```

These settings follow from the measurements below. Don't raise the basis or
tighten `energy_conv` without measuring one species first.

**Delocalised cations need iterations, and a reachable `energy_conv`.**
MEASURED on the same 27-atom cation, changing only these two knobs:

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
does not. A correlation energy computed from an unconverged density can look
reproducible and still shift when anything upstream changes, because the
density was never converged.

**Checkpoint every species as it finishes.** A crash at species 6 of 8 should
cost one species, not the whole run.

**The method and basis decide whether the study is possible at all.**
MEASURED on the same 27-atom cation, single point (both the basis and the
functional differ between the two rows, so this does not isolate the cost of
the basis):

    def2-SVP / B3LYP : >22 min, 6.04 GB, SIGKILLed before converging
    6-31G   / PBE    :  4.9 min, 1.81 GB, CONVERGED

6-31G/PBE used a third of the memory and finished. Don't plan a def2-SVP
cascade on a workstation without measuring one species first.

**Treat the optimization time as an estimate.** An optimization takes tens of
gradient steps. ESTIMATED: at ~5 min per 6-31G single point, 30–60 steps would
take roughly 2.5–5 hours per species *if* each gradient step costs about as much
as a single point. Neither a gradient step nor a full optimization of this
system has been timed. Later steps start from the previous density and may
converge faster than a cold start. Time one optimization before budgeting a
cascade.

**Use a release build.** MEASURED: the same cation at B3LYP/def2-SVP ran for
more than 34 minutes on a *debug* binary, sharing the machine with one other
job, without converging a single point. The wheel and `cargo build --release`
are both release builds.

**A 27-atom cation at def2-SVP needs ~6 GB.** Budget the study with that in
mind. Running eight species of that size at once is an outage, not a plan.
Always wrap the run in `scripts/ferric-limited --max=8G --high=7G --`.
[For agents](agents.md#memory-the-budget-predicts-the-cgroup-enforces)
explains why the in-process budget alone is not enough.

**Size it before you start.** MEASURED: the tert-butyl cation (C4H9+, 101
basis functions at def2-SVP) is a single point that takes seconds. A C10
cascade cation (C10H17+) has 225 basis functions. ESTIMATED at N^3.5 scaling,
that is about **16x** the single-point cost. That's still cheap for an energy,
but an optimization multiplies it by the step count. So the optimizations of
the C10 species are the dominant cost of the whole study. Run one species end
to end and time it before you queue eight.

### Step 3 — confirm the ordering survives the functional

Run single points on the optimized geometries with at least two more
functionals, for example **B3LYP** and **wB97X-V** (both available;
`wB97X-V` resolves to libxc's `HYB_GGA_XC_WB97X_V`), at a basis that fitted in
Step 2.

If the functionals disagree on the *ordering* of intermediates, **report
the disagreement**. Do not average them and do not pick the one that matches
your hypothesis. A cascade whose ordering is functional-dependent is a finding
about the system, not a number to be cleaned up.

### Step 4 — properties at the key intermediate

Hirshfeld and Löwdin charges, the ESP at the nuclei, and the static
polarizability of the π-stabilized cation. From Python these are
`ferric.hirshfeld_charges`, `ferric.lowdin_charges` and `ferric.esp_at_atoms`.

**Cost warning:** the polarizability comes from `ferric-rpa` and is an
RPA-level calculation, not a cheap add-on to the DFT run. Budget it
separately. It can cost more than the optimization before it.

### Step 5 (optional) — barriers

Intermediates alone give you a thermodynamic profile. If you need barriers,
ferric has a Python-only transition-state toolchain. None of it is wired to
the CLI's `method.task`:

| Call | What it does | Scope |
|---|---|---|
| `ferric.run_saddle(mol, basis, xc=...)` | P-RFO search for a first-order saddle point | Closed shell only (multiplicity 1), HF or KS. It raises if the start has no negative Hessian mode, so start from a guessed TS, not a minimum. The Hessian is built twice by central differences and Bofill-updated in between: `2(6N+1) + (steps+1)` gradients. |
| `ferric.run_irc(mol, basis, mode=...)` | Follows the reaction path downhill in both directions, from the saddle to the two minima it connects | Closed shell only. Pass `SaddleResult.imaginary_mode` as `mode`. MEASURED ~71 gradients per direction on NH3 inversion. |
| `ferric.run_frequencies(mol, basis, reference=..., xc=...)` (CLI: `task = "frequencies"`) | Harmonic frequencies from finite differences of the analytic gradient (6N gradients). Negative entries are imaginary modes. `.normal_modes` gives the vectors. | RHF/UHF/ROHF and their KS variants. Check `.asymmetry` to judge whether the step size suited the system. The CLI refuses `[dft] dispersion` with this task. |

One imaginary frequency is necessary but not sufficient for a transition state:
a methyl rotor also gives one. Use `run_irc` to confirm that the saddle connects
the two intermediates you meant.

### What this cannot tell you

State these in any write-up. They aren't hedging. They define the scope of
the claim.

* **Gas-phase cluster models.** There's no enzyme environment unless you add
  QM/MM (golden path B, and [QM/MM](qmmm.md)).
* **Analytic Hessians for closed-shell RHF only.** RHF frequencies use the
  analytic Hessian; KS-DFT, open-shell and embedded frequencies, and every TS
  search, use finite differences of analytic gradients (Step 5). That costs 6N
  gradients per Hessian, so a C10 cation Hessian is hundreds of gradient
  evaluations. Without Step 5 you
  have intermediate energies, not barriers, and *kcat depends on barriers*.
* **No thermochemistry.** There's no entropy, enthalpy or free-energy
  correction anywhere, from the CLI or Python. You can compute a zero-point
  energy yourself from the frequencies (½Σhν over the real modes). Python's
  `FrequencyResult` doesn't provide one.
* **Relative energies only.** Absolute totals carry basis-set and functional
  errors far larger than the differences you are interpreting.

---

## Golden path B — Which residue should I mutate?

**Answers:** which active-site residues most influence a reactive center, so
you have a ranked short list.

**Does not answer:** what mutation to make. ferric ranks hypotheses. It does
not design, and it does not predict ΔΔG.

### Step 1 — classical pre-screen (cheap, all residues)

`derive_pocket_charges` (in `tools/active_site/pocket_charges.py`) runs
pdb2pqr on the pocket and records each charge's residue
(`residue_ids`, `res_names`). `pocket_field_at_atoms` returns the potential and
field `[phi, Ex, Ey, Ez]` (atomic units) at the sites you give it. It returns
the *total*. To rank residues, group the charges by residue and evaluate each
group separately. [Recipes](recipes.md) §5 has a sketch of the code.

### Step 2 — QM/MM the top few (expensive, short list only)

Use [QM/MM](qmmm.md) (`ferric.QmmmSystem` + `ferric.run_qmmm`). The embedding
energy shift matches `pyscf.qmmm.mm_charge` to <1e-8 Ha. This step is what
turns the answer into physics rather than electrostatics.

The pre-screen exists to make this step affordable. Running QM/MM on every
residue is the thing the funnel pattern is designed to avoid.

### Limits

Classical point charges: no protein polarization response, no sterics, no
conformational change on mutation, no ΔΔG. A flagged residue is a **candidate
for QM/MM**, not a designed mutation.

---

## Golden path C — Screen many ligands down to DFT

**Answers:** of N candidates, which few deserve expensive QM.

Use `tools.pipeline.run_funnel` (repository `tools/`, not the wheel). Don't
write your own loop. [Recipes](recipes.md) §4 has a funnel you can run
(force field → xtb → DFT on three isomers, with its measured output). The
[pipeline notes](../reference/pipeline-golden-path.md) §0b show the
substituent version: parent-relative gating, liability flags and the optional
docking tier.

Why use it instead of your own loop:

* **Failed candidates are dropped and counted, never ranked.** A hand-written
  screen that sorts ascending on a sentinel value silently promotes its
  failures to the top. The funnel exists to prevent that bug.
* **Per-tier wall times.** The tier that actually dominates is often not the
  one the cost table predicts. MEASURED in both recorded runs: the DFT tier
  took 90–96% of the wall time.
* **Early stop** on an empty population.
* **Ascending rank** at every tier, since each tier reports an energy-like
  score. Rank only candidates with the same formula, or rank relative to a
  parent.

**The funnel will produce an ordering that the noise doesn't support.**
MEASURED on a real campaign: the best available ΔΔE noise over a pose ensemble
was 4.07 kcal/mol, against substituent effects of 1–2. Read the
[pharma coverage notes](../reference/pharma-use-case-coverage.md) before you
rank anything.

---

## Reading the output

| Signal | Meaning |
|---|---|
| `converged = true` | Trust the energy. Absence of this line is not success. |
| `converged = false` | An energy is still printed. It is NOT a result. |
| `exit Some(Stalled)` | The SCF could not progress. Suspect the GEOMETRY -- check the minimum interatomic distance and relax with xtb. |
| `exit Some(MaxIter)` | It WAS progressing and ran out of iterations. Raise `max_iter` (173 is normal for a delocalised cation) and check `energy_conv` is reachable under DF. |
| `exit Some(Diverged)` | The energy climbed for several iterations in a row. Suspect the geometry or the charge/multiplicity before the solver. |
| charge/multiplicity error | Your `.xyz` atom count is wrong. Read the arithmetic it prints. |
| Disagreement with another code ~1e-5 Ha | Check the grid: ferric `(75,110)` vs PySCF ~`(75,302)`. Scales with **atom count**. |
| Larger KS-DFT disagreement that grows with size | ferric density-fits Coulomb by default in KS-DFT. See [Recipes](recipes.md) §6. |
| A correlation energy that moves when nothing physical changed | `density_conv` was never set. |

## Before publishing any number

Check the capability's grade in
[Capabilities and validation](../reference/validation.md). A method being available
from the CLI doesn't mean its numbers are production-grade. The grades
are Proven, Proven (narrow), Smoke and Spike; a few kinds are not graded.
