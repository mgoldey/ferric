# ferric for agents

Read this first if you are an automated agent driving ferric. It is a map, not
a tutorial: what exists, where it lives, and which mistakes cost the most time.

## Start here

**Do not build from source unless you need to.** A prebuilt wheel exists and
gets you a working CLI in seconds instead of ~45 minutes:

```bash
pip install ferric
ferric examples/water-rhf.toml
```

`pip install` puts a `ferric` executable on PATH that runs the same CLI.

### Three channels, and which one you want

| Channel | What | How |
|---|---|---|
| **PyPI release** | Published on a `v*` tag only. The default. | `pip install ferric` |
| **PyPI pre-release** | Release candidates, same tag pipeline | `pip install --pre ferric` |
| **GitHub nightly** | Built every night (`53 10 * * *` UTC) from `main`; a run artifact, not a PyPI upload | download from the Wheels workflow run, then `pip install ./<wheel>` |

Use the **nightly artifact** when you need a fix that has landed on `main` but
has not been tagged — which is common on an actively developed branch. Do not
build from source just to get an unreleased fix.

**`ferric-mpi` is NOT on PyPI.** The wheels workflow explicitly refuses to
upload it (OpenMPI 4.x ABI lock-in plus an unbundled system-MPI runtime
requirement), so `pip install ferric-mpi` will not get you the MPI build even
though `site/src/using/installation.md` currently says it will. For MPI, build
from source with `cargo build --release --workspace --features mpi`.

Building from source is for changing ferric itself. If you are running
calculations, the wheel is the entry point. (A plan this repo was handed once
budgeted 1.5 hours to "build from source, no PyPI wheel is published" — the
premise was simply false, and it was the single largest line item.)

From a source checkout instead:

```bash
cargo run --release -- examples/water-rhf.toml
```

Either way, `converged = true` plus an energy means it works. If not, stop and
fix that ([Installation](installation.md)) — nothing else will work and the
failures will be confusing.

## Capability → entry point

| You want | Use | Notes |
|---|---|---|
| Energy of a molecule | CLI TOML, `method.kind` | [Methods overview](../methods/index.md) lists every `kind` |
| Ion / radical / metal center | `[molecule] charge`, `multiplicity` | **No example sets these.** [Recipes](recipes.md) §2 |
| Optimized geometry | `method.task = "optimize"` | `examples/h2-lda-opt.toml` |
| Screen many ligands | `tools/pipeline/funnel.py` | dock→xtb→DFT, tiered. Do not rebuild it. §4 |
| Pose relaxation, binding energy | `tools/active_site/` | 13 modules; see its own README |
| Rank residues for mutation | `pocket_charges` + `pocket_field` + QM/MM | Ranks hypotheses, does not design. §5 |
| QM/MM embedding | `ferric_scf::qmmm` | Validated vs `pyscf.qmmm`, <1e-8 |
| Properties for ML | NPZ export | 27 fields; `ferric-export` |
| Python instead of TOML | `ferric` module | `run_rhf`, `run_ksdft`, `run_rimp2`, ... |

## Before you trust a number

**Check the maturity grade.** Not every capability in this repo is equally
validated. `wiki/VALIDATION.md` is the per-capability proven/smoke/stub matrix
and it is the authority — a method existing in the CLI does not mean its
numbers are production-grade. Read the row before quoting the value.

**Check convergence, not just exit code.** `converged = true` is the signal.
A run that hit `max_iter` still prints an energy.

**Check the grid if comparing to another code.** Default `(75, 110)` vs PySCF's
~`(75, 302)`; ~1e-5 Ha on water, scaling with **atom count** not basis size.

## Failure modes that cost the most time

1. **Charge/multiplicity parity.** Odd electron count needs even multiplicity.
   The error message prints the arithmetic and the rule — read it; it is
   telling you your `.xyz` is wrong, not that ferric lacks the feature.
2. **Debug vs release.** A debug build is ~10-50x slower. If a job seems hung,
   check which binary you are running before concluding anything. This is the
   single easiest mistake to make: `cargo run --release` is correct, but a
   hand-invoked `./target/debug/ferric` silently is not. Confirmed the hard way
   while writing these docs.
3. **MPI is a WORKSPACE feature, not a CLI one.** Reading `Cargo.toml` files
   per-crate will tell you `ferric-cli` has no `mpi` feature -- true, and
   misleading. It lives on `ferric-scf` and `ferric-core`. The working build is
   `cargo build --release --workspace --features mpi`; the natural guess
   `-p ferric-cli --features mpi` fails.
4. **Memory: the budget predicts, the cgroup enforces. Use both.**
   THE FULL ANSWER, in one place -- other pages point here.

   ferric resolves a memory budget (`[memory] budget_gb`, `FERRIC_MEM_BUDGET_GB`,
   or auto at `0.8 x available RAM`) and prints it at startup. Where a code path
   has been migrated to the shared `MemoryPlan`, that budget is genuinely
   enforcing: the job is REFUSED up front with a breakdown naming the dominant
   term. Grid planes, the Laplace sparsify peak, co-resident VVVV blocks and the
   Hirshfeld grid scan are all charged against it.

   **UPDATED 2026-09-18.** The budget is now a shared, DEBITED ledger
   (`MemoryPool`), not a ceiling each call site re-reads. SCF, KS-DFT, MP2,
   RPA, CC, GW and the gradient paths all charge against one pool, so
   whichever plane asks second sees only what the first left. Before this, the
   failure below happened with NOT ONE GATE FAILING: the DF 3-index tensor
   asked "do I fit in 4.72 GiB?" (yes), the grid AO cache asked the same
   question (yes), and the process held the sum.

   **Coverage is still not total, and the budget is still not an RSS cap.** It
   bounds the dominant tensor planes, not every allocation -- libint2's
   C++-side engine pools, for one, are not modelled. Treat the printed number
   as a floor on what the job will use. MEASURED here, 27-atom KS-DFT,
   def2-SVP, release binary, BEFORE the pool landed:

   ```text
   [ferric] memory budget: 4.72 GiB  [source: auto]
   WALL 1344 s   MAXRSS 6.04 GiB   SP_EXIT=137      <- SIGKILL, 28% over
   ```

   Lowering `FERRIC_MEM_BUDGET_GB` does not make an over-budget job fit: it
   changes blocking behaviour, not the process total. (Before the pool, it did
   not even compose -- a smaller number was handed independently to each call
   site. That specific defect is fixed.)

   **Size the budget from a GRADIENT, not from a single-point energy.**
   MEASURED, benzene/cc-pVDZ/PBE/RI-J:

   ```text
   SCF peak alone      : 0.616 GB
   SCF + gradient peak : 3.033 GB      <- the gradient adds 3.9x the SCF
   ```

   The KS gradient carries `ddchi`, the AO SECOND derivatives on the grid --
   `(3, 3, nbf, npts)`, nine times the `chi` plane. It scales as `nbf^1.93`:
   1.8 GB at 66 basis functions, 3.0 GB at 114. A budget sized from an energy
   run under-provisions a geometry optimization by roughly 4x, and the first
   optimization step is where it fails.

   **The cgroup is the only hard ceiling**, and it is what keeps a miss from
   becoming everyone else's problem:

   ```bash
   scripts/ferric-limited --max=8G --high=7G -- ferric input.toml
   ```

   An unbounded overshoot fires a GLOBAL OOM (`CONSTRAINT_NONE`), which picks
   victims machine-wide: the run above took the browser, the terminal
   multiplexer and the editing session with it. Inside a cgroup, only the job
   dies.

   Rules of thumb: **~6 GB for 27 atoms at def2-SVP**; serialize QM against
   builds and test suites; and `dmesg | grep -i oom` is how you confirm a kill,
   never the log's appearance.

5. **`pgrep -f` LIES. Never decide a job's state from a process-name match.**
   This is the single most expensive mistake in this session: it fired SIX
   times, each costing an iteration, and every instance exited 0 and looked
   exactly like "still running".

   ```bash
   # WRONG -- matches your own grep, a defunct wrapper, or a waiter script
   pgrep -f "cargo build --release"     # matched the bash -c that launched it
   pgrep -f "release/ferric input.toml" # matched a 0%-CPU defunct wrapper

   # RIGHT -- sort by actual work and read the top of the list
   ps -eo pid,stat,pcpu,rss,etimes,comm --sort=-pcpu | head -5
   ```

   The forms it takes, all observed here: the pattern matches your own `grep`
   or `pgrep` command line; it matches the `bash -c` wrapper rather than the
   worker; it matches a defunct/zombie entry reading 0% CPU while the real
   worker runs under a different pid; or the process genuinely never started
   because a redirect failed, and the match is your own waiter loop.

   **Check for evidence of WORK, not for a name.** `%CPU`, RSS, load average,
   `pgrep -x rustc | wc -l`, log byte growth. A build with zero `rustc` and
   load 0.4 is not building, whatever `pgrep -f "cargo build"` says.

6. **A frozen log is not a dead job.** stdout is buffered, so a long QM run
   legitimately writes nothing for tens of minutes. Distinguishing the three
   states needs three different checks:

   | symptom | what to actually check | the wrong inference |
   |---|---|---|
   | truncated log, no error | top of `ps --sort=-pcpu` (NOT `ps -p <pid>` -- the pid is the thing you got wrong) | "it crashed" |
   | `Terminated` | was there a `timeout`? only then `dmesg \| grep -i oom` | "OOM" -- SIGTERM from timeout(1) is far more common |
   | 0% CPU on "the" pid | find the real worker by CPU, not by name | "dead" |

   Swap occupancy is likewise not swap pressure: a full swap with `si=0 so=0`
   in `vmstat` is fine. Nonzero si/so is the tell.

7. **SERIALIZE QM JOBS.** Two concurrent ferric runs on a 12-core box did not
   halve throughput -- they pushed a seconds-long single-point past a 15-minute
   timeout. Contention is superlinear here because each run wants all cores and
   several GB. Run one at a time, or use the funnel
   ([Golden paths](applications.md)), whose per-tier timing exists for this.

8. **Threading.** Set `OPENBLAS_NUM_THREADS=1`. BLAS>1 under rayon segfaults or
   miscomputes in this repo. This is not a performance tip; it is correctness.
9. **Filters that match nothing.** Grepping runner output for a pattern that
   never appears looks exactly like "the job produced nothing." Confirm your
   filter against known-good output before believing a null result.
10. **Under-converged density.** `energy_conv` alone does not converge the
   density. Correlation energies and properties inherit the full first-order
   density error; E_HF does not, because it is variational.

## Where the documentation lives

| Page | Audience |
|---|---|
| [Quickstart](quickstart.md) | First run, 60 seconds |
| [Recipes](recipes.md) | Task-driven: runnable workflows |
| [Golden paths](applications.md) | End-to-end studies |
| [Installation](installation.md) | Building from source |
| [Python API](python.md) | Driving ferric from Python |
| [Methods overview](../methods/index.md) | Which `method.kind` does what |
| [What is validated](../reference/validation.md) | **Per-capability maturity. Read before trusting numbers.** |

## Honest scope

ferric is a quantum chemistry engine: energies, gradients, properties. It is
**not** a protein-engineering tool, a docking program, or an MD code. The
`tools/` layer wires it to RDKit/xtb/docking for screening workflows, but the
QM is the product. When a task asks for something ferric cannot do — designing
mutations, predicting ΔΔG, characterizing transition states (no analytic
Hessians) — say so rather than approximating it with something adjacent.
