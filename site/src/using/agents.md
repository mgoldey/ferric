# ferric for agents

Read this first if you are an automated agent driving ferric. It's a routing
page: what exists, where each thing is documented, and which mistakes cost the
most time. The linked pages hold the details.

## Start here

**Don't build from source unless you need to.** The prebuilt wheel installs in
about a minute. A source build takes ~30 minutes, most of it libint2.

```bash
pip install ferric
git clone https://github.com/mgoldey/ferric && cd ferric   # for examples/, testdata/, tools/, scripts/
ferric examples/water-rhf.toml
```

Expect `converged = true` and `energy = -74.9631468000 Hartree`. If you don't
get that, stop and fix the install ([Installation](installation.md)). Nothing
else will work, and the failures will be confusing.

`pip install` puts a `ferric` command on `PATH` that runs the same CLI as the
source build. The wheel holds only the compiled library and that command. The
`examples/`, `testdata/`, `scripts/` and `tools/` directories need the clone.

### Which wheel

| Channel | What | How |
|---|---|---|
| **PyPI** | Built by the Wheels workflow on a `v*` tag. So far only pre-releases (`0.1.0rc*`) exist, and pip and uv pick them when there's no final release. | `pip install ferric` |
| **GitHub nightly** | Built every night (`53 10 * * *` UTC) from `main`. It's a workflow-run artifact, not a PyPI upload. | Download it from the Wheels workflow run, then `pip install ./<wheel>` |

Use the **nightly artifact** when you need a fix that has landed on `main` but
hasn't been tagged. Don't build from source just to get an unreleased fix.

Build from source when you are changing ferric or need MPI. MPI builds are
source-only (see [Installation](installation.md)).

## Capability → entry point

| You want | Use | Where it's documented |
|---|---|---|
| Which methods exist, with which tasks and validation grade | — | [Capabilities](../reference/capabilities.md) |
| An energy from a SMILES string | `tools.structure.from_smiles` + `ferric.run_dft` | [Recipes](recipes.md) §0 |
| An energy from a TOML file | CLI, `method.kind` | [Recipes](recipes.md) §1, [input reference](../reference/input.md) |
| An ion, radical or metal center | `[molecule] charge`, `multiplicity` (Python: on `Molecule.from_xyz`) | [Recipes](recipes.md) §2 |
| An optimized geometry | `method.task = "optimize"` | [Recipes](recipes.md) §3 |
| Harmonic frequencies | `method.task = "frequencies"`, `ferric.run_frequencies` | Finite differences of analytic gradients; `examples/water-frequencies.toml` |
| A transition state and its reaction path | `ferric.run_saddle`, `ferric.run_irc` (Python only, closed shell) | [Golden paths](applications.md) Step 5 |
| A screen of many ligands | `tools.pipeline.run_funnel` | [Recipes](recipes.md) §4 |
| A pose relaxation or binding energy | `tools/active_site/` | [Pipeline notes](../reference/pipeline-golden-path.md) |
| A residue ranking for mutation | `pocket_charges` + `pocket_field`, then QM/MM | [Recipes](recipes.md) §5. It ranks hypotheses and doesn't design mutations. |
| QM/MM embedding | `ferric.QmmmSystem`, `ferric.run_qmmm`, CLI `[qmmm]` | [QM/MM](qmmm.md) |
| Toxicity and liability flags | `python -m tools.tox` | [Toxicity screening](toxicity.md) |
| A machine-readable record of a run | `<input>.ferric.jsonl`, written by default | [Run logs](run-logs.md) |
| Python instead of TOML | the `ferric` module | [Python API](python.md) |

## Before you trust a number

**Check the grade.** Not every capability is equally validated. A method being
available from the CLI doesn't mean its numbers are production-grade. Read its
row in [Capabilities](../reference/capabilities.md) and
[What is validated](../reference/validation.md) before quoting a value.

**Check convergence, not the exit code.** `converged = true` (CLI) or
`.converged` (Python) is the signal. A run that hit `max_iter` still prints an
energy. In the run log, read `run_end.converged`, and for correlated methods
read the `result` record ([Run logs](run-logs.md)).

**When comparing to another code, check density fitting and the grid.**
[Recipes](recipes.md) §6 covers both.

## Failure modes that cost the most time

1. **Charge/multiplicity parity.** An odd electron count needs an even
   multiplicity. The error message prints the arithmetic and the rule, and it
   means your `.xyz` or your charge is wrong. ferric can handle ions.
2. **Debug instead of release.** A debug binary is far slower. MEASURED: a
   27-atom cation didn't converge a single point in 34 minutes on one. If a job
   seems hung, check which binary is running. `cargo run --release --bin ferric`
   and the wheel are both release builds. `./target/debug/ferric` is not.
3. **MPI is a workspace feature, not a CLI one.** `ferric-cli` has no `mpi`
   feature of its own, so `-p ferric-cli --features mpi` fails. The working
   build is `cargo build --release --workspace --features mpi`.
4. **Threading.** Don't set `OPENBLAS_NUM_THREADS` above 1. The CLI and
   `import ferric` pin OpenBLAS to one thread when the variable is unset and
   honour an explicit value, and `cargo` sets it to 1 only when it is unset
   (an exported value wins). Multithreaded BLAS
   under ferric's rayon parallelism can crash (LU routines) or oversubscribe
   the machine. It's a correctness setting, not a performance tip.
5. **Under-converged density.** `energy_conv` alone doesn't converge the
   density. Correlation energies and properties inherit the full first-order
   density error. E_HF doesn't, because it's variational. Set `density_conv`.
6. **Serialize QM jobs.** Two concurrent ferric runs don't just halve
   throughput. MEASURED: a pair of them pushed a single point that normally
   takes seconds past a 15-minute timeout. Each run wants all the cores and
   several GB. Run one at a time, or use the funnel, which times each tier.

### Memory: the budget predicts, the cgroup enforces

ferric works out a memory budget from `[memory] budget_gb`,
`FERRIC_MEM_BUDGET_GB`, or, if neither is set, 0.8 × available RAM. The CLI
prints that budget at startup and installs it as one shared, debited pool for
the whole process: the three-index RI tensors, the DFT grid's AO cache and the
large tensors of the MP2, RPA, GW and CC methods reserve their bytes from it,
so two allocations alive at the same time cannot each claim the whole budget.
From Python, `memory_budget_gb=` sets the same per-allocation limits but
installs no shared pool, so each check compares its own allocation with the
whole budget. In both, an allocation that does not fit is spilled to disk,
recomputed on demand, or refused with an error naming it, depending on the
allocation (see
[Sharp bits](sharp-bits.md#memory-budget_gb-does-not-cap-the-whole-process)).

**The budget isn't a cap on process memory (RSS).** It covers the dominant
tensors, not every allocation. Basis-sized matrices, integral engines (for
example libint2's C++-side pools), BLAS and per-thread scratch and allocator
overhead aren't charged, so treat the printed number as a floor on what the
process needs. Lowering `FERRIC_MEM_BUDGET_GB` changes how the charged work
is split into blocks. It doesn't shrink the uncharged part.

**Size the budget from a gradient, not a single point.** MEASURED on
benzene/cc-pVDZ/PBE/RI-J: the SCF alone peaked at 0.616 GB, and the SCF plus
gradient at 3.033 GB. The KS gradient holds the AO second derivatives on the
grid. A budget sized from an energy run under-provisions a geometry
optimization by roughly 4x, and the first optimization step is where it fails.

**The cgroup is the only hard ceiling.** On Linux with systemd:

```bash
scripts/ferric-limited --max=8G --high=7G -- ferric input.toml
```

An unbounded overshoot triggers the system-wide OOM killer, which can kill
unrelated processes. Inside a cgroup, only the job dies. `dmesg | grep -i oom`
confirms a kill. A truncated log doesn't prove one. As a rule of thumb, plan on
**~6 GB for 27 atoms at def2-SVP**.

### Telling a running job from a dead one

- **Look for evidence of work, not a process name.** `pgrep -f` also matches
  your own `grep`, a `bash -c` wrapper, or a defunct entry at 0% CPU. Sort by
  CPU instead: `ps -eo pid,stat,pcpu,rss,etimes,comm --sort=-pcpu | head`.
- **A log that has stopped growing doesn't mean the job is dead.** stdout is
  buffered, so a long QM run can legitimately print nothing for tens of
  minutes. The run log ([Run logs](run-logs.md)) is flushed record by record,
  so watch it instead.
- **`Terminated` usually means timeout(1), not the OOM killer.** Check whether
  a `timeout` wrapped the command before you look in `dmesg`.
- **A grep filter that matches nothing** looks exactly like "the job produced
  nothing". Test the filter against known-good output first.

## Honest scope

ferric is a quantum chemistry engine: energies, gradients, finite-difference
Hessians and properties. It isn't a protein-engineering tool, a docking
program or an MD code. The `tools/` layer connects it to RDKit, xtb and
docking for screening, but the QM is the product.

It has no analytic Hessians, and no thermochemistry (entropy, enthalpy, free
energy). Transition-state search and IRC are Python-only and closed-shell only
([Golden paths](applications.md) Step 5). If a task asks for something ferric
can't do, such as designing mutations or predicting ΔΔG, say so instead of
approximating it with something adjacent.
