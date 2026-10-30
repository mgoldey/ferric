# Run logs

Every ferric CLI run writes a machine-readable JSON Lines log, without being
asked.

The format is ferric's own. It is **not QCSchema**, and there is no QCSchema
exporter. Each record's `schema` field (currently `1`, on `run_start`) versions
this format. It is not a QCSchema version.

```
$ ferric water-rhf.toml
[ferric] JSON run log: water-rhf.ferric.jsonl
...
```

## Why it is on by default

A result whose run left no artifact can't be checked. Stdout captures get
truncated, and a claim such as "173 iterations, converged" that survives only
in a truncated capture can be repeated but never checked. The run log keeps
the record on disk whether or not anyone captured stdout, which is why it is
on by default.

Opt out explicitly if you need to:

```toml
[output]
json = false          # no log
# json = "runs/a.jsonl"  # or put it somewhere specific
```

or from the command line:

```
ferric --no-json input.toml
ferric --json runs/benzene.jsonl input.toml
```

The default path is the input file's name with `.ferric.jsonl` in place of its
extension, written beside the input.

**A log never fails a calculation.** If the path cannot be opened or the disk
fills, ferric warns once on stderr and the run continues.

## Format

One JSON object per line, written and flushed as it happens — not buffered to
the end of the run. A job that is OOM-killed, hits a wall-clock limit, or is
Ctrl-C'd at iteration 90 of 100 leaves those 90 iterations on disk. Only the
final line may be truncated; everything before it parses.

```
jq -c 'select(.record=="scf_iter") | {iter, energy, dp_rms}' water-rhf.ferric.jsonl
```

Every record carries `record` (its type), `seq` (a gap-free counter, so a
missing record is detectable) and `t` (seconds since the run started).

### `run_start`

Written before any expensive work, so it survives a job killed in the first
iteration. Carries the ferric version, the git SHA the binary was built from
(`null` if the build had no checkout; a `-dirty` suffix means uncommitted
changes were compiled in), a UTC timestamp, the resolved config — including the
memory budget actually used, not the key as written — and the molecule.

### `scf_iter`

One per SCF iteration: `iter`, `energy`, `de`, `dp_rms`, `dp_max`, `err_max`,
and `grad_rms` where the solver forms one. `rung` says which ladder rung the
iteration belongs to. `method` is `rhf`/`uhf`/`rohf` or their KS counterparts
`rks`/`uks`/`roks`.

These are the quantities the solver already computes for its convergence
decision (see `scf_converged`) — the log reports them, it does not create them.

`dp_rms` and `dp_max` are `"inf"` on the first iteration of each SCF, before
there is a previous density to compare against. JSON has no infinity, so
non-finite values are written as the strings `"inf"`, `"-inf"` and `"NaN"`
rather than as `null`, which would make a NaN indistinguishable from a field
that was simply not reported.

### `guess_scf_iter`

Identical shape, but for the free-atom SCFs the SAD/MINAO initial guess runs
per element. They are real work and can fail, so they are logged — under a
different record type so that free oxygen's energy is never mistaken for the
molecule's.

### `ladder_rung`

One per convergence-ladder rung: `rung`, `tricks` (which accelerators it
enables), `iterations`, `exit`, `energy`, `converged` — and **`max_iter`**.

That last field matters more than it looks. In the CLI, `kind = "rhf"` and
`kind = "ksdft"` run through a convergence ladder, and unless you define your
own `[[scf.ladder]]` rungs, the default ladder sets its own per-rung iteration
caps:

| `kind` | rung 0 | rungs 1–4 |
|---|---|---|
| `rhf` | 60 | 60 / 60 / 80 / 100 |
| `ksdft` | `[scf] max_iter` | 60 / 60 / 80 / 100 |

So for `rhf`, `[scf] max_iter` does not set the per-rung cap. A run that reports
"iterations = 100" is usually rung 4 using up its own budget, not a single
hundred-iteration SCF. The cap and the count are recorded together so the log
can't be misread that way.

### `run_end`

The terminal record: `energy`, `converged`, `exit`, `wall_s`, `cpu_s` and
`peak_rss_bytes` (the high-water mark, not the RSS at exit).

For a post-SCF method the `energy` is the **SCF reference**, not the method's
total. `extra.energy_is` says which. Per-iteration records for MP2/RPA/CC/GW
are not implemented yet.

**Only `task = "energy"` and periodic `task = "optimize"` write `run_end`.**
A molecular `task = "optimize"` and `task = "frequencies"` currently write
`run_start` and the SCF iteration records, but no `run_end`, `ladder_rung` or
`result`. A periodic `[cell]` optimization writes `run_end` (see below), also
when it stops at `max_steps` without converging; it writes none when it ends
in an error. Read the final geometry
and energy of an optimization from stdout. A finished `task = "frequencies"`
writes one `frequencies` record instead (below). A missing `run_end` in such a log
doesn't mean the run failed (MEASURED on `examples/h2-lda-opt.toml`,
2026-09-23).

### Precision provenance (`precision`)

`run_end.extra.precision` records which mixed-precision kernels the **whole
run** went through: every kernel, counted from the start of the run to
`run_end`, whichever code called it (the SCF, the method, and MP2 variants such
as SCS-MP2, SCS-MP2(2terfc), terfc `att-rimp2` and the double hybrids, which
carry no record of their own). `run_end.extra.scf_precision` is the SCF's own
record. A run that writes no `run_end` from the molecular dispatch (open-shell
`uhf`/`rohf` energies and their KS forms, `task = "optimize"`,
`task = "frequencies"`, `wb97x-l-v`, `b2plyp`, `dsd-pbep86`) writes the
whole-run record as a `precision` record instead. A `rimp2` run's
`result.components.precision` covers the MP2 correlation energy alone (closed
shell and unrestricted). The Python results carry
the same object as `RhfResult.precision`, `UhfResult.precision`,
`DftResult.precision`, `RiMp2Result.precision` and
`RiMp2Result.scf_precision`. An all-f64 run, which is every run that does not
set `[precision] mode = "mixed"`, says so explicitly:

```json
{"precision": "f64", "counter_scope": "process", "kernels": []}
```

Otherwise `kernels` has one entry per kernel that was allowed to run mixed
for that result, or that ran mixed:

| field | meaning |
|---|---|
| `kernel` | the kernel name (`rimp2-energy`, `xc-vxc`, `dfj-pack`, `dfk-occ`, `cosx-kern`) |
| `requested` | `mixed` when `[precision]` allowed it, `f64` when only a test seam ran it |
| `mode` | `mixed` when at least one call ran mixed, else `f64` (allowed, but nothing ran mixed) |
| `device` | where the mixed calls ran: `"cpu"`, or `{"ordinal", "name", "cc"}` of the CUDA device; `null` when `calls_mixed` is 0 |
| `calls_mixed` | calls that ran mixed: one RI-MP2 energy (per spin block when unrestricted), one XC potential pass, one RI-J or DF-K build, one COSX K build with f32 units |
| `calls_f64_fallback` | `{"device_f64", "f32_range", "cpu_f64"}`: calls, or parts of calls, that ran f64 instead (mixed kernel unavailable on the device, a value beyond the f32 range, or the whole call on the CPU) |
| `ladder` | for `xc-vxc`, `dfj-pack` and `dfk-occ`: `{"builds_mixed", "builds_total", "exit_rebuilt_f64"}` of the SCF precision ladder; `null` for the other kernels |
| `error_bound` | `{"kind": "none_computed", "why": ...}` for every kernel today. No kernel computes a per-run error bound yet; the `why` says what exists instead (the measured error maps on the [validation page](../reference/validation.md)). A computed bound would read `{"kind": "a_posteriori", "eh", "method"}`; a number is never invented |

`precision` is `mixed` when any kernel's `mode` is `mixed`. The counts are
process-wide counter differences over the call, which `counter_scope:
"process"` states in every record: two calculations running at the same time
in one Python process can see each other's counts; a sequential run is
exact. The record is observation only: an f64 energy is bit-identical
with or without it.

### `frequencies`

Written once by a finished `task = "frequencies"` run, which writes no
`run_end` or `result`. It carries the numbers the printout rounds, at full
precision:

| field | meaning |
|---|---|
| `hessian_source` | `analytic`, `analytic-moving-grid` or `finite-difference` |
| `energy` | energy at the input geometry, Hartree; with `[dft] dispersion` the corrected total, KS + dispersion |
| `correction_energy` | the dispersion part of `energy` (0 without `[dft] dispersion`) |
| `frequencies_cm1` | vibrational wavenumbers, cm⁻¹, ascending; negative = imaginary |
| `trans_rot_frequencies_cm1` | the projected-out translation/rotation modes (≈ 0, a diagnostic) |
| `asymmetry` | largest \|H_ij − H_ji\| of the raw Hessian, Hartree/Bohr² |
| `n_gradient_evaluations` | gradients the finite-difference Hessian took (0 when analytic) |
| `cartesian_hessian` | the Cartesian Hessian before mass-weighting, row-major 3N × 3N, Hartree/Bohr² |

Non-finite floats are written as the strings `"NaN"`, `"inf"`, `"-inf"`, as
everywhere in the log.

### `result`

**The number a correlated run was launched to produce.** It's written once,
after the method finishes. Plain SCF runs (`rhf`, `uhf`, `rohf`, `ksdft`) don't
write one, because for them `run_end.energy` *is* the answer.

| field | meaning |
|---|---|
| `kind` | the method that produced it, e.g. `"rimp2"`, `"ccsd"`, `"gw"` |
| `total` | the headline number a user would quote |
| `components` | the decomposition that makes `total` checkable — `e_corr`, `e_os`/`e_ss`, the reference energy, and whatever else the method defines |

`total` and `run_end.energy` are **different numbers** for every post-SCF
method. A `kind="rimp2"` run's `run_end.energy` is the RHF energy it was built
on; its `result.total` is the RI-MP2 total. Reading the wrong one silently
gives an uncorrelated answer.

`components` exists because a total alone cannot be reconciled against a
reference implementation — you need the pieces to see *where* a disagreement
comes from.

### `result_unlogged`

A post-SCF method that isn't wired up to `result` yet writes this instead,
naming the `kind`. The methods that do write `result` are `rimp2`,
`oo-rimp2`, `att-rimp2`, `laplace-mp2`, `laplace-sos-mp2`, `scs-mp2`,
`scs-mp2-2terfc`, `mp3`, `ccsd`, `ccd`, `ccsd(t)`, `linlccd`, `drpa`, `mp2-v` and
`rs-mp2-rpa`. For `rimp2`, `drpa` and `linlccd` the record's `components`
carry `"local"`: `null` for the exact method, else `{"scheme", "eps",
"keep_fraction", "integral_direct"}`.

It exists because **silence is ambiguous**: a log with no `result` record could
mean the method does not record one yet, or that the run died before producing
one. Those demand opposite responses from whatever reads the log. This makes
the first case explicit and leaves "the run failed" as the only remaining
reading of a genuinely missing record.

## Example

A water/STO-3G RHF run, abridged:

```json
{"record":"run_start","schema":1,"seq":0,"t":0.002,"ferric_version":"0.1.0","git_sha":"396e0d61eded-dirty","timestamp":"2026-09-18T16:04:12Z","config":{"method":"rhf","basis":"sto-3g","max_iter":100,"density_conv":1e-7,"memory_budget_bytes":13018454425,"rayon_num_threads":12,"mpi_ranks":1},"molecule":{"formula":"H2O","n_atoms":3,"n_basis":7,"n_electrons":10,"charge":0,"multiplicity":1}}
{"record":"guess_scf_iter","seq":1,"t":0.014,"method":"uhf","iter":1,"energy":-0.4665818503784864,"dp_rms":"inf","dp_max":"inf","de":0.4665818503784864,"err_max":0.0,"grad_rms":null,"rung":0}
{"record":"scf_iter","seq":5,"t":0.017,"method":"rhf","iter":1,"energy":-74.68273456362847,"de":74.68273456362847,"dp_rms":"inf","dp_max":"inf","err_max":1.6088693243641559,"grad_rms":0.5469406324593924,"rung":0}
{"record":"scf_iter","seq":13,"t":0.019,"method":"rhf","iter":9,"energy":-74.9631468000391,"de":5.684341886080802e-14,"dp_rms":1.356320515468342e-10,"dp_max":4.726057323267696e-10,"err_max":3.5214886562329184e-14,"grad_rms":1.2777284834157483e-14,"rung":0}
{"record":"ladder_rung","seq":14,"t":0.019,"rung":0,"tricks":["diis"],"max_iter":60,"iterations":9,"exit":"Converged","energy":-74.9631468000391,"converged":true}
{"record":"run_end","seq":15,"t":0.019,"energy":-74.9631468000391,"converged":true,"exit":"Converged","wall_s":0.0189,"cpu_s":0.05,"peak_rss_bytes":128815104,"extra":{"method":"rhf","task":"energy","scf_iterations":9,"energy_is":"total","precision":{"precision":"f64","counter_scope":"process","kernels":[]}}}
```

## Periodic k-point stress

A periodic `[cell]` run with `kmesh` and `stress = true` adds
`extra.stress_hartree_per_bohr3` (the analytic 3×3 stress tensor
σ = (1/Ω) dE/dε of the converged SCF, rows and columns in the Cartesian x, y, z
of the lattice frame, full f64 precision), `extra.stress_gpa` (the same tensor
in GPa) and `extra.stress_unit = "hartree_per_bohr3"`. `run_end.energy` is the
total energy per cell. `run_start.config.stress` records the request; `run_end.extra`
also carries `stress_requested` and `scf_iterations`. If the SCF does not converge
the tensor is absent and `run_end` has `converged = false`, `exit = "not_converged"`;
a library error closes the log with `exit = "error"` and `extra.error`.

## Periodic geometry optimization

A periodic `[cell]` run with `task = "optimize"` (Gamma point or `kmesh`) adds
`extra.final_geometry_bohr` (the returned Cartesian positions in Bohr, flat
`[x0, y0, z0, x1, ...]`, full f64 precision) and
`extra.final_gradient_hartree_per_bohr` (the analytic gradient dE/dR per cell
evaluated at exactly that geometry, same layout). `run_end.energy` is the
energy per cell at that geometry. When the optimization ends in an error (a refusal or a non-converged SCF
step) the run exits with the error and writes no `run_end` record, so neither
key exists; an optimization that stops at `max_steps` without converging
still writes both (`run_end.converged = false`).

## Point-group-blocked runs

With `[scf] symmetry = true` the terminal record carries
`extra.symmetry = {point_group, max_symmetrization_shift_bohr, irrep_nelec, geometry}`.
`irrep_nelec` maps each irrep to its electron count in the converged state,
`geometry` is `"symmetrized"`, and `run_end.energy` is the energy of that
symmetrized geometry (it differs from the input geometry's by the effect of
`max_symmetrization_shift_bohr`, at most 1e-4 Bohr). Without the key there is
no `symmetry` field.

## Does it change the answer?

No, and that is a regression test, not a claim: SCF energies are bit-identical
(`f64::to_bits()`) with the log on and off, asserted across RHF, RKS, UHF and
ROHF in `crates/ferric-scf/tests/runlog_bit_identity.rs`. Logging is
observation, never participation — every value written is one the solver had
already computed for its own convergence decision, and nothing is ever read
back.

## Does it cost anything?

The per-record cost has not yet been measured on a quiet machine. Run
`scripts/bench-runlog-overhead.sh` to measure it; the script explains why the
emit path must be timed directly rather than by diffing whole-process wall
times, and what to compare the result against.
