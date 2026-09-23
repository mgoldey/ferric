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

A result whose run left no artifact can't be checked. The log is on by default
because a key measurement was once lost that way. A 27-atom PBE/6-31G run was
reported as "173 iterations, converged", but its stdout capture had been
truncated, so the claim could only be repeated, never checked.

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

**Only `task = "energy"` writes `run_end`.** `task = "optimize"` and
`task = "frequencies"` currently write `run_start` and the SCF iteration
records, but no `run_end`, `ladder_rung` or `result`. Read the final geometry
and energy from stdout for those tasks. A missing `run_end` in such a log
doesn't mean the run failed (MEASURED on `examples/h2-lda-opt.toml`,
2026-09-23).

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
`scs-mp2-2terfc`, `mp3`, `ccsd`, `linlccd`, `lmp2`, `lmp2-direct`, `mp2-v` and
`rs-mp2-rpa`.

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
{"record":"run_end","seq":15,"t":0.019,"energy":-74.9631468000391,"converged":true,"exit":"Converged","wall_s":0.0189,"cpu_s":0.05,"peak_rss_bytes":128815104,"extra":{"method":"rhf","task":"energy","scf_iterations":9,"energy_is":"total"}}
```

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
