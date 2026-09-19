# Run logs

Every ferric run writes a machine-readable JSON Lines log, without being asked.

```
$ ferric water-rhf.toml
[ferric] JSON run log: water-rhf.ferric.jsonl
...
```

## Why it is on by default

A result whose run left no artifact cannot be checked. Ferric turned this on
because a load-bearing measurement was lost exactly that way: a 27-atom
PBE/6-31G run reported as "173 iterations, converged, E = -390.3794234093" had
no surviving log — the stdout capture had been truncated to 275 bytes and two
sibling captures were empty — so the claim could not be verified, only repeated.

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

That last field matters more than it looks. The ladder *hardcodes* a per-rung
iteration cap (60/60/60/80/100) that overrides whatever `[scf] max_iter` says.
A run that reports "iterations = 100" is usually rung 4 exhausting its own
budget, not a hundred-iteration run; reading it the other way has already
produced a wrong diagnosis. The cap and the count that hit it are recorded
together so the log cannot be misread that way.

### `run_end`

The terminal record: `energy`, `converged`, `exit`, `wall_s`, `cpu_s` and
`peak_rss_bytes` (the high-water mark, not the RSS at exit).

For a post-SCF method the `energy` is the **SCF reference**, not the method's
total — `extra.energy_is` says which. Per-iteration coverage of MP2/RPA/CC/GW
is not implemented yet.

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
