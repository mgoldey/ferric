# scripts/ — reference generators, benchmark sweeps, scratch

## Correctness CI gate (`ci-gate.sh`)

This repo has no hosted CI: no `.github/workflows/`, no `.gitlab-ci.yml`, and
**no git remote at all**. `scripts/ci-gate.sh` is the local stand-in --
a script that runs `cargo test --workspace` (the default, non-`--ignored`
set -- ~910/1065 tests as of 2026-07-21, counted via
`grep -rc '#\[ignore' crates/*/tests/*.rs crates/*/src/*.rs`
(155 `#[ignore]`d, deliberately-slow dispersion/CPKS/BSE/RPA benchmarks
excluded from routine runs on purpose) plus
`cargo clippy --workspace --all-targets` checked against the numerical-lint
allow-list in root `Cargo.toml`'s `[workspace.lints]` block. These counts
drift as tests get added -- re-run the grep above rather than trusting this
number if it matters to what you're doing.

**Install it once per checkout** (git hooks aren't tracked/auto-installed):

```
scripts/install-hooks.sh
```

This wires `scripts/ci-gate.sh` as a `pre-push` hook. Git hooks live in the
*common* `.git/hooks` dir, shared across every linked worktree of this repo
(not per-worktree) -- installing once from any worktree covers all of them.
After installing, every `git push` runs the gate first and blocks the push
on a nonzero exit.

Run it by hand any time without pushing:

```
scripts/ci-gate.sh
```

**Allow-listed clippy lints** (see the Cargo.toml comment block above
`[workspace.lints]` for the full per-lint rationale; the short version --
these are numerical/formatting idioms this codebase already tolerates
elsewhere per its "prod lib is 0-warning except numerical lints" policy, not
guessed): `excessive_precision`, `neg_multiply`, `needless_range_loop`,
`doc_lazy_continuation`, `too_many_arguments`, `derivable_impls`,
`field_reassign_with_default`, `redundant_pattern_matching`,
`unnecessary_map_or`, `unnecessary_min_or_max`, `type_complexity`,
`needless_borrows_for_generic_args`, `if_same_then_else`, and the
non-clippy `unused_parens`. This list was derived empirically (ran clippy on
current `main`, categorized every warning that fired) -- not copy-pasted.
Everything else still fails the gate.

**Load awareness / timeouts**: a test either passes or fails regardless of
how busy the box is, so gate correctness itself is load-independent. The
practical risk is the gate script *timing out* under heavy contention
(shared `/tmp/ferric-cargo.lock` with other agents/jobs) and that being
misread as "the gate found nothing" rather than "the box was busy." The
script prints `uptime` load average at start and annotates a timeout exit
with whether load exceeded `nproc/2` at the time. Default per-step timeout
is 30 minutes (`CI_GATE_TIMEOUT_SECS` to override) -- generous for a
warm/incremental build (single-digit minutes per `docs/performance.md`) but
not for a cold/clean build (~29 min) stacked with lock contention. If you
hit a timeout:
  - Retry once the box is quieter (check `uptime`), or
  - Raise `CI_GATE_TIMEOUT_SECS` for one run, or
  - `git push --no-verify` to skip the gate for that push -- only do this
    with an explicit acknowledgment (commit message / PR note / chat to
    whoever reviews) that the gate was bypassed, not silently passed. Never
    treat a `--no-verify` push as equivalent to a green gate.

**Complexity regression check (Step 3)**: alongside tests/clippy, the gate
also runs `complexity_gate.py`, which uses `rust-code-analysis-cli` (install
via `cargo install rust-code-analysis-cli --locked`) to compute per-function
cyclomatic complexity (CC) and maintainability index (MI) across `crates/`
and compare them against a checked-in snapshot (`complexity_baseline.json`).
This is deliberately a **regression tracker, not an absolute threshold** --
this repo already tolerates high-CC numerical kernels that mirror the
underlying physics (`solve_rhf` CC=134, etc. -- see the `too_many_arguments`
allow-list rationale above), so an absolute ceiling would either catch
nothing or immediately fail on already-accepted code. The gate instead fails
only when a function gets *worse* than its baseline by more than a small
noise tolerance, or when a brand-new function appears with complexity far
above the codebase's own historical worst-case. If a regression is a
deliberate, reviewed change (a genuine refactor or an intentionally-grown
function), regenerate the baseline and commit it alongside the change so the
shift is visible in review:

```
python3 scripts/complexity_gate.py --update-baseline
```

This step **soft-skips** (does not fail the gate) if `rust-code-analysis-cli`
isn't installed, or if `CI_GATE_SKIP_COMPLEXITY=1` is set.

## What belongs in git

- **Reference generators** (`gen_pyscf_*.py`, `cc/`, `cpks/`, `minimax/`): scripts
  that produce `testdata/reference/*.json`. Committed so every reference number
  is reproducible.
- **Benchmark sweeps** (`dosd*/`, `gw100/`): the generator + runner + analysis
  scripts, input geometries/TOMLs, and the *final* results (JSON/CSV/MD) plus
  curated evidence files that docs cite. Each sweep dir must keep the scripts
  that regenerate it.
- **The CI gate** (`ci-gate.sh`, `install-hooks.sh`, `complexity_gate.py`,
  `complexity_baseline.json`): the only local correctness-gate mechanism this
  repo has (see above). Always committed.

## What does NOT belong in git

- Raw run logs, timestamped stdout, per-job `.npz`/`.log` artifacts (regenerable
  from the committed TOMLs), one-off probe/debug scripts, `__pycache__`.

## Scratch zone

`scripts/queue/out/` is **gitignored**. It is the dumping ground for load-gated
queue stdout, one-off PySCF cross-checks, and debugging logs. To promote a file
into the permanent record (evidence a doc cites), use `git add -f` deliberately —
and prefer moving it next to the doc that cites it.

## Failure patterns to avoid (learned the hard way)

- **Don't clone a sweep dir** (`dosd` → `dosd2` → `dosd3`): parameterize the
  existing generator instead. `dosd3` was cloned without its generator/runner
  scripts, so its TOMLs are now hand-maintained.
- **Don't share one mutable results file across concurrent runs**: the legacy
  `gw100/results.json` was clobbered by same-basis concurrency. Write per-run
  or per-basis files and merge afterward (`gw100/run_sweep.py` per-basis files;
  `gw100/fast_lane.sh` + `merge_driver_log.py`).
- **Don't hardcode absolute paths** (`sys.path.insert(0, "/home/matt/qc/pyscf")`):
  guard with an env var or `pathlib` relative to the repo root.
- **Commit results promptly**: sweep result files left dirty in the working tree
  for weeks are the raw material for the next accidental clobber.
