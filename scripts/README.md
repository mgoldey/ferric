# scripts/ — reference generators, benchmark sweeps, scratch

## What belongs in git

- **Reference generators** (`gen_pyscf_*.py`, `cc/`, `cpks/`, `minimax/`): scripts
  that produce `testdata/reference/*.json`. Committed so every reference number
  is reproducible.
- **Benchmark sweeps** (`dosd*/`, `gw100/`): the generator + runner + analysis
  scripts, input geometries/TOMLs, and the *final* results (JSON/CSV/MD) plus
  curated evidence files that docs cite. Each sweep dir must keep the scripts
  that regenerate it.

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
