# CI: splitting build from test, and sharding the fast tier (2026-09-18)

A dated analysis (2026-09-18/19) of the CI critical path and the sharding
design that followed. The current workflow is `.github/workflows/ci.yml` and
`.config/nextest.toml`; where they differ from this page, they are
authoritative. All timings are from GitHub Actions job and step records, not
local runs. The caching research is in a separate note.

## 1. Where the wall clock goes (measured, from CI)

`build-and-test` is the critical path in 33/33 sampled push/PR runs.

| step | run 35377097468 (main, green) | share |
|---|---|---|
| Free runner disk space | 1.6 min | 3% |
| Install system deps | 0.3 min | <1% |
| Restore libint2 cache | 0.0 min | 0% |
| Restore cargo cache | 0.0 min | 0% |
| **Check (all targets)** | **2.9 min** | 4% |
| **Test (fast tier)** | **63.2 min** | **89%** |
| Save cargo cache | 0.1 min | <1% |

Job total 70.9 min on this run. These are main-branch figures. A PR run is
shorter (40.3 min fast tier, 51.4 min median job), but main approaches the
step's 100-minute cap, so it sets the headroom. ci.yml's comment giving the
fast tier as "~23 min" held on 2026-09-16; the tier has grown ~2.7x since. The
second-longest job is `mpi` at 8.6 min.

ci.yml has no `needs:` edges, so all seven jobs start together. Parallelising
the existing jobs saves nothing.

Read step durations only from completed runs. An in-flight run reports a step
with no `completedAt` as `-1065422800.6 min`.

## 2. The fast tier is already six invocations

The fast tier runs as six sequential `cargo test` invocations (ci.yml:385-401):

```
cargo test --workspace --exclude ferric-scf --exclude ferric-cc \
                       --exclude ferric-mp2 --exclude ferric-dft --exclude ferric-rpa
for crate in ferric-scf ferric-cc ferric-mp2 ferric-dft ferric-rpa; do
    cargo test -p $crate --lib --test <each non-SLOW target>
done
```

So sharding needs no `--partition` machinery and no new test runner: the five
per-crate invocations plus the blanket pass are already disjoint units, run
one after another in a single job. Turning them into a matrix is a workflow
change, not a test-infrastructure change.

## 3. A per-crate matrix is unbalanced

Test-target counts (`ls crates/*/tests/*.rs`), measured 2026-09-18:

| crate | test targets | share |
|---|---|---|
| ferric-scf | 83 | 32% |
| ferric-rpa | 45 | 18% |
| ferric-mp2 | 34 | 13% |
| ferric-cc | 18 | 7% |
| ferric-dft | 18 | 7% |
| everything else (blanket pass) | 59 | 23% |
| **total** | **257** | |

A shard containing ferric-scf carries a third of the targets on its own, so the
critical path would be set by that shard, not by 1/N of the total. Target
count is also a poor proxy for time, since a few binaries dominate (the reason
the SLOW list exists). Shard balance has to come from per-target durations,
which CI did not record per target.

## 4. The fast tier is dominated by execution

The CI log records each test binary's run time in its `finished in <N>s` line.
Summing them on run 35377097468 (checked twice):

```
$ gh run view 35377097468 --log | grep -oE 'finished in [0-9.]+s' \
    | awk '{s+=$3} END {printf "%.1f min across %d binaries\n", s/60, NR}'
52.3 min across 344 binaries
```

52.3 min of the 63.2 min step is test execution; ~11 min is compile and link.
A caching change can address at most ~11 min. Sharding is a question of how
execution is scheduled, not of build reuse.

### Execution time is concentrated (same run)

| rank | duration | share of 52.3 min |
|---|---|---|
| 1 | 801.8 s (13.4 min) | 25.6% |
| 2 | 285.1 s (4.8 min) | 9.1% |
| 3 | 243.1 s (4.1 min) | 7.7% |
| 4 | 124.6 s (2.1 min) | 4.0% |
| 5 | 116.4 s (1.9 min) | 3.7% |

The top binary is `cdft_coupling_hene`: 4 test functions running def2-SVP cDFT
He-Ne coupling scans in parallel threads on a 4-core runner, so it is
CPU-saturated.

13.4 min is a floor on the critical path for any scheme that keeps that binary
intact, including nextest and sharding. Sharding cannot split one saturated
binary.

Runner concurrency for this repo is unknown. With only ~2 runners, three shards
queue instead of running concurrently and save nothing. `gh` does not show it
without a billing query.

## 5. Build and test as separate jobs: moving artifacts

One shape is a `build` job running `cargo test --no-run`, then N test jobs
consuming the binaries. Moving compiled artifacts between jobs has two
constraints:

- `target/` is not cached (ci.yml:279). It measured ~9 GB per PR merge ref
  against a 10 GB limit and evicted the libint2 cache, which costs ~80 min to
  rebuild. An artifact-passing scheme must not recreate that.
- The test binaries alone are smaller than `target/`, but they link
  dynamically against OpenBLAS/libxc/libint2 and carry absolute paths. Whether
  they run unmodified on a fresh runner was not verified.

`cargo-nextest` was not installed at the time. Its `--partition count:i/N` is
the standard way to shard a Rust suite, and it records per-test timings, which
section 3 needs for balancing.

## 6. Ranked options

1. **`cargo-nextest`.** `cargo test` runs test BINARIES serially (each binary
   parallelises only its own tests) -- a documented cargo limitation,
   rust-lang/cargo#5609. ferric runs 344 binaries and most are tiny, so the
   short ones run one at a time with cores idle. nextest runs them
   concurrently. Estimated 63 -> ~25 min; the 13.4 min floor is measured.
   nextest is process-per-test, so N concurrent processes each spawn a rayon
   pool and each read `FERRIC_MEM_BUDGET_GB` as a per-process budget: an
   N-times memory overcommit. It needs a `.config/nextest.toml`
   (`threads-required`, `test-groups`). nextest does not run doctests, so
   `cargo test --doc` must stay as a separate step.
2. **Split or defer `cdft_coupling_hene`.** It is 25.6% of all execution time
   and sets the floor for every parallel scheme above. Split it into separate
   binaries or defer it to nightly; do not shrink the physics. ci.yml's
   deferrals move slow tests without shrinking them, because an exactness
   anchor shrunk until it stops firing is inert.
3. **Then shard or split build from test.** `cargo nextest archive` moves built
   binaries between jobs and handles the path/`CARGO_MANIFEST_DIR` remapping
   that breaks hand-rolled `--no-run` + upload-artifact schemes. It gains
   nothing over plain nextest until #2 is done, and its archive size was
   unmeasured against a runner with ~14 GB free that has already failed with
   "No space left on device".
4. **Leave caching unchanged.**
   - Not caching `target/` is correct: a cache saved on a PR is scoped to the
     merge ref and cannot be restored by main or another PR, so it only
     consumes quota and evicts the 80-min libint2 artifact.
   - The 23 apparent duplicate cache keys share a key but differ in scope
     (20 x `refs/pull/N/merge`, 4 x main, 1 branch). Caches are keyed on
     (key, scope) and are immutable; a same-scope re-save is skipped with a
     warning. Total usage is 0.98 GB of 10 GB.
   - `Swatinem/rust-cache` does not cache workspace crates, which is where
     ferric's multi-hundred-MB test binaries live.
   - `sccache` cannot cache crates that invoke the system linker (every `bin`,
     so every integration test binary here), per its own docs.
5. **Keep the libint2 prefix cache.** 42 MB, ~0 restore time, 0 misses in 14
   sampled runs, pinned version. Caching the installed artifact skips
   cmake+make entirely; a compiler cache would not.
6. **Leave `fmt` and `mpi` alone.** Merging `fmt` into `pre-commit` or trimming
   `mpi`'s preamble saves nothing: both run in parallel with a 63-min critical
   path.

## 6b. Lint steps gated behind the test tier

PR #88's `build-and-test` failed on Clippy (`function 'pairing_at' is never
used` in `crates/ferric-scf/tests/cdft_coupling_hene.rs`) after 30 minutes,
because the lint steps ran after the tests in the same job. Clippy takes
1.8 min and Check 2.9 min against the 63-min test step. Running them as their
own job, with no `needs:` edge and no sharding, reports this class of failure
in ~3 min instead of ~30.

The dead code came from an edit that cut that binary's cost:
`pairing_at_capped` superseded `pairing_at`. The same 800-second binary is
both the critical-path floor and the file with the most cost-reduction edits,
so option #2 above touches it too.

## 6c. Result: separate lint job

PR #94 runs check, clippy and rustdoc as a separate job. First CI run
(run 35407496129, job 105800175202):

```
lint (check, clippy, rustdoc): PASS
  started  2026-09-18T23:55:04Z
  finished 2026-09-19T00:01:25Z
  => 6 min 21 s
```

The same three commands inside `build-and-test` ran after the 63-min test
tier; PR #88's clippy failure took ~30 min to surface. The separate job gives
the same signal in 6m21s. It re-pays ~2 min of setup (disk-free, apt, cache
restore), which accounts for the difference from the ~6 min sum of the three
steps. libint2 restored from cache.

## 6d. Design: 4-way binary-level sharding (2026-09-19)

Decision: 4 shards, cdft deferred, nextest with a config, archive-based
build/test split. The design inputs below are measured.

### Why 4, and where the curve flattens

Per-binary times from CI run 35377097468 (324 binaries paired to their
`finished in Ns` lines), LPT-partitioned:

| shards | makespan | ideal |
|---|---|---|
| 2 | 26.1 min | 26.1 |
| 3 | 17.4 min | 17.4 |
| 4 | 13.4 min | 13.1 |
| 6 | 13.4 min | 8.7 |

Binary-level LPT matches the ideal up to 3 shards. From 4 shards the floor is
`cdft_coupling_hene` at 13.4 min, so more shards gain nothing.

Crate-level sharding does not help. ferric-scf is 33.7 of the 52.3 min (64%),
so a crate-granularity split saturates at 33.7 min at any shard count:

| shards | crate-level makespan |
|---|---|
| 2 | 33.7 min |
| 3 | 33.7 min |
| 4 | 33.7 min |

The design therefore shards by binary (`--partition hash:i/4`), not by crate.

### With the five deferrals

The five slowest integration binaries -- `cdft_coupling_hene` (13.4 min),
`cdft_state_selection` (4.8), `rimp2_qqr3_screening` (4.1),
`cosx_group_screen_anchors` (2.1), `cosx_sparse_anchors` (1.9) -- total
26.2 min. Deferring them leaves 26.1 min over 4 shards, ~6.5 min each.

### Memory budget under nextest

nextest is process-per-test, and the workflow sets `FERRIC_MEM_BUDGET_GB=3`,
which every ferric process reads as its own budget. Under `cargo test` at most
one test binary ran at a time, so 3 GB meant 3 GB. Under nextest, N concurrent
processes each assume 3 GB; on a runner with ~14 GB free that is an N-times
overcommit, the same class as the ferric-batch defect. `.config/nextest.toml`
caps the pool-touching and BLAS-heavy tests at `max-threads = 2` (2 x 3 GB =
6 GB). The cap is a conservative choice, not a measurement; raise it only with
a measured peak RSS for the group.

Two coverage requirements:
- nextest does not run doctests. `cargo test --workspace --doc` stays as its
  own step, or doctest coverage is lost (the same failure mode as a SLOW entry
  with no matching nightly `--test` line).
- The SLOW list becomes a nextest filterset (`not (binary(a) + binary(b) +
  ...)`, 17 entries). That is a third copy of the list, after ci.yml's SLOW
  and the nightly `--test` lines, and needs the same sync check.

### Check: the 17-binary filterset (2026-09-19)

`cargo nextest list -E "not (binary(a) + binary(b) + ...)"` on a quiet box:

```
EXIT=0
996 tests selected across 214 binaries
all 17 deferred binaries: ABSENT
controls still present:  dft_gradient_rsh 4, linlccd_direct 8, lmp2_direct 10
```

A filterset that excluded everything would also show "all 17 absent"; the
control binaries show it does not over-filter. Count from the full output
captured to a file (2339 lines here): a `tail -8` of the same output shows
"7 tests from 1 binary".

### Unmeasured risk

`cargo nextest archive` produces a zstd tarball of ~190 binaries that ci.yml
records at 200-335 MB each uncompressed. They share most of their symbols so
zstd should compress it well, but the compressed size has not been measured
on a runner, and the runner has ~14 GB free and has already failed once with
"No space left on device" (run 33210162947). The build job prints `ls -lh` on
the archive so the first real run answers this.

## 6e. The MPI job has a ~7% launch-flake rate

Counted across the last 90 workflow runs, taking each run's `mpi` job
conclusion:

```
42 success
 3 failure     -> 3/45 completed = 6.7%
 4 cancelled   (excluded; superseded runs)
```

Every failure carries the same check-run annotation:

    expected 2 passing 'test result: ok.' summaries (one per rank),
    found 1 -- a rank failed or never started

(once `found 0`). This is an mpirun launch failure, not a test assertion. In
each instance (PRs #88, #92, #93, #95, #96) a plain re-run passed with no code
change; the #96 diff touched two Python files and no Rust.

1. **Triage.** At ~7% per run and 5+ pushes per PR, most PRs hit it at least
   once. Read the annotation with `gh api
   repos/:owner/:repo/check-runs/<id>/annotations`: the job log is not
   retrievable while its parent run is in progress, and `gh run rerun
   --failed` is refused while any job in the run is live. Read the
   annotation, wait for the run to finish, then re-run.
2. **Retry in the job.** `cargo-nextest`'s `retries = 1` absorbs an
   assertion-level flake, but this failure kills the launch, so the retry has
   to wrap the `mpirun` invocation in the `2-rank correctness check` step. The
   change is independent of the sharding work.
3. **Keep the summary count.** "expected 2 summaries, found 1" is what catches
   a dead rank. A retry around the launch keeps that assertion; relaxing it to
   ">= 1" would hide a real single-rank failure.

## 7. Getting per-binary times

`grep -oE 'finished in [0-9.]+s'` over `gh run view --log` gives per-binary
execution times on CI's own hardware, without a local `cargo test --no-run`
timing on a contended box.
