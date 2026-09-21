# CI: splitting build from test, and sharding the fast tier (2026-09-18)

All timings MEASURED from GitHub Actions job/step records. Nothing reproduced
locally — the box was contended for most of the session, and CI already has the
numbers. Companion: the caching research is in a separate note.

## 1. Where the wall clock goes (MEASURED, from CI)

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

Job total 70.9 min on this run. These are MAIN-branch figures: a PR run is
shorter (40.3 min fast tier, 51.4 min median job), but main is what approaches
the step's own 100-minute cap, so it is the number that matters for headroom.
ci.yml's comment claiming the deferrals brought the fast tier to "~23 min" was
true on 2026-09-16 and has drifted ~2.7x since. Second-longest job `mpi` at
8.6 min. `needs:` is
EMPTY throughout ci.yml, so all seven jobs already start together — **there is
no serialization to remove.** Any proposal claiming a win from "parallelising
existing jobs" is claiming a saving that does not exist.

CAVEAT on method: extracting step durations from an IN-FLIGHT run yields
garbage (a step with no `completedAt` gave `-1065422800.6 min`). Only use
completed runs.

## 2. The split point already exists in the script

This is the finding that reframes the work. The fast tier is NOT one monolithic
`cargo test`. Reading ci.yml:385-401, it is already **six sequential
invocations**:

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

## 3. But a naive per-crate matrix is badly unbalanced

Test-target counts (`ls crates/*/tests/*.rs`), MEASURED today:

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
critical path would be set by that shard, not by 1/N of the total. Target COUNT
is also a poor proxy for TIME — the SLOW list exists precisely because a few
binaries dominate — so shard balance should be derived from per-target
durations, which CI does not currently record per target.

## 4. RESOLVED: it is EXECUTION, not compilation

I originally called the compile-vs-run split "the load-bearing unknown" and
proposed timing `cargo test --no-run` to get it. That measurement was
unnecessary — **the CI log already contains it**, in every `finished in <N>s`
line.

Summing them on run 35377097468 (VERIFIED independently, twice):

```
$ gh run view 35377097468 --log | grep -oE 'finished in [0-9.]+s' \
    | awk '{s+=$3} END {printf "%.1f min across %d binaries\n", s/60, NR}'
52.3 min across 344 binaries
```

**52.3 min of the 63.2 min step is tests RUNNING; only ~11 min is compile and
link.** So no caching change can address more than ~11 min, and the sharding
question is entirely about how execution is scheduled, not about build reuse.

### The concentration is extreme (MEASURED, same run)

| rank | duration | share of 52.3 min |
|---|---|---|
| 1 | **801.8 s (13.4 min)** | **25.6%** |
| 2 | 285.1 s (4.8 min) | 9.1% |
| 3 | 243.1 s (4.1 min) | 7.7% |
| 4 | 124.6 s (2.1 min) | 4.0% |
| 5 | 116.4 s (1.9 min) | 3.7% |

The top binary is `cdft_coupling_hene` — 4 test functions, def2-SVP cDFT He-Ne
coupling scans, already running its 4 tests in parallel threads on a 4-core
runner, i.e. CPU-saturated.

**13.4 min is therefore a hard floor on the critical path for ANY scheme that
keeps that binary intact** — including nextest and including sharding. Sharding
a suite N ways cannot split one saturated binary.

Second unknown: **runner concurrency for this repo.** If only ~2 runners are
available, three shards queue rather than run concurrently and save nothing.
Not visible through `gh` without a billing query.

## 5. Build-then-test as separate JOBS: the artifact problem

The tempting shape is one `build` job doing `cargo test --no-run`, then N test
jobs consuming the binaries. The difficulty is moving compiled artifacts
between jobs:

- `target/` is deliberately NOT cached here (ci.yml:279) because it measured
  ~9 GB per PR merge ref against a 10 GB limit, evicting the libint2 cache that
  costs ~80 min to rebuild. Any artifact-passing scheme must not recreate that.
- Uploading just the test binaries is smaller than `target/`, but they are
  dynamically linked against OpenBLAS/libxc/libint2 and carry absolute paths;
  whether they run unmodified on a fresh runner is UNVERIFIED here.

`cargo-nextest` is **not used and not installed** in this repo (verified). Its
`--partition count:i/N` is the standard way to shard a Rust suite, and it
would also give per-test timing data, which is exactly what section 3 says is
missing for balancing. Evaluating it is a prerequisite, not a conclusion.

## 6. Ranked (revised after the execution-vs-compile measurement)

1. **`cargo-nextest`.** `cargo test` runs test BINARIES serially (each binary
   parallelises only its own tests) -- a documented cargo limitation,
   rust-lang/cargo#5609. ferric runs 344 binaries and most are tiny, so the
   short ones are executed one at a time with cores idle. nextest runs them
   concurrently. ESTIMATED 63 -> ~25 min; the 13.4 min FLOOR is measured.
   NOT free: nextest is process-per-test, so N concurrent processes each spawn
   a rayon pool and each read `FERRIC_MEM_BUDGET_GB` as a PER-PROCESS budget --
   the N-times-overcommit class already recorded in this repo's memory notes.
   Requires a deliberate `.config/nextest.toml` (`threads-required`,
   `test-groups`), which does not exist today. nextest also does NOT run
   doctests: `cargo test --doc` must be kept, or coverage is silently deleted.
2. **Split or defer `cdft_coupling_hene`.** 25.6% of all execution time in one
   binary, and it sets the floor for every parallel scheme above. SPLIT into
   separate binaries or DEFER to nightly -- do NOT shrink the physics. ci.yml's
   existing deferrals are explicit that slow tests are "MOVED, not shrunk",
   and shrinking an exactness anchor until it stops firing makes it inert.
3. **Only then consider sharding / build-test split.** `cargo nextest archive`
   is the real mechanism for moving built binaries between jobs (it handles the
   path/`CARGO_MANIFEST_DIR` remapping that makes hand-rolled `--no-run` +
   upload-artifact schemes fail). But it buys nothing over plain nextest until
   #2 is done, and the archive size is UNMEASURED against a runner ci.yml says
   has ~14 GB free and has already died with "No space left on device".
4. **Change nothing about caching.** Both "problems" in my original framing
   dissolved on inspection:
   - "Is not caching target/ still right?" YES, and for a stronger reason than
     the in-file note gives: a cache saved on a PR is scoped to the merge ref
     and can NEVER be restored by main or another PR, so it is write-only
     quota consumption that evicts the 80-min libint2 artifact.
   - "23 duplicate cache keys" -- REFUTED. They share a KEY but differ in
     SCOPE (20 x `refs/pull/N/merge`, 4 x main, 1 branch). Caches are keyed on
     (key, scope) and are immutable; a genuine same-scope re-save is skipped
     with a warning, never duplicated. Nothing to fix. Total usage 0.98 GB of
     10 GB -- healthy.
   - `Swatinem/rust-cache` would NOT help: its pruning explicitly does not
     cache workspace crates, which is exactly where ferric's multi-hundred-MB
     test binaries live.
   - `sccache` is disqualified by its own docs -- crates that invoke the system
     linker (every `bin`, i.e. every integration test binary here) cannot be
     cached.
5. **Keep the libint2 prefix cache as-is.** 42 MB, ~0 restore time, 0 misses in
   14 sampled runs, pinned version. Caching the installed artifact skips
   cmake+make entirely; a compiler cache would not.
6. **Do NOT** merge `fmt` into `pre-commit` or trim `mpi`'s preamble. Both are
   fully parallel against a 63-min critical path; the saving is exactly zero.

## 6b. Live evidence for #2, from an unrelated CI failure

While writing this, PR #88's `build-and-test` failed -- not on a test, but on
**Clippy**, with `function 'pairing_at' is never used` in
`crates/ferric-scf/tests/cdft_coupling_hene.rs`. That is the same binary
identified above as 25.6% of all execution time.

Three things this demonstrates, none of them hypothetical:

1. **The job ran 30 minutes and died at the lint step AFTER the tests.**
   Clippy is 1.8 min and Check is 2.9 min, both far cheaper than the 63-min
   test step, yet both are gated behind it in the same job. A one-line dead-code
   violation cost a full test cycle to discover. **Running clippy/check as their
   own job -- which needs no `needs:` and no sharding -- would surface this
   class of failure in ~3 min instead of ~30.** That is a smaller, safer change
   than anything in the ranked list above and it is worth doing FIRST.

2. **The dead code came from editing that same binary to make it cheaper.** The
   surrounding comment records the candidate loop being added to cut its cost,
   which superseded `pairing_at` with `pairing_at_capped` and orphaned the
   original. Recommendation #2 (split or defer this binary) touches exactly this
   file, so expect the same class of churn.

3. **It confirms the concentration is a maintenance problem, not only a
   scheduling one.** One 800-second binary is both the critical-path floor AND
   the file accumulating the most edits to reduce its own cost.

## 6c. RESULT: the lint split is merged-ready and MEASURED

PR #94 implements 6b's recommendation. First CI run on it
(run 35407496129, job 105800175202):

```
lint (check, clippy, rustdoc): PASS
  started  2026-09-18T23:55:04Z
  finished 2026-09-19T00:01:25Z
  => 6 min 21 s
```

Compare against what it replaces: the identical three commands, run inside
`build-and-test` AFTER the 63-min test tier. PR #88's clippy failure took
~30 min of runner time to surface. **6m21s vs ~30 min for the same signal**,
with no sharding, no new test runner, and no `needs:` edge.

Worth noting the cost side came in as predicted rather than better: the job
re-pays ~2 min of setup (disk-free + apt + cache restore), so 6m21s against a
~6 min sum of the three steps is almost exactly the setup overhead and nothing
more. libint2 restored from cache as expected.

This is now the CHEAPEST real win in this document and it is already
implemented. Everything above it (nextest, splitting cdft_coupling_hene,
sharding) remains unimplemented and still ranked as written.

## 6d. DESIGN: 4-way binary-level sharding (decided 2026-09-19)

Approved: 4 shards, cdft deferred, nextest with a config, archive-based
build/test split. Design is measured, not estimated.

### Why 4, and where the curve flattens

Per-binary times from CI run 35377097468 (324 binaries paired to their
`finished in Ns` lines), LPT-partitioned:

| shards | makespan | ideal |
|---|---|---|
| 2 | 26.1 min | 26.1 |
| 3 | 17.4 min | 17.4 |
| 4 | **13.4 min** | 13.1 |
| 6 | 13.4 min | 8.7 |

Binary-level LPT is near-PERFECT to 3 shards and then hits a wall: the floor is
`cdft_coupling_hene` at 13.4 min on its own. Past 4 shards you buy nothing.

**Crate-level sharding does not work at all here.** ferric-scf is 33.7 of the
52.3 min (64%), so a crate-granularity split saturates at 33.7 min at ANY shard
count:

| shards | crate-level makespan |
|---|---|
| 2 | 33.7 min |
| 3 | 33.7 min |
| 4 | 33.7 min |

That is why the design shards by BINARY (`--partition hash:i/4`), not by crate.

### With the five deferrals

The five slowest integration binaries -- `cdft_coupling_hene` (13.4 min),
`cdft_state_selection` (4.8), `rimp2_qqr3_screening` (4.1),
`cosx_group_screen_anchors` (2.1), `cosx_sparse_anchors` (1.9) -- total
26.2 min. Deferring them leaves **26.1 min over 4 shards ~= 6.5 min each**.

### The nextest hazard this design has to manage

nextest is PROCESS-PER-TEST and the workflow sets `FERRIC_MEM_BUDGET_GB=3`,
which every ferric process reads as ITS OWN budget. Under `cargo test` at most
one test binary ran at a time, so 3 GB meant 3 GB; under nextest, N concurrent
processes each believe they may use 3 GB. On a runner with ~14 GB free that is
an N-times overcommit -- the same class as the ferric-batch defect. The
`.config/nextest.toml` caps the pool-touching and BLAS-heavy tests at
`max-threads = 2` (2 x 3 GB = 6 GB, leaving headroom). That number is a
DELIBERATE CONSERVATIVE CHOICE, not a measurement; raise it only with a
measured peak RSS for the group.

Two more things that would silently delete coverage if missed:
- **nextest does not run doctests.** `cargo test --workspace --doc` must stay
  as its own step or doctest coverage vanishes -- the same failure mode as a
  SLOW entry with no matching nightly `--test` line.
- **The SLOW list becomes a nextest filterset** (`not (binary(a) + binary(b)
  + ...)`, 17 entries). That is a THIRD copy of the list, after ci.yml's SLOW
  and the nightly `--test` lines. It needs the same sync check.

### VERIFIED: the 17-binary filterset (2026-09-19)

`cargo nextest list -E "not (binary(a) + binary(b) + ...)"` on a quiet box:

```
EXIT=0
996 tests selected across 214 binaries
all 17 deferred binaries: ABSENT
controls still present:  dft_gradient_rsh 4, linlccd_direct 8, lmp2_direct 10
```

Both directions checked, and the second one is the point: a filterset that
excluded EVERYTHING would also show "all 17 absent". The control binaries prove
it is not over-filtering.

METHOD NOTE, because I got this wrong first: my initial check read a `tail -8`
of the output and concluded "7 tests from 1 binary -- catastrophically
over-filtering". That was the TAIL, not the listing. Re-running with the full
output redirected to a file gave 2339 lines and the numbers above. A truncated
view of a result is not the result -- capture to a file before counting.

### Unmeasured risk, stated

`cargo nextest archive` produces a zstd tarball of ~190 binaries that ci.yml
records at 200-335 MB each uncompressed. They share most of their symbols so
zstd should collapse it hard, but **the compressed size has not been measured
on a runner**, and the runner has ~14 GB free and has already died once with
"No space left on device" (run 33210162947). The build job prints `ls -lh` on
the archive so the first real run answers this.

## 6e. The MPI job has a MEASURED ~7% launch-flake rate

Counted across the last 90 workflow runs, taking each run's `mpi` job
conclusion:

```
42 success
 3 failure     -> 3/45 completed = 6.7%
 4 cancelled   (excluded; superseded runs)
```

Every failure carries the SAME check-run annotation:

    expected 2 passing 'test result: ok.' summaries (one per rank),
    found 1 -- a rank failed or never started

(once `found 0`). It is an mpirun LAUNCH failure, not a test assertion: in
every instance this session -- PRs #88, #92, #93, #95, #96 -- a plain re-run
passed with no code change. On #96 the diff was TWO PYTHON FILES and zero Rust,
which alone rules out the code.

**Consequences worth acting on:**

1. **Triage cost.** At ~7% per run and 5+ pushes per PR, most PRs will hit it
   at least once. The annotation is the fastest discriminator -- `gh api
   repos/:owner/:repo/check-runs/<id>/annotations` -- because the job log is
   NOT retrievable while its parent run is still in progress, and
   `gh run rerun --failed` is REFUSED while any job in the run is live. So the
   loop is: read the annotation, wait for the run to finish, then re-run.
2. **The retry belongs in the job, not in the human.** `cargo-nextest`'s
   `retries = 1` (already in the proposed `.config/nextest.toml`) would absorb
   an assertion-level flake, but this one kills the LAUNCH, so a retry has to
   wrap the `mpirun` invocation itself. That is a small, self-contained change
   to the `2-rank correctness check` step and is worth doing independently of
   the sharding work.
3. **Do NOT silence it by loosening the summary count.** The assertion
   "expected 2 summaries, found 1" is exactly what catches a genuinely dead
   rank. A retry around the launch keeps that assertion intact; relaxing it to
   ">= 1" would make a real single-rank failure invisible.

## 7. Method note, for the next person

My first pass proposed running `cargo test --no-run` locally to get the
compile/run split. That was wrong twice over: the box was contended (so the
timing would have been noise), and **the number was already in the CI log**.
`grep -oE 'finished in [0-9.]+s'` over `gh run view --log` gives per-binary
execution times for free, on the exact hardware CI uses. Mine it before
measuring anything locally.
