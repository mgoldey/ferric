# Unattended quiet-window run of the remaining #228 plan

`overnight_228.py` runs the rest of the #228 timing plan from ONE fixed binary, interleaving ferric and
PySCF back to back, and keeps only samples taken in a quiet window. `analyze_228.py` turns the results
into medians, spreads and ratios with bootstrap CIs. Nothing here has been timed yet: the only run so far
is the `--smoke` toy cell, and its timings are invalid by construction.

## Launch (one command)

```
reference/pbc/bench/overnight_228.sh --build ~/qc/pbc228-runs/night1
```

This refreshes `~/qc/ferric-mainline` to origin/main and builds `ferric-cli` and `ferric-python` there in
release mode. It then starts the driver detached (`nohup setsid`), logging to `night1/driver.log`. The
driver's defaults are tiers `anchor,core,pbe-gated`, `--hours 10`, PySCF `cell.precision 1e-13` and
`--match-ferric-conv`. Extra arguments go to the driver, e.g. `--tiers anchor,core,pbe-gated,n1-large`.

`--build` is required today: the mainline build is from 2026-09-24. It predates #508 and has no
`--version`, so the driver would refuse it.

**Resume** after a kill, a reboot, the deadline (exit 3) or exhausted keys (exit 4) with the same
command WITHOUT `--build`, plus the same extra arguments if any:

```
reference/pbc/bench/overnight_228.sh ~/qc/pbc228-runs/night1
```

The launcher refuses `--build` once `night1/manifest.json` exists, or while a driver holds
`night1/lock`. Rebuilding the shared `~/qc/ferric-mainline` is never part of a resume: the main venv's
`ferric` `.so` points at it. On resume, the driver re-hashes both `night1/bin` files and re-checks the
python environment against the manifest; it refuses on any mismatch. Finished keys are skipped.

Exit codes: 0 = every key has a gated sample; 3 = deadline; 4 = some keys used all `--max-attempts`
without a gated sample. Those keys are listed in `driver.log`, and "plan complete" is NOT printed.

To check the setup first without running any SCF:
`python reference/pbc/bench/overnight_228.py --run-dir X --dry-run`. To prove the pipeline end to end, add
`--smoke`: it runs H2/STO-3G once, with the gate not enforced.

Afterwards:
`~/qc/ferric/.venv/bin/python reference/pbc/bench/analyze_228.py ~/qc/pbc228-runs/night1/results.jsonl`.
For between-session scatter, pass several run dirs. Each night needs its own run dir, all on the same
binary: pass `--ferric-bin night1/bin/ferric --ferric-so night1/bin/libferric.so`.

## Fixed binary

On the first launch, `ferric` and `libferric.so` are copied into `RUN_DIR/bin`. The copies are private
inodes, so a rebuild cannot SIGBUS the run. `RUN_DIR/manifest.json` records:
- `ferric --version`;
- the extension's `__build__`, loaded by file path in `python -I`;
- both sha256s;
- whether the commit contains #508 (fc0ccc041);
- the PySCF side's python path and its python, pyscf, numpy and scipy versions.

The driver refuses to start unless all of these hold:
- both files are clean;
- the extension HAS a `__build__` with a commit (a missing or unknown one is refused, never skipped);
- they name the same commit;
- that commit contains #508.

`--allow-unverified` overrides this, but its rows are then not quotable. Every row carries the same
`bin_sha256`. Every ferric run re-hashes the binary, through `scatter_driver.check_binary_unchanged`.

## The quiet-window gate (read before AND after every sample, logged on the row)

One reading takes 5 s of `/proc/stat` and `/proc/<pid>/stat` (never `pgrep`). CPU fractions are
measured against the 6 PHYSICAL cores, not the 12 logical ones: the SMT siblings of the pinned cores
0-5 are outside the taskset. A reading fails if any of these holds:

| check | fails when |
|---|---|
| 1-min loadavg (before / recheck ONLY) | `load1 >= 1.5`, read from `/proc/loadavg` as is: the pre-registered gate, and the default. After an N=6 sample, the next sample waits for the driver's own load tail to decay; that wait is about 90 s after a long run and is polled every 10 s. `--load-mode adjusted` gates on `load1 - own_load` instead. `own_load` models the driver's own recent samples under the kernel's 60 s exponential average. Both values are logged. |
| box CPU busy over the window | `>=` the busy bar: the measured idle floor + 0.02 of the physical cores (see below) |
| iowait / steal over the window | `>= 2%` / `>= 1%` (`--no-steal-iowait` disables) |
| compiler alive | any `rustc`, `cc1plus`, `cc1`, `collect2`, `ld`, `rust-lld` |
| build driver busy | `cargo`, `ninja` or `sccache` above 1% of a core. A blocked cargo shim is idle and does not count. |
| other processes | any `python*`, `pytest`, `uv` or `uvx` above 10% of a core; any other process above 25% |

### The idle floor (bars are floor + margin, not an absolute 0.02)

Night 1 (2026-10-28/29) showed that an absolute 0.02 bar cannot be met on this box. With only the
desktop running (Zoom, conky, several Claude sessions, a node worker), the box reads 0.035-0.045
of the physical cores busy. The gate waited from 00:14 to 02:27 and took no sample. The unit tests
had mocked a near-idle box, so none of them could see it.

So, at every launch and before any sample, the driver measures the box's **idle floor**:
- `--floor-samples 6` back-to-back gate readings (5 s each), while nothing of ours runs.
- Readings with a heavy process (a compile, a python sweep) are left out, since they are not the
  idle desktop.
- `busy_floor` is the MEDIAN `busy_phys` of the clean readings: what one window is judged on.
- `foreign_floor` is their MEAN: the run-window foreign CPU is an average over the run, and with the
  driver idle, busy and foreign are the same quantity.

The bars are then `busy_floor + --busy-margin` and `foreign_floor + --foreign-margin`. Both margins
default to 0.02: the pre-registered quiet bar, now read as a margin above the floor. An explicit
`--max-busy` / `--max-foreign` is still an ABSOLUTE bar and wins over the floor.

The driver **refuses to start** (smoke runs only warn) if:
- the busy or foreign floor is above `--max-floor 0.10`, meaning the box is not idle; or
- fewer than half the floor readings were free of heavy processes.

A bar built on a loaded box would quietly accept loaded samples, so the driver stops instead. The
dry run measures the floor too and prints `WOULD REFUSE` instead of stopping.

It **never waits silently.** If the before gate has failed for `--unreachable-after 30` minutes on
busy readings alone, at idle-desktop level (`<= --max-floor`, no heavy process, load fine), it logs
one `[gate] WARNING: the busy bar ... may be UNREACHABLE` line per wait. That line gives the bar, its
mode, the floor, and the min/median/max of the failing readings. It then keeps waiting until
`--hours`.

**Recorded:**
- each launch's floor (every reading) is appended to `manifest.json` under `idle_floors`;
- every row carries `idle_floor`: the floors, the bars it was judged by, and each bar's mode
  (`floor+margin` or `absolute`);
- every gate reading carries `busy_bar`, and the after reading carries `foreign_bar`.

The floor is re-measured per launch, so the bars can differ between resumed launches. Every row also
keeps `foreign_cpu_frac`, so any fixed bar (for example the old absolute 0.02) can be re-applied
after the fact.

The order around the heavy-slot locks (`/tmp/ferric-compile*.lock`), in `scatter_driver.run_cmd`:
1. **before**: poll every 60 s until a reading is quiet, holding NO lock. A busy box therefore never
   blocks anyone else's build. If `--hours` runs out first, the driver exits 3.
2. Take both locks, then **recheck**: one 5 s reading. If it fails (e.g. a build started in between),
   release the locks and go back to step 1.
3. Run the sample, then **after**: one reading with NO loadavg check, since load1 still holds the
   sample's own ~N(1 − e^{−wall/60}). Judging it would discard every N=6 sample longer than ~17 s and
   keep the fast ones, a selection bias. The after phase judges only FOREIGN activity:
   - busy CPU, processes, iowait and steal in the post-run window (the driver is idle then);
   - foreign CPU over the run (scatter_driver's accounting, per physical core) below the foreign
     bar: the measured idle floor + the pre-registered 0.02 margin;
   - steal over the run. Run-window iowait is logged but not judged, because PySCF writes its own
     cderi file.
4. Release the locks.

A sample is **gated** only if both the recheck and the after reading pass. A failed sample's row is still
written, with `valid = false`, and the key is re-run, up to `--max-attempts 3`. Only gated, converged,
exit-0 rows enter any statistic.

## Resumability, memory and order

- **Resumability:** `RUN_DIR/results.jsonl` is append-only, one row per sample. Rows are keyed by
  (plan id, code, variant, threads, rep). If a kill leaves a partial last line, the next append moves it
  to `results.jsonl.truncated` before writing. Readers skip a partial LAST line with a warning, and
  refuse a bad line anywhere else. A second launch on the same dir exits at once, through a flock
  on `RUN_DIR/lock`. `run_id` = `session` = the dir name plus a hash of its absolute path (two `night1` dirs never pair),
  so reps pair across resumed launches of one dir. The
  `invocation` field tells the launches apart.
- **Memory:** every process runs under `scripts/ferric-limited --max=14G --high=14G`. MemoryHigh is set
  equal to MemoryMax, so a runaway is killed rather than throttled, which would silently distort its
  timing. The largest earlier PySCF row was 6 GB, at precision 1e-11.
- **Order:** samples run rep-major (rep 0 of every cell, then rep 1, and so on), so each cell's samples
  are spread over the night. Inside a rep, the order is [ferric variants..., PySCF], rotated by rep.

## Plan and budget (ESTIMATE)

Per-sample walls come from the session tables in README.md:
- N=6 is measured. N=1 is measured where a table has it, otherwise taken as 5 × N=6.
- dryice_112 ferric is taken as 314 s / 2, after #508.
- PySCF is scaled by an ASSUMED 1.3× for precision 1e-13.
- k333, dryice PBE and the dryice_112 omega variants have never been timed, so their numbers are guesses.

Each sample also gets 15 s of gate overhead, plus the raw-load wait for its own load tail to decay.
Waits for OTHER load and retries are NOT included. The dry run prints this table.

| plan id | N | ferric inputs | samples | est. min |
|---|---|---|---|---|
| dryice_pbe_ladder (anchors, untimed, ungated) | 6 | 7 grids, 75x302 … 250x590 | 14 | 24 |
| diamond_prim_rhf | 6 / 1 | default + ω 0.3/0.5/0.7 / default | 25 / 6 | 12 / 7 |
| dryice_rhf | 6 / 1 | default + ω 0.3/0.5/0.7 / default + ω 0.5 | 25 / 9 | 21 / 18 |
| diamond_prim_k222_rhf | 6 / 1 | default | 10 / 6 | 9 / 10 |
| diamond_prim_k333_rhf | 6 | default | 10 | 59 |
| diamond_conv_rhf | 6 | default | 10 | 19 |
| diamond_prim222_rhf | 6 | default | 10 | 43 |
| dryice_112_rhf | 6 | ω 0.3/0.5/0.7/1.0 | 25 | 111 |
| diamond_prim_pbe | 6 / 1 | ω 0.3/0.5/0.7/1.0 / default | 25 / 6 | 22 / 11 |
| dryice_pbe (only if the ladder gate passes) | 6 / 1 | 75x302 | 10 / 6 | 19 / 25 |
| **default tiers total** | | | | **~6.8 h** |
| n1-large (opt-in): diamond_conv, diamond_prim222, dryice_112 | 1 | default | 6 each | 24 / 80 / 87 = **~3.2 h** |

Reps are 5 at N=6 and 3 at N=1, the pre-registered counts. PySCF uses its automatic ω and is labelled
as such.

## Choices made here that Matt has not ruled on (decisions.txt item 12)

- **Precision:** PySCF runs at 1e-13, as the issue comment asks. The session-4 rule was "the loosest
  value that passes the energy gate", which gave 1e-11. Earlier rows are therefore not
  precision-matched to these.
- **Excluded inputs:** k444, MgO/LiF k-mesh and the jkfit rows are not in the plan. No inputs exist for
  them, or they are out of #228.
- **Omega sweeps:** these cells get an ω sweep: diamond_prim, dryice, dryice_112 and diamond_prim PBE.
- **N=1 large cells:** opt-in, with 3 reps.
- **Spread flag:** `analyze_228.py` flags `SPREAD>2%` on range/median over gated samples. That is
  stricter than summarize.py's NOISY (IQR/median > 10%) and H12 (≤ 3%). It shows `n<3` when fewer than 3
  samples were gated.
- **Metrics:** spreads and ratios are reported for `wall`, `t_setup_s`, `t_scf_s` and `s_per_iter`, and
  each 2% flag is computed on the metric of its own block. `wall` contains the `systemd-run --scope`
  start (~0.1-0.3 s, close to the 2% bar on ~15 s cells) and PySCF's ~1.5 s import, so on short cells read
  `t_setup_s` / `t_scf_s` first. `s_per_iter` has different definitions in the two codes (README.md).
- **Quotability:** `summarize.py --quiet-only` still applies its pre-registered QUOTABLE rule to the same
  rows. `box_state` comes from scatter_driver's own ABSOLUTE 0.02 foreign bar, not the idle-floor
  bar. A gated row on this desktop can therefore carry `box_state = ambient`, and that alone does not
  make it unquotable. `box_state` is also read a few milliseconds after the gate, so a loadavg tick can
  split the two. Check before quoting.
- **Deadline:** with `--hours 10` the default tiers fit one night with about 3 h of slack for waits on
  other load. n1-large needs another ~3.2 h: run it as a second night in a second run dir on the same
  binary. That second night also gives the between-session scatter.
