# Launching the two SR-MP2 + LR-RPA sweeps

## TL;DR — copy/paste relaunch (both are idempotent; safe to re-run anytime)

```bash
cd "$(git rev-parse --show-toplevel)"

# 1. terfc/terf small-case r0 sweep (serial, ~0.1-1.5 GB/job, coexists with anything)
TERFC_CONC=1 TERFC_PER_JOB_GB=2 \
  setsid nohup python3 benchmarks/omega_diag/terfc_sweep.py \
  >> benchmarks/omega_diag/terfc_sweep.log 2>&1 & disown

# 2. benzene aTZ erf/erfc crossing (CONC=1 while GW100 holds ~8 GB; bump to 2-3 if the box is clear)
BZ_CONC=1 BZ_PER_JOB_GB=7 BZ_YIELD_TO= \
  setsid nohup python3 benchmarks/omega_diag/finish_benzene_atz_par.py \
  >> benchmarks/omega_diag/finish_benzene.log 2>&1 & disown
```

Both skip already-finished jobs (marker-checked), so relaunching after a death
resumes exactly where it stopped — nothing is recomputed. `BZ_YIELD_TO=` (empty)
is REQUIRED (see the pgrep gotcha below). The terfc driver auto-resolves
FERRIC_TERF_TABLE_DIR. Pick `BZ_CONC` by free RAM: 1 job ≈ 6.4 GB, and GW100 (if
running) holds ~8 GB — check `free -g` first. `BZ_CONC=3` ONLY if nothing else
(GW, a build) is resident.

**Check what's alive** (never trust `pgrep -f`):
```bash
pgrep -x ferric-cli | while read p; do tr '\0' ' ' </proc/$p/cmdline; echo " [$p]"; done
tail -f benchmarks/omega_diag/terfc_sweep.log        # or finish_benzene.log
free -g                                              # watch avail vs GW100's ~8 GB
```

---



Two independent sweeps. **They share one 23 GB box — read the memory rule before
running them together.**

## Memory rule (learned the hard way, 2026-07-08)

The box is 23 GB. A benzene aTZ job peaks ~6.4 GB (post-T12) — **this includes the
CP ghost-monomers (cpA/cpB), which are NOT cheap**: they climb from ~2 GB at
startup to 6.4 GB at the RPA eigensolve, same as the dimer. Do NOT size
concurrency off startup RSS — measure the PEAK (see 2026-07-10 note below). A
`cargo`/`rustc`/`rust-lld` build is multi-GB. **3 benzene jobs + a concurrent
build = OOM cascade** (it killed a job, the driver, and tmux). So:

- **BZ_CONC=1 is the standing safe default for aTZ.** MEASURED 2026-07-10: a single
  benzene aTZ RS-MP2-RPA job — INCLUDING the CP ghost-monomers (cpA/cpB) — peaks at
  **~10.6 GB, and does so TWICE** (two eigensolve stages), with a ~1 GB dip between.
  On a 23 GB box with ~4 GB of desktop/claude processes resident, that leaves room
  for exactly ONE job. CONC=2 (2 × 10.6 = 21 GB + 4 GB other = 25 GB) OOMs. CONC=6
  OOM-killed tmux. **Do not exceed CONC=1 unless the box has >21 GB genuinely free
  AND you have re-measured the peak.**
- Startup RSS (~2 GB) is a LIE — the peak is 5× that at the eigensolve. Always
  measure peak, not startup, when sizing concurrency.
- Before any launch, check `free -m` swap: if swap > ~500 MB (residue of a prior
  crash), a fresh 10 GB peak will thrash against it and run slow. Let it drain or
  accept the slowdown, but never stack a second peak on a full swap.
- **3 benzene jobs XOR a build, never both.**
- If code must be rebuilt (e.g. to land the terf arm), first **stop the benzene
  driver** (or run it at `BZ_CONC=2`), build, then relaunch.
- The terfc small-case sweep is `CONC=1`, <1.5 GB/job — it can coexist with a
  build, but NOT with 3 benzene jobs. Pair it with benzene at `BZ_CONC=2` if you
  must run both, or just run them one at a time (they're both idempotent).

Always verify real processes with `pgrep -x ferric-cli` or a `/proc/*/cmdline`
scan — `pgrep -f <pattern>` phantom-matches your own diagnostic shells.

---

## Sweep 1 — benzene aTZ erf/erfc crossing (READY, this is the production grid)

Driver: `finish_benzene_atz_par.py` (concurrent) or `finish_benzene_atz.py` (serial).
27 remaining jobs: B ω∈{0.30,0.42,0.55,0.673,0.80}, T ω∈{0.20,0.30,0.42}, RHF×3
frags, on the S22 #11 benzene parallel-displaced dimer + CP monomers, aug-cc-pVTZ,
full-rank (trunc_thresh=0). Idempotent/additive over `derisk/out/`.

Launch (detached, 2-concurrent = safe alongside a build; 3 only if nothing else
is building):

    cd "$(git rev-parse --show-toplevel)"
    BZ_CONC=2 BZ_PER_JOB_GB=7 BZ_YIELD_TO= \
      setsid nohup python3 benchmarks/omega_diag/finish_benzene_atz_par.py \
      >> benchmarks/omega_diag/finish_benzene.log 2>&1 & disown

`BZ_YIELD_TO=` (empty) is REQUIRED — the default yields to `run_aconf_cli`, and the
pgrep -f match false-fires on diagnostic shells, stalling the driver forever.

Monitor:  `tail -f benchmarks/omega_diag/finish_benzene.log`
On completion it auto-runs `derisk_atz_cp.py` → `DERISK_ATZ_CP.md` (the crossings).

---

## Sweep 2 — terfc/terf r0 sweep on SMALL cases (BLOCKED on task #20)

Driver: `terfc_sweep.py`. Sweeps the *tempered* attenuator on **r0** (Bohr) for
B and T, plus a matched erf/erfc arm at the derived ω (=1/(r0·√2)) for shape
comparison. Systems: water, water dimer, ethene dimer / cc-pVDZ (each <1.5 GB).

**BLOCKER:** the terf arm is not in the binary yet. The CLI `[mp2]` section must
accept `attenuator = "terf"` and `r0 = <Bohr>` and thread it into `RsMp2RpaConfig`
(`rs_mp2_rpa.rs`). Good news confirmed 2026-07-08: the 3-index Terfc RI path
ALREADY works (`engine.rs` new_3center/new_2center dispatch `OperatorKind::Terfc`
to the table engine; `compute_eri3` handles it) — so this is a config-wiring +
LR-realization change, NOT new integral code. terf = Coulomb − terfc (terf+terfc=1),
same exact limits as erf/erfc. Tables: set `FERRIC_TERF_TABLE_DIR` if not default.

Preflight FIRST (proves the binary understands the stanza; the driver also runs
this and ABORTS if it doesn't — it will NOT silently record erf-as-terf):

    cd "$(git rev-parse --show-toplevel)"
    TERFC_PREFLIGHT_ONLY=1 python3 benchmarks/omega_diag/terfc_sweep.py

When preflight passes, launch (serial, memory-safe):

    TERFC_CONC=1 \
      setsid nohup python3 benchmarks/omega_diag/terfc_sweep.py \
      >> benchmarks/omega_diag/terfc_sweep.log 2>&1 & disown

Monitor:  `tail -f benchmarks/omega_diag/terfc_sweep.log`
Tunables: `TERFC_R0_LIST` (comma Bohr), `TERFC_SYSTEMS` (water,water_dimer,
ethene_dimer), `TERFC_TIMEOUT` (7200s).

r0 grid default: 0.30, 1.00, 2.00, 3.18, 5.00, 12.0 Bohr. r0=3.18 → ω=0.2224
Bohr⁻¹ = 0.42 Å⁻¹ (the erf arm's operating point — the direct comparison row).
Extremes (0.30, 12.0) verify the limits: r0→0 ⇒ MP2+ΔdRPA[Coulomb]; r0→∞ ⇒ plain
MP2.
