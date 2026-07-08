# Launching the two SR-MP2 + LR-RPA sweeps

Two independent sweeps. **They share one 23 GB box — read the memory rule before
running them together.**

## Memory rule (learned the hard way, 2026-07-08)

The box is 23 GB. A benzene aTZ job peaks ~6.4 GB (post-T12). A `cargo`/`rustc`/
`rust-lld` build is multi-GB. **3 benzene jobs + a concurrent build = OOM cascade**
(it killed a job, the driver, and tmux). So:

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

    cd /home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa
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

    cd /home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa
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
