# Sodium clusters (Na4/Na6/K2) — root cause + fix

## Symptom
Na4 (aDZ) and Na4/Na6 (aTZ) hung the GW100 sweep: 4+ CPU-HOURS each, no output.
Worked around with a per-molecule stall-watchdog (commit d4de76f) that marks them
FAILED and skips. This doc is the actual FIX.

## Diagnosis (measured, not assumed)

NOT divergence — the SCF converges, and the QP Newton loop (sigma.rs `for _ in
0..30`) and evGW iterations (`0..max_ev_iter`) are all BOUNDED. It's raw compute
on a big system:

- **Na4 dimensions:** nbf=108, nocc=22, nvir=86, **nov=1892** — vs CO's nbf=28,
  nov=130 (≈14× the occ-vir space).
- **Oversized aux:** Na has NO aug-cc-pVDZ RI-fit aux upstream, so it's paired
  with cross-family **def2-TZVP-RIFIT** (~29 shells/Na) — a larger naux (≈M, the
  PDEP mode count) than a same-family aDZ aux would be.
- **Cost structure (from FERRIC_TIMING):** the freq_quad stage
  (`eval_inv_dielectric_matrices`) does **K full M×M inversions**, O(K·M³).
  `project_b_into_pdep` is O(naux·nov·M). Both scale steeply in M=naux and nov.
- **×22 PDEP solves per molecule:** the sweep runs G0W0+COHSEX+evGW0+evGW(≤8) per
  reference × 2 (@HF and @PBE) ≈ 22 PDEP builds. On CO each is ~0.15s (trivial);
  on Na4 each is large, so 22× a large number = hours.

## The fix (by leverage)

### 1. PDEP truncation @1e-4 — the big lever (already built, = Task #7)
freq_quad is O(K·M³) in the PDEP mode count M. The sweep runs `trunc_thresh=0`
(full rank, for apples-to-apples). Truncating at the 1e-4 default drops the
weakly-screening modes (~75% on large systems, per the trust-map):
- keep 25% of M → inversions cost 0.25³ = **~64× cheaper**
- keep 50% of M → **8× cheaper**
This is ~lossless (the trust-map proves IP/α/C6 unchanged at 1e-4). **Likely turns
Na4 from hours into minutes.** Test: `pdep_trunc_trustmap na4 aug-cc-pvdz` (Na4
geometry added to the driver) — measure IP-vs-thresh0 AND wall-time-vs-thresh.

### 2. Share W₀ across the W₀-methods — ~3× fewer PDEP builds
G0W0, COHSEX, evGW0 all use the SAME W₀ (one PDEP). Currently rebuilt per method.
On small molecules this saves <1% (measured, db47859); on Na4 where each PDEP is
the bottleneck, collapsing ~22→~3 builds is a real ~3× win. Driver refactor.

### 3. Same-family / right-sized aux
The cross-family def2-TZVP-RIFIT aux is larger than aDZ needs. A right-sized aux
(or running these in the def2 orbital basis to match) shrinks naux=M directly.
Basis-curation chore for a few molecules.

### 4. Laplace / Boys PDEP backends (already in ferric, off by default)
`Chi0Backend::Laplace` / `Chi0Sparsity::BoysScreened` are built for large systems
where Dense PDEP blows up (boys-screening-crossover memory: helps ≳10 atoms). Na4
is a candidate; needs benchmarking vs truncation.

## Recommendation
Fix #1 (truncation) is highest-leverage and already the queued Task #7 — it's the
fix, not just a characterization. Run it on Na4 to confirm hours→minutes at
unchanged IP, then the sweep can run the alkali clusters WITH truncation instead
of marking them FAILED. Combine with #2 for production GW100.
