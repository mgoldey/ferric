# THC lane, ported onto main — measurement record (2026-10)

Branch `feat/thc-port` (worktree `~/qc/thc-port`), ported by hand from the
pre-rewrite `feat/thc` (`~/qc/ferric-thc`, 116e565df). Everything is behind
the `thc` cargo feature of `ferric-scf` / `ferric-rpa` / `ferric-benchmarks`.
The pre-registration (§3) was written before the timing run; the tables come
first only for readability.

## 1. Raw measurements — THC-K vs DF-K

**Run:** 2026-10-24 00:50:06 to 01:45:33 EDT. **Commit:** b98d17705
(`target/release/examples/thc_k_vs_dfk_timing`). **Box:** i7-6800K (6
physical cores). Both heavy-op slots (`/tmp/ferric-compile{,.2}.lock`) were
held for the whole run, so no other cargo or PySCF jobs ran. Load average was
12.0 at the start (the trailing average of earlier jobs) and 1.6-2.5 by the
end. **Settings, identical for both builders:**
- `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`.
- f64, run under `scripts/ferric-limited` (MemoryHigh 10G, MemoryMax 12G).
- obs cc-pVDZ, aux cc-pVDZ-RI. This is the smaller (RI) aux, which favours
  THC because `N_IP` scales with naux.
- Both builders use the same lindep-filtered `V^{-1/2}`.

**What was timed:** `build_from_occ` (DF-K's production occupied-MO path),
one warm-up and then the median of 3. THC used `c_isdf = 8.5` (the Lee et al.
HF-K floor) on a (30,50) Becke parent grid, with `rank_tol 1e-18`.

### 1a. Full mode: real setup, accuracy vs DF-K on the same aux

`C_occ` comes from a DF-RHF converged to energy 1e-6 / density 1e-5.
`ΔE_x = −tr(C Cᵀ ΔK)` is the RHF exchange-energy error.

| system | nbf | naux | nocc | N_IP | c_eff | DF-K setup s | DF-K build s | THC setup s | THC build s | build THC/DF-K | rel max\|ΔK\| | ΔE_x mHa | VmHWM GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alkane_1 | 34 | 112 | 5 | 564 | 5.04 | 0.030 | 0.0003 | 1.00 | 0.0012 | 3.88 | 2.80e-6 | −0.019 | 0.12 |
| alkane_2 | 58 | 196 | 9 | 1507 | 7.69 | 0.060 | 0.0013 | 9.71 | 0.0146 | 11.48 | 1.79e-6 | +0.020 | 0.23 |
| alkane_3 | 82 | 280 | 13 | 2380 | 8.50 | 0.193 | 0.0043 | 32.01 | 0.0617 | 14.34 | 7.55e-6 | −0.008 | 0.40 |
| alkane_4 | 106 | 364 | 17 | 3094 | 8.50 | 0.357 | 0.0120 | 69.03 | 0.1142 | 9.48 | 8.86e-6 | −0.042 | 0.58 |
| alkane_6 | 154 | 532 | 25 | 4522 | 8.50 | 1.275 | 0.0437 | 208.87 | 0.2864 | 6.55 | 1.62e-5 | +0.028 | 1.17 |
| alkane_8 | 202 | 700 | 33 | 5950 | 8.50 | 3.012 | 0.1159 | 473.69 | 0.5767 | 4.98 | 1.07e-5 | +0.040 | 2.02 |
| alkane_10 | 250 | 868 | 41 | 7378 | 8.50 | 6.200 | 0.2570 | 888.56 | 1.0088 | 3.93 | 2.43e-5 | −0.203 | 3.14 |

alkane_1 and alkane_2 are rank-clamped (`c_eff < 8.5`: the product space or
the parent grid ran out first).

### 1b. Build mode: per-build cost at `N_IP = ceil(8.5·naux)`

These rows use synthetic factors (`ThcK::from_parts`) and a synthetic
`C_occ`, so they measure cost only and carry no accuracy.

| system | nbf | naux | nocc | N_IP | DF-K setup s | DF-K build s | THC build s | build THC/DF-K | VmHWM GB |
|---|---|---|---|---|---|---|---|---|---|
| alkane_4 | 106 | 364 | 17 | 3094 | 0.341 | 0.0117 | 0.1151 | 9.80 | 3.14* |
| alkane_8 | 202 | 700 | 33 | 5950 | 3.031 | 0.1173 | 0.5718 | 4.88 | 3.14* |
| alkane_12 | 298 | 1036 | 49 | 8806 | 14.18 | 0.4567 | 1.6207 | 3.55 | 3.14* |
| alkane_16 | 394 | 1372 | 65 | 11662 | 37.49 | 1.3378 | 3.4545 | 2.58 | 3.41 |
| alkane_20 | 490 | 1708 | 81 | 14518 | 96.39 | 3.0519 | 6.2182 | 2.04 | 6.52 |
| alkane_24 | 586 | 2044† | 97 | — | — | — | — | **CUT** | 10.8 |

\* VmHWM is process-wide and inherited from the alkane_10 full row.
† naux not printed; computed from the measured 56 aux functions per C and 14 per H (N_IP would be 17374).
**alkane_24 cut:** RSS reached 10.8 GB, above the cgroup's MemoryHigh (10G),
with 69% memory pressure and 428k `high` events. The process was reclaim-
throttled 15 min into a roughly 5 min row, so it was killed and gives no
number. Peak memory is the DF-K raw plus dressed tensors held together with
the THC-side buffers. The harness sets `DFK_BUDGET = 10 GiB`
(`thc_k_vs_dfk_timing.rs`, the `DFK_BUDGET` const), which equals the cgroup
MemoryHigh. DF-K was therefore allowed to stay in core right up to the
throttle point, so the cut follows from how the harness was configured,
not from a property of either method. alkane_24 was also the FIRST row inside
the pre-registered crossover window (alkane_24-48): the window itself was
never reached.

**Control (artifact a):** synthetic and real THC builds at the same `N_IP`
agree to within 1%: 0.1151 vs 0.1142 s at alkane_4, 0.5718 vs 0.5767 s at
alkane_8. DF-K build is also reproducible between the two modes
(0.0117/0.0120 and 0.1173/0.1159 s).

### 1c. Controls at alkane_4 (full mode)

| c_isdf | parent grid | N_IP | THC setup s | THC build s | build THC/DF-K | rel max\|ΔK\| | ΔE_x mHa |
|---|---|---|---|---|---|---|---|
| 2.0 | (30,50) | 728 | 5.41 | 0.0042 | **0.36** | 3.17e-3 | **−20.949** |
| 4.0 | (30,50) | 1456 | 16.90 | 0.0187 | 1.58 | 2.38e-4 | +2.225 |
| 8.5 | (30,50) | 3094 | 69.03 | 0.1142 | 9.48 | 8.86e-6 | −0.042 |
| 8.5 | (50,110) | 3094 | 230.04 | 0.1120 | 9.32 | 6.97e-6 | −0.074 |

### 1d. Derived fits over the last three build-mode points (alkane_12/16/20)

| quantity | global exponent in nbf (4..20) | tail-3 | last-2 |
|---|---|---|---|
| THC build | 2.61 | 2.70 | 2.70 |
| DF-K build | 3.62 | 3.82 | 3.78 |
| ratio THC/DF-K | −1.01 | −1.12 | −1.09 |
| THC setup (full, alkane_6/8/10) | — | 2.99 | — |
| DF-K setup (full, alkane_6/8/10) | — | 3.26 | — |

- `ratio × nocc` = 167, 161, 174, 168, 165 at alkane_4..20.
- Effective flop rates against the cost models: THC 21-40 GFLOP/s, DF-K 24-44
  GFLOP/s (model `4·naux·nbf²·nocc`). Both rise with size.
- Tail-3 extrapolation puts the per-build ratio at 1 near **nbf ≈ 925-945
  (alkane_38, nocc ≈ 155)**. This is an extrapolation, not a measurement.

## 2. Exactness anchors and mutation tests (2026-10-24, commit b98d17705)

**THC-K: `thc_k_matches_exact_in_the_trivial_limit`** (water, STO-3G, aux
cc-pVDZ-RI, mock non-orthogonal `C_occ`). Selection runs to the pair-space
rank, n_ip 37 ≥ 28. THC-K vs `DfK` agrees to **4.9e-10** relative max|ΔK| on
both the density path and the occ path; the test asserts < 1e-9.
- This is a conditioning floor, not machine epsilon. The normal-equation fit
  squares the collocation condition number, and `M` carries large cancelling
  entries. At rank_tol 1e-14 (n_ip 28) the floor is 2.3e-9; at pinv_tol 1e-16
  it blows up to 0.36.
- Normalising the `ω` columns (an exact invariance of THC) was needed to
  reach this floor at all. Before normalising it was 5.1e-9.
- At cc-pVDZ, full-rank selection stalls at n_ip 293 of 300, with a floor of
  **1.6e-4**. This is pinned as a regression bar (`thc_k_full_rank_floor_ccpvdz`).
- **Bar replaced (2026-10-24, PR #539 CI):** the fixed `< 1e-9` was 2× one
  machine's measurement, and CI's lint job (AMD EPYC 7763, Ubuntu OpenBLAS)
  failed it at 2.68e-9. Locally, OPENBLAS_CORETYPE=Nehalem/Prescott give
  1.58e-9/1.42e-9 with **n_ip 38**, against Haswell/Zen's 4.9e-10 with n_ip 37.
  The pivot count moves because rank_tol 1e-18 sits below round-off. In every
  configuration the kept spectrum of `S` is exactly 28 (pair rank), with
  dropped eigenvalues at ~1e-16 relative and κ_kept(S) = 8.1e7..8.7e7. The
  test now asserts kept rank == 28 and err < ε·κ_kept(S) ≈ 1.9e-8, the
  first-order sensitivity of the normal-equation solve. Every observed error
  is 0.03..0.15 of that bar. Re-run mutants against the new bar: M1/M2/M3 as
  above, `M·(1+1e-6)` FAIL 1.0e-6, `S⁺` cut at 1e-7 FAIL 2.4e-3, and 37
  points with duplicates (rank 20) FAIL the rank assertion, which
  `n_ip ≥ 28` alone passes.

| ThcK mutant | anchor result |
|---|---|
| M0 none | pass |
| M1 Hadamard → GEMM in `M ∘ P` | FAIL 2.02e1 |
| M2 drop right `S⁺` | FAIL 1.50e1 |
| M3 drop `V^{-1/2}` | FAIL 2.78e1 |
| M4 `S` not squared | FAIL 1.62e-1 |
| S1 selection: skip the `resid -= c²` downdate | killed by `thc_k_selection_matches_column_pivoted_qr` (pivots diverge from the 2nd pivot on), `thc_k_truncated_rank_error_is_bounded` (7.00e-1) and `thc_k_full_rank_floor_ccpvdz` (6.99e-1). The anchor does NOT kill it cleanly: with no downdate the rank stop never fires, and at c_isdf 1e6 it walks the whole parent grid (killed at 19 min, 5 GB RSS). |

**Added in review (2026-10-24):**
- `thc_k_selection_matches_column_pivoted_qr`: the pivoted-Cholesky pivots
  equal those of an independent column-pivoted QR (Gram-Schmidt on the
  explicit `nbf² × npts` product matrix) at k = 60 on water/cc-pVDZ. The
  reviewer also found them identical at k = 120 and 200.
- `thc_k_truncated_rank_error_is_bounded` replaces the old
  `thc_k_default_c_isdf_is_close_to_dfk`. At water/cc-pVDZ-RI (naux 84),
  c_isdf 8.5 rank-clamps at 293 points, so the old test was the full-rank
  floor again, and its "few percent" doc was false (the real error was
  1.63e-4). The new test requests 200 points and measures 3.169e-3. Its
  bounds are derived from both sides: below 6.4e-3 (2× measured) and above
  8.2e-4 (5× the full-rank floor).

**THC χ₀: `thc_chi0_matches_exact_in_the_trivial_limit`** builds generic
factors and constructs `y_α,ia = Σ_K Ω_αK φ_iK φ_aK` explicitly. THC matches
the dense Laplace dielectric to **3.2e-16** at ω = 0, 0.5 and 2.0.
`thc_chi0_error_vs_dense_chi0_is_quadrature_only` measures the residual
against the EXACT dense χ₀ at ω = 0.3: 1.0e-4 / 1.5e-6 / 1.2e-8 at
n_quad 3/5/7, monotone.

- **Pre-port defect, found by mutation:** the pre-port kernel computed
  `Ω (M ∘ Σ c_l G∘H) Ωᵀ`. That is χ₀ of a single `y` only if `M ≡ 1`. Its only
  exactness test used `M = ones`, and **deleting `M` passed all 4 pre-port
  tests**. `M` was removed in the port (the core belongs inside `Ω`).

| ThcChi0 mutant (new tests) | result |
|---|---|
| C1 occ exponent sign flipped | FAIL (anchor 8.6e1; quad-only test) |
| C2 Hadamard → GEMM | FAIL (2.7e-1) |
| C3 drop cos(ωt) | FAIL (3.8e-2 at ω=0.5) |
| C4 re-inject pre-port core `M` | FAIL (7.4e-2) |
| P1 negate the response coefficient | FAIL: anchor, quadrature-only, and the new static-PSD test (min eigenvalue −0.39) |

The former `thc_dielectric_is_symmetric` could not fail, because the kernel
symmetrizes its result. It is replaced by
`thc_dielectric_static_is_at_least_identity`: at ω = 0 every eigenvalue of
`ε̃` must be ≥ 1, since `ε̃ − I` is a positive-weight sum of Gram matrices.

## 3. Pre-registration (written 2026-10-24 00:35 EDT, BEFORE the timing run)

Question: does THC-K's per-iteration build beat ferric's current DF-K
(`DfK::build_from_occ`, the occupied half-transform path) on cc-pVDZ alkanes,
and is the SCF-level total (setup + n builds) ever a win?

Cost model, both builders dense GEMM:
THC `≈ 2·nbf·N_IP²` with `N_IP = 8.5·naux`; DF-K `≈ 4·naux·nbf²·nocc`.
Ratio THC/DF-K `≈ 36·naux/(nbf·nocc)`; with `naux/nbf ≈ 2.6` (cc-pVDZ-RI on
alkanes) that is `≈ 95/nocc`, or `≈ 190/nocc` if DF-K's GEMMs run at twice
THC's flop rate.
(Review note, added after the run: the THC model above leaves out forming
`P = ΦᵀΦ`, `2·nocc·N_IP²`, which is about `nocc/nbf ≈ 17%` of the dominant
term at alkane_20. The measured GFLOP/s in §1d include it.)

- **If a real crossover exists I expect:** the build ratio to rise at small
  sizes (rank clamp, overheads), peak, then FALL roughly as `1/nocc`; tail
  exponents near `N^3` (THC) and `N^4` (DF-K), and a per-build crossover near
  `nocc ≈ 95-190`, i.e. alkane_24 to alkane_48. The SCF-level crossover lies
  later than the per-build one by however much THC's one-time setup costs.
- **If the benchmark is mis-built I expect:**
  (a) synthetic-factor THC builds that disagree with real-factor builds at
  the same `N_IP` (checked on the overlap sizes, scaled by `N_IP²`);
  (b) DF-K spilling to disk at the largest sizes, which shows up as a DF-K
  tail exponent well above 4 and a sudden ratio drop (checked with VmHWM vs
  the in-core tensor size);
  (c) a ratio trend that is non-monotone from row to row, which on this
  shared box means contention, not physics;
  (d) the THC rank silently clamped below `8.5·naux`, which would make THC
  look cheaper than its own recipe (the full rows report the achieved
  `N_IP` and `c_eff = N_IP/naux`).
  A crossover that appears only under (a)-(d) is an artifact.
- **Accuracy gate:** a THC row only counts if `|ΔE_x|` vs DF-K on the same aux
  stays well under 1 mHa (chemical accuracy is 1.6 mHa total).

**Outcome vs pre-registration:**
- (a) passed, to within 1%.
- (b) did not occur through alkane_20: the DF-K tail exponent is 3.8, not
  above 4, and VmHWM is below the in-core sizes. At alkane_24 the cgroup
  throttled instead, and that row was cut.
- (c) the ratio is monotone from alkane_3 onward.
- (d) applies only to alkane_1 and alkane_2.
- The ratio falls as `nocc^-1` with constant `≈ 165`, inside the predicted
  95-190 band. The pre-registered `naux/nbf ≈ 2.6` was low: it is 3.5 measured.
  At equal flop rates the model therefore gives about 125/nocc, so DF-K runs
  slightly faster per flop. The predicted crossover window was alkane_24-48; the
  extrapolation lands at alkane_38.
- "Too clean" check: the constant `ratio × nocc` is arithmetic of the shapes.
  Both builders sit at similar GEMM rates and `naux/nbf` is fixed along the
  series. It confirms the cost model; it says nothing about chemistry.

## 4. Interpretation (2026-10-24, PROVISIONAL)

1. **The old premise is refuted.** The parked lane assumed DF-K ran at
   1.4-3.8 GFLOP/s, so THC would cross over early on wall clock. DF-K on main
   runs at 24-44 GFLOP/s on one thread, the same rate as THC's GEMMs, so the
   flop model governs. THC-K loses on every measured per-build row: 2.0× slower
   at alkane_20 (nbf 490), with the gap closing as about `1/nocc`.
   **The measured range ends BELOW the pre-registered crossover window**
   (alkane_24-48; alkane_24 was cut, §1b). The per-build negative is
   therefore a below-onset negative. It shows the gap is still closing; it does
   NOT show that no per-build crossover exists.
2. **Per-build crossover: extrapolated, not measured.** The extrapolation
   puts it at nbf ≈ 930 (alkane_38). That is beyond where DF-K's in-core
   tensor fits this box (about 22 GB there), so in that regime THC would be
   competing with spilled DF-K, LinK or COSX, none of which were measured here.
3. **SCF-level: no win in any measured regime.** Beyond it, the following is
   extrapolation:
   - MEASURED: THC setup is 889 s at alkane_10, about 3500 DF-K builds, and
     it grows as N^3.0 over alkane_6-10.
   - EXTRAPOLATED (tail-3 ratio fit): past the per-build crossover, the
     per-build saving grows only as about N^1.1. INTERPRETATION: setup cannot
     be amortised over an SCF of about 20 iterations near the crossover
     unless setup falls by about 10³.
   - ESTIMATES (N^3.0 setup fit, and `npts·N_IP·8` bytes for the selection
     factor): about 13 h of setup and about 38 GB of selection memory at
     alkane_38.
   - The setup was NEVER broken down by stage. Selection is an unblocked
     BLAS2 pivoted Cholesky at about `npts·N_IP²` flops: about 2.6e12 at
     alkane_10 (npts 48000, N_IP 7378). At BLAS2 rates that is LIKELY most of
     the 889 s, but this is inferred, not timed. A blocked BLAS3 selection
     could change the setup prefactor substantially.
4. **A speedup bought with accuracy is not a win.** c_isdf = 2 makes THC 2.8×
   faster than DF-K at alkane_4, but ΔE_x = −21 mHa. c_isdf = 4 is still
   1.6× slower, with ΔE_x = 2.2 mHa (above chemical accuracy). Only c ≈ 8.5
   reaches sub-0.1 mHa through alkane_8; alkane_10 already shows −0.20 mHa.
5. **Verdict (provisional, below onset):** in its current form (AO-THC, dense M, Gram/
   Cholesky selection), THC-K is not a DF-K replacement on this box at any
   reachable size. Keep it feature-gated as a correctness artifact. Do not
   wire it into SCF.

## 5. NOT measured (do not infer)

- Any THC-K row above alkane_10 with real factors: no accuracy, rank or setup
  past nbf 250. The alkane_12-20 rows are cost-only.
- The per-build crossover itself (extrapolated only). Anything past
  alkane_20; alkane_24 was cut.
- An SCF loop driven by THC-K: the variational-collapse question (Lee et al.
  §7.1) remains open.
- The JK-fit aux (def2-universal-jkfit): naux is larger there, which raises
  `N_IP` and moves the crossover later.
- Multi-thread timings; BLAS > 1. Comparisons vs LinK, COSX and spilled DF-K.
- A per-stage setup breakdown (grid/collocation, selection, eri3 collocation,
  `S⁺`, the `M` GEMMs), and a blocked BLAS3 selection. The "setup must fall
  about 10³" in §4.3 is interpretation that rests on this unmeasured split.
- Selection on NORMALISED product columns. Reviewer probe: QR on the
  unnormalised `X` gives 5.5e-5 at cc-pVDZ water full rank, against the
  Cholesky selection's 1.63e-4, so the selection metric is itself a lever.
- Shell-pair-sparse M formation, ISDF/CVT point selection, a QR-based
  (non-normal-equation) fit, and a symmetric (dsyrk/dsymm) THC build. Each is
  an untested lever with a plausible 2× class effect. None changes the
  setup-amortisation argument in §4.3 unless setup falls by about 10³.
- THC χ₀ on any real molecule: no ISDF builder for the projected pair
  densities exists, and no timing was taken.

## 6. Port record

- **Ported:**
  - `crates/ferric-scf/src/thc_k.rs`. Setup rewritten with exact identities,
    so neither the `(nbf², npts)` product matrix nor the `(nbf², nbf²)` RI-ERI
    matrix is formed. Pivoted-Cholesky selection replaces `dgeqp3`, so there
    is no new dependency and `Cargo.lock` is unchanged. Added `from_parts`.
  - `crates/ferric-rpa/src/thc_chi0.rs`, with the `M` defect fixed.
  - `benchmarks/harness/examples/thc_k_vs_dfk_timing.rs`, rewritten. The
    pre-port version had never been run, and its `ThcK::new` could not have
    fit in memory past about alkane_3.
- **Left out:**
  - Phase-0/1 one-off probes (`thc_ao_z_density`, `thc_fit_gate`,
    `thc_k_costmodel`, `thc_x_sparsity`). Their numbers are recorded in the
    pre-port docs, and the cost-model probe is superseded by §1.
  - The branch's `docs/`, `scripts/queue`, `benchmarks/grid` and `CLAUDE.md`
    purge residue.
