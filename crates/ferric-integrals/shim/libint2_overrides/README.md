# libint2 header overrides

Each `<version>/libint2/boys.h` here is a patched copy of that libint2 version's stock `include/libint2/boys.h`, and
`<version>/libint2-<version>-gmeval-no-copy.patch` is the exact diff from stock to that copy. `patch -p1` applied
inside an install prefix of that version reproduces the file byte for byte.

| Directory | Upstream | Stock `boys.h` sha256 | FNV-1a 64 (`build_support/boys_policy.rs`) |
|-----------|----------|-----------------------|----------------------|
| `2.7.2/` | `libint-2.7.2-mpqc4.tgz` | `880592b9026352c871c96877a43788edd22edb0b1057ab5a73657c40b3a61505` | `0x1d51ac52f2076ad8` |
| `2.13.1/` | conda-forge `libint-2.13.1-h83f5b4b_0.conda` (`scripts/install-libint.sh`) | `890ab0261a2be2517d3d4cedfd7d07e2a1c6be7b6f057dc7ea2ea418e17067c1` | `0x0f6f0c23194e18ad` |

Override sha256 (what `patch -p1` of the committed `.patch` onto the stock file reproduces): 2.7.2
`8f5140d5b4e4132e0f90cfe3fee03bdae7971aee4b14dffc7292e09463854dc5`, 2.13.1
`14c8bdfd4d8fb41e0c4e5091de1c45e84fe5f5840c6cdcfb1c40419a368f343e`.

The #433 fix is the same in both. The sections below describe it with the 2.7.2 names. The `TennoGmEval` table-edge
changes (issue #436, section "Third change") exist ONLY in the 2.7.2 override; 2.13.1 already has them in stock form.
The `eval_urr` overflow fix exists in both (#436 for 2.7.2, #439 for 2.13.1, section "Fourth change"). The small-`T`
Yukawa fix (#443, section "Fifth change") also exists in both. In 2.13.1, one evaluator,
`erfx_coulomb_gm_eval`, replaces `erfc_coulomb_gm_eval`. It serves `erfc_coulomb` (with lambda=0, sigma=1),
`erfx_coulomb`, and the 1-body `erfc_nuclear`/`erfx_nuclear`. `erf_coulomb_gm_eval` serves `erf_coulomb` and
`erf_nuclear`. The patch applies to both evaluators.

## What it fixes

For `Operator::erf_coulomb` and `Operator::erfc_coulomb`, `Engine::compute2` calls
`GenericGmEval<...>::eval` once per surviving primitive quartet. Stock `eval` does
`(GmEvalFunction(*this))(...)`, which copy-constructs the evaluator on every call:

- `erf_coulomb_gm_eval` and `erfc_coulomb_gm_eval` each hold a `shared_ptr` to the process-wide
  `FmEval_Chebyshev7::instance()` singleton. Each copy does an atomic increment and an atomic decrement on that
  singleton's control block, which is one cache line shared by every engine in every thread.
- `erfc_coulomb_gm_eval` also inherits a `std::vector` scratch (`CoreEvalScratch`), so each copy also does
  `operator new`, `memmove` and `operator delete`.

Measured on ferric (`examples/pbc_parallel_contention.rs`, 20000 calls, best of 3, i7-6800K): six threads in one
process inflate per-call time 3.28x (erfc 3-centre) and 4.07x (erfc Gaussian-nucleus hcore). Six separate
single-thread processes inflate it only 1.00-1.08x. The FP-only control inflates 1.13x. So the cost comes from state
shared inside the process.

## The change (boys.h only)

1. `detail::gm_eval_call_by_const_ref<F>` is a new trait, true only for `erf_coulomb_gm_eval` and
   `erfc_coulomb_gm_eval`.
2. `GenericGmEval::eval` dispatches on that trait. For those two evaluators it calls
   `static_cast<const GmEvalFunction&>(*this)(...)`, so there is no copy, no refcount change and no allocation.
   Every other evaluator (`delta`, `r12_xx_K<1>`) keeps the stock copy path unchanged.
3. `erfc_coulomb_gm_eval` no longer inherits `CoreEvalScratch`. Its `operator()` is `const` and uses a
   `Real[64]` stack buffer. `FmEval_Chebyshev7` rejects `mmax > 40`, so that buffer always suffices; a local
   `std::vector` fallback exists only for `LIBINT_USER_DEFINED_REAL` builds. The arithmetic is the stock sequence:
   the same `fm_eval_->eval` calls with the same arguments, the same copy into `Gm`, and the same loop. Only the
   buffer's address changes. `FmEval_Chebyshev7::eval` stores with unaligned stores (`convert`) or scalar stores,
   so alignment does not matter.
4. `erf_coulomb_gm_eval` was already `const` and needed no change.

Thread safety: the const call path only reads `fm_eval_`, whose pointee is immutable, and writes to the stack. It is
therefore race-free even if one `GenericGmEval` were shared by several engines. That sharing happens when you copy an
`Engine`, because its `core_eval_pack_` holds a `shared_ptr`.

Plain `Operator::coulomb` never goes through `GenericGmEval`: `FmEval_Chebyshev7::eval` is called through a const
pointer. It is untouched.

## Second change: `FmEval_Chebyshev7::eval` table edge (issue #433)

`eval` took the asymptotic branch only for `x > T_crit` (`T_crit = cheb_table_tmax = 117`) and interpolated
otherwise with `iv = int(x * 7)` into a table of `819 = 117 * 7` intervals (rows 0..=818). At `x == 117.0` exactly,
`iv == 819` reads the coefficients one row past the end of the `posix_memalign`'d table, i.e. uninitialised heap.
The override uses `x >= T_crit`. Every `x != 117.0` takes the same branch as before (bit-identical results); at
`x == 117.0` the asymptotic value replaces garbage. That value is what any `x > T_crit` already gets: its omitted
`e^-x/(2x)` term is about 1e-16 relative at m = 0 and 9e-17 at m = 40, below the roughly 4e-15 rounding of the
upward recursion.

### Exposure (who can hit it)

ANY integral family at ANY precision, conditional only on a Boys argument that is exactly `117.0` in floating point.
That is `rho * |PQ|^2` for the Coulomb kernel, and for erf/erfc both `T` and `T * omega^2/(omega^2 + rho)` are
evaluated. Generic geometries essentially never produce it; integer or round coordinates and exponents do, and so do
lattice shifts. A plain-libint counter-example with no ferric code involved: Coulomb 3-centre `(P|AA)`, `P` an s shell of
exponent 2 at the origin, `A` an s shell of exponent 1 at (9, 6, 0) Bohr. Then `rho` is exactly 1 and `|PQ|^2` exactly
117, so at precision 1e-14 and 1e-10 stock libint returns 4.6e-310 (default allocator) or -1.4e304 (`MALLOC_PERTURB_=1`)
where the patched header returns 0.218157161196848, the closed form. In the default allocator the excess memory is
usually zero, so the stock result is a silently wrong, plausible-looking number rather than an obvious failure.

The precision dependence seen on the original reproducer (garbage at 1e-20 and below, clean at 1e-14) is NOT a
property of the defect. It only says whether the one primitive quartet that hits 117.0 survives primitive screening:
that quartet's contribution was about 1e-17, so a coarse precision screened it away. Do not read "clean at 1e-14" as
"safe at 1e-14".

This is a libint2 defect (upstream `boys.h`, every version that ships `boys_cheb7_v2.h`: both 2.7.2 and 2.13.1 are
confirmed). A prefix whose `boys.h` is not one of the stock files above gets no override and MAY still read
uninitialised memory; build.rs warns that the fix is NOT applied, or fails under `FERRIC_REQUIRE_PATCHED_BOYS=1`.

### Evidence and its limits

- Tests: `crates/ferric-integrals/tests/boys_chebyshev_table_edge.rs` (direct evaluator pin via `scf_boys_fm`,
  exact-integer-geometry reproducer, point-site reproducer, allocator-fill sweep). `tests/boys_chebyshev_manifest.rs`
  is a PRINT-ONLY helper for before/after bit comparisons by hand, not a guard.
- Hit-count survey of existing tests (2026-10-16; an instrumented STOCK 2.13.1 build counting evaluations at exactly
  117.0, default allocator, 400 s timeout per binary): 256 test binaries of ferric-integrals, ferric-pbc, ferric-mp2,
  ferric-rpa; 218 passed, 26 failed with 0 hits (environmental: 25 re-run serially all pass, 0 hits), 12 timed out
  (rc=124): hirshfeld_symmetry, pbc_grad_rsgdf, pbc_kgrad, pbc_kpair_symmetry, pbc_krsh_rsgdf, pbc_kstress,
  pbc_kuxcstress, pbc_kxcgrad_hybrid, pbc_kxcstress, pbc_kxcstress_mgga, pbc_parallel_bitwise, pbc_sr_rotation. Of
  those 12, 3 were measured afterwards (per test: 39 tests, 0 hits); the issue does not name which. The one known real
  hit is the #421 ECP k-point test (80 evaluations at 117.0 per run; failing under `MALLOC_PERTURB_` 1 and 170 before
  the fix, passing after).
- Perturbed-allocator survey (#433 "impact survey"; 2026-10-17/18; VERDICT PROVISIONAL). It measures exit status
  under `MALLOC_PERTURB_` against each test's existing bars. It does NOT count evaluations at 117.0, so it does not
  show that any run evaluated a Boys argument at exactly 117.0, and it does not replace the hit-count measurement: the
  hit counts of 9 of the 12 timed-out binaries above remain unmeasured (UNMET, #433).
  - Scope: 28 ferric-pbc binaries (of 69 test binaries), ferric-rpa `hirshfeld_symmetry`, and the molecular files of
    harnesses F and G. The 12 timed-out binaries are all covered at `MALLOC_PERTURB_=170` in some mode below. Not perturbed by this survey (40 ferric-pbc
    binaries, not examined for whether they reach an integral far-field path): pbc_auto_omega pbc_dense_aft_scf
    pbc_drpa pbc_ecp pbc_forces_heavy_atoms pbc_grsh pbc_hcore pbc_kcorr pbc_kks_open_ecp pbc_kpoint_ecp pbc_krhf
    pbc_krks pbc_krks_shifted pbc_kroks pbc_krsh pbc_krsh_more pbc_kuhf pbc_kuks pbc_kuxcstress_ro_rsgdf pbc_lindep
    pbc_lmp2 pbc_lr_force_bitwise pbc_memory_gates pbc_molecular_limit pbc_mp2 pbc_pair_symmetry pbc_rks pbc_rohf
    pbc_rsgdf pbc_rsgdf_split pbc_shifted_overlap pbc_sr_deriv_frame pbc_sr_rotation_k pbc_sr_rotation_kderiv
    pbc_sr_screening pbc_stage0 pbc_timings pbc_ucorr pbc_uhf pbc_uks. `pbc_kkeep` and `pbc_kkeep_ks` (k-point
    selection) are perturbed in harness G. `pbc_kpoint_ecp_rotation` (ferric-pbc) and the ferric-integrals
    `boys_chebyshev_table_edge` and `tenno_table_edge` are covered by the CI `malloc-perturb` job (1, 170, 2). Observed
    on CI: each `nextest run --archive-file` invocation costs ~3.5-4 min of archive extraction regardless of the tests run; a per-binary loop (6 binaries x 3 values = 18
    invocations) was cancelled at the 60 min job limit, and the 3-binary version (9 invocations) took ~38 min. The job
    now makes 3 invocations (one per value, union filterset, per-binary PASS check); measured job wall time 35m46s
    (run 39199767707) and 31m07s (run 39233925174).
  - Harness A, `MALLOC_PERTURB_=170`, whole binaries with `--include-ignored` (libtest-style, `--test-threads=1`), 1800 s
    cap, release build, OPENBLAS_NUM_THREADS=1, 22 ferric-pbc binaries: the 15 under Harness D plus `pbc_kxcstress`,
    `pbc_grad_ecp_nucleus_scan`, `pbc_kuxcstress`, `pbc_kuxcstress_ro`, `pbc_kxcgrad`, `pbc_kxcstress_hybrid` and
    `pbc_kstress`. Result: 20 of 22 exit 0, 2 exit 124 (`pbc_kxcstress_hybrid`, `pbc_kstress`; inconclusive as whole
    binaries). The raw per-binary logs of this run are not retained, so its wall seconds are UNRECORDED / approximate and
    are not repeated here (the longest were roughly 25-28 min, unverified).
  - Harness B, `MALLOC_PERTURB_=170`, nextest, release build, non-ignored tests only, one test process at a time,
    1200 s cap per test: `pbc_kstress` (33 tests) and `pbc_kxcstress_hybrid` (11 tests), all 44 exit 0 (1-102 s and
    7-52 s per test; 670 s and 232 s totals). Their `#[ignore]`d tests were not run in this mode.
  - Harness C, `MALLOC_PERTURB_=170`, nextest, release build, non-ignored tests only (these binaries have no
    `#[ignore]`), `--test-threads 1`, 1200 s cap per binary: pbc_kpair_symmetry (10 tests, 214 s), pbc_krsh_rsgdf
    (13, 175 s), pbc_parallel_bitwise (25, 739 s), pbc_sr_rotation (15, 505 s), ferric-rpa hirshfeld_symmetry (4,
    161 s); all pass, rc=0.
  - Harness D, `MALLOC_PERTURB_=1`, nextest, release build, non-ignored tests only, `--test-threads 1`, 1200 s cap per
    binary: 15 of the 22 exit 0 (nextest summary, all tests passed): pbc_krsgdf (13 s), pbc_krsgdf_split (53),
    pbc_grad_ecp (29), pbc_grad_open (146), pbc_grad (20), pbc_grad_rsgdf_split (89), pbc_grad_ro (217),
    pbc_kxcgrad_mgga (96), pbc_stress (52), pbc_grad_rsgdf (214), pbc_kxcgrad_hybrid (373), pbc_kuxcgrad_ro (54),
    pbc_kgrad (524), pbc_kxcstress_mgga (1000), pbc_kuxcgrad (684). Harness D's `pbc_kxcstress_mgga` at 1000 s versus
    the 574 s of the earlier Harness A run is a different harness and box load, not a comparison.
  - Harness E, `MALLOC_PERTURB_=1`, `cargo test --release` (libtest, `RUST_TEST_THREADS=3`, `OPENBLAS_NUM_THREADS=1`),
    build of origin/main d3be28fc, the remaining 7 of the 22 binaries, non-ignored tests only unless stated; every
    binary exit 0 (libtest "test result: ok", 0 failed). Seconds are libtest's "finished in": pbc_kxcstress_hybrid (11
    passed, 123 s), pbc_kuxcstress_ro (7, 71), pbc_kuxcstress (7, 160), pbc_kxcgrad (13, 115), pbc_kxcstress (18, 212),
    pbc_kstress (33, 322), pbc_grad_ecp_nucleus_scan (both `#[ignore]`d diagnostics run with `--ignored
    --test-threads=1`: 2 passed, 418 s). With harness D, all 22 binaries now have a passing run at both `=1` and
    `=170`; for `pbc_kstress` and `pbc_kxcstress_hybrid` the `=170` evidence is harness B (non-ignored tests only, one
    test process at a time), not a whole-binary exit 0 (harness A timed out on both). `#[ignore]`d tests: run at `=170`
    only (harness A, so not for those two binaries), except `pbc_grad_ecp_nucleus_scan`, whose ignored tests also ran
    at `=1` here.
  - Harness F, molecular attenuated (erfc) paths, same build and settings as E, `MALLOC_PERTURB_` 1 and 170, all exit 0
    with identical pass counts at both values: ferric-mp2 `--lib attenuated` (7 passed; the 2 `#[ignore]`d decane
    timing benchmarks not run), ferric-mp2 `validation_attenuated_mp2 --include-ignored` (3 passed, PySCF anchors +
    trivial-limit anchor), ferric-rpa `--lib rs_mp2_rpa` (9 passed), ferric-rpa `validation_rs_mp2_rpa
    --include-ignored` (3 passed), ferric-rpa `validation_attenuated_rpa --include-ignored` (4 passed); each under
    9 s.
  - Harness G, same build and settings as E. At `MALLOC_PERTURB_` 1 and 170, all exit 0 with identical pass counts
    (non-ignored tests): ferric-mp2 `lmp2_direct_terfc_omega` (3 passed, 1 ignored), `lmp2_direct_terfc_schwarz` (1
    passed, 2 ignored; 24-25 s), ferric-rpa `mwe_rs_mp2_rpa_second_intermediate` (4 passed), ferric-pbc `pbc_kkeep` (4
    passed, 20-22 s), `pbc_kkeep_ks` (3 passed, 39-43 s). ferric-mp2 `gpu_rimp2_mixed_attenuated` is gated on the
    `gpu` and `test-seams` features and compiled to 0 tests in this build, so it is NOT exercised. At `=1`, the
    harness C binaries, all exit 0: pbc_kpair_symmetry (10 passed, 145 s), pbc_krsh_rsgdf (13, 175), pbc_parallel_bitwise
    (25, 366), pbc_sr_rotation (15, 364), ferric-rpa hirshfeld_symmetry (4, 184).
    All elapsed times in harnesses A-G are local wall-clock on one developer box, not CI-runner measurements.
  - Interpretation (provisional): no allocator-dependent failure was seen in any run. Exit 0 against each test's
    existing bars is not proof of no uninitialised reads: the tests may not reach the table edge, and a
    perturbation-insensitive wrong answer inside a loose bar would pass.
  - Not perturbed (not exhaustive): the 40 ferric-pbc binaries listed under Scope; `gpu_rimp2_mixed_attenuated` (needs
    the `gpu` feature); the `#[ignore]`d tests of the harness G files; the fully `#[ignore]`d molecular files
    `validation_scs_mp2_2terfc`, `attenuated_alpha_c6_probe`, `erfc_vs_terfc_timing` and the ferric-mp2 `terfc_*`
    scratch probes.
  - Status of #433 item 2: met for the 22 binaries of harnesses A-E, the harness C binaries, `pbc_kkeep`,
    `pbc_kkeep_ks` and the molecular attenuated suites listed in F and G (non-ignored tests at `=1` and `=170`); not
    met for the 40 unperturbed ferric-pbc binaries listed under Scope.
- Not checked (unmet, #433): aarch64 and FMA-contracting builds (whether a given geometry lands on exactly 117.0
  depends on rounding, which is why the integer-geometry test exists).
- Not done (maintainer-owned): notifying upstream libint.

### Sibling defect: see the third change below (issue #436)

## Third change (2.7.2 only): `TennoGmEval` table edges and overflow (issue #436)

`TennoGmEval` is the evaluator behind `Operator::yukawa` and `Operator::slater_geminal` (libint2 `stg_x_coulomb`,
`stg`). `eval_yukawa` / `eval_slater` take the Chebyshev table for `T <= Tmax = 1024` and, for Yukawa, `U >= Umin`, with
`U = zeta^2 / (4 rho)`, `Umin = 1e-7`, `Umax = 1e3`. Four defects, found by the independent review of #433 and
reproduced here with a standalone probe under AddressSanitizer (`probe.cc`-style, `TennoGmEval::instance(m)->eval_yukawa`):

1. **T index.** `Tint = floor(log2(T))` with the table's octaves 0..=9 (the last is [512, 1024]). `T == 1024` exactly,
   and the doubles just below it whose `log2` rounds to 10 (1023.9999999999999 does), give `Tint == 10`:
   `one_over_2K[10]` and the coefficient block `c_[interval ...]` are read past their ends. ASan:
   stack-buffer-overflow in `interpolate_Gm`; a non-NDEBUG build aborts on the `assert(Tint < 10)`.
2. **U index.** Likewise `Uint = floor(log10(U / Umin))`: `U == 1000` and the two doubles below it
   (999.9999999999998, 999.9999999999999) give `Uint == 10`.
3. **NaN in `Gm[0]`.** The explicit `G_0` / `G_{-1}` form `exp(x*x - T) * erfc(x)` with `x = sqrt(U) +- sqrt(T)`:
   for `x` above ~26.5 `erfc(x)` is subnormal and then 0 while `exp(x*x - T)` overflows once `U + 2 sqrt(U T)`
   exceeds ~709.8, so the product is `inf * 0 = NaN`. This is not an edge: stock 2.7.2 returned NaN or inf
   for `Gm[0]` over the whole upper part of the nominal domain (measured, 184 of 12640 grid points; e.g. every T at
   U = 1000, and T = 562 at U = 158). The `T > Tmax` upward recursion (`eval_urr`) and the `T == 0` branch share the pattern.
4. **No `U` range check.** `U > Umax` was only an `assert` inside `interpolate_Gm` (and no check at all on the
   `T > Tmax` path); `eval_slater` also sent `T <= Tmax` with `U < Umin` to `interpolate_Gm`, where `Uint` is
   negative (index before the arrays).

### What the override does

* `Tint` and `Uint` are clamped with `std::min(..., 9)`. Only an index of 10 changes, to the last octave/decade at its
  upper end (`t = +0.5` / `u = +0.5`); every other argument keeps its index and its result bit for bit. libint 2.13.1
  fixes (1) and (2) by scaling the logarithm by `(1 - eps)`, which would also move `T = 2^k` and its ulp neighbours to
  the lower octave; the clamp is the smaller change.
* `exp_x2_erfc(x, x*x - T, -T)` returns the stock `exp(x*x - T) * erfc(x)` while `erfc(x)` is a normal number
  (bit-identical), and otherwise `exp(-T) * erfcx(x)` with the asymptotic series
  `erfcx(x) = (1 / (x sqrt(pi))) sum_n (-1)^n (2n-1)!! / (2 x^2)^n` truncated after n = 8 (first omitted term
  ~4e-21 at the switch point x = 26.5). The `T == 0` branch does the same with `G_{-1} = S / (2U)`, `G_0 = 1 - S`.
  Where the stock expression was a subnormal-precision value (erfc between 5e-324 and 2.2e-308) it is replaced by the
  accurate one, so those few points DO change: of the 12640 swept grid points, 211 differ from stock, all of them
  points where stock returned NaN/inf or such a degraded value (accuracy of the new values: see Evidence).
* `eval_yukawa` / `eval_slater` throw `std::invalid_argument` for `U > Umax` (or NaN) with `T <= Tmax` (above both
  limits the true value is exactly 0 and the upward recursion returns it), and `eval_slater` for
  `U < Umin` with `T <= Tmax`; stock either aborted (assert) or read out of range or returned NaN there. A throw inside
  an `Engine` compute is caught by the shim (`SCF_EINTERNAL`) and becomes a Rust panic carrying the status, the same
  path as every other libint exception. libint 2.13.1 instead falls back to a Gaussian fit of the kernel for these;
  2.7.2 has no such fit, so the override refuses rather than returning NaN. Revert these three `throw`s (and keep the
  rest) if a caller needs the NaN.

Not changed: `eval_yukawa` with `U < Umin` or `T > Tmax` still uses the upward recursion, which is unstable at small `T`
(garbage, or an assert abort at exactly `T == 0` in a non-NDEBUG build), as documented on `Operator::yukawa`.

### Evidence and its limits

* Tests: `crates/ferric-integrals/tests/tenno_table_edge.rs` (direct evaluator through `tenno_gm` against the defining
  integral by Gauss-Legendre, exact-integer engine reproducers for `T == 1024.0` and `U == 1000.0`, allocator-fill
  sweep under `MALLOC_PERTURB_` 1, 170, 2). On a build whose boys.h is not the patched 2.7.2 one they skip with a
  message (2.7.2 stock aborts); on 2.13.1 they run as version-independent regressions.
* To build and run the tests against 2.7.2 (the PyPI-wheel libint, `~/.local`) here: `LIBINT2_PREFIX=$HOME/.local
  CARGO_TARGET_DIR=<scratch> cargo test -p ferric-integrals --test tenno_table_edge`; build.rs then selects this
  directory by the stock hash and prints "libint2 2.7.2 boys.h override applied".
* Bit-identity (C++ sweeps against stock 2.7.2 headers, `-O2`, asserts on): `eval_yukawa`/`eval_slater` over
  T in {1e-6 ... 2000, every 2^k and its two ulp neighbours, 0, 1e5} x U in {1e-7 ... ~900, every 10^k and neighbours} x
  m <= 6, 12640 points: 12429 bit-identical to stock; 211 differ, comprising the 184 points where stock had a
  non-finite slot (now 0 non-finite) and the rest where stock's erfc was subnormal. Whole `Engine` 2-, 3- and 4-centre Yukawa and Slater blocks,
  s/p/d shells x 7 centres (same-centre, 9-6-0 integer, 5.5 along x ...) x zeta in {0.3, 1, 3, 10}, 28592 blocks:
  FNV-1a hash `25864f7f62d9114b` for stock, for the pre-#436 override and for this one, 0 non-finite values.
  Rust-side, `tenno_table_edge::sweep_child_prints_hashes` prints `SWEEP_HASH edge=bbf3ce42b1378db6
  engine=de36c9077af007c0 n=296` on 2.7.2+override and, on stock 2.13.1 (before #439), `edge=2acf9250b8e24a57
  engine=0ed77033630946fd`; with the 2.13.1 override of #439 the `edge` hash is `b445e0973c928c4d` (its T = ulp(1024) points at U = 100..1000 were NaN on stock 2.13.1 and are now 0) and `engine` is unchanged (see the Fourth change).
* Accuracy of what changed. Points where stock was non-finite (184 of the 12640 grid points) are now finite and
  accurate: of the 210 changed grid points checked against a 500-digit mpmath evaluation, the largest absolute error
  is 4e-19 on the swept grid. That is NOT true of every changed point in general. `G_0 = pfac (erfc_k - erfc_l) /
  sqrt(T)` cancels at small `T`, and where stock was FINITE but `erfc_l` alone crossed to the series (`U` about
  700 to 720), the new value carries that cancellation: an independent dense scan (T in 1e-8..5, U in 690..720, 488
  points) found the largest absolute error on a changed point 3.9e-13 for `G_0` at T = 1e-6, U = 704.5 (stock there
  2.3e-13), and 12 points where the new value is worse than stock although stock was finite, by up to 9.7x (e.g.
  T = 1e-3, U = 703.5: new 1.87e-14, stock 7.1e-15). The same `exp(x*x)` rounding amplification (about `x^2 eps`)
  limits the unchanged stock-form branch: worst relative error of `Gm[0]` 4e-10 (T = 1e-6, U = 570). The set of
  switched points was deliberately NOT enlarged to chase these (the bit-identity set stays exactly "erfc subnormal").
  In the `T == 0` branch `G_0 = 1 - S` is evaluated as `y * tail(y)`, avoiding the 1.7e-14 to 6.8e-14 relative loss
  of `1 - S`; that code only runs where stock was non-finite or subnormal-precision, so it does not move the set.
* libint 2.13.1 has the same `T > Tmax` overflow in `eval_urr`; it is fixed in the 2.13.1 override (issue #439, section
  "Fourth change" below). Its table indices are clean at T == 1024 and U == 1000 (ASan/UBSan probe, both edges and one
  ulp each side).
* CI coverage gap: CI runs only the 2.13.1 build, so the 2.7.2-specific assertions of `tenno_table_edge` (guarded by
  `FERRIC_LIBINT2_BOYS_OVERRIDE`) are protected only by a manual run (`LIBINT2_PREFIX=$HOME/.local CARGO_TARGET_DIR=...
  cargo test -p ferric-integrals --test tenno_table_edge`). The cheap protection is a `wheels.yml` step inside the
  manylinux image (which has the 2.7.2 libint) running that same test.
* Follow-up, not in this change: the engine constructors could return a typed `Err` up front instead of the shim's
  `SCF_EINTERNAL` panic: compute `rho_min` from the smallest primitive exponents and return `Err` when
  `zeta^2 / (4 rho_min) > 1e3` on 2.7.2 builds.
* A standalone ASan/UBSan probe of the patched header over the edge set (`T` and `U` on, one ulp each side of, and away
  from the edges, both kernels, m <= 6) and the full safe grid is clean; the same probe against stock 2.7.2 aborts or
  reports the overflow at the first edge point.

## Fourth change (2.13.1 only): `TennoGmEval::eval_urr` overflow (issue #439)

libint 2.13.1 already guards the table indices (no #436 items 1, 2, 4: `U > Umax` falls back to a Gaussian fit of the
kernel, `T == 1024` / `U == 1000` are clean under ASan/UBSan) and `eval_G0_and_maybe_Gm1` (the `T <= Tmax`
path) returns `nullopt` and falls back to the Gaussian fit when `lambda^2 - T > 709`. What it does NOT guard is
`eval_urr`, the `T > Tmax` upward recursion (and `eval_yukawa` with `U < Umin`): it forms
`erfc_k = exp(kappa*kappa - T) * erfc(kappa)` and `erfc_l = exp(lambda*lambda - T) * erfc(lambda)`,
`kappa, lambda = sqrt(U) -+ sqrt(T)`, which is `inf * 0 = NaN` once `lambda*lambda - T = U + 2 sqrt(U T)` exceeds about
709.78 (`exp` overflows while `erfc(lambda)` is already 0). Both kernels, every `m` (the recursion propagates the NaN).

### The change

The same `exp_x2_erfc(x, x*x - T, -T)` helper as the 2.7.2 override (copied verbatim, with `erfcx_asymptotic`,
`erfcx_series`, `erfcx_series_tail`), used for `erfc_k` and `erfc_l` in `eval_urr` only: the stock product while
`erfc(x)` is a normal number (bit-identical), `exp(-T) * erfcx(x)` from the 8-term asymptotic series otherwise.
`eval_G0_and_maybe_Gm1` is untouched (its `lambda^2 - T > 709` check already routes to the Gaussian fit). Unlike the 2.7.2
override there are no new `throw`s (2.13.1 has its own fallback) and no index clamps.

### Evidence and its limits

* Reproduction, stock 2.13.1 headers (`~/.local/libint2-2.13.1`, `scripts/install-libint.sh`), standalone C++ probe,
  `-O2` and a non-NDEBUG `-O1` ASan/UBSan build, `TennoGmEval::eval_yukawa` / `eval_slater` over
  `T` in {0, 1e-6 ... 5000, 1e4, 1e5, every 2^k (k <= 17) and its two ulp neighbours} x `U` in {1e-7 ... 1e3 every
  quarter decade, every 10^k and neighbours, 50 ... 900} x `m <= 6`, both kernels, `zeta = 1`: 11680 points, 1390 with a
  non-finite slot. With `U > Umax` (1001 ... 1e8) and `T` up to 1e12 added: 12948 points, 2020 non-finite. Every one has
  `T > 1024`; no point with `T <= 1024` is non-finite. On this grid every point with `T > 1024` and
  `U + 2 sqrt(U T) >= 725` is NaN and every one below 690 is finite (no grid point lies between). The
  region is NOT only `U >= 100`: 568 of the 2020 points have `U < 100` (e.g. `U = 1`, `T = 131072`), since only
  `U + zeta R` matters (`U T = zeta^2 R^2 / 4`).
* Bit-identity, C++ evaluator sweep (above): of 12948 points, exactly the 2020 formerly non-finite points change
  (all now finite, 0 non-finite); 0 formerly finite points change. Whole `Engine` blocks (2-, 3-, 4-centre Yukawa and
  Slater, s/p/d shells x 7 centres x zeta in {0.3, 1, 3, 10}, 28592 blocks, FNV-1a over the values):
  `55f65bd5925d35ff` for stock 2.13.1, for the previous 2.13.1 override and for this one. With three far centres added
  ((90,10,0), (-120,0,30), (0,200,0) Bohr) and zeta in {1, 10, 30} (31260 blocks): stock and the previous override have
  51528 non-finite values (`f9c2cf2d9697d90a`), this override 0 (`c21f9e3c033cbad8`).
* Accuracy of the 2020 changed points against a 400-digit mpmath reference (closed form
  `sqrt(pi)/4 (e^{k^2-T} erfc k +- e^{l^2-T} erfc l)` + upward recursion at 400 digits, anchored on `mpmath.quad` for
  `T < 1e-3`): maximum absolute error 6.9e-247 (on this grid all true values are below 1e-230 and the largest the override returns is
  3.7e-234, from `exp(U - 2 sqrt(U T))`, because `e^-T` underflows to 0 for `T > 745`; an independent denser sweep
  reached 7.7e-230 (Yukawa) and 1.7e-228 (Slater, rho = 1), so the grid-independent statement is "below 1e-200", which
  is what the tests assert; the analytic ceiling is about e^-524, times zeta/(2 rho) for Slater). Largest relative error where
  both values exceed 1e-300: 1.3e-12 (300 slots; the Slater difference `G_{m-1} - G_m` cancels).
* ASan/UBSan (`-O1`, asserts on) over the patched header on the whole grid: clean.
* NOT addressed (recorded): `eval_yukawa` with `U < Umin` (1e-7) uses `eval_urr` at ANY `T` and the upward recursion is
  unstable at small `T`. Measured on 2.13.1 against the same reference (`zeta = 1`, `U` from 1e-14 to 1e-7):
  absolute error 1e-14 or better for `T >= 1` (FOR `m <= 6` ONLY), up to 2e-12 at `T = 0.5`, 6e-8 at 0.1, 0.2 at 0.01, 5e5 at 1e-3, 2e25 at 1e-6
  and 9e63 at 1e-12. `T == 0` exactly returns `false` from `eval_urr` and takes the Gaussian fit (3e-7 absolute error
  at `U = 1e-14`, 2e-12 at `U = 1e-8`); no assert fires at `T == 0` (non-NDEBUG, ASan) on 2.13.1. The Slater kernel
  with `U < Umin` is routed to the Gaussian fit (accurate to 5e-14 there). It was left alone in the #439 change; the Fifth change below (issue #443) fixes it. The error grows steeply with `m` (independent 600-digit check, `U` about 1e-14..1e-8, `T = 1`: 1e-15 for
  `m <= 6`, 6e-10 for `m <= 12`, 2.3 for `m <= 20`, 2e23 for `m <= 36`; `T = 3`: 1e-14 for `m <= 12`, 8e-9 for `m <= 20`;
  `T = 5`: 3e-13 for `m <= 20`, 0.3 for `m <= 36`; outputs violate `0 <= G_m <= 1/(2m+1)` up to `T` about 5.6 at high
  `m` and are inf/NaN at `T` about 1e-8, `m = 36`). d/f/g 4-centre derivative integrals reach `m` of 9 to 18, reachable
  from the library with `zeta` below about 0.01 to 0.1 and tight core primitives or collinear pairs. Fixed by the Fifth change.
* Not repaired in 2.13.1 (the 2.7.2 override does): the `T == 0` branch of `eval_G0_and_maybe_Gm1` loses precision
  for `U` in [704.6, 709], where `erfc(sqrt(U))` is subnormal: Slater `m = 0` abs 1.5e-12, rel 6e-8 at `U = 707.25`;
  Yukawa `m = 0` 5.7e-14 at `U = 692`. Documented only.
* Tests (`tenno_table_edge`): `nan_region_grid_is_finite_and_vanishingly_small`, `engine_far_pair_is_finite_on_the_override`,
  `large_u_plus_large_t_is_finite_on_the_override` (now on both overrides), `u_above_umax_is_rejected_only_below_tmax`
  (2.13.1 branch: Gaussian-fit fallback finite for `T <= Tmax`, exactly 0 for `T > Tmax`), and the allocator-fill sweep
  prints an extra `nanreg=` hash.
* How each path picks it up: build.rs hashes `<libint_root>/include/libint2/boys.h` (FNV-1a 64) and, for the stock
  2.13.1 file (the conda-forge `libint-2.13.1-h83f5b4b_0` header that `scripts/install-libint.sh` installs, which is what
  every `ci.yml` job uses), puts `shim/libint2_overrides/2.13.1` first on the shim include path. This change leaves the
  stock hash and the selection untouched, so every build that already got the #433 fix gets this one when the shim
  is recompiled (cargo reruns build.rs because `rerun-if-changed` covers the override `boys.h`). The PyPI wheels
  (`wheels.yml`, libint 2.7.2) are unaffected by this change. A conda-forge package build of ferric is not in this
  checkout (no `conda/` directory), so its path was not checked; if it compiles the shim against the same stock
  2.13.1 `boys.h` the override is selected the same way. A prefix whose `boys.h` is not byte-identical to a pinned stock file
  (distro libint, terf-patched build) gets no override and keeps the NaN; build.rs prints a warning, or fails
  under `FERRIC_REQUIRE_PATCHED_BOYS=1` (see "How it is wired in").

## Only header code is affected

`Engine`, `GenericGmEval` and the evaluators are header-only (`__libint2_engine_inline`). They are instantiated in
`shim/shim.cc` and compiled with the shim's flags. `nm -C libint2.a` (2.7.2) and `nm -DC libint2.so` (2.13.1) show no `GenericGmEval`, `Engine` or
`*_gm_eval` symbols. The generated VRR/HRR kernels in `libint2.a` (2.7.2) or `libint2.so` (2.13.1) do not call the core evaluator,
so **the libint2 library does not need a rebuild**. Recompiling the shim picks the fix up.

## How it is wired in

`build.rs` (`libint2_boys_override`) hashes `<libint_root>/include/libint2/boys.h`, the single libint2 root the shim
compiles and links against. If the hash matches one of the stock files in the table, build.rs puts that version's
directory (`2.7.2/` or `2.13.1/`) **first** on the shim's include path, ahead of `<libint_root>/include`.
`engine.impl.h` includes `<libint2/boys.h>`, and the first `-I` wins. The override is a whole-file replacement, so
any other header gets no override. That covers other versions, the terf-patched prefix from
`wiki/perf-tasks/patches/libint2-terf-*.patch`, and distro builds. In those cases the stock header is used and cargo
prints a warning. Setting `FERRIC_LIBINT2_STOCK_BOYS=1` forces the stock header for A/B timing. That variable is checked
for presence only: any value, including `0`, forces the stock header.

`FERRIC_REQUIRE_PATCHED_BOYS=1` turns every case where the override is not applied (unknown hash, unreadable
`boys.h`, or `FERRIC_LIBINT2_STOCK_BOYS`) into a build error. The message names the reason (for an unknown header, the
hash found and the known stock hashes), a remedy that fits it (point `LIBINT2_PREFIX` at a stock libint2, or unset
`FERRIC_LIBINT2_STOCK_BOYS`), and the opt-out. `FERRIC_ALLOW_UNPATCHED_BOYS=1` is that opt-out: it downgrades the error back to the warning. `ci.yml` (every
job), `wheels.yml` (both wheel builds and the `MALLOC_PERTURB_` job) and the `Dockerfile` set
`FERRIC_REQUIRE_PATCHED_BOYS=1`, because each installs a sha256-pinned libint2 whose `boys.h` is stock (conda-forge
2.13.1, or the `libint-2.7.2-ferric-small-1` export, whose `boys.h` has the stock 2.7.2 sha256 above). A new libint2
pin or a stale cache there therefore fails the build instead of silently dropping the fixes. A plain local or sdist
build does not set it, so a libint2 the table does not list (another version, a distro or from-source build, the
terf-patched prefix) still builds with the warning. The decision is `build_support/boys_policy.rs`, shared by
`build.rs` and `tests/boys_override_policy.rs`; the test rebuilds each stock `boys.h` by reverse-applying the
committed `.patch` and checks it selects its override.

The terf patch adds `terf_gm_eval`, which also goes through `GenericGmEval` with `CoreEvalScratch`. So it has the
same per-call copy, and this override does not fix it.

License: libint is LGPL-3.0-or-later. This file keeps the upstream notice and adds a modification notice.

## Fifth change (both versions): Yukawa `U < Umin` at small `T` (issue #443)

`eval_yukawa` sends `U = zeta^2 / (4 rho) < Umin = 1e-7` to `eval_urr` at any `T`. That is the upward recursion
`G_{m+1} = [(2m+1) G_m + 2U G_{m-1} - e^-T] / (2T)`, which multiplies the rounding error of `G_0` by about
`prod (2m+1)/(2T)` (independent of `U`): O(1) at `m = 20`, `T = 1`, inf/NaN at `m = 36`, `T = 1e-8`, and
`G_m > 1/(2m+1)` up to `T` about 5.6 at high `m`. At `T == 0` 2.7.2 divides by zero and 2.13.1 takes a Gaussian fit
(3e-7 error at `U = 1e-14`). At `mmax == 0` there is no recursion, but the closed form `G_0 = pfac (erfc_k - erfc_l) /
sqrt(T)` loses digits like `1e-16 / sqrt(T)`: for coincident bra and ket centres, where rounding leaves `T` near 1e-32, it is
O(1) noise. Measured on libint 2.13.1: H2/STO-3G `(10|10)` at `zeta = 1e-8` differed from Coulomb by 4e-3 (expected 1e-8).

The override adds `yukawa_urr_unusable(T, U, mmax)` (`U < Umin` and `T < 0.5 mmax + 6`, or `T < 1e-2` at `mmax == 0`) and
`eval_yukawa_quadrature`, which evaluates `G_m = int_0^1 t^{2m} exp(U(1-t^-2) - T t^2) dt` by 24-point Gauss-Legendre
on graded panels (a ratio-4 ladder from `max(0.01 sqrt(U), 1e-20)` to 1e-3 for the `exp(-U/t^2)` boundary layer at
`t ~ sqrt(U)`, then 12 panels up to 1). Every term is positive, so there is no cancellation. Everywhere else
`eval_yukawa` is unchanged: `crates/ferric-integrals/tests/yukawa_urr_small_t.rs` compares the library with an in-binary port of the stock
recursion to 2e-15 relative on `U < Umin`, `T >= 0.5 mmax + 6` (or `T >= 1e-2` at `mmax == 0`), and independently against mpmath
(`REF_STOCK`, mmax 1..36, T from 0.7x the threshold to 1000: worst abs 3.7e-15, worst rel 6.3e-13, at the threshold for mmax = 36;
bars abs 1e-13 / rel 1e-11), and checks the `U >= Umin` table region against
50-digit values and the bounds `0 <= G_m <= 1/(2m+1)`. Those gate CI. A bit-level golden (FNV-1a of the same grid: `84f833f18f8c180e`, n=168859 on
2.13.1; `da400bb5bddb82a4`, n=55579 on 2.7.2, whose Tenno table stops at `mmax` 20) is `#[ignore]`d because libm and compiler
contraction change low bits on other machines; it was identical with the fallback disabled and enabled on the dev box. `eval_slater` is untouched (2.7.2 still throws for
`U < Umin`, `T <= Tmax`; 2.13.1 goes to its Gaussian fit).

Accuracy against 50-digit mpmath values (`U = 1e-14` and `1e-9`, `T` in 1e-8..20, `m <= 36`; the unmodified region is covered separately out to `T = 1000`, see above): at most 1e-15 absolute,
and `0 <= G_m <= 1/(2m+1)` holds. The test also checks the `U -> 0` limit `G_m -> F_m(T)` (Boys function), and
water/cc-pVDZ four-centre ERIs at `zeta = 1e-6` against Coulomb.

Cost of the quadrature path (2.13.1, per `eval_yukawa` call): about 8 to 13 microseconds, against about 90 ns for the old
recursion and 1 to 3.6 microseconds for the table. A small `zeta` (below about 0.07 for first-row cc-pVDZ) routes every
quartet through it, roughly 3 to 4 times slower than the table path. No production caller uses the Yukawa operator today.
