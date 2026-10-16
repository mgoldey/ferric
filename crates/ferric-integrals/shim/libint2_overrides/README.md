# libint2 header overrides

Each `<version>/libint2/boys.h` here is a patched copy of that libint2 version's stock `include/libint2/boys.h`, and
`<version>/libint2-<version>-gmeval-no-copy.patch` is the exact diff from stock to that copy. `patch -p1` applied
inside an install prefix of that version reproduces the file byte for byte.

| Directory | Upstream | Stock `boys.h` sha256 | FNV-1a 64 (build.rs) |
|-----------|----------|-----------------------|----------------------|
| `2.7.2/` | `libint-2.7.2-mpqc4.tgz` | `880592b9026352c871c96877a43788edd22edb0b1057ab5a73657c40b3a61505` | `0x1d51ac52f2076ad8` |
| `2.13.1/` | conda-forge `libint-2.13.1-h83f5b4b_0.conda` (`scripts/install-libint.sh`) | `890ab0261a2be2517d3d4cedfd7d07e2a1c6be7b6f057dc7ea2ea418e17067c1` | `0x0f6f0c23194e18ad` |

Override sha256 (what `patch -p1` of the committed `.patch` onto the stock file reproduces): 2.7.2
`6817d7ab53d801f868778cd14a446fcc5cb2ea8252e16e42a02815f6743e29b5`, 2.13.1
`658cffae31ce33b4ecc5ce6846f7feeac1ab63dfe8271286b654f1dad79fa868`.

The #433 fix is the same in both. The sections below describe it with the 2.7.2 names. The `TennoGmEval` changes
(issue #436, section "Third change") exist ONLY in the 2.7.2 override; 2.13.1 already has them in stock form. In 2.13.1, one evaluator,
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
uninitialised memory; build.rs warns that the fix is NOT applied.

### Evidence and its limits

- Tests: `crates/ferric-integrals/tests/boys_chebyshev_table_edge.rs` (direct evaluator pin via `scf_boys_fm`,
  exact-integer-geometry reproducer, point-site reproducer, allocator-fill sweep). `tests/boys_chebyshev_manifest.rs`
  is a PRINT-ONLY helper for before/after bit comparisons by hand, not a guard.
- Survey of existing tests (an instrumented STOCK 2.13.1 build counting evaluations at exactly 117.0, default
  allocator, 400 s timeout per binary): 256 test binaries of ferric-integrals, ferric-pbc, ferric-mp2, ferric-rpa;
  218 passed, 26 failed with 0 hits (environmental: 25 re-run serially all pass, 0 hits), 12 timed out. Of those 12,
  only 3 were measured afterwards (per test: 39 tests, 0 hits); **9 timed-out binaries remain unmeasured**. The one
  known real hit is the #421 ECP k-point test (80 evaluations at 117.0 per run; failing under `MALLOC_PERTURB_` 1 and
  170 before the fix, passing after).
- Not checked: aarch64 and FMA-contracting builds (whether a given geometry lands on exactly 117.0 depends on
  rounding, which is why the integer-geometry test exists).

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
* `eval_yukawa` / `eval_slater` throw `std::invalid_argument` for `U > Umax` (or NaN), and `eval_slater` for
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
  engine=de36c9077af007c0 n=296` on 2.7.2+override and `edge=2acf9250b8e24a57 engine=0ed77033630946fd` on 2.13.1.
* Accuracy of what changed: at every one of the 210 changed points checked against a 500-digit mpmath evaluation of the
  closed form, abs error <= 4e-19, rel <= 3e-11. At tiny `T` and large `U` the stock-form branch (erfc normal) keeps its
  inherent `x^2 eps` amplification of the `exp(x*x)` rounding: worst measured relative error of `Gm[0]` 4e-10 (T = 1e-6,
  U = 570), unchanged by this patch.
* libint 2.13.1 (stock, and this repo's 2.13.1 override) is NOT fixed for the overflow: `eval_yukawa` and `eval_slater`
  return NaN in the `T > Tmax` upward recursion once `U + 2 sqrt(U T)` exceeds ~709 (measured: 84 of 12640 grid points,
  T > 1024 with U >= 100, both kernels). The true values there are < 1e-270, so no physical integral is affected unless a
  primitive that large is not screened; recorded, not patched. Its table indices are clean at T == 1024 and U == 1000
  (ASan/UBSan probe, both edges and one ulp each side).
* A standalone ASan/UBSan probe of the patched header over the edge set (`T` and `U` on, one ulp each side of, and away
  from the edges, both kernels, m <= 6) and the full safe grid is clean; the same probe against stock 2.7.2 aborts or
  reports the overflow at the first edge point.

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
prints a warning. Setting `FERRIC_LIBINT2_STOCK_BOYS=1` forces the stock header for A/B timing.

The terf patch adds `terf_gm_eval`, which also goes through `GenericGmEval` with `CoreEvalScratch`. So it has the
same per-call copy, and this override does not fix it.

License: libint is LGPL-3.0-or-later. This file keeps the upstream notice and adds a modification notice.
