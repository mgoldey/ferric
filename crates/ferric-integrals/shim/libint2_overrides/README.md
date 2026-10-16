# libint2 header overrides

Each `<version>/libint2/boys.h` here is a patched copy of that libint2 version's stock `include/libint2/boys.h`, and
`<version>/libint2-<version>-gmeval-no-copy.patch` is the exact diff from stock to that copy. `patch -p1` applied
inside an install prefix of that version reproduces the file byte for byte.

| Directory | Upstream | Stock `boys.h` sha256 | FNV-1a 64 (build.rs) |
|-----------|----------|-----------------------|----------------------|
| `2.7.2/` | `libint-2.7.2-mpqc4.tgz` | `880592b9026352c871c96877a43788edd22edb0b1057ab5a73657c40b3a61505` | `0x1d51ac52f2076ad8` |
| `2.13.1/` | conda-forge `libint-2.13.1-h83f5b4b_0.conda` (`scripts/install-libint.sh`) | `890ab0261a2be2517d3d4cedfd7d07e2a1c6be7b6f057dc7ea2ea418e17067c1` | `0x0f6f0c23194e18ad` |

The fix is the same in both. The sections below describe it with the 2.7.2 names. In 2.13.1, one evaluator,
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

### Sibling defect, not fixed here

libint 2.7.2's `TennoGmEval` reads out of range at `T == Tmax` and `U == Umax` (Yukawa/SlaterGeminal on the wheel):
issue #436. Same class, different table, different file; this override does not touch it.

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
