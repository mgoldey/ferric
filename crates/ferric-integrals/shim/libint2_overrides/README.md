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
