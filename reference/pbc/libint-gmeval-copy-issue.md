# Draft upstream issue: evaleev/libint

**Title:** `GenericGmEval::eval` copies the erf/erfc(/erfx) evaluator per primitive quartet: atomic refcount on the
Boys singleton and heap allocation, so erf/erfc integrals scale poorly across threads

(Draft, 2026-09-29. Not filed. The code was verified in 2.7.2 and 2.13.1. For master, a fetch on 2026-09-29 by a
previous session showed `GmEvalFunction(*this)` still present; re-check before filing.)

## Summary

`GenericGmEval<GmEvalFunction>::eval` (`include/libint2/boys.h`) invokes the wrapped core evaluator through a
temporary copy:

```cpp
template <typename Real, typename... ExtraArgs>
void eval(Real* Gm, Real rho, Real T, int mmax, ExtraArgs... args) const {
  assert(mmax <= mmax_);
  (GmEvalFunction(*this))(Gm, rho, T, mmax, std::forward<ExtraArgs>(args)...);
}
```

`Engine::compute2` calls this once per surviving primitive quartet for `erf_coulomb` and `erfc_coulomb` (2.7.2), and
for `erfx_coulomb` in 2.13.x. The copied evaluators hold:

- `std::shared_ptr<const FmEval_Chebyshev7<double>> fm_eval_`. It points to the process-wide
  `FmEval_Chebyshev7::instance()`, so every engine in every thread shares one control block. Each copy performs an
  atomic increment and each destruction an atomic decrement on that one cache line.
- For erfc (2.7.2) and erfx (2.13.x), a `std::vector<Real>` scratch inherited from `detail::CoreEvalScratch`, so each
  copy also costs `operator new` + `memmove` + `operator delete`.

With N threads, these per-primitive atomics on a single line become a throughput ceiling. Plain Coulomb is unaffected
because `FmEval_Chebyshev7::eval` is called through a const pointer.

GCC 12 `-O2` does not elide the copy. Disassembly of a TU that instantiates
`GenericGmEval<erfc_coulomb_gm_eval<double>>::eval` shows `operator new`, `memmove`, `lock add`, `_M_release`
(`lock xadd`) and `operator delete` on every call.

## Measurement

Environment: libint 2.7.2 (mpqc4 tarball, `LIBINT_MAX_AM 6`, contracted ints), GCC 12.4 `-O2`, i7-6800K (6C/12T
Broadwell-E). The test calls `Engine::compute` repeatedly on fixed shell sets, one pre-built `Engine` per thread with
no sharing, 20000 calls per thread, best of 3:

| workload | 6 threads, 1 process: per-call time vs 1 thread | 6 processes × 1 thread, run concurrently |
|---|---|---|
| erfc 3-centre (`xs_xx`, cc-pVDZ C / 1-primitive aux) | **3.28x** | 1.05-1.07x |
| erfc 3-centre with a Gaussian nucleus (short-range hcore) | **4.07x** | 1.00-1.08x |
| FP-only control loop | 1.13x | – |

The same work in separate processes does not slow down, so the cost comes from state shared within a process, not
from the hardware (frequency, caches, memory bandwidth).

## Proposed fix (arithmetic unchanged)

1. Make the evaluators' call operators `const` and keep their scratch on the stack.
   `FmEval_Chebyshev7` already caps `mmax` at `cheb_table_mmax` (40), so a fixed `Real[cheb_table_mmax+1]` or
   similar is enough; a heap fallback can cover `LIBINT_USER_DEFINED_REAL`.
2. Call through a const reference in `GenericGmEval::eval`:
   `static_cast<const GmEvalFunction&>(*this)(Gm, rho, T, mmax, args...)`.

This works for the erf, erfc and erfx evaluators. `r12_xx_K_gm_eval<1>` has the same vector-plus-shared_ptr pattern
and can get the same treatment. The operations run in the same order on the same values; only the scratch buffer's
address changes. So integrals are bit-identical. It is also race-free: the const call reads only the immutable Boys
table. Keeping the member scratch and `const_cast`-ing instead would introduce a data race, because `Engine` copies
share `core_eval_pack_` by `shared_ptr`.

A patch against 2.7.2 that does this for erf and erfc is attached (`libint2-2.7.2-gmeval-no-copy.patch`). It adds a
trait `detail::gm_eval_call_by_const_ref` so that only the two converted evaluators take the no-copy path. Upstream
could skip the trait and convert all evaluators.

Verification done downstream:
- A standalone TU hashed the `Gm` output over a grid (mmax 0..28, rho 1e-3..1e4, T 0..200, omega ∈ {0, 0.1, 0.5,
  2, 11}) for erf and erfc. The stock and patched headers gave the same hash (`10ad81a93f1d84c9`). The patched
  `main` has no `lock` instructions and no `operator new`/`delete` calls in the loop.
- Full-engine checksums: [to be filled in from ferric's contention example before and after].
- Scaling after the patch: [to be filled in].

## Related

- CHANGES 2.7.0: "fixed excessive lock contention when computing 1-e Coulomb ("nuclear") ints on many threads". That
  entry is the same class of defect in a different path.
- psi4 #2491 (libint2 one-electron integrals on a grid scaled 1.4x at 8 cores).
