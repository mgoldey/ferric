<!-- Filed as https://github.com/evaleev/libint/issues/431 on 2026-09-30 -->
**Title:** Per-call evaluator copy in GenericGmEval::eval limits erf/erfc thread scaling

---

## Summary

`GenericGmEval<GmEvalFunction>::eval` calls the wrapped core evaluator through a temporary copy of it. `Engine` calls
`eval` once per primitive quartet (2-body) or primitive pair (1-body), so for the operators that go through
`GenericGmEval` (erf/erfc/erfx Coulomb, the erf/erfc/erfx nuclear operators, `q_gau`/`op_q_gau_op`, `r12`) each
primitive costs:

- an atomic increment and decrement of the `std::shared_ptr` to the process-wide `FmEval_Chebyshev7` instance, which
  every engine in every thread shares, so all threads write to the same cache line; and
- for evaluators that carry a `std::vector` scratch (`erfx_coulomb_gm_eval`, `q_gau_gm_eval`,
  `r12_xx_K_gm_eval<Real,1>`; `erfc_coulomb_gm_eval` in 2.7.x), a heap allocation, copy and free.

With one `Engine` per thread, stock erf/erfc integrals slow down 4.5-5.7x per call at 6 threads. The same work in 6
separate processes slows down only 1.2-1.3x, which matches plain Coulomb. Calling the evaluator through a const
reference with the scratch on the stack removes the difference. The integrals come out bit-identical.

## Affected versions

- 2.7.2: measured below.
- 2.13.1 (latest release): same code; measurements are in the section "Results on 2.13.1" below.
- master @ [`f47820b`](https://github.com/evaleev/libint/tree/f47820bd2b8eb1b71b3076249a8f516f1a5316e4) (2026-05-20,
  current as of 2026-09-30): same code; `q_gau_gm_eval` (#408) adds a fourth evaluator with a vector scratch.

## Where

[`include/libint2/boys.h` L1963-L1967](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L1963-L1967):

```cpp
  template <typename Real, typename... ExtraArgs>
  void eval(Real* Gm, Real rho, Real T, int mmax, ExtraArgs... args) const {
    assert(mmax <= mmax_);
    (GmEvalFunction(*this))(Gm, rho, T, mmax, std::forward<ExtraArgs>(args)...);
  }
```

The evaluators that are copied (all on master @ `f47820b`):

| evaluator | copy costs | used by (`engine.h` `core_eval_type`) |
|---|---|---|
| [`erf_coulomb_gm_eval`](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L2077-L2109) | `shared_ptr` refcount | `erf_coulomb`, `erf_nuclear` |
| [`erfx_coulomb_gm_eval`](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L2115-L2158) | `shared_ptr` refcount + `std::vector` alloc/copy/free | `erfc_coulomb`, `erfx_coulomb`, `erfc_nuclear`, `erfx_nuclear` |
| [`q_gau_gm_eval`](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L2168-L2229) | `shared_ptr` refcount + `std::vector` alloc/copy/free | `q_gau`, `op_q_gau_op` |
| [`r12_xx_K_gm_eval<Real,1>`](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L2045-L2074) | `shared_ptr` refcount + `std::vector` alloc/copy/free | `r12` |
| `delta_gm_eval` | nothing (empty class) | `delta` |

Call sites: 2-body
[`engine.impl.h` L1540-L1573](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/engine.impl.h#L1540-L1573)
(once per primitive quartet in `compute2`); 1-body
[`engine.impl.h` L1169-L1215](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/engine.impl.h#L1169-L1215)
(once per primitive pair and charge center). Plain `coulomb` and `nuclear` call `FmEval_Chebyshev7::eval` through a
const pointer and are not affected.

## Mechanism

The copy appears to exist so that each call gets private scratch. The comment on
[`detail::CoreEvalScratch`](https://github.com/evaleev/libint/blob/f47820bd2b8eb1b71b3076249a8f516f1a5316e4/include/libint2/boys.h#L1725)
reads "some evaluators need thread-local scratch". But the copy also copies `fm_eval_`, a
`std::shared_ptr<const FmEval_Chebyshev7<double>>` that points to the one instance returned by
`FmEval_Chebyshev7::instance()`. All threads therefore do a `lock add` / `lock xadd` pair on one control block for each
primitive quartet. With N threads that cache line bounces between cores and caps throughput.

GCC 12 at `-O2` does not elide the copy. The disassembly of `GenericGmEval<erfc_coulomb_gm_eval<double>>::eval`
(2.7.2) shows `operator new`, `memmove`, `lock add`, `_M_release` (`lock xadd`) and `operator delete` on every call.
Stock erf copies only the `shared_ptr`, with no allocation, and it degrades *more* than erfc in the table below. That
points to the atomic refcount, not the allocator, as the main cost.

CHANGES for 2.7.0 lists "fixed excessive lock contention when computing 1-e Coulomb ("nuclear") ints on many threads",
which appears to be the same class of problem on a different path.

## Reproducer

A standalone program that links only libint2. It builds three carbon-like centers, each with a contracted s shell
(9 primitives) and a contracted p shell (4 primitives). Every thread runs the same 1296 shell quartets through its own
`Engine`.

<details><summary><code>mwe.cc</code> (140 lines)</summary>

```cpp
// Reproducer: erf/erfc two-electron integrals scale poorly across threads,
// while plain Coulomb scales normally.
//
// Each worker thread owns its own Engine and runs the same fixed list of
// shell quartets `calls` times. Nothing is shared between threads except what
// libint2 itself shares. Prints the mean wall ns per Engine::compute call,
// the per-thread values, and a checksum of the integrals.
//
// Build: g++ -std=c++17 -O2 -pthread -I/usr/include/eigen3 -I$PREFIX/include \
//          mwe.cc -L$PREFIX/lib -lint2 -o mwe
// Usage: mwe <coulomb|erf|erfc> <nthreads> <calls>
//   e.g. ./mwe erfc 1 200000; ./mwe erfc 6 200000;
//        for i in 1 2 3 4 5 6; do ./mwe erfc 1 200000 & done; wait

#include <libint2.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

using libint2::BraKet;
using libint2::Engine;
using libint2::Operator;
using libint2::Shell;

// Carbon-like contracted s and p shells (cc-pVDZ-style exponents, several
// primitives each => many primitive quartets per call, so the per-quartet
// evaluator copy dominates).
std::vector<Shell> make_shells() {
  const libint2::svector<double> e_s = {6665.0, 1000.0, 228.0, 64.71, 21.06, 7.495, 2.797, 0.5215, 0.1596};
  const libint2::svector<double> c_s = {0.000692, 0.005329, 0.027077, 0.101718, 0.27474,
                                   0.448564, 0.285074, 0.015204, -0.003191};
  const libint2::svector<double> e_p = {9.439, 2.002, 0.5456, 0.1517};
  const libint2::svector<double> c_p = {0.038109, 0.20948, 0.508557, 0.468842};
  std::vector<Shell> shells;
  const double centers[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 2.2}, {1.9, 0.4, -0.8}};
  for (const auto& c : centers) {
    shells.push_back(Shell{e_s, {{0, false, c_s}}, {{c[0], c[1], c[2]}}});
    shells.push_back(Shell{e_p, {{1, false, c_p}}, {{c[0], c[1], c[2]}}});
  }
  return shells;
}

Engine make_engine(const char* op, size_t max_nprim, int max_l) {
  const double omega = 1.0;  // bohr^-1
  const double eps = std::numeric_limits<double>::epsilon();
  if (std::strcmp(op, "erfc") == 0)
    return Engine(Operator::erfc_coulomb, max_nprim, max_l, 0, eps, omega);
  if (std::strcmp(op, "erf") == 0)
    return Engine(Operator::erf_coulomb, max_nprim, max_l, 0, eps, omega);
  return Engine(Operator::coulomb, max_nprim, max_l, 0, eps);
}

uint64_t mix(uint64_t h, double v) {
  uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  h ^= b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <coulomb|erf|erfc> <nthreads> <calls>\n", argv[0]);
    return 2;
  }
  const char* op = argv[1];
  const int nthreads = std::atoi(argv[2]);
  const long calls = std::atol(argv[3]);

  libint2::initialize();
  const auto shells = make_shells();
  size_t max_nprim = 0;
  int max_l = 0;
  for (const auto& s : shells) {
    max_nprim = std::max(max_nprim, s.nprim());
    max_l = std::max(max_l, s.contr[0].l);
  }
  // A fixed quartet list: every (i j | k l) over the 6 shells.
  std::vector<std::array<int, 4>> quartets;
  const int n = static_cast<int>(shells.size());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l) quartets.push_back({i, j, k, l});

  std::vector<double> ns_per_call(nthreads, 0.0);
  std::vector<uint64_t> hashes(nthreads, 0);
  std::atomic<int> ready{0};

  auto worker = [&](int t) {
    Engine engine = make_engine(op, max_nprim, max_l);  // one engine per thread
    const auto& buf = engine.results();
    ready.fetch_add(1);
    while (ready.load() < nthreads) {
    }  // start together
    uint64_t h = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long c = 0; c < calls; ++c) {
      const auto& q = quartets[c % quartets.size()];
      engine.compute(shells[q[0]], shells[q[1]], shells[q[2]], shells[q[3]]);
      if (c < static_cast<long>(quartets.size()) && buf[0] != nullptr) {
        const size_t len = shells[q[0]].size() * shells[q[1]].size() * shells[q[2]].size() *
                           shells[q[3]].size();
        for (size_t x = 0; x < len; ++x) h = mix(h, buf[0][x]);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    ns_per_call[t] =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls);
    hashes[t] = h;
  };

  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  double mean = 0.0;
  for (double v : ns_per_call) mean += v / nthreads;
  std::printf("op %s threads %d calls/thread %ld quartets %zu: mean %.1f ns/call", op, nthreads,
              calls, quartets.size(), mean);
  for (double v : ns_per_call) std::printf(" %.1f", v);
  std::printf("  checksum %016llx\n", static_cast<unsigned long long>(hashes[0]));
  for (int t = 1; t < nthreads; ++t)
    if (hashes[t] != hashes[0]) std::printf("WARNING: thread %d checksum differs\n", t);

  libint2::finalize();
  return 0;
}
```

</details>

## Results on 2.7.2

libint 2.7.2 (`LIBINT_MAX_AM 6`, contracted integrals), GCC 12.4 `-O2`, Intel i7-6800K (6 cores / 12 threads,
Broadwell-E), omega = 1.0. Values are the mean wall ns per `Engine::compute` call, averaged over threads. "Patched" is
the same program built against headers with the fix described below.

| operator | 1 thread | 6 threads | 6 separate 1-thread processes | threads / 1 | processes / 1 |
|---|---|---|---|---|---|
| coulomb | 77,009 | 96,529 | ~100,700 | 1.25x | 1.31x |
| erf (stock) | 91,742 | 518,823 | ~118,000 | **5.66x** | 1.29x |
| erfc (stock) | 131,619 | 587,367 | ~160,700 | **4.46x** | 1.22x |
| erf (patched) | 80,648 | 102,164 | ~103,000 | 1.27x | 1.28x |
| erfc (patched) | 100,696 | 123,964 | ~127,000 | 1.23x | 1.26x |

- Stock and patched builds give the same integral checksum for every operator.
- The ~1.25x that remains after the patch also shows up for Coulomb and for separate processes, so it is the cost of
  loading all cores on this CPU. With the patch, erf and erfc scale like Coulomb.
- The fix also helps single-threaded runs: the 1-thread erfc time drops by 23% (131.6 to 100.7 µs per call) because the
  per-primitive allocation is gone.

The same effect appears in a production code (one `Engine` per thread, 3-center erfc integrals over cc-pVDZ carbon
shells with a 1-primitive auxiliary shell): 3.28x per-call slowdown at 6 threads, against 1.05-1.07x for 6 concurrent
1-thread processes. A plain floating-point control loop on the same machine gave 1.13x.

## Results on 2.13.1

Same reproducer and machine, built against conda-forge libint 2.13.1 (`libint2.so`), mean ns per call. In 2.13.x,
`erfc_coulomb` uses `erfx_coulomb_gm_eval`, which has the vector scratch. Measured 2026-09-30 with some desktop load
on the machine, so the 6-thread and 6-process columns carry about ±25% noise; the 1-thread column is steady.

| operator | 1 thread | 6 threads | 6 separate 1-thread processes | threads / 1 | processes / 1 |
|---|---|---|---|---|---|
| coulomb | 79,289 | 97,338 | 100,657 | 1.23x | 1.27x |
| erf (stock) | 94,364 | 510,193 | 119,953 | 5.41x | 1.27x |
| erfc (stock) | 135,760 | 581,771 | 162,311 | 4.28x | 1.20x |
| erf (patched) | 85,009 | 108,640 | 107,311 | 1.28x | 1.26x |
| erfc (patched) | 100,709 | 120,698 | 126,131 | 1.20x | 1.25x |

Checksums of all integrals are identical between stock and patched for every operator. The same session re-measured
2.7.2 at the same load: erf 5.61x / erfc 4.37x stock, 1.36x / 1.21x patched, and single-thread times within 2% of
2.13.1 for every operator.

## Proposed fix

Invoke the evaluator through a const reference, and make the four evaluators' call operators `const` with their Boys
scratch on the stack. A small `detail::FmStackScratch` holds 64 values inline and falls back to the heap for larger
`mmax`, so no cap on `mmax` is assumed; for example, `LIBINT_USER_DEFINED_REAL` builds use `FmEval_Reference`. The
three `CoreEvalScratch` specializations that only existed to be copied are removed, along with their redundant forward
declarations; `boys_fwd.h` already declares these evaluators. `GaussianGmEval`'s `CoreEvalScratch` is not touched.

Diff against master @ `f47820b` (applies with `git apply`; 1 file, +51/-62):

```diff
diff --git a/include/libint2/boys.h b/include/libint2/boys.h
index fcd11ad..439f7e5 100644
--- a/include/libint2/boys.h
+++ b/include/libint2/boys.h
@@ -1963,7 +1963,11 @@ struct GenericGmEval : private GmEvalFunction {
   template <typename Real, typename... ExtraArgs>
   void eval(Real* Gm, Real rho, Real T, int mmax, ExtraArgs... args) const {
     assert(mmax <= mmax_);
-    (GmEvalFunction(*this))(Gm, rho, T, mmax, std::forward<ExtraArgs>(args)...);
+    // call through a const reference: copying the evaluator here costs an
+    // atomic refcount update on the shared Boys engine (and, for evaluators
+    // with a scratch vector, a heap allocation) per call
+    static_cast<const GmEvalFunction&>(*this)(
+        Gm, rho, T, mmax, std::forward<ExtraArgs>(args)...);
   }
 
   /// @return the maximum value of m for which the \f$ G_m(\rho, T) \f$ can be
@@ -1978,42 +1982,28 @@ struct GenericGmEval : private GmEvalFunction {
   Real precision_;
 };
 
-// these Gm engines need extra scratch data
-namespace os_core_ints {
-template <typename Real, int K>
-struct r12_xx_K_gm_eval;
-template <typename Real>
-struct erfx_coulomb_gm_eval;
-template <typename Real>
-struct q_gau_gm_eval;
-}  // namespace os_core_ints
-
 namespace detail {
-/// r12_xx_K_gm_eval<1> needs extra scratch data
-template <typename Real>
-struct CoreEvalScratch<os_core_ints::r12_xx_K_gm_eval<Real, 1>> {
-  std::vector<Real> Fm_;
-  CoreEvalScratch(const CoreEvalScratch&) = default;
-  CoreEvalScratch(CoreEvalScratch&&) = default;
-  // need to store Fm(T) for m = 0 .. mmax+1
-  explicit CoreEvalScratch(int mmax) { Fm_.resize(mmax + 2); }
-};
-/// erfx_coulomb_gm_eval needs extra scratch data
-template <typename Real>
-struct CoreEvalScratch<os_core_ints::erfx_coulomb_gm_eval<Real>> {
-  std::vector<Real> Fm_;
-  CoreEvalScratch(const CoreEvalScratch&) = default;
-  CoreEvalScratch(CoreEvalScratch&&) = default;
-  // need to store Fm(T) for m = 0 .. mmax
-  explicit CoreEvalScratch(int mmax) { Fm_.resize(mmax + 1); }
-};
-/// q_gau_gm_eval needs extra scratch data
-template <typename Real>
-struct CoreEvalScratch<os_core_ints::q_gau_gm_eval<Real>> {
-  std::vector<Real> Fm_;
-  CoreEvalScratch(const CoreEvalScratch&) = default;
-  CoreEvalScratch(CoreEvalScratch&&) = default;
-  explicit CoreEvalScratch(int mmax) { Fm_.resize(mmax + 1); }
+/// Boys function scratch for core evaluators that call the Boys engine more
+/// than once per invocation; lives on the caller's stack, so that evaluators
+/// can be invoked through a const reference from any number of threads
+template <typename Real, int StackSize = 64>
+class FmStackScratch {
+ public:
+  explicit FmStackScratch(int size) : data_(stack_) {
+    if (size > StackSize) {
+      heap_.resize(size);
+      data_ = heap_.data();
+    }
+  }
+  FmStackScratch(const FmStackScratch&) = delete;
+  FmStackScratch& operator=(const FmStackScratch&) = delete;
+  Real* data() { return data_; }
+  Real& operator[](int i) { return data_[i]; }
+
+ private:
+  Real stack_[StackSize];
+  std::vector<Real> heap_;  // used only if size > StackSize
+  Real* data_;
 };
 }  // namespace detail
 
@@ -2043,9 +2033,7 @@ template <typename Real, int K>
 struct r12_xx_K_gm_eval;
 
 template <typename Real>
-struct r12_xx_K_gm_eval<Real, 1>
-    : private detail::CoreEvalScratch<r12_xx_K_gm_eval<Real, 1>> {
-  typedef detail::CoreEvalScratch<r12_xx_K_gm_eval<Real, 1>> base_type;
+struct r12_xx_K_gm_eval<Real, 1> {
   typedef Real value_type;
 
 #ifndef LIBINT_USER_DEFINED_REAL
@@ -2054,18 +2042,20 @@ struct r12_xx_K_gm_eval<Real, 1>
   using FmEvalType = libint2::FmEval_Reference<scalar_type>;
 #endif
 
-  r12_xx_K_gm_eval(unsigned int mmax, Real precision) : base_type(mmax) {
+  r12_xx_K_gm_eval(unsigned int mmax, Real precision) {
     fm_eval_ = FmEvalType::instance(mmax + 1, precision);
   }
-  void operator()(Real* Gm, Real rho, Real T, int mmax) {
-    fm_eval_->eval(&base_type::Fm_[0], T, mmax + 1);
+  void operator()(Real* Gm, Real rho, Real T, int mmax) const {
+    // need Fm(T) for m = 0 .. mmax+1
+    detail::FmStackScratch<Real> Fm(mmax + 2);
+    fm_eval_->eval(Fm.data(), T, mmax + 1);
     auto T_plus_m_plus_one = T + 1.0;
-    Gm[0] = T_plus_m_plus_one * base_type::Fm_[0] - T * base_type::Fm_[1];
+    Gm[0] = T_plus_m_plus_one * Fm[0] - T * Fm[1];
     auto minus_m = -1.0;
     T_plus_m_plus_one += 1.0;
     for (auto m = 1; m <= mmax; ++m, minus_m -= 1.0, T_plus_m_plus_one += 1.0) {
-      Gm[m] = minus_m * base_type::Fm_[m - 1] +
-              T_plus_m_plus_one * base_type::Fm_[m] - T * base_type::Fm_[m + 1];
+      Gm[m] = minus_m * Fm[m - 1] +
+              T_plus_m_plus_one * Fm[m] - T * Fm[m + 1];
     }
   }
 
@@ -2113,9 +2103,7 @@ struct erf_coulomb_gm_eval {
 /// @note need extra scratch for Boys function values,
 ///       since need to call Boys engine twice
 template <typename Real>
-struct erfx_coulomb_gm_eval
-    : private detail::CoreEvalScratch<erfx_coulomb_gm_eval<Real>> {
-  typedef detail::CoreEvalScratch<erfx_coulomb_gm_eval<Real>> base_type;
+struct erfx_coulomb_gm_eval {
   typedef Real value_type;
 
 #ifndef LIBINT_USER_DEFINED_REAL
@@ -2124,16 +2112,17 @@ struct erfx_coulomb_gm_eval
   using FmEvalType = libint2::FmEval_Reference<scalar_type>;
 #endif
 
-  erfx_coulomb_gm_eval(unsigned int mmax, Real precision) : base_type(mmax) {
+  erfx_coulomb_gm_eval(unsigned int mmax, Real precision) {
     fm_eval_ = FmEvalType::instance(mmax, precision);
   }
   void operator()(Real* Gm, Real rho, Real T, int mmax, Real omega, Real lambda,
-                  Real sigma) {
+                  Real sigma) const {
+    detail::FmStackScratch<Real> Fm(mmax + 1);
     // \lambda \mathrm{erf}(\omega r) + \sigma \mathrm{erfc}(\omega r) = \sigma
     // - (\sigma - \lambda) \mathrm{erf}(\omega r)
     if (sigma != 0) {
-      fm_eval_->eval(&base_type::Fm_[0], T, mmax);
-      for (auto m = 0; m <= mmax; ++m) Gm[m] = sigma * base_type::Fm_[m];
+      fm_eval_->eval(Fm.data(), T, mmax);
+      for (auto m = 0; m <= mmax; ++m) Gm[m] = sigma * Fm[m];
     } else {
       std::fill(Gm, Gm + mmax + 1, Real{0});
     }
@@ -2141,14 +2130,14 @@ struct erfx_coulomb_gm_eval
     if (omega > 0 && sigma_minus_lambda != 0) {
       auto omega2 = omega * omega;
       auto omega2_over_omega2_plus_rho = omega2 / (omega2 + rho);
-      fm_eval_->eval(&base_type::Fm_[0], T * omega2_over_omega2_plus_rho, mmax);
+      fm_eval_->eval(Fm.data(), T * omega2_over_omega2_plus_rho, mmax);
 
       using std::sqrt;
       auto ooversqrto2prho_exp_2mplus1 =
           sqrt(omega2_over_omega2_plus_rho) * sigma_minus_lambda;
       for (auto m = 0; m <= mmax;
            ++m, ooversqrto2prho_exp_2mplus1 *= omega2_over_omega2_plus_rho) {
-        Gm[m] -= ooversqrto2prho_exp_2mplus1 * base_type::Fm_[m];
+        Gm[m] -= ooversqrto2prho_exp_2mplus1 * Fm[m];
       }
     }
   }
@@ -2166,8 +2155,7 @@ struct erfx_coulomb_gm_eval
 /// potential expansion (point nuclear, finite nuclear, SAP, erf, etc.).
 /// @note needs extra scratch for Boys function values
 template <typename Real>
-struct q_gau_gm_eval : private detail::CoreEvalScratch<q_gau_gm_eval<Real>> {
-  typedef detail::CoreEvalScratch<q_gau_gm_eval<Real>> base_type;
+struct q_gau_gm_eval {
   typedef Real value_type;
 
 #ifndef LIBINT_USER_DEFINED_REAL
@@ -2176,7 +2164,7 @@ struct q_gau_gm_eval : private detail::CoreEvalScratch<q_gau_gm_eval<Real>> {
   using FmEvalType = libint2::FmEval_Reference<scalar_type>;
 #endif
 
-  q_gau_gm_eval(unsigned int mmax, Real precision) : base_type(mmax) {
+  q_gau_gm_eval(unsigned int mmax, Real precision) {
     fm_eval_ = FmEvalType::instance(mmax, precision);
   }
 
@@ -2185,7 +2173,7 @@ struct q_gau_gm_eval : private detail::CoreEvalScratch<q_gau_gm_eval<Real>> {
   ///         .coefficient members
   template <typename PrimitivesContainer>
   void operator()(Real* Gm, Real rho, Real T, int mmax,
-                  const PrimitivesContainer& primitives) {
+                  const PrimitivesContainer& primitives) const {
     using std::isfinite;
     using std::isinf;
     using std::isnan;
@@ -2198,6 +2186,7 @@ struct q_gau_gm_eval : private detail::CoreEvalScratch<q_gau_gm_eval<Real>> {
 
     std::fill(Gm, Gm + mmax + 1, Real{0});
 
+    detail::FmStackScratch<Real> Fm(mmax + 1);
     for (const auto& prim : primitives) {
       assert(!isnan(prim.exponent) &&
              "q_gau_gm_eval: primitive exponent is NaN");
@@ -2211,15 +2200,15 @@ struct q_gau_gm_eval : private detail::CoreEvalScratch<q_gau_gm_eval<Real>> {
       // produces NaN from -inf/(-inf+rho) — visibly wrong rather than silently.
       if (isinf(prim.exponent) && prim.exponent > 0.0) {
         // α = ∞ => (∞/(∞+ρ)) = 1, contributes c_i * F_m(T)
-        fm_eval_->eval(&base_type::Fm_[0], T, mmax);
+        fm_eval_->eval(Fm.data(), T, mmax);
         for (auto m = 0; m <= mmax; ++m)
-          Gm[m] += prim.coefficient * base_type::Fm_[m];
+          Gm[m] += prim.coefficient * Fm[m];
       } else {
         const auto r = prim.exponent / (prim.exponent + rho);
-        fm_eval_->eval(&base_type::Fm_[0], T * r, mmax);
+        fm_eval_->eval(Fm.data(), T * r, mmax);
         auto factor = prim.coefficient * sqrt(r);
         for (auto m = 0; m <= mmax; ++m, factor *= r)
-          Gm[m] += factor * base_type::Fm_[m];
+          Gm[m] += factor * Fm[m];
       }
     }
   }
```

### Why the arithmetic does not change

Each evaluator performs the same floating-point operations in the same order on the same values; only the address of
the Boys scratch buffer changes. Every scratch element that is read has been written earlier in the same call by
`fm_eval_->eval`, so dropping the stock vector's zero-initialization cannot change any result.

The change is also race-free. The const call operators touch only the immutable Boys table and their own stack. This
matters because `Engine` copies share `core_eval_pack_` through a `shared_ptr`, so the alternative of keeping the
member scratch and calling through a `const_cast` would introduce a data race.

### Verification done so far

- **Master, this diff:** a standalone TU hashed the `Gm` output of `erf_coulomb_gm_eval`, `erfx_coulomb_gm_eval`
  (erfc and a mixed lambda/sigma case), `r12_xx_K_gm_eval<double,1>` and `q_gau_gm_eval` (a point charge plus two
  Gaussian primitives). The grid was mmax 0..28, rho 1e-3..2e4, T 0..200 and omega in {0, 0.1, 0.5, 2, 11}. Stock and
  patched master headers gave the same hash, `949b4fa191d2092e`. A 4-thread run sharing one evaluator instance also
  agreed with itself.
- **2.7.2 and 2.13.1:** we ship a narrower version downstream, which converts only erf and erfc/erfx and gates the
  no-copy path with a trait. It gives bit-identical `Gm` on a similar grid, and the patched build's hot loop has no
  `lock`-prefixed instructions and no `operator new`/`delete`. It also produced the "patched" rows above. That narrower
  patch also applies to master (with line offsets), but converting all evaluators as above makes the trait unnecessary.

I have not run the libint test suite against this diff. I am happy to open a PR with the change, run the unit tests,
and add a scaling or regression test if you would like one.

## Related

- #408 (Direct evaluation of SAP integrals) added `q_gau_gm_eval`, which follows the same vector-scratch-plus-copy
  pattern, so the Gaussian-nucleus/SAP 1-body path pays the same per-primitive cost. The
  `kshitij/experimental/sap_boys_timing` branch instruments Boys timing on that path and may see the effect.
- #383 / #386 (`erfx_coulomb`, `erfx_nuclear`) route erfc through `erfx_coulomb_gm_eval`.
- CHANGES 2.7.0: "fixed excessive lock contention when computing 1-e Coulomb ("nuclear") ints on many threads".
- I found no open or closed issue or PR about the `GenericGmEval` copy or erf/erfc thread scaling; #44 (2015) concerns
  hf++ Coulomb-K scaling and is unrelated.
