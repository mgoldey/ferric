# libint2 thread scaling: research note for the PBC SR per-call inflation (2026-09-29)

Scope: this is research only (web plus reading local source and objdumping the already-built shim.o). Nothing was built
or benchmarked. It covers the per-call inflation measured in FINDINGS.md, "Parallel contention diagnosis
(2026-09-27)": about 3x ns/call at 6 threads vs 1 for `erfc` 3-centre and hcore-SR libint calls, while pure FP scales 5.5x.

Evidence tags: **[SRC]** read in source on this box; **[BIN]** seen in ferric's compiled `shim.o`; **[MEAS-ext]**
measured by someone else; **[DOC]** a vendor or maintainer statement; **[INF]** my inference, untested.

---

## 0. Headline finding: every erfc core-integral evaluation copies an evaluator that holds a shared_ptr to the global Boys singleton

**Claim.** `libint2::GenericGmEval<F>::eval` is `const`. It calls the operator by **copy-constructing the wrapped
functor on every call**:

```cpp
// boys.h, GenericGmEval (2.7.2 line ~1588; identical in 2.13.1 line 1966 and in master today)
void eval(Real* Gm, Real rho, Real T, int mmax, ExtraArgs... args) const {
  (GmEvalFunction(*this))(Gm, rho, T, mmax, std::forward<ExtraArgs>(args)...);
}
```

`erfc_coulomb_gm_eval` (2.7.2) and `erfx_coulomb_gm_eval` (2.13.1/master, which serves erf and erfc) hold two members:
- a `std::vector<Real> Fm_` scratch, inherited from `detail::CoreEvalScratch<...>`, so each copy costs a **heap
  allocation, a memmove and a free**;
- a `std::shared_ptr<const FmEval_Chebyshev7<double>> fm_eval_`. It points to the **process-wide
  `FmEval_Chebyshev7::instance()` singleton** (a function-local `static` shared_ptr). Every engine in every thread shares
  that one control block. Copying the functor does an **atomic increment**, and destroying the copy does an
  **atomic decrement**, both on the **same cache line from every thread**. That is true sharing, not false sharing.

`Engine::compute2` calls this once per **surviving primitive quartet** (`case Operator::erfc_coulomb:
core_eval_ptr->eval(gm_ptr, rho, T, mmax, omega)` in engine.impl.h ~1373). So one shell-triplet call makes
2·N_primquartet contended atomics and N_primquartet malloc/free pairs.

**Plain `Operator::coulomb` does not take this path.** Its core evaluator is `FmEval_Chebyshev7` itself, called through
a `const&` (`core_eval_ptr->eval(gm_ptr, T, mmax)`), with no copy, no refcount and no allocation.

Evidence:
- [SRC] `~/.local/include/libint2/boys.h` 1576-1760 (2.7.2, the build ferric links),
  `~/.local/libint2-2.13.1/include/libint2/boys.h` 1964-1966 and 2105-2147, and the psi4 env's libint (the same code).
- [SRC] upstream master still has it: https://raw.githubusercontent.com/evaleev/libint/master/include/libint2/boys.h
  (fetched 2026-09-29: `GmEvalFunction(*this)` present; erfx evaluator holds the shared_ptr and the vector).
- [BIN] ferric's own `target/release/build/ferric-integrals-*/out/*-shim.o`. The out-of-line
  `GenericGmEval<erfc_coulomb_gm_eval<double>>::eval<double,double>` contains, in order: `operator new`, `memmove`,
  a `__libc_single_threaded` check, `FmEval_Chebyshev7::eval`, `memcpy`, `_Sp_counted_base::_M_release` (atomic dec),
  `operator delete`, a second `FmEval_Chebyshev7::eval`, `lock addl $0x1,0x8(%r13)` (atomic inc), and
  `_M_release`. GCC -O2 did **not** elide the copy. `__libc_single_threaded` goes false permanently once rayon spawns
  a thread, so the atomics are real even in the 1-thread pool baseline. They are uncontended there, which explains why
  1 thread looks fine.
- The engine code is **header-only and instantiated inside shim.cc**, so it is compiled with ferric's `-O2`. The
  generated VRR/HRR kernels in `libint2.a` are not involved in this path.

Why this fits the 2026-09-27 measurements [INF, but it fits every row]:
- It is inside libint's integral call. Stripping the shim copies (`sr3-zero-plain`) did not help. Own-engine and pool
  give the same result, because the shared line is the global singleton's control block, not anything ferric owns.
- SMT and pinning do not matter. A contended line bounces between cores whatever the placement. Physical pinning got
  slightly *worse* (3.22x vs 2.78x), which is consistent with faster cores contending harder.
- cpu/wall = 1.000. There is no blocking, so it is not a mutex: spinning on coherence traffic counts as CPU time.
- ctl-alloc alone gave 1.43x, so malloc explains only a part. The shared-line atomics are the rest.
- hcore-SR inflates the **most** (3.3-4.0x). It uses Gaussian-nucleus **erfc** eri3 with few primitives on the
  nucleus side, so its per-call work is smaller relative to the per-primitive overhead.
- Order of magnitude: Broadwell L2-to-L2 modified intervention is about 40 ns
  ([MEAS-ext] https://community.intel.com/t5/Intel-Moderncode-for-Parallel/Core-to-Core-Communication-Latency-in-Skylake-Kaby-Lake/td-p/1061658,
  https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html). The serial cost is 1826 ns/call for tens of
  primitive quartets per (cc-pVDZ C, 1-primitive aux) triplet. A few tens of line transfers at about 40 ns each is a few µs
  per call, which is the size of the measured +3.5 µs inflation. A single line can only be owned by one core at a time,
  so this is a *throughput ceiling*, and it predicts that aggregate speed-up saturates as threads are added.
- The FINDINGS plan records "the molecular analogue, a raw 3-index build, measured 4.4x on this box". That build is
  full-range Coulomb (no copy path) as far as I know. **Verify that before leaning on it.**

**Fix options (all [INF], none tried):**
1. **Patch the header.** Ferric already carries libint header patches: `wiki/perf-tasks/patches/libint2-terf-*.patch`
   via `LIBINT2_PREFIX`. Because the path is header-only, **no libint2.a rebuild is needed**, only a shim recompile.
   Make `erfc_coulomb_gm_eval::operator()` `const`, use a stack scratch (`Real Fm[LIBINT_MAX_AM*4+2+deriv]`, or
   alloca/`std::array<Real,64>`) instead of the member vector, and have `GenericGmEval::eval` call
   `static_cast<const GmEvalFunction&>(*this)(...)`, with no copy. The arithmetic is unchanged, so the result should
   be **bit-identical**; the existing checksums (sr3 `ea59bc1f7b8425ad`, hcore `2814b1e9d483fc38`) confirm that for free.
   Do NOT just `const_cast` and call the member operator on the shared object while keeping the member vector. That is
   a data race if two engines ever share one `GenericGmEval`. `Engine`'s copy ctor copies `core_eval_pack_`, which is a
   shared_ptr copy. `EnginePool::from_fn` constructs each engine fresh, so ferric does not share today, but a stack
   scratch is race-free regardless.
2. **Do it in the shim without patching libint.** Compute SR as `coulomb − erf` from two engines. **Not
   recommended:** erf also copies the shared_ptr (no vector), it doubles the work, and it loses precision. libcint's
   README says direct SR is about 10x more accurate than subtraction.
3. **Report upstream.** A search found no evaleev/libint issue about this: the issue search for "thread" returned only
   #349 and #153. The fix is small and generic; it helps erf, erfc, erfx and r12 (`r12_xx_K_gm_eval<1>` has the same
   vector plus shared_ptr pattern).

---

## 1. Known libint2 thread-scaling problems (Q1)

| # | Claim | Source | Strength |
|---|---|---|---|
| 1.1 | libint 2.7.0 (2021-09-14) CHANGES: "fixed excessive lock contention when computing 1-e Coulomb ("nuclear") ints on many threads". A precedent of the same class: core-evaluator and singleton state contended across threads. | https://raw.githubusercontent.com/evaleev/libint/master/CHANGES | [DOC] |
| 1.2 | psi4 issue #2491: libint2 one-electron integrals (electrostatic potential on grid points) scaled **1.45x at 2 cores and 1.41x at 8, and did not improve at 18**. The old Obara-Saika code scaled 16.6x at 18 cores. The reporter suspected "some kind of global lock". psi4 1.6 says it was fixed "by using new Libint2" (#2491, #2413), which matches 1.1. | https://github.com/psi4/psi4/issues/2491 ; https://psicode.org/posts/v16/ | [MEAS-ext] numbers; root cause [INF] (no diagnosis posted) |
| 1.3 | 2.5.0-beta.2: "2-body core engine singleton re-initialization is thread-safe". Singletons (`FmEval_Chebyshev7::instance`, `TennoGmEval::instance`, `FmEval_Taylor::instance`) take a `static std::mutex` **only while growing mmax** (`while (instance_->max_m() < m_max)`). In 2.7.2 steady state there is no lock per call. | CHANGES; [SRC] boys.h 298-313 | [SRC] |
| 1.4 | `GenericGmEval::instance()` is **not** a singleton: `make_shared` per engine. Each engine gets its own GenericGmEval, but they all point at the one global `FmEval_Chebyshev7` (section 0). | [SRC] boys.h 1583 | [SRC] |
| 1.5 | Per-call allocation elsewhere in `compute2`: `targets_` uses a stack allocator (`ext_stack_allocator`); `spbra_`/`spket_` `ShellPair::init` is recomputed per call **when no precomputed ShellPair is passed**. That is vector growth inside a member ShellPair, amortised after warm-up. `scratch_` is sized once in `reset_scratch()`. I found no other per-call heap traffic. | [SRC] engine.h 419-422, 926-964; engine.impl.h 1195-1197 | [SRC] |
| 1.6 | The `any_cast` used per primitive is a `typeid` compare (pointer-equal in one binary) plus a virtual `type()`. It is read-only and not contended. | [SRC] util/any.h 192-205 | [SRC] |
| 1.7 | `libint2::initialize()`/`finalize()` are called once in ferric's shim (shim.cc ~251-261). The CHANGES entries show no per-call cost. | [SRC] | [SRC] |
| 1.8 | Engine ctor: the singleton mutex fires when an engine requests a larger mmax than the current table. FINDINGS already notes "the libint ctor mutex makes per-chunk construction a contention trap". It does not matter for pre-built pools. | [SRC], FINDINGS 3315 | [SRC] |

## 2. Hardware causes of per-call slowdown for small batches (Q2)

| # | Claim | Source | Strength |
|---|---|---|---|
| 2.1 | Box: i7-6800K Broadwell-E. 6C/12T, L1d 32 KiB/core, **L2 256 KiB/core**, L3 15 MiB shared (2.5 MiB/core), AVX2+FMA, **no AVX-512**, so the AVX-512 licence-drop mechanism does not exist here. | `lscpu` | [SRC] |
| 2.2 | Frequency: base 3.4 GHz, Turbo 2.0 3.6 GHz, Turbo Boost Max 3.0 up to 3.8 GHz on one favoured core. The worst-case all-core vs single-core ratio is about 3.8/3.4 = 1.12, which cannot produce 3x. ctl-fp already measured 1.05-1.11x. | https://www.tomshardware.com/reviews/intel-core-i7-broadwell-e-6950x-6900k-6850k-6800k,4587-11.html | [DOC] + ferric [MEAS] |
| 2.3 | AVX offset (Haswell/Broadwell) applies to 256-bit AVX. The libint kernels are **scalar VEX FMA** (`vfmadd132sd`, `vmovsd`, no ymm in a sampled `eri3_aB_S__0__D__1` kernel), so there is no AVX frequency offset. | [BIN] objdump of libint2.a member | [BIN] |
| 2.4 | Per-engine working set [INF]: `Libint_t` has about 275 fields (roughly 2.2 KB). For contracted ints `primdata_` holds max_nprim^3 of them for 3-centre, and the used prefix per call is N_primquartet × 2.2 KB. That is ~180 KB at 81 quartets, plus the touched VRR stack and `scratch_`. It is close to or above the 256 KiB L2, but far below L3/core. The predicted effect is L2 spills into a shared L3 ring; ctl-mem at 32 KiB already gave 1.3-1.4x. This is a plausible secondary contributor of about 1.3x, not 3x. | [SRC] libint2_types.h, engine.impl.h 672 | [INF] |
| 2.5 | Single-socket box, so NUMA does not apply. SMT is ruled out by ferric's own pinning table. | FINDINGS 4246-4264 | ferric [MEAS] |

## 3. How other codes parallelise libint2 (Q3)

| # | Claim | Source | Strength |
|---|---|---|---|
| 3.1 | The canonical pattern is one `Engine` per thread, copied from a prototype. libint's `hartree-fock++` test does this with `std::vector<Engine> engines(nthreads, engines[0])` and precomputed `ShellPair` lists (`compute_shellpairs`). Note that copying from a prototype shares the `core_eval_pack_` shared_ptr, which is harmless for Coulomb and would be racy if the fix in section 0 kept member scratch. | libint `tests/hartree-fock/hartree-fock++.cc` (read from memory of upstream; the repo was not re-fetched) | [INF/DOC] |
| 3.2 | Psi4 uses one libint2 engine per thread. Its 2022 thread-scaling bug (1.2) was fixed by upgrading libint, and SCF gradients "took longer with more threads" until #2559/#2581. | https://psicode.org/posts/v16/ , https://github.com/psi4/psi4/issues/2442 | [DOC] |
| 3.3 | Passing precomputed `ShellPair` data (`compute2(..., spbra, spket)`) avoids per-call `ShellPair::init`. For 3-centre `xs_xx` the bra is the 1-function aux shell and the ket is the pair. A precomputed ket pair per (μ, ν, image) is cheap to cache. This is a serial win and orthogonal to the scaling problem. | [SRC] engine.impl.h 1190-1197 | [SRC] |
| 3.4 | I found no published libint2 per-thread scaling numbers for erf, erfc or SR operators. | searches listed below | - |

## 4. libcint design differences (Q4)

| # | Claim | Source | Strength |
|---|---|---|---|
| 4.1 | libcint is pure C with **caller-supplied `cache` buffers**. PySCF allocates one buffer per OpenMP thread outside the job loop (`#pragma omp parallel { double *buf = malloc(...); #pragma omp for ... (*intor)(buf, NULL, shls, ..., cintopt, cache); }`). That gives no per-call heap traffic and no refcounted shared objects. | https://raw.githubusercontent.com/pyscf/pyscf/master/pyscf/lib/gto/fill_nr_3c.c | [SRC-ext] |
| 4.2 | Its Rys roots and weights come from static polynomial tables (read-only) or are computed on the stack. SR (erfc) is handled through `env[PTR_RANGE_OMEGA] < 0`. In v6, low-L SR ERIs are computed as full − LR, because v5 computed SR Rys quadratures from scratch at runtime and was slow. | https://arxiv.org/pdf/2302.11307 (PySCF PBC screening paper) ; https://github.com/sunqm/libcint | [DOC] |
| 4.3 | libcint's README: "Thread safe", "Small memory usage ... typically less than 1 Mega bytes". SR errors are larger than full-range, but "comparing to computing integrals via 'regular ERI − long-range ERI', errors are roughly one order of magnitude better". | https://github.com/sunqm/libcint | [DOC] |
| 4.4 | Raw speed: Sun (2015) reports libcint comparable to libint. A 2025 SIMD-MD paper says both reach <10% of AVX2 peak. | https://onlinelibrary.wiley.com/doi/10.1002/jcc.23981 ; https://arxiv.org/html/2506.12501 | [MEAS-ext] |
| 4.5 | PySCF's PBC SR 3c (`PBCnr3c_fill`) sums all lattice images inside one C call per shell triplet, using the per-thread cache. The shared work is amortised over images, whereas ferric makes one call per triplet × image. That is a serial-efficiency difference; with no shared mutable state, the scaling difference is explained by section 0. | from knowledge of PySCF `pbc/df` C kernels; not re-fetched | [INF] |

## 5. Build and config flags (Q5)

Ferric's 2.7.2 build (`~/.local/include/libint2/config.h`): `LIBINT_MAX_AM 6` (`"6,4"`), `LIBINT_OPT_AM 3`,
`LIBINT_CONTRACTED_INTS 1`, `LIBINT_GENERATE_FMA 1`, `LIBINT2_REALTYPE double`, no `LIBINT_VECTOR_LENGTH`
(scalar), `ERI3_PURE_SH 1`, `LIBINT_ENABLE_UNROLLING 100`, GCC 12.4. The kernels are compiled with AVX2/FMA (VEX
scalar); the shim uses `-O2` with no `-march`.

- Nothing in config.h touches the core-evaluator copy. It is in the C++ header layer and is the same for every
  configuration and for 2.7.2, 2.13.1 and master [SRC].
- `LIBINT_CONTRACTED_INTS` makes `primdata_` scale as nprim^rank of `Libint_t` (2.4). That is inherent, and
  sensible for cc-pVDZ contractions.
- `LIBINT_VECTOR_LENGTH` (SIMD over primitive batches) is off. Turning it on changes serial speed, not the scaling
  defect.
- Moving to 2.13.1 (installed at `~/.local/libint2-2.13.1`) does **not** fix this: the same copy is present. 2.13
  brings `erfx_coulomb` (a combined erf and erfc), second derivatives, and the 2.7.0 nuclear-lock fix, which 2.7.2
  already has.

---

## 6. Ranked, testable hypotheses (no perf needed)

The shared cost of each experiment is one `pbc_parallel_contention` run (1 vs 6 threads, 100k calls) on a quiet box.

**H1 (most likely): contended shared_ptr refcount on the global `FmEval_Chebyshev7` singleton, plus per-primitive
malloc, both from `GenericGmEval::eval` copying the erfc functor.**
- **E1a: operator swap.** Add a `sr3-coulomb` variant (same shells and images, `Operator::coulomb`) and an `sr3-erf`
  variant. *If H1 is true:* Coulomb inflates about 1.1-1.4x (ctl-mem-like); erf inflates close to erfc (it has the
  shared_ptr, lacks the vector), which shows the atomic dominates; erfc stays about 3x. *If H1 is false:* all three
  inflate about equally, about 3x.
- **E1b: header patch** (section 0 fix 1: const operator, stack Fm scratch, no copy). *If true:* erfc inflation drops to
  about 1.1-1.4x, 6-thread speed-up rises from 2-3x toward about 5x, the 1-thread ns/call also drops (tens of
  malloc/free and atomics removed per call), and the checksums are **bit-identical**. *If false:* inflation stays about
  3x, and only the serial time moves slightly.
- **E1c: 6 processes instead of 6 threads** (6 copies of the example with `--threads 1`, launched together). Each
  process has its own singleton, so there is no shared line; cache and L3 bandwidth effects remain. *If H1:*
  per-process ns/call is about 1.1-1.4x the solo run. *If hardware (H2/H3):* about 3x, the same as threads. This is
  the cleanest shared-state vs hardware separator and needs no code change.
- Supporting shape check: aggregate throughput vs threads 1,2,3,6. A single contended line gives an early plateau
  (speed-up about flat past 2-3 threads). Bandwidth or cache gives a gradual roll-off.

**H2: L2 capacity (per-call working set = primdata_ prefix + VRR stack + scratch, roughly 256 KiB), so six cores
thrash a shared L3 ring.**
- **E2:** after E1b, rerun ctl-mem with `--mem-kb` at 128, 256, 512 and 2048, and rerun sr3 with a basis that has fewer
  primitives per shell (e.g. STO-3G or a decontracted-light set) at similar call counts. *If true:* the residual
  inflation tracks working-set size and jumps as it crosses 256 KiB. *If false:* the residual is flat about 1.1x.
  Expect at most about 1.3-1.4x, based on the existing ctl-mem numbers.

**H3: frequency, turbo, thermal.** Already bounded at 1.05-1.11x by ctl-fp.
- **E3:** during the 6-thread run, `watch -n0.5 "grep MHz /proc/cpuinfo"`, or `turbostat --quiet` if it is available.
  *If true:* the loaded MHz is at least 25% below the single-thread MHz. *If false:* within about 10% (3.4-3.8 GHz
  envelope). Expect false.

**H4 (low): memory-allocator arena contention separate from H1.** ctl-alloc gave 1.43x in isolation, and E1b removes
the allocations anyway. If needed, run E1a with `LD_PRELOAD` jemalloc or mimalloc: *if true*, erfc inflation drops
noticeably but not fully; *if false*, no change.

**Top 3 to run first:** E1c (processes vs threads: no code, decisive on shared state vs hardware), E1a (a Coulomb or erf
variant in the example: one-line operator swap, decisive on the operator path), and E1b (the header patch: the actual
fix, with free bit-identity via the existing checksums).

---

Searches run (for completeness): libint GitHub issues ("thread"), libint CHANGES, psi4 #2491/#2413/v1.6 notes, libcint
README, PySCF fill_nr_3c.c, libcint SR (PySCF PBC screening paper), Broadwell-E turbo and core-to-core latency. I found
no public report of the `GenericGmEval` copy issue.
