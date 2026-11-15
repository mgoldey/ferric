# 2.3 Determinism, threads and the memory budget

<!-- section: 2.3; status: edit; registry: ;
     theory: -->

Three rules of the code affect a number a user can read: how the threads are
set up, in what order a sum is added, and what happens when a job meets its
memory budget. This section states them once. A chapter that depends on one
refers back here.

| Setting | CLI | Python | Environment |
|---|---|---|---|
| Serial-section BLAS threads | `[parallel] blas_threads` | `ferric.set_blas_threads` | `FERRIC_BLAS_THREADS` |
| Rayon threads | | | `RAYON_NUM_THREADS` |
| BLAS threads in parallel regions | | | `OPENBLAS_NUM_THREADS` |
| Memory budget | `[memory] budget_gb` | keyword arguments where offered | `FERRIC_MEM_BUDGET_GB` |

| Entry point | Location |
|---|---|
| Threads | `ferric-core/src/blas_threads.rs::{init_threading,with_blas_threads,opt_in_blas_threads}` |
| Budget | `ferric-core/src/memory.rs::{resolve_budget_bytes,check_alloc}`, `ferric-core/src/memory/pool.rs::MemoryPool` |

## 2.3.1 Two BLAS thread counts

rayon owns the cores inside a parallel region (integrals, J and K builds, the
XC quadrature). OpenBLAS runs there at one thread. This is the process
baseline: `ferric-core/src/blas_threads.rs::init_threading` sets it to one
unless the user set `OPENBLAS_NUM_THREADS`. A larger value multiplies rayon
threads by BLAS threads and oversubscribes the machine. That is a user
choice, and the code does not work around it.

Serial sections run outside any rayon region, where the cores would otherwise
idle: the per-iteration Fock diagonalization, the DIIS commutator and the
orthogonalizer and density products. They wrap their work in
`with_blas_threads(opt_in_blas_threads(), ..)`. The count comes from, highest
first,

| Order | Source |
|---|---|
| 1 | `[parallel] blas_threads` or `ferric.set_blas_threads` |
| 2 | `FERRIC_BLAS_THREADS` |
| 3 | the size of the rayon pool, so `RAYON_NUM_THREADS` is the one count a user sets |

Called from a rayon worker, the resolver returns 1. A request is clamped to
the build's `MAX_THREADS`, with a one-time warning, so the printed count is
the count that runs. `with_blas_threads` restores the process baseline when
the scope ends, and not the value it saw on entry: OpenBLAS's count is
process-global, and an interleaved save and restore by two threads would leave
a raised count behind.

**Stack size.** OpenBLAS's threaded LU needs about 527 KiB of stack per
recursion frame, more than the 2 MiB that Rust gives a spawned thread.
`init_threading` sets `RUST_MIN_STACK` to 64 MiB and builds the global rayon
pool with that stack. It must run before any thread is spawned, so the CLI
`main` and the Python module call it first.

## 2.3.2 Bit-identical sums

A floating-point sum depends on the order of its terms. Reductions in ferric
fold in a fixed ascending order that does not depend on the thread count, so
results are bit-identical across `RAYON_NUM_THREADS`. Examples used later in
the book: the per-occupied-orbital partial sums of RI-MP2 are added in order
of the orbital index on one thread (section 10.1), and the contraction engine's
row-banded matrix product gives the same bits for every worker count because
every even band width rounds identically
(`ferric-tensors/src/einsum.rs`, test `row_banded_gemm_is_thread_count_independent`).
A tree reduction would be faster to write and would change the bits with the
worker count.

The serial BLAS thread count is the one exception. With
`blas_threads = 1` results are bit-identical across thread counts. Any other
value lets OpenBLAS split reductions by thread count, which moves energies at
the last bit; the user guide puts the size at about $10^{-13}$ Ha.

What "the same" means across machines is weaker. A different BLAS or CPU can
round differently, so a test that compares to a stored golden energy uses a
tolerance, and a committed value never depends on the machine that produced it.

## 2.3.3 The memory budget

The budget is a ceiling on the **ledgered** allocations, in bytes. It resolves
once, with this precedence (`ferric-core/src/memory.rs::resolve_budget_bytes`):

| Order | Source |
|---|---|
| 1 | the explicit value: `[memory] budget_gb` or a keyword argument |
| 2 | `FERRIC_MEM_BUDGET_GB` |
| 3 | the legacy `FERRIC_OOC_BUDGET_GB` and `FERRIC_ERI3_BUDGET_GB` (the smaller wins) |
| 4 | $0.8\times$ the available memory (the cgroup limit and `MemAvailable` on Linux) |
| 5 | 2 GiB, with a one-time warning, when detection fails |

The CLI installs one process-wide `MemoryPool` of that size. A large,
size-dependent allocation reserves its bytes from the pool before it
allocates, and releases them when it is done. Two allocations alive at once
therefore cannot each claim the whole budget. Before the pool, each check
compared its own request with the whole ceiling, and a job whose pieces each
fit could still be killed.

What a request that does not fit does depends on the allocation:

| Allocation | If it does not fit |
|---|---|
| a tensor with a fallback (spill to disk, recompute, a packed layout) | takes the fallback, slower |
| a dense path with none | `check_alloc` refuses it before allocating, with a message naming the size and the budget |

The budget is not the process memory. Basis-sized matrices, integral engines
and per-thread scratch are not charged, so the resident size can exceed
`budget_gb`. The user guide's "sharp bits" page has the measured example.

**Budget-derived widths and results.** Where a block width comes from the
budget, the width can set the order of a floating-point sum. At an ample
budget the width is a constant. Two runs at different budgets can then differ
in the last digits, and two runs at one budget are bit-identical (section
10.1 gives the RI-MP2 case).

## Known limits

- A library or Python caller installs no pool, so each check there compares
  its own allocation with the whole budget.
- `OPENBLAS_NUM_THREADS` above 1 is honored and oversubscribes rayon regions.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-core/src/blas_threads.rs` module documentation and code | sections 2.3.1 and the stack size |
| `crates/ferric-core/src/memory.rs`, `crates/ferric-core/src/memory/pool.rs` | section 2.3.3 |
| `site/src/reference/architecture.md`, "Cross-cutting conventions" | the statements on determinism and the pool |
| `site/src/reference/input.md`, `[memory]` and `[parallel]` | the precedence tables |
| `site/src/using/sharp-bits.md` | what the budget does not cap |

Changes from the sources: the order of precedence is given as one table per
setting.
