# Determinism, threads and the memory budget {#determinism-and-memory}

<!-- status: edit; registry: ;
     theory: -->

Run the same input twice, once on 4 threads and once on 12. Should the
energies agree? To the last bit, if the code is careful about the order in
which it adds numbers. Without that care they differ in the last digits, and
a test that compares the two runs sees a difference that has nothing to do
with the chemistry. A second class of failure is a job that the operating system
kills because two parts of the code each assumed they had all the memory.

Three rules of the code control these failures: how the threads are set up,
in what order a sum is added, and what happens when a job meets its memory
budget. This section states them once. A chapter that depends on one refers
back here.

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

## Two BLAS thread counts {#determinism-and-memory-blas-thread-counts}

Two libraries can start threads in ferric. rayon runs the parallel loops
that ferric writes itself. OpenBLAS runs the matrix products and
factorizations, both inside those loops and outside them. If both use every core at once, a 12-core
machine runs $12 \times 12$ threads on 12 cores. So the code gives each
library its own part of the run.

rayon owns the cores inside a parallel region (integrals, J and K builds, the
XC quadrature). OpenBLAS runs there at one thread. That count is the process
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

Called from a rayon worker, the resolver returns 1. That covers a serial
section that runs as one task of a parallel region, such as the free-atom
SCFs of the SAD guess. The first lines of
`ferric-core/src/blas_threads.rs::opt_in_blas_threads_with` are the guard and
the precedence:

```rust
if rayon::current_thread_index().is_some() {
    return 1;                                  // inside a parallel region: rayon owns the cores
}
let explicit = EXPLICIT_SERIAL.load(Ordering::SeqCst);   // TOML / Python setting, 0 = unset
effective_blas_threads(resolve_serial_blas_threads(
    (explicit > 0).then_some(explicit),        // 1. explicit
    get,                                       // 2. FERRIC_BLAS_THREADS
    rayon::current_num_threads(),              // 3. rayon pool size
))
```

`effective_blas_threads` clamps a request to the build's `MAX_THREADS`, with a
one-time warning, so the printed count is the count that runs.

### Restoring the count

`with_blas_threads(n, f)` raises the count, runs `f`, and puts a count back.
Which count? The obvious choice, the value it saw on entry, fails when two
threads open scopes at once. OpenBLAS's count is process-global. Suppose
thread 1 raises it from 1 to 8, and thread 2 then enters and records 8 as its
entry value. Thread 1 exits and restores 1, then thread 2 exits and restores
8. Every later parallel region now runs OpenBLAS at 8 threads inside rayon.

So the outermost scope restores the process baseline recorded by
`init_threading`, not the value it saw on entry. A nested scope on the same
thread restores the count its enclosing scope set. From
`ferric-core/src/blas_threads.rs::with_blas_threads`:

```rust
let outer = SCOPE.with(|s| s.get());            // (nesting depth, count) on this thread
let restore_to = if outer.0 > 0 {
    outer.1                                     // nested: the enclosing scope's count
} else {
    let now = unsafe { openblas_get_num_threads() }.max(1);
    match BASELINE.compare_exchange(0, now, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => now,                           // first scope ever: record the baseline
        Err(b) => b,                            // otherwise: the recorded baseline
    }
};
let _restore = Restore(restore_to, outer);      // restores on drop, also on panic
SCOPE.with(|s| s.set((outer.0 + 1, n)));
unsafe { openblas_set_num_threads(n) };
f()
```

The excerpt starts after the clamp and the debug assertion that `n > 1` is
never requested on a rayon worker, and the definition of the `Restore` guard
is removed.

**Stack size.** OpenBLAS's threaded LU (`dgetrf_parallel`) takes about
527 KiB of stack per frame and recurses, 4 frames deep at $n = 113$. That is
more than the 2 MiB that Rust gives a spawned thread, and the overflow
crashes the process. `init_threading` sets `RUST_MIN_STACK` to 64 MiB and
builds the global rayon pool with that stack. Rust reads `RUST_MIN_STACK`
once, at the first thread spawn, so `init_threading` must run before any
thread is spawned, and the CLI `main` and the Python module call it first.
Stack memory is reserved, not committed, until a thread touches it, so an
idle worker costs address space only.

## Bit-identical sums {#determinism-and-memory-bit-identical-sums}

Floating-point addition is not associative. In double precision,
$(10^{16} + 1) - 10^{16} = 0$ while $(10^{16} - 10^{16}) + 1 = 1$, because
$10^{16}+1$ is not representable and rounds to $10^{16}$. With terms of
ordinary size the difference is in the last bits. In both cases a sum
depends on the order of its terms.

A parallel reduction that combines partial sums in a tree has a tree shape
that depends on the worker count. Its result can therefore change with
`RAYON_NUM_THREADS`. Reductions in ferric fold in a fixed ascending order
that does not depend on the thread count, so results are bit-identical across
`RAYON_NUM_THREADS`.

The RI-MP2 energy (section @sec:closed-shell-ri-mp2) shows the pattern. Each occupied orbital
$i$ gets its own partial sums on some worker, the partials are collected
into a vector indexed by $i$, and one thread adds them in order. From
`ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, with the
body of the per-$i$ closure removed:

```rust
let partials: Vec<(f64, f64)> = (0..nocc)
    .into_par_iter()                     // parallel over i
    .map(|i| {
        // ... (e_os_i, e_ss_i) = sum over j >= i of the pair energies
        (e_os_i, e_ss_i)
    })
    .collect();                          // ordered by i, whatever worker ran it
let mut e_os = 0.0;
let mut e_ss = 0.0;
for (e_os_i, e_ss_i) in partials {       // serial, ascending i
    e_os += e_os_i;
    e_ss += e_ss_i;
}
```

A rayon `reduce` would be shorter and would change the bits with the worker
count; the code comment puts the variation at the µHa level. The cost of the
fixed order is one vector of `nocc` pairs and a serial loop over it.

The contraction engine's row-banded matrix product gives the same bits for
every worker count because every even band width rounds identically
(`ferric-tensors/src/einsum.rs`, test `row_banded_gemm_is_thread_count_independent`).
Section @sec:tensors-reproducibility explains why the width has to be even.

The serial BLAS thread count is the one exception. With
`blas_threads = 1` results are bit-identical across thread counts. Any other
value lets OpenBLAS split reductions by thread count, which moves energies at
the last bit; the user guide puts the size at about $10^{-13}$ Ha. Since the
default serial count is the rayon pool size (@sec:determinism-and-memory-blas-thread-counts), changing
`RAYON_NUM_THREADS` alone also changes the serial count. For two runs that
must agree bit for bit at different `RAYON_NUM_THREADS`, set
`blas_threads = 1`.

What "the same" means across machines is weaker. A different BLAS or CPU can
round differently, so a test that compares to a stored golden energy uses a
tolerance, and a committed value never depends on the machine that produced it.

## The memory budget {#determinism-and-memory-memory-budget}

A large calculation is limited by memory as often as by time. The worst
outcome is a job that starts, does part of its work, and is then killed by
the operating system. The budget
exists so that a job that cannot fit fails at the start with a message, or
takes a slower path that fits.

The budget is a ceiling on the **ledgered** allocations, in bytes. It resolves
once, with this precedence (`ferric-core/src/memory.rs::resolve_budget_bytes`):

| Order | Source |
|---|---|
| 1 | the explicit value: `[memory] budget_gb` or a keyword argument |
| 2 | `FERRIC_MEM_BUDGET_GB` |
| 3 | the legacy `FERRIC_OOC_BUDGET_GB` and `FERRIC_ERI3_BUDGET_GB` (the smaller wins) |
| 4 | $0.8\times$ the available memory (the cgroup limit and `MemAvailable` on Linux) |
| 5 | 2 GiB, with a one-time warning, when detection fails |

In step 4, the available memory is the smaller of the process's cgroup limit
and `MemAvailable`, and the factor 0.8 leaves headroom.

### One pool for the whole process

A ceiling alone is not enough. Suppose the budget is 4.72 GiB and two
allocations each check "do I fit in 4.72 GiB?". Each answer is yes, and the
process holds the sum. The module documentation of
`ferric-core/src/memory/pool.rs` records this case: a 27-atom def2-SVP B3LYP
RI-JK single point where the density-fitting tensor and the grid AO cache
each passed that check, and the process was killed at 6.04 GiB resident.

The CLI therefore installs one process-wide `MemoryPool` of the budget's
size. A large, size-dependent allocation reserves its bytes from the pool
before it allocates, and releases them when it is done. Two allocations alive
at once cannot each claim the whole budget: whichever asks second sees only
what the first left.

The reservation is one atomic compare-and-swap loop, in
`ferric-core/src/memory/pool.rs::MemoryPool::reserve`. The excerpt is the core of the
private `reserve_once` it calls, with the source comments, a lint attribute,
the bookkeeping of the success branch and the error text removed:

```rust
let outcome =
    self.inner
        .outstanding                                   // bytes currently reserved
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
            let next = cur.saturating_add(bytes);
            (next <= cap).then_some(next)              // commit only if it fits
        });
match outcome {
    Ok(prev) => {
        // ... record the peak and the label, return a Reservation guard
    }
    Err(cur) => Err(FerricError::MemoryExhausted(format!(
        // ... label, request, free bytes, shortfall, occupancy
    ))),
}
```

`fetch_update` retries if another thread changed `outstanding` in between,
so two rayon workers that each fit alone cannot both commit past the ceiling.
The returned `Reservation` credits the bytes back when it is dropped, which
lets a transient buffer inside an SCF iteration give its bytes back every
iteration.

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
@sec:closed-shell-ri-mp2 gives the RI-MP2 case).

## Known limits

- A library or Python caller installs no pool, so each check there compares
  its own allocation with the whole budget.
- `OPENBLAS_NUM_THREADS` above 1 is honored and oversubscribes rayon regions.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-core/src/blas_threads.rs` module documentation and code | sections @sec:determinism-and-memory-blas-thread-counts and the stack size |
| `crates/ferric-core/src/memory.rs`, `crates/ferric-core/src/memory/pool.rs` | section @sec:determinism-and-memory-memory-budget |
| `site/src/reference/architecture.md`, "Cross-cutting conventions" | the statements on determinism and the pool |
| `site/src/reference/input.md`, `[memory]` and `[parallel]` | the precedence tables |
| `site/src/using/sharp-bits.md` | what the budget does not cap |

Changes from the sources: the order of precedence is given as one table per
setting.
