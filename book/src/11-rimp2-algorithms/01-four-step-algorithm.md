<!-- status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting", "2. Closed-shell RI-MP2" -->

# The four-step algorithm, 2014 and now {#four-step-algorithm}

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` |
| CLI | `method.kind = "rimp2"`, sections `[mp2]` and `[memory]` |
| Python | `run_rimp2` (`crates/ferric-python/ferric.pyi`) |

An RI-MP2 run spends its time in one place and its memory in another. The
time goes into forming $(ia|jb)$ for every pair of occupied orbitals, which
grows as $N^5$. The memory goes into the raw three-index AO integrals, which
grow as $N^3$ but with a large prefactor. This section lays out the four
steps of the 2014 algorithm, the same four steps in the code, and the
storage decision that differs between them.

The closed-shell RI-MP2 energy (section @sec:closed-shell-ri-mp2) needs three objects: the
metric factor $M$, the dressed factor $B$, and, for each occupied pair, the
block of $(ia|jb)$ integrals.

$$ V_{PQ} = (P|Q), \qquad M^{\mathsf T}M = V^{-1} . \label{eq:four-step-algorithm-1} $$

$$ (P|ia) = \sum_{\mu\nu} C_{\mu i}\,(P|\mu\nu)\,C_{\nu a}, \qquad B^P_{ia} = \sum_Q M_{PQ}\,(Q|ia) . \label{eq:four-step-algorithm-2} $$

$$ (ia|jb) \approx \sum_P B^P_{ia} B^P_{jb} . \label{eq:four-step-algorithm-3} $$

Equations @eq:four-step-algorithm-1 to @eq:four-step-algorithm-3 are equation @eq:closed-shell-ri-mp2-4 split into the steps that the
algorithm schedules.

## The 2014 algorithm {#four-step-algorithm-2014-algorithm}

The dissertation's algorithm (chapter 4, section 4.2) has four functions,
one per row of its Table 4.1. Its intermediates went to disk between
functions. The table carries its cost, memory and disk columns in this
book's symbols.

| Function | Quantity | Computation | Memory | Disk |
|---|---|---|---|---|
| 1 | $(P\vert Q)^{-1/2}$ | $N_{\mathrm{aux}}^3$ | $3N_{\mathrm{aux}}^2$ | $N_{\mathrm{aux}}^2$ |
| 2 | $(ia\vert P)$ | $N n_o (N + n_v) N_{\mathrm{aux}}$ | $2N^2 n_t N_{\mathrm{aux}}$ | $n_o n_v N_{\mathrm{aux}}$ |
| 3 | $B^Q_{ia}$ | $n_o n_v N_{\mathrm{aux}}^2$ | $2 n_t n_o n_v N_{\mathrm{aux}}$ | $n_o n_v N_{\mathrm{aux}}$ |
| 4 | $(ia\vert jb)$ and the energy | $n_o^2 n_v^2 N_{\mathrm{aux}}$ | $n_B n_v N_{\mathrm{aux}}$ | 0 |

Here $n_t$ is the thread count and $n_B$ the occupied batch of Function 4.
The printed Table 4.1 gives Function 2's cost as $O(N+V)X$. The
transformation it describes costs $N^2 n_o N_{\mathrm{aux}}$ for the first
quarter-transform and $N n_o n_v N_{\mathrm{aux}}$ for the second. Their sum
is $N n_o (N + n_v) N_{\mathrm{aux}}$, the entry above.

How each function ran (chapter @sec:operators, section @sec:terf-table-engine):

- **Function 1.** The metric was diagonalized with ScaLAPACK and the result
  written to disk.
- **Function 2.** The transformation was split into single-threaded matrix
  products, one per OpenMP thread, over batches of $n_X$ auxiliary
  functions. $(ia|P)$ was stored in $(a, P, i)$ order.
- **Function 3.** $B$ was formed in batches of $n_O$ occupied orbitals,
  sized by memory and the core count, with one thread per $i$.
- **Function 4.** A batch of $n_B$ $B$ matrices was held in memory, $n_B$ a
  multiple of the core count. The remaining $B$ matrices were read from disk
  one at a time, with $j$ running down to $i+1$. This lopsided batching
  reduces the disk reads from a maximum of
  $\tfrac12 n_o(n_o+1)\,n_v N_{\mathrm{aux}}$ to
  $\tfrac{1}{2n_B}\, n_o(n_o+1)\,n_v N_{\mathrm{aux}}$.

The occupied index was the batch index because $n_v \gg n_o$. That choice
survives unchanged.

## The algorithm in the code {#four-step-algorithm-algorithm-code}

| Step | Quantity | Cost | Lives in | Function |
|---|---|---|---|---|
| 1 | $V$, then $M = L^{-1}$ | $N_{\mathrm{aux}}^3$ | memory, $N_{\mathrm{aux}}^2$ | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c`, `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` |
| 2a | raw $(P\vert\mu\nu)$ | integrals | memory, packed memory or disk (section @sec:memory-budget) | `ferric-integrals/src/three_index_source.rs::ThreeIndexSource` |
| 2b, 3 | $(P\vert ia)$ and $B$, one pass | $N^2 n_o N_{\mathrm{aux}} + N n_o n_v N_{\mathrm{aux}} + n_o n_v N_{\mathrm{aux}}^2$ | $B$ in memory, $N_{\mathrm{aux}} n_o n_v$ | `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band_budgeted` |
| 4 | $G^{(i)}$ blocks and the energy | $\tfrac12 n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ | one block per worker | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, `ferric-mp2/src/rimp2.rs::pair_energy` |

### Where the time goes

Count the powers. Steps 1 to 3 each carry four factors that grow with the
molecule ($N_{\mathrm{aux}}^3$ is three, the others four). Step 4 carries
five. Divide step 4 by the dressing, the largest of the $N^4$ terms:

$$ \frac{\tfrac12 n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}}{n_o n_v N_{\mathrm{aux}}^2} = \frac{(n_o+1)\,n_v}{2N_{\mathrm{aux}}} . $$

The numerator grows as the square of the size and the denominator as the
first power, so the ratio grows linearly with the molecule. For benzene in
cc-pVDZ ($N = 114$, $N_{\mathrm{aux}} = 420$, $n_o = 21$, $n_v = 93$) the
steps cost

| Step | Multiply-adds |
|---|---|
| 1, $N_{\mathrm{aux}}^3$ | $7.4 \times 10^7$ |
| 2b, first quarter | $1.2 \times 10^8$ |
| 2b, second quarter | $9.4 \times 10^7$ |
| 3, dressing | $3.5 \times 10^8$ |
| 4, pair GEMMs | $8.4 \times 10^8$ |

Step 4 is 57% of the count here. For benzene in aug-cc-pVTZ with a frozen
core ($N = 414$, $N_{\mathrm{aux}} = 912$, $n_o = 15$, $n_v = 393$) it is
62%. These are operation counts, not timings: section @sec:parallel-performance records what has
and has not been timed. The dissertation measured the same trend in wall
time, 60% to 95% from 4 to 32 polyglycine subunits (section @sec:parallel-performance-dissertation-measured).

### The order of the driver

The driver runs these steps in this order:

1. The metric $V$ and its factor $M$.
2. The budget, resolved once: `ferric-mp2/src/rimp2.rs::eri3_budget_bytes`.
3. The admission check on the MO-side blocks:
   `ferric-mp2/src/rimp2.rs::check_mo_side_alloc` (section @sec:memory-budget).
4. The raw three-index source, optionally screened by distance
   (`eri3_screen_thresh`).
5. The streamed transform and dressing into $B$ (section @sec:streaming-transform).
6. `drop(src)`. The raw tensor is freed before the energy loop, which never
   reads it.
7. The pair energies (section @sec:pair-kernel), on the device when one is configured
   and usable (section @sec:device-mixed-precision), otherwise on the CPU.

Here is the body of `ri_mp2_spin_components` with the comments, the
screening setup, the pool charges, the timer tokens, the `kappa` validation
and the definitions of `dims` and `mixed_ok` removed, and the arguments of
the transform call joined onto one line. The comments are added:

```rust
let v2c = threeindex::coulomb_metric_2c(met_op, dfbs)?;       // V, step 1
let v2c_inv_sqrt = metric_inverse_sqrt(&v2c, met_op)?;        // M = L^{-1}
let budget_bytes = eri3_budget_bytes(config.memory_budget_bytes);   // resolved once
check_mo_side_alloc("RI-MP2", dfbs.nbasis(), nocc, nvir, false, budget_bytes)?; // (11.9)
// ...
let mut src =
    ThreeIndexSource::build_band_screened(op, obs, dfbs, budget_bytes, 0, naux_all, screen)?; // (P|mu nu)
// ...
let b_flat = stream_dressed_mo_band_budgeted(
    &mut src, &v2c_inv_sqrt, &c_occ, &c_vir, None, Some(budget_bytes),
)?;                                                           // B, (11.4)-(11.5)
drop(src);                                                    // raw tensor freed
// ...
let sc = acc.time("pair energies", || {
    dispatch_spin_components(&b_flat, eps, dims, config.kappa, mixed_ok)
});                                                           // step 4, device or CPU
```

The same `budget_bytes` reaches the admission check, the raw-tensor tier
and the chunk width. Section @sec:streaming-transform-reproducibility explains why the transform must not
re-resolve it.

## What changed {#four-step-algorithm-changed}

| | 2014 (dissertation) | Now (code) |
|---|---|---|
| Metric factor | $V^{-1/2}$ by ScaLAPACK, on disk | $L^{-1}$ by Cholesky, in memory; eigendecomposition with a cut for erf and terf (section @sec:closed-shell-ri-mp2-density-fitting-making-affordable) |
| $(P\vert ia)$ | written to disk, $(a,P,i)$ order | never stored whole; each auxiliary chunk is dressed as soon as it is transformed |
| $B$ | written to disk | in memory for the whole energy evaluation |
| What spills | $(ia\vert P)$ and $B$ | only the raw AO tensor $(P\vert\mu\nu)$, when it does not fit the budget |
| Function 4 | batch of $n_B$ in memory, the rest read from disk | all of $B$ in memory; one $G^{(i)}$ block per task |
| Pair symmetry | $j > i$ read from disk | $j \ge i$ tail GEMM, off-diagonal weight 2 (section @sec:pair-kernel) |
| Parallelism | OpenMP threads | rayon tasks over $i$ (CPU), a device GEMM (section @sec:device-mixed-precision), auxiliary bands over MPI ranks (section @sec:distributed-rimp2) |
| Memory control | batch sizes | a resolved budget, admission checks and a shared pool (section @sec:memory-budget) |

Why keep $B$ and spill the raw tensor? Compare their sizes. $B$ holds
$N_{\mathrm{aux}} n_o n_v$ doubles and the raw tensor
$N_{\mathrm{aux}} N^2$, so the ratio is $n_o n_v / N^2$. The active
occupied and virtual orbitals are disjoint subsets of at most $N$ orbitals,
so $n_o + n_v \le N$. For a fixed sum the product is largest when the two
are equal:

$$ \Big(\frac{n_o + n_v}{2}\Big)^2 - n_o n_v = \frac{(n_o - n_v)^2}{4} \ge 0
\quad\Rightarrow\quad n_o n_v \le \frac{(n_o+n_v)^2}{4} \le \frac{N^2}{4} . $$

$B$ is therefore at most a quarter of the raw tensor in every basis. For
benzene in cc-pVDZ the ratio is $21 \cdot 93 / 114^2 = 0.15$: 6.56 MB of $B$
against 43.67 MB of raw integrals. Keeping the smaller tensor in memory and
spilling the larger one is the reverse of the 2014 choice, which wrote both
MO tensors.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.2 (Table 4.1, Functions 1 to 4) | section @sec:four-step-algorithm-2014-algorithm and the left column of @sec:four-step-algorithm-changed |
| `book/sample-ri-mp2.md`, section 10.1.3 | the step table, rewritten for this chapter |
| `crates/ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` | the order of steps in @sec:four-step-algorithm-algorithm-code and the excerpt |

Changes from the sources: the dissertation's $O, V, X$ are written $n_o, n_v,
N_{\mathrm{aux}}$; its thread count $n$ is written $n_t$; Function 2's cost
entry carries the factor $N$ that the printed table lacks; the dressing
matrix is $M$, and the code's $M$ is the Cholesky inverse, not
$V^{-1/2}$. The step counts for benzene and the bound $n_on_v \le N^2/4$
are computed for this section.
