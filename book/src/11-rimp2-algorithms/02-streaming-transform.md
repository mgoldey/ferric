<!-- status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting" -->

# Streaming the MO transform and the dressing {#streaming-transform}

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band_budgeted`, `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band`, `ferric-mp2/src/rimp2.rs::mo_stream_chunk_for` |
| Other callers | `ferric-mp2/src/rimp2.rs::eri3_mo_block_dressed_band` (MPI, section @sec:distributed-rimp2); the GW $B$ build in `ferric-gw/src/mo_b.rs` |
| Trace | `FERRIC_MO_STREAM_TRACE=1` logs the resolved chunk width |

Steps 2 and 3 of section @sec:four-step-algorithm turn the raw AO integrals into $B$. Done
naively, they hold two full MO tensors at once: the undressed $(P|ia)$ and
the dressed $B$, both $N_{\mathrm{aux}} n_o n_v$ doubles. The code runs the
two steps in one pass over the raw tensor, a few auxiliary rows at a time,
so the undressed $(P|ia)$ never exists whole. The plan: write the pass,
count its memory, derive the chunk width from the budget, then state what
the width does to the last digits.

## The pass {#streaming-transform-pass}

The source delivers the raw integrals in blocks of auxiliary rows. Inside a
block the code walks chunks $\mathcal{C}$ of at most $k$ rows. For each chunk
it transforms the rows to the MO basis, then adds their contribution to
every row of $B$:

$$ (Q|ia) = \sum_{\mu} C_{\mu i} \Big( \sum_{\nu} (Q|\mu\nu)\,C_{\nu a} \Big), \qquad Q \in \mathcal{C}, \label{eq:streaming-transform-1} $$

$$ B^P_{ia} \;\mathrel{+}=\; \sum_{Q \in \mathcal{C}} M_{PQ}\,(Q|ia) \qquad \text{for every } P \text{ in the output band.} \label{eq:rimp2alg-chunked-dressing} $$

Sum (@eq:rimp2alg-chunked-dressing) over all chunks and the chunks tile the full range of $Q$, which
gives equation @eq:four-step-algorithm-2. The order of the chunks fixes the order of the
floating-point sum over $Q$.

> **Unverified equation @eq:rimp2alg-chunked-dressing** · chunked dressing equals the full dressing
>
> For any partition of the auxiliary rows into chunks, the sum of (@eq:rimp2alg-chunked-dressing) over
> the chunks equals $B = M\,(Q|ia)$ in exact arithmetic. In floating point the
> results differ by summation order only.
>
> | | |
> |---|---|
> | Proof | none committed. The identity is linearity of the matrix product. To add: a claim in `scripts/mp2_variant_identities.py` that dresses a random $(Q\vert ia)$ with chunk widths 1, 32, 256 and the full width, with a dropped-chunk negative control. |
> | Code tests | `tests/mwe_mo_stream_chunk_budget.rs` pins the width rule (@sec:streaming-transform-chunk-width), not the identity |
>
> <!-- unverified: id=rimp2alg.chunked-dressing; todo=add chunk-partition claim to scripts/mp2_variant_identities.py -->

Here is the loop body of `stream_dressed_mo_band_budgeted`, with the long
comments replaced by the term each line computes and the serial branch of
the row loop removed:

```rust
let q1 = (q0 + chunk).min(qb);                  // chunk C = rows [q0, q1) of the block
let qc = q1 - q0;
let mut mo_blk = Array2::<f64>::zeros((qc, width)); // (Q|ia), k x n_o n_v
let mo_transform_row = |q: usize, mut row: ndarray::ArrayViewMut1<f64>| {
    let bq_ao = blk.data.slice(ndarray::s![q0 + q, .., ..]); // (Q|mu nu), N x N
    let half = c_left.t().dot(&bq_ao);          // sum_mu C_{mu i} (Q|mu nu): N^2 n_o
    let bq_mo = half.dot(c_right);              // ... C_{nu a}: N n_o n_v
    row.assign(&bq_mo.into_shape_with_order(width).unwrap());
};
// ... rows in parallel above PAR_MO_TRANSFORM_WORK_THRESHOLD ...
let msub = v_inv_sqrt.slice(ndarray::s![band_p0..band_p1, blk.p0 + q0..blk.p0 + q1]); // M_{P,C}
ferric_core::linalg::par_gemm(1.0, &msub, &mo_blk.view(), 1.0, &mut b_flat); // B += M_{P,C} (C|ia), beta = 1
```

**Order of the quarter-transforms.** Equation @eq:streaming-transform-1 contracts the occupied
index first. Per auxiliary row, `half = c_left.t().dot(&bq_ao)` costs
$N^2 n_o$, and `half.dot(c_right)` costs $N n_o n_v$. Contracting the
virtual index first would cost $N^2 n_v$ for the first product, a factor
$n_v/n_o$ more. Over both products the ratio is

$$ \frac{N^2 n_v + N n_o n_v}{N^2 n_o + N n_o n_v} , $$

which for benzene in aug-cc-pVTZ with a frozen core ($N = 414$,
$n_o = 15$, $n_v = 393$) is 13.9, against $n_v/n_o = 26.2$ for the first
product alone. The comment at the call site names PySCF, Psi4 and NWChem as
using the same occupied-first order.

**Parallelism.** Each auxiliary row of the chunk is an independent pair of
matrix products with a disjoint output row. Below
`PAR_MO_TRANSFORM_WORK_THRESHOLD` (200 000 output elements per chunk) the
rows run serially. Above it they run as rayon tasks, with BLAS pinned to one
thread inside each task. The dressing (@eq:rimp2alg-chunked-dressing) is one call to
`ferric-core/src/linalg.rs::par_gemm` with $\beta = 1$. `par_gemm` splits the
output into fixed column blocks, so its result does not depend on the thread
count.

**Why the dressing is one GEMM per chunk.** The dressing costs
$2\,(p_1 - p_0)\,k\,n_o n_v$ flops per chunk, $2N_{\mathrm{aux}}^2 n_o n_v$
over the whole pass for the full band. A matrix product runs near the
peak rate of the core only when it does many flops per byte it moves.
Suppose each operand of one chunk's product crosses the memory bus once: $M$
restricted to the chunk ($N_{\mathrm{aux}} \times k$), the MO chunk
($k \times n_o n_v$), and $B$, which $\beta = 1$ reads and writes
($2 N_{\mathrm{aux}} n_o n_v$). Then

$$ \frac{\text{flops}}{\text{bytes}} = \frac{2 N_{\mathrm{aux}}\, k\, n_o n_v}{8\,\big(N_{\mathrm{aux}} k + k\, n_o n_v + 2 N_{\mathrm{aux}} n_o n_v\big)} \;\longrightarrow\; \frac{k}{8} \qquad (N_{\mathrm{aux}},\, n_o n_v \gg k) . $$

The intensity is set by the chunk width alone: 32 flops per byte at
$k = 256$, 4 at $k = 32$. That is why the width has a floor (@sec:streaming-transform-chunk-width), and
why the doc comment of `MO_STREAM_CHUNK_MIN` says that a very small width
destroys BLAS3 efficiency.

**The output band.** `output_band = Some((p0, p1))` keeps only the rows
$P \in [p_0, p_1)$ of $B$. The pass still reads every raw auxiliary row,
because every $Q$ contributes to every $P$ through $M_{PQ}$. Section @sec:distributed-rimp2
uses this.

## Memory of the pass {#streaming-transform-memory-pass}

| Quantity | Size in doubles | Lifetime |
|---|---|---|
| Output $B$ (band) | $(p_1 - p_0)\,n_o n_v$ | returned to the caller |
| MO chunk $(Q\vert ia)$, $Q \in \mathcal{C}$ | $k\,n_o n_v$ | one chunk |
| Per-row half transform | $n_o N$ per active task | one row |
| Raw block from the source | set by the source tier (section @sec:memory-budget) | one block |

Only the first row grows as $N^3$. The MO chunk is $k/N_{\mathrm{aux}}$ of
it, at most 256 rows. A two-tensor pass would hold the undressed
$(P|ia)$ beside $B$, doubling the first row.

## The chunk width {#streaming-transform-chunk-width}

The width $k$ is set once per call:

| Call | Width |
|---|---|
| `stream_dressed_mo_band` (no budget) | `MO_STREAM_CHUNK` = 256 |
| `stream_dressed_mo_band_budgeted(.., Some(budget))` | `mo_stream_chunk_for(n_o n_v, budget)` |

Here is `mo_stream_chunk_for` with the source comments:

<!-- excerpt: ferric-mp2/src/rimp2.rs::mo_stream_chunk_for -->
```rust
let per_row_bytes = width.max(1).saturating_mul(8).saturating_mul(2);
let share = budget_bytes / 8;
let raw = (share / per_row_bytes.max(1)).max(MO_STREAM_CHUNK_MIN);
```

and after the ceiling and the rounding, the code returns
`(capped & !1).max(MO_STREAM_CHUNK_MIN)`. Read it line by line. The
transient may take one eighth of the budget. Each auxiliary row of the
chunk is charged two rows of $n_o n_v$ doubles: the MO chunk being filled,
and, by the function's doc comment, the raw slab that feeds it. (The raw
rows are $N^2$ wide and belong to the source's block, which the tier of
section @sec:memory-budget accounts for; the second row here is the code's allowance, not
the slab's size.) Divide, floor at 32, cap at 256, round down to even:

$$ k_{\mathrm{raw}} = \max\!\Big(32,\ \Big\lfloor \frac{\lfloor \mathrm{budget}/8 \rfloor}{16\,n_o n_v} \Big\rfloor\Big), \qquad k = \max\big(32,\ 2\lfloor \min(k_{\mathrm{raw}}, 256)/2 \rfloor\big) . \label{eq:streaming-transform-3} $$

The budget is in bytes. When does a budget narrow the chunk? $k = 256$
exactly when $\lfloor\lfloor \mathrm{budget}/8\rfloor/(16 n_o n_v)\rfloor \ge 256$,
that is $\lfloor \mathrm{budget}/8 \rfloor \ge 4096\,n_o n_v$. The right side
is an integer, so the floor can be dropped:

$$ k = 256 \iff \mathrm{budget} \ge 32768\; n_o n_v \ \text{bytes} . $$

For water in cc-pVDZ ($n_o n_v = 95$) that is 3.11 MB. For benzene in
cc-pVDZ ($n_o n_v = 1953$) it is 64.0 MB, so every budget in the benzene
band of section @sec:memory-budget-shared-pool (43.67 to 50.23 MB) narrows the chunk, to between 174
and 200 rows.
`[memory] budget_gb` is read in GiB (`ferric-core/src/memory.rs::gib_to_bytes`).

| Property | Why | Test in `crates/ferric-mp2/tests/mwe_mo_stream_chunk_budget.rs` |
|---|---|---|
| An ample budget gives 256 | the budgeted and unbudgeted calls agree bit for bit | `an_ample_budget_reproduces_the_historical_width` |
| A tight budget narrows $k$ | bounds the transient | `a_tight_budget_narrows_the_chunk`, `a_wider_mo_block_narrows_the_chunk` |
| $k$ is even | odd GEMM splits perturbed the last bits on the development machine (doc comment of `mo_stream_chunk_for`) | `the_width_is_always_even` |
| $k \ge 32$ | the pass always progresses, and the dressing keeps 4 flops per byte or more | `the_width_never_reaches_zero` |
| $k$ depends on $(n_o n_v, \mathrm{budget})$ only | reproducibility | `the_width_is_deterministic` |

## Reproducibility {#streaming-transform-reproducibility}

Because (@eq:rimp2alg-chunked-dressing) accumulates with $\beta = 1$, the chunk width is the blocking
of a floating-point sum over $Q$. Change $k$ and the partial sums change,
and with them the rounding. The MO-transform rows write disjoint outputs and
accumulate nothing, so only the dressing constrains $k$. Three rules follow.

- The width reads the resolved budget and the problem shape. It never reads
  the thread count, the free memory at call time or the shared pool's
  ledger. The driver passes the budget resolved at the start of the run,
  not `pool.available_bytes()`.
- Two runs at the same `[memory] budget_gb` give the same $B$ bit for bit.
- Two runs at different budgets can differ in the last digits, through $k$
  and through the block size of a spilled source (section @sec:memory-budget). Pin
  `budget_gb` to pin the digits.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/rimp2.rs`, doc comments of `mo_stream_chunk_for`, `MO_STREAM_CHUNK`, `MO_STREAM_CHUNK_MIN` and `stream_dressed_mo_band_budgeted`, and the loop body | equations @eq:streaming-transform-1 to @eq:streaming-transform-3, the excerpts and the rules in @sec:streaming-transform-reproducibility |
| `crates/ferric-mp2/tests/mwe_mo_stream_chunk_budget.rs` | the property table |
| `crates/ferric-mp2/theory.md`, "1. Density fitting" | the definition of $B$ |

Changes from the sources: the code's `v_inv_sqrt` is written $M$; the code's
`c_left`, `c_right` are $C_o$, $C_v$ for RI-MP2. The quarter-transform
ratio, the flops-per-byte limit and the $k = 256$ threshold are derived for
this section and checked numerically against a reimplementation of
`mo_stream_chunk_for`.
