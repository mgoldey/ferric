<!-- section: 11.2; status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting" -->

# 11.2 Streaming the MO transform and the dressing

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band_budgeted`, `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band`, `ferric-mp2/src/rimp2.rs::mo_stream_chunk_for` |
| Other callers | `ferric-mp2/src/rimp2.rs::eri3_mo_block_dressed_band` (MPI, section 11.7); the GW $B$ build in `ferric-gw/src/mo_b.rs` |
| Trace | `FERRIC_MO_STREAM_TRACE=1` logs the resolved chunk width |

Steps 2 and 3 of section 11.1 run in one pass over the raw three-index
tensor. The undressed $(P|ia)$ is never held whole.

## 11.2.1 The pass

The source delivers the raw integrals in blocks of auxiliary rows. Inside a
block the code walks chunks $\mathcal{C}$ of at most $k$ rows. For each chunk:

$$ (Q|ia) = \sum_{\mu} C_{\mu i} \Big( \sum_{\nu} (Q|\mu\nu)\,C_{\nu a} \Big), \qquad Q \in \mathcal{C}, \tag{11.4} $$

$$ B^P_{ia} \;\mathrel{+}=\; \sum_{Q \in \mathcal{C}} M_{PQ}\,(Q|ia) \qquad \text{for every } P \text{ in the output band.} \tag{11.5} $$

The sum of (11.5) over all chunks is equation 11.2. The order of the chunks
fixes the order of the floating-point sum over $Q$.

> **Unverified equation 11.5** · chunked dressing equals the full dressing
>
> For any partition of the auxiliary rows into chunks, the sum of (11.5) over
> the chunks equals $B = M\,(Q|ia)$ in exact arithmetic. In floating point the
> results differ by summation order only.
>
> | | |
> |---|---|
> | Proof | none committed. The identity is linearity of the matrix product. To add: a claim in `scripts/mp2_variant_identities.py` that dresses a random $(Q\vert ia)$ with chunk widths 1, 32, 256 and the full width, with a dropped-chunk negative control. |
> | Code tests | `tests/mwe_mo_stream_chunk_budget.rs` pins the width rule (11.2.3), not the identity |
>
> <!-- unverified: id=rimp2alg.chunked-dressing; todo=add chunk-partition claim to scripts/mp2_variant_identities.py -->

**Order of the quarter-transforms.** Equation 11.4 contracts the occupied
index first. Per auxiliary row, `half = c_left.t().dot(&bq_ao)` costs
$N^2 n_o$, and `half.dot(c_right)` costs $N n_o n_v$. Contracting the
virtual index first would cost $N^2 n_v$ for the first product, a factor
$n_v/n_o$ more. The comment at the call site names PySCF, Psi4 and NWChem as
using the same occupied-first order.

**Parallelism.** Each auxiliary row of the chunk is an independent pair of
matrix products with a disjoint output row. Below
`PAR_MO_TRANSFORM_WORK_THRESHOLD` (200 000 output elements per chunk) the
rows run serially. Above it they run as rayon tasks, with BLAS pinned to one
thread inside each task. The dressing (11.5) is one call to
`ferric-core/src/linalg.rs::par_gemm` with $\beta = 1$. `par_gemm` splits the
output into fixed column blocks, so its result does not depend on the thread
count.

**The output band.** `output_band = Some((p0, p1))` keeps only the rows
$P \in [p_0, p_1)$ of $B$. The pass still reads every raw auxiliary row,
because every $Q$ contributes to every $P$ through $M_{PQ}$. Section 11.7
uses this.

## 11.2.2 Memory of the pass

| Quantity | Size in doubles | Lifetime |
|---|---|---|
| Output $B$ (band) | $(p_1 - p_0)\,n_o n_v$ | returned to the caller |
| MO chunk $(Q\vert ia)$, $Q \in \mathcal{C}$ | $k\,n_o n_v$ | one chunk |
| Per-row half transform | $n_o N$ per active task | one row |
| Raw block from the source | set by the source tier (section 11.4) | one block |

## 11.2.3 The chunk width

The width $k$ is set once per call:

| Call | Width |
|---|---|
| `stream_dressed_mo_band` (no budget) | `MO_STREAM_CHUNK` = 256 |
| `stream_dressed_mo_band_budgeted(.., Some(budget))` | `mo_stream_chunk_for(n_o n_v, budget)` |

`mo_stream_chunk_for` takes one eighth of the budget and divides it by two
rows of $n_o n_v$ doubles (the MO chunk and the matching raw slab):

$$ k_{\mathrm{raw}} = \max\!\Big(32,\ \Big\lfloor \frac{\lfloor \mathrm{budget}/8 \rfloor}{16\,n_o n_v} \Big\rfloor\Big), \qquad k = \max\big(32,\ 2\lfloor \min(k_{\mathrm{raw}}, 256)/2 \rfloor\big) . \tag{11.6} $$

The budget is in bytes. The width is capped at 256, rounded down to an even
number and floored at 32.

| Property | Why | Test in `crates/ferric-mp2/tests/mwe_mo_stream_chunk_budget.rs` |
|---|---|---|
| An ample budget gives 256 | the budgeted and unbudgeted calls agree bit for bit | `an_ample_budget_reproduces_the_historical_width` |
| A tight budget narrows $k$ | bounds the transient | `a_tight_budget_narrows_the_chunk`, `a_wider_mo_block_narrows_the_chunk` |
| $k$ is even | odd GEMM splits perturbed the last bits on the development machine (doc comment of `mo_stream_chunk_for`) | `the_width_is_always_even` |
| $k \ge 32$ | the pass always progresses | `the_width_never_reaches_zero` |
| $k$ depends on $(n_o n_v, \mathrm{budget})$ only | reproducibility | `the_width_is_deterministic` |

## 11.2.4 Reproducibility

Because (11.5) accumulates with $\beta = 1$, the chunk width is the blocking
of a floating-point sum. Three rules follow.

- The width reads the resolved budget and the problem shape. It never reads
  the thread count, the free memory at call time or the shared pool's
  ledger. The driver passes the budget resolved at the start of the run,
  not `pool.available_bytes()`.
- Two runs at the same `[memory] budget_gb` give the same $B$ bit for bit.
- Two runs at different budgets can differ in the last digits, through $k$
  and through the block size of a spilled source (section 11.4). Pin
  `budget_gb` to pin the digits.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/rimp2.rs`, doc comments of `mo_stream_chunk_for` and `stream_dressed_mo_band_budgeted`, and the loop body | equations 11.4 to 11.6 and the rules in 11.2.4 |
| `crates/ferric-mp2/tests/mwe_mo_stream_chunk_budget.rs` | the property table |
| `crates/ferric-mp2/theory.md`, "1. Density fitting" | the definition of $B$ |

Changes from the sources: the code's `v_inv_sqrt` is written $M$; the code's
`c_left`, `c_right` are $C_o$, $C_v$ for RI-MP2.
