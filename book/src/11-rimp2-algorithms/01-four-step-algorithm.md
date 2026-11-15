<!-- section: 11.1; status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting", "2. Closed-shell RI-MP2" -->

# 11.1 The four-step algorithm, 2014 and now

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` |
| CLI | `method.kind = "rimp2"`, sections `[mp2]` and `[memory]` |
| Python | `run_rimp2` (`crates/ferric-python/ferric.pyi`) |

The closed-shell RI-MP2 energy (section 10.1) needs three objects: the
metric factor $M$, the dressed factor $B$, and, for each occupied pair, the
block of $(ia|jb)$ integrals.

$$ V_{PQ} = (P|Q), \qquad M^{\mathsf T}M = V^{-1} . \tag{11.1} $$

$$ (P|ia) = \sum_{\mu\nu} C_{\mu i}\,(P|\mu\nu)\,C_{\nu a}, \qquad B^P_{ia} = \sum_Q M_{PQ}\,(Q|ia) . \tag{11.2} $$

$$ (ia|jb) \approx \sum_P B^P_{ia} B^P_{jb} . \tag{11.3} $$

Equations 11.1 to 11.3 are equation 10.4 split into the steps that the
algorithm schedules.

## 11.1.1 The 2014 algorithm

The dissertation's algorithm (chapter 4, section 4.2) has four functions,
one per row of its Table 4.1. Its intermediates went to disk between
functions. The table below carries its cost, memory and disk columns in this
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
quarter-transform and $N n_o n_v N_{\mathrm{aux}}$ for the second, which is
the entry above.

How each function ran (chapter 4, section 4.2):

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

## 11.1.2 The algorithm in the code

| Step | Quantity | Cost | Lives in | Function |
|---|---|---|---|---|
| 1 | $V$, then $M = L^{-1}$ | $N_{\mathrm{aux}}^3$ | memory, $N_{\mathrm{aux}}^2$ | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c`, `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` |
| 2a | raw $(P\vert\mu\nu)$ | integrals | memory, packed memory or disk (section 11.4) | `ferric-integrals/src/three_index_source.rs::ThreeIndexSource` |
| 2b, 3 | $(P\vert ia)$ and $B$, one pass | $N^2 n_o N_{\mathrm{aux}} + N n_o n_v N_{\mathrm{aux}} + n_o n_v N_{\mathrm{aux}}^2$ | $B$ in memory, $N_{\mathrm{aux}} n_o n_v$ | `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band_budgeted` |
| 4 | $G^{(i)}$ blocks and the energy | $\tfrac12 n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ | one block per worker | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, `ferric-mp2/src/rimp2.rs::pair_energy` |

The driver runs these steps in this order:

1. The metric $V$ and its factor $M$.
2. The budget, resolved once: `ferric-mp2/src/rimp2.rs::eri3_budget_bytes`.
3. The admission check on the MO-side blocks:
   `ferric-mp2/src/rimp2.rs::check_mo_side_alloc` (section 11.4).
4. The raw three-index source, optionally screened by distance
   (`eri3_screen_thresh`).
5. The streamed transform and dressing into $B$ (section 11.2).
6. `drop(src)`. The raw tensor is freed before the energy loop, which never
   reads it.
7. The pair energies (section 11.3), on the device when one is configured
   and usable (section 11.6), otherwise on the CPU.

## 11.1.3 What changed

| | 2014 (dissertation) | Now (code) |
|---|---|---|
| Metric factor | $V^{-1/2}$ by ScaLAPACK, on disk | $L^{-1}$ by Cholesky, in memory; eigendecomposition with a cut for erf and terf (section 10.1.2) |
| $(P\vert ia)$ | written to disk, $(a,P,i)$ order | never stored whole; each auxiliary chunk is dressed as soon as it is transformed |
| $B$ | written to disk | in memory for the whole energy evaluation |
| What spills | $(ia\vert P)$ and $B$ | only the raw AO tensor $(P\vert\mu\nu)$, when it does not fit the budget |
| Function 4 | batch of $n_B$ in memory, the rest read from disk | all of $B$ in memory; one $G^{(i)}$ block per task |
| Pair symmetry | $j > i$ read from disk | $j \ge i$ tail GEMM, off-diagonal weight 2 (section 11.3) |
| Parallelism | OpenMP threads | rayon tasks over $i$ (CPU), a device GEMM (section 11.6), auxiliary bands over MPI ranks (section 11.7) |
| Memory control | batch sizes | a resolved budget, admission checks and a shared pool (section 11.4) |

The change of storage follows from the memory of current nodes: $B$ is
$N_{\mathrm{aux}} n_o n_v$ doubles, smaller than the raw AO tensor
$N_{\mathrm{aux}} N^2$ whenever $n_o n_v < N^2$, which holds for every basis.
Keeping the smaller tensor in memory and spilling the larger one is the
reverse of the 2014 choice, which wrote both MO tensors.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.2 (Table 4.1, Functions 1 to 4) | section 11.1.1 and the left column of 11.1.3 |
| `book/sample-ri-mp2.md`, section 10.1.3 | the step table, rewritten for this chapter |
| `crates/ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` | the order of steps in 11.1.2 |

Changes from the sources: the dissertation's $O, V, X$ are written $n_o, n_v,
N_{\mathrm{aux}}$; its thread count $n$ is written $n_t$; Function 2's cost
entry carries the factor $N$ that the printed table lacks; the dressing
matrix is $M$, and the code's $M$ is the Cholesky inverse rather than
$V^{-1/2}$.
