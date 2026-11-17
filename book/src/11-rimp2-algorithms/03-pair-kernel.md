<!-- status: ready; registry: ri-mp2 -->

# The pair-energy kernel {#pair-kernel}

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, `ferric-mp2/src/rimp2.rs::pair_energy` |
| f64-only wrapper | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov` (RS-MP2+RPA, OO-RI-MP2) |
| Dispatcher | `dispatch_spin_components` in `crates/ferric-mp2/src/rimp2.rs` (private): device first, then CPU |

Step 4 of section @sec:four-step-algorithm is the only $O(N^5)$ step. It forms every $(ia|jb)$ by
equation @eq:four-step-algorithm-3 and contracts it into the two spin components of equation @eq:mp2-cs-spin-adaptation.
There are $n_o^2 n_v^2$ integrals, and each is a sum of $N_{\mathrm{aux}}$
products. The kernel has to do that work as matrix products, visit each
pair once, hold no more than one block of integrals per worker, and add the
results in an order that does not depend on the thread count. The plan:
the block per $i$, the pair symmetry, the inner kernel, then the parallel
loop.

## Batching over the occupied index {#pair-kernel-batching-over-occupied-index}

For each $i$ the code forms one block:

$$ G^{(i)}_{a,\,(j-i)n_v + b} = \sum_P B^P_{ia}\,B^P_{jb}, \qquad j = i, \dots, n_o - 1 . \label{eq:pair-kernel-1} $$

In the code `b_i` is columns $[i n_v, (i+1) n_v)$ of `b_ov`, `b_tail` is
columns $[i n_v, n_o n_v)$, and `g_i = b_i.t().dot(&b_tail)`. The product is
$n_v \times N_{\mathrm{aux}}$ times $N_{\mathrm{aux}} \times (n_o - i) n_v$.
It is one GEMM per $i$, wide in the $j$ direction. This matches the
dissertation's choice of batch index (chapter 4, section 4.2) without the
disk: all of $B$ is in memory, so no $B$ matrix is read twice.

**The flop count.** The GEMM for $i$ does $2 \cdot n_v \cdot N_{\mathrm{aux}}
\cdot (n_o - i) n_v$ flops. Sum over $i$, using
$\sum_{i=0}^{n_o-1}(n_o - i) = \tfrac12 n_o(n_o+1)$:

$$ \sum_{i=0}^{n_o-1} 2\,n_v^2 (n_o - i)\, N_{\mathrm{aux}} = n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}} . $$

A full-width block (all $j$) would cost $2 n_o^2 n_v^2 N_{\mathrm{aux}}$.
The fraction the tail saves is

$$ 1 - \frac{n_o(n_o+1)}{2 n_o^2} = \frac{n_o - 1}{2 n_o} , $$

47% at $n_o = 15$. The $j < i$ blocks would be mirror images of blocks
another task already uses (section @sec:pair-kernel-symmetry-only), so they are not formed. Check
the limit: at $n_o = 1$ the tail is the whole block and the saving is zero.

**Why a GEMM and not dot products.** Each $(ia|jb)$ is a dot product of
two columns of $B$ of length $N_{\mathrm{aux}}$: $2N_{\mathrm{aux}}$ flops
against $16N_{\mathrm{aux}}$ bytes read, $1/8$ flop per byte if nothing is
reused. The block product reads $B_i$ once for all $j$. If each operand of
the GEMM for $i$ crosses the memory bus once, with tail width
$n = (n_o - i)n_v$, the intensity tends for a wide tail ($n$ large against
$n_v$ and $N_{\mathrm{aux}}$) to

$$ \frac{2\,n_v N_{\mathrm{aux}}\, n}{8\,(n_v N_{\mathrm{aux}} + N_{\mathrm{aux}} n + n_v n)} \;\longrightarrow\; \frac{n_v N_{\mathrm{aux}}}{4\,(n_v + N_{\mathrm{aux}})} , $$

69 flops per byte for benzene in aug-cc-pVTZ ($n_v = 393$,
$N_{\mathrm{aux}} = 912$). The flop count is the same as for the dot
products; the GEMM moves about 550 times fewer bytes per flop in this
model.

| Quantity | Value |
|---|---|
| GEMM flops, all $i$ | $\sum_i 2\,n_v^2 (n_o - i) N_{\mathrm{aux}} = n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ |
| Pair-loop work | a few operations per $(a,b)$ per pair, about $N_{\mathrm{aux}}$ times less than the GEMM |
| Largest block | $n_v \times n_o n_v$ doubles, at $i = 0$ |

The largest block is $8 n_o n_v^2$ bytes, 18.5 MB for benzene in
aug-cc-pVTZ. Holding every block at once would take $8 n_o^2 n_v^2$ bytes,
0.28 GB there, and it grows as the fourth power of the size.

## Symmetry: only $j \ge i$ {#pair-kernel-symmetry-only}

Both spin components are invariant under the joint swap
$(i,a) \leftrightarrow (j,b)$: the swap maps $(ia|jb)$ to itself and
$\Delta_{ij}^{ab}$ to itself, and relabelling $a \leftrightarrow b$ maps the
exchange partner $(ib|ja)$ back to itself. So each off-diagonal pair stands
in for its mirror with weight 2. Verified equation @eq:mp2-cs-unique-pairs states this and its
proof (`scripts/mp2_variant_identities.py`, claim 1, with the weight-1
negative control).

$$ E_{\mathrm{OS}} = \sum_i \sum_{j \ge i} f_{ij}\, e^{\mathrm{OS}}_{ij}, \qquad E_{\mathrm{SS}} = \sum_i \sum_{j \ge i} f_{ij}\, e^{\mathrm{SS}}_{ij}, \qquad f_{ij} = \begin{cases} 1 & i = j \\ 2 & i < j \end{cases} \label{eq:rimp2alg-tail-block-pairs} $$

> **Verified equation @eq:rimp2alg-tail-block-pairs** · the unique-pair loop of the production kernel
>
> The loop over $j \ge i$ with weights $f_{ij}$, reading $(ia|jb)$ and
> $(ib|ja)$ from the tail block (@eq:pair-kernel-1), reproduces the full $(i,j)$ sum.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 (unique-pair checks) |
> | Negative control | off-diagonal weight 1 must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu` |
> | Code test | `spin_components_symmetry_matches_naive_double_loop` in `crates/ferric-mp2/src/rimp2.rs`: the production loop against a full $(i,j)$ double loop on H2O/cc-pVDZ, relative $10^{-12}$ |
>
> <!-- verified: id=rimp2alg.tail-block-pairs; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu; rust_test=ferric-mp2/src/rimp2.rs::spin_components_symmetry_matches_naive_double_loop -->

## The kernel {#pair-kernel-kernel}

`pair_energy` reads one $j$ from the tail block. The column offset of $j$ is
`jcol` $= (j - i)\,n_v$.

<!-- excerpt: ferric-mp2/src/rimp2.rs::pair_energy -->
```rust
let g_ab = g_i[(a, jcol + b)]; // (ia|jb)
let g_ba = g_i[(b, jcol + a)]; // (ib|ja)
let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
e_os_ij += g_ab * g_ab / denom;
e_ss_ij += g_ab * (g_ab - g_ba) / denom;
```

`denom` is $-\Delta_{ij}^{ab}$, so the kernel has no leading minus sign
(section @sec:closed-shell-ri-mp2-energy-splits). With `kappa` set, each term is multiplied by
`(1 - (k * denom).exp())^2` $= (1 - e^{-\kappa\Delta_{ij}^{ab}})^2$
(section @sec:scs-and-kappa).

The exchange partner $(ib|ja)$ is in the same block as $(ia|jb)$, with $a$
and $b$ exchanged. So one block per $i$ holds every integral the pair needs.
Section @sec:closed-shell-ri-mp2-energy-splits reads this loop line by line.

## The parallel loop and its fixed summation order {#pair-kernel-parallel-loop-fixed-summation}

Here is the parallel loop of `spin_components_from_b_ov_kappa_cpu` with the
source comments replaced by the term each line computes:

```rust
let partials: Vec<(f64, f64)> = (0..nocc)
    .into_par_iter()                                  // one rayon task per i
    .map(|i| {
        let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]); // B_{P,ia}
        let b_tail = b_ov.slice(ndarray::s![.., i * nvir..]);           // B_{P,jb}, j >= i
        let g_i = b_i.t().dot(&b_tail);                                 // G^(i), (11.7)

        let (mut e_os_i, mut e_ss_i) = (0.0, 0.0);
        for j in i..nocc {
            let fac = if i == j { 1.0 } else { 2.0 };                   // f_ij, (11.8)
            let (e_os_ij, e_ss_ij) =
                pair_energy(&g_i.view(), i, j, nvir, eps, first_occ, nocc_total, kappa);
            e_os_i += fac * e_os_ij;
            e_ss_i += fac * e_ss_ij;
        }
        (e_os_i, e_ss_i)
    })
    .collect();                                       // partials kept in order of i
```

After the loop, a serial `for` adds the partials in order of $i$.

| Choice | Effect |
|---|---|
| One rayon task per $i$, with its GEMM inside the task | the GEMMs, which carry most of the work, run in parallel |
| BLAS at one thread inside each task | no nested threading (section @sec:determinism-and-memory) |
| Partials collected in $i$ order, summed on one thread | the total does not depend on the worker count; a rayon `reduce` would combine in a tree whose shape does |
| $j, a, b$ loops serial inside a task | the per-$i$ partial is fixed |

**Load balance.** Task $i$ does $(n_o - i)/n_o$ of the work of task 0. Its
share of the total is

$$ \frac{n_o}{\tfrac12 n_o(n_o+1)} = \frac{2}{n_o+1} \quad\text{for } i = 0 . $$

No schedule can finish before its largest task, so the parallel speedup of
this stage is at most $(n_o+1)/2$, whatever the worker count: 8 at
$n_o = 15$. Rayon's work stealing balances the tasks when $n_o$ is several
times the worker count. When $n_o + 1 < 2 n_w$, task 0 alone is longer than
an even share $1/n_w$, and the first tasks set the wall time.

**What is pinned.** The CPU stage is designed to be bit-identical across
`RAYON_NUM_THREADS`.
`mo_stream_dressing_uses_the_pool_and_is_thread_count_independent` in
`crates/ferric-mp2/tests/mo_stream_dressing_threads.rs` runs
`ri_mp2_spin_components` on H2O/cc-pVDZ in rayon pools of 1, 2 and 6 threads
and asserts the same bits for $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ and $B$.
The device pair fold has its own test (section @sec:device-mixed-precision). The related test
`ri_mp2_spin_components_consistent_across_blas_thread_counts` in
`crates/ferric-mp2/tests/blas_raise_identity.rs` varies
`FERRIC_BLAS_THREADS` and asserts agreement near $10^{-13}$, not bit
identity.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.2, Function 4 | the batch index |
| `crates/ferric-mp2/src/rimp2.rs`, the loop body and comments of `spin_components_from_b_ov_kappa_cpu` | equations @eq:pair-kernel-1 and @eq:rimp2alg-tail-block-pairs, the order rules, the excerpt |
| `scripts/mp2_variant_identities.py`, claim 1 | verified equation @eq:rimp2alg-tail-block-pairs |
| `crates/ferric-mp2/tests/mo_stream_dressing_threads.rs` | the thread-count test |
| `book/sample-ri-mp2.md`, sections 10.1.3 and 10.1.4 | the kernel excerpt, extended here |

Changes from the sources: the tail-block column index is written
$(j - i)\,n_v + b$; the weight is written $f_{ij}$ (the code's `fac`). The
flop sum, the saved fraction, the flops-per-byte limits and the speedup
bound $(n_o+1)/2$ are derived for this section and checked with sympy.
