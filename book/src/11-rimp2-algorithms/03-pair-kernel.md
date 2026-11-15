<!-- section: 11.3; status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"2. Closed-shell RI-MP2" -->

# 11.3 The pair-energy kernel

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, `ferric-mp2/src/rimp2.rs::pair_energy` |
| f64-only wrapper | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov` (RS-MP2+RPA, OO-RI-MP2) |
| Dispatcher | `dispatch_spin_components` in `crates/ferric-mp2/src/rimp2.rs` (private): device first, then CPU |

Step 4 of section 11.1 is the only $O(N^5)$ step. It forms every $(ia|jb)$ by
equation 11.3 and contracts it into the two spin components of equation 10.2.
The integrals are never stored beyond one block.

## 11.3.1 Batching over the occupied index

For each $i$ the code forms one block:

$$ G^{(i)}_{a,\,(j-i)n_v + b} = \sum_P B^P_{ia}\,B^P_{jb}, \qquad j = i, \dots, n_o - 1 . \tag{11.7} $$

In the code `b_i` is columns $[i n_v, (i+1) n_v)$ of `b_ov`, `b_tail` is
columns $[i n_v, n_o n_v)$, and `g_i = b_i.t().dot(&b_tail)`. The product is
$n_v \times N_{\mathrm{aux}}$ times $N_{\mathrm{aux}} \times (n_o - i) n_v$.
It is one GEMM per $i$, wide in the $j$ direction. This is the dissertation's
choice of batch index (chapter 4, section 4.2) without the disk: all of $B$
is in memory, so no $B$ matrix is read twice.

Only the $j \ge i$ tail is formed. The $j < i$ blocks would be the mirror
images of blocks another task already uses (section 11.3.2), so computing
them would add $(n_o - 1)/(2 n_o)$ of the GEMM work for nothing.

| Quantity | Value |
|---|---|
| GEMM flops, all $i$ | $\sum_i 2\,n_v^2 (n_o - i) N_{\mathrm{aux}} = n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ |
| Pair-loop work | a few operations per $(a,b)$ per pair, about $N_{\mathrm{aux}}$ times less than the GEMM |
| Largest block | $n_v \times n_o n_v$ doubles, at $i = 0$ |

## 11.3.2 Symmetry: only $j \ge i$

Both spin components are invariant under the joint swap
$(i,a) \leftrightarrow (j,b)$, so each off-diagonal pair stands in for its
mirror with weight 2. Verified equation 10.6 states this and its proof
(`scripts/mp2_variant_identities.py`, claim 1, with the weight-1 negative
control).

$$ E_{\mathrm{OS}} = \sum_i \sum_{j \ge i} f_{ij}\, e^{\mathrm{OS}}_{ij}, \qquad E_{\mathrm{SS}} = \sum_i \sum_{j \ge i} f_{ij}\, e^{\mathrm{SS}}_{ij}, \qquad f_{ij} = \begin{cases} 1 & i = j \\ 2 & i < j \end{cases} \tag{11.8} $$

> **Verified equation 11.8** · the unique-pair loop of the production kernel
>
> The loop over $j \ge i$ with weights $f_{ij}$, reading $(ia|jb)$ and
> $(ib|ja)$ from the tail block (11.7), reproduces the full $(i,j)$ sum.
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

## 11.3.3 The kernel

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
(section 10.1.1). With `kappa` set, each term is multiplied by
`(1 - (k * denom).exp())^2` $= (1 - e^{-\kappa\Delta_{ij}^{ab}})^2$
(section 10.3).

The exchange partner $(ib|ja)$ is in the same block as $(ia|jb)$, with $a$
and $b$ exchanged. That is why one block per $i$ holds every integral the
pair needs.

## 11.3.4 The parallel loop and its fixed summation order

Abridged from `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu` (not a verbatim excerpt):
```rust
let partials: Vec<(f64, f64)> = (0..nocc)
    .into_par_iter()
    .map(|i| {
        let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
        let b_tail = b_ov.slice(ndarray::s![.., i * nvir..]);
        let g_i = b_i.t().dot(&b_tail);
        // j = i..nocc, fac = 1 or 2, pair_energy(..)
        (e_os_i, e_ss_i)
    })
    .collect();
// then a serial sum over partials in order of i
```

| Choice | Effect |
|---|---|
| One rayon task per $i$, with its GEMM inside the task | the GEMMs, which carry most of the work, run in parallel |
| BLAS at one thread inside each task | no nested threading (section 2.3) |
| Partials collected in $i$ order, summed on one thread | the total does not depend on the worker count; a rayon `reduce` would combine in a tree whose shape does |
| $j, a, b$ loops serial inside a task | the per-$i$ partial is fixed |

The tasks have unequal sizes: task $i$ does $(n_o - i)/n_o$ of the work of
task 0. Rayon's work stealing balances them when $n_o$ is several times the
worker count. When $n_o$ is close to the worker count, the first tasks set
the wall time.

**What is and is not pinned.** The CPU stage is designed to be bit-identical
across `RAYON_NUM_THREADS`. No committed test runs the CPU closed-shell
energy at two rayon thread counts and compares the bits. The device pair fold
has such a test (section 11.6). The related committed test,
`ri_mp2_spin_components_consistent_across_blas_thread_counts` in
`crates/ferric-mp2/tests/blas_raise_identity.rs`, varies `FERRIC_BLAS_THREADS`
and asserts agreement near $10^{-13}$, not bit identity.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.2, Function 4 | the batch index |
| `crates/ferric-mp2/src/rimp2.rs`, the loop body and comments of `spin_components_from_b_ov_kappa_cpu` | equations 11.7 and 11.8, the order rules |
| `scripts/mp2_variant_identities.py`, claim 1 | verified equation 11.8 |
| `book/sample-ri-mp2.md`, sections 10.1.3 and 10.1.4 | the kernel excerpt, extended here |

Changes from the sources: the tail-block column index is written
$(j - i)\,n_v + b$; the weight is written $f_{ij}$ (the code's `fac`).
