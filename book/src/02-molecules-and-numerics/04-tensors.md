# Dense tensors and the einsum macro {#tensors}

<!-- status: edit; registry: none -->

Correlation methods are written as sums over products of tensors with up to
four indices: amplitudes $t_{ij}^{ab}$, integrals $(ia|jb)$, fitted factors
$B^P_{ia}$. Written as nested loops, each such sum runs at a fraction of the
machine's floating-point rate. Written as a matrix product, the same sum runs
in BLAS near peak FLOP rate. The tensor crate lets a method write a contraction once, as
an index string, and turns it into matrix products.

| Entry point | Location |
|---|---|
| Labeled tensor | `ferric-tensors/src/tensor.rs::Tensor`, `ferric-tensors/src/axis.rs::Axis` |
| Contraction | `ferric-tensors/src/einsum.rs::{einsum_binary,einsum_binary_batched,einsum_dyn}`, `ferric-tensors/src/einsum_chain.rs::einsum_chain_dyn` |
| Macro | `einsum!`, in `crates/ferric-tensor-macro/src/lib.rs` |
| Python | `ferric.einsum` |

## Storage {#tensors-storage}

`Tensor<N>` wraps a dense `f64` array of $N$ axes and adds one label per axis:
`Ao`, `O`, `V`, `Aux` or `Any`. The label checks run in debug builds only.

What do the labels catch? A debug build checks that each contracted and each
batch axis carries the same label in both operands. If a caller passes a
virtual axis where an occupied one belongs, the shapes catch it only when
$n_o \ne n_v$, and in
the small test systems where code is debugged $n_o$ and $n_v$ are often both
small and may coincide. The labels catch the swap whatever the sizes. In a
release build they compile out and cost nothing. `Tensor` dereferences to the
inner `ndarray` array, so all ndarray code works on it unchanged. The crate
has no sparse format.

## The contraction {#tensors-contraction}

A contraction spec such as `ijab,iajb->` names the index of each operand. The
grammar is stricter than numpy's. Every index falls in one class:

| Appears in | Class | Effect |
|---|---|---|
| both inputs, not the output | contracted | summed; the inner axis of the matrix product |
| both inputs and the output | batch | one matrix product per batch slice |
| one input and the output | free | kept |
| one input only | error | no implicit sum |

The classes map onto a matrix product directly. Collect the free indices of
the left operand into one row index, the contracted indices into one inner
index, and the free indices of the right operand into one column index. Then
the contraction is $C_{\text{row},\text{col}} = \sum_{\text{inner}}
A_{\text{row},\text{inner}}B_{\text{inner},\text{col}}$.

Rank-4 coupled-cluster contractions are matrix products in the same way:

$$ C_{ij}^{ab} = \sum_{kc} A_{ik}^{ac}B_{kj}^{cb} \;\Longleftrightarrow\; C_{(ia),(jb)} = \sum_{(kc)} A_{(ia),(kc)}B_{(kc),(jb)} . $$

Here $(ia)$ is one row index of length $n_on_v$. To get there, permute
$A_{ik}^{ac}$ so its axes run $i,a,k,c$ and reshape to an
$n_on_v \times n_on_v$ matrix, likewise $B$ to $k,c,j,b$, multiply, and
reshape the product back to four axes. One contraction is one GEMM of
$n_o^3n_v^3$ multiply-adds, the same count as the four nested loops over
$i, a, j, b$ and the sum over $k, c$.

The engine does those steps. It permutes each operand so that each group is
contiguous, reshapes to a matrix, and calls a k-blocked matrix product. A
permutation can cost a full copy of the operand, so the code avoids it where
it can. From `ferric-tensors/src/einsum.rs::to_2d_or_transpose`, with the
row and column counts computed above it removed:

<!-- excerpt: ferric-tensors/src/einsum.rs::to_2d_or_transpose -->
```rust
let fwd: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
if fwd.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
    if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(rows, cols)) {
        return Operand2d::Borrowed(v);                 // already (first, second): a view
    }
}
let rev: Vec<usize> = second.iter().chain(first.iter()).copied().collect();
if rev.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
    if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(cols, rows)) {
        log::debug!("einsum: {which} operand consumed transposed (no copy)");
        return Operand2d::Borrowed(v.reversed_axes()); // (second, first): transposed view
    }
}
Operand2d::Owned(to_2d(op, first, second, which))  // interleaved: permute and copy
```

The second branch is the transpose case. A permutation that swaps the two
groups is absorbed into the transpose flag of the product and costs no copy.
The source comment in `einsum_binary` counts 72 of the roughly 130 specs in
the workspace with at least one operand in this class, the CCSD VVVV term
`ijcd,abcd->ijab` among them. Only an interleaved order, where the two groups
alternate in memory, is copied.

Three or more operands are a fixed left-to-right chain of binary
contractions, with no search for a cheaper order, so the result is a
deterministic function of the spec. The full grammar and examples are in the
user guide, `site/src/reference/einsum.md`.

The output lists its indices in group order: batch, then the left operand's
free indices, then the right operand's. The rank-4 example above therefore
returns $C$ with axes $i,a,j,b$, and the caller permutes it to $i,j,a,b$ if it
needs that order. A spec that interleaves the two groups, such as
`ikac,kjcb->ijab`, is an error.

### The `einsum!` macro {#tensors-einsum-macro}

Method code writes a contraction as one macro call. The MP3
particle–particle ladder is

```rust
let x: ArrayD<f64> = einsum!("ijab,abcd->ijcd", &t, &vvvv);
```

with $a, b$ contracted and $i, j, c, d$ free. Three properties come from
`crates/ferric-tensor-macro/src/lib.rs`:

- **The spec is parsed at compile time.** A malformed spec, or one that
  breaks a class rule of the table above, fails the build. The macro emits
  a call to `einsum_binary_batched` with the axis groups already worked out.
- **A scalar spec returns `f64`.** `einsum!("ijab,ijab->", &t, &v)` is the
  full contraction of two tensors, the form every energy takes.
- **A shape mismatch panics, by design.** Inside one function every operand
  is built from the same validated dimensions, so a mismatch can only be a
  bug in the spec, and the macro treats it like an out-of-bounds index.
  `einsum_dyn` returns an error instead.

An optional third argument scales the result. With a batch index the
product runs once per slice. In

```rust
let c: ArrayD<f64> = einsum!("kij,kij->ij", &a, &a, 0.5);
```

$k$ is contracted and $i, j$ are batch indices, so the result is
$c_{ij} = \tfrac12\sum_k a_{kij}^2$, one length-$k$ dot product per $(i, j)$.

### Why the matrix product is k-blocked

The product the engine calls does not hand the whole inner dimension to one
GEMM. It splits the contracted axis into blocks of `GEMM_K_BLOCK` = 128 and
accumulates the blocks in ascending order, in
`ferric-tensors/src/einsum.rs::gemm_kblocked`:

<!-- excerpt: ferric-tensors/src/einsum.rs::gemm_kblocked -->
```rust
let mut k0 = 0;
while k0 < k {
    let k1 = (k0 + GEMM_K_BLOCK).min(k);
    let lb = left.slice(ndarray::s![.., k0..k1]);      // A[:, k0..k1]
    let rb = right.slice(ndarray::s![k0..k1, ..]);     // B[k0..k1, :]
    let beta = if k0 == 0 { 0.0 } else { 1.0 };        // first block initializes C
    general_mat_mul(1.0, &lb, &rb, beta, out);         // C += A_blk B_blk
    k0 = k1;
}
```

The early return for $k \le 128$, which makes one plain GEMM, has been
removed. Each block's partial product is formed on its own and then added,
which is a coarse pairwise summation: rounding error grows like
$\log(k/128)$ instead of $k$. The function's documentation measures it at
1.10 to 1.40 times more accurate than one full-$k$ GEMM against a compensated
reference. Blocking does not by itself make the product independent of the
OpenBLAS thread count, since OpenBLAS also splits work inside a block.

## Reproducibility {#tensors-reproducibility}

The result is a deterministic function of the spec and the operands, and does
not depend on the worker count (section @sec:determinism-and-memory). The batched path rounds each
slice exactly as an unbatched call would, which a test pins
(`batched_is_bit_identical_to_loop_of_unbatched` in
`crates/ferric-tensors/tests/batched_parity.rs`).

For a large product the engine splits the output rows into bands and gives
each band to a rayon worker, with OpenBLAS at one thread inside. Row $i$ of
$AB$ depends only on row $i$ of $A$, and each band runs the same ascending-k
block loop, so in exact arithmetic the split changes nothing. OpenBLAS does
not honor that at the bit level for every split. Its dgemm microkernel
unrolls the row axis by 2, so a band with an odd row count finishes its last
row on a different code path with a different accumulation order, and the
result differs from the unbanded product by far more than rounding. So every
band has an even row count; the test
`row_banded_gemm_is_thread_count_independent` in the same file pins it.
With that rule every band width gives the same bits as the unbanded product,
and the width may depend on the worker count without changing any result.

## Known limits

Checked 2026-11-19.

- Single-operand sums, traces and diagonals are errors by design.
- The crate has no sparse format.

## Sources

| Source | Used for |
|---|---|
| `site/src/reference/einsum.md` | the grammar and classes |
| `crates/ferric-tensors/src/einsum.rs`, module documentation | the engine steps |

Changes from the sources: none to the equations.
