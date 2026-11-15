# 2.4 Dense tensors and the einsum macro

<!-- section: 2.4; status: edit; registry: ;
     theory: crates/ferric-tensors/theory.md#"Tensor Operations and Sparsity" -->

The correlation methods contract tensors with up to four indices. They write
a contraction once, as a spec string, and the engine turns it into matrix
products.

| Entry point | Location |
|---|---|
| Labeled tensor | `ferric-tensors/src/tensor.rs::Tensor`, `ferric-tensors/src/axis.rs::Axis` |
| Contraction | `ferric-tensors/src/einsum.rs::{einsum_binary,einsum_binary_batched,einsum_dyn}`, `ferric-tensors/src/einsum_chain.rs::einsum_chain_dyn` |
| Python | `ferric.einsum` |

## 2.4.1 Storage

`Tensor<N>` wraps a dense `f64` array of $N$ axes and adds one label per axis:
`Ao`, `O`, `V`, `Aux` or `Any`. The label checks run in debug builds only. They
catch an occupied–virtual swap in a contraction, which the shapes alone cannot,
because $n_o$ and $n_v$ are often both small and may coincide. The crate has no
sparse format.

## 2.4.2 The contraction

A contraction spec such as `ijab,iajb->` names the index of each operand. The
grammar is stricter than numpy's. Every index falls in one class:

| Appears in | Class | Effect |
|---|---|---|
| both inputs, not the output | contracted | summed; the inner axis of the matrix product |
| both inputs and the output | batch | one matrix product per batch slice |
| one input and the output | free | kept |
| one input only | error | no implicit sum |

The engine permutes each operand so that each group is contiguous, reshapes to
a matrix, and calls a k-blocked matrix product. A permutation that is a swap
of the two groups is absorbed into the transpose flag of the product and
costs no copy. Three or more operands are a fixed left-to-right chain of binary
contractions, with no search for a cheaper order, so the result is a
deterministic function of the spec. The full grammar and examples are in the
user guide, `site/src/reference/einsum.md`.

Rank-4 coupled-cluster contractions are matrix products in the same way:

$$ C_{ij}^{ab} = \sum_{kc} A_{ik}^{ac}B_{kj}^{cb} \;\Longleftrightarrow\; C_{(ia),(jb)} = \sum_{(kc)} A_{(ia),(kc)}B_{(kc),(jb)} . $$

## 2.4.3 Reproducibility

The result is a deterministic function of the spec and the operands, and does
not depend on the worker count (section 2.3). The batched path rounds each
slice exactly as an unbatched call would, which a test pins
(`batched_is_bit_identical_to_loop_of_unbatched` in
`crates/ferric-tensors/tests/batched_parity.rs`).

## Known limits

- Single-operand sums, traces and diagonals are errors by design.
- `crates/ferric-tensors/theory.md` states that the AO-Laplace code is dense,
  and that no $O(N)$ scaling is implemented or measured.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-tensors/theory.md` | storage and the matrix form of a contraction |
| `site/src/reference/einsum.md` | the grammar and classes |
| `crates/ferric-tensors/src/einsum.rs`, module documentation | the engine steps |

Changes from the sources: none to the equations.
