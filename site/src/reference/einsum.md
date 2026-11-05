# Einsum spec grammar

ferric contracts two tensors under a spec string such as `ij,jk->ik`. The same parser serves the Rust `einsum!` macro, `ferric_tensors::einsum_dyn` and the Python function `ferric.einsum`.

The grammar is stricter than `numpy.einsum`. A spec that breaks a rule is an error; there is no fallback to another algorithm.

Every example below is a test in `crates/ferric-tensors/tests/einsum_spec_examples.rs` with a hand-computed expected value. A further test fails if a spec tested there is missing from this page. The parser is `crates/ferric-einsum-spec/src/lib.rs`.

## Entry points

| Entry point | Spec checked | Scale | Scalar spec returns |
|---|---|---|---|
| `einsum!(spec, &a, &b)` (Rust macro) | at compile time; the spec must be a string literal | optional fourth argument, default `1.0` | `f64` |
| `einsum_dyn(spec, a, b, scale)` (Rust) | at run time, as `TensorError` | `scale: f64` | 0-dimensional `ArrayD<f64>` |
| `ferric.einsum(spec, a, b, scale=1.0)` (Python) | at run time, as `ValueError` | `scale` keyword | Python `float` |

The macro takes `Tensor` or `ArrayD<f64>` operands. For `Tensor` operands a debug build also checks that contracted and batch axes carry the same `Axis` label. A shape mismatch inside a macro call panics; `einsum_dyn` returns an error instead.

## Grammar

```text
spec   = left "," right "->" output
left, right, output = zero or more index characters
```

- Spaces are removed before parsing.
- An index is one `char`, and `i` and `I` are different indices. The parser splits on the first `->` and on `,`, so those characters cannot be indices.
- The spec must contain `->` and exactly two comma-separated inputs.
- No index may repeat within one term, including the output. Traces and diagonals of one operand (`ii,...`) are rejected.
- Each operand has as many axes as its term has indices; `einsum_dyn` checks this.

## Index classes

Each index falls in one class by where it appears.

| Appears in | Class | Effect |
|---|---|---|
| both inputs, not the output | contracted | summed; the GEMM inner axis |
| both inputs and the output | batch | element-wise; one GEMM per batch slice |
| one input and the output | free | kept in the output |
| one input only | error | implicit single-operand sums are not supported |

An index in the output that is in neither input is also an error.

## Output order

The output lists indices in three groups: batch indices, then free indices of the left operand, then free indices of the right operand. Order within a group is free and sets the output axis order. A group that appears out of sequence is an error.

## Examples

Operands used below: `A = [[1,2,3],[4,5,6]]` (2x3), `B = [[1,0],[0,1],[1,1]]` (3x2).

| Spec | Operation | Result |
|---|---|---|
| `ij,jk->ik` | matrix product `A B` | `[[4,5],[10,11]]` |
| `ij , jk -> ik` | same; spaces ignored | `[[4,5],[10,11]]` |
| `ab,bc->ac` | same; letters are labels only | `[[4,5],[10,11]]` |
| `ij,j->i` | `A` times `[1,1,1]` | `[6,15]` |
| `i,j->ij` | outer product of `[1,2]` and `[3,4,5]` | `[[3,4,5],[6,8,10]]` |
| `ij,ij->` | full contraction `A:A` | `91` |
| `ij,ij->ij` | Hadamard product `A*A` | `[[1,4,9],[16,25,36]]` |
| `kij,kij->ij` | dot over `k` per `(i,j)`, `k=2`, shape `(2,1,3)` | `[[17,29,45]]` |
| `ijk,kl->jil` | `i,j` free in operand order `j,i`; shape `(1,2,2)` with `(2,1)` | shape `(2,1,1)` |

### Scale

The scale multiplies the whole result. With scale `-0.5`, `ij,jk->ik` gives `[[-2,-2.5],[-5,-5.5]]`. For a scalar spec, scale `2` gives `182`.

```rust,ignore
let s: f64 = einsum!("ij,ij->", &a, &a, 2.0); // 182.0
let d = einsum_dyn("ij,ij->", a.view(), a.view(), 2.0).unwrap();
assert_eq!(s, d[IxDyn(&[])]);
```

## Errors

Each message is asserted verbatim in `malformed_specs_are_errors` and `operand_mismatches_are_errors_at_run_time`. `einsum!` reports parse errors as compile errors with the same text.

| Spec | Error |
|---|---|
| `ij,jk` | `einsum spec must contain '->'` |
| `ij->ij` | `einsum spec must have exactly two comma-separated inputs` |
| `ij,jk,kl->il` | `einsum is binary: exactly two inputs` |
| `ii,jk->jk` | `index 'i' is repeated in the left operand (traces / diagonals of one operand are not supported)` |
| `ij,jk->kk` | `index 'k' is repeated in the output (traces / diagonals of one operand are not supported)` |
| `ij,jk->k` | `index 'i' appears in an input but is neither contracted nor in the output (implicit single-operand sums are not supported)` |
| `ij,jk->ikz` | `output index 'z' is not a free or batch index of either input` |
| `ij,jk->ki` | `output indices must be listed as (batch indices, then left-operand free indices, then right-operand free indices) to match the runtime axis order` |

`einsum_dyn` adds two checks that are `TensorError::Operand` (Python `ValueError`):

- an operand whose number of axes differs from its term;
- a shared index whose extents differ between the operands.

An operand with a zero-extent axis returns an array of zeros with the output shape.

## Execution

The contraction is one binary product, not a loop over indices.

1. Each operand is permuted so batch, free and contracted axes are contiguous. When the needed order is the identity or the swap of the two axis groups, the operand is passed as a (transposed) view with no copy.
2. Axes are flattened to a matrix and multiplied with `ndarray::linalg::general_mat_mul`. `ferric-tensors` builds ndarray with its `blas` feature through `ndarray-linalg`, so the product is an OpenBLAS `dgemm`.
3. With batch indices, one GEMM runs per batch slice, in a sequential loop.
4. The contracted axis is summed in blocks of `GEMM_K_BLOCK` (128). Large products split output rows over rayon bands; this path is bit-identical to the serial product.
5. The scale is applied (`alpha` of the batched GEMM, or a pass over the result when there is no batch index) and the result is reshaped to the output shape.

An outer product (`i,j->ij`) is a GEMM with inner dimension 1. Contractions run on the CPU unless the crate is built with the `gpu` feature.

Set `OPENBLAS_NUM_THREADS=1` as described in [Installation](../using/installation.md).
