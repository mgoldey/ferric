# Tensor Operations and Sparsity

This library provides the tensor abstractions used throughout `ferric` by the correlation methods.

## Storage
`Tensor<N>` stores dense `f64` data with labelled axes (`crates/ferric-tensors/src/tensor.rs:11`). The crate has no sparse format.
The AO-Laplace MP2 code in `ferric-mp2` (`crates/ferric-mp2/src/laplace.rs`) is dense: no $O(N)$ scaling is implemented or measured.

## Trace Contractions
Many energy expressions can be written as traces of matrix products:
$$\text{Tr}(A B C \dots) = \sum_{ijk\dots} A_{ij} B_{jk} C_{ki} \dots$$
The AO-Laplace method computes:
$$J_{PQ} = \text{Tr}(B^P P B^Q Q)$$
The implementation evaluates this contraction with dense matrix products.

## Rank-4 Tensor Contractions
Coupled Cluster methods contract rank-4 tensors:
$$C_{ij}^{ab} = \sum_{kc} A_{ik}^{ac} B_{kj}^{cb}$$
These run on BLAS/MKL kernels by flattening the tensors into matrices:
$$C_{(ia),(jb)} = \sum_{(kc)} A_{(ia),(kc)} B_{(kc),(jb)}$$
Each contraction is then one `gemm` call.
