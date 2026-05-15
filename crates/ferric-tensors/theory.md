# Tensor Operations and Sparsity

This library provides the tensor abstractions used throughout `ferric` to enable high-performance correlation methods and linear scaling.

## Sparse Matrix Formats
We use Compressed Sparse Row (CSR) storage for AO-basis matrices:
$$A_{ij} \neq 0 \implies \text{store } (i, j, A_{ij})$$
This is critical for $O(N)$ AO-Laplace MP2, where the pseudo-densities $P(t)$ and $Q(t)$ become sparse for large systems.

## Trace Contractions
Many energy expressions can be written as traces of matrix products:
$$\text{Tr}(A B C \dots) = \sum_{ijk\dots} A_{ij} B_{jk} C_{ki} \dots$$
In the AO-Laplace method, we compute:
$$J_{PQ} = \text{Tr}(B^P P B^Q Q)$$
Using sparse SpGEMM (Sparse General Matrix Multiplication), this contraction scales linearly with system size.

## Rank-4 Tensor Contractions
For Coupled Cluster methods, we contract rank-4 tensors:
$$C_{ij}^{ab} = \sum_{kc} A_{ik}^{ac} B_{kj}^{cb}$$
These are implemented using optimized BLAS/MKL kernels by flattening the tensors into matrices:
$$C_{(ia),(jb)} = \sum_{(kc)} A_{(ia),(kc)} B_{(kc),(jb)}$$
This allows us to leverage high-performance `gemm` routines.
