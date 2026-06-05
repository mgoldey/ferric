//! High-performance tensor library for Coupled Cluster methods.
//!
//! Provides labeled-axis tensors and a compile-time [`einsum!`] macro for
//! quantum-chemistry contractions. Each `einsum!` is one binary contraction that
//! lowers to BLAS3 GEMM (permute → reshape → `general_mat_mul` → reshape):
//!
//! ```
//! use ferric_tensors::{einsum, Axis, Tensor};
//! use ndarray::{Array, IxDyn};
//! let a = Tensor::new(
//!     Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap(),
//!     [Axis::O, Axis::Aux]);
//! let b = Tensor::new(
//!     Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap(),
//!     [Axis::Aux, Axis::V]);
//! let c: ndarray::ArrayD<f64> = einsum!("ik,kj->ij", &a, &b);
//! assert_eq!(c.shape(), &[2, 2]);
//! ```
//!
//! Index classification follows numpy-einsum rules: an index in both inputs but
//! not the output is **contracted** (the GEMM axis); an index in both inputs and
//! the output is a **batch/diagonal** axis (iterated element-wise, one GEMM per
//! batch slice — e.g. Hadamard `"ij,ij->ij"` or per-element dot `"kij,kij->ij"`);
//! an index in one input and the output is **free**. An optional trailing scalar
//! scales the whole result: `einsum!("ai,ai->", &mu, &u, -4.0)`. Implicit
//! single-operand sums (an input index that is neither contracted, batched, nor
//! in the output) are rejected at compile time.
//!
//! Also retains the sparse tensor types ([`SparseTensor2`], [`FlatSparse`], …)
//! used by screening / link-K paths.

use ndarray::{Array2, Array3, Array4};
use sprs::{CsMat, TriMat};

pub mod axis;
pub use axis::Axis;
pub mod tensor;
pub use tensor::{MaybeLabeled, Tensor};
pub mod einsum;
pub use einsum::{einsum_binary, einsum_binary_batched, TensorError};
pub use ferric_tensor_macro::einsum;

/// A rank-4 tensor typically used for T2 amplitudes or 2-electron integrals.
///
/// Storage layout is [dim1, dim2, dim3, dim4].
pub struct Tensor4 {
    pub data: Array4<f64>,
}

impl Tensor4 {
    pub fn new(d1: usize, d2: usize, d3: usize, d4: usize) -> Self {
        Self {
            data: Array4::zeros((d1, d2, d3, d4)),
        }
    }

    /// Contract (ia, kc) * (kc, jb) -> (ia, jb)
    /// This is a common pattern in CCSD where we contract amplitudes with integrals.
    pub fn contract_ia_kc_jb(&self, other: &Tensor4) -> Tensor4 {
        let (ni, na, nk, nc) = self.data.dim();
        let (nk2, nc2, nj, nb) = other.data.dim();
        assert_eq!(nk, nk2);
        assert_eq!(nc, nc2);

        let a_flat = self.data.view().into_shape_with_order((ni * na, nk * nc)).unwrap();
        let b_flat = other.data.view().into_shape_with_order((nk * nc, nj * nb)).unwrap();

        let c_flat = a_flat.dot(&b_flat);
        let c_data = c_flat.into_shape_with_order((ni, na, nj, nb)).unwrap();

        Tensor4 { data: c_data }
    }
}

/// A rank-2 sparse tensor (matrix) in CSR format.
/// Used by CC routines that need sprs interop.
pub struct SparseTensor2 {
    pub data: CsMat<f64>,
}

impl SparseTensor2 {
    /// Create a sparse tensor from a dense matrix using a threshold.
    pub fn from_dense(dense: &Array2<f64>, threshold: f64) -> Self {
        let (rows, cols) = dense.dim();
        let mut tri = TriMat::new((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let val = dense[(i, j)];
                if val.abs() > threshold {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        Self { data: tri.to_csr() }
    }

    /// Compute the trace of the product of two sparse matrices: Tr(A * B).
    /// This is equivalent to sum(A_ij * B_ji).
    pub fn trace_product(&self, other: &SparseTensor2) -> f64 {
        let mut tr = 0.0;
        for (val, (i, j)) in self.data.iter() {
            if let Some(&val_b) = other.data.get(j, i) {
                tr += val * val_b;
            }
        }
        tr
    }
}

/// A rank-3 tensor stored as a collection of sparse slices.
/// Typically used for B^P_mu_nu 3-center integrals.
pub struct SparseTensor3 {
    pub slices: Vec<SparseTensor2>,
    pub shape: (usize, usize, usize),
}

impl SparseTensor3 {
    /// Create a sparse rank-3 tensor from a dense tensor.
    pub fn from_dense(dense: &Array3<f64>, threshold: f64) -> Self {
        let (n1, n2, n3) = dense.dim();
        let mut slices = Vec::with_capacity(n1);
        for i in 0..n1 {
            let slice = dense.slice(ndarray::s![i, .., ..]).to_owned();
            slices.push(SparseTensor2::from_dense(&slice, threshold));
        }
        Self { slices, shape: (n1, n2, n3) }
    }

    pub fn contract_pqp(&self, p_idx: usize, p_mat: &SparseTensor2, q_mat: &SparseTensor2) -> SparseTensor2 {
        let bp = &self.slices[p_idx];
        let tmp = &p_mat.data * &bp.data;
        let l = &tmp * &q_mat.data;
        SparseTensor2 { data: l }
    }
}

/// Flat sorted COO sparse vector over a (μ,ν) index space.
///
/// Entries are stored as `(flat_index: u32, value: f64)` sorted by flat_index.
/// Flat index = μ * ncols + ν.  All operations (hadamard, trace_dot) use
/// two-pointer merge: O(nnz_A + nnz_B) with no allocations beyond the output.
#[derive(Clone)]
pub struct FlatSparse {
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl FlatSparse {
    pub fn from_dense_flat(data: &[f64], threshold: f64) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in data.iter().enumerate() {
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }
        Self { indices, values }
    }

    pub fn nnz(&self) -> usize { self.indices.len() }

    /// Element-wise product of two flat-sparse vectors; output sorted by index.
    pub fn hadamard(&self, other: &FlatSparse) -> FlatSparse {
        let mut out_idx = Vec::new();
        let mut out_val = Vec::new();
        let (mut ai, mut bi) = (0, 0);
        while ai < self.nnz() && bi < other.nnz() {
            match self.indices[ai].cmp(&other.indices[bi]) {
                std::cmp::Ordering::Equal => {
                    let v = self.values[ai] * other.values[bi];
                    if v != 0.0 {
                        out_idx.push(self.indices[ai]);
                        out_val.push(v);
                    }
                    ai += 1; bi += 1;
                }
                std::cmp::Ordering::Less => { ai += 1; }
                std::cmp::Ordering::Greater => { bi += 1; }
            }
        }
        FlatSparse { indices: out_idx, values: out_val }
    }

    /// Dot product (sum of element-wise products) via two-pointer merge.
    pub fn dot(&self, other: &FlatSparse) -> f64 {
        let mut acc = 0.0;
        let (mut ai, mut bi) = (0, 0);
        while ai < self.nnz() && bi < other.nnz() {
            match self.indices[ai].cmp(&other.indices[bi]) {
                std::cmp::Ordering::Equal => {
                    acc += self.values[ai] * other.values[bi];
                    ai += 1; bi += 1;
                }
                std::cmp::Ordering::Less => { ai += 1; }
                std::cmp::Ordering::Greater => { bi += 1; }
            }
        }
        acc
    }

    /// Squared Frobenius norm: sum of squares of values.
    pub fn norm_sq(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum()
    }
}

/// Collection of FlatSparse slices for the B^P_μν 3-center integrals.
pub struct FlatSparse3 {
    pub slices: Vec<FlatSparse>,
}

impl FlatSparse3 {
    pub fn from_dense(dense: &Array3<f64>, threshold: f64) -> Self {
        let (naux, _, _) = dense.dim();
        let slices = (0..naux).map(|p| {
            let slice = dense.slice(ndarray::s![p, .., ..]).to_owned();
            FlatSparse::from_dense_flat(slice.as_slice().unwrap(), threshold)
        }).collect();
        Self { slices }
    }
}

/// Brainstorming: Perturbation Theory with Modified Coulomb Operators in Coupled Cluster
///
/// 1. Attenuated Coupled Cluster:
///    Replace the 1/r operator in the CC equations with an attenuated version
///    v_att(r) = erfc(omega*r)/r. This can be used to implement Local CC (LCC)
///    where long-range interactions are treated perturbatively (e.g., via MP2)
///    while short-range correlation is handled by CC.
///
/// 2. Range-Separated Coupled Cluster (RS-CC):
///    Split the Hamiltonian: H = H_sr + H_lr.
///    Solve CC for the short-range part and use PT for the long-range part.
///    This is particularly useful for avoiding the "intruder state" problem
///    in multireference cases or for combining CC with DFT.
///
/// 3. Perturbative (T) with Modified Operators:
///    The (T) correction in CCSD(T) is O(N^7). Using an attenuated operator
///    for the triples correction could allow for a "screened (T)" that captures
///    the most important local triples contributions at a lower cost.
pub mod brainstorming {
    // This module exists for design notes and future implementation paths.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor4_contraction() {
        let mut a = Tensor4::new(2, 2, 2, 2);
        let mut b = Tensor4::new(2, 2, 2, 2);
        
        // Simple data
        a.data.fill(1.0);
        b.data.fill(2.0);
        
        let c = a.contract_ia_kc_jb(&b);
        
        // (2*2) contraction: 1.0 * 2.0 * (nk*nc) = 1.0 * 2.0 * 4 = 8.0
        assert_eq!(c.data[(0, 0, 0, 0)], 8.0);
        assert_eq!(c.data.dim(), (2, 2, 2, 2));
    }

    #[test]
    fn test_sparse_tensor2_trace() {
        let mut a_dense = Array2::zeros((2, 2));
        a_dense[(0, 0)] = 1.0;
        a_dense[(1, 1)] = 2.0;

        let mut b_dense = Array2::zeros((2, 2));
        b_dense[(0, 0)] = 3.0;
        b_dense[(1, 1)] = 4.0;

        let a = SparseTensor2::from_dense(&a_dense, 1e-10);
        let b = SparseTensor2::from_dense(&b_dense, 1e-10);

        // Tr(A*B) = 1*3 + 2*4 = 11
        assert_eq!(a.trace_product(&b), 11.0);
    }

    #[test]
    fn test_flat_sparse_hadamard_dot() {
        // 2x2 matrices flattened: [a,b,c,d] = [M00,M01,M10,M11]
        let m = vec![1.0, 2.0, 3.0, 4.0];
        let n = vec![5.0, 6.0, 7.0, 8.0];
        let p = vec![1.0, 0.0, 0.0, 2.0]; // diagonal
        let q = vec![3.0, 0.0, 0.0, 4.0];

        let bm = FlatSparse::from_dense_flat(&m, 1e-10);
        let bn = FlatSparse::from_dense_flat(&n, 1e-10);
        let bp = FlatSparse::from_dense_flat(&p, 1e-10);
        let bq = FlatSparse::from_dense_flat(&q, 1e-10);

        // M[p] = bm ⊙ bp = [1,0,0,8], N[p] = bn ⊙ bp = [5,0,0,16]
        let mvec = bm.hadamard(&bp);
        let nvec = bn.hadamard(&bq);

        // J = sum M[p]*N[q] = 1*3*5 + 4*4*8 = 15 + 128 = 143... let me just check dot
        let dot_mn = mvec.dot(&nvec);
        // mvec = [1*1,0,0,4*2]=[1,0,0,8], nvec=[5*3,0,0,8*4]=[15,0,0,32]
        // dot = 1*15 + 8*32 = 15 + 256 = 271
        assert_eq!(dot_mn, 271.0);

        // hadamard of mvec and nvec
        let l = mvec.hadamard(&nvec);
        // l = [1*15, 0, 0, 8*32] = [15, 0, 0, 256]
        let norm_sq = l.norm_sq();
        // 15² + 256² = 225 + 65536 = 65761
        assert_eq!(norm_sq, 65761.0);
    }
}
