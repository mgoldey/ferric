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

pub mod axis;
pub use axis::Axis;
pub mod tensor;
pub use tensor::{MaybeLabeled, Tensor};
pub mod einsum;
pub use einsum::{einsum_binary, einsum_binary_batched, permute_to_owned, TensorError};
pub use ferric_tensor_macro::einsum;
