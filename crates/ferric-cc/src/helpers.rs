use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array4, ArrayD, IxDyn};

// --- Permutation antisymmetrizers on the i,j,a,b axes of X[i,j,a,b] ---
// Shared by the spin-orbital CCD and CCSD residual builders (ccd.rs, ccsd.rs),
// which apply the same P(ij)/P(ab) projectors to their `[O,O,V,V]` tensors.

/// Swap the i,j axes (0,1) of an `[i,j,a,b]` tensor.
fn swap_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.view().permuted_axes(IxDyn(&[1, 0, 2, 3])).as_standard_layout().into_owned()
}
/// Swap the a,b axes (2,3) of an `[i,j,a,b]` tensor.
fn swap_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.view().permuted_axes(IxDyn(&[0, 1, 3, 2])).as_standard_layout().into_owned()
}
/// P(ij) x = x - swap_ij(x)
pub fn p_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ij(x)
}
/// P(ab) x = x - swap_ab(x)
pub fn p_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ab(x)
}
/// P(ij)P(ab) x = x - swap_ij(x) - swap_ab(x) + swap_ab(swap_ij(x))
pub fn p_ij_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    let sij = swap_ij(x);
    let sab = swap_ab(x);
    let sijab = swap_ab(&sij);
    x - &sij - &sab + &sijab
}

/// Compute the Particle-Particle ladder term: L_iajb = sum_P B^P_ab * (sum_cd B^P_cd * t_icjd)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `b_ab_t` is the dressed RI block `[Aux, V, V]` (loop-invariant — wrapped once
/// by the caller and reused across CCD iterations). `t2_t` is the current
/// amplitude tensor stored `(i,c,j,d)` labeled `[O, V, O, V]`. Intermediate
/// `X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]`, then `R[i,a,j,b] = sum_P B^P_ab X[P,i,j]`.
/// Output layout `(i,a,j,b)`.
pub fn contract_pp_ladder(b_ab_t: &Tensor<3>, t2_t: &Tensor<4>) -> Array4<f64> {
    let nocc = t2_t.shape()[0];
    let nvir = t2_t.shape()[1];

    // X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]  (contract c,d=V)
    let x: ndarray::ArrayD<f64> = einsum!("Pcd,icjd->Pij", b_ab_t, t2_t);
    let x_t = Tensor::new(x, [Axis::Aux, Axis::O, Axis::O]);

    // R[a,b,i,j] = sum_P B^P_ab X[P,i,j]; then permute (a,b,i,j)->(i,a,j,b) = axes [2,0,3,1]
    let r_abij: ndarray::ArrayD<f64> = einsum!("Pab,Pij->abij", b_ab_t, &x_t);
    let res = r_abij
        .permuted_axes(IxDyn(&[2, 0, 3, 1]))
        .as_standard_layout()
        .into_owned();
    res.into_dimensionality::<ndarray::Ix4>()
        .unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir))
        .unwrap()
}

/// Compute the Hole-Hole ladder term: H_iajb = sum_P B^P_ij * (sum_kl B^P_kl * t_kalb)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `b_ij_t` is the dressed RI block `[Aux, O, O]` (loop-invariant — wrapped once
/// by the caller and reused across CCD iterations). `t2_t` is the current
/// amplitude tensor stored `(k,a,l,b)` labeled `[O, V, O, V]`. Intermediate
/// `Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]`, then `R[i,a,j,b] = sum_P B^P_ij Y[P,a,b]`.
/// Output layout `(i,a,j,b)`.
pub fn contract_hh_ladder(b_ij_t: &Tensor<3>, t2_t: &Tensor<4>) -> Array4<f64> {
    let nocc = t2_t.shape()[0];
    let nvir = t2_t.shape()[1];

    // Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]  (contract k,l=O)
    let y: ndarray::ArrayD<f64> = einsum!("Pkl,kalb->Pab", b_ij_t, t2_t);
    let y_t = Tensor::new(y, [Axis::Aux, Axis::V, Axis::V]);

    // R[i,j,a,b] = sum_P B^P_ij Y[P,a,b]; then permute (i,j,a,b)->(i,a,j,b) = axes [0,2,1,3]
    let r_ijab: ndarray::ArrayD<f64> = einsum!("Pij,Pab->ijab", b_ij_t, &y_t);
    let res = r_ijab
        .permuted_axes(IxDyn(&[0, 2, 1, 3]))
        .as_standard_layout()
        .into_owned();
    res.into_dimensionality::<ndarray::Ix4>()
        .unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir))
        .unwrap()
}
