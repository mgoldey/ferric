use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array3, Array4, IxDyn};

/// Compute the Particle-Particle ladder term: L_iajb = sum_P B^P_ab * (sum_cd B^P_cd * t_icjd)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `t2` is stored `(i,c,j,d)`. Intermediate `X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]`,
/// then `R[i,a,j,b] = sum_P B^P_ab X[P,i,j]`. Output layout `(i,a,j,b)`.
pub fn contract_pp_ladder(
    b_ab: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (nocc, nvir, _, _) = t2.dim();

    let b_ab_t = Tensor::new(b_ab.clone().into_dyn(), [Axis::Aux, Axis::V, Axis::V]);
    let t2_t = Tensor::new(t2.clone().into_dyn(), [Axis::O, Axis::V, Axis::O, Axis::V]);

    // X[P,i,j] = sum_cd B^P_cd t2[i,c,j,d]  (contract c,d=V)
    let x: ndarray::ArrayD<f64> = einsum!("Pcd,icjd->Pij", &b_ab_t, &t2_t);
    let x_t = Tensor::new(x, [Axis::Aux, Axis::O, Axis::O]);

    // R[a,b,i,j] = sum_P B^P_ab X[P,i,j]; then permute (a,b,i,j)->(i,a,j,b) = axes [2,0,3,1]
    let r_abij: ndarray::ArrayD<f64> = einsum!("Pab,Pij->abij", &b_ab_t, &x_t);
    let res = r_abij
        .permuted_axes(IxDyn(&[2, 0, 3, 1]))
        .as_standard_layout()
        .into_owned();
    res.into_dimensionality::<ndarray::Ix4>().unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir)).unwrap()
}

/// Compute the Hole-Hole ladder term: H_iajb = sum_P B^P_ij * (sum_kl B^P_kl * t_kalb)
/// RI complexity: O(N^5). All contractions routed through `einsum!` (BLAS3 GEMM).
///
/// `t2` is stored `(k,a,l,b)`. Intermediate `Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]`,
/// then `R[i,a,j,b] = sum_P B^P_ij Y[P,a,b]`. Output layout `(i,a,j,b)`.
pub fn contract_hh_ladder(
    b_ij: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (nocc, nvir, _, _) = t2.dim();

    let b_ij_t = Tensor::new(b_ij.clone().into_dyn(), [Axis::Aux, Axis::O, Axis::O]);
    let t2_t = Tensor::new(t2.clone().into_dyn(), [Axis::O, Axis::V, Axis::O, Axis::V]);

    // Y[P,a,b] = sum_kl B^P_kl t2[k,a,l,b]  (contract k,l=O)
    let y: ndarray::ArrayD<f64> = einsum!("Pkl,kalb->Pab", &b_ij_t, &t2_t);
    let y_t = Tensor::new(y, [Axis::Aux, Axis::V, Axis::V]);

    // R[i,j,a,b] = sum_P B^P_ij Y[P,a,b]; then permute (i,j,a,b)->(i,a,j,b) = axes [0,2,1,3]
    let r_ijab: ndarray::ArrayD<f64> = einsum!("Pij,Pab->ijab", &b_ij_t, &y_t);
    let res = r_ijab
        .permuted_axes(IxDyn(&[0, 2, 1, 3]))
        .as_standard_layout()
        .into_owned();
    res.into_dimensionality::<ndarray::Ix4>().unwrap()
        .into_shape_with_order((nocc, nvir, nocc, nvir)).unwrap()
}
