//! AO-to-MO integral transformation for 3-center integrals.

use ndarray::{Array2, Array3};

/// Transform (P|mu nu) -> (P|ia) where i=occupied, a=virtual.
pub fn transform_3center_ov(
    eri3_ao: &Array3<f64>,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
) -> Array3<f64> {
    let naux = eri3_ao.shape()[0];
    let nocc = c_occ.ncols();
    let nvir = c_vir.ncols();

    let mut mo = Array3::zeros((naux, nocc, nvir));
    for p in 0..naux {
        let bp_ao = eri3_ao.slice(ndarray::s![p, .., ..]);
        // B^P_ia = C_occ^T * B^P_AO * C_vir
        let tmp = bp_ao.dot(c_vir);
        let bp_mo = c_occ.t().dot(&tmp);
        mo.slice_mut(ndarray::s![p, .., ..]).assign(&bp_mo);
    }
    mo
}

/// Transform (P|mu nu) -> (P|ij) where i,j=occupied.
pub fn transform_3center_oo(
    eri3_ao: &Array3<f64>,
    c_occ: &Array2<f64>,
) -> Array3<f64> {
    let naux = eri3_ao.shape()[0];
    let _nbas = eri3_ao.shape()[1];
    let nocc = c_occ.ncols();

    let mut mo = Array3::zeros((naux, nocc, nocc));
    for p in 0..naux {
        let bp_ao = eri3_ao.slice(ndarray::s![p, .., ..]);
        // B^P_ij = C_occ^T * B^P_AO * C_occ
        let tmp = bp_ao.dot(c_occ);
        let bp_mo = c_occ.t().dot(&tmp);
        mo.slice_mut(ndarray::s![p, .., ..]).assign(&bp_mo);
    }
    mo
}

/// Transform (P|mu nu) -> (P|ab) where a,b=virtual.
pub fn transform_3center_vv(
    eri3_ao: &Array3<f64>,
    c_vir: &Array2<f64>,
) -> Array3<f64> {
    let naux = eri3_ao.shape()[0];
    let _nbas = eri3_ao.shape()[1];
    let nvir = c_vir.ncols();

    let mut mo = Array3::zeros((naux, nvir, nvir));
    for p in 0..naux {
        let bp_ao = eri3_ao.slice(ndarray::s![p, .., ..]);
        // B^P_ab = C_vir^T * B^P_AO * C_vir
        let tmp = bp_ao.dot(c_vir);
        let bp_mo = c_vir.t().dot(&tmp);
        mo.slice_mut(ndarray::s![p, .., ..]).assign(&bp_mo);
    }
    mo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_identity() {
        // With identity MO coefficients, (P|ia) should equal (P|i,a)
        let naux = 2;
        let nbas = 3;
        let nocc = 1;
        let nvir = 2;

        let mut eri3 = Array3::zeros((naux, nbas, nbas));
        eri3[(0, 0, 1)] = 1.0;
        eri3[(0, 1, 0)] = 1.0;
        eri3[(1, 0, 2)] = 2.0;
        eri3[(1, 2, 0)] = 2.0;

        // c_occ picks out column 0
        let mut c_occ = Array2::zeros((nbas, nocc));
        c_occ[(0, 0)] = 1.0;

        // c_vir picks out columns 1,2
        let mut c_vir = Array2::zeros((nbas, nvir));
        c_vir[(1, 0)] = 1.0;
        c_vir[(2, 1)] = 1.0;

        let mo = transform_3center_ov(&eri3, &c_occ, &c_vir);
        assert_eq!(mo.shape(), &[naux, nocc, nvir]);
        // (0|0,1) via identity: eri3[(0,0,1)] = 1.0 -> mo[(0,0,0)] = 1.0
        assert!((mo[(0, 0, 0)] - 1.0).abs() < 1e-12);
        // (1|0,2) via identity: eri3[(1,0,2)] = 2.0 -> mo[(1,0,1)] = 2.0
        assert!((mo[(1, 0, 1)] - 2.0).abs() < 1e-12);
    }
}
