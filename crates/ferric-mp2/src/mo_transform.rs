//! AO-to-MO integral transformation for 3-center integrals.

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

/// Transform a raw AO 3-center tensor `(P|μν)` to an MO block `(P|pq)` by
/// contracting each aux row P with `c_left^T (P|μν) c_right`.
///
/// The aux index P is embarrassingly parallel: each P owns a disjoint output
/// slab `mo[P,:,:]` and reads only its own `(P|μν)` AO slab, so we fan the
/// per-P GEMM pair across rayon (the old T4). BLAS stays serial inside each
/// closure (OPENBLAS_NUM_THREADS=1) — nested BLAS threads under rayon is the
/// documented dgetrf-crash footgun. The result is bit-identical to a serial
/// loop: each P's two GEMMs are computed independently and written to disjoint
/// rows, so there is no cross-P accumulation whose order could shift.
fn transform_3center(
    eri3_ao: &Array3<f64>,
    c_left: &Array2<f64>,
    c_right: &Array2<f64>,
) -> Array3<f64> {
    let naux = eri3_ao.shape()[0];
    let nleft = c_left.ncols();
    let nright = c_right.ncols();

    let mut mo = Array3::zeros((naux, nleft, nright));
    // par over aux P: disjoint output bands (one lane per aux row), independent
    // read of the matching AO slab. Zip the outer axes so each task gets its
    // own (P|μν) view and (P|pq) output view.
    ndarray::Zip::from(mo.axis_iter_mut(Axis(0)))
        .and(eri3_ao.axis_iter(Axis(0)))
        .into_par_iter()
        .for_each(|(mut mo_p, bp_ao)| {
            // B^P_pq = c_left^T * B^P_AO * c_right
            let tmp = bp_ao.dot(c_right);
            let bp_mo = c_left.t().dot(&tmp);
            mo_p.assign(&bp_mo);
        });
    mo
}

/// Transform (P|mu nu) -> (P|ia) where i=occupied, a=virtual.
pub fn transform_3center_ov(
    eri3_ao: &Array3<f64>,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
) -> Array3<f64> {
    transform_3center(eri3_ao, c_occ, c_vir)
}

/// Transform (P|mu nu) -> (P|ij) where i,j=occupied.
pub fn transform_3center_oo(
    eri3_ao: &Array3<f64>,
    c_occ: &Array2<f64>,
) -> Array3<f64> {
    transform_3center(eri3_ao, c_occ, c_occ)
}

/// Transform (P|mu nu) -> (P|ab) where a,b=virtual.
pub fn transform_3center_vv(
    eri3_ao: &Array3<f64>,
    c_vir: &Array2<f64>,
) -> Array3<f64> {
    transform_3center(eri3_ao, c_vir, c_vir)
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

    /// Serial reference for the par-over-aux-P transform: exactly the old
    /// per-P loop, kept here so the test proves the rayon version is bit-for-bit
    /// identical (f64::to_bits), not merely close.
    fn transform_serial(
        eri3_ao: &Array3<f64>,
        c_left: &Array2<f64>,
        c_right: &Array2<f64>,
    ) -> Array3<f64> {
        let naux = eri3_ao.shape()[0];
        let mut mo = Array3::zeros((naux, c_left.ncols(), c_right.ncols()));
        for p in 0..naux {
            let bp_ao = eri3_ao.slice(ndarray::s![p, .., ..]);
            let tmp = bp_ao.dot(c_right);
            let bp_mo = c_left.t().dot(&tmp);
            mo.slice_mut(ndarray::s![p, .., ..]).assign(&bp_mo);
        }
        mo
    }

    #[test]
    fn par_transform_bit_identical_to_serial() {
        // Deterministic pseudo-random tensors (LCG) so the test is reproducible
        // and every element is nonzero — the par/serial paths must agree to the
        // last bit for ov, oo, and vv transforms.
        let naux = 37;
        let nao = 11;
        let nleft = 4;
        let nright = 6;

        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };

        let mut eri3 = Array3::<f64>::zeros((naux, nao, nao));
        for v in eri3.iter_mut() {
            *v = next();
        }
        let mut c_left = Array2::<f64>::zeros((nao, nleft));
        for v in c_left.iter_mut() {
            *v = next();
        }
        let mut c_right = Array2::<f64>::zeros((nao, nright));
        for v in c_right.iter_mut() {
            *v = next();
        }

        let par = transform_3center(&eri3, &c_left, &c_right);
        let serial = transform_serial(&eri3, &c_left, &c_right);
        assert_eq!(par.shape(), serial.shape());
        for (a, b) in par.iter().zip(serial.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "par transform not bit-identical to serial: {a} vs {b}"
            );
        }

        // Same guarantee for the square (oo / vv) transform.
        let par_sq = transform_3center(&eri3, &c_left, &c_left);
        let serial_sq = transform_serial(&eri3, &c_left, &c_left);
        for (a, b) in par_sq.iter().zip(serial_sq.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "square par transform not bit-identical");
        }
    }
}
