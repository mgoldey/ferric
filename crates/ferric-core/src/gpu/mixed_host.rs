//! Host twin of `crate::gpu::mixed::gemm_f32_f64acc`: the SAME algorithm
//! (operands rounded to f32, k-panelled sgemm into an f32 scratch, each panel
//! added into an f64 accumulator) on OpenBLAS. Three uses: (1) CI tests of the
//! error model without a device; (2) the CPU-side counterpart arm of the panel
//! sweep (plan §3.6); (3) a second implementation whose error class the device
//! kernel must match (same bound, different sgemm internals).
//!
//! Error model, one element, compared with the exact product
//! (Higham 2nd ed. §3.1/3.5; standard model fl(x op y) = (x op y)(1+δ), |δ| ≤ u):
//!   rounding a, b to f32: each term a_l b_l perturbed by ≤ 2u32 (first order)
//!   a panel of b terms in f32 (any order inside sgemm): ≤ γ_b(u32)
//!   ⌈k/b⌉ panel sums in f64:                               ≤ γ_{⌈k/b⌉}(u64)
//! so |ĉ − c| ≤ (2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64)) · Σ_l |a_l||b_l|.
//! The k-dependence sits in the u64 term, 2^29 smaller than the f32 terms, so
//! for every reachable k the bound is ≈ (b + 2)·u32 and independent of k. The
//! two-precision optimum b* = sqrt(k·u64/u32) is < 1 for k < 5e8: smaller b is
//! always more accurate, and b is chosen by throughput (see `mixed.rs`).
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Zip};

/// Unit roundoff of binary32 (2^-24). NOT machine epsilon (2^-23).
pub const U32: f64 = 5.960_464_477_539_063e-8;
/// Unit roundoff of binary64 (2^-53).
pub const U64: f64 = 1.110_223_024_625_156_5e-16;

/// γ_n(u) = n u / (1 − n u).
pub fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// The f32-side factor of the bound above: 2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64),
/// with b clamped to [1, k]. Callers comparing against an f64 reference add
/// that reference's own γ_k(u64).
pub fn mixed_error_factor(k: usize, k_panel: usize) -> f64 {
    let k = k.max(1);
    let b = k_panel.clamp(1, k);
    let panels = k.div_ceil(b);
    2.0 * U32 + gamma(b, U32) + gamma(panels, U64)
}

/// `x as f32 as f64` elementwise (IEEE round-to-nearest-even, Rust `as`).
pub fn round_trip_f32(v: &ArrayView2<f64>) -> Array2<f64> {
    v.mapv(|x| x as f32 as f64)
}

/// `out = left · right` by the mixed algorithm on the host. `k_panel >= k`
/// gives one panel, i.e. plain sgemm plus one f64 conversion; `k_panel == 0`
/// is clamped to 1.
pub fn gemm_f32_f64acc_host(
    left: &ArrayView2<f64>,
    right: &ArrayView2<f64>,
    out: &mut ArrayViewMut2<f64>,
    k_panel: usize,
) {
    let (m, k) = (left.nrows(), left.ncols());
    let n = right.ncols();
    assert_eq!(right.nrows(), k, "inner dimensions differ");
    assert_eq!(out.dim(), (m, n), "output shape");
    out.fill(0.0);
    if k == 0 || m == 0 || n == 0 {
        return;
    }
    let a32: Array2<f32> = left.mapv(|x| x as f32);
    let b32: Array2<f32> = right.mapv(|x| x as f32);
    let kb = k_panel.max(1);
    let mut c32 = Array2::<f32>::zeros((m, n));
    let mut k0 = 0usize;
    while k0 < k {
        let k1 = (k0 + kb).min(k);
        general_mat_mul(
            1.0f32,
            &a32.slice(s![.., k0..k1]),
            &b32.slice(s![k0..k1, ..]),
            0.0f32,
            &mut c32,
        );
        Zip::from(out.view_mut())
            .and(&c32)
            .for_each(|o, &c| *o += f64::from(c));
        k0 = k1;
    }
}
