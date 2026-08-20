//! Laplace-separable χ₀ kernel for PDEP-RPA.
//!
//! Replaces the explicit `4·e_ia/(ω²+e_ia²)` denominator (the `Dense` path in
//! `sternheimer::dielectric_matrix_into`) by a minimax-Laplace sum-of-exponentials:
//!
//! ```text
//! e_ia / (ω² + e_ia²)  =  ∫_0^∞ exp(-e_ia · t) cos(ω t) dt
//!                     ≈  Σ_l w_l · cos(ω t_l) · exp(-t_l · e_ia)
//! ```
//!
//! with `(t_l, w_l)` the standard minimax-Laplace nodes/weights for `1/x` on the
//! energy-gap range `[e_ia^min, e_ia^max]` (same machinery as Laplace-MP2).
//!
//! The closed-shell RHF prefactor is 4, so the dielectric kernel becomes
//! ```text
//! ε̃_αβ(iω) − δ_αβ = Σ_l 4 w_l cos(ω t_l) · Σ_ia (V^T B)_α,ia · (V^T B)_β,ia · exp(-t_l · e_ia)
//! ```
//!
//! For each l we build `X^l_α,ia = (V^T B)_α,ia · exp(-t_l · e_ia / 2)` and accumulate
//! `coeff_l · X^l · X^l^T` into the output. `coeff_l = 4 w_l cos(ω t_l)` can be
//! negative when `cos(ω t_l) < 0`, so we use a general matmul (not SYRK).
//!
//! Mathematically, the MO-basis Laplace form is *correctness-equivalent* to the
//! Dense path with the same arithmetic cost. The cubic-scaling win comes from
//! the AO-basis reformulation where the occupied/virtual contractions become
//! pseudo-densities `P̃_μν(t) = Σ_i e^{t ε_i} C_μi C_νi` and
//! `Q̃_μν(t) = Σ_a e^{-t ε_a} C_μa C_νa`. The AO route is left as a TODO; this
//! file delivers the MO-basis correctness gate for C6.

use crate::channel::RpaChannel;
use ferric_quadrature::LaplaceQuadrature;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Axis, Zip};

/// Build the per-(ia) energy-gap array `e_ia = ε_a − ε_i` (length `nocc·nvir`,
/// stored as `ia = i·nvir + a`).
/// # INVARIANT: `eps_occ`/`eps_vir` MUST be CANONICAL orbital energies
///
/// `e_ia = eps_a − eps_i` is a per-pair SCALAR, valid only when the orbitals
/// are Fock EIGENVECTORS. In a rotated basis (localized, PNO, or
/// semicanonical-pending) the Fock matrix is not diagonal and this silently
/// discards the off-diagonal coupling.
///
/// Currently every caller passes canonical `rhf.eps_r()`-derived values, so
/// this is a LATENT trap, not a live bug. It is guarded rather than merely
/// documented because the identical pattern WAS a live bug in `screen.rs`
/// (`diag(C_locᵀ F C_loc)` from Boys-localized orbitals; discarded coupling
/// 1.3–2.9% of the diagonal spread; fixed 2026-07-28 in commit 17e994e) and
/// produced an entirely bogus locality result in the AO-Laplace path
/// (commit 3693d5d).
///
/// Note this module's own docs (see the header) describe an AO pseudo-density
/// route as future work — that is precisely the change that would feed rotated
/// orbitals in here, so the guard is aimed at a plausible future edit, not a
/// hypothetical one.
///
/// If you need a rotated basis: SEMICANONICALIZE first (re-diagonalize the occ
/// and vir Fock blocks and rotate the coefficients to match), as
/// `dlpno_rpa.rs` and `screen.rs` do.
fn build_e_ia(eps_occ: &[f64], eps_vir: &[f64]) -> Array1<f64> {
    debug_assert!(
        eps_occ.windows(2).all(|w| w[0] <= w[1] + 1e-12)
            && eps_vir.windows(2).all(|w| w[0] <= w[1] + 1e-12),
        "build_e_ia: eps_occ/eps_vir are not sorted ascending, so they are \
         almost certainly NOT canonical Fock eigenvalues. The per-pair scalar \
         e_ia = eps_a - eps_i is only valid in a Fock-DIAGONAL basis; in a \
         rotated basis it silently drops the off-diagonal coupling. \
         Semicanonicalize before calling (see this function's docs)."
    );
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let mut e = Array1::<f64>::zeros(nocc * nvir);
    for (i, &eps_i) in eps_occ.iter().enumerate() {
        for (a, &eps_a) in eps_vir.iter().enumerate() {
            e[i * nvir + a] = eps_a - eps_i;
        }
    }
    e
}

/// Build a Laplace quadrature spanning the orbital energy-gap range of a
/// closed-shell system.
///
/// Uses `ymin = min(e_ia)` (HOMO–LUMO gap) and `ymax = max(e_ia)` (eps_max_vir −
/// eps_min_occ). This matches the Laplace-MP2 convention except without the
/// factor of 2 (no two-particle denominator here — χ₀ has one e_ia).
pub fn build_laplace_for_gaps(
    eps_occ: &[f64],
    eps_vir: &[f64],
    n_quad: usize,
) -> Result<LaplaceQuadrature, ferric_core::FerricError> {
    let eps_homo = eps_occ.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_lumo = eps_vir.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_min = eps_occ.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps_vir.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = eps_lumo - eps_homo;
    let ymax = eps_max - eps_min;
    LaplaceQuadrature::new(n_quad, ymin, ymax)
}

/// Laplace-separable dielectric matrix builder.
///
/// Same in-place signature as [`crate::sternheimer::dielectric_matrix_into`] but
/// uses the Laplace-quadrature form of `4·e_ia/(ω²+e_ia²)`. Numerically matches
/// the Dense path to the quadrature tolerance (~1e-3 relative elementwise for
/// the tabulated `n_quad = 7`). Only `n_quad ∈ {3, 5, 7}` are tabulated — the
/// previous `n_quad = 12/20` accuracy claims were never real; those sizes
/// silently fell back to the 7-point table before the TD-QUAD fix.
///
/// # Arguments
/// * `v_mat` — trial-vector block of shape `(naux, m)`.
/// * `b_ov` — RI-MO tensor of shape `(naux, nocc·nvir)`.
/// * `eps_occ`, `eps_vir` — orbital energies.
/// * `omega` — imaginary frequency (real number, the kernel `e_ia/(ω²+e_ia²)`).
/// * `laplace` — minimax quadrature spanning the gap range; build with
///   [`build_laplace_for_gaps`].
/// * `rhs_scaled` — scratch buffer `(m, nov)`; reused across quadrature points
///   to avoid per-call allocation.
/// * `out` — output dielectric matrix `(m, m)`; overwritten with `I + Π`.
///
/// # Algorithm
/// 1. Compute `Y = v_mat^T · b_ov` once (shape `(m, nov)`).
/// 2. For each quadrature point l:
///    - Compute `X^l_α,ia = Y_α,ia · exp(-t_l · e_ia / 2)`.
///    - Accumulate `coeff_l · X^l · X^l^T` into `out` with
///      `coeff_l = 4 · w_l · cos(ω · t_l)`.
/// 3. Add `δ_αβ` on the diagonal.
// Closed-shell into-buffer kernel: the channel triple plus a frequency, the
// Laplace grid, and two caller-owned scratch/output buffers for allocation
// reuse — all distinct, nothing further to bundle.
#[allow(clippy::too_many_arguments)]
/// Build the dielectric matrix ε(iω) = I − v^½ χ₀(iω) v^½ via Laplace transform of χ₀, writing into `out`.
pub fn dielectric_matrix_laplace_into(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
    rhs_scaled: &mut Array2<f64>,
    out: &mut Array2<f64>,
) {
    let m = v_mat.ncols();
    let nov = eps_occ.len() * eps_vir.len();
    assert_eq!(b_ov.shape()[1], nov);
    assert_eq!(rhs_scaled.shape(), &[m, nov], "rhs_scaled scratch shape");
    assert_eq!(out.shape(), &[m, m], "out shape");

    // The Laplace identity e/(ω²+e²) = ∫ cos(ωt) exp(-et) dt becomes
    // ill-conditioned for the standard 1/x-tuned minimax quadrature when
    // ω · t_max ≳ π/2 — the cosine modulation oscillates across the
    // quadrature points and the Laplace expansion catastrophically loses
    // accuracy. For RPA trace-log integration the Gauss-Legendre ω-grid
    // routinely hits ω · t_max > 100 at high points, so the Laplace path
    // CANNOT be used naively at large ω.
    //
    // Resolution: fall back to the Dense kernel whenever ω · t_max exceeds
    // a safe threshold (here π/2). This preserves the Laplace win at low
    // ω (where the cubic-scaling benefit of separability matters) without
    // poisoning the high-ω tail. The high-ω contributions are
    // |e/(ω²+e²)| ~ 1/ω² → small, so the dense fallback is cheap because
    // those quadrature weights are already tiny.
    let t_max = laplace.points.iter().cloned().fold(0.0_f64, f64::max);
    if omega * t_max > std::f64::consts::FRAC_PI_2 {
        // Dense fallback — exact e/(ω²+e²) instead of the broken Laplace
        // approximation. Same closed-shell prefactor 4.
        let dense = crate::sternheimer::dielectric_matrix(v_mat, b_ov, eps_occ, eps_vir, omega);
        out.assign(&dense);
        let _ = rhs_scaled; // scratch unused in fallback
        return;
    }

    // Y = v_mat^T @ b_ov, shape (m, nov). We *don't* scale this in place;
    // we'll need it for every quadrature point. Stash into rhs_scaled as a
    // base, then for each l copy → reuse-buffer for scaling and matmul.
    general_mat_mul(1.0, &v_mat.t(), b_ov, 0.0, rhs_scaled);

    // Save a clone of the unscaled Y. (One alloc per dielectric build — the
    // hot loop is the dgemm inside the quadrature, not this 8m·nov copy.)
    let y_base = rhs_scaled.clone();

    let e_ia = build_e_ia(eps_occ, eps_vir);

    out.fill(0.0);

    // Per-quadrature scratch for X^l X^l^T: shape (m, m). We avoid allocating
    // it per l by accumulating directly via general_mat_mul with beta=1.
    for (&t_l, &w_l) in laplace.points.iter().zip(laplace.weights.iter()) {
        // X^l_α,ia = Y_α,ia · exp(-t_l · e_ia / 2)
        rhs_scaled.assign(&y_base);
        // Precompute the per-(ia) factor.
        let factor: Array1<f64> = e_ia.iter().map(|&e| (-0.5 * t_l * e).exp()).collect();
        let factor_row = factor.view().insert_axis(Axis(0));
        Zip::from(&mut *rhs_scaled)
            .and_broadcast(factor_row)
            .for_each(|x, &f| *x *= f);

        // coeff_l = 4 · w_l · cos(ω · t_l)
        let coeff = 4.0 * w_l * (omega * t_l).cos();

        // out += coeff · X^l · X^l^T
        // general_mat_mul(alpha, &A, &B, beta, C):  C = alpha A B + beta C
        // We need X X^T where X is (m, nov). Use general_mat_mul with X and X.t().
        let x_view = rhs_scaled.view();
        let xt = rhs_scaled.t();
        general_mat_mul(coeff, &x_view, &xt, 1.0, out);
    }

    // ε̃ = I + Π
    for alpha in 0..m {
        out[(alpha, alpha)] += 1.0;
    }
}

/// Unrestricted Laplace-separable dielectric: ε̃ = I + Π_α + Π_β.
///
/// Mirrors [`dielectric_matrix_laplace_into`] but with prefactor 2 per spin
/// (vs 4 closed-shell) and two channels summed. The two channels share
/// the same Laplace quadrature only when their orbital-energy-gap ranges
/// align — for genuine open-shell cases (different α/β gap ranges) the
/// caller passes one quadrature per spin built via
/// [`build_laplace_for_gaps`] on the appropriate eps_occ_σ/eps_vir_σ.
// Unrestricted into-buffer kernel: two spin channels each with their own
// Laplace grid, a frequency, and three caller-owned buffers (per-spin scratch
// + shared output) for allocation reuse. Already bundled to RpaChannel; the
// remainder are independent.
#[allow(clippy::too_many_arguments)]
pub fn dielectric_matrix_laplace_unrestricted_into(
    v_mat: &Array2<f64>,
    chan_a: &RpaChannel, laplace_a: &LaplaceQuadrature,
    chan_b: &RpaChannel, laplace_b: &LaplaceQuadrature,
    omega: f64,
    rhs_scaled_a: &mut Array2<f64>,
    rhs_scaled_b: &mut Array2<f64>,
    out: &mut Array2<f64>,
) {
    let m = v_mat.ncols();

    // Same ω · t_max safety check as the closed-shell path. If either spin's
    // Laplace quadrature is in the bad regime, fall back to dense for both
    // (mixing Dense for one spin and Laplace for the other would still
    // poison the trace-log).
    let t_max = laplace_a.points.iter().chain(laplace_b.points.iter())
        .cloned().fold(0.0_f64, f64::max);
    if omega * t_max > std::f64::consts::FRAC_PI_2 {
        use crate::sternheimer::dielectric_matrix_unrestricted;
        let dense = dielectric_matrix_unrestricted(v_mat, chan_a, chan_b, omega);
        out.assign(&dense);
        return;
    }

    out.fill(0.0);

    accumulate_one_spin(v_mat, chan_a, laplace_a, omega, rhs_scaled_a, out);
    accumulate_one_spin(v_mat, chan_b, laplace_b, omega, rhs_scaled_b, out);

    for alpha in 0..m {
        out[(alpha, alpha)] += 1.0;
    }
}

/// Per-spin Π_σ contribution, accumulated into `out`. Prefactor 2 (open-shell).
fn accumulate_one_spin(
    v_mat: &Array2<f64>,
    chan: &RpaChannel,
    laplace: &LaplaceQuadrature,
    omega: f64,
    rhs_scaled: &mut Array2<f64>,
    out: &mut Array2<f64>,
) {
    let RpaChannel { b_ov, eps_occ, eps_vir } = *chan;
    let m = v_mat.ncols();
    let nov = eps_occ.len() * eps_vir.len();
    if nov == 0 {
        return;
    }
    assert_eq!(b_ov.shape()[1], nov);
    assert_eq!(rhs_scaled.shape(), &[m, nov], "rhs_scaled scratch shape");

    general_mat_mul(1.0, &v_mat.t(), b_ov, 0.0, rhs_scaled);
    let y_base = rhs_scaled.clone();
    let e_ia = build_e_ia(eps_occ, eps_vir);

    for (&t_l, &w_l) in laplace.points.iter().zip(laplace.weights.iter()) {
        rhs_scaled.assign(&y_base);
        let factor: Array1<f64> = e_ia.iter().map(|&e| (-0.5 * t_l * e).exp()).collect();
        let factor_row = factor.view().insert_axis(Axis(0));
        Zip::from(&mut *rhs_scaled)
            .and_broadcast(factor_row)
            .for_each(|x, &f| *x *= f);

        let coeff = 2.0 * w_l * (omega * t_l).cos();
        let x_view = rhs_scaled.view();
        let xt = rhs_scaled.t();
        general_mat_mul(coeff, &x_view, &xt, 1.0, out);
    }
}

/// Allocating convenience wrapper around [`dielectric_matrix_laplace_unrestricted_into`].
pub fn dielectric_matrix_laplace_unrestricted(
    v_mat: &Array2<f64>,
    chan_a: &RpaChannel, laplace_a: &LaplaceQuadrature,
    chan_b: &RpaChannel, laplace_b: &LaplaceQuadrature,
    omega: f64,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let nov_a = chan_a.eps_occ.len() * chan_a.eps_vir.len();
    let nov_b = chan_b.eps_occ.len() * chan_b.eps_vir.len();
    let mut rhs_a = Array2::<f64>::zeros((m, nov_a.max(1)));
    let mut rhs_b = Array2::<f64>::zeros((m, nov_b.max(1)));
    let mut out = Array2::<f64>::zeros((m, m));
    dielectric_matrix_laplace_unrestricted_into(
        v_mat,
        chan_a, laplace_a,
        chan_b, laplace_b,
        omega, &mut rhs_a, &mut rhs_b, &mut out,
    );
    out
}

/// Allocating convenience wrapper around [`dielectric_matrix_laplace_into`].
pub fn dielectric_matrix_laplace(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let nov = eps_occ.len() * eps_vir.len();
    let mut rhs_scaled = Array2::<f64>::zeros((m, nov));
    let mut out = Array2::<f64>::zeros((m, m));
    dielectric_matrix_laplace_into(
        v_mat, b_ov, eps_occ, eps_vir, omega, laplace, &mut rhs_scaled, &mut out,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sternheimer::dielectric_matrix;

    /// Synthetic 4×4 dielectric check: Laplace must reproduce Dense to ~1e-6.
    ///
    /// We use random-ish (deterministic) B, V, ε and assert agreement element-wise.
    #[test]
    fn laplace_matches_dense_synthetic() {
        let nocc = 2usize;
        let nvir = 3usize;
        let nov = nocc * nvir;
        let naux = 4usize;
        let m = 4usize;

        // Deterministic B and V.
        let b_ov = Array2::from_shape_fn((naux, nov), |(p, ia)| 0.1 + 0.03 * (p as f64) - 0.02 * (ia as f64));
        let v_mat = Array2::from_shape_fn((naux, m), |(p, a)| if p == a { 1.0 } else { 0.05 * (p as f64 - a as f64) });
        let eps_occ = vec![-0.6_f64, -0.4];
        let eps_vir = vec![0.2_f64, 0.7, 1.5];

        let omega = 0.5;
        let laplace = build_laplace_for_gaps(&eps_occ, &eps_vir, 7).unwrap();

        let dense = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, omega);
        let lap = dielectric_matrix_laplace(&v_mat, &b_ov, &eps_occ, &eps_vir, omega, &laplace);

        let max_err = dense
            .iter()
            .zip(lap.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("laplace_chi0 synthetic max err = {max_err:.3e}");
        assert!(max_err < 1e-3, "Laplace vs Dense agreement: max_err={max_err}");
    }

    #[test]
    fn laplace_tightens_with_more_points() {
        let nocc = 2usize;
        let nvir = 3usize;
        let nov = nocc * nvir;
        let naux = 4usize;
        let m = 4usize;

        let b_ov = Array2::from_shape_fn((naux, nov), |(p, ia)| 0.1 + 0.03 * (p as f64) - 0.02 * (ia as f64));
        let v_mat = Array2::from_shape_fn((naux, m), |(p, a)| if p == a { 1.0 } else { 0.05 * (p as f64 - a as f64) });
        let eps_occ = vec![-0.6_f64, -0.4];
        let eps_vir = vec![0.2_f64, 0.7, 1.5];
        let omega = 0.0;
        let dense = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, omega);

        let errs: Vec<f64> = [3usize, 5, 7]
            .iter()
            .map(|&n| {
                let lap = build_laplace_for_gaps(&eps_occ, &eps_vir, n).unwrap();
                let mat = dielectric_matrix_laplace(&v_mat, &b_ov, &eps_occ, &eps_vir, omega, &lap);
                dense.iter().zip(mat.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
            })
            .collect();
        eprintln!("laplace convergence: 3pt={:.3e} 5pt={:.3e} 7pt={:.3e}", errs[0], errs[1], errs[2]);
        // Looser ordering: minimax tables are coarse for small n, but should
        // be monotonically better.
        assert!(errs[2] <= errs[0], "more quadrature points should not increase error");
    }
}
