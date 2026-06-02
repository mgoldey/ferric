//! PDEP Sternheimer response: independent-particle polarizability kernel.
//!
//! For a trial potential V (vector in the RI auxiliary basis), computes the
//! scalar χ(V, V; iω) = ⟨V|χ₀(iω)|V⟩ using the RI-MO Sternheimer equation.
//!
//! In the PDEP formalism all response is expressed via MO-basis B^P_ia
//! tensors — no AO-space Fock rebuild is needed at this level.

use ndarray::{Array1, Array2, Axis, Zip};

// Direct BLAS DSYRK binding — OpenBLAS is already linked via openblas-src.
// Avoids adding a cblas/blas-sys dependency just for a symmetric rank-k update.
// Fortran interface: dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc).
extern "C" {
    fn dsyrk_(
        uplo: *const u8,
        trans: *const u8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

/// Compute C = A · A^T using DSYRK (symmetric rank-k update).
///
/// `a` is a row-major (m, k) matrix; the result is a row-major (m, m) symmetric
/// matrix. Only the lower triangle is computed by BLAS, then mirrored to the
/// upper triangle so callers see a fully-populated symmetric matrix.
///
/// Reading row-major (m, k) as Fortran-order gives a (k, m) matrix with lda=k.
/// Computing A_F^T · A_F (n=m, k=k, trans='T', lda=k) yields the symmetric
/// (m, m) result we want.
pub(crate) fn syrk_aat(a: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let kdim = a.ncols();
    let mut c = Array2::<f64>::zeros((m, m));
    if m == 0 || kdim == 0 {
        return c;
    }
    // Require contiguous row-major storage so the Fortran view is valid.
    let a_slice = a
        .as_slice()
        .expect("syrk_aat: input must be contiguous (row-major)");
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let n_i32 = m as i32;
    let k_i32 = kdim as i32;
    let lda = kdim as i32; // Fortran leading dim of the transposed view
    let ldc = m as i32;
    // Compute LOWER triangle (Fortran 'L') of A^T·A in Fortran view.
    // In row-major terms this populates the UPPER triangle of C — symmetric in
    // either reading, so we mirror to fill both halves.
    unsafe {
        dsyrk_(
            b"L\0".as_ptr(),
            b"T\0".as_ptr(),
            &n_i32,
            &k_i32,
            &alpha,
            a_slice.as_ptr(),
            &lda,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
    // Mirror upper → lower (row-major). BLAS wrote one triangle; copy across.
    for i in 0..m {
        for j in 0..i {
            c[(i, j)] = c[(j, i)];
        }
    }
    c
}

/// Build the (nov,) array of per-(ia) scale factors s_ia = sqrt(4·e_ia / (ω²+e_ia²)).
///
/// Hoisted out of `dielectric_matrix` so callers iterating over many frequencies
/// can keep the dominant GEMM/SYRK costs in BLAS and avoid scalar work entirely.
///
/// The factor 4 = 2 (spin pairs) · 2 (occupation, closed-shell). For
/// open-shell U-RPA, build per-spin factors with prefactor 2 each via
/// [`build_scale_factors_with_prefactor`] and sum the two contributions.
#[inline]
pub fn build_scale_factors(eps_occ: &[f64], eps_vir: &[f64], omega: f64) -> Array1<f64> {
    build_scale_factors_with_prefactor(eps_occ, eps_vir, omega, 4.0)
}

/// Build scale factors s_ia = sqrt(prefactor·e_ia / (ω²+e_ia²)).
///
/// Use `prefactor=4` for closed-shell (single B_ov), `prefactor=2` per spin
/// channel for open-shell (two B_ov tensors summed).
#[inline]
pub fn build_scale_factors_with_prefactor(
    eps_occ: &[f64], eps_vir: &[f64], omega: f64, prefactor: f64,
) -> Array1<f64> {
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let omega2 = omega * omega;
    let mut s = Array1::<f64>::zeros(nocc * nvir);
    for (i, &eps_i) in eps_occ.iter().enumerate() {
        for (a, &eps_a) in eps_vir.iter().enumerate() {
            let e_ia = eps_a - eps_i;
            s[i * nvir + a] = (prefactor * e_ia / (omega2 + e_ia * e_ia)).sqrt();
        }
    }
    s
}

/// Compute ⟨V|χ₀(iω)|V⟩ for a single trial potential V.
///
/// Returns −2 Σ_{ia} (Σ_P V_P B^P_ia)² / (ε_a − ε_i + ω). This is ≤ 0.
pub fn chi_from_trial_potential(
    v: &Array1<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> f64 {
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let nov = nocc * nvir;
    assert_eq!(b_ov.shape()[1], nov);

    // rhs_ia = Σ_P V_P B^P_ia  (shape: nov)
    let rhs = v.dot(b_ov); // shape (nov,)

    let mut chi = 0.0f64;
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            let e_ia = eps_vir[a] - eps_occ[i];
            chi -= 2.0 * e_ia / (omega * omega + e_ia * e_ia) * rhs[ia] * rhs[ia];
        }
    }
    chi
}

/// Compute the dielectric matrix ε̃_αβ(iω) = δ_αβ − χ₀_αβ(iω) in a subspace.
///
/// Given trial potentials V of shape (naux, m) (columns = trial vecs),
/// stores positive 2 Σ_{ia} and adds 1 to diagonal to form I − χ₀.
/// Returns the m×m symmetric matrix with eigenvalues ≥ 1 for a physical system.
pub fn dielectric_matrix(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let scale = build_scale_factors(eps_occ, eps_vir, omega);
    dielectric_matrix_with_scale(v_mat, b_ov, &scale)
}

/// Same as `dielectric_matrix` but takes a precomputed scale-factor array.
///
/// Hot-path entry point for callers that evaluate the dielectric matrix at many
/// frequencies: scale factors depend only on (ω, ε_occ, ε_vir), not on `v_mat`
/// or `b_ov`, so they can be built once per (ω, orbital set).
pub fn dielectric_matrix_with_scale(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    scale: &Array1<f64>,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let nov = scale.len();
    assert_eq!(b_ov.shape()[1], nov);

    // rhs_mat: (m, nov) — rhs for each trial vector
    //
    // PySCF: chi0 = 2·e_ov·f_ov/(ω²+e_ov²) with e_ov = e_occ−e_vir < 0, f_ov = 2
    //        = 4 · (-e_ia) / (ω²+e_ia²) — negative
    // Ferric stores +|χ₀| = 4·e_ia/(ω²+e_ia²) so that ε̃ = I − Π matches PySCF I − χ₀.
    //
    // SYRK path: rhs_scaled[α,ia] = rhs_mat[α,ia] * sqrt(4·e_ia/(ω²+e_ia²))
    //   chi = rhs_scaled @ rhs_scaled^T  (one DSYRK replaces DGEMM, ~2× faster).
    let mut rhs_scaled = v_mat.t().dot(b_ov); // (m, nov), owned & contiguous row-major
    // Broadcast row-scale: multiply column `ia` by scale[ia] in place.
    let scale_row = scale.view().insert_axis(Axis(0)); // shape (1, nov)
    Zip::from(&mut rhs_scaled)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);

    // chi = rhs_scaled @ rhs_scaled^T via DSYRK; result is fully symmetrized.
    let mut eps_mat = syrk_aat(&rhs_scaled);

    // Return ε̃ = I − χ₀ = I + Π (since Π = −χ₀, χ₀ < 0 at iω).
    // Eigenvalues are 1 + μ_α ≥ 1 for physical systems.
    for alpha in 0..m {
        eps_mat[(alpha, alpha)] += 1.0;
    }
    eps_mat
}

/// Apply the dielectric matrix to a block of trial vectors: returns ε̃ · V.
///
/// Used by the block-Lanczos eigensolver, which needs A·V rather than V^T·A·V.
pub fn dielectric_apply(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let scale = build_scale_factors(eps_occ, eps_vir, omega);
    let nov = scale.len();
    assert_eq!(b_ov.shape()[1], nov);

    // y = V^T · B_ov   (m × nov)
    let mut y: Array2<f64> = v_mat.t().dot(b_ov);

    // Scale columns by s_ia²: ε̃ = I + B^T diag(s²) B.
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut y)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s * s);

    // out = V + B_ov · y^T   (naux × m)
    let mut out: Array2<f64> = v_mat.to_owned();
    general_mat_mul(1.0, b_ov, &y.t(), 1.0, &mut out);
    out
}

/// Unrestricted dielectric apply: ε̃_U = I + Π_α + Π_β.
///
/// Each spin channel contributes Π_σ = B_ov_σ diag(2·e_iaσ/(ω²+e_iaσ²)) B_ov_σ^T.
/// Closed-shell `dielectric_apply` is the special case with B_α = B_β,
/// e_α = e_β, prefactor 4 instead of 2+2 — both produce the same Π for a
/// closed-shell density when run on the spin-symmetric SCF result.
pub fn dielectric_apply_unrestricted(
    v_mat: &Array2<f64>,
    b_ov_a: &Array2<f64>, eps_occ_a: &[f64], eps_vir_a: &[f64],
    b_ov_b: &Array2<f64>, eps_occ_b: &[f64], eps_vir_b: &[f64],
    omega: f64,
) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;

    let mut out: Array2<f64> = v_mat.to_owned();

    for (b_ov, eps_occ, eps_vir) in [
        (b_ov_a, eps_occ_a, eps_vir_a),
        (b_ov_b, eps_occ_b, eps_vir_b),
    ] {
        let scale = build_scale_factors_with_prefactor(eps_occ, eps_vir, omega, 2.0);
        let nov = scale.len();
        assert_eq!(b_ov.shape()[1], nov);

        let mut y: Array2<f64> = v_mat.t().dot(b_ov);
        let scale_row = scale.view().insert_axis(Axis(0));
        Zip::from(&mut y)
            .and_broadcast(scale_row)
            .for_each(|x, &s| *x *= s * s);

        // out += B_ov · y^T
        general_mat_mul(1.0, b_ov, &y.t(), 1.0, &mut out);
    }
    out
}

/// Unrestricted ε̃ in subspace V: V^T (I + Π_α + Π_β) V.
pub fn dielectric_matrix_unrestricted(
    v_mat: &Array2<f64>,
    b_ov_a: &Array2<f64>, eps_occ_a: &[f64], eps_vir_a: &[f64],
    b_ov_b: &Array2<f64>, eps_occ_b: &[f64], eps_vir_b: &[f64],
    omega: f64,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let mut eps_mat = Array2::<f64>::zeros((m, m));
    for alpha in 0..m { eps_mat[(alpha, alpha)] = 1.0; }
    for (b_ov, eps_occ, eps_vir) in [
        (b_ov_a, eps_occ_a, eps_vir_a),
        (b_ov_b, eps_occ_b, eps_vir_b),
    ] {
        let scale = build_scale_factors_with_prefactor(eps_occ, eps_vir, omega, 2.0);
        let mut rhs_scaled = v_mat.t().dot(b_ov);
        let scale_row = scale.view().insert_axis(Axis(0));
        Zip::from(&mut rhs_scaled)
            .and_broadcast(scale_row)
            .for_each(|x, &s| *x *= s);
        let chi_sigma = syrk_aat(&rhs_scaled);
        eps_mat += &chi_sigma;
    }
    eps_mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_static_two_level_system() {
        // 1 occupied, 1 virtual, 1 aux function.
        // B^P_ia = 1.0, eps_occ = -0.5, eps_vir = 0.5
        // χ₀(0) = -2 * B^2 / (eps_vir - eps_occ) = -2 * 1.0 / 1.0 = -2.0
        let b_ov = ndarray::array![[1.0f64]]; // shape (1, 1): naux=1, nocc*nvir=1
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64];
        let v = ndarray::array![1.0f64]; // trial potential, naux=1
        let chi = chi_from_trial_potential(&v, &b_ov, &eps_occ, &eps_vir, 0.0);
        assert!(
            (chi + 2.0).abs() < 1e-12,
            "expected χ = -2.0, got {}",
            chi
        );
    }

    #[test]
    fn chi_freq_shift() {
        // Same system at omega=1.0
        // χ₀(i·1) = -2 * 1.0 / (1.0 + 1.0) = -1.0
        let b_ov = ndarray::array![[1.0f64]];
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64];
        let v = ndarray::array![1.0f64];
        let chi = chi_from_trial_potential(&v, &b_ov, &eps_occ, &eps_vir, 1.0);
        assert!(
            (chi + 1.0).abs() < 1e-12,
            "expected χ = -1.0, got {}",
            chi
        );
    }

    #[test]
    fn dielectric_matrix_identity_at_zero_coupling() {
        // With B=0, dielectric should be identity.
        use ndarray::Array2;
        let b_ov = Array2::zeros((2, 2));
        let v_mat = ndarray::array![[1.0, 0.0], [0.0, 1.0f64]];
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64, 1.5f64];
        let eps = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, 0.0);
        assert!((eps[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((eps[(1, 1)] - 1.0).abs() < 1e-12);
        assert!(eps[(0, 1)].abs() < 1e-12);
    }

    #[test]
    fn dielectric_matrix_two_level_system() {
        use ndarray_linalg::{Eigh, UPLO};
        // 1 occ, 1 vir, 1 aux. B=1, e_ia=1, ω=0:
        // Π = 4*e_ia/(0+e_ia²)·B² = 4  (RHF factor 4)
        // ε̃ = I + Π = 1 + 4 = 5 (= I − χ₀ since χ₀ = −Π < 0)
        let b_ov = ndarray::array![[1.0f64]];
        let v_mat = ndarray::array![[1.0f64]];
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64];
        let eps = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, 0.0);

        let (evals, _) = eps.eigh(UPLO::Upper).expect("Failed to diagonalize");
        assert!(
            (evals[0] - 5.0).abs() < 1e-12,
            "expected ε̃=5 (I + Π with RHF factor 4), got {}",
            evals[0]
        );
    }
}
