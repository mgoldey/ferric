//! PDEP Sternheimer response: independent-particle polarizability kernel.
//!
//! For a trial potential V (vector in the RI auxiliary basis), computes the
//! scalar χ(V, V; iω) = ⟨V|χ₀(iω)|V⟩ using the RI-MO Sternheimer equation.
//!
//! In the PDEP formalism all response is expressed via MO-basis B^P_ia
//! tensors — no AO-space Fock rebuild is needed at this level.

use crate::channel::RpaChannel;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

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
        // dsyrk_ takes *const u8 (Fortran char args); b"..\0" matches directly,
        // whereas a c"" literal is *const c_char and would need a cast.
        #[allow(clippy::manual_c_str_literals)]
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

/// Same as [`syrk_aat`], writing into a caller-provided `(m, m)` buffer instead
/// of allocating. Lets per-frequency callers reuse one output buffer across
/// many calls (see `dielectric_matrix_from_projection_into`).
pub(crate) fn syrk_aat_into(a: &Array2<f64>, c: &mut Array2<f64>) {
    let m = a.nrows();
    let kdim = a.ncols();
    assert_eq!(c.shape(), &[m, m], "syrk_aat_into: output buffer shape mismatch");
    if m == 0 || kdim == 0 {
        c.fill(0.0);
        return;
    }
    let a_slice = a
        .as_slice()
        .expect("syrk_aat_into: input must be contiguous (row-major)");
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let n_i32 = m as i32;
    let k_i32 = kdim as i32;
    let lda = kdim as i32;
    let ldc = m as i32;
    unsafe {
        #[allow(clippy::manual_c_str_literals)]
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
    for i in 0..m {
        for j in 0..i {
            c[(i, j)] = c[(j, i)];
        }
    }
}

/// Same as [`syrk_aat_into`], but accumulates (`beta=1`) into the caller's
/// `(m, m)` buffer instead of overwriting it (`beta=0`). Used by the
/// nov-panelled dielectric assembly ([`dielectric_matrix_from_projection_into_panelled`])
/// to sum `Σ_panels rhs_scaled_panel · rhs_scaled_panelᵀ` across panels without
/// ever materializing the full `(m, nov)` scaled projection at once. Caller is
/// responsible for zeroing `c` (or pre-seeding it, e.g. with the identity
/// diagonal) before the first panel.
///
/// Takes a view (not `&Array2`) so a full-width panel — which is already a
/// contiguous row-major sub-slice of the caller's `(m, panel_width)` scratch —
/// can be passed directly with no copy; only the caller's ragged final panel
/// (`w < panel_width`) needs an owned, right-sized `to_owned()` first.
fn syrk_aat_accumulate_into(a: ArrayView2<f64>, c: &mut Array2<f64>) {
    let m = a.nrows();
    let kdim = a.ncols();
    assert_eq!(c.shape(), &[m, m], "syrk_aat_accumulate_into: output buffer shape mismatch");
    if m == 0 || kdim == 0 {
        return;
    }
    let a_slice = a
        .as_slice()
        .expect("syrk_aat_accumulate_into: input must be contiguous (row-major)");
    let alpha = 1.0f64;
    let beta = 1.0f64; // accumulate, don't overwrite
    let n_i32 = m as i32;
    let k_i32 = kdim as i32;
    let lda = kdim as i32;
    let ldc = m as i32;
    unsafe {
        #[allow(clippy::manual_c_str_literals)]
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
    // BLAS only accumulated into the triangle it was asked to write (lower,
    // in the Fortran view = upper in row-major, same convention as
    // syrk_aat/syrk_aat_into). Mirror after EVERY panel is wasteful (the
    // off-triangle is stale from the previous panel's mirror); instead the
    // caller mirrors ONCE after the last panel — see
    // `dielectric_matrix_from_projection_into_panelled`.
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
    assert_eq!(b_ov.shape()[1], scale.len());

    // PySCF: chi0 = 2·e_ov·f_ov/(ω²+e_ov²) with e_ov = e_occ−e_vir < 0, f_ov = 2
    //        = 4 · (-e_ia) / (ω²+e_ia²) — negative
    // Ferric stores +|χ₀| = 4·e_ia/(ω²+e_ia²) so that ε̃ = I − Π matches PySCF I − χ₀.
    //
    // The projection y = Vᵀ·B_ov (m × nov) is frequency-independent; the per-ω
    // work (scale + DSYRK) lives in dielectric_matrix_from_projection.
    let y = v_mat.t().dot(b_ov); // (m, nov), owned & contiguous row-major
    dielectric_matrix_from_projection(&y, scale)
}

/// Build ε̃(iω) from a *precomputed* projection `y = Vᵀ·B_ov` (m × nov).
///
/// The projection is frequency-independent, so callers that sweep many
/// frequencies (the RPA quadrature loop) compute `y` once and call this per ω
/// with only the ω-dependent `scale`. Bit-identical to
/// [`dielectric_matrix_with_scale`], which recomputes `y` each call.
pub fn dielectric_matrix_from_projection(y: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let m = y.shape()[0];
    let nov = scale.len();
    assert_eq!(y.shape()[1], nov);

    // rhs_scaled[α,ia] = y[α,ia] · sqrt(4·e_ia/(ω²+e_ia²)); χ = rhs·rhsᵀ via DSYRK.
    let mut rhs_scaled = y.clone();
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut rhs_scaled)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);

    let mut eps_mat = syrk_aat(&rhs_scaled);
    for alpha in 0..m {
        eps_mat[(alpha, alpha)] += 1.0;
    }
    eps_mat
}

/// Same as [`dielectric_matrix_from_projection`], writing into caller-provided
/// `(m, nov)` and `(m, m)` scratch buffers instead of allocating a fresh
/// `y.clone()` + output on every call.
///
/// Exists because the RPA quadrature loop (`energy.rs::eval_eigenvalues_at_frequencies`)
/// calls this once per frequency inside a `rayon::par_iter` — allocating a full
/// `(naux, nov)` clone per call means up to `min(n_quad, active_threads)`
/// concurrent multi-GB clones, entirely unbounded by `[memory] budget_gb`
/// (measured: ~1.4 GB per clone at naux=2976/nov=61740, i.e. benzene
/// aug-cc-pVQZ). Reusing one `(rhs_scaled, out)` pair per rayon worker via
/// `map_init` (already the pattern in `dielectric_matrix_laplace_into`) caps
/// this at one buffer pair per THREAD instead of per FREQUENCY.
pub fn dielectric_matrix_from_projection_into(
    y: &Array2<f64>, scale: &Array1<f64>, rhs_scaled: &mut Array2<f64>, out: &mut Array2<f64>,
) {
    let m = y.shape()[0];
    let nov = scale.len();
    assert_eq!(y.shape()[1], nov);
    assert_eq!(rhs_scaled.shape(), &[m, nov], "rhs_scaled scratch shape");
    assert_eq!(out.shape(), &[m, m], "out shape");

    rhs_scaled.assign(y);
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut *rhs_scaled)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);

    syrk_aat_into(rhs_scaled, out);
    for alpha in 0..m {
        out[(alpha, alpha)] += 1.0;
    }
}

/// Budget-aware nov-panelled sibling of [`dielectric_matrix_from_projection_into`].
///
/// [`dielectric_matrix_from_projection_into`]'s `rhs_scaled` scratch is a full
/// `(m, nov)` buffer PER RAYON WORKER (via `map_init` in
/// `energy.rs::eval_eigenvalues_at_frequencies`) — the per-worker scratch
/// scales with `n_workers·(m·nov + m²)·8` bytes, budget-blind (this is the
/// term `budget.rs::estimate_peak_bytes` flags as the crux of the 2026-07-21
/// incident: more rayon workers ⇒ more concurrent scratch, with no cap).
///
/// This function instead scales `y`'s `nov` columns in contiguous panels of
/// width `panel_width`, SYRK-accumulating (`beta=1`) each panel's contribution
/// into `out` rather than materializing the whole `(m, nov)` scaled `y` at
/// once. Peak resident scratch per call is `O(m·panel_width)` instead of
/// `O(m·nov)` — and, since every full-width panel is passed to
/// `syrk_aat_accumulate_into` as a contiguous view with no extra copy (only a
/// possible ragged *final* panel, `w < panel_width`, needs one small owned
/// `to_owned()` copy strictly smaller than the persistent scratch), the
/// steady-state live set never doubles the `(m, panel_width)` buffer the way
/// an unconditional per-panel copy would. Mathematically identical to the
/// non-panelled version: SYRK is linear in its accumulation
/// (`Σ_panels A_panel·A_panelᵀ = A·Aᵀ` when the `A_panel`s partition `A`'s
/// columns), so panelling changes nothing about the result, only the peak
/// memory of computing it.
///
/// `rhs_scaled_panel` must have exactly `m` rows and at least 1 column; its
/// column count is read as the (fixed) panel width (the last panel may use
/// fewer columns via a sub-slice — see the loop body). `out` must be `(m, m)`.
///
/// Falls back to a single full-width "panel" (mathematically the same as
/// [`dielectric_matrix_from_projection_into`], just routed through the
/// panelled accumulation path) when `panel_width >= nov` — callers should
/// prefer the plain function in that case; this one is exposed primarily for
/// [`crate::energy::eval_eigenvalues_at_frequencies`] to dispatch on a
/// resolved budget.
pub fn dielectric_matrix_from_projection_into_panelled(
    y: &Array2<f64>,
    scale: &Array1<f64>,
    rhs_scaled_panel: &mut Array2<f64>,
    out: &mut Array2<f64>,
) {
    let m = y.shape()[0];
    let nov = scale.len();
    assert_eq!(y.shape()[1], nov);
    assert_eq!(out.shape(), &[m, m], "out shape");
    assert_eq!(rhs_scaled_panel.shape()[0], m, "rhs_scaled_panel row count must be m");
    let panel_width = rhs_scaled_panel.shape()[1].max(1);

    out.fill(0.0);
    let mut col0 = 0usize;
    while col0 < nov {
        let col1 = (col0 + panel_width).min(nov);
        let w = col1 - col0;

        // View the panel's columns of y and scale, sub-slicing the scratch
        // buffer down to the (possibly narrower) final panel width so
        // syrk_aat_accumulate_into sees exactly (m, w).
        let y_panel = y.slice(ndarray::s![.., col0..col1]);
        let mut panel_view = rhs_scaled_panel.slice_mut(ndarray::s![.., ..w]);
        panel_view.assign(&y_panel);
        let scale_panel = scale.slice(ndarray::s![col0..col1]);
        let scale_row = scale_panel.view().insert_axis(Axis(0));
        Zip::from(&mut panel_view)
            .and_broadcast(scale_row)
            .for_each(|x, &s| *x *= s);

        // syrk_aat_accumulate_into requires a contiguous row-major view. A
        // `..w` column sub-slice of the (m, panel_width) scratch IS contiguous
        // whenever w == panel_width (a full-column-range slice of a row-major
        // array is the whole array, unchanged strides) — true for every panel
        // except a ragged final one (w < panel_width), where the sub-slice's
        // row stride (panel_width) no longer matches its row length (w), so
        // `.as_slice()` would fail. Only that ragged tail needs an owned,
        // right-sized copy; every full-width panel is passed straight through
        // with no copy at all.
        if w == panel_width {
            syrk_aat_accumulate_into(panel_view.view(), out);
        } else {
            let owned_panel = panel_view.to_owned();
            syrk_aat_accumulate_into(owned_panel.view(), out);
        }

        col0 = col1;
    }
    for alpha in 0..m {
        out[(alpha, alpha)] += 1.0;
    }
    // Mirror upper → lower ONCE, after all panels have accumulated (each
    // panel's dsyrk_ call only wrote/accumulated into one triangle).
    for i in 0..m {
        for j in 0..i {
            out[(i, j)] = out[(j, i)];
        }
    }
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
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    omega: f64,
) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;

    let mut out: Array2<f64> = v_mat.to_owned();

    for chan in [chan_a, chan_b] {
        let RpaChannel { b_ov, eps_occ, eps_vir } = *chan;
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
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    omega: f64,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let mut eps_mat = Array2::<f64>::zeros((m, m));
    for alpha in 0..m { eps_mat[(alpha, alpha)] = 1.0; }
    for chan in [chan_a, chan_b] {
        let RpaChannel { b_ov, eps_occ, eps_vir } = *chan;
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

    /// Regression for the 2026-07-21 per-frequency-clone memory fix: the
    /// scratch-reusing `_into` variant (introduced so
    /// `energy.rs::eval_eigenvalues_at_frequencies` can use `map_init` instead
    /// of allocating a fresh `y.clone()` per quadrature frequency) must produce
    /// bit-identical output to the original allocating version on a
    /// non-trivial (m > 1, nov > 1) case.
    #[test]
    fn dielectric_matrix_from_projection_into_matches_allocating_version() {
        let y = ndarray::array![
            [1.0f64, 2.0, -0.5, 0.25],
            [0.3, -1.2, 0.8, 1.1],
            [-0.6, 0.4, 2.0, -0.9],
        ];
        let scale = ndarray::array![0.7f64, 1.3, 0.9, 2.1];

        let expected = dielectric_matrix_from_projection(&y, &scale);

        let m = y.shape()[0];
        let nov = y.shape()[1];
        let mut rhs_scaled = Array2::<f64>::zeros((m, nov));
        let mut out = Array2::<f64>::zeros((m, m));
        dielectric_matrix_from_projection_into(&y, &scale, &mut rhs_scaled, &mut out);

        assert_eq!(expected.shape(), out.shape());
        for ((i, j), &e) in expected.indexed_iter() {
            assert!(
                (out[(i, j)] - e).abs() < 1e-14,
                "mismatch at ({i},{j}): into={:?} allocating={e:?}",
                out[(i, j)]
            );
        }
    }

    /// The `_into` variant must be safely callable many times in a row with
    /// the SAME scratch buffers (the whole point of `map_init` reuse) — each
    /// call must fully overwrite its scratch, not accumulate stale data from a
    /// previous frequency.
    #[test]
    fn dielectric_matrix_from_projection_into_reuses_scratch_correctly_across_calls() {
        let y = ndarray::array![[1.0f64, -2.0], [0.5, 3.0]];
        let scale_a = ndarray::array![1.0f64, 1.0];
        let scale_b = ndarray::array![2.0f64, 0.5];

        let mut rhs_scaled = Array2::<f64>::zeros((2, 2));
        let mut out = Array2::<f64>::zeros((2, 2));

        dielectric_matrix_from_projection_into(&y, &scale_a, &mut rhs_scaled, &mut out);
        let first = out.clone();
        dielectric_matrix_from_projection_into(&y, &scale_b, &mut rhs_scaled, &mut out);
        let second = out.clone();

        // Different scale factors must give different results (proves the
        // second call actually recomputed rather than reusing stale `out`).
        // Element (0,0) happens to coincide for these particular scale/y
        // values (1²·1²+(-2)²·1² == 1²·2²+(-2)²·0.5² == 5, both +1 = 6) — use
        // (1,1), which genuinely differs (10.25 vs 4.25), so this assertion
        // can't pass by numerical coincidence.
        assert!(
            (first[(1, 1)] - second[(1, 1)]).abs() > 1e-10,
            "second call with a different scale should not match the first"
        );

        // And the second call's result must match a fresh allocating call
        // with the same inputs (proves reuse doesn't corrupt correctness).
        let expected_b = dielectric_matrix_from_projection(&y, &scale_b);
        for ((i, j), &e) in expected_b.indexed_iter() {
            assert!(
                (second[(i, j)] - e).abs() < 1e-14,
                "reused-scratch call diverged from a fresh allocating call at ({i},{j})"
            );
        }
    }

    /// Item 2b regression: the nov-panelled dielectric assembly
    /// (`dielectric_matrix_from_projection_into_panelled`) must reproduce the
    /// non-panelled allocating reference (`dielectric_matrix_from_projection`)
    /// to ~1e-12 at several panel widths spanning the full range: k=1 (every
    /// column its own panel — the narrowest possible, worst case for the
    /// accumulate-and-mirror-once bookkeeping), k=nov/2 (a ragged split when
    /// nov is odd), and k=nov (a single "panel" covering everything, the
    /// fallback-to-full-width case explicitly called out in the function's
    /// doc comment).
    #[test]
    fn dielectric_matrix_from_projection_into_panelled_matches_allocating_at_several_widths() {
        let y = ndarray::array![
            [1.0f64, 2.0, -0.5, 0.25, 0.6, -1.1],
            [0.3, -1.2, 0.8, 1.1, -0.4, 0.9],
            [-0.6, 0.4, 2.0, -0.9, 1.3, 0.2],
        ];
        let scale = ndarray::array![0.7f64, 1.3, 0.9, 2.1, 1.5, 0.4];
        let m = y.shape()[0];
        let nov = y.shape()[1];

        let expected = dielectric_matrix_from_projection(&y, &scale);

        for panel_width in [1usize, nov / 2, nov] {
            let panel_width = panel_width.max(1);
            let mut rhs_scaled_panel = Array2::<f64>::zeros((m, panel_width));
            let mut out = Array2::<f64>::zeros((m, m));
            dielectric_matrix_from_projection_into_panelled(&y, &scale, &mut rhs_scaled_panel, &mut out);

            for ((i, j), &e) in expected.indexed_iter() {
                assert!(
                    (out[(i, j)] - e).abs() < 1e-12,
                    "panel_width={panel_width}: mismatch at ({i},{j}): panelled={:?} allocating={e:?}",
                    out[(i, j)]
                );
            }
        }
    }

    /// The panelled path must also match the scratch-reusing `_into` version
    /// (not just the plain allocating one), since that's the direct
    /// non-panelled sibling `eval_eigenvalues_at_frequencies` falls back to
    /// when panelling isn't needed.
    #[test]
    fn dielectric_matrix_from_projection_into_panelled_matches_into_version() {
        let y = ndarray::array![
            [2.0f64, -0.3, 1.1, 0.4],
            [0.5, 1.7, -0.9, 0.2],
        ];
        let scale = ndarray::array![1.1f64, 0.6, 2.3, 0.9];
        let m = y.shape()[0];
        let nov = y.shape()[1];

        let mut rhs_scaled = Array2::<f64>::zeros((m, nov));
        let mut expected = Array2::<f64>::zeros((m, m));
        dielectric_matrix_from_projection_into(&y, &scale, &mut rhs_scaled, &mut expected);

        for panel_width in [1usize, 2, nov] {
            let mut rhs_scaled_panel = Array2::<f64>::zeros((m, panel_width));
            let mut out = Array2::<f64>::zeros((m, m));
            dielectric_matrix_from_projection_into_panelled(&y, &scale, &mut rhs_scaled_panel, &mut out);
            for ((i, j), &e) in expected.indexed_iter() {
                assert!(
                    (out[(i, j)] - e).abs() < 1e-12,
                    "panel_width={panel_width}: mismatch at ({i},{j}) vs _into version: {:?} vs {e:?}",
                    out[(i, j)]
                );
            }
        }
    }

    /// Repeated calls with the SAME scratch buffers (the `map_init` reuse
    /// pattern) must not accumulate stale data across calls — `out.fill(0.0)`
    /// at the top of the panelled function must genuinely reset state.
    #[test]
    fn dielectric_matrix_from_projection_into_panelled_reuses_scratch_correctly_across_calls() {
        let y = ndarray::array![[1.0f64, -2.0, 0.5], [0.5, 3.0, -1.0]];
        let scale_a = ndarray::array![1.0f64, 1.0, 1.0];
        let scale_b = ndarray::array![2.0f64, 0.5, 1.5];

        let mut rhs_scaled_panel = Array2::<f64>::zeros((2, 2)); // panel width 2 < nov=3
        let mut out = Array2::<f64>::zeros((2, 2));

        dielectric_matrix_from_projection_into_panelled(&y, &scale_a, &mut rhs_scaled_panel, &mut out);
        let first = out.clone();
        dielectric_matrix_from_projection_into_panelled(&y, &scale_b, &mut rhs_scaled_panel, &mut out);
        let second = out.clone();

        assert!(
            (first[(0, 0)] - second[(0, 0)]).abs() > 1e-10,
            "second call with a different scale should not match the first"
        );

        let expected_b = dielectric_matrix_from_projection(&y, &scale_b);
        for ((i, j), &e) in expected_b.indexed_iter() {
            assert!(
                (second[(i, j)] - e).abs() < 1e-12,
                "reused-scratch panelled call diverged from a fresh allocating call at ({i},{j})"
            );
        }
    }
}
