//! Per-orbital screened-tile Sternheimer kernel.
//!
//! Sparse equivalent of `sternheimer::dielectric_matrix` and
//! `sternheimer::dielectric_apply`. Consumes a `ScreenedBov` (per-orbital
//! tiles on Boys-localized occupied orbitals) instead of a dense
//! (naux × nocc·nvir) `b_ov`.
//!
//! # Subspace dielectric (ε̃ in the trial-vector basis)
//!
//! Mirrors `sternheimer::dielectric_matrix_with_scale`. For each i_loc:
//!   1. Gather rows of `v_mat` at `p_lists[i_loc]` → `v_gather` shape
//!      (m_i, msub) where msub = v_mat.ncols().
//!   2. Compute `rhs_i = v_gather.T @ tile_i` → shape (msub, nvir).
//!   3. Scale columns by `s_ia = sqrt(4·e_ia / (ω² + e_ia²))` with
//!      `e_ia = eps_vir[a] - eps_loc[i_loc]`.
//!   4. Accumulate `out += rhs_i @ rhs_i.T` (SYRK over msub-msub).
//!
//! Finally `out += I` to form `ε̃ = I + Π`.
//!
//! # Apply form (ε̃ · V)
//!
//! Used by block-Lanczos. For each i_loc:
//!   1. Gather rows as above.
//!   2. `y_i = v_gather.T @ tile_i` shape (msub, nvir).
//!   3. Scale columns by `s_ia^2`.
//!   4. Compute `contrib = tile_i @ y_i.T` shape (m_i, msub).
//!   5. Scatter rows of `contrib` back into `out` at `p_lists[i_loc]`.
//!
//! # Why this is correct
//!
//! Dense form: `Π_{PQ} = Σ_{ia} (4·e_ia/(ω²+e_ia²)) B^P_{ia} B^Q_{ia}`.
//! Split by occ index:
//!   `Π_{PQ} = Σ_i [ Σ_a (s_ia²/4 · 4) B^P_{ia} B^Q_{ia} ]`
//!          = `Σ_i (B_{i,:}^P diag(s_i²) B_{i,:}^Q,T)`
//! Per-i_loc contributions are independent and add. Screening drops aux rows
//! P with negligible `B^P_{i_loc, a}` for all a; the dropped contributions
//! are bounded by `thresh² · nvir`.

use crate::screen::ScreenedBov;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, Axis, Zip};

/// Build per-i_loc scale factors `s_ia = sqrt(4·e_ia/(ω²+e_ia²))`.
#[inline]
fn build_scale_for_iloc(eps_loc_i: f64, eps_vir: &[f64], omega: f64) -> Array1<f64> {
    let omega2 = omega * omega;
    let nvir = eps_vir.len();
    let mut s = Array1::<f64>::zeros(nvir);
    for (a, &eps_a) in eps_vir.iter().enumerate() {
        let e_ia = eps_a - eps_loc_i;
        s[a] = (4.0 * e_ia / (omega2 + e_ia * e_ia)).sqrt();
    }
    s
}

/// Subspace dielectric matrix `ε̃(iω)` evaluated through a screened B-tile
/// representation.
///
/// `v_mat` is (naux × msub). Returns an (msub × msub) symmetric matrix
/// `ε̃ = I + Π` matching `sternheimer::dielectric_matrix`.
pub fn dielectric_matrix_screened(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let msub = v_mat.ncols();
    let mut out = Array2::<f64>::zeros((msub, msub));

    for i_loc in 0..bov.n_occ_loc {
        let p_list = &bov.p_lists[i_loc];
        let tile = &bov.tiles[i_loc];
        let m_i = p_list.len();
        if m_i == 0 {
            continue;
        }

        // Gather rows of v_mat at p_list → v_gather shape (m_i, msub).
        let mut v_gather = Array2::<f64>::zeros((m_i, msub));
        for (slot, &p) in p_list.iter().enumerate() {
            v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
        }

        // rhs_i = v_gather.T @ tile  →  shape (msub, nvir).
        let mut rhs_i: Array2<f64> = v_gather.t().dot(tile);

        // Scale columns by s_ia.
        let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
        let scale_row = scale.view().insert_axis(Axis(0));
        Zip::from(&mut rhs_i)
            .and_broadcast(scale_row)
            .for_each(|x, &s| *x *= s);

        // Accumulate out += rhs_i @ rhs_i.T (msub × msub).
        // Plain GEMM accumulate keeps the code path simple; SYRK upgrade is
        // a follow-up if profiling demands it.
        let rhs_t = rhs_i.t().to_owned();
        general_mat_mul(1.0, &rhs_i, &rhs_t, 1.0, &mut out);
    }

    // Symmetrize (defensive — GEMM-of-A·A^T is symmetric in exact arithmetic,
    // but floating-point drift gives O(ε) asymmetry that the eigensolver does
    // not like).
    let out_sym = 0.5 * (&out + &out.t());

    // ε̃ = I + Π.
    let mut eps_mat = out_sym;
    for alpha in 0..msub {
        eps_mat[(alpha, alpha)] += 1.0;
    }
    eps_mat
}

/// Apply form: returns `ε̃(iω) · V` in the naux × msub space.
///
/// Used by block-Lanczos which needs A·V rather than V^T·A·V.
pub fn dielectric_apply_screened(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let naux = v_mat.nrows();
    let msub = v_mat.ncols();
    let mut out: Array2<f64> = v_mat.to_owned(); // identity contribution

    for i_loc in 0..bov.n_occ_loc {
        let p_list = &bov.p_lists[i_loc];
        let tile = &bov.tiles[i_loc];
        let m_i = p_list.len();
        if m_i == 0 {
            continue;
        }

        // Gather rows.
        let mut v_gather = Array2::<f64>::zeros((m_i, msub));
        for (slot, &p) in p_list.iter().enumerate() {
            v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
        }

        // y_i = v_gather.T @ tile  →  (msub, nvir)
        let mut y_i: Array2<f64> = v_gather.t().dot(tile);

        // Scale columns by s_ia².
        let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
        let scale_row = scale.view().insert_axis(Axis(0));
        Zip::from(&mut y_i)
            .and_broadcast(scale_row)
            .for_each(|x, &s| *x *= s * s);

        // contrib = tile @ y_i.T  →  (m_i, msub)
        let y_t = y_i.t().to_owned();
        let mut contrib = Array2::<f64>::zeros((m_i, msub));
        general_mat_mul(1.0, tile, &y_t, 0.0, &mut contrib);

        // Scatter rows back into out.
        for (slot, &p) in p_list.iter().enumerate() {
            let mut row = out.slice_mut(s![p, ..]);
            let crow = contrib.slice(s![slot, ..]);
            for col in 0..msub {
                row[col] += crow[col];
            }
        }
    }

    let _ = naux;
    out
}
