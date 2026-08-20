//! cDFT electron-transfer coupling (Wu–Van Voorhis H_ab) between two
//! charge-constrained UHF diabatic states.
//!
//! The kernel is the matrix element of a one-body operator between two
//! non-orthogonal UHF determinants with different MO sets. Per spin, SVD the
//! occupied-MO overlap M = (C_a^occ)ᵀ S C_b^occ = U Σ Vᵀ (Löwdin pairing);
//! the determinant overlap is the product of singular values, and one-body
//! elements use the reduced-overlap form, with the 0/1/≥2 near-zero
//! singular-value cases handled explicitly.
//!
//! Reference: Wu & Van Voorhis, J. Chem. Phys. 125, 164105 (2006).

use ndarray::{Array1, Array2};
use ndarray_linalg::SVD;

/// Singular-value threshold below which a paired orbital is "zero overlap".
const S_TOL: f64 = 1e-8;

/// Löwdin pairing of two occupied MO sets for one spin channel.
#[derive(Debug, Clone)]
pub struct Pairing {
    /// det(M) = Π singular values (non-negative; sign handled via the SVD
    /// rotations folded into c_tilde).
    pub det_m: f64,
    /// Singular values, descending.
    pub s_vals: Array1<f64>,
    /// Rotated occupied coeffs C̃_a = C_a^occ · U, shape (nbf, nocc).
    pub c_tilde_a: Array2<f64>,
    /// Rotated occupied coeffs C̃_b = C_b^occ · V, shape (nbf, nocc).
    pub c_tilde_b: Array2<f64>,
}

/// SVD-pair two occupied MO sets. `c_occ_a`/`c_occ_b` are (nbf, nocc).
pub fn biorth_pairing(
    c_occ_a: &Array2<f64>,
    c_occ_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Pairing {
    // M = C_aᵀ S C_b, shape (nocc, nocc).
    let m = c_occ_a.t().dot(s).dot(c_occ_b);
    let (u_opt, sigma, vt_opt) = m.svd(true, true).expect("SVD of MO-overlap failed");
    let u = u_opt.expect("svd U");
    let vt = vt_opt.expect("svd Vt");
    // Rotate: C̃_a = C_a U,  C̃_b = C_b V = C_b (Vt)ᵀ.
    let c_tilde_a = c_occ_a.dot(&u);
    let c_tilde_b = c_occ_b.dot(&vt.t());
    let det_m: f64 = sigma.iter().product();
    Pairing { det_m, s_vals: sigma, c_tilde_a, c_tilde_b }
}

/// ⟨Ψ_a|Ô|Ψ_b⟩ for a one-body AO operator Ô, given both spins' pairings and
/// the total overlap S_ab = pair_α.det_m · pair_β.det_m.
///
/// Per spin the reduced contribution is Σ_i ⟨ã_i|Ô|b̃_i⟩ / s_i, valid when all
/// s_i > S_TOL. Near-zero s_i are handled by the cofactor rules:
///   - one spin with exactly one zero s_k: that spin contributes only the k-th
///     paired term with the zero excluded from the det prefactor; the other
///     spin contributes its full det. The element is finite though S_ab = 0.
///   - ≥2 zeros total (in one spin, or one in each spin): a one-body operator
///     connects determinants differing by ≤1 orbital, so the element is 0.
pub fn cross_one_body(
    op_ao: &Array2<f64>,
    pair_alpha: &Pairing,
    pair_beta: &Pairing,
    s_ab: f64,
) -> f64 {
    // Per-spin paired diagonal d_i = ⟨ã_i|Ô|b̃_i⟩.
    let diag = |p: &Pairing| -> Vec<f64> {
        let n = p.s_vals.len();
        (0..n)
            .map(|i| {
                let a = p.c_tilde_a.column(i);
                let b = p.c_tilde_b.column(i);
                // aᵀ Op b
                a.dot(&op_ao.dot(&b))
            })
            .collect()
    };
    let da = diag(pair_alpha);
    let db = diag(pair_beta);

    // Count near-zero singular values per spin.
    let zeros = |p: &Pairing| -> Vec<usize> {
        p.s_vals.iter().enumerate()
            .filter(|(_, &s)| s < S_TOL)
            .map(|(i, _)| i).collect()
    };
    let za = zeros(pair_alpha);
    let zb = zeros(pair_beta);
    let nz = za.len() + zb.len();

    if nz == 0 {
        // Generic: S_ab · Σ_σ Σ_i d_i / s_i.
        let red = |p: &Pairing, d: &[f64]| -> f64 {
            p.s_vals.iter().zip(d).map(|(s, di)| di / s).sum::<f64>()
        };
        s_ab * (red(pair_alpha, &da) + red(pair_beta, &db))
    } else if nz == 1 {
        // Exactly one zero, in one spin. The element is
        //   (Π_{i≠k} s_i^that-spin) · d_k^that-spin · (det of the other spin),
        // i.e. cofactor: the zero orbital is the only connecting pair.
        let (p_zero, d_zero, k, det_other) = if za.len() == 1 {
            (pair_alpha, &da, za[0], pair_beta.det_m)
        } else {
            (pair_beta, &db, zb[0], pair_alpha.det_m)
        };
        let prod_nonzero: f64 = p_zero.s_vals.iter().enumerate()
            .filter(|(i, _)| *i != k)
            .map(|(_, s)| *s)
            .product();
        prod_nonzero * d_zero[k] * det_other
    } else {
        // ≥2 zeros: one-body operator cannot connect determinants differing by
        // more than one orbital.
        0.0
    }
}

/// A converged constrained UHF diabatic state, viewed for coupling.
#[derive(Debug)]
pub struct DiabaticState<'a> {
    /// α MO coefficients (nbf, nbf); occupied = first `nocc_a` columns.
    pub c_a: &'a Array2<f64>,
    /// β MO coefficients (nbf, nbf); occupied = first `nocc_b` columns.
    pub c_b: &'a Array2<f64>,
    pub nocc_a: usize,
    pub nocc_b: usize,
    /// State energy (CdftResult.scf.energy).
    pub energy: f64,
    /// Constraint multiplier (single constraint, k=1).
    pub lambda: f64,
    /// Constraint weight operator W^C in AO basis.
    pub w: &'a Array2<f64>,
}

/// Result of a coupling evaluation.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct HabResult {
    pub h_ab: f64,
    pub s_ab: f64,
    pub e_a: f64,
    pub e_b: f64,
}

/// Wu–VV electronic coupling between two constrained UHF states.
pub fn coupling_hab(
    state_a: &DiabaticState,
    state_b: &DiabaticState,
    s: &Array2<f64>,
) -> HabResult {
    let ca_occ_a = state_a.c_a.slice(ndarray::s![.., ..state_a.nocc_a]).to_owned();
    let ca_occ_b = state_a.c_b.slice(ndarray::s![.., ..state_a.nocc_b]).to_owned();
    let cb_occ_a = state_b.c_a.slice(ndarray::s![.., ..state_b.nocc_a]).to_owned();
    let cb_occ_b = state_b.c_b.slice(ndarray::s![.., ..state_b.nocc_b]).to_owned();

    let pair_alpha = biorth_pairing(&ca_occ_a, &cb_occ_a, s);
    let pair_beta = biorth_pairing(&ca_occ_b, &cb_occ_b, s);
    let s_ab = pair_alpha.det_m * pair_beta.det_m;

    // ⟨Ψ_a|Ŵ_a|Ψ_b⟩ and ⟨Ψ_a|Ŵ_b|Ψ_b⟩.
    let w_a_elem = cross_one_body(state_a.w, &pair_alpha, &pair_beta, s_ab);
    let w_b_elem = cross_one_body(state_b.w, &pair_alpha, &pair_beta, s_ab);

    // Symmetric Wu–VV raw element:
    //   H_raw = ½[(E_b S_ab − λ_b⟨a|W_b|b⟩) + (E_a S_ab − λ_a⟨a|W_a|b⟩)].
    let e_a = state_a.energy;
    let e_b = state_b.energy;
    let h_raw = 0.5
        * ((e_b * s_ab - state_b.lambda * w_b_elem)
            + (e_a * s_ab - state_a.lambda * w_a_elem));

    // Symmetric orthogonalization. Guard the degenerate denominator.
    let denom = 1.0 - s_ab * s_ab;
    let h_ab = if denom.abs() < 1e-10 {
        // |S_ab| → 1: the two states are (nearly) identical; the coupling is
        // undefined / the off-diagonal collapses to the diagonal. Report E_a.
        e_a
    } else {
        (h_raw - 0.5 * (e_a + e_b) * s_ab) / denom
    };

    HabResult { h_ab, s_ab, e_a, e_b }
}
