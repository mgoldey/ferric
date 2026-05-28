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
