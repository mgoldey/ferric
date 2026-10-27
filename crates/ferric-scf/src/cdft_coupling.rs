//! cDFT electron-transfer coupling (Wu–Van Voorhis H_ab) between two
//! charge-constrained UHF diabatic states.
//!
//! The kernel is the matrix element of a one-body operator between two
//! non-orthogonal UHF determinants with different MO sets. Per spin, SVD the
//! occupied-MO overlap M = (C_a^occ)ᵀ S C_b^occ = U Σ Vᵀ (Löwdin pairing);
//! the determinant overlap is the product of singular values, and one-body
//! elements use the division-free cofactor form of
//! [`cross_one_body`](crate::cdft_coupling::cross_one_body), which is exact
//! and continuous for any number of zero singular values.
//!
//! # The raw element (constraint-offset invariant form)
//!
//! A converged constrained state Φ_X is the stationary point of
//! E\[ρ\] + λ_X(∫w_X ρ − N_X), so it is treated (Wu & Van Voorhis) as an
//! eigenfunction of Ĥ + λ_X ŵ_X with eigenvalue F_X = E_X + λ_X N_X, where
//! ŵ_X = Σ_i w_X(r_i) is the constraint's population operator and
//! N_X = ⟨Φ_X|ŵ_X|Φ_X⟩. Projecting onto the other state:
//!
//! ```text
//! ⟨Φ_A|Ĥ|Φ_B⟩ ≈ F_B S_AB − λ_B ⟨Φ_A|ŵ_B|Φ_B⟩,   F_B = E_B + λ_B N_B
//! H_raw = ½[(F_A + F_B) S_AB − λ_A⟨A|ŵ_A|B⟩ − λ_B⟨A|ŵ_B|B⟩]
//! H_ab  = [H_raw − ½(E_A + E_B) S_AB] / (1 − S_AB²)
//! ```
//!
//! Equivalently each λ multiplies ⟨A|ŵ_X − N_X|B⟩, the matrix element of the
//! constraint RESIDUAL operator. That is what makes H_ab invariant under the
//! physically empty redefinition ŵ → ŵ − c·N̂ (target N → N − c·N_e): the
//! states, λ and E are unchanged, and so is H_ab. The form with E_B in place
//! of F_B is NOT invariant and, on a symmetric dimer whose two fragment
//! weights partition unity, overstates |H_ab| by exactly (N_e/2)/(N_e/2 − N)
//! (3× for He₂⁺ with N = 1). Measured against NWChem's direct ⟨A|Ĥ|B⟩ on
//! He₂⁺/PBE (`tests/validation_cdft_et.rs`), the F form is within 1.4–6% of
//! the direct coupling at every point and the E-only form is about 3× high.
//!
//! N_X is evaluated as the state's OWN population ⟨Φ_X|ŵ_X|Φ_X⟩ from its
//! occupied MOs (equal to the target at convergence), so the invariance holds
//! exactly for any operator offset, not only to the outer-loop tolerance.
//! Charge (`SpinChannel::Total`) constraints only: the same ŵ acts on both
//! spins here, as in [`cross_one_body`](crate::cdft_coupling::cross_one_body).
//!
//! References: Q. Wu & T. Van Voorhis, J. Chem. Phys. 125, 164105 (2006);
//! B. Kaduk, T. Kowalczyk & T. Van Voorhis, Chem. Rev. 112, 321 (2012).

use ndarray::{Array1, Array2};
use ndarray_linalg::SVD;

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
pub fn biorth_pairing(c_occ_a: &Array2<f64>, c_occ_b: &Array2<f64>, s: &Array2<f64>) -> Pairing {
    // M = C_aᵀ S C_b, shape (nocc, nocc).
    let m = c_occ_a.t().dot(s).dot(c_occ_b);
    let (u_opt, sigma, vt_opt) = m.svd(true, true).expect("SVD of MO-overlap failed");
    let u = u_opt.expect("svd U");
    let vt = vt_opt.expect("svd Vt");
    // Rotate: C̃_a = C_a U,  C̃_b = C_b V = C_b (Vt)ᵀ.
    let c_tilde_a = c_occ_a.dot(&u);
    let c_tilde_b = c_occ_b.dot(&vt.t());
    let det_m: f64 = sigma.iter().product();
    Pairing {
        det_m,
        s_vals: sigma,
        c_tilde_a,
        c_tilde_b,
    }
}

/// Σ_i d_i Π_{j≠i} s_j, the derivative of Π_j (s_j + t·d_j) at t = 0.
///
/// Built from prefix and suffix products, so no s_j is ever divided by: the
/// result is a polynomial in the singular values and stays exact (and
/// continuous) when any of them is zero.
fn cofactor_sum(s_vals: &Array1<f64>, d: &[f64]) -> f64 {
    let n = s_vals.len();
    let mut suffix = vec![1.0; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] * s_vals[i];
    }
    let mut prefix = 1.0;
    let mut total = 0.0;
    for i in 0..n {
        total += d[i] * prefix * suffix[i + 1];
        prefix *= s_vals[i];
    }
    total
}

/// Paired diagonal d_i = ⟨ã_i|Ô|b̃_i⟩ of a one-body AO operator for one spin.
fn paired_diagonal(op_ao: &Array2<f64>, p: &Pairing) -> Vec<f64> {
    (0..p.s_vals.len())
        .map(|i| {
            p.c_tilde_a
                .column(i)
                .dot(&op_ao.dot(&p.c_tilde_b.column(i)))
        })
        .collect()
}

/// ⟨Ψ_a|Ô|Ψ_b⟩ for a one-body AO operator Ô, given both spins' pairings.
///
/// In the Löwdin-paired basis the spin-σ overlap matrix is diag(s^σ), and the
/// generalized Slater–Condon rule gives the one-body element as the adjugate
/// trace tr(adj(M_σ) O_σ) times the other spin's determinant:
///
/// ```text
/// ⟨Ψ_a|Ô|Ψ_b⟩ = Σ_σ [Σ_i d_i^σ Π_{j≠i} s_j^σ] · det_σ'
/// ```
///
/// with d_i^σ = ⟨ã_i|Ô|b̃_i⟩ and det_σ' = Π s^σ'. This is the derivative of
/// det(M + t·O) and contains no division, so it is exact for any number of
/// zero singular values: one zero leaves only that pair's cofactor term, two or
/// more give zero, and the value is continuous as a singular value passes
/// through zero. When every s_i is nonzero it equals the reduced-overlap form
/// S_ab · Σ_σ Σ_i d_i^σ / s_i^σ.
///
/// It is linear in Ô, so W → W − c·S shifts it by exactly −c·N_e·S_ab (each
/// d_i^σ moves by −c·s_i^σ). An earlier thresholded version treated
/// s_k < 1e-8 as an exact zero and kept only the k-th term, which shifted by
/// −c·S_ab instead and left ½·c·(N_e − 1)·S_ab·(λ_a + λ_b) in H_ab.
pub fn cross_one_body(op_ao: &Array2<f64>, pair_alpha: &Pairing, pair_beta: &Pairing) -> f64 {
    let alpha = cofactor_sum(&pair_alpha.s_vals, &paired_diagonal(op_ao, pair_alpha));
    let beta = cofactor_sum(&pair_beta.s_vals, &paired_diagonal(op_ao, pair_beta));
    alpha * pair_beta.det_m + beta * pair_alpha.det_m
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
    /// Constraint weight operator W^C in AO basis (the population operator
    /// whose expectation the constraint drives to its target). Any constant
    /// offset W − c·S gives the same H_ab; the state's own N = ⟨W⟩ is
    /// recomputed from its MOs.
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

/// ⟨Ψ|Ŵ|Ψ⟩ = Σ_σ tr(C_σ,occᵀ W C_σ,occ) for the state's own weight operator.
fn own_population(st: &DiabaticState, c_occ_a: &Array2<f64>, c_occ_b: &Array2<f64>) -> f64 {
    let tr = |c: &Array2<f64>| -> f64 { c.t().dot(st.w).dot(c).diag().sum() };
    tr(c_occ_a) + tr(c_occ_b)
}

/// Wu–VV electronic coupling between two constrained UHF states, in the
/// constraint-offset-invariant form derived in the module docs.
pub fn coupling_hab(
    state_a: &DiabaticState,
    state_b: &DiabaticState,
    s: &Array2<f64>,
) -> HabResult {
    let ca_occ_a = state_a
        .c_a
        .slice(ndarray::s![.., ..state_a.nocc_a])
        .to_owned();
    let ca_occ_b = state_a
        .c_b
        .slice(ndarray::s![.., ..state_a.nocc_b])
        .to_owned();
    let cb_occ_a = state_b
        .c_a
        .slice(ndarray::s![.., ..state_b.nocc_a])
        .to_owned();
    let cb_occ_b = state_b
        .c_b
        .slice(ndarray::s![.., ..state_b.nocc_b])
        .to_owned();

    let pair_alpha = biorth_pairing(&ca_occ_a, &cb_occ_a, s);
    let pair_beta = biorth_pairing(&ca_occ_b, &cb_occ_b, s);
    let s_ab = pair_alpha.det_m * pair_beta.det_m;

    // ⟨Ψ_a|Ŵ_a|Ψ_b⟩ and ⟨Ψ_a|Ŵ_b|Ψ_b⟩.
    let w_a_elem = cross_one_body(state_a.w, &pair_alpha, &pair_beta);
    let w_b_elem = cross_one_body(state_b.w, &pair_alpha, &pair_beta);

    // Each state's own constraint population N_X = ⟨Ψ_X|Ŵ_X|Ψ_X⟩.
    let n_a = own_population(state_a, &ca_occ_a, &ca_occ_b);
    let n_b = own_population(state_b, &cb_occ_a, &cb_occ_b);

    // Symmetric Wu–VV raw element with F_X = E_X + λ_X N_X (module docs):
    //   H_raw = ½[(F_b S_ab − λ_b⟨a|W_b|b⟩) + (F_a S_ab − λ_a⟨a|W_a|b⟩)].
    let e_a = state_a.energy;
    let e_b = state_b.energy;
    let f_a = e_a + state_a.lambda * n_a;
    let f_b = e_b + state_b.lambda * n_b;
    let h_raw =
        0.5 * ((f_b * s_ab - state_b.lambda * w_b_elem) + (f_a * s_ab - state_a.lambda * w_a_elem));

    // Symmetric orthogonalization. Guard the degenerate denominator.
    let denom = 1.0 - s_ab * s_ab;
    let h_ab = if denom.abs() < 1e-10 {
        // |S_ab| → 1: the two states are (nearly) identical; the coupling is
        // undefined / the off-diagonal collapses to the diagonal. Report E_a.
        e_a
    } else {
        (h_raw - 0.5 * (e_a + e_b) * s_ab) / denom
    };

    HabResult {
        h_ab,
        s_ab,
        e_a,
        e_b,
    }
}
