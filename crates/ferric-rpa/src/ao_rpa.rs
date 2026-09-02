//! AO-basis (imaginary-time) formulation of dRPA correlation.
//!
//! Reference: Kaltak, Klimeš, Kresse, J. Chem. Theory Comput. 10, 2498 (2014).
//!
//! # Overview
//!
//! Standard MO-basis dRPA has cost O(N⁴) per ω-point because the
//! cosine-modulated Laplace expansion couples ω to per-orbital energies
//! (see commit 108ee8f and task #32 for the resulting accuracy issues).
//!
//! Kaltak-Kresse moves the cosine out of the inner loop by going through
//! the imaginary-time / Laplace conjugate of imaginary frequency:
//!
//!   χ₀(iω, μν, λσ) = -∫_0^∞ dτ cos(ωτ) [G^o_{λμ}(τ) G^v_{νσ}(τ)
//!                                       + G^o_{νσ}(τ) G^v_{λμ}(τ)]
//!
//! where the imaginary-time occupied/virtual Green's functions
//!
//!   G^o_{μν}(τ) = -Σ_i C_{μi} C_{νi} exp(-(ε_i - μ_F)·τ)   τ > 0
//!   G^v_{μν}(τ) =  Σ_a C_{μa} C_{νa} exp(-(ε_a - μ_F)·τ)
//!
//! factor into MO sums independent of ω. With AO sparsity (atom-pair
//! cutoffs on C_μi, C_νa via local orbitals), the (μν,λσ) contraction
//! drops to O(N²) per (τ,ω) pair, and the τ-quadrature size is O(log N)
//! → total cost O(N³ log²N) in the best case.
//!
//! # First-land scope
//!
//! This module implements the **MO-basis** version of the imaginary-time
//! formulation — same scaling as Dense RPA, but with the τ↔ω separation
//! that's the prerequisite for the AO-sparse extension. Validates the
//! τ-grid + cosine-Fourier integration against the existing Dense path.
//!
//! AO sparsity (the real scaling win) is the C9 follow-up after this
//! module proves the conceptual machinery works.

use ferric_core::FerricError;
use ferric_quadrature::LaplaceQuadrature;
use ndarray::{Array2, Array3};
use crate::quadrature::MinimaxJointQuadrature;

/// Build the imaginary-time τ-quadrature for the energy-gap range.
///
/// Uses the minimax-Laplace nodes/weights for `1/x` on `[ymin, ymax]`,
/// which are exactly the right nodes for the Laplace transform of the
/// occupied/virtual exponentials. ymin = smallest e_ia (HOMO-LUMO gap),
/// ymax = largest e_ia.
pub fn build_tau_quadrature(
    eps_occ: &[f64],
    eps_vir: &[f64],
    n_quad: usize,
) -> Result<LaplaceQuadrature, FerricError> {
    let eps_homo = eps_occ.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_lumo = eps_vir.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_min = eps_occ.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps_vir.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = eps_lumo - eps_homo;
    let ymax = eps_max - eps_min;
    LaplaceQuadrature::new(n_quad, ymin, ymax)
}

/// MO-basis χ₀(iω) matrix in the RI-aux basis via imaginary-time formulation.
///
/// Returns Π(iω) = -χ₀(iω) of shape (naux, naux), with prefactor 4 baked
/// in for closed-shell. The dielectric is then ε̃ = I + Π.
///
/// Math: in MO basis the (i,a) decomposition gives
///
///   Π^P^Q = 4 Σ_{ia} B^P_ia · e_ia/(ω²+e_ia²) · B^Q_ia
///
/// Going through τ:
///
///   e_ia/(ω²+e_ia²) = ∫_0^∞ dτ cos(ωτ) exp(-e_ia·τ)
///                  ≈ Σ_l w_l cos(ω t_l) exp(-e_ia t_l)
///
/// The Laplace nodes (t_l, w_l) are tuned for 1/x on [ymin, ymax].
/// At a given ω, the per-spin contribution becomes
///
///   Π(ω) = 4 Σ_l w_l cos(ωt_l) X^l (X^l)^T,
///   where X^l_{P,ia} = B^P_ia · exp(-e_ia t_l / 2)
///
/// **This is bit-for-bit the same formula as crate::laplace_chi0**, but
/// implemented in this module as the foundation for the AO-basis
/// extension. We expect identical results to dielectric_matrix_laplace
/// at the same (n_quad, ω, B_ov).
///
/// At ω·t_max ≫ 1 the cosine modulation is faster than the quadrature
/// can resolve; the [bounded-ω fallback fix from commit 108ee8f
/// applies if used inside run_pdep_rpa](crate::laplace_chi0).
pub fn pi_via_imag_time(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
) -> Array2<f64> {
    use ndarray::{Axis, Zip};

    let naux = b_ov.shape()[0];
    let nov = b_ov.shape()[1];
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    assert_eq!(nov, nocc * nvir);

    let mut e_ia = Vec::with_capacity(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia.push(eps_vir[a] - eps_occ[i]);
        }
    }

    let mut pi = Array2::<f64>::zeros((naux, naux));

    // Scratch for X^l: shape (naux, nov).
    let mut x_l = Array2::<f64>::zeros((naux, nov));
    for (&t_l, &w_l) in laplace.points.iter().zip(laplace.weights.iter()) {
        // X^l_{P,ia} = B^P_ia · exp(-e_ia · t_l / 2)
        x_l.assign(b_ov);
        let factor: Vec<f64> = e_ia.iter().map(|&e| (-0.5 * t_l * e).exp()).collect();
        let factor_arr = ndarray::Array1::from(factor);
        let factor_row = factor_arr.view().insert_axis(Axis(0));
        Zip::from(&mut x_l).and_broadcast(factor_row).for_each(|x, &f| *x *= f);

        // Π(ω) += 4 · w_l cos(ω t_l) · X^l (X^l)^T
        let coeff = 4.0 * w_l * (omega * t_l).cos();
        // pi += coeff · x_l · x_l.t()
        let xt = x_l.t();
        ndarray::linalg::general_mat_mul(coeff, &x_l.view(), &xt, 1.0, &mut pi);
    }

    pi
}

/// Full AO-basis dielectric ε̃(iω) = I + Π via imaginary-time MO route.
///
/// Same input/output contract as
/// [`crate::laplace_chi0::dielectric_matrix_laplace`], implemented through
/// the τ-grid. Used here for validation against the dense path.
pub fn dielectric_matrix_imag_time(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
) -> Result<Array2<f64>, FerricError> {
    let pi_naux = pi_via_imag_time(b_ov, eps_occ, eps_vir, omega, laplace);
    // Project into the trial-vector subspace: ε̃_proj = V^T (I + Π) V
    let m = v_mat.ncols();
    let i_naux = Array2::<f64>::eye(pi_naux.nrows());
    let mut eps_naux = i_naux;
    eps_naux += &pi_naux;
    let eps_proj = v_mat.t().dot(&eps_naux).dot(v_mat);
    let _ = m;
    Ok(eps_proj)
}

// ---------------------------------------------------------------------------
// AO-basis imaginary-time pseudo-densities and χ⁰_PQ(τ)
// ---------------------------------------------------------------------------
//
// Task #43 (AO-sparse extension). The functions below mirror the
// Kaltak-Kresse construction directly in the AO basis:
//
//   P_{μλ}(τ) = Σ_i  C_{μi}  exp(+ε_i τ) C_{λi}              (nbasis × nbasis)
//   Q_{νσ}(τ) = Σ_a  C_{νa}  exp(-ε_a τ) C_{σa}              (nbasis × nbasis)
//
//   χ⁰_{PQ}(τ) = -2 Σ_{μνλσ} (P|μν) P_{μλ}(τ) Q_{νσ}(τ) (λσ|Q)
//
// The contraction over (μ,ν,λ,σ) is reordered as four matmuls so no
// O(nbasis^4) intermediate is built. With nbasis = N, naux = M this gives
// O(N³ · M) per τ point — same cost as MO χ⁰ assembly but the inputs are
// AO tensors, ready for AO sparsity (task #43 follow-up).
//
// Closed-shell prefactor 2 comes from spin sum on the doubly-occupied
// reference; the sign convention matches the χ⁰ used in pi_via_imag_time.

/// Build the occupied AO-basis pseudo-density at imaginary time τ.
///
///   P_{μλ}(τ) = Σ_i  C_{μi}  exp(+ε_i τ) C_{λi}
///
/// Note: ε_occ are negative for bound occupied orbitals, so `exp(+ε_i τ)`
/// decays for τ > 0 — this is the "occupied propagator" weight.
///
/// Shapes:
/// * `c_occ`  — (nbasis, nocc)
/// * `eps_occ` — (nocc,)
/// * returns — (nbasis, nbasis)
pub fn pseudo_density_occ(c_occ: &Array2<f64>, eps_occ: &[f64], tau: f64) -> Array2<f64> {
    let (nbasis, nocc) = c_occ.dim();
    assert_eq!(nocc, eps_occ.len(), "c_occ ncols must match eps_occ length");
    // Scaled coefficients C̃_{μi} = C_{μi} · exp(+½ ε_i τ); P = C̃ · C̃^T splits
    // the exp factor across both columns, avoiding overflow at large τ when
    // ε_i is large-negative (occupied energies are negative).
    let mut c_scaled = c_occ.clone();
    for i in 0..nocc {
        let w = (0.5 * eps_occ[i] * tau).exp();
        for mu in 0..nbasis {
            c_scaled[(mu, i)] *= w;
        }
    }
    c_scaled.dot(&c_scaled.t())
}

/// Build the virtual AO-basis pseudo-density at imaginary time τ.
///
///   Q_{νσ}(τ) = Σ_a  C_{νa}  exp(-ε_a τ) C_{σa}
///
/// Shapes:
/// * `c_vir`  — (nbasis, nvir)
/// * `eps_vir` — (nvir,)
/// * returns — (nbasis, nbasis)
pub fn pseudo_density_vir(c_vir: &Array2<f64>, eps_vir: &[f64], tau: f64) -> Array2<f64> {
    let (nbasis, nvir) = c_vir.dim();
    assert_eq!(nvir, eps_vir.len(), "c_vir ncols must match eps_vir length");
    let mut c_scaled = c_vir.clone();
    for a in 0..nvir {
        let w = (-0.5 * eps_vir[a] * tau).exp();
        for nu in 0..nbasis {
            c_scaled[(nu, a)] *= w;
        }
    }
    c_scaled.dot(&c_scaled.t())
}

/// Apply `exp(s · F τ / 2)` to a coefficient block via eigendecomposition.
///
/// `F = U diag(λ) Uᵀ` ⇒ `exp(s F τ/2) = U diag(exp(s λ τ/2)) Uᵀ`, so
/// `C · exp(s F τ/2) = (C U) · diag(exp(s λ τ/2))` — the column-scaled
/// rotated coefficients. The pseudo-density is then `(scaled)(scaled)ᵀ`,
/// splitting the exp factor across both sides exactly as the scalar path does.
///
/// `f_mo` is the Fock matrix in the orbital basis spanning `c`'s columns
/// (norb × norb). When the orbitals are canonical, `f_mo = diag(ε)`, `U = I`,
/// `λ = ε`, and this reduces bit-for-bit to the scalar path. When the orbitals
/// are localized (non-canonical), `f_mo` is non-diagonal and this is the only
/// correct form — a per-orbital scalar `exp(ε_i τ)` would be wrong.
fn pseudo_density_fock(c: &Array2<f64>, f_mo: &Array2<f64>, tau: f64, sign: f64) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let (nbasis, norb) = c.dim();
    assert_eq!(f_mo.dim(), (norb, norb), "f_mo must be (norb, norb)");
    // F = U diag(λ) Uᵀ
    let (lambda, u) = f_mo
        .eigh(UPLO::Upper)
        .expect("Fock-block eigendecomposition failed");
    // Rotated coefficients C·U, then column-scale by exp(sign·½·λ·τ).
    let mut c_rot = c.dot(&u); // (nbasis, norb)
    for k in 0..norb {
        let w = (sign * 0.5 * lambda[k] * tau).exp();
        for mu in 0..nbasis {
            c_rot[(mu, k)] *= w;
        }
    }
    c_rot.dot(&c_rot.t())
}

/// Occupied AO-basis pseudo-density for NON-CANONICAL (e.g. localized) orbitals.
///
///   P_{μλ}(τ) = Σ_ij C_{μi} [exp(F τ)]_ij C_{λj}
///
/// where `f_occ` is the Fock matrix in the occupied orbital basis (nocc × nocc).
/// Reduces to [`pseudo_density_occ`] bit-for-bit when `f_occ = diag(ε_occ)`.
/// This is the form required to feed Boys/PM-localized occupied orbitals into the
/// AO-time χ⁰ — see `localization-first-rpa-blocker`.
pub fn pseudo_density_occ_fock(c_occ: &Array2<f64>, f_occ: &Array2<f64>, tau: f64) -> Array2<f64> {
    // Occupied propagator weight is exp(+ε τ) (ε_occ negative ⇒ decays).
    pseudo_density_fock(c_occ, f_occ, tau, 1.0)
}

/// Virtual AO-basis pseudo-density for NON-CANONICAL (e.g. localized) orbitals.
///
///   Q_{νσ}(τ) = Σ_ab C_{νa} [exp(-F τ)]_ab C_{σb}
///
/// `f_vir` is the Fock matrix in the virtual orbital basis (nvir × nvir).
/// Reduces to [`pseudo_density_vir`] bit-for-bit when `f_vir = diag(ε_vir)`.
pub fn pseudo_density_vir_fock(c_vir: &Array2<f64>, f_vir: &Array2<f64>, tau: f64) -> Array2<f64> {
    // Virtual propagator weight is exp(-ε τ) (ε_vir positive ⇒ decays).
    pseudo_density_fock(c_vir, f_vir, tau, -1.0)
}

/// VIRTUAL-FREE (projector-form) virtual pseudo-density.
///
///   Q(τ) = C_vir exp(-ε_vir τ) C_virᵀ
///        = exp(-τ S⁻¹F) · (S⁻¹ − C_occ C_occᵀ)
///
/// The virtual space is the S-orthogonal complement of the occupied space, so it
/// is represented by the projector `Π_v = S⁻¹ − C_occ C_occᵀ` (= C_vir C_virᵀ)
/// WITHOUT any explicit virtual orbitals. Propagating Π_v by the AO Fock yields the
/// imaginary-time virtual propagator — the time-domain analog of the Sternheimer
/// occupied-projector trick. This eliminates the virtual-localization problem: there
/// are no virtual orbitals to localize.
///
/// Implemented in the symmetric (Löwdin) basis: with `S = X Xᵀ` (`X = S^{1/2}`),
/// `F̄ = X⁻¹ F X⁻ᵀ` is symmetric and `C̄_occ = Xᵀ C_occ` is orthonormal, so the
/// whole thing reduces to the orthonormal case `Q̄(τ) = exp(−F̄τ)(I − C̄_occ C̄_occᵀ)`,
/// symmetrized, then transformed back: `Q = X⁻ᵀ Q̄ X⁻¹`.
///
/// Reduces (to machine precision) to [`pseudo_density_vir`] built from the explicit
/// virtual orbitals. See memory `localization-first-rpa-blocker`.
pub fn pseudo_density_vir_projector(
    f_ao: &Array2<f64>,
    s: &Array2<f64>,
    c_occ: &Array2<f64>,
    tau: f64,
) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let n = s.dim().0;
    // S = U d Uᵀ ⇒ X = S^{1/2} = U d^{1/2} Uᵀ, X⁻¹ = U d^{-1/2} Uᵀ.
    let (sd, su) = s.eigh(UPLO::Upper).expect("overlap eigendecomposition failed");
    let x = su.dot(&Array2::from_diag(&sd.mapv(|v| v.sqrt()))).dot(&su.t());
    let x_inv = su.dot(&Array2::from_diag(&sd.mapv(|v| 1.0 / v.sqrt()))).dot(&su.t());

    // Symmetric Fock in the orthonormal basis: F̄ = X⁻¹ F X⁻¹ (X symmetric).
    let f_bar = x_inv.dot(f_ao).dot(&x_inv);
    // Orthonormal occupied coeffs: C̄_occ = Xᵀ C_occ = X C_occ (X symmetric).
    let c_occ_bar = x.dot(c_occ);
    // Virtual projector in the orthonormal basis: Π̄_v = I − C̄_occ C̄_occᵀ.
    let pi_v = Array2::<f64>::eye(n) - c_occ_bar.dot(&c_occ_bar.t());

    // Q̄ = exp(−F̄_v τ) · Π̄_v, where F̄_v = Π̄_v F̄ Π̄_v is the Fock PROJECTED into
    // the virtual space. Projecting the exponent first is essential: a bare
    // exp(−F̄ τ) also exponentiates the OCCUPIED modes (ε < 0 ⇒ exp(+|ε|τ)), which
    // overflows at large τ for real core orbitals (ε ≈ −11 ⇒ exp(66·τ)). With
    // F̄_v, occupied modes have eigenvalue 0 ⇒ exp(0)=1 (harmless), and the virtual
    // space is F̄-invariant so the result is exact. The right-multiply by Π̄_v then
    // strips the residual occupied identity, leaving exactly C̄_vir exp(−ε_vir τ) C̄_virᵀ.
    let f_bar_v = pi_v.dot(&f_bar).dot(&pi_v);
    let (lambda, w) = f_bar_v.eigh(UPLO::Upper).expect("Fock-bar_v eigendecomposition failed");
    let e_mat = {
        let scaled = w.dot(&Array2::from_diag(&lambda.mapv(|l| (-l * tau).exp())));
        scaled.dot(&w.t())
    };
    let q_bar = e_mat.dot(&pi_v);

    // Back-transform to the AO basis: Q = X⁻ᵀ Q̄ X⁻¹ = X⁻¹ Q̄ X⁻¹ (X symmetric).
    x_inv.dot(&q_bar).dot(&x_inv)
}

/// Independent-particle χ⁰_{PQ}(τ) in the auxiliary basis at one τ.
///
///   χ⁰_{PQ}(τ) = -2 Σ_{μνλσ} (P|μν) P_{μλ}(τ) Q_{νσ}(τ) (λσ|Q)
///
/// Implemented as a four-step contraction (each step is a single GEMM
/// over rolled-up indices); no nbasis⁴ intermediate is materialized:
///
///   step 1: X^P_{λν} = Σ_μ  (P|μν) P_{μλ}                 — (naux·nbasis, nbasis)
///   step 2: Y^P_{λσ} = Σ_ν  X^P_{λν} Q_{νσ}               — (naux·nbasis, nbasis)
///   step 3: χ⁰_{PQ}  = Σ_{λσ} Y^P_{λσ} (λσ|Q)             — (naux, naux)
///
/// Cost: O(naux · nbasis³) per τ-point. With nbasis = N and naux ≈ 3N,
/// that's ~3N⁴ flops — the same as MO χ⁰. The win comes when AO sparsity
/// (task #43 follow-up) reduces the effective nbasis in the inner steps.
///
/// Shapes:
/// * `eri3` — (naux, nbasis, nbasis), the 3-index tensor (P|μν) from
///   [`ferric_integrals::threeindex::eri3_tensor`]
/// * `p_occ` — (nbasis, nbasis), occupied pseudo-density P_{μλ}(τ)
/// * `q_vir` — (nbasis, nbasis), virtual  pseudo-density Q_{νσ}(τ)
/// * returns — (naux, naux), χ⁰_{PQ}(τ)
pub fn chi0_ao_at_tau(
    eri3: &Array3<f64>,
    p_occ: &Array2<f64>,
    q_vir: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let (naux, nbasis_1, nbasis_2) = eri3.dim();
    if nbasis_1 != nbasis_2 {
        return Err(FerricError::General(
            "chi0_ao_at_tau: eri3 must have square (μ,ν) dims".into(),
        ));
    }
    let nbasis = nbasis_1;
    if p_occ.dim() != (nbasis, nbasis) || q_vir.dim() != (nbasis, nbasis) {
        return Err(FerricError::General(
            "chi0_ao_at_tau: pseudo-density shapes must match eri3 AO dims".into(),
        ));
    }

    // Flatten eri3 into 2D for matmul. eri3 stored row-major (naux, μ, ν);
    // view as (naux · μ, ν) for the ν contraction in step 1.
    // Step 1: contract μ via   X[P,λ,ν] = Σ_μ eri3[P,μ,ν] · p_occ[μ,λ]
    //                       = Σ_μ eri3_view[P·μ, ν] indexed against p_occ[μ,λ]
    // For maximum BLAS leverage we use a different reshape:
    //   reshape eri3 → (naux, nbasis * nbasis) — view as (naux, μν)
    // is wrong here because the μ index isn't contiguous across (P, μ, ν).
    // Instead contract via two matmul passes per aux row, or use einsum-style
    // tensordot. The cleanest form is to reshape eri3 → (naux · nbasis, nbasis)
    // with leading index (P, μ); then p_occ is (μ, λ); but the contraction
    // is over μ which is the outer of (P, μ), not the inner of eri3_2d.
    //
    // Easiest correct path: build a 3D intermediate Y[P, λ, σ] by looping
    // over P (small loop, naux ~ 3·nbasis) and doing two GEMMs per P:
    //   E_P[μ, ν] = eri3[P, ·, ·]              (nbasis, nbasis)
    //   M_P[λ, ν] = p_occ^T[λ, μ] · E_P[μ, ν]  (nbasis, nbasis)
    //   N_P[λ, σ] = M_P[λ, ν]   · q_vir[ν, σ]  (nbasis, nbasis)
    // Then
    //   χ⁰[P, Q] = -2 Σ_{λ,σ} N_P[λ, σ] · eri3[Q, λ, σ]
    //            = -2 · (N_flat) · (eri3_flat)^T,   N_flat[P, λσ], eri3_flat[Q, λσ]
    //
    // Memory: one (nbasis, nbasis) intermediate per P, plus a final
    // (naux, nbasis²) buffer for N_flat. The final dot is one big GEMM
    // (naux, nbasis²) × (nbasis², naux) — well-vectorized.

    let nbasis_sq = nbasis * nbasis;
    let mut n_flat = Array2::<f64>::zeros((naux, nbasis_sq));

    // p_occ_t[λ, μ] view, so the first GEMM is p_occ_t · E_P
    let p_occ_t = p_occ.t();  // (nbasis, nbasis)

    for p in 0..naux {
        // E_P[μ, ν] = eri3[p, :, :]
        let e_p = eri3.slice(ndarray::s![p, .., ..]);
        // M_P[λ, ν] = p_occ^T[λ, μ] · E_P[μ, ν]
        let m_p = p_occ_t.dot(&e_p);
        // N_P[λ, σ] = M_P[λ, ν] · Q[ν, σ]
        let n_p = m_p.dot(q_vir);
        // Flatten N_P (nbasis, nbasis) → row p of n_flat (nbasis², )
        let mut row = n_flat.row_mut(p);
        for la in 0..nbasis {
            for sg in 0..nbasis {
                row[la * nbasis + sg] = n_p[(la, sg)];
            }
        }
    }

    // eri3_flat[Q, λσ] view: same memory, reshape (naux, nbasis, nbasis) → (naux, nbasis²).
    // eri3 is owned and row-major; this reshape is a zero-copy view.
    let eri3_flat = eri3.view().into_shape_with_order((naux, nbasis_sq)).map_err(|e| {
        FerricError::General(format!("chi0_ao_at_tau: eri3 reshape failed: {e}"))
    })?;

    // χ⁰[P, Q] = -2 · n_flat[P, λσ] · eri3_flat[Q, λσ]^T
    //         = -2 · n_flat · eri3_flat^T
    let mut chi0 = n_flat.dot(&eri3_flat.t());
    chi0 *= -2.0;
    Ok(chi0)
}

/// AO-basis χ⁰(τ) on the full τ-quadrature grid.
///
/// Returns a stack of `(naux, naux)` matrices for each τ-point.
///
/// Shape: `(n_tau, naux, naux)`.
pub fn chi0_ao_full_time(
    eri3: &Array3<f64>,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    laplace: &LaplaceQuadrature,
) -> Result<Array3<f64>, FerricError> {
    let (naux, _, _) = eri3.dim();
    let n_tau = laplace.points.len();
    let mut out = Array3::<f64>::zeros((n_tau, naux, naux));
    for (k, &tau_k) in laplace.points.iter().enumerate() {
        let p_occ = pseudo_density_occ(c_occ, eps_occ, tau_k);
        let q_vir = pseudo_density_vir(c_vir, eps_vir, tau_k);
        let chi0 = chi0_ao_at_tau(eri3, &p_occ, &q_vir)?;
        out.slice_mut(ndarray::s![k, .., ..]).assign(&chi0);
    }
    Ok(out)
}

/// Dress the 3-index ERI with V^{-1/2}: `B̃[P,μ,ν] = Σ_Q V^{-1/2}[P,Q] · eri3[Q,μ,ν]`.
///
/// This is the AO analogue of the standard MO `b_ov = V^{-1/2} · (P|ia)`
/// dressing: by absorbing the metric inverse into one factor of the
/// 3-index tensor, the resulting χ⁰_PQ(τ) is already in the V^{-1/2}-dressed
/// basis where ε̃ = I + Π reads directly without further metric work, and Π
/// is positive semidefinite for closed shell.
///
/// Shapes: `eri3` (naux, nbasis, nbasis), `v_inv_sqrt` (naux, naux),
/// returns (naux, nbasis, nbasis).
pub fn dress_eri3_with_metric(
    eri3: &Array3<f64>,
    v_inv_sqrt: &Array2<f64>,
) -> Array3<f64> {
    let (naux, nbasis, _) = eri3.dim();
    // Reshape eri3 (naux, nbasis²) so the dressing is a single naux²·nbasis² GEMM.
    let eri3_flat = eri3.view().into_shape_with_order((naux, nbasis * nbasis)).unwrap();
    let dressed_flat = v_inv_sqrt.dot(&eri3_flat);
    dressed_flat.into_shape_with_order((naux, nbasis, nbasis)).unwrap()
}

/// Cosine-Fourier transform Π_PQ(iω) = Σ_l (-2·w_l·cos(ω·τ_l)) · χ⁰_PQ(τ_l).
///
/// Derivation (closed shell, V^{-1/2}-dressed basis):
///
///   Π(iω) = 2 · ∫₀^∞ dτ cos(ω τ) · [-χ⁰(τ)]
///         ≈ Σ_l (2 · w_l · cos(ω τ_l)) · [-χ⁰(τ_l)]
///
/// The factor 2 comes from extending the τ integral to the full real axis
/// (cosine-Fourier conjugate of the Laplace transform), and `χ⁰_PQ(τ)` as
/// implemented in [`chi0_ao_at_tau`] carries its own factor -2 (closed-shell
/// spin sum). Net coefficient: -2·w_l·cos(ω·τ_l). The result Π(iω) is
/// positive semidefinite for closed shell.
///
/// Shapes: `chi0_tau_stack` (n_tau, naux, naux), returns (naux, naux).
pub fn pi_ao_at_omega(
    chi0_tau_stack: &Array3<f64>,
    laplace: &LaplaceQuadrature,
    omega: f64,
) -> Array2<f64> {
    let (n_tau, naux, _) = chi0_tau_stack.dim();
    assert_eq!(n_tau, laplace.points.len());
    let mut pi = Array2::<f64>::zeros((naux, naux));
    for l in 0..n_tau {
        let coeff = -2.0 * laplace.weights[l] * (omega * laplace.points[l]).cos();
        let slab = chi0_tau_stack.slice(ndarray::s![l, .., ..]);
        // pi += coeff · slab
        pi.scaled_add(coeff, &slab);
    }
    pi
}

/// Fourier transform using pre-optimized Minimax Joint weights (Kaltak-Kresse).
///
/// Π_PQ(iω_k) = Σ_l (-2·W_lk) · χ⁰_PQ(τ_l)
///
/// This avoids evaluating highly oscillatory cos(ωτ) functions,
/// guaranteeing accuracy without aliasing.
pub fn pi_ao_at_omega_minimax(
    chi0_tau_stack: &Array3<f64>,
    joint: &MinimaxJointQuadrature,
    k_omega: usize,
) -> Array2<f64> {
    let (n_tau, naux, _) = chi0_tau_stack.dim();
    assert_eq!(n_tau, joint.tau_points.len());
    
    let mut pi = Array2::<f64>::zeros((naux, naux));
    let offset = k_omega * n_tau;
    let w_k_row = &joint.w_transform[offset..offset + n_tau];
    
    for l in 0..n_tau {
        let coeff = -2.0 * w_k_row[l];
        let slab = chi0_tau_stack.slice(ndarray::s![l, .., ..]);
        pi.scaled_add(coeff, &slab);
    }
    pi
}

/// Build Π(iω) in the V^{-1/2}-dressed aux basis directly from the MO B-tensor.
///
/// Used as a high-ω fallback when the cosine-Fourier τ-quadrature becomes
/// inaccurate. Same Π that `crate::diagnostics::ri_drpa_eigenvalues` uses
/// internally, but exposed for the AO-RPA driver. Cost: O(naux² · nov).
pub fn pi_mo_dressed(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    use crate::sternheimer::build_scale_factors;
    use ndarray::{Axis, Zip};
    let scale = build_scale_factors(eps_occ, eps_vir, omega);
    let mut bs = b_ov.to_owned();
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut bs).and_broadcast(scale_row).for_each(|x, &s| *x *= s);
    bs.dot(&bs.t())
}

/// AO-basis RI-dRPA correlation energy via the Kaltak-Kresse imaginary-time path.
///
/// Pipeline:
/// 1. dress the 3-index ERI with `V^{-1/2}` once: `B̃[P,μ,ν]`
/// 2. build χ⁰(τ_k) AO-basis stack on the minimax-Laplace τ-grid (O(naux·N³) per τ)
/// 3. cosine-Fourier to each ω_k: `Π(iω_k) = -2·Σ_l w_l cos(ω_k τ_l) χ⁰(τ_l)`
/// 4. trace-log: `E_c = (1/2π) Σ_k w_k Σ_α [ln λ_α + (1 − λ_α)]`, λ = eigvals(I + Π)
///
/// Cost summary (no AO sparsity yet):
///   step 1: one O(naux²·N²) GEMM
///   step 2: n_tau passes, each O(naux·N³) — dominant for small/medium N
///   step 3: n_tau · n_ω naux² adds
///   step 4: n_ω diagonalizations of naux × naux + trace-log
///
/// AO sparsity will reduce step 2 to O(naux · N²) by truncating the
/// pseudo-densities — that's the task #43 follow-up.
///
/// Returns `(e_c, n_tau, n_omega)` for benchmarking.
/// `b_ov_fallback`: if `Some`, used to build Π(iω) directly from the MO B-tensor at
/// any ω where the cosine-Fourier τ-quadrature is inaccurate (ω·t_max > π/2,
/// matching the bounded-ω guard in `crate::laplace_chi0`). At those high ω the
/// integrand `e_ia/(ω²+e²) ~ 1/ω²` is small but non-negligible for the trace-log,
/// and the τ-quadrature aliases. If `None`, we use the AO τ path at all ω
/// (will be inaccurate for high-ω quadrature points — useful for diagnostics).
#[allow(clippy::too_many_arguments)]
pub fn ao_rpa_correlation_energy(
    eri3: &Array3<f64>,
    v_inv_sqrt: &Array2<f64>,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    quad_weights: &[f64],
    n_tau: usize,
    b_ov_fallback: Option<&Array2<f64>>,
) -> Result<(f64, usize, usize, usize), FerricError> {
    use ferric_core::memory::plan::{Lifetime, MemoryPlan};
    use ferric_integrals::blas_threads::with_blas_threads;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    // Fail-fast size guard, mirroring `ao_rpa_correlation_energy_minimax`'s.
    // This sibling allocates the SAME shapes — the dense input `eri3`
    // co-resident with `dress_eri3_with_metric`'s copy, plus the `n_tau`-deep
    // χ⁰(τ) stack — but had no gate at all, so the identical footprint was
    // governed on one entry point and ungoverned on the other.
    //
    // It also charges a term the minimax gate misses: the per-frequency
    // closure below builds `eps_mat` (an `naux x naux` from `pi`) and hands it
    // to `eigh`, whose eigenvector output is a second `naux²` — all inside a
    // `par_iter`, so both scale with the rayon worker count.
    {
        let (naux_g, nbf1, nbf2) = eri3.dim();
        let n_workers = rayon::current_num_threads().max(1);
        let mut plan = MemoryPlan::with_budget_bytes(
            ferric_core::memory::resolve_budget_bytes(None),
            format!(
                "AO-RPA (naux={naux_g}, nbf={nbf1}, n_tau={n_tau}, n_omega={}, \
                 n_workers={n_workers})",
                quad_freqs.len()
            ),
        );
        let eri3_elems = naux_g.saturating_mul(nbf1).saturating_mul(nbf2);
        plan.reserve("eri3 (naux,nbf,nbf) input", eri3_elems, Lifetime::Resident);
        plan.reserve("eri3 dressed copy", eri3_elems, Lifetime::Resident);
        plan.reserve(
            "chi0(tau) stack (n_tau,naux,naux)",
            n_tau.saturating_mul(naux_g).saturating_mul(naux_g),
            Lifetime::Resident,
        );
        plan.reserve_per_worker(
            "per-omega eps_mat + eigh evecs",
            naux_g.saturating_mul(naux_g).saturating_mul(2),
            n_workers,
        );
        plan.check()?;
    }

    // Step 1: dress ERI.
    let eri3_dressed = dress_eri3_with_metric(eri3, v_inv_sqrt);

    // Step 2: AO-basis χ⁰(τ) stack on minimax grid.
    let laplace = build_tau_quadrature(eps_occ, eps_vir, n_tau)?;
    let chi0_stack = chi0_ao_full_time(
        &eri3_dressed, c_occ, c_vir, eps_occ, eps_vir, &laplace,
    )?;
    let t_max = laplace.points.iter().cloned().fold(0.0_f64, f64::max);
    let omega_cutoff = std::f64::consts::FRAC_PI_2 / t_max;

    // Steps 3+4: per ω_k, build Π(iω_k), add I, diagonalize, accumulate trace-log.
    // High ω (ω·t_max > π/2) → use MO Π fallback if provided.
    let naux = eri3.dim().0;
    let n_fallback = std::sync::atomic::AtomicUsize::new(0);
    // Pin BLAS to 1 inside the rayon region: `eigh` below runs per-frequency
    // under rayon workers — nested OpenBLAS threads oversubscribe and can
    // overflow the 2 MB rayon worker stack (openblas-rayon-dgetrf-crash).
    let contribs: Result<Vec<f64>, FerricError> = with_blas_threads(1, || {
        quad_freqs
            .par_iter()
            .zip(quad_weights.par_iter())
            .map(|(&omega, &wk)| {
                let pi = if omega > omega_cutoff {
                    match b_ov_fallback {
                        Some(b) => {
                            n_fallback.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            pi_mo_dressed(b, eps_occ, eps_vir, omega)
                        }
                        None => pi_ao_at_omega(&chi0_stack, &laplace, omega),
                    }
                } else {
                    pi_ao_at_omega(&chi0_stack, &laplace, omega)
                };
                let mut eps_mat = pi;
                for p in 0..naux { eps_mat[(p, p)] += 1.0; }
                let (evals, _) = eps_mat.eigh(UPLO::Upper)
                    .map_err(|e| FerricError::General(format!("AO-RPA eigh: {e}")))?;
                let contrib: f64 = evals.iter().map(|&lam| lam.ln() + (1.0 - lam)).sum();
                Ok(wk * contrib)
            })
            .collect()
    });

    let e_c: f64 = contribs?.iter().sum::<f64>() / (2.0 * std::f64::consts::PI);
    let n_fb = n_fallback.load(std::sync::atomic::Ordering::Relaxed);
    Ok((e_c, n_tau, quad_freqs.len(), n_fb))
}

/// True AO-basis RI-dRPA using Joint Minimax weights.
/// 
/// This eliminates the dense MO-basis fallback entirely because
/// the pre-computed joint weights guarantee microhartree precision
/// for all relevant frequencies without oscillatory aliasing.
#[allow(clippy::too_many_arguments)]
pub fn ao_rpa_correlation_energy_minimax(
    eri3: &Array3<f64>,
    v_inv_sqrt: &Array2<f64>,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    joint_grids: &MinimaxJointQuadrature,
) -> Result<(f64, usize, usize), FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    // Fail-fast size guard: the dense input eri3 (naux·nbf², passed in) is held
    // co-resident with the dressed copy from dress_eri3_with_metric (:663 → naux·nbf²)
    // and the χ⁰(τ) stack (n_tau·naux², chi0_ao_full_time :672). Peak ≈ 2× naux·nbf²
    // plus the τ-stack. No config budget on this reference path. Keep next to the
    // dress + full-time build.
    let (naux, nbf1, nbf2) = eri3.dim();
    let n_tau = joint_grids.tau_points.len();
    let eri3_bytes = naux.saturating_mul(nbf1).saturating_mul(nbf2).saturating_mul(8);
    let chi0_stack_bytes = n_tau.saturating_mul(naux).saturating_mul(naux).saturating_mul(8);
    let peak = eri3_bytes
        .saturating_mul(2) // input eri3 + dressed copy
        .saturating_add(chi0_stack_bytes);
    ferric_core::memory::check_alloc(
        &format!("AO-RPA minimax (naux={naux}, nbf={nbf1}, n_tau={n_tau}; dense eri3 + χ⁰ stack)"),
        peak,
        ferric_core::memory::resolve_budget_bytes(None),
    )?;

    let eri3_dressed = dress_eri3_with_metric(eri3, v_inv_sqrt);

    // Build the τ-quadrature mock from the joint arrays
    let laplace = LaplaceQuadrature {
        points: joint_grids.tau_points.clone(),
        weights: vec![0.0; joint_grids.tau_points.len()], // Not used in chi0_ao
        n_quad: joint_grids.tau_points.len(),
    };
    
    let chi0_stack = chi0_ao_full_time(
        &eri3_dressed, c_occ, c_vir, eps_occ, eps_vir, &laplace,
    )?;

    let naux = eri3.dim().0;

    // Evaluate trace-log for each frequency using the minimax mapping.
    // Pin BLAS to 1 inside the rayon region: `eigh` below runs per-frequency
    // under rayon workers — nested OpenBLAS threads oversubscribe and can
    // overflow the 2 MB rayon worker stack (openblas-rayon-dgetrf-crash).
    let contribs: Result<Vec<f64>, FerricError> = with_blas_threads(1, || {
        joint_grids.omega_points
            .par_iter()
            .zip(joint_grids.omega_weights.par_iter())
            .enumerate()
            .map(|(k, (&_omega, &wk))| {
                let pi = pi_ao_at_omega_minimax(&chi0_stack, joint_grids, k);

                let mut eps_mat = pi;
                for p in 0..naux { eps_mat[(p, p)] += 1.0; }

                let (evals, _) = eps_mat.eigh(UPLO::Upper)
                    .map_err(|e| FerricError::General(format!("AO-RPA minimax eigh: {e}")))?;

                let contrib: f64 = evals.iter().map(|&lam| lam.ln() + (1.0 - lam)).sum();
                Ok(wk * contrib)
            })
            .collect()
    });

    let e_c: f64 = contribs?.iter().sum::<f64>() / (2.0 * std::f64::consts::PI);
    Ok((e_c, joint_grids.tau_points.len(), joint_grids.omega_points.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sternheimer::dielectric_matrix;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize env-mutating tests
    // via the crate-wide lock (a private per-module lock cannot stop a
    // cross-module race — see TEST_BUDGET_ENV_LOCK's doc in lib.rs).
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;

    #[test]
    fn ao_rpa_minimax_fails_fast_under_tiny_env_budget() {
        // M2 size guard: a tiny env budget must ERROR before dress_eri3_with_metric
        // duplicates the dense eri3. Synthetic inputs — the guard fires first, so
        // no numerics are exercised.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (nbasis, nocc, nvir, naux) = (4usize, 1usize, 3usize, 5usize);
        let eri3 = Array3::<f64>::zeros((naux, nbasis, nbasis));
        let v_inv_sqrt = Array2::<f64>::eye(naux);
        let c_occ = Array2::<f64>::zeros((nbasis, nocc));
        let c_vir = Array2::<f64>::zeros((nbasis, nvir));
        let eps_occ = vec![-0.5];
        let eps_vir = vec![0.2, 0.6, 1.1];
        let joint = MinimaxJointQuadrature {
            tau_points: vec![0.1, 0.5],
            omega_points: vec![0.2],
            omega_weights: vec![1.0],
            w_transform: vec![0.0, 0.0],
        };
        // 1e-7 GiB ≈ 107 bytes; even this tiny synthetic tensor set exceeds it.
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "0.0000001");
        let res = ao_rpa_correlation_energy_minimax(
            &eri3, &v_inv_sqrt, &c_occ, &c_vir, &eps_occ, &eps_vir, &joint,
        );
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
        let err = res.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("AO-RPA") && msg.contains("budget is"), "unexpected: {msg}");
    }

    /// Build B^P_ia = Σ_{μν} C_{μi} (P|μν) C_{νa} for synthetic test below.
    fn build_b_ov(eri3: &Array3<f64>, c_occ: &Array2<f64>, c_vir: &Array2<f64>) -> Array2<f64> {
        let (naux, _, _) = eri3.dim();
        let nocc = c_occ.shape()[1];
        let nvir = c_vir.shape()[1];
        let mut b = Array2::<f64>::zeros((naux, nocc * nvir));
        for p in 0..naux {
            let e_p = eri3.slice(ndarray::s![p, .., ..]);
            // T[i, ν] = C^T[i, μ] · E_P[μ, ν]
            let t = c_occ.t().dot(&e_p);
            // B[P, ia] = T[i, ν] · C_vir[ν, a]
            let b_pia = t.dot(c_vir);  // (nocc, nvir)
            for i in 0..nocc {
                for a in 0..nvir {
                    b[(p, i * nvir + a)] = b_pia[(i, a)];
                }
            }
        }
        b
    }

    #[test]
    fn ao_chi0_matches_mo_chi0_synthetic() {
        // Tiny closed-shell system: nbasis=6, nocc=2, nvir=4.
        let nbasis = 6;
        let nocc = 2;
        let nvir = 4;
        let naux = 7;

        // Random-ish AO MO coefficients; orthonormality not required for the
        // identity (the AO-basis χ⁰ formula doesn't assume orthonormal MOs —
        // it's just a sum over occ/vir spaces).
        let c_occ = Array2::from_shape_fn((nbasis, nocc), |(mu, i)| {
            0.2 + 0.05 * mu as f64 + 0.07 * i as f64
                - 0.01 * (mu as f64 * i as f64)
        });
        let c_vir = Array2::from_shape_fn((nbasis, nvir), |(mu, a)| {
            0.1 - 0.04 * mu as f64 + 0.06 * a as f64
                + 0.02 * (mu as f64 - a as f64)
        });
        let eps_occ = vec![-0.5_f64, -0.3];
        let eps_vir = vec![0.2_f64, 0.6, 1.1, 1.8];

        // Random symmetric 3-index ERI tensor (μ↔ν symmetric).
        let mut eri3 = Array3::<f64>::zeros((naux, nbasis, nbasis));
        for p in 0..naux {
            for mu in 0..nbasis {
                for nu in 0..=mu {
                    let v = 0.05 + 0.02 * p as f64 - 0.01 * (mu + nu) as f64
                        + 0.003 * (mu as f64 * nu as f64);
                    eri3[(p, mu, nu)] = v;
                    eri3[(p, nu, mu)] = v;
                }
            }
        }

        let b_ov = build_b_ov(&eri3, &c_occ, &c_vir);

        // Pick a few τ-points to compare. n_quad=8 is not tabulated (only
        // {3,5,7} are); round explicitly to preserve the original test's
        // 7-point coverage instead of relying on the old silent fallback.
        let n_quad = ferric_quadrature::minimax::nearest_supported_n_quad(8);
        let laplace = build_tau_quadrature(&eps_occ, &eps_vir, n_quad).unwrap();
        for &tau in &[0.05_f64, 0.2, 0.5, 1.0] {
            // AO-basis route
            let p_occ = pseudo_density_occ(&c_occ, &eps_occ, tau);
            let q_vir = pseudo_density_vir(&c_vir, &eps_vir, tau);
            let chi0_ao = chi0_ao_at_tau(&eri3, &p_occ, &q_vir).unwrap();

            // MO-basis route:
            // χ⁰_{PQ}(τ) = -2 Σ_{ia} B^P_ia exp(-e_ia τ) B^Q_ia
            // (closed-shell prefactor 2 matches the AO formula's factor).
            let nov = nocc * nvir;
            let mut e_ia = Vec::with_capacity(nov);
            for i in 0..nocc {
                for a in 0..nvir {
                    e_ia.push(eps_vir[a] - eps_occ[i]);
                }
            }
            let mut b_scaled = b_ov.clone();
            for (col, &e) in e_ia.iter().enumerate() {
                let w = (-0.5 * e * tau).exp();
                for p in 0..naux {
                    b_scaled[(p, col)] *= w;
                }
            }
            let mut chi0_mo = b_scaled.dot(&b_scaled.t());
            chi0_mo *= -2.0;

            let max_err = chi0_ao.iter().zip(chi0_mo.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            let max_ref = chi0_mo.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            eprintln!("τ={tau:.2}: AO-vs-MO χ⁰ max_err={max_err:.3e} (|χ⁰|max={max_ref:.3e})");
            assert!(max_err < 1e-10,
                "AO χ⁰(τ={tau}) should match MO route: max_err={max_err:.3e}");
        }

        // Full-time stack sanity: shape and first slice match a direct call.
        let stack = chi0_ao_full_time(&eri3, &c_occ, &c_vir, &eps_occ, &eps_vir, &laplace).unwrap();
        assert_eq!(stack.dim(), (laplace.points.len(), naux, naux));
        let tau0 = laplace.points[0];
        let p0 = pseudo_density_occ(&c_occ, &eps_occ, tau0);
        let q0 = pseudo_density_vir(&c_vir, &eps_vir, tau0);
        let chi0_direct = chi0_ao_at_tau(&eri3, &p0, &q0).unwrap();
        let max_err = stack.slice(ndarray::s![0, .., ..]).iter()
            .zip(chi0_direct.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(max_err < 1e-14, "full-time stack slice should match direct call");
    }

    #[test]
    fn imag_time_matches_dense_synthetic() {
        // Tiny synthetic: B random, e_ia spaced.
        let naux = 5;
        let nocc = 2;
        let nvir = 3;
        let nov = nocc * nvir;
        let b_ov = Array2::from_shape_fn((naux, nov), |(p, ia)| {
            0.1 + 0.03 * p as f64 - 0.02 * ia as f64
        });
        let eps_occ = vec![-0.5_f64, -0.3];
        let eps_vir = vec![0.2_f64, 0.6, 1.4];
        // Identity V — Π in aux basis directly.
        let v_mat = Array2::<f64>::eye(naux);
        let omega = 0.5;

        // n_quad=8 is not tabulated (only {3,5,7} are); round explicitly to
        // preserve the original test's 7-point coverage instead of relying
        // on the old silent fallback.
        let n_quad = ferric_quadrature::minimax::nearest_supported_n_quad(8);
        let laplace = build_tau_quadrature(&eps_occ, &eps_vir, n_quad).unwrap();
        let eps_imag = dielectric_matrix_imag_time(&v_mat, &b_ov, &eps_occ, &eps_vir, omega, &laplace).unwrap();
        let eps_dense = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, omega);

        let max_err = eps_imag.iter().zip(eps_dense.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        eprintln!("imag-time vs dense max elementwise error: {max_err:.3e}");
        assert!(max_err < 5e-3,
            "imag-time τ-route should match dense at low ω: max_err={max_err:.3e}");
    }

    /// GATE: the Fock-matrix pseudo-density must reproduce the canonical scalar
    /// path BIT-FOR-BIT when the Fock matrix is diagonal (i.e. the orbitals are
    /// already canonical eigenstates). Diagonal F = diag(ε) ⇒ exp(Fτ) = diag(exp(ε τ)).
    #[test]
    fn pseudo_density_occ_fock_matches_canonical_when_diagonal() {
        let nbasis = 6;
        let nocc = 3;
        let c_occ = Array2::from_shape_fn((nbasis, nocc), |(mu, i)| {
            0.2 + 0.05 * mu as f64 - 0.03 * i as f64 + 0.01 * (mu * i) as f64
        });
        let eps_occ = vec![-0.9_f64, -0.55, -0.3];
        let f_diag = Array2::from_diag(&ndarray::arr1(&eps_occ));

        for &tau in &[0.05_f64, 0.4, 1.3, 5.0] {
            let p_scalar = pseudo_density_occ(&c_occ, &eps_occ, tau);
            let p_fock = pseudo_density_occ_fock(&c_occ, &f_diag, tau);
            let max_err = p_scalar.iter().zip(p_fock.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-12,
                "occ Fock path must match canonical scalar path at τ={tau}: max_err={max_err:.3e}");
        }
    }

    #[test]
    fn pseudo_density_vir_fock_matches_canonical_when_diagonal() {
        let nbasis = 6;
        let nvir = 4;
        let c_vir = Array2::from_shape_fn((nbasis, nvir), |(mu, a)| {
            0.1 - 0.04 * mu as f64 + 0.06 * a as f64 + 0.02 * (mu as f64 - a as f64)
        });
        let eps_vir = vec![0.2_f64, 0.6, 1.1, 1.8];
        let f_diag = Array2::from_diag(&ndarray::arr1(&eps_vir));

        for &tau in &[0.05_f64, 0.4, 1.3, 5.0] {
            let q_scalar = pseudo_density_vir(&c_vir, &eps_vir, tau);
            let q_fock = pseudo_density_vir_fock(&c_vir, &f_diag, tau);
            let max_err = q_scalar.iter().zip(q_fock.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-12,
                "vir Fock path must match canonical scalar path at τ={tau}: max_err={max_err:.3e}");
        }
    }

    /// PROJECTOR-FORM Q̃: built from the AO Fock + the occupied projector ALONE
    /// (no c_vir), it must equal the explicit-virtual pseudo-density. This is the
    /// virtual-free construction — the virtual space is the S-orthogonal complement
    /// of the occupied space, propagated by exp(-Fτ). Test in an ORTHONORMAL AO
    /// basis (S = I) so the projector is `I - C_occ C_occᵀ`.
    #[test]
    fn pseudo_density_vir_projector_matches_explicit_orthonormal() {
        use ndarray_linalg::Eigh;
        let n = 6;
        let nocc = 2;
        // Orthonormal full MO set: eigenvectors of a symmetric matrix ⇒ Cᵀ C = I.
        let sym = Array2::from_shape_fn((n, n), |(i, j)| {
            0.4 * ((i + 1) as f64).sin() * ((j + 2) as f64).cos()
                + if i == j { 1.5 + i as f64 } else { 0.0 }
        });
        let (_w, c_all) = sym.eigh(ndarray_linalg::UPLO::Upper).unwrap(); // (n, n) orthonormal
        let eps_all = vec![-0.9_f64, -0.5, 0.3, 0.7, 1.1, 1.9];
        let c_occ = c_all.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c_all.slice(ndarray::s![.., nocc..]).to_owned();
        let eps_vir: Vec<f64> = eps_all[nocc..].to_vec();

        // AO Fock with S = I: F = C diag(ε) Cᵀ.
        let f_ao = c_all.dot(&Array2::from_diag(&ndarray::arr1(&eps_all))).dot(&c_all.t());
        // Identity overlap.
        let s = Array2::<f64>::eye(n);

        // Include LARGE τ: exp(−F̄τ) over occupied modes (negative ε) would
        // overflow as exp(+|ε|τ) unless the virtual projection is applied to the
        // exponent. The minimax τ-grid reaches large values, so this must hold.
        for &tau in &[0.05_f64, 0.4, 1.3, 3.0, 8.0, 15.0] {
            let q_explicit = pseudo_density_vir(&c_vir, &eps_vir, tau);
            let q_proj = pseudo_density_vir_projector(&f_ao, &s, &c_occ, tau);
            let max_err = q_explicit.iter().zip(q_proj.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-10,
                "projector Q̃ must match explicit-virtual Q̃ (S=I) at τ={tau}: max_err={max_err:.3e}");
        }
    }

    /// Same identity in a NON-ORTHOGONAL AO basis (S ≠ I): the projector form must
    /// still reproduce the explicit-virtual Q̃. This pins the metric handling.
    #[test]
    fn pseudo_density_vir_projector_matches_explicit_nonorthogonal() {
        use ndarray_linalg::Eigh;
        let n = 6;
        let nocc = 2;
        // A symmetric positive-definite overlap S.
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            0.2 * ((i + 1) as f64 / (j + 1) as f64).min((j + 1) as f64 / (i + 1) as f64)
        });
        let s = a.dot(&a.t()) + Array2::<f64>::eye(n); // SPD
        // Build S-orthonormal MOs: C with Cᵀ S C = I. Take eigh of S = U d Uᵀ,
        // S^{-1/2} = U d^{-1/2} Uᵀ, then any orthogonal R gives C = S^{-1/2} R.
        let (sd, su) = s.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let s_inv_sqrt = su.dot(&Array2::from_diag(&sd.mapv(|x| 1.0 / x.sqrt()))).dot(&su.t());
        let sym = Array2::from_shape_fn((n, n), |(i, j)| {
            0.3 * ((i + 2) as f64).cos() * ((j + 1) as f64).sin() + if i == j { 2.0 } else { 0.1 }
        });
        let (_w, r) = sym.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let c_all = s_inv_sqrt.dot(&r); // Cᵀ S C = Rᵀ S^{-1/2} S S^{-1/2} R = I
        let eps_all = vec![-0.8_f64, -0.4, 0.25, 0.6, 1.0, 1.7];
        let c_occ = c_all.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c_all.slice(ndarray::s![.., nocc..]).to_owned();
        let eps_vir: Vec<f64> = eps_all[nocc..].to_vec();
        // AO Fock: F = S C diag(ε) Cᵀ S (so that Cᵀ F C = diag(ε)).
        let f_ao = s.dot(&c_all).dot(&Array2::from_diag(&ndarray::arr1(&eps_all)))
            .dot(&c_all.t()).dot(&s);

        for &tau in &[0.05_f64, 0.4, 1.3, 3.0] {
            let q_explicit = pseudo_density_vir(&c_vir, &eps_vir, tau);
            let q_proj = pseudo_density_vir_projector(&f_ao, &s, &c_occ, tau);
            let max_err = q_explicit.iter().zip(q_proj.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-9,
                "projector Q̃ must match explicit-virtual Q̃ (S≠I) at τ={tau}: max_err={max_err:.3e}");
        }
    }

    /// The pseudo-density is a property of the occupied SUBSPACE, not the orbital
    /// basis: rotating the canonical orbitals by any orthogonal U (with F → UᵀFU)
    /// must leave P(τ) unchanged. This is the property that makes localized
    /// (non-canonical) orbitals valid inputs — it FAILS for the scalar path.
    #[test]
    fn pseudo_density_occ_fock_invariant_under_orbital_rotation() {
        use ndarray_linalg::Eigh;
        let nbasis = 6;
        let nocc = 3;
        let c_occ = Array2::from_shape_fn((nbasis, nocc), |(mu, i)| {
            0.2 + 0.05 * mu as f64 - 0.03 * i as f64 + 0.01 * (mu * i) as f64
        });
        let eps_occ = vec![-0.9_f64, -0.55, -0.3];
        let f_canon = Array2::from_diag(&ndarray::arr1(&eps_occ));

        // A fixed orthogonal rotation U (nocc×nocc) from eigenvectors of a sym matrix.
        let sym = Array2::from_shape_fn((nocc, nocc), |(i, j)| {
            0.3 * (i as f64 + 1.0) * (j as f64 + 1.0) + if i == j { 1.0 } else { 0.2 }
        });
        let (_w, u) = sym.eigh(ndarray_linalg::UPLO::Upper).unwrap();

        // Rotated orbitals C' = C U ; rotated Fock F' = Uᵀ F U (non-diagonal).
        let c_rot = c_occ.dot(&u);
        let f_rot = u.t().dot(&f_canon).dot(&u);

        for &tau in &[0.05_f64, 0.4, 1.3, 5.0] {
            let p_canon = pseudo_density_occ_fock(&c_occ, &f_canon, tau);
            let p_rot = pseudo_density_occ_fock(&c_rot, &f_rot, tau);
            let max_err = p_canon.iter().zip(p_rot.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-11,
                "occ Fock pseudo-density must be rotation-invariant at τ={tau}: max_err={max_err:.3e}");
        }
    }
}
