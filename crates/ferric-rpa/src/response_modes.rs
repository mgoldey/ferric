//! Dielectric response eigenmodes as real-space functions.
//!
//! The closed-shell RPA dielectric at imaginary frequency iω, in the
//! metric-dressed auxiliary basis, is
//!
//! ```text
//!   ε̃(iω) = I + B̃ D(ω) B̃ᵀ,   B̃ = F·(Q|ia),   D(ω) = diag(4Δ_ia / (Δ_ia² + ω²))
//! ```
//!
//! with `F` the metric factor of [`ferric_mp2::rimp2::RpaIntermediates`]
//! (`F V Fᵀ = I`, `V = (P|Q)`). Its eigenvectors `u_α` are the PDEP
//! eigenvectors. The physical aux coefficients are `c_α = Fᵀ u_α`; they solve
//! the generalized problem
//!
//! ```text
//!   (V + Π(iω)) c_α = λ_α V c_α,   c_αᵀ V c_β = δ_αβ,
//!   Π(iω) = (P|ia) D(ω) (ia|Q)
//! ```
//!
//! # What the function `ρ_α(r) = Σ_P c_αP χ_P(r)` is
//!
//! `χ_P` are fitting functions for a density, so `ρ_α` is a charge density,
//! normalised to unit Coulomb self-energy `(ρ_α|ρ_α) = c_αᵀ V c_α = 1`. Read it
//! as an electron density added to the molecule:
//!
//! * its Hartree potential `δV_α(r) = ∫ ρ_α(r') / |r − r'| dr'` is the
//!   perturbation, with MO matrix elements `⟨i|δV_α|a⟩ = Σ_P (ia|P) c_αP`;
//! * the independent-particle density it induces, Coulomb-fitted in the same
//!   aux basis, is `d = −V⁻¹ Π c_α = −(λ_α − 1) ρ_α` (the eigen-equation), so
//!   the mode is its own induced density up to the factor `−(λ_α − 1)`;
//! * self-consistently (RPA), an added density `ρ_α` is screened to `ρ_α / λ_α`.
//!
//! The perturbing potential `δV_α` is the Coulomb potential of `ρ_α`, not an
//! expansion in the `χ_P`; reading `Σ_P c_αP χ_P(r)` as a potential is wrong.
//!
//! # Coupling to a uniform field
//!
//! With `μ_d` the MO dipole integrals and `D_ω = diag(Δ / (Δ² + ω²))`,
//!
//! ```text
//!   α(iω) = 4 μᵀ D_ω μ − 16 w_ωᵀ ε̃(iω)⁻¹ w_ω,   w_ω = B̃ D_ω μ
//!         = α_bare(iω) − Σ_α s_α s_αᵀ / λ_α,      s_α,d = 4 u_αᵀ w_ω,d
//! ```
//!
//! the second line holding exactly when every mode is kept. In physical terms
//! `s_α,d = Σ_P c_αP (P|δn_d)` with `δn_d = 4 Σ_ia μ_d,ia Δ_ia/(Δ_ia²+ω²) φ_i φ_a`,
//! the Coulomb overlap of the mode with the bare induced density of a unit
//! field along `d`: `s_α` is how strongly mode α is driven by a field, and the
//! sum of `s_α s_αᵀ / λ_α` over modes is the RPA screening of the polarizability.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{Array1, Array2, Axis};

/// Leading eigenmodes of the closed-shell RPA dielectric at one imaginary
/// frequency, as physical auxiliary-basis densities. See the module docs.
#[derive(Debug, Clone)]
#[must_use]
pub struct ResponseModes {
    /// Imaginary frequency ω (Hartree) the dielectric was built at; 0 is static.
    pub omega: f64,
    /// Number of auxiliary functions.
    pub naux: usize,
    /// Eigenvalues λ_α(iω) ≥ 1, descending.
    pub eigenvalues: Vec<f64>,
    /// Physical aux coefficients, shape `(naux, M)`; column α gives `ρ_α`, with
    /// `c_αᵀ V c_β = δ_αβ`. Sign: the largest-magnitude coefficient is positive.
    pub coefficients: Array2<f64>,
    /// `‖ρ_α‖₂ = sqrt(c_αᵀ S c_α)` with `S` the aux overlap; divide a column by
    /// it for a mode with `∫ ρ_α² dr = 1`.
    pub l2_norms: Vec<f64>,
    /// Field couplings `s_α,d`, shape `(M, 3)` (see the module docs).
    pub field_couplings: Array2<f64>,
    /// Uncoupled polarizability `α_bare(iω) = 4 μᵀ D_ω μ` (a.u.).
    pub alpha_bare: [[f64; 3]; 3],
}

impl ResponseModes {
    /// `α_bare − Σ_{α<M} s_α s_αᵀ / λ_α`: the RPA polarizability at iω when
    /// every mode was kept (`M = naux`), its truncation to the leading `M`
    /// modes otherwise.
    pub fn polarizability_from_modes(&self) -> [[f64; 3]; 3] {
        let mut a = self.alpha_bare;
        for (k, &lam) in self.eigenvalues.iter().enumerate() {
            for d in 0..3 {
                for e in 0..3 {
                    a[d][e] -= self.field_couplings[(k, d)] * self.field_couplings[(k, e)] / lam;
                }
            }
        }
        a
    }
}

/// The `n_modes` largest-eigenvalue modes of the closed-shell RPA dielectric
/// ε̃(iω) built from `rhf` (all occupied orbitals active), by a dense eigensolve.
///
/// `n_modes` is clamped to the dressed aux dimension. Cost: one `(naux, nov)`
/// metric-dressed tensor, an `naux²·nov` product and an `naux³` eigensolve; the
/// memory pre-flight is the one [`crate::properties::pdep_polarizability_static`]
/// uses. Errors on an open-shell reference, a negative or non-finite `omega`,
/// or `n_modes == 0`.
#[allow(clippy::too_many_arguments)]
pub fn response_modes(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    n_modes: usize,
    omega: f64,
    memory_budget_bytes: Option<usize>,
) -> Result<ResponseModes, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "response_modes: closed-shell (restricted) reference only".into(),
        ));
    }
    if !(omega.is_finite() && omega >= 0.0) {
        return Err(FerricError::General(format!(
            "response_modes: omega must be a finite frequency >= 0, got {omega}"
        )));
    }
    if n_modes == 0 {
        return Err(FerricError::General(
            "response_modes: n_modes must be >= 1".into(),
        ));
    }

    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        memory_budget_bytes,
        ..Default::default()
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let (nocc, nvir) = (inter.nocc, inter.nvir);
    let nov = nocc * nvir;
    let n_dressed = b_ov.nrows();
    let naux = inter.v_inv_sqrt.ncols();

    let (_budget, _charge) = crate::properties::preflight_molecular_path(
        &format!("response_modes (naux={naux}, nocc={nocc}, nvir={nvir})"),
        memory_budget_bytes,
        n_dressed,
        nov,
        1,
    )?;

    let eps = rhf.eps_r();
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    // d_w[ia] = Δ/(Δ²+ω²); the dielectric uses 4·d_w.
    let mut d_w = Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            let de = eps[nocc_total + a] - eps[first_occ + i];
            d_w[i * nvir + a] = de / (de * de + omega * omega);
        }
    }

    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]);
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]);
    let mu: [Array1<f64>; 3] = std::array::from_fn(|d| {
        let m = c_occ.t().dot(&dip_ao[d]).dot(&c_vir);
        Array1::from_iter(m.iter().copied())
    });

    let mut alpha_bare = [[0.0_f64; 3]; 3];
    for d in 0..3 {
        for e in 0..3 {
            alpha_bare[d][e] = 4.0 * (&mu[d] * &d_w).dot(&mu[e]);
        }
    }
    let w: [Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&(&mu[d] * &d_w)));

    // ε̃ = I + (B̃ √(4 d_w)) (B̃ √(4 d_w))ᵀ
    let mut b_scaled = b_ov.clone();
    for (ia, mut col) in b_scaled.axis_iter_mut(Axis(1)).enumerate() {
        let s = (4.0 * d_w[ia]).sqrt();
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat = crate::sternheimer::syrk_aat(&b_scaled);
    drop(b_scaled);
    for p in 0..n_dressed {
        eps_mat[(p, p)] += 1.0;
    }
    let (evals, evecs) = eps_mat
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::Lapack(format!("response_modes: dielectric eigh: {e}")))?;
    drop(eps_mat);

    // eigh is ascending; take the n_modes largest, descending.
    let m = n_modes.min(n_dressed);
    let order: Vec<usize> = (0..n_dressed).rev().take(m).collect();
    let eigenvalues: Vec<f64> = order.iter().map(|&k| evals[k]).collect();
    let mut u = Array2::<f64>::zeros((n_dressed, m));
    for (j, &k) in order.iter().enumerate() {
        u.column_mut(j).assign(&evecs.column(k));
    }
    let mut coefficients = inter.v_inv_sqrt.t().dot(&u);
    for (j, mut col) in coefficients.axis_iter_mut(Axis(1)).enumerate() {
        let mut pivot = 0.0_f64;
        for &x in col.iter() {
            if x.abs() > pivot.abs() {
                pivot = x;
            }
        }
        if pivot < 0.0 {
            col.mapv_inplace(|x| -x);
            u.column_mut(j).mapv_inplace(|x| -x);
        }
    }

    let mut field_couplings = Array2::<f64>::zeros((m, 3));
    for d in 0..3 {
        let s = u.t().dot(&w[d]) * 4.0;
        field_couplings.column_mut(d).assign(&s);
    }

    let s_aux = oneelectron::overlap(dfbs);
    let sc = s_aux.dot(&coefficients);
    let l2_norms: Vec<f64> = (0..m)
        .map(|j| coefficients.column(j).dot(&sc.column(j)).max(0.0).sqrt())
        .collect();

    Ok(ResponseModes {
        omega,
        naux,
        eigenvalues,
        coefficients,
        l2_norms,
        field_couplings,
        alpha_bare,
    })
}
