//! Analytic RHF Hessian pieces for density-fitted EXCHANGE (RI-K).
//!
//! The energy this differentiates is the closed-shell RI-K energy of
//! [`crate::df_k::DfK`] / [`crate::df_gradient`]:
//!
//! ```text
//!   E_K = α Σ_ij L_ij · M L_ij ,   α = −¼,   D = X Xᵀ (X = √2 C_occ, m columns)
//!   L_{P,ij} = Σ_μν X_μi (P|μν) X_νj ,   M = V^{-1/2} V^{-1/2},   V_PQ = (P|Q)
//! ```
//!
//! which equals `−¼ Σ D_μλ D_νσ (μν|λσ)_fit` with
//! `(μν|λσ)_fit = (μν|P) M_PQ (Q|λσ)`. The exchange Fock matrix is
//! `F_K = ∂E_K/∂D = 2α K[D]`, `K[D]_μλ = Σ_νσ D_νσ (μν|λσ)_fit` (`−½K`).
//!
//! # Scope: no dropped metric modes
//!
//! `DfK` drops metric modes below `DFK_LINDEP_THRESH`. With none dropped,
//! `M = V⁻¹` exactly (the product of the two `V^{-1/2}` factors), so every
//! derivative below follows from `∂M = −M V_x M`. A metric with a dropped mode
//! is refused with a typed error naming the count (the energy is then a
//! spectral function of `V` and its second derivative needs the
//! Daleckii–Krein formula to second order, which is not implemented). Modes in
//! `[1e-10, 1e-7)` are kept (the Hessian warns above cond 1e8; the pre-SCF
//! probe that `Auto` consults refuses them and cond above 1e8, see
//! `probe_exchange_metric`).
//!
//! # Derivation (fixed `D`, i.e. fixed `X`; derivatives of integrals only)
//!
//! Index `ij` is carried along; `⟨a, b⟩ = Σ_{P,ij} a_{P,ij} b_{P,ij}`.
//! `Γ = M L` (`naux × m²`), `L_x = ∂L/∂x` through the integrals (the orbital
//! functions AND the auxiliary function `P` move with their atoms).
//!
//! ```text
//!   E_x  = α [ 2⟨L_x, Γ⟩ − ⟨Γ, V_x Γ⟩ ]
//!   ∂_y Γ = M Z_y ,         Z_y := L_y − V_y Γ
//!   E_xy = α [ 2⟨L_xy, Γ⟩ − ⟨Γ, V_xy Γ⟩ + 2⟨Z_x, M Z_y⟩ ]
//! ```
//!
//! (`∂_y` of `2⟨L_x,Γ⟩ = 2⟨L_xy,Γ⟩ + 2⟨L_x, M Z_y⟩` and of
//! `−⟨Γ, V_x Γ⟩ = −⟨Γ, V_xy Γ⟩ − 2⟨M Z_y, V_x Γ⟩`; the two cross pieces
//! combine to `2⟨(L_x − V_x Γ), M Z_y⟩`. This is the RI-J result
//! `d_xyᵀc − ½cᵀV_xy c + Z_xᵀ M Z_y` with `d → L` and `½ → α`.) The cross term
//! is symmetric in `x ↔ y`.
//!
//! The second derivatives enter only through two contractions, never as
//! tensors, both from [`crate::hessian_df_stencil`]:
//!
//! * `2α⟨L_xy, Γ⟩ = Σ_{P,μν} W_{Pμν} ∂²(P|μν)` with the per-auxiliary-function
//!   weight `W_P = 2α X Γ_P Xᵀ` (`Γ_P` is the symmetric `m × m` matrix of row
//!   `P`; `L` is symmetric in `ij`, hence so is `Γ`, which lets the swapped
//!   order of an off-diagonal shell pair carry the factor 2);
//! * `−α⟨Γ, V_xy Γ⟩ = Σ_PQ (−α Γ Γᵀ)_PQ ∂²(P|Q)`.
//!
//! ## First-order Fock matrix `K^x` (AO, fixed `D`)
//!
//! With the fit coefficients `C_{P,μσ} = M_PQ (Q|μσ)`,
//!
//! ```text
//!   K^x_μλ = Σ_{P,ν} ∂_x(P|μν) T_{P,νλ} + (μ↔λ) − Σ_PQ (V_x)_PQ Σ_j Ĉ_{P,μj} Ĉ_{Q,λj}
//!   Ĉ_{P,λj} = Σ_σ C_{P,λσ} X_σj = [M B]_{P,λj},  B_{P,λj} = Σ_σ (P|λσ) X_σj
//!   T_{P,νλ} = Σ_j X_νj Ĉ_{P,λj} = Σ_σ D_νσ C_{P,λσ}
//! ```
//!
//! (differentiate `(μν|P) M (Q|λσ) D_νσ`: the two integral derivatives give the
//! first two terms, `∂M = −M V_x M` the third.) The Fock contribution is
//! `2α K^x`. The matrix-valued piece the exact path does not have is the third
//! term (the aux-centre / metric response of the fit in the orbital basis); the
//! first piece contains the auxiliary-centre derivative of `(P|μν)` too.
//!
//! ## CPHF
//!
//! `G(D1) = J(D1) − ½K(D1)` is linear in `D1`; the response uses the SCF's own
//! `DfK` (the dressed `V^{-1/2}(Q|μν)` source) through its density path
//! `K = Σ_P B_P D1 B_P`. That path is written for a general matrix: it never
//! factors `D1`, so the non-positive-semidefinite response density
//! `D1 = C_v U C_oᵀ + h.c.` and the occupied-block density are handled with no
//! change (the occupied-orbital path `build_from_occ` is NOT used, it needs
//! `D = C Cᵀ`).
//!
//! # UHF: one channel per spin
//!
//! The unrestricted energy is a sum of per-spin exchange channels over ONE
//! shared fit, plus the Coulomb term of the TOTAL density:
//!
//! ```text
//!   E = Tr[D h] + ½ dᵀ M d(D_α + D_β) + Σ_σ α_σ Σ_ij L^σ_ij · M L^σ_ij + V_nn
//!   α_σ = −½ ,  D_σ = X_σ X_σᵀ ,  X_σ = C_σ,occ (unit occupations)
//!   L^σ_{P,ij} = Σ_μν X_σ,μi (P|μν) X_σ,νj ,  M = V⁻¹ (RI-K: the SCF's truncated V^{-1/2}V^{-1/2})
//! ```
//!
//! (`E_K = −½ Σ_σ tr(D_σ K[D_σ])`, `K[D]_μλ = Σ_νσ D_νσ (μν|λσ)_fit`; an αβ
//! exchange term does not exist, so no cross-channel term appears anywhere.)
//!
//! * **RI-J.** `d_P = Σ (P|μν) (D_α + D_β)_μν`: the Coulomb energy sees the
//!   spins only through the total density, so the merged RI-J terms
//!   ([`crate::hessian_df`]: skeleton, `J^x`) are used unchanged with
//!   `D = D_α + D_β`, and the one `J^x` is added to both spin Fock derivatives.
//! * **RI-K skeleton.** Every piece above is linear in `α_σ` at fixed `X_σ`,
//!   and `M`, `V_x`, `V_xy` are shared, so `Γ_σ = M L^σ`, `Z^σ_x = L^σ_x −
//!   V_x Γ_σ` give
//!   `E_xy = Σ_σ α_σ [2⟨L^σ_xy, Γ_σ⟩ − ⟨Γ_σ, V_xy Γ_σ⟩ + 2⟨Z^σ_x, M Z^σ_y⟩]`.
//!   The second derivatives enter through two contractions with SUMMED weights
//!   (`W_P = Σ_σ 2α_σ X_σ Γ_σ,P X_σᵀ` for the 3c stencil, `−Σ_σ α_σ Γ_σ Γ_σᵀ`
//!   for the 2c stencil): one pass each however many channels there are. The
//!   cross term is a sum of per-channel terms (`∂_y Γ_σ = M Z^σ_y` involves only
//!   channel `σ`).
//! * **`K^x`.** Per channel, with `X → X_σ` in the closed-shell expression;
//!   the spin Fock derivative gets `2α_σ K^x[D_σ] = −K^x[D_σ]`.
//! * **CPHF kernel.** `v_σ(D1_α, D1_β) = J_DF(D1_α + D1_β) − K_DF(D1_σ)`:
//!   `DfJ` on the TOTAL response density (one Coulomb matrix, both spins), `DfK`
//!   on each spin's own response density (its density path takes the indefinite
//!   `C_v U C_oᵀ + h.c.` as it is). The α and β rotations stay ONE coupled
//!   system through the Coulomb term.
//! * **Closed-shell limit.** `D_α = D_β = D/2`: `X_σ = X/√2`, `L^σ = L/2`, so
//!   `Σ_σ α_σ L^σ·M L^σ = 2(−½)(¼) L·M L = −¼ L·M L`, the closed-shell
//!   energy; every piece above reduces to the closed-shell piece, which is the
//!   exactness anchor of `tests/df_hessian_uhf.rs`.
//!
//! The channel code is `rik_channel_terms`; the closed-shell `rik_terms` is
//! the one-channel case `(X = √2 C_occ, α = −¼)`. Channels are processed one
//! after the other: `M` and `V_x` are built once, each channel's `L^σ`, `Γ_σ`,
//! `L^σ_x`, `Z^σ_x` and `K^x_σ` are built and (but for `Γ_σ`, `K^x_σ`) released
//! before the next, and what a finished channel leaves behind is charged by
//! `rik_peak_bytes_channels`.
//!
//! ## Factorization
//!
//! `M` is built as `W·W` with `W = U Λ^{-1/2} Uᵀ` from the same `eigh` as
//! `DfK::v_inv_sqrt_lindep`, i.e. the SCF's own construction (no explicit
//! `V⁻¹` from LU). With nothing dropped it agrees with `V⁻¹` to `~ε·cond(V)`.
//!
//! ## What the tests establish (and what they cannot see)
//!
//! * No exactly-spanning auxiliary basis exists for exchange: the pair
//!   densities `φ_i φ_j` of orbitals on different atoms are two-centre
//!   products, and with both on one atom every integral is one-centre (zero
//!   Hessian), so the RI-J exact-span anchor has no K analogue.
//! * For ONE occupied column `E_K = 2α E_J^DF`: every skeleton piece equals
//!   `2α` times the RI-J piece through an independent code path (blind to the
//!   pair index `ij`, i.e. to `m > 1`).
//! * `K^x` against the SCF's own `DfK` differenced; the skeleton pieces and the
//!   whole skeleton against differences of the first-derivative pieces and of
//!   `df_jk_gradient` at fixed density; the total against the differenced DF
//!   gradient with the SCF re-converged; PySCF's analytic DF Hessian.
//!
//! ## Memory and cost
//!
//! With `A = naux`, `n = nbf`, `m` = occupied count, `3N` coordinates, the
//! arrays alive together at the widest point are `M` and `V_x` (`A² + 3N·A²`),
//! `Γ` and `Ĉ` (`A·m² + A·n·m`), then per phase: `L_x` (`3N·A·m²`) with the
//! `K1` matrices (`3N·n²`) in the first-derivative pass; `L_x → Z_x` and
//! `∂_xΓ` (`2·3N·A·m²`) in the cross term; five `A·n·m` copies of `Ĉ` (the
//! stored layout, its flat and row views, the product with `V_x` and its
//! standard-layout copy) with `K^x` and its Fock-scaled copy in the exchange
//! Fock matrices; the metric factorization (`6·A²`) before them; and the CPHF
//! after them (`F^x`, `K^x` and the `G^x` accumulators, plus the dressed
//! `A·n²` tensors of the `DfJ` and `DfK` rebuilt there). The charge
//! (`rik_peak_bytes`) is the largest phase, reserved BEFORE the metric
//! factorization allocates; a typed error when it does not fit the global pool.
//! The pre-SCF probe `Auto` consults applies the same estimate (sizes alone), so
//! an over-budget job runs finite differences instead of failing after the SCF;
//! with no pool installed (library use) there is no budget and nothing is
//! checked. Peak: 2.7 MB for water cc-pVDZ / def2-universal-jkfit (3 atoms,
//! `A = 113`, `m = 5`); 9.8 GB for 30 atoms with 300 basis functions, `m = 60`,
//! `A = 1500` (`L_x` and `∂_xΓ` 3.9 GB each, `V_x` 1.6 GB). It grows like
//! **N⁴** (the `3N·A·m²` terms; fitted exponent 3.85 between 30 and 71 atoms at
//! 50 auxiliary functions, 10 basis functions and 2 occupied orbitals per
//! atom): about 270 GB for 71 atoms of that shape. Everything is built once per Hessian, not per CPHF
//! iteration; the integral work is one streamed `(P|μν)` pass, one
//! first-derivative 3c pass (`L_x` and `K1` together), one 2c pass, and the two
//! stencil contractions (about 30 times the engine calls of a gradient pass).
//! The `K^x` metric term costs `3N · (A²·n·m + A·m·n²)` flops (the sparsity of
//! `V_x` in the moved atom is not exploited).
//!
//! ## Conditioning
//!
//! The Cholesky-free `M` amplifies integral noise by `cond(V)`. Measured (water
//! 6-31G, Hessian vs the Richardson finite difference of ferric's own DF-K
//! gradient, fitting bases with no metric mode in the gradient's noise band):
//! cond 2.5e5 (cc-pVDZ-RI) 1.4e-10, 2.4e6 (aug-cc-pVDZ-RIFIT) 1.6e-10, 9.1e6
//! (aug-cc-pVTZ-RIFIT) 2.5e-10, 2.5e7 (def2-universal-jkfit) 1.3e-10, 2.4e7
//! (def2-QZVPP-RIFIT) 4.4e-10, **2.0e8 (aug-cc-pVQZ-RIFIT) 6.7e-9**: the error
//! begins to differ near cond 1e8, hence the warning above `RIK_COND_WARN`.
//! The RI-J analogue on the same fit (same metric, 3c integrals) stays at
//! 1.7e-10 there, and the two finite-difference references (h = 2e-3 and 4e-3,
//! Richardson) differ by only 3e-10, but PySCF's analytic DF Hessian agrees
//! with ferric as well at cond 2e8 (2.7e-8) as at cond 2.5e5 (2.2e-8), at its own
//! raw-asymmetry floor of 1.8e-8: the larger miss against the differenced
//! K gradient is attributed to that gradient's Daleckii–Krein roundoff
//! (unproven; no K-specific loss is established for the Hessian). Fixed-density
//! twin-augmented bases up to cond 7.9e8 show K and J skeletons with the same
//! 8e-8 (h = 1e-4) finite-difference truncation floor, so they cannot see 1e-9
//! effects. One real point above the threshold, hence the conservative
//! refusal in the probe. A
//! metric with a mode below the DF-K cut is refused (see Scope). The
//! reference gradient itself treats kept modes in `[1e-10, 1e-7)` as dropped
//! (its noise floor), so a basis with such a mode has no valid finite-difference
//! reference and yields no accuracy datum.

use crate::df_gradient::aux_basis;
use crate::hessian_df::{check_metric_spectrum, chunk_len, MetricUse};
use crate::hessian_df_stencil::{
    eri2_deriv2_contract, eri3_deriv2_contract, weight_matrix, StencilConfig,
};
use crate::result::DfJkRoute;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::{s, Array1, Array2, Array3, Axis};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use rayon::prelude::*;

/// Condition number of the exchange metric above which the Hessian warns and
/// the pre-SCF probe (`Auto`) falls back to finite differences. Measured (water
/// 6-31G, Hessian vs the differenced DF-K gradient): 1.3e-10 .. 4.4e-10 for
/// cond 2.5e5 .. 2.5e7 and 6.7e-9 at cond 2e8 (the RI-J analogue on the same
/// metric: 1.7e-10; the two differenced references differ by 3e-10). Against
/// PySCF's analytic DF Hessian the agreement is 2.2e-8 at cond 2.5e5 and
/// 2.7e-8 at 2e8 (PySCF's own raw asymmetry is 1.8e-8), so the larger
/// disagreement with the differenced gradient is not established as an error
/// of the Hessian; above 1e8 accuracy is simply not established.
pub const RIK_COND_WARN: f64 = 1e8;

/// Closed-shell exchange weight `α` of `E_K = α Σ_ij L_ij · M L_ij`.
pub const RIK_ALPHA: f64 = -0.25;

/// Deliberately broken constructions, wired through the production code path
/// for the mutation tests of `tests/df_hessian_rik.rs`. `Default` is the
/// production construction. NOT public API.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct RikMutants {
    /// Drop `−α⟨Γ, V_xy Γ⟩` (the 2c second derivative).
    pub drop_metric_second: bool,
    /// Drop the cross terms of the skeleton (`Z_x := L_x`).
    pub drop_cross: bool,
    /// Drop the whole metric-response term of `K^x` (`−V_x Ĉ Ĉ`).
    pub drop_kx_metric: bool,
    /// Drop `K^x` altogether (the fixed-density exchange Fock derivative).
    pub drop_kx: bool,
    /// Omit the auxiliary-centre derivative of `(P|μν)` in `K^x`.
    pub omit_aux_centre_kx: bool,
    /// Use the transposed pair `T_{P,λν}` in `K^x` (the `(ij)` pair swapped).
    pub transpose_pair_kx: bool,
    /// Use an erfc(0.4) metric for `V`, `V_x`, `V_xy`, `M`, with Coulomb 3c.
    pub swap_metric: bool,
    /// Apply an explicit `V⁻¹` (LU) where the SCF used `V^{-1/2}V^{-1/2}`.
    pub explicit_inverse: bool,
    /// Multiplier on the exchange weight `α` everywhere (1 = production).
    pub alpha_factor: f64,
    /// CPHF kernel mutant: the exact four-centre `K` inside `G` where the SCF
    /// used the fitted one.
    pub cphf_exact_k: bool,
    /// CPHF kernel mutant: `G = K − ½J` (J and K swapped).
    pub cphf_swap_jk: bool,
    /// Stencil construction (step, order, shell displacement).
    pub stencil: StencilConfig,
    /// Skip the conditioning refusal (measurement only).
    pub allow_ill_conditioned: bool,
    /// Channel mutant: `(channel, cut)` makes that channel use a metric
    /// inverse with every mode below `cut` dropped, while the other channels
    /// and `V_x` keep the shared one (a "shared M recomputed with a different
    /// cut" defect).
    pub channel_metric_cut: Option<(usize, f64)>,
}

impl Default for RikMutants {
    fn default() -> Self {
        RikMutants {
            drop_metric_second: false,
            drop_cross: false,
            drop_kx_metric: false,
            drop_kx: false,
            omit_aux_centre_kx: false,
            transpose_pair_kx: false,
            swap_metric: false,
            explicit_inverse: false,
            alpha_factor: 1.0,
            cphf_exact_k: false,
            cphf_swap_jk: false,
            stencil: StencilConfig::default(),
            allow_ill_conditioned: false,
            channel_metric_cut: None,
        }
    }
}

/// Deliberately broken constructions of the UHF channel wiring (mutation tests
/// of `tests/df_hessian_uhf.rs`). `Default` is the production construction.
/// NOT public API.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UhfDfMutants {
    /// Build `J^x` and the RI-J skeleton from the α density alone.
    pub j_from_alpha: bool,
    /// Add the α channel's `K^x` to the β Fock derivative and vice versa.
    pub swap_k_channels: bool,
    /// Use the closed-shell channel (`α = −¼`, `X = √2 C_σ`) for each spin.
    pub closed_shell_channel: bool,
    /// Leave the β channel out of the skeleton and of `K^x`.
    pub drop_beta_channel: bool,
    /// The CPHF kernel applies the exchange of the OTHER spin's response
    /// density: `v_α = J − K(D1_β)`, `v_β = J − K(D1_α)`.
    pub cphf_wrong_spin_k: bool,
    /// The CPHF kernel uses the exact four-centre `K` where the SCF fitted it.
    pub cphf_exact_k: bool,
    /// The CPHF kernel's Coulomb part is built from the α response density
    /// only (not the total).
    pub cphf_j_from_alpha: bool,
}

/// Exchange weight of one UHF spin channel: `E_K = −½ Σ_σ tr(D_σ K[D_σ])`,
/// `X_σ = C_σ,occ`.
pub const RIK_ALPHA_SPIN: f64 = -0.5;

/// The RI-K pieces of the Hessian.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikTerms {
    /// `∂²E_K^DF/∂x∂y` at fixed `D`, `3N × 3N` (symmetrized).
    pub skeleton: Array2<f64>,
    /// `max |S − Sᵀ|` of the skeleton before symmetrization.
    pub skeleton_asymmetry: f64,
    /// `2α⟨L_xy, Γ⟩`.
    pub three_centre: Array2<f64>,
    /// `−α⟨Γ, V_xy Γ⟩`.
    pub metric_second: Array2<f64>,
    /// `2α⟨Z_x, M Z_y⟩`.
    pub cross: Array2<f64>,
    /// `∂K[D]/∂x` (unscaled), one `n × n` AO matrix per coordinate.
    pub first_order_k: Vec<Array2<f64>>,
    /// `2α ∂K[D]/∂x`: what is ADDED to the Fock derivative (`−½K^x`).
    pub first_order_fock: Vec<Array2<f64>>,
}

/// First-derivative quantities of the fit for the piece tests.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikFirst {
    /// `Γ = M L` at this geometry, `naux × m²`.
    pub gamma: Array2<f64>,
    /// `⟨Γ*, L_x⟩` for each `x` (`Γ*` = own `Γ` or the fixed one).
    pub lg: Array1<f64>,
    /// `⟨Γ*, V_x Γ*⟩` for each `x`.
    pub gvg: Array1<f64>,
    /// `Z_x` for the own `Γ`, `3N × (naux·m²)`.
    pub z: Array2<f64>,
}

/// Metric, `M` and the operators of one exchange fit.
struct Fit {
    dfbs: PreparedBasis,
    m: Array2<f64>,
    /// Mutation tests only: the metric inverse of the one channel whose
    /// eigenvalue cut was raised (`RikMutants::channel_metric_cut`).
    m_cut: Option<(usize, Array2<f64>)>,
    op: Operator,
    metric_op: Operator,
    budget: usize,
}

impl Fit {
    fn new(
        dfbs: PreparedBasis,
        op: Operator,
        budget: usize,
        mutants: &RikMutants,
    ) -> Result<Self, FerricError> {
        let metric_op = if mutants.swap_metric {
            Operator::erfc(0.4)
        } else {
            op
        };
        let v = coulomb_metric_2c(metric_op, &dfbs)?;
        // ONE eigendecomposition, the (BLAS-thread-wrapped) call DfK makes, so
        // the dropped-mode classification and M come from the same numbers.
        let (lam, u) = with_blas_threads(opt_in_blas_threads(), || v.eigh(UPLO::Upper))
            .map_err(|e| FerricError::Lapack(format!("analytic RI-K Hessian: eigh: {e}")))?;
        if !mutants.allow_ill_conditioned {
            check_metric_spectrum(&lam, MetricUse::ExchangeFit)?;
        }
        let m = if mutants.explicit_inverse {
            v.inv()
                .map_err(|e| FerricError::Lapack(format!("analytic RI-K Hessian: inverse: {e}")))?
        } else {
            // The SCF's own construction (DfK::v_inv_sqrt_lindep) with nothing
            // dropped: W = U Λ^{-1/2} Uᵀ, M = W W.
            let mut us = u.clone();
            for k in 0..lam.len() {
                let sc = 1.0 / lam[k].sqrt();
                us.column_mut(k).mapv_inplace(|x| x * sc);
            }
            let w = us.dot(&u.t());
            w.dot(&w)
        };
        let m_cut = mutants.channel_metric_cut.map(|(channel, cut)| {
            let mut us = u.clone();
            for k in 0..lam.len() {
                let sc = if lam[k] >= cut {
                    1.0 / lam[k].sqrt()
                } else {
                    0.0
                };
                us.column_mut(k).mapv_inplace(|x| x * sc);
            }
            let w = us.dot(&u.t());
            (channel, w.dot(&w).as_standard_layout().to_owned())
        });
        Ok(Fit {
            dfbs,
            m: m.as_standard_layout().to_owned(),
            m_cut,
            op,
            metric_op,
            budget,
        })
    }

    /// The metric inverse channel `channel` uses (the shared `M` in production).
    fn m_for(&self, channel: usize) -> &Array2<f64> {
        match &self.m_cut {
            Some((c, m)) if *c == channel => m,
            _ => &self.m,
        }
    }
}

/// Bytes the Hessian pieces keep resident beyond the budget-bounded raw
/// tensor (see the module doc); charged to the pool before allocation.
/// Closed shell: one channel.
#[doc(hidden)]
pub fn rik_peak_bytes(natoms: usize, nbf: usize, nocc: usize, naux: usize) -> usize {
    rik_peak_bytes_channels(natoms, nbf, &[nocc], naux)
}

/// [`rik_peak_bytes`] for one exchange channel per entry of `noccs` (UHF: the
/// α and β occupied counts). The channels are processed one after the other;
/// the metric `M` and `V_x` are shared, and what a finished channel leaves
/// behind (`Γ_s`, `K^x_s` and its Fock-scaled copy) stays resident for the
/// later channels and the CPHF.
#[doc(hidden)]
pub fn rik_peak_bytes_channels(natoms: usize, nbf: usize, noccs: &[usize], naux: usize) -> usize {
    let (n3, a, n) = (3 * natoms, naux, nbf);
    let k1 = n3 * n * n;
    // The metric factorization (V, eigenvectors, scaled copy, W, M, check),
    // before anything else is alive.
    let mut peak = 6 * a * a;
    // What the finished channels leave resident: Γ_s and the two K^x copies.
    let mut prev = 0usize;
    for &m in noccs {
        let m2 = m * m;
        // M, V_x, Γ and Ĉ in (A, m, n) layout alive from the end of the
        // first-derivative passes to the outputs.
        let base = a * a + n3 * a * a + a * m2 + a * n * m;
        // B/Ĉ construction (B, B flat, Ĉ flat, permuted copy; L, Γ; M, W) with
        // M and V_x already built,
        let pre = 4 * a * n * m + 4 * a * m2 + 2 * a * a + n3 * a * a + prev;
        // the first-derivative 3c pass (L_x, K1 and the per-chunk partials),
        let pass2 = base + n3 * a * m2 + k1 + chunk_len(n3, nbf) * n3 * n * n + prev;
        // Z_x and ∂_xΓ (plus two (A, m²) temporaries),
        let cross = base + 2 * n3 * a * m2 + 2 * a * m2 + k1 + prev;
        // K^x: Ĉ, its flat and row copies, W and its standard-layout copy, and
        // K1, K^x, the unscaled and the Fock-scaled output matrices.
        let kx = base + k1 + 5 * a * n * m + 3 * n3 * n * n + prev;
        peak = peak.max(pre).max(pass2).max(cross).max(kx);
        prev += a * m2 + 2 * n3 * n * n;
    }
    // The CPHF that follows: per channel the first-order Fock matrices of its
    // spin, K^x and its Fock-scaled copy (3 per channel), the F^x / G^x
    // accumulators, and the SCF's DfJ and DfK rebuilt with the full budget
    // (dressed A n² each).
    let cphf = 3 * noccs.len() * n3 * n * n + 2 * a * n * n;
    8usize.saturating_mul(peak.max(cphf))
}

/// The RI-K Hessian pieces at the occupied factor `x` (`D = X Xᵀ`).
#[doc(hidden)]
pub fn rik_terms(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    x: &Array2<f64>,
    mutants: &RikMutants,
) -> Result<RikTerms, FerricError> {
    let name = route.k_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-K Hessian: the SCF route has no RI-K basis".into())
    })?;
    rik_terms_with_aux(
        mol.atoms.len(),
        obs,
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        x,
        mutants,
    )
}

/// One exchange channel `E = α Σ_ij L_ij · M L_ij` with `L = Xᵀ(P|··)X`:
/// closed shell `(√2 C_occ, −¼)`, one UHF spin `(C_σ,occ, −½)`.
#[doc(hidden)]
pub struct RikChannel<'a> {
    pub x: &'a Array2<f64>,
    pub alpha: f64,
}

/// The first-order exchange matrices of one channel.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikChannelFock {
    /// `∂K[D_s]/∂x` (unscaled), one `n × n` AO matrix per coordinate.
    pub first_order_k: Vec<Array2<f64>>,
    /// `2α_s ∂K[D_s]/∂x`: what is ADDED to this channel's Fock derivative.
    pub first_order_fock: Vec<Array2<f64>>,
}

/// The skeleton summed over channels and the per-channel first-order matrices.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikChannelTerms {
    pub skeleton: Array2<f64>,
    pub skeleton_asymmetry: f64,
    pub three_centre: Array2<f64>,
    pub metric_second: Array2<f64>,
    pub cross: Array2<f64>,
    pub channels: Vec<RikChannelFock>,
}

/// Quantities of one channel shared by the production terms and the piece tests.
struct Core {
    gamma: Array2<f64>,
    /// `L_x`, `3N × (A·m²)`.
    lx: Array2<f64>,
    /// `Ĉ` as `(A, m, n)`, the layout `K^x`'s metric term contracts.
    chat: Array3<f64>,
    /// `K1_x = Σ ∂_x(P|μν) T_{P,νλ}`, `3N × n²` (not symmetrized).
    k1: Array2<f64>,
}

#[allow(clippy::too_many_arguments)]
fn build_core(
    natoms: usize,
    obs: &PreparedBasis,
    fit: &Fit,
    m_metric: &Array2<f64>,
    x: &Array2<f64>,
    mutants: &RikMutants,
    with_k1: bool,
) -> Result<Core, FerricError> {
    let (n, naux, m) = (obs.nbasis(), fit.dfbs.nbasis(), x.ncols());
    let m2 = m * m;

    // Pass 1: L_P = Xᵀ (P|··) X and B_{P,λj} = Σ_σ (P|λσ) X_σj.
    let mut source = ThreeIndexSource::build(fit.op, obs, &fit.dfbs, fit.budget)?;
    let mut l = Array2::<f64>::zeros((naux, m2));
    let mut b = Array3::<f64>::zeros((naux, n, m));
    source.for_each_block(|blk| {
        let nb = blk.data.shape()[0];
        let rows: Vec<(Array2<f64>, Array2<f64>)> = (0..nb)
            .into_par_iter()
            .map(|q| {
                let t = blk.data.index_axis(Axis(0), q).dot(x);
                let lq = x.t().dot(&t);
                (t, lq)
            })
            .collect();
        for (q, (t, lq)) in rows.into_iter().enumerate() {
            b.index_axis_mut(Axis(0), blk.p0 + q).assign(&t);
            for (dst, src) in l.row_mut(blk.p0 + q).iter_mut().zip(lq.iter()) {
                *dst = *src;
            }
        }
        Ok(())
    })?;
    drop(source);

    let gamma = m_metric.dot(&l).as_standard_layout().to_owned();
    drop(l);
    let b_flat = b
        .into_shape_with_order((naux, n * m))
        .map_err(|e| FerricError::General(format!("B reshape: {e}")))?;
    let chat_flat = m_metric.dot(&b_flat).as_standard_layout().to_owned();
    drop(b_flat);
    let chat = chat_flat
        .into_shape_with_order((naux, n, m))
        .map_err(|e| FerricError::General(format!("Chat reshape: {e}")))?;

    // (A, n, m) -> (A, m, n)
    let chat_p = chat
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .to_owned();

    // Pass 2: first-derivative 3c blocks: L_x always, K1_x on request.
    let (lx, k1) = first_derivative_3c(obs, fit, natoms, x, &chat_p, mutants, with_k1)?;
    Ok(Core {
        gamma,
        lx,
        chat: chat_p,
        k1,
    })
}

/// `rik_terms` with an explicit auxiliary basis (the exactness anchors hand
/// it a basis that spans the pair densities exactly).
#[doc(hidden)]
pub fn rik_terms_with_aux(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    budget: usize,
    x: &Array2<f64>,
    mutants: &RikMutants,
) -> Result<RikTerms, FerricError> {
    let channel = RikChannel {
        x,
        alpha: RIK_ALPHA,
    };
    let mut t = rik_channel_terms_with_aux(natoms, obs, dfbs, op, budget, &[channel], mutants)?;
    let c = t.channels.remove(0);
    Ok(RikTerms {
        skeleton: t.skeleton,
        skeleton_asymmetry: t.skeleton_asymmetry,
        three_centre: t.three_centre,
        metric_second: t.metric_second,
        cross: t.cross,
        first_order_k: c.first_order_k,
        first_order_fock: c.first_order_fock,
    })
}

/// [`rik_channel_terms_with_aux`] with the auxiliary basis the SCF route names.
#[doc(hidden)]
pub fn rik_channel_terms(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    channels: &[RikChannel<'_>],
    mutants: &RikMutants,
) -> Result<RikChannelTerms, FerricError> {
    let name = route.k_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-K Hessian: the SCF route has no RI-K basis".into())
    })?;
    rik_channel_terms_with_aux(
        mol.atoms.len(),
        obs,
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        channels,
        mutants,
    )
}

/// The RI-K Hessian pieces for a set of exchange channels sharing ONE metric:
/// `E_K = Σ_s α_s Σ_ij L^s_ij · M L^s_ij` (cross-channel terms vanish for
/// exchange). `M` and `V_x` are built once; each channel then has its own
/// `L^s`, `Γ_s = M L^s`, `Z^s_x`, and `K^x_s`; the skeleton is the sum over
/// channels (the stencil contractions take the summed weights, so there is one
/// 3c and one 2c second-derivative pass however many channels).
#[doc(hidden)]
pub fn rik_channel_terms_with_aux(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    budget: usize,
    channels: &[RikChannel<'_>],
    mutants: &RikMutants,
) -> Result<RikChannelTerms, FerricError> {
    let (n, naux) = (obs.nbasis(), dfbs.nbasis());
    if channels.is_empty() {
        let z = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
        return Ok(RikChannelTerms {
            skeleton: z.clone(),
            skeleton_asymmetry: 0.0,
            three_centre: z.clone(),
            metric_second: z.clone(),
            cross: z,
            channels: Vec::new(),
        });
    }
    let noccs: Vec<usize> = channels.iter().map(|c| c.x.ncols()).collect();
    // Charged BEFORE the metric factorization allocates its (A, A) work arrays.
    let _charge = ferric_core::memory::pool::reserve_global(
        "RI-K Hessian fit tensors, L^x/Z^x, V^x and K^x matrices",
        rik_peak_bytes_channels(natoms, n, &noccs, naux),
    )?;
    let fit = Fit::new(dfbs, op, budget, mutants)?;
    let n3 = 3 * natoms;
    let vx = first_derivative_2c(&fit, natoms)?;

    let mut gammas: Vec<Array2<f64>> = Vec::with_capacity(channels.len());
    let mut cross: Option<Array2<f64>> = None;
    let mut fock_out = Vec::with_capacity(channels.len());
    for (ci, ch) in channels.iter().enumerate() {
        let (x, m) = (ch.x, ch.x.ncols());
        let m2 = m * m;
        let alpha = ch.alpha * mutants.alpha_factor;
        let m_metric = fit.m_for(ci);
        let mut core = build_core(natoms, obs, &fit, m_metric, x, mutants, true)?;
        let gamma = core.gamma.clone();

        // Z_x = L_x - V_x Γ (in place), then ∂_xΓ = M Z_x.
        if !mutants.drop_cross {
            for xi in 0..n3 {
                let vg = vx.index_axis(Axis(0), xi).dot(&gamma.view());
                let mut row = core.lx.row_mut(xi);
                let flat = row.as_slice_mut().expect("contiguous");
                for (dst, src) in flat.iter_mut().zip(vg.iter()) {
                    *dst -= *src;
                }
            }
        }
        let z = core.lx;
        let mut gamma_x = Array2::<f64>::zeros((n3, naux * m2));
        for xi in 0..n3 {
            let zx = z
                .row(xi)
                .to_owned()
                .into_shape_with_order((naux, m2))
                .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;
            let gx = m_metric.dot(&zx);
            for (dst, src) in gamma_x.row_mut(xi).iter_mut().zip(gx.iter()) {
                *dst = *src;
            }
        }
        let cross_s = (z.dot(&gamma_x.t())) * (2.0 * alpha);
        drop(gamma_x);
        drop(z);
        match cross.as_mut() {
            None => cross = Some(cross_s),
            Some(acc) => *acc += &cross_s,
        }

        // K^x = K1 + K1ᵀ - Σ_j Ĉ_jᵀ V_x Ĉ_j
        let chat_flat = core
            .chat
            .view()
            .into_shape_with_order((naux, m * n))
            .map_err(|e| FerricError::General(format!("Chat flat: {e}")))?
            .to_owned();
        let chat_rows = core
            .chat
            .view()
            .into_shape_with_order((naux * m, n))
            .map_err(|e| FerricError::General(format!("Chat rows: {e}")))?
            .to_owned();
        let mut first_order_k = Vec::with_capacity(n3);
        let mut first_order_fock = Vec::with_capacity(n3);
        for xi in 0..n3 {
            let k1 = core
                .k1
                .row(xi)
                .to_owned()
                .into_shape_with_order((n, n))
                .map_err(|e| FerricError::General(format!("K1 reshape: {e}")))?;
            let mut kx = &k1 + &k1.t();
            if !mutants.drop_kx_metric {
                let w = vx
                    .index_axis(Axis(0), xi)
                    .dot(&chat_flat)
                    .as_standard_layout()
                    .to_owned();
                let w_rows = w
                    .into_shape_with_order((naux * m, n))
                    .map_err(|e| FerricError::General(format!("W reshape: {e}")))?;
                kx -= &chat_rows.t().dot(&w_rows);
            }
            let scale = if mutants.drop_kx { 0.0 } else { 2.0 * alpha };
            first_order_fock.push(&kx * scale);
            first_order_k.push(kx);
        }
        fock_out.push(RikChannelFock {
            first_order_k,
            first_order_fock,
        });
        gammas.push(gamma);
    }
    drop(vx);
    let cross = cross.unwrap_or_else(|| Array2::zeros((n3, n3)));

    // Skeleton second derivatives through the stencil: the weights of all
    // channels are summed, one pass each for the 3c and the 2c integrals.
    let cfg = &mutants.stencil;
    let dfbs = &fit.dfbs;
    let alphas: Vec<f64> = channels
        .iter()
        .map(|c| c.alpha * mutants.alpha_factor)
        .collect();
    let three_centre = eri3_deriv2_contract(obs, dfbs, fit.op, cfg, |sp, s1, s2, w| {
        for (ci, ch) in channels.iter().enumerate() {
            let scale = 2.0 * alphas[ci];
            if ci == 0 {
                weight_gamma(obs, dfbs, &gammas[ci], ch.x, scale, (sp, s1, s2), w);
            } else {
                let mut extra = Vec::new();
                weight_gamma(
                    obs,
                    dfbs,
                    &gammas[ci],
                    ch.x,
                    scale,
                    (sp, s1, s2),
                    &mut extra,
                );
                for (a, b) in w.iter_mut().zip(&extra) {
                    *a += *b;
                }
            }
        }
        true
    })?;
    let metric_second = if mutants.drop_metric_second {
        Array2::zeros((n3, n3))
    } else {
        let mut m2w = gammas[0].dot(&gammas[0].t()) * (-alphas[0]);
        for ci in 1..gammas.len() {
            m2w += &(gammas[ci].dot(&gammas[ci].t()) * (-alphas[ci]));
        }
        eri2_deriv2_contract(dfbs, fit.metric_op, cfg, |sp, sq, w| {
            weight_matrix(dfbs, &m2w, (sp, sq), w)
        })?
    };
    let raw = &three_centre + &metric_second + &cross;
    let skeleton_asymmetry = raw
        .iter()
        .zip(raw.t().iter())
        .fold(0.0f64, |mx, (a, b)| mx.max((a - b).abs()));
    let skeleton = (&raw + &raw.t()) * 0.5;
    Ok(RikChannelTerms {
        skeleton,
        skeleton_asymmetry,
        three_centre,
        metric_second,
        cross,
        channels: fock_out,
    })
}

/// First-derivative pieces at the occupied factor `x` for the per-term finite
/// difference tests: `⟨Γ*, L_x⟩`, `⟨Γ*, V_x Γ*⟩` (with `Γ*` the fixed
/// `gamma_fixed` when given, else this geometry's own) and `Z_x`.
#[doc(hidden)]
pub fn rik_first(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    x: &Array2<f64>,
    gamma_fixed: Option<&Array2<f64>>,
) -> Result<RikFirst, FerricError> {
    let mutants = RikMutants::default();
    let name = route.k_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-K Hessian: the SCF route has no RI-K basis".into())
    })?;
    let natoms = mol.atoms.len();
    let fit = Fit::new(
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        &mutants,
    )?;
    let n3 = 3 * natoms;
    let vx = first_derivative_2c(&fit, natoms)?;
    let core = build_core(natoms, obs, &fit, &fit.m, x, &mutants, false)?;
    let g_star = gamma_fixed.unwrap_or(&core.gamma);
    let mut lg = Array1::<f64>::zeros(n3);
    let mut gvg = Array1::<f64>::zeros(n3);
    let mut z = core.lx.clone();
    for xi in 0..n3 {
        lg[xi] = core
            .lx
            .row(xi)
            .iter()
            .zip(g_star.iter())
            .map(|(a, b)| a * b)
            .sum();
        let vg_star = vx.index_axis(Axis(0), xi).dot(g_star);
        gvg[xi] = vg_star.iter().zip(g_star.iter()).map(|(a, b)| a * b).sum();
        let vg_own = vx.index_axis(Axis(0), xi).dot(&core.gamma);
        for (dst, src) in z.row_mut(xi).iter_mut().zip(vg_own.iter()) {
            *dst -= *src;
        }
    }
    Ok(RikFirst {
        gamma: core.gamma,
        lg,
        gvg,
        z,
    })
}

/// Block weights `w_{P,μν} = scale · (X Γ_P Xᵀ)_{μν}` for the 3c stencil;
/// `Γ_P` is symmetric, so an off-diagonal shell pair carries both orders as 2×.
fn weight_gamma(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    gamma: &Array2<f64>,
    x: &Array2<f64>,
    scale: f64,
    shells: (usize, usize, usize),
    w: &mut Vec<f64>,
) -> bool {
    let (sp, s1, s2) = shells;
    let m = x.ncols();
    let (od, oo) = (obs.shell_dims(), obs.shell_offsets());
    let (dd, doff) = (dfbs.shell_dims(), dfbs.shell_offsets());
    let x1 = x.slice(s![oo[s1]..oo[s1] + od[s1], ..]);
    let x2 = x.slice(s![oo[s2]..oo[s2] + od[s2], ..]);
    let fac = if s1 != s2 { 2.0 * scale } else { scale };
    for p in 0..dd[sp] {
        let g = gamma
            .slice(s![doff[sp] + p, ..])
            .into_shape_with_order((m, m))
            .expect("contiguous gamma row");
        let blk = x1.dot(&g).dot(&x2.t());
        w.extend(blk.iter().map(|v| fac * v));
    }
    true
}

/// `L_x` (`3N × A·m²`) and optionally `K1_x` (`3N × n²`) from the
/// first-derivative three-centre blocks, one auxiliary shell per task.
fn first_derivative_3c(
    obs: &PreparedBasis,
    fit: &Fit,
    natoms: usize,
    x: &Array2<f64>,
    chat: &Array3<f64>,
    mutants: &RikMutants,
    with_k1: bool,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n3 = 3 * natoms;
    let (n, naux, m) = (obs.nbasis(), fit.dfbs.nbasis(), x.ncols());
    let m2 = m * m;
    let dfbs = &fit.dfbs;
    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)?;
    let mut lx = Array2::<f64>::zeros((n3, naux * m2));
    let mut k1 = Array2::<f64>::zeros((n3, if with_k1 { n * n } else { 0 }));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(chunk_len(n3, n)) {
        let parts: Vec<(usize, Array2<f64>, Array2<f64>)> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)
                        .expect("3-center deriv engine (pre-validated)")
                },
                |eng, &sp| shell_partial(eng, obs, dfbs, sp, x, chat, n3, with_k1, mutants),
            )
            .collect();
        for (sp, lxp, k1p) in parts {
            let p0 = dfbs.shell_offsets()[sp];
            let width = lxp.ncols();
            lx.slice_mut(s![.., p0 * m2..p0 * m2 + width]).assign(&lxp);
            if with_k1 {
                k1 += &k1p;
            }
        }
    }
    Ok((lx, k1))
}

/// One auxiliary shell's `L_x` columns and `K1_x` partial.
#[allow(clippy::too_many_arguments)]
fn shell_partial(
    eng: &mut Engine,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    sp: usize,
    x: &Array2<f64>,
    chat: &Array3<f64>,
    n3: usize,
    with_k1: bool,
    mutants: &RikMutants,
) -> (usize, Array2<f64>, Array2<f64>) {
    let n = obs.nbasis();
    let m = x.ncols();
    let m2 = m * m;
    let (od, oo, o_at) = (obs.shell_dims(), obs.shell_offsets(), obs.shell_to_atom());
    let np = dfbs.shell_dims()[sp];
    let pf0 = dfbs.shell_offsets()[sp];
    let atom_p = dfbs.shell_to_atom()[sp];
    let mut lxp = Array2::<f64>::zeros((n3, np * m2));
    let mut k1p = Array2::<f64>::zeros((n3, if with_k1 { n * n } else { 0 }));
    // T_p[ν, λ] = Σ_j X_νj Ĉ_{p,λ,j}  (Ĉ stored (A, m, n): chat[p, j, λ]).
    let t: Vec<Array2<f64>> = if with_k1 {
        (0..np)
            .map(|p| {
                let cp = chat.index_axis(Axis(0), pf0 + p); // (m, n): [j, λ]
                let tp = x.dot(&cp);
                if mutants.transpose_pair_kx {
                    tp.t().as_standard_layout().to_owned()
                } else {
                    tp
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    for s1 in 0..obs.nshells() {
        for s2 in 0..=s1 {
            let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                continue;
            };
            let (n1, n2) = (od[s1], od[s2]);
            let bs = np * n1 * n2;
            let atoms = [atom_p, o_at[s1], o_at[s2]];
            let x1 = x.slice(s![oo[s1]..oo[s1] + n1, ..]);
            let x2 = x.slice(s![oo[s2]..oo[s2] + n2, ..]);
            for (centre, &atom) in atoms.iter().enumerate() {
                for q in 0..3 {
                    let xi = 3 * atom + q;
                    for p in 0..np {
                        let off = (3 * centre + q) * bs + p * n1 * n2;
                        let dm =
                            ndarray::ArrayView2::from_shape((n1, n2), &deriv[off..off + n1 * n2])
                                .expect("block shape");
                        // L_x[P] += X1ᵀ Dm X2 (+ transpose for s1 != s2)
                        let y = x1.t().dot(&dm).dot(&x2);
                        let mut row = lxp.row_mut(xi);
                        let seg =
                            &mut row.as_slice_mut().expect("contiguous")[p * m2..(p + 1) * m2];
                        for i in 0..m {
                            for j in 0..m {
                                let v = if s1 != s2 {
                                    y[(i, j)] + y[(j, i)]
                                } else {
                                    y[(i, j)]
                                };
                                seg[i * m + j] += v;
                            }
                        }
                        if with_k1 && !(mutants.omit_aux_centre_kx && centre == 0) {
                            let tp = &t[p];
                            let r1 = dm.dot(&tp.slice(s![oo[s2]..oo[s2] + n2, ..]));
                            let mut k1row = k1p.row_mut(xi);
                            let kseg = k1row.as_slice_mut().expect("contiguous");
                            for a in 0..n1 {
                                let base = (oo[s1] + a) * n;
                                for lam in 0..n {
                                    kseg[base + lam] += r1[(a, lam)];
                                }
                            }
                            if s1 != s2 {
                                let r2 = dm.t().dot(&tp.slice(s![oo[s1]..oo[s1] + n1, ..]));
                                for b2 in 0..n2 {
                                    let base = (oo[s2] + b2) * n;
                                    for lam in 0..n {
                                        kseg[base + lam] += r2[(b2, lam)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (sp, lxp, k1p)
}

/// Dense `V_x = ∂(P|Q)/∂x` (`3N × A × A`) from the first-derivative 2c blocks.
fn first_derivative_2c(fit: &Fit, natoms: usize) -> Result<Array3<f64>, FerricError> {
    let n3 = 3 * natoms;
    let dfbs = &fit.dfbs;
    let naux = dfbs.nbasis();
    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)?;
    let (dims, offs, at) = (
        dfbs.shell_dims().to_vec(),
        dfbs.shell_offsets().to_vec(),
        dfbs.shell_to_atom().to_vec(),
    );
    let mut vx = Array3::<f64>::zeros((n3, naux, naux));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(32) {
        type Blocks = Vec<(usize, usize, Array3<f64>)>;
        let parts: Vec<Blocks> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)
                        .expect("2-center deriv engine (pre-validated)")
                },
                |eng, &sp| {
                    let mut out = Vec::new();
                    for sq in 0..=sp {
                        let Some(deriv) = eng.compute_eri2_deriv(dfbs, sp, sq) else {
                            continue;
                        };
                        let (np, nq) = (dims[sp], dims[sq]);
                        let bs = np * nq;
                        let mut blk = Array3::<f64>::zeros((n3, np, nq));
                        for (centre, &atom) in [at[sp], at[sq]].iter().enumerate() {
                            for k in 0..3 {
                                for p in 0..np {
                                    for q in 0..nq {
                                        blk[(3 * atom + k, p, q)] +=
                                            deriv[(3 * centre + k) * bs + p * nq + q];
                                    }
                                }
                            }
                        }
                        out.push((sp, sq, blk));
                    }
                    out
                },
            )
            .collect();
        for blocks in parts {
            for (sp, sq, blk) in blocks {
                let (np, nq) = (dims[sp], dims[sq]);
                for xi in 0..n3 {
                    for p in 0..np {
                        for q in 0..nq {
                            let v = blk[(xi, p, q)];
                            vx[(xi, offs[sp] + p, offs[sq] + q)] = v;
                            vx[(xi, offs[sq] + q, offs[sp] + p)] = v;
                        }
                    }
                }
            }
        }
    }
    Ok(vx)
}
