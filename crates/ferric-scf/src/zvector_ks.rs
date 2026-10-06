//! Kohn–Sham Z-vector (closed-shell RKS, unrestricted UKS and restricted
//! open-shell ROKS references): the
//! orbital-relaxation term of the nuclear gradient of a post-SCF quantity
//! Q(R, D).
//!
//! # What it computes
//!
//! For a quantity Q that depends on the converged closed-shell density D (and
//! on the geometry directly), the total nuclear derivative is
//!
//! ```text
//!   dQ/dR = ∂Q/∂R |_{occupied orbitals held fixed}  +  Σ_ai Z_ai ∂F_ai/∂R
//! ```
//!
//! "Held fixed" is the path on which the reference occupied orbitals follow
//! their basis functions and are kept orthonormal (dC_occ = −½ C_occ S^x_oo,
//! so dD = −½ D S^x D): the caller supplies that part. This module supplies
//! the second, the orbital relaxation, from one linear solve:
//!
//! ```text
//!   H Z = −4 V_vo,     V = ∂Q/∂D (AO, symmetric),   V_vo = C_virᵀ V C_occ
//! ```
//!
//! with H = ∂F_ai/∂κ_bj the closed-shell orbital Hessian (the Jacobian of the
//! Brillouin condition F_ai = 0) in the convention of
//! [`crate::rhf_newton::hessian_matvec`]: a rotation κ changes the density by
//! δD = 2 C δD_MO Cᵀ with δD_MO\[a,i\] = δD_MO\[i,a\] = κ_ai, so dE_SCF = 4 Σ κ_ai F_ai
//! and dQ = 4 Σ κ_ai V_ai.
//!
//! # The contraction Σ Z_ai ∂F_ai/∂R without new derivative integrals
//!
//! F_ai = ¼ ∂E_SCF/∂κ_ai, so Σ_ai Z_ai ∂F_ai/∂R = ¼ d/dt [∂E_SCF/∂R] at the
//! orbitals rotated by t·Z. The nuclear derivative of E_SCF along the
//! fixed-orbital path is the ordinary gradient EXPRESSION g(D, W) with
//! W = ½ D F\[D\] D (which is the usual 2 Σ ε_i C_i C_iᵀ at convergence), so
//!
//! ```text
//!   Σ Z_ai ∂F_ai/∂R = ¼ d/dt g(D + t δD_Z, W + t Ẇ) |_{t=0},
//!   δD_Z = 2 (C_vir Z C_occᵀ + C_occ Zᵀ C_virᵀ),
//!   Ẇ    = ½ (δD_Z F D + D F δD_Z + D δF D),   δF = response Fock of δD_Z,
//! ```
//!
//! (with D(t) built from the occupied orbitals rotated by t·Z, an exact
//! projector) evaluated by a central difference in t of
//! [`crate::ks_gradient::ks_gradient_closed_for_density`] — the SAME gradient
//! code, routing (RI-J, fitted or COSX exchange) and XC grid response the SCF
//! gradient uses. g is linear in W and linear or bilinear in D for the
//! one-electron, ECP and two-electron parts, so the central difference is exact
//! for them up to rounding; only the XC part carries an O(t²) truncation, and
//! t is chosen so that t·max|δD_Z| = `T_SCALE`.
//!
//! # Scope
//!
//! Closed-shell KS (LDA, GGA, global hybrid, range-separated hybrid GGA) on a
//! converged `Spin::Restricted` result. Refused, with a reason, rather than
//! answered approximately: HF (no `xc`), meta-GGA (no τ f_xc kernel), VV10
//! (no nonlocal kernel), an `xc_omega` override, COSMO/PCM/polarizable
//! embedding and cDFT constraints (their response is not in H), fractional
//! occupation and smearing (not a closed-shell idempotent density).
//!
//! # Unrestricted (UKS) references
//!
//! [`crate::zvector_ks::relaxation_gradient_unrestricted`] is the same construction with one
//! occ→virt rotation per spin, unit occupations (δD_σ = C_σ(κ^σ + κ^σᵀ)C_σᵀ,
//! so dE = 2 Σ_σ Σ κ^σ_ai F^σ_ai), the coupled α/β Hessian (δJ on the total
//! perturbation, δK and f_xc spin-resolved), H Z = −2 V_vo per spin for a Q of
//! the total density, W = Σ_σ D_σ F_σ D_σ, and a factor ½ (not ¼) on the
//! directional derivative of
//! [`crate::ks_gradient::ks_gradient_uks_for_density`]. The same references
//! are refused, plus COSX exchange (no UKS COSX gradient) and spin densities
//! that are not the aufbau projectors of the MOs (MOM).
//!
//! # Restricted open-shell (ROKS) references
//!
//! [`crate::zvector_ks::relaxation_gradient_roks`]. One set of spatial
//! orbitals C = (C_c | C_o | C_v) (closed, open, virtual), occupation vectors
//! n_α = (1, 1, 0) and n_β = (1, 0, 0) over the three blocks, so
//! D_α = P_c + P_o and D_β = P_c with P_x = C_x C_xᵀ. The ROKS energy is the
//! UKS functional E\[D_α, D_β\] restricted to such densities.
//!
//! **Rotations and the gradient.** C(κ) = C exp(κ), κ antisymmetric with the
//! three density-changing blocks κ_vc, κ_vo, κ_oc (closed–closed, open–open
//! and virtual–virtual rotations leave both densities unchanged). To first
//! order δD_σ = C (κ n_σ − n_σ κ) C ᵀ, i.e. in the MO basis
//! δD_σ\[p,q\] = (n_σ\[q\] − n_σ\[p\]) κ\[p,q\]:
//!
//! ```text
//!   κ_vc moves α and β,  κ_vo moves α only,  κ_oc moves β only,
//!   dE = Σ_σ tr(F_σ δD_σ) = 2 (κ_vc·g_vc + κ_vo·g_vo + κ_oc·g_oc),
//!   g_vc = (F_α + F_β)_vc,  g_vo = F_α,vo,  g_oc = F_β,oc,
//! ```
//!
//! the blocks and normalization of `rohf_newton::gradient_blocks`
//! (F_σ the spin Focks in the MO basis). Converged ROKS means g = 0, which
//! does NOT make F_α,vc, F_β,vc, F_α,co or F_β,vo vanish individually
//! (F_β,vc = −F_α,vc).
//!
//! **(a) The orbitals-held-fixed path.** The occupied orbitals follow their
//! basis functions and are Löwdin re-orthonormalized as ONE set (closed and
//! open together, since they must stay mutually orthogonal):
//! dC_occ = −½ C_occ M, M = C_occᵀ Sˣ C_occ. Then
//!
//! ```text
//!   dD_α = −D_α Sˣ D_α,
//!   dD_β = −P_c Sˣ P_c − ½ (P_o Sˣ P_c + P_c Sˣ P_o),
//! ```
//!
//! the closed–open cross term coming from M_oc: the closed orbitals pick up
//! −½ C_o M_oc. For a quantity Q(D_total) with V = ∂Q/∂D_total this is the
//! orthonormality term −Tr\[Sˣ W_Q\],
//! W_Q = D_α V D_α + P_c V P_c + ½ (P_c V P_o + P_o V P_c); for the energy it
//! is −Tr\[Sˣ W\] with
//!
//! ```text
//!   W = D_α F_α D_α + P_c F_β P_c + ½ (P_c F_β P_o + P_o F_β P_c),
//! ```
//!
//! whose cross term vanishes at convergence (P_c F_β P_o = C_c F_β,co C_oᵀ
//! and F_β,oc = g_oc = 0) but whose t-derivative below does not. V_co does
//! not vanish, so the cross term of W_Q is a real contribution, not the UKS
//! form −Σ_σ D_σ Sˣ D_σ.
//!
//! **(b) The Z-vector equation.** The local gradient g(κ) is read from the
//! MO spin Focks at C exp(κ): F_σ^MO(κ) ≈ F_σ + \[F_σ, κ\] + C ᵀ δF_σ C, with
//! δF_σ the spin response Fock of (δD_α, δD_β) (δJ on the total, δK and f_xc
//! spin-resolved — `rohf_newton::hessian_matvec`'s δF). Its Jacobian
//! H = ∂g/∂κ, written with the stationarity conditions so that it is
//! symmetric:
//!
//! ```text
//!   (H x)_vc = F⁺_vv x_vc − x_vc F⁺_cc + F_β,vo x_oc − x_vo F_α,oc + (δF_α + δF_β)_vc
//!   (H x)_vo = F_α,vv x_vo − x_vo F_α,oo + G_vc x_ocᵀ − x_vc F_α,co + δF_α,vo
//!   (H x)_oc = F_β,oo x_oc − x_oc F_β,cc + F_β,ov x_vc + x_voᵀ G_vc + δF_β,oc
//!   F⁺ = F_α + F_β,   G_vc = ½ (F_β − F_α)_vc   (= F_β,vc = −F_α,vc at convergence)
//! ```
//!
//! (the F_β,vo / F_α,oc terms of the vc row use F_α,vo = 0 and F_β,oc = 0).
//! `rohf_newton::hessian_matvec` keeps only the diagonal Fock
//! entries and drops these off-diagonal Fock couplings — adequate for a
//! damped Newton step, not for an exact linear response — so the Z-vector
//! uses this full product in its parametrization and normalization. With
//! dQ = 2 κ·(2 V_vc, V_vo, V_oc) (κ_vc moves both spins of D_total):
//!
//! ```text
//!   H Z = −(4 V_vc, 2 V_vo, 2 V_oc),   V_xy = C_xᵀ V C_y.
//! ```
//!
//! **(c) The contraction.** g = ½ ∂E/∂κ at κ = 0 for every geometry, so
//! Z·∂g/∂R = ½ d/dt \[∂E/∂R at C exp(tZ)\]. ∂E/∂R at the rotated orbitals,
//! with the occupied set re-orthonormalized as in (a), is the ROKS gradient
//! expression [`crate::ks_gradient::ks_gradient_roks_for_density`] at
//! (D_α(t), D_β(t), W(t)); any other connection differs at O(t) only along
//! density-changing rotations, whose energy gradient is itself O(t), so the
//! t-derivative is unchanged. D_σ(t) comes from the Cayley rotation
//! C (I − tκ/2)⁻¹(I + tκ/2) (exact projectors, dD/dt = δD_Z), W(t) = W + t Ẇ
//! with Ẇ the first-order change of W above (δP_c = δD_β,
//! δP_o = δD_α − δD_β, δF_σ the response Fock of δD_Z), and the factor is ½.
//!
//! For a closed shell run through this path (no open orbitals) H is twice the
//! closed-shell Hessian, the right-hand side is the same −4 V_vc, so Z is half
//! the closed-shell Z and the ½ contraction of the half-sized rotation equals
//! the closed-shell ¼: the exactness anchor of the tests.
//!
//! H uses exact four-centre J and K. When the SCF fitted J and/or K (RI-J is
//! `run_dft`'s default), Z solves a Hessian that differs from the SCF's by the
//! fitting error; the contraction itself uses the SCF's own route.

use crate::engine_pool::EnginePool;
use crate::result::{ScfResult, Spin};
use crate::rhf::{build_jk_with_pool, RhfConfig};
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array2};

/// t · max|δD_Z| for the central difference in t (see the module doc).
pub const T_SCALE: f64 = 1e-4;
/// Converged when max|residual| ≤ `CG_REL_TOL` · max|rhs|.
pub const CG_REL_TOL: f64 = 1e-10;
/// PCG iteration cap; exceeding it is an error, never a silent answer.
pub const CG_MAX_ITER: usize = 200;
/// When the curvature along the search direction is numerically zero (PCG
/// at its rounding floor), the solve is accepted if max|residual| is already
/// ≤ `CG_FLOOR_TOL` · max|rhs|, and is a convergence error otherwise.
pub const CG_FLOOR_TOL: f64 = 1e-8;

/// Curvature verdict for one PCG step, with `php = p·Hp` and `pmp = p·M p`
/// (M = the positive orbital-energy-gap preconditioner, so `pmp` is the
/// natural scale of `php`).
///
/// * Positive curvature: proceed.
/// * Otherwise, if max|residual| is already ≤ `CG_FLOOR_TOL` · max|rhs|, PCG
///   has reached the rounding floor of the Hessian product: p is noise and the
///   sign of p·Hp carries no information (measured on UKS OH/STO-3G PBE:
///   p·Hp = −8.4e-19 against p·Mp = 1.1e-16). Stop and accept.
/// * Otherwise negative curvature means H is indefinite (the SCF is not a
///   minimum) and zero curvature means PCG stalled: both errors.
fn pcg_curvature_ok(php: f64, pmp: f64, residual: f64, rhs_max: f64) -> Result<bool, FerricError> {
    const ROUNDING: f64 = 1e-12;
    if php > ROUNDING * pmp {
        return Ok(true);
    }
    if residual <= CG_FLOOR_TOL * rhs_max {
        return Ok(false);
    }
    if php < 0.0 {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: the orbital Hessian is not positive definite along the \
             search direction (p.Hp = {php:.3e}, p.Mp = {pmp:.3e}, max|r| = {residual:.3e}, max|rhs| = {rhs_max:.3e}); \
             the SCF is not a minimum, or (open shell) a near-degenerate state makes the \
             orbital Hessian near-singular. The axis-rotation null mode of a linear molecule \
             is removed automatically (NullModeDiagnostics); any other near-null mode is not"
        )));
    }
    Err(FerricError::Convergence(format!(
        "Z-vector relaxation: PCG stalled (p.Hp = {php:.3e}, p.Mp = {pmp:.3e}) with \
         max|r| = {residual:.3e} > {:.3e}",
        CG_FLOOR_TOL * rhs_max
    )))
}

/// The relaxation term and its solve diagnostics.
#[derive(Debug, Clone)]
pub struct RelaxationGradient {
    /// Σ_ai Z_ai ∂F_ai/∂R, (natoms, 3), in the units of Q per Bohr.
    pub gradient: Array2<f64>,
    /// Z, (nvir, nocc).
    pub z: Array2<f64>,
    /// PCG iterations used.
    pub iterations: usize,
    /// Final max|residual| of H Z = −4 V_vo.
    pub residual: f64,
    /// `true` when PCG stopped at the rounding floor of the Hessian product
    /// (numerically zero curvature) with `residual` between `CG_REL_TOL` and
    /// `CG_FLOOR_TOL` times max|rhs| instead of reaching `CG_REL_TOL`. The
    /// relaxation term then carries a relative error of order
    /// `residual / max|rhs|` (≤ `CG_FLOOR_TOL`).
    pub stopped_at_floor: bool,
}

/// Why a closed-shell Z-vector cannot be formed for `config`, or `None`.
pub fn unsupported_reason(config: &RhfConfig) -> Option<String> {
    let xc = match config.xc.as_deref() {
        None => {
            return Some(
                "the Z-vector relaxation term is implemented for Kohn-Sham references only \
                 (config.xc is None)"
                    .into(),
            )
        }
        Some(x) => x,
    };
    if crate::rohf::xc_is_metagga(Some(xc)) {
        return Some(format!(
            "'{xc}' is a meta-GGA: there is no tau f_xc kernel for the orbital Hessian"
        ));
    }
    match ferric_dft::libxc::xc_def_from_name(xc) {
        Err(e) => return Some(format!("unknown functional '{xc}': {e:?}")),
        Ok(def) if def.vv10.is_some() => {
            return Some(format!(
                "'{xc}' carries VV10 nonlocal correlation: there is no VV10 kernel for the \
                 orbital Hessian"
            ))
        }
        Ok(_) => {}
    }
    if config.xc_omega.is_some() {
        return Some("an xc_omega override is not threaded into the orbital Hessian".into());
    }
    if config.cosmo.is_some() || config.pcm.is_some() {
        return Some(
            "implicit solvation: the solvent response is not in the orbital Hessian".into(),
        );
    }
    if config.polarizable.is_some() {
        return Some(
            "polarizable embedding: the induced-dipole response is not in the orbital Hessian"
                .into(),
        );
    }
    if !config.constraints.is_empty() {
        return Some("cDFT constraints are not in the orbital Hessian".into());
    }
    if config.fractional_occ || config.smearing_sigma.is_some() {
        return Some(
            "fractional occupation / smearing: the density is not a closed-shell projector".into(),
        );
    }
    None
}

/// Closed-shell response Fock δF[δD] = δJ − ½(c_sr δK_sr + c_lr δK_lr) + δV_xc
/// for a total-density perturbation `dd` (ω = 0: one exact K scaled by c_sr).
struct ResponseFock<'a> {
    prep: &'a PreparedBasis,
    thresh: f64,
    band_bytes: usize,
    coul: (SchwarzBounds, EnginePool),
    /// (bounds, pool, coefficient) per exchange operator.
    exch: Vec<(SchwarzBounds, EnginePool, f64)>,
    /// Coulomb-operator K reuses `coul`; its coefficient.
    k_coul: f64,
    fxc: crate::rohf::FxcKernelStore,
}

impl<'a> ResponseFock<'a> {
    /// `d_a`, `d_b`: the reference spin densities of the f_xc kernel (½D
    /// each for a closed-shell reference).
    fn new(
        mol: &Molecule,
        prep: &'a PreparedBasis,
        config: &RhfConfig,
        xc: &str,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<Self, FerricError> {
        let def = ferric_dft::libxc::xc_def_from_name(xc)
            .map_err(|e| FerricError::General(format!("zvector: libxc '{xc}': {e:?}")))?;
        let k_mix = ferric_dft::libxc::k_mix_from_xc_def(&def);
        let precision = ferric_integrals::engine_pool::eri_precision();
        let coul_op = Operator::coulomb();
        let coul = (
            SchwarzBounds::compute(coul_op, prep)?,
            EnginePool::new(coul_op, prep, precision)?,
        );
        let mut exch = Vec::new();
        let k_coul = if k_mix.omega > 0.0 {
            for (op, c) in [
                (Operator::erfc(k_mix.omega), k_mix.sr),
                (Operator::erf(k_mix.omega), k_mix.lr),
            ] {
                if c != 0.0 {
                    exch.push((
                        SchwarzBounds::compute(op, prep)?,
                        EnginePool::new(op, prep, precision)?,
                        c,
                    ));
                }
            }
            0.0
        } else {
            k_mix.sr
        };
        let grid = config.dft_grid.clone().unwrap_or_default();
        // `ks_hessian_unsupported_reason` refuses an `xc_omega` run outright, so
        // this is always `None` in practice; passed through rather than hardcoded
        // so the two places cannot drift apart silently.
        let fxc =
            crate::rohf::FxcKernelStore::build(mol, prep, &grid, xc, config.xc_omega, d_a, d_b)?;
        let budget = crate::rhf::resolve_three_index_budget(config.three_index_budget_bytes);
        Ok(Self {
            prep,
            thresh: config.integral_thresh,
            band_bytes: crate::reduce::resolve_band_bytes(budget),
            coul,
            exch,
            k_coul,
            fxc,
        })
    }

    /// δF[dd]. The operator is linear, so it is applied to dd / max|dd| and
    /// rescaled: the JK builders screen against an ABSOLUTE threshold, and the
    /// Z-vector's trial densities are small (~1e-5), so unscaled they lose a
    /// relative 4e-8 to screening and PCG stalls there (measured, H2O/6-31G).
    fn apply(&self, ctx: &ParallelContext, dd: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let scale = max_abs(dd);
        if scale == 0.0 {
            return Ok(Array2::zeros(dd.dim()));
        }
        let unit = dd / scale;
        Ok(self.apply_unscaled(ctx, &unit)? * scale)
    }

    fn apply_unscaled(
        &self,
        ctx: &ParallelContext,
        dd: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        let n = dd.nrows();
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk_with_pool(
            ctx,
            self.prep,
            &self.coul.0,
            self.thresh,
            dd,
            &mut j,
            &mut k,
            &self.coul.1,
            self.band_bytes,
        )?;
        let mut df = &j - &(0.5 * self.k_coul * &k);
        for (bounds, pool, c) in &self.exch {
            let mut jx = Array2::<f64>::zeros((n, n));
            let mut kx = Array2::<f64>::zeros((n, n));
            build_jk_with_pool(
                ctx,
                self.prep,
                bounds,
                self.thresh,
                dd,
                &mut jx,
                &mut kx,
                pool,
                self.band_bytes,
            )?;
            df = &df - &(0.5 * *c * &kx);
        }
        // Per-spin perturbation of a restricted point: δD_α = δD_β = ½ δD.
        let half = 0.5 * dd;
        let (dvxc, _) = (self.fxc.response())(&half, &half);
        Ok(&df + &dvxc)
    }

    /// Unrestricted response Fock for spin-density perturbations
    /// (`dd_a`, `dd_b`):
    /// δF_σ = δJ[δD_α + δD_β] − (c_sr δK_sr[δD_σ] + c_lr δK_lr[δD_σ]) + δV_xc^σ.
    /// J couples the spins; K and f_xc are spin-resolved. Applied to the pair
    /// divided by one common scale (it is linear in the pair), as [`Self::apply`].
    fn apply_spin(
        &self,
        ctx: &ParallelContext,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let scale = max_abs(dd_a).max(max_abs(dd_b));
        if scale == 0.0 {
            return Ok((Array2::zeros(dd_a.dim()), Array2::zeros(dd_b.dim())));
        }
        let (fa, fb) = self.apply_spin_unscaled(ctx, &(dd_a / scale), &(dd_b / scale))?;
        Ok((fa * scale, fb * scale))
    }

    fn apply_spin_unscaled(
        &self,
        ctx: &ParallelContext,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let n = dd_a.nrows();
        let jk = |bounds: &SchwarzBounds,
                  pool: &EnginePool,
                  dd: &Array2<f64>|
         -> Result<(Array2<f64>, Array2<f64>), FerricError> {
            let mut j = Array2::<f64>::zeros((n, n));
            let mut k = Array2::<f64>::zeros((n, n));
            build_jk_with_pool(
                ctx,
                self.prep,
                bounds,
                self.thresh,
                dd,
                &mut j,
                &mut k,
                pool,
                self.band_bytes,
            )?;
            Ok((j, k))
        };
        let (j_a, k_a) = jk(&self.coul.0, &self.coul.1, dd_a)?;
        let (j_b, k_b) = jk(&self.coul.0, &self.coul.1, dd_b)?;
        let dj = &j_a + &j_b;
        let mut df_a = &dj - &(self.k_coul * &k_a);
        let mut df_b = &dj - &(self.k_coul * &k_b);
        for (bounds, pool, c) in &self.exch {
            let (_, kx_a) = jk(bounds, pool, dd_a)?;
            let (_, kx_b) = jk(bounds, pool, dd_b)?;
            df_a = &df_a - &(*c * &kx_a);
            df_b = &df_b - &(*c * &kx_b);
        }
        let (dv_a, dv_b) = (self.fxc.response())(dd_a, dd_b);
        Ok((&df_a + &dv_a, &df_b + &dv_b))
    }
}

/// δD = 2 (C_v X C_oᵀ + C_o Xᵀ C_vᵀ) for an occ→virt block X (nvir, nocc).
fn ao_density_from_ov(c: &Array2<f64>, x: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let co = c.slice(s![.., ..nocc]);
    let cv = c.slice(s![.., nocc..]);
    let a = cv.dot(x).dot(&co.t());
    2.0 * (&a + &a.t())
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// What the open-shell Z-vector did about the axis-rotation null mode of a
/// linear molecule (issue #265).
///
/// A linear molecule's energy is invariant under a rotation of the electrons
/// about the molecular axis. A state that breaks that symmetry (a ²Π radical:
/// one π orbital of a spin occupied and its partner empty) turns the rotation
/// into an occ→virt orbital rotation κ_L that is an exact zero mode of the
/// orbital Hessian in the continuum; ferric's XC grid lifts it only by its
/// anisotropy (measured gap-metric Rayleigh quotients from +9.9e-3 to −7.7e-5
/// on OH, CH, NO at STO-3G, 6-31G and cc-pVDZ; the rest of the spectrum ≥ 0.57
/// in magnitude on every minimum; the softest eigenvector is κ_L to
/// |cos| ≥ 0.99999). Its sign is grid noise, so
/// PCG meets zero or negative curvature along it. The Z-vector removes it:
/// the solve runs on the complement of κ_L (Z_L := 0), which is exact in the
/// symmetric limit (a symmetric property has no right-hand side along κ_L, and
/// κ_L·∂F/∂R = −κ_L·H dκ*/dR = 0 there).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NullModeDiagnostics {
    /// κ_L·Hκ_L / κ_L·Mκ_L (M the orbital-gap preconditioner).
    pub rayleigh: f64,
    /// |b·κ_L| / (‖b‖ ‖κ_L‖) of the right-hand side b before projection.
    pub rhs_overlap: f64,
}

/// Largest |gap-metric Rayleigh quotient| of κ_L for it to count as the null
/// mode. Placed between the measured null (≤ 9.9e-3 in magnitude, OH UKS-PBE
/// STO-3G at the default (75,110) grid; −7.7e-5 at (75,302); −3.5e-3 OH
/// cc-pVDZ) and the smallest other eigenvalue magnitude measured on the same
/// systems (0.127, the genuinely negative second mode of the CH UKS saddle;
/// ≥ 0.57 on every minimum): 0.05 is 5x above the first and 2.5x below the
/// second.
pub const NULL_RAYLEIGH_TOL: f64 = 0.05;

/// Largest |b·κ̂_L|/‖b‖ accepted. A property that is NOT invariant under the
/// axis rotation has a right-hand side along κ_L, and its derivative is then
/// ill-conditioned rather than merely conventional: refused.
pub const NULL_RHS_TOL: f64 = 1e-3;

/// κ_L of `max|κ_L|` below this counts as absent (a Σ state, or both π
/// partners of every spin occupied: the rotation is occ–occ).
const KAPPA_VANISH: f64 = 1e-8;

/// Euclidean inner product over a set of blocks.
fn inner_blocks(a: &[Array2<f64>], b: &[Array2<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| inner(x, y)).sum()
}

/// x ← x − (u·x) u for a unit `u`.
fn project_out(u: &[Array2<f64>], x: &mut [Array2<f64>]) {
    let d = inner_blocks(u, x);
    for (xk, uk) in x.iter_mut().zip(u) {
        xk.scaled_add(-d, uk);
    }
}

/// The axis-rotation mode in the Z-vector parametrization: block k is
/// `left_kᵀ S Gᵀ right_k` (the occ→virt part of δC = Gᵀ C), normalized to a
/// Euclidean unit vector. `None` for a non-linear molecule or when κ_L
/// vanishes.
fn axis_rotation_mode(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    prep: &PreparedBasis,
    blocks: &[(&Array2<f64>, &Array2<f64>)],
) -> Result<Option<Vec<Array2<f64>>>, FerricError> {
    let Some(axis) = linear_axis(mol) else {
        return Ok(None);
    };
    let g = axis_rotation_generator(mol, bs, axis)?;
    let sgt = ferric_integrals::oneelectron::overlap(prep).dot(&g.t());
    let k: Vec<Array2<f64>> = blocks
        .iter()
        .map(|(l, r)| l.t().dot(&sgt).dot(*r))
        .collect();
    let kmax = k.iter().fold(0.0_f64, |m, b| m.max(max_abs(b)));
    if kmax < KAPPA_VANISH {
        return Ok(None);
    }
    let nrm = inner_blocks(&k, &k).sqrt();
    Ok(Some(k.into_iter().map(|b| b / nrm).collect()))
}

/// Accept or refuse a detected axis-rotation mode: `u` unit, `hu = H u`,
/// `gap` the preconditioner diagonals (same block layout), `rhs` the
/// unprojected right-hand side. `Ok(Some(..))`: project it out;
/// `Ok(None)`: κ_L is not near-null (an ordinary direction, nothing to do).
fn accept_null_mode(
    u: &[Array2<f64>],
    hu: &[Array2<f64>],
    gap: &[Array2<f64>],
    rhs: &[Array2<f64>],
) -> Result<Option<NullModeDiagnostics>, FerricError> {
    let uhu = inner_blocks(u, hu);
    let umu: f64 = u.iter().zip(gap).map(|(x, g)| inner(x, &(x * g))).sum();
    let rayleigh = uhu / umu;
    if rayleigh.abs() >= NULL_RAYLEIGH_TOL {
        return Ok(None);
    }
    let bn = inner_blocks(rhs, rhs).sqrt();
    let rhs_overlap = if bn == 0.0 {
        0.0
    } else {
        inner_blocks(rhs, u).abs() / bn
    };
    if rhs_overlap > NULL_RHS_TOL {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: the molecule is linear and its state breaks the cylindrical \
             symmetry, so the orbital Hessian has a null mode (rotation about the axis, \
             Rayleigh {rayleigh:.3e}); the property's right-hand side is not orthogonal to it \
             (overlap {rhs_overlap:.3e} > {NULL_RHS_TOL:.0e}), so its derivative is not \
             determined by the Z-vector equation"
        )));
    }
    Ok(Some(NullModeDiagnostics {
        rayleigh,
        rhs_overlap,
    }))
}

/// (mode to project out, its diagnostics) — see [`detect_axis_null_mode`].
type NullModeChoice = (Option<Vec<Array2<f64>>>, Option<NullModeDiagnostics>);

/// Detect and vet the axis-rotation null mode for one solve: `(Some(unit
/// mode), Some(diagnostics))` when it is to be projected out, `(None, None)`
/// otherwise. `hx` is the Hessian product on the same block layout.
fn detect_axis_null_mode(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    prep: &PreparedBasis,
    blocks: &[(&Array2<f64>, &Array2<f64>)],
    gap: &[Array2<f64>],
    rhs: &[Array2<f64>],
    hx: impl Fn(&[Array2<f64>]) -> Result<Vec<Array2<f64>>, FerricError>,
) -> Result<NullModeChoice, FerricError> {
    let Some(u) = axis_rotation_mode(mol, bs, prep, blocks)? else {
        return Ok((None, None));
    };
    let hu = hx(&u)?;
    Ok(match accept_null_mode(&u, &hu, gap, rhs)? {
        Some(d) => (Some(u), Some(d)),
        None => (None, None),
    })
}

/// [`project_out`] when a mode was accepted; a no-op otherwise.
fn project_opt(mode: &Option<Vec<Array2<f64>>>, x: &mut [Array2<f64>]) {
    if let Some(u) = mode {
        project_out(u, x);
    }
}

/// Largest perpendicular distance (Bohr) of an atom from the molecular axis
/// for the molecule to count as linear.
const LINEAR_TOL: f64 = 1e-8;

/// Unit vector along the line every atom lies on, or `None` (fewer than two
/// atoms, or not collinear to `LINEAR_TOL`).
fn linear_axis(mol: &Molecule) -> Option<[f64; 3]> {
    let pts: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let p0 = *pts.first()?;
    let rel = |p: &[f64; 3]| [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
    let nrm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let far = pts
        .iter()
        .map(rel)
        .max_by(|a, b| nrm(a).total_cmp(&nrm(b)))?;
    let d = nrm(&far);
    if d < LINEAR_TOL {
        return None;
    }
    let n = [far[0] / d, far[1] / d, far[2] / d];
    for p in &pts {
        let v = rel(p);
        let t = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let perp = [v[0] - t * n[0], v[1] - t * n[1], v[2] - t * n[2]];
        if nrm(&perp) > LINEAR_TOL {
            return None;
        }
    }
    Some(n)
}

/// AO representation `G` of the generator of a rotation of the electrons
/// about `axis` (through the nuclei, all of which lie on it):
/// `(n × (r − A))·∇χ_μ(r) = Σ_ν G_μν χ_ν(r)` for every AO μ on centre A.
/// Every shell maps onto itself (its centre is on the axis and solid
/// harmonics / Cartesian monomials of one l are closed under rotations), so
/// `G` is block-diagonal by shell. A rotated MO keeps coefficients
/// `δC = Gᵀ C`.
///
/// Each shell block is fitted from AO values and gradients at points on a
/// sphere around its centre (the radial factor cancels: (n × d)·d = 0), and
/// the fit residual is checked: the relation is exact, so a residual above
/// rounding is an error, never a silently wrong generator.
fn axis_rotation_generator(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    axis: [f64; 3],
) -> Result<Array2<f64>, FerricError> {
    use ferric_integrals::ao_grid::{collect_shells, eval_shell_and_grad};
    use ndarray_linalg::Solve;
    let shells = collect_shells(mol, bs)
        .map_err(|e| FerricError::General(format!("axis rotation generator: {e:?}")))?;
    let ncomp = |l: i32, pure: bool| -> usize {
        let l = l as usize;
        if pure {
            2 * l + 1
        } else {
            (l + 1) * (l + 2) / 2
        }
    };
    let nbf: usize = shells.iter().map(|s| ncomp(s.l, s.pure)).sum();
    // Deterministic golden-spiral directions: well spread, none on the axis
    // by construction of a generic axis.
    const NPTS: usize = 48;
    let dirs: Vec<[f64; 3]> = (0..NPTS)
        .map(|k| {
            let z = 1.0 - (2.0 * k as f64 + 1.0) / NPTS as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = k as f64 * 2.399_963_229_728_653;
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect();
    let mut g = Array2::<f64>::zeros((nbf, nbf));
    let mut off = 0usize;
    for sh in &shells {
        let m = ncomp(sh.l, sh.pure);
        if sh.l == 0 {
            off += m;
            continue;
        }
        // Radius at which the contracted radial factor is far from a node.
        let amin = sh.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
        let radius = (0.5 / amin).sqrt().clamp(0.05, 3.0);
        let mut v = Array2::<f64>::zeros((m, NPTS));
        let mut lv = Array2::<f64>::zeros((m, NPTS));
        let mut out = [0.0_f64; 15];
        let mut grad = [[0.0_f64; 15]; 3];
        for (k, u) in dirs.iter().enumerate() {
            let d = [radius * u[0], radius * u[1], radius * u[2]];
            out.iter_mut().for_each(|x| *x = 0.0);
            grad.iter_mut()
                .for_each(|row| row.iter_mut().for_each(|x| *x = 0.0));
            eval_shell_and_grad(sh, d[0], d[1], d[2], &mut out, &mut grad)
                .map_err(|e| FerricError::General(format!("axis rotation generator: {e:?}")))?;
            // n × d
            let t = [
                axis[1] * d[2] - axis[2] * d[1],
                axis[2] * d[0] - axis[0] * d[2],
                axis[0] * d[1] - axis[1] * d[0],
            ];
            for j in 0..m {
                v[(j, k)] = out[j];
                lv[(j, k)] = t[0] * grad[0][j] + t[1] * grad[1][j] + t[2] * grad[2][j];
            }
        }
        // Normalize away the common radial factor before the fit.
        let scale = max_abs(&v);
        if !(scale > 0.0) {
            return Err(FerricError::General(
                "axis rotation generator: a shell vanishes on its fitting sphere".into(),
            ));
        }
        let v = &v / scale;
        let lv = &lv / scale;
        // G_s = LV Vᵀ (V Vᵀ)⁻¹, solved column by column of Gᵀ.
        let vvt = v.dot(&v.t());
        let rhs = v.dot(&lv.t()); // (V Vᵀ) G_sᵀ = V LVᵀ
        let mut gs_t = Array2::<f64>::zeros((m, m));
        for c in 0..m {
            let col = vvt
                .solve(&rhs.column(c).to_owned())
                .map_err(|e| FerricError::Lapack(format!("axis rotation generator: {e}")))?;
            gs_t.column_mut(c).assign(&col);
        }
        let gs = gs_t.t().to_owned();
        let resid = max_abs(&(&lv - &gs.dot(&v)));
        if resid > 1e-10 * max_abs(&lv).max(1.0) {
            return Err(FerricError::General(format!(
                "axis rotation generator: shell (l = {}, pure = {}) is not closed under the axis \
                 rotation (fit residual {resid:.3e}); the AO convention is not the one assumed",
                sh.l, sh.pure
            )));
        }
        g.slice_mut(s![off..off + m, off..off + m]).assign(&gs);
        off += m;
    }
    Ok(g)
}

/// The orbital-relaxation term Σ_ai Z_ai ∂F_ai/∂R of a post-SCF quantity with
/// AO density derivative `v_ao` = ∂Q/∂D (symmetric), for the converged
/// closed-shell KS `result` of `config` on `mol`. See the module doc.
///
/// # Errors
///
/// Any [`unsupported_reason`]; a non-restricted or unconverged `result`; a
/// non-symmetric or mis-sized `v_ao`; a PCG solve that does not reach
/// [`CG_REL_TOL`] in [`CG_MAX_ITER`] iterations, unless it stops at the
/// rounding floor with max|residual| ≤ [`CG_FLOOR_TOL`] · max|rhs| (then it
/// succeeds with `stopped_at_floor = true`); any integral, kernel or
/// gradient error.
#[allow(clippy::too_many_arguments)]
pub fn relaxation_gradient_closed(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    result: &ScfResult,
    v_ao: &Array2<f64>,
) -> Result<RelaxationGradient, FerricError> {
    if let Some(r) = unsupported_reason(config) {
        return Err(FerricError::General(format!("Z-vector relaxation: {r}")));
    }
    let xc = config.xc.as_deref().unwrap_or_default();
    if !matches!(result.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF result is not closed-shell (Spin::Restricted)".into(),
        ));
    }
    if !result.converged {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF is not converged, so F_ai != 0 and the Z-vector \
             relaxation term is not the derivative of anything"
                .into(),
        ));
    }
    let c = result.mos_r();
    let n = c.nrows();
    let nmo = c.ncols();
    if v_ao.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is {:?}, expected ({n}, {n})",
            v_ao.dim()
        )));
    }
    let asym = max_abs(&(v_ao - &v_ao.t()));
    if asym > 1e-12 * max_abs(v_ao).max(1e-300) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is not symmetric (max|V - V^T| = {asym:.3e})"
        )));
    }
    let nocc = (mol.nelec() / 2) as usize;
    if nocc == 0 || nocc >= nmo {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: {nocc} occupied of {nmo} orbitals leaves no occ-virt space"
        )));
    }
    let d = result.density_r();
    let f_ao = result.fock_r();
    let f_mo = c.t().dot(f_ao).dot(c);
    let f_oo = f_mo.slice(s![..nocc, ..nocc]).to_owned();
    let f_vv = f_mo.slice(s![nocc.., nocc..]).to_owned();
    let gap = {
        let mut g = Array2::<f64>::zeros((nmo - nocc, nocc));
        for a in 0..nmo - nocc {
            for i in 0..nocc {
                g[(a, i)] = f_vv[(a, a)] - f_oo[(i, i)];
            }
        }
        g
    };
    if gap.iter().any(|&x| !(x > 0.0)) {
        return Err(FerricError::General(
            "Z-vector relaxation: a non-positive orbital-energy gap (F_aa - F_ii <= 0); the \
             reference is not an aufbau closed-shell minimum"
                .into(),
        ));
    }

    let d_half = 0.5 * d;
    let resp = ResponseFock::new(mol, prep, config, xc, &d_half, &d_half)?;
    let cv = c.slice(s![.., nocc..]).to_owned();
    let co = c.slice(s![.., ..nocc]).to_owned();
    // H x = F_vv x − x F_oo + C_vᵀ δF[δD(x)] C_o.
    let hx = |x: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let dd = ao_density_from_ov(c, x, nocc);
        let df = resp.apply(ctx, &dd)?;
        let mut h = f_vv.dot(x) - x.dot(&f_oo);
        h += &cv.t().dot(&df).dot(&co);
        Ok(h)
    };

    // H Z = −4 V_vo, PCG with the orbital-energy-gap preconditioner.
    let rhs = -4.0 * cv.t().dot(v_ao).dot(&co);
    let rhs_max = max_abs(&rhs);
    let mut z = &rhs / &gap;
    let mut iterations = 0usize;
    let mut stopped_at_floor = false;
    let mut residual;
    if rhs_max == 0.0 {
        z.fill(0.0);
        residual = 0.0;
    } else {
        let mut r = &rhs - &hx(&z)?;
        let mut p = &r / &gap;
        let mut rz = inner(&r, &p);
        residual = max_abs(&r);
        while residual > CG_REL_TOL * rhs_max {
            if iterations == CG_MAX_ITER {
                return Err(FerricError::Convergence(format!(
                    "Z-vector relaxation: PCG did not converge in {CG_MAX_ITER} iterations \
                     (max|r| = {residual:.3e}, target {:.3e})",
                    CG_REL_TOL * rhs_max
                )));
            }
            let hp = hx(&p)?;
            let php = inner(&p, &hp);
            let pmp = inner(&p, &(&p * &gap));
            if !pcg_curvature_ok(php, pmp, residual, rhs_max)? {
                stopped_at_floor = true;
                break;
            }
            let alpha = rz / php;
            z.scaled_add(alpha, &p);
            r.scaled_add(-alpha, &hp);
            let zr = &r / &gap;
            let rz_new = inner(&r, &zr);
            p = &zr + &(rz_new / rz * &p);
            rz = rz_new;
            residual = max_abs(&r);
            iterations += 1;
        }
    }

    let gradient = if max_abs(&z) == 0.0 {
        Array2::<f64>::zeros((mol.atoms.len(), 3))
    } else {
        let dd = ao_density_from_ov(c, &z, nocc);
        let df = resp.apply(ctx, &dd)?;
        let w_dot = 0.5 * (dd.dot(f_ao).dot(d) + d.dot(f_ao).dot(&dd) + d.dot(&df).dot(d));
        let nocc_w = nocc;
        let w0 = crate::gradient::build_energy_weighted_density(result, nocc_w);
        let t = T_SCALE / max_abs(&dd);
        let ext = config.external_potential.as_ref();
        let cosx =
            crate::cosx_gradient::scf_exchange_is_cosx(config, false)?.then_some(&config.cosx);
        // D(t) from the occupied orbitals rotated by t·Z and re-orthonormalized:
        // C_occ(t) = (C_occ + t C_vir Z) (I + t² ZᵀZ)^{-1/2}. It is an exact
        // projector (positive semidefinite, which the fitted-exchange gradient
        // requires: D ± t·δD is not) with dD/dt = δD_Z at t = 0, and its O(t²)
        // part is even in t, so the central difference cancels it.
        // D(t) from the occupied orbitals rotated by t·Z and re-orthonormalized
        // (see `rotated_projector`): an exact projector (positive semidefinite,
        // which the fitted-exchange gradient requires: D ± t·δD is not) with
        // dD/dt = δD_Z at t = 0, and its O(t²) part is even in t, so the
        // central difference cancels it.
        let rotated_density = |tt: f64| rotated_projector(&co, &cv, &z, tt, 2.0);
        let g_at = |sgn: f64| -> Result<Array2<f64>, FerricError> {
            let dt = rotated_density(sgn * t)?;
            let wt = &w0 + &(sgn * t * &w_dot);
            crate::ks_gradient::ks_gradient_closed_for_density(
                mol, prep, bs, op, bounds, xc, result, ext, cosx, &dt, &wt,
            )
        };
        let gp = g_at(1.0)?;
        let gm = g_at(-1.0)?;
        0.25 * (&gp - &gm) / (2.0 * t)
    };
    Ok(RelaxationGradient {
        gradient,
        z,
        iterations,
        residual,
        stopped_at_floor,
    })
}

/// The unrestricted relaxation term and its solve diagnostics.
#[derive(Debug, Clone)]
pub struct UnrestrictedRelaxationGradient {
    /// Σ_σ Σ_ai Z^σ_ai ∂F^σ_ai/∂R, (natoms, 3), in the units of Q per Bohr.
    pub gradient: Array2<f64>,
    /// Z^α, (nvir_α, nocc_α).
    pub z_alpha: Array2<f64>,
    /// Z^β, (nvir_β, nocc_β).
    pub z_beta: Array2<f64>,
    /// PCG iterations used (one per coupled α/β matvec).
    pub iterations: usize,
    /// Final max|residual| over both spins of H Z = −2 V_vo.
    pub residual: f64,
    /// `true` when PCG stopped at the rounding floor of the Hessian product
    /// (numerically zero curvature) with `residual` between `CG_REL_TOL` and
    /// `CG_FLOOR_TOL` times max|rhs| instead of reaching `CG_REL_TOL`. The
    /// relaxation term then carries a relative error of order
    /// `residual / max|rhs|` (≤ `CG_FLOOR_TOL`).
    pub stopped_at_floor: bool,
    /// The axis-rotation null mode removed from the solve (linear molecules
    /// whose state breaks the cylindrical symmetry, issue #265), or `None`
    /// when no such mode exists (every non-linear molecule; Σ states).
    pub null_mode: Option<NullModeDiagnostics>,
}

/// Why an unrestricted (UKS) Z-vector cannot be formed for `config`, or
/// `None`: every [`unsupported_reason`], plus COSX exchange (the UKS gradient
/// has no COSX form).
pub fn unsupported_reason_unrestricted(config: &RhfConfig) -> Option<String> {
    if let Some(r) = unsupported_reason(config) {
        return Some(r);
    }
    match crate::cosx_gradient::scf_exchange_is_cosx(config, true) {
        Err(e) => Some(format!("COSX exchange check failed: {e}")),
        Ok(true) => Some("COSX exchange: the UKS gradient has no COSX form".into()),
        Ok(false) => None,
    }
}

/// C_o(t) = (C_o + t C_v Z)(I + t² ZᵀZ)^{-1/2} and `occ · C_o(t) C_o(t)ᵀ`: the
/// occupied orbitals rotated by t·Z and Löwdin re-orthonormalized. An exact
/// projector (times `occ`) with d/dt = occ (C_v Z C_oᵀ + h.c.) at t = 0 and
/// an O(t²) part even in t.
fn rotated_projector(
    co: &Array2<f64>,
    cv: &Array2<f64>,
    z: &Array2<f64>,
    tt: f64,
    occ: f64,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let nocc = co.ncols();
    if nocc == 0 {
        return Ok(Array2::zeros((co.nrows(), co.nrows())));
    }
    let ztz = z.t().dot(z);
    let m = Array2::<f64>::eye(nocc) + &(tt * tt * &ztz);
    let (w, v) = m
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::Lapack(format!("Z-vector relaxation: {e}")))?;
    let mut vs = v.clone();
    for (j, wj) in w.iter().enumerate() {
        let f = 1.0 / wj.sqrt();
        vs.column_mut(j).mapv_inplace(|x| x * f);
    }
    let m_inv_sqrt = vs.dot(&v.t());
    let ct = (co + &(tt * &cv.dot(z))).dot(&m_inv_sqrt);
    Ok(occ * ct.dot(&ct.t()))
}

/// Per-spin data of a converged UKS reference used by the Z-vector.
struct SpinBlock {
    co: Array2<f64>,
    cv: Array2<f64>,
    f_ao: Array2<f64>,
    f_oo: Array2<f64>,
    f_vv: Array2<f64>,
    gap: Array2<f64>,
    /// D_σ = C_o C_oᵀ.
    d: Array2<f64>,
}

impl SpinBlock {
    fn new(
        label: &str,
        c: &Array2<f64>,
        f_ao: &Array2<f64>,
        d_scf: &Array2<f64>,
        nocc: usize,
    ) -> Result<Self, FerricError> {
        let nmo = c.ncols();
        if nocc >= nmo {
            return Err(FerricError::General(format!(
                "Z-vector relaxation: {nocc} {label} occupied of {nmo} orbitals leaves no \
                 occ-virt space"
            )));
        }
        let co = c.slice(s![.., ..nocc]).to_owned();
        let cv = c.slice(s![.., nocc..]).to_owned();
        let d = co.dot(&co.t());
        let dev = max_abs(&(&d - d_scf));
        if dev > 1e-6 * max_abs(d_scf).max(1.0) {
            return Err(FerricError::General(format!(
                "Z-vector relaxation: the {label} density is not the projector onto the first \
                 {nocc} {label} MOs (max|D - C_occ C_occ^T| = {dev:.3e}); a non-aufbau (MOM) \
                 or fractionally occupied reference is not supported"
            )));
        }
        let f_mo = c.t().dot(f_ao).dot(c);
        let f_oo = f_mo.slice(s![..nocc, ..nocc]).to_owned();
        let f_vv = f_mo.slice(s![nocc.., nocc..]).to_owned();
        let mut gap = Array2::<f64>::zeros((nmo - nocc, nocc));
        for a in 0..nmo - nocc {
            for i in 0..nocc {
                gap[(a, i)] = f_vv[(a, a)] - f_oo[(i, i)];
            }
        }
        if gap.iter().any(|&x| !(x > 0.0)) {
            return Err(FerricError::General(format!(
                "Z-vector relaxation: a non-positive {label} orbital-energy gap \
                 (F_aa - F_ii <= 0); the reference is not an aufbau UKS minimum"
            )));
        }
        Ok(Self {
            co,
            cv,
            f_ao: f_ao.clone(),
            f_oo,
            f_vv,
            gap,
            d,
        })
    }

    /// δD_σ = C_v X C_oᵀ + C_o Xᵀ C_vᵀ (unit occupation: no factor 2).
    fn density(&self, x: &Array2<f64>) -> Array2<f64> {
        let a = self.cv.dot(x).dot(&self.co.t());
        &a + &a.t()
    }
}

/// The orbital-relaxation term Σ_σ Σ_ai Z^σ_ai ∂F^σ_ai/∂R of a post-SCF
/// quantity Q that depends on the TOTAL density, with AO density derivative
/// `v_ao` = ∂Q/∂D_total (symmetric), for the converged UKS `result` of
/// `config` on `mol`. The unrestricted sibling of
/// [`relaxation_gradient_closed`]:
///
/// ```text
///   δD_σ = C_σ (κ^σ + κ^σᵀ) C_σᵀ  ⇒  dE = 2 Σ_σ Σ_ai κ^σ_ai F^σ_ai,
///                                     dQ = 2 Σ_σ Σ_ai κ^σ_ai V^σ_ai,
///   H Z = −2 V_vo,   V^σ_vo = C_σ,virᵀ V C_σ,occ   (the same V for both spins)
///   Σ_σ Z^σ_ai ∂F^σ_ai/∂R = ½ d/dt g(D_α(t), D_β(t), W + t Ẇ) |_{t=0},
///   W = Σ_σ D_σ F_σ D_σ,
///   Ẇ = Σ_σ (δD_σ F_σ D_σ + D_σ F_σ δD_σ + D_σ δF_σ D_σ),
/// ```
///
/// with H the coupled α/β orbital Hessian of [`crate::uhf_newton`] (δJ on
/// the total perturbation, δK and f_xc spin-resolved; range-separated
/// exchange as in the closed-shell response), D_σ(t) from the spin orbitals
/// rotated by t·Z^σ and re-orthonormalized, and g the UKS gradient expression
/// [`crate::ks_gradient::ks_gradient_uks_for_density`]. For a closed-shell
/// molecule run unrestricted (α = β) this is the closed-shell term: Z^σ is
/// half the closed-shell Z and both contractions differentiate the same
/// energy.
///
/// # Errors
///
/// Any [`unsupported_reason_unrestricted`]; a non-`Unrestricted` or
/// unconverged `result`; spin densities that are not the aufbau projectors of
/// the MOs; a non-positive orbital gap; a non-symmetric or mis-sized `v_ao`;
/// a PCG solve that does not reach [`CG_REL_TOL`] in [`CG_MAX_ITER`]
/// iterations, unless it stops at the rounding floor with max|residual| ≤
/// [`CG_FLOOR_TOL`] · max|rhs| (then it succeeds with `stopped_at_floor = true`); any integral, kernel or gradient error.
#[allow(clippy::too_many_arguments)]
pub fn relaxation_gradient_unrestricted(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    result: &ScfResult,
    v_ao: &Array2<f64>,
) -> Result<UnrestrictedRelaxationGradient, FerricError> {
    if let Some(r) = unsupported_reason_unrestricted(config) {
        return Err(FerricError::General(format!("Z-vector relaxation: {r}")));
    }
    let xc = config.xc.as_deref().unwrap_or_default();
    if !matches!(result.spin, Spin::Unrestricted) {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF result is not unrestricted (Spin::Unrestricted); \
             a ROKS reference takes relaxation_gradient_roks"
                .into(),
        ));
    }
    if !result.converged {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF is not converged, so F_ai != 0 and the Z-vector \
             relaxation term is not the derivative of anything"
                .into(),
        ));
    }
    let (c_b, f_b, d_b) = match (
        result.mos_beta.as_ref(),
        result.fock_beta.as_ref(),
        result.density_beta.as_ref(),
    ) {
        (Some(c), Some(f), Some(d)) => (c, f, d),
        _ => {
            return Err(FerricError::General(
                "Z-vector relaxation: the unrestricted result has no beta MOs/Fock/density".into(),
            ))
        }
    };
    let n = result.mos_alpha.nrows();
    if v_ao.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is {:?}, expected ({n}, {n})",
            v_ao.dim()
        )));
    }
    let asym = max_abs(&(v_ao - &v_ao.t()));
    if asym > 1e-12 * max_abs(v_ao).max(1e-300) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is not symmetric (max|V - V^T| = {asym:.3e})"
        )));
    }
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || (nelec - two_s) < 0 || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: {nelec} electrons with multiplicity {} is not a valid \
             spin state",
            mol.multiplicity
        )));
    }
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    if nocc_a == 0 {
        return Err(FerricError::General(
            "Z-vector relaxation: no occupied alpha orbitals".into(),
        ));
    }
    let sa = SpinBlock::new(
        "alpha",
        &result.mos_alpha,
        &result.fock_alpha,
        &result.density_alpha,
        nocc_a,
    )?;
    let sb = SpinBlock::new("beta", c_b, f_b, d_b, nocc_b)?;

    let resp = ResponseFock::new(mol, prep, config, xc, &result.density_alpha, d_b)?;
    // H (x_α, x_β): per spin F_vv x − x F_oo + C_vᵀ δF_σ[δD_α, δD_β] C_o.
    let hx = |x: &[Array2<f64>; 2]| -> Result<[Array2<f64>; 2], FerricError> {
        let (df_a, df_b) = resp.apply_spin(ctx, &sa.density(&x[0]), &sb.density(&x[1]))?;
        let h = |sp: &SpinBlock, x: &Array2<f64>, df: &Array2<f64>| {
            let mut h = sp.f_vv.dot(x) - x.dot(&sp.f_oo);
            h += &sp.cv.t().dot(df).dot(&sp.co);
            h
        };
        Ok([h(&sa, &x[0], &df_a), h(&sb, &x[1], &df_b)])
    };
    let inner2 =
        |a: &[Array2<f64>; 2], b: &[Array2<f64>; 2]| inner(&a[0], &b[0]) + inner(&a[1], &b[1]);
    let max2 = |a: &[Array2<f64>; 2]| max_abs(&a[0]).max(max_abs(&a[1]));
    let precond = |r: &[Array2<f64>; 2]| [&r[0] / &sa.gap, &r[1] / &sb.gap];

    // H Z = −2 V_vo per spin, PCG on the coupled pair.
    let mut rhs = [
        -2.0 * sa.cv.t().dot(v_ao).dot(&sa.co),
        -2.0 * sb.cv.t().dot(v_ao).dot(&sb.co),
    ];
    // Linear molecule in a symmetry-broken state: remove the axis-rotation
    // null mode (issue #265; `NullModeDiagnostics`). `None` everywhere else,
    // and then every `proj` below is a no-op (the solve is unchanged).
    let (mode, null_mode) = detect_axis_null_mode(
        mol,
        bs,
        prep,
        &[(&sa.cv, &sa.co), (&sb.cv, &sb.co)],
        &[sa.gap.clone(), sb.gap.clone()],
        &rhs,
        |u| hx(&[u[0].clone(), u[1].clone()]).map(Vec::from),
    )?;
    let proj = |x: &mut [Array2<f64>; 2]| project_opt(&mode, x);
    proj(&mut rhs);
    let rhs_max = max2(&rhs);
    let mut z = precond(&rhs);
    proj(&mut z);
    let mut iterations = 0usize;
    let mut stopped_at_floor = false;
    let mut residual;
    if rhs_max == 0.0 {
        z[0].fill(0.0);
        z[1].fill(0.0);
        residual = 0.0;
    } else {
        let hz = hx(&z)?;
        let mut r = [&rhs[0] - &hz[0], &rhs[1] - &hz[1]];
        proj(&mut r);
        let mut p = precond(&r);
        proj(&mut p);
        let mut rz = inner2(&r, &p);
        residual = max2(&r);
        while residual > CG_REL_TOL * rhs_max {
            if iterations == CG_MAX_ITER {
                return Err(FerricError::Convergence(format!(
                    "Z-vector relaxation: PCG did not converge in {CG_MAX_ITER} iterations \
                     (max|r| = {residual:.3e}, target {:.3e})",
                    CG_REL_TOL * rhs_max
                )));
            }
            let mut hp = hx(&p)?;
            proj(&mut hp);
            let php = inner2(&p, &hp);
            let pmp = inner(&p[0], &(&p[0] * &sa.gap)) + inner(&p[1], &(&p[1] * &sb.gap));
            if !pcg_curvature_ok(php, pmp, residual, rhs_max)? {
                stopped_at_floor = true;
                break;
            }
            let alpha = rz / php;
            for s in 0..2 {
                z[s].scaled_add(alpha, &p[s]);
                r[s].scaled_add(-alpha, &hp[s]);
            }
            let mut zr = precond(&r);
            proj(&mut zr);
            let rz_new = inner2(&r, &zr);
            let beta = rz_new / rz;
            p = [&zr[0] + &(beta * &p[0]), &zr[1] + &(beta * &p[1])];
            rz = rz_new;
            residual = max2(&r);
            iterations += 1;
        }
    }

    let gradient = if max2(&z) == 0.0 {
        Array2::<f64>::zeros((mol.atoms.len(), 3))
    } else {
        let dd_a = sa.density(&z[0]);
        let dd_b = sb.density(&z[1]);
        let (df_a, df_b) = resp.apply_spin(ctx, &dd_a, &dd_b)?;
        let w_dot_spin = |sp: &SpinBlock, dd: &Array2<f64>, df: &Array2<f64>| {
            dd.dot(&sp.f_ao).dot(&sp.d) + sp.d.dot(&sp.f_ao).dot(dd) + sp.d.dot(df).dot(&sp.d)
        };
        let w_dot = w_dot_spin(&sa, &dd_a, &df_a) + w_dot_spin(&sb, &dd_b, &df_b);
        let w0 = crate::gradient::build_energy_weighted_density_uhf(result, nocc_a, nocc_b);
        let t = T_SCALE / max_abs(&dd_a).max(max_abs(&dd_b));
        let ext = config.external_potential.as_ref();
        let g_at = |sgn: f64| -> Result<Array2<f64>, FerricError> {
            let da = rotated_projector(&sa.co, &sa.cv, &z[0], sgn * t, 1.0)?;
            let db = rotated_projector(&sb.co, &sb.cv, &z[1], sgn * t, 1.0)?;
            let wt = &w0 + &(sgn * t * &w_dot);
            crate::ks_gradient::ks_gradient_uks_for_density(
                mol, prep, bs, op, bounds, xc, result, ext, &da, &db, &wt,
            )
        };
        let gp = g_at(1.0)?;
        let gm = g_at(-1.0)?;
        0.5 * (&gp - &gm) / (2.0 * t)
    };
    let [z_alpha, z_beta] = z;
    Ok(UnrestrictedRelaxationGradient {
        gradient,
        z_alpha,
        z_beta,
        iterations,
        residual,
        stopped_at_floor,
        null_mode,
    })
}

/// The ROKS relaxation term and its solve diagnostics.
#[derive(Debug, Clone)]
pub struct RestrictedOpenRelaxationGradient {
    /// Z·∂g/∂R summed over the three rotation blocks, (natoms, 3), in the
    /// units of Q per Bohr.
    pub gradient: Array2<f64>,
    /// Z_vc, closed → virtual, (nvir, nclosed).
    pub z_vc: Array2<f64>,
    /// Z_vo, open → virtual, (nvir, nopen).
    pub z_vo: Array2<f64>,
    /// Z_oc, closed → open, (nopen, nclosed).
    pub z_oc: Array2<f64>,
    /// PCG iterations used (one per three-block matvec).
    pub iterations: usize,
    /// Final max|residual| over the three blocks of
    /// H Z = −(4 V_vc, 2 V_vo, 2 V_oc).
    pub residual: f64,
    /// `true` when PCG stopped at the rounding floor of the Hessian product
    /// with `residual` between `CG_REL_TOL` and `CG_FLOOR_TOL` times max|rhs|
    /// (see `RelaxationGradient::stopped_at_floor`).
    pub stopped_at_floor: bool,
    /// The axis-rotation null mode removed from the solve (linear molecules
    /// whose state breaks the cylindrical symmetry, issue #265), or `None`
    /// when no such mode exists (every non-linear molecule; Σ states).
    pub null_mode: Option<NullModeDiagnostics>,
}

/// Why a restricted open-shell (ROKS) Z-vector cannot be formed for
/// `config`, or `None`: every [`unsupported_reason`], plus COSX exchange (the
/// ROKS gradient has no COSX form). Nothing ROKS-specific beyond that: the
/// limits of `rohf_newton::hessian_matvec` (diagonal Fock entries
/// only, range-separated exchange dropped) do not apply, because the Z-vector
/// builds its own exact product (see the module doc) with the same response
/// Fock as the closed-shell and UKS paths, range-separated exchange included.
pub fn unsupported_reason_roks(config: &RhfConfig) -> Option<String> {
    if let Some(r) = unsupported_reason(config) {
        return Some(r);
    }
    match crate::cosx_gradient::scf_exchange_is_cosx(config, true) {
        Err(e) => Some(format!("COSX exchange check failed: {e}")),
        Ok(true) => Some("COSX exchange: the ROKS gradient has no COSX form".into()),
        Ok(false) => None,
    }
}

/// The three ROKS rotation blocks (vc, vo, oc) as one PCG vector.
type Blocks3 = [Array2<f64>; 3];

/// Converged ROKS data the Z-vector needs: the shared MOs split by block and
/// the MO spin-Fock blocks of the exact Hessian (module doc, (b)).
struct RoksBlocks {
    cc: Array2<f64>,
    co: Array2<f64>,
    cv: Array2<f64>,
    /// F⁺ = F_α + F_β blocks.
    fs_vv: Array2<f64>,
    fs_cc: Array2<f64>,
    fa_vv: Array2<f64>,
    fa_oo: Array2<f64>,
    fa_oc: Array2<f64>,
    fb_oo: Array2<f64>,
    fb_cc: Array2<f64>,
    fb_vo: Array2<f64>,
    /// G_vc = ½ (F_β − F_α)_vc.
    g_vc: Array2<f64>,
    /// Diagonal preconditioner per block.
    diag: Blocks3,
}

impl RoksBlocks {
    /// δD_α = C_v x_vc C_cᵀ + C_v x_vo C_oᵀ + h.c.,
    /// δD_β = C_v x_vc C_cᵀ + C_o x_oc C_cᵀ + h.c.
    fn densities(&self, x: &Blocks3) -> (Array2<f64>, Array2<f64>) {
        let vc = self.cv.dot(&x[0]).dot(&self.cc.t());
        let vo = self.cv.dot(&x[1]).dot(&self.co.t());
        let oc = self.co.dot(&x[2]).dot(&self.cc.t());
        let a = &vc + &vo;
        let b = &vc + &oc;
        (&a + &a.t(), &b + &b.t())
    }

    /// H x from the response spin Focks (δF_α, δF_β) of `densities(x)`.
    fn product(&self, x: &Blocks3, df_a: &Array2<f64>, df_b: &Array2<f64>) -> Blocks3 {
        let (x_vc, x_vo, x_oc) = (&x[0], &x[1], &x[2]);
        let df_s = df_a + df_b;
        let mut h_vc = self.fs_vv.dot(x_vc) - x_vc.dot(&self.fs_cc);
        h_vc += &self.fb_vo.dot(x_oc);
        h_vc -= &x_vo.dot(&self.fa_oc);
        h_vc += &self.cv.t().dot(&df_s).dot(&self.cc);
        let mut h_vo = self.fa_vv.dot(x_vo) - x_vo.dot(&self.fa_oo);
        h_vo += &self.g_vc.dot(&x_oc.t());
        h_vo -= &x_vc.dot(&self.fa_oc.t());
        h_vo += &self.cv.t().dot(df_a).dot(&self.co);
        let mut h_oc = self.fb_oo.dot(x_oc) - x_oc.dot(&self.fb_cc);
        h_oc += &self.fb_vo.t().dot(x_vc);
        h_oc += &x_vo.t().dot(&self.g_vc);
        h_oc += &self.co.t().dot(df_b).dot(&self.cc);
        [h_vc, h_vo, h_oc]
    }
}

/// The orbital-relaxation term Z·∂g/∂R of a post-SCF quantity Q that depends
/// on the TOTAL density, with AO density derivative `v_ao` = ∂Q/∂D_total
/// (symmetric), for the converged ROKS `result` of `config` on `mol` (high
/// spin: 2S open orbitals, all α). The restricted open-shell sibling of
/// [`relaxation_gradient_closed`] and [`relaxation_gradient_unrestricted`];
/// see the module doc for the derivation:
///
/// ```text
///   H Z = −(4 V_vc, 2 V_vo, 2 V_oc),
///   Z·∂g/∂R = ½ d/dt g_ROKS(D_α(t), D_β(t), W + t Ẇ) |_{t=0},
///   W = D_α F_α D_α + P_c F_β P_c + ½ (P_c F_β P_o + P_o F_β P_c),
/// ```
///
/// with H the exact three-block ROKS orbital Hessian, D_σ(t) from the shared
/// orbitals rotated by the Cayley transform of t·Z, and g_ROKS
/// [`crate::ks_gradient::ks_gradient_roks_for_density`]. The "held fixed"
/// path this term completes re-orthonormalizes closed and open orbitals as
/// one set (module doc, (a)).
///
/// # Errors
///
/// Any [`unsupported_reason_roks`]; a non-`RestrictedOpen` or unconverged
/// `result`, or one without its spin Focks (`rohf_spin_focks`); spin
/// densities that are not the projectors onto the first closed (+ open) MOs;
/// a non-positive diagonal of the Hessian (not an aufbau ROKS minimum); a
/// non-symmetric or mis-sized `v_ao`; a PCG solve that does not reach
/// [`CG_REL_TOL`] in [`CG_MAX_ITER`] iterations or meets non-positive
/// curvature; any integral, kernel or gradient error.
#[allow(clippy::too_many_arguments)]
pub fn relaxation_gradient_roks(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    result: &ScfResult,
    v_ao: &Array2<f64>,
) -> Result<RestrictedOpenRelaxationGradient, FerricError> {
    if let Some(r) = unsupported_reason_roks(config) {
        return Err(FerricError::General(format!("Z-vector relaxation: {r}")));
    }
    let xc = config.xc.as_deref().unwrap_or_default();
    if !matches!(result.spin, Spin::RestrictedOpen) {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF result is not restricted open-shell \
             (Spin::RestrictedOpen)"
                .into(),
        ));
    }
    if !result.converged {
        return Err(FerricError::General(
            "Z-vector relaxation: the SCF is not converged, so the ROKS orbital gradient is \
             not zero and the Z-vector relaxation term is not the derivative of anything"
                .into(),
        ));
    }
    let (f_a, f_b) = result.rohf_spin_focks.as_ref().ok_or_else(|| {
        FerricError::General(
            "Z-vector relaxation: the ROKS result carries no spin Fock matrices \
             (rohf_spin_focks); produce it with solve_rohf"
                .into(),
        )
    })?;
    let d_b = result.density_beta.as_ref().ok_or_else(|| {
        FerricError::General("Z-vector relaxation: the ROKS result has no beta density".into())
    })?;
    let d_a = &result.density_alpha;
    let c = &result.mos_alpha;
    let n = c.nrows();
    let nmo = c.ncols();
    if v_ao.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is {:?}, expected ({n}, {n})",
            v_ao.dim()
        )));
    }
    let asym = max_abs(&(v_ao - &v_ao.t()));
    if asym > 1e-12 * max_abs(v_ao).max(1e-300) {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: v_ao is not symmetric (max|V - V^T| = {asym:.3e})"
        )));
    }
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || (nelec - two_s) < 0 || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: {nelec} electrons with multiplicity {} is not a valid \
             spin state",
            mol.multiplicity
        )));
    }
    let nc = ((nelec - two_s) / 2) as usize;
    let no = two_s as usize;
    let na = nc + no;
    if na == 0 || na >= nmo {
        return Err(FerricError::General(format!(
            "Z-vector relaxation: {na} occupied of {nmo} orbitals leaves no occ-virt space"
        )));
    }
    let cc = c.slice(s![.., ..nc]).to_owned();
    let co = c.slice(s![.., nc..na]).to_owned();
    let cv = c.slice(s![.., na..]).to_owned();
    let p_c = cc.dot(&cc.t());
    let p_o = co.dot(&co.t());
    for (label, d, proj) in [("alpha", d_a, &p_c + &p_o), ("beta", d_b, p_c.clone())] {
        let dev = max_abs(&(&proj - d));
        if dev > 1e-6 * max_abs(d).max(1.0) {
            return Err(FerricError::General(format!(
                "Z-vector relaxation: the ROKS {label} density is not the projector onto the \
                 first {} MOs (max|D - C C^T| = {dev:.3e}); a non-aufbau (MOM) reference is \
                 not supported",
                if label == "alpha" { na } else { nc }
            )));
        }
    }
    let fa_mo = c.t().dot(f_a).dot(c);
    let fb_mo = c.t().dot(f_b).dot(c);
    let blk = |m: &Array2<f64>, r: std::ops::Range<usize>, q: std::ops::Range<usize>| {
        m.slice(s![r, q]).to_owned()
    };
    let (rc, ro, rv) = (0..nc, nc..na, na..nmo);
    let fs_mo = &fa_mo + &fb_mo;
    let fs_vv = blk(&fs_mo, rv.clone(), rv.clone());
    let fs_cc = blk(&fs_mo, rc.clone(), rc.clone());
    let fa_vv = blk(&fa_mo, rv.clone(), rv.clone());
    let fa_oo = blk(&fa_mo, ro.clone(), ro.clone());
    let fb_oo = blk(&fb_mo, ro.clone(), ro.clone());
    let fb_cc = blk(&fb_mo, rc.clone(), rc.clone());
    let diag_of = |vv: &Array2<f64>, oo: &Array2<f64>| {
        let mut d = Array2::<f64>::zeros((vv.nrows(), oo.nrows()));
        for a in 0..vv.nrows() {
            for i in 0..oo.nrows() {
                d[(a, i)] = vv[(a, a)] - oo[(i, i)];
            }
        }
        d
    };
    let diag = [
        diag_of(&fs_vv, &fs_cc),
        diag_of(&fa_vv, &fa_oo),
        diag_of(&fb_oo, &fb_cc),
    ];
    for (label, d) in ["closed->virtual", "open->virtual", "closed->open"]
        .iter()
        .zip(&diag)
    {
        if d.iter().any(|&x| !(x > 0.0)) {
            return Err(FerricError::General(format!(
                "Z-vector relaxation: a non-positive {label} diagonal of the ROKS orbital \
                 Hessian; the reference is not an aufbau ROKS minimum"
            )));
        }
    }
    let rb = RoksBlocks {
        fa_oc: blk(&fa_mo, ro.clone(), rc.clone()),
        fb_vo: blk(&fb_mo, rv.clone(), ro.clone()),
        g_vc: 0.5 * (&blk(&fb_mo, rv.clone(), rc.clone()) - &blk(&fa_mo, rv, rc)),
        cc,
        co,
        cv,
        fs_vv,
        fs_cc,
        fa_vv,
        fa_oo,
        fb_oo,
        fb_cc,
        diag,
    };

    let resp = ResponseFock::new(mol, prep, config, xc, d_a, d_b)?;
    let hx = |x: &Blocks3| -> Result<Blocks3, FerricError> {
        let (dd_a, dd_b) = rb.densities(x);
        let (df_a, df_b) = resp.apply_spin(ctx, &dd_a, &dd_b)?;
        Ok(rb.product(x, &df_a, &df_b))
    };
    let inner3 = |a: &Blocks3, b: &Blocks3| (0..3).map(|k| inner(&a[k], &b[k])).sum::<f64>();
    let max3 = |a: &Blocks3| (0..3).fold(0.0_f64, |m, k| m.max(max_abs(&a[k])));
    let precond = |r: &Blocks3| -> Blocks3 {
        [
            &r[0] / &rb.diag[0],
            &r[1] / &rb.diag[1],
            &r[2] / &rb.diag[2],
        ]
    };

    // H Z = −(4 V_vc, 2 V_vo, 2 V_oc).
    let mut rhs: Blocks3 = [
        -4.0 * rb.cv.t().dot(v_ao).dot(&rb.cc),
        -2.0 * rb.cv.t().dot(v_ao).dot(&rb.co),
        -2.0 * rb.co.t().dot(v_ao).dot(&rb.cc),
    ];
    // Axis-rotation null mode of a linear molecule (issue #265), as in the
    // UKS solve; `None` (no-op projections) everywhere else.
    let (mode, null_mode) = detect_axis_null_mode(
        mol,
        bs,
        prep,
        &[(&rb.cv, &rb.cc), (&rb.cv, &rb.co), (&rb.co, &rb.cc)],
        &rb.diag,
        &rhs,
        |u| hx(&[u[0].clone(), u[1].clone(), u[2].clone()]).map(Vec::from),
    )?;
    let proj = |x: &mut Blocks3| project_opt(&mode, x);
    proj(&mut rhs);
    let rhs_max = max3(&rhs);
    let mut z = precond(&rhs);
    proj(&mut z);
    let mut iterations = 0usize;
    let mut stopped_at_floor = false;
    let mut residual;
    if rhs_max == 0.0 {
        for zk in &mut z {
            zk.fill(0.0);
        }
        residual = 0.0;
    } else {
        let hz = hx(&z)?;
        let mut r: Blocks3 = [&rhs[0] - &hz[0], &rhs[1] - &hz[1], &rhs[2] - &hz[2]];
        proj(&mut r);
        let mut p = precond(&r);
        proj(&mut p);
        let mut rz = inner3(&r, &p);
        residual = max3(&r);
        while residual > CG_REL_TOL * rhs_max {
            if iterations == CG_MAX_ITER {
                return Err(FerricError::Convergence(format!(
                    "Z-vector relaxation: PCG did not converge in {CG_MAX_ITER} iterations \
                     (max|r| = {residual:.3e}, target {:.3e})",
                    CG_REL_TOL * rhs_max
                )));
            }
            let mut hp = hx(&p)?;
            proj(&mut hp);
            let php = inner3(&p, &hp);
            let pmp: f64 = (0..3).map(|k| inner(&p[k], &(&p[k] * &rb.diag[k]))).sum();
            if !pcg_curvature_ok(php, pmp, residual, rhs_max)? {
                stopped_at_floor = true;
                break;
            }
            let alpha = rz / php;
            for k in 0..3 {
                z[k].scaled_add(alpha, &p[k]);
                r[k].scaled_add(-alpha, &hp[k]);
            }
            let mut zr = precond(&r);
            proj(&mut zr);
            let rz_new = inner3(&r, &zr);
            let beta = rz_new / rz;
            p = [
                &zr[0] + &(beta * &p[0]),
                &zr[1] + &(beta * &p[1]),
                &zr[2] + &(beta * &p[2]),
            ];
            rz = rz_new;
            residual = max3(&r);
            iterations += 1;
        }
    }

    let gradient = if max3(&z) == 0.0 {
        Array2::<f64>::zeros((mol.atoms.len(), 3))
    } else {
        let (dd_a, dd_b) = rb.densities(&z);
        let (df_a, df_b) = resp.apply_spin(ctx, &dd_a, &dd_b)?;
        let w0 = roks_energy_weighted_density(&p_c, &p_o, f_a, f_b);
        let w_dot = roks_energy_weighted_density_derivative(
            &p_c, &p_o, f_a, f_b, &dd_a, &dd_b, &df_a, &df_b,
        );
        let t = T_SCALE / max_abs(&dd_a).max(max_abs(&dd_b));
        let ext = config.external_potential.as_ref();
        let kappa = roks_kappa(&z, nc, no, nmo);
        let g_at = |sgn: f64| -> Result<Array2<f64>, FerricError> {
            let ct = c.dot(&cayley(&kappa, sgn * t)?);
            let cct = ct.slice(s![.., ..nc]);
            let cat = ct.slice(s![.., ..na]);
            let da = cat.dot(&cat.t());
            let db = cct.dot(&cct.t());
            let wt = &w0 + &(sgn * t * &w_dot);
            crate::ks_gradient::ks_gradient_roks_for_density(
                mol, prep, bs, op, bounds, xc, result, ext, &da, &db, &wt,
            )
        };
        let gp = g_at(1.0)?;
        let gm = g_at(-1.0)?;
        0.5 * (&gp - &gm) / (2.0 * t)
    };
    let [z_vc, z_vo, z_oc] = z;
    Ok(RestrictedOpenRelaxationGradient {
        gradient,
        z_vc,
        z_vo,
        z_oc,
        iterations,
        residual,
        stopped_at_floor,
        null_mode,
    })
}

/// W = D_α F_α D_α + P_c F_β P_c + ½ (P_c F_β P_o + P_o F_β P_c), D_α = P_c +
/// P_o: the energy-weighted density of the ROKS energy on the path that
/// re-orthonormalizes closed and open orbitals as one set (module doc, (a)).
fn roks_energy_weighted_density(
    p_c: &Array2<f64>,
    p_o: &Array2<f64>,
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
) -> Array2<f64> {
    let d_a = p_c + p_o;
    let cross = p_c.dot(f_b).dot(p_o);
    d_a.dot(f_a).dot(&d_a) + p_c.dot(f_b).dot(p_c) + 0.5 * (&cross + &cross.t())
}

/// First-order change of [`roks_energy_weighted_density`] for spin-density
/// perturbations (δD_α, δD_β) = (δP_c + δP_o, δP_c) and response spin Focks
/// (δF_α, δF_β).
#[allow(clippy::too_many_arguments)]
fn roks_energy_weighted_density_derivative(
    p_c: &Array2<f64>,
    p_o: &Array2<f64>,
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    dd_a: &Array2<f64>,
    dd_b: &Array2<f64>,
    df_a: &Array2<f64>,
    df_b: &Array2<f64>,
) -> Array2<f64> {
    let d_a = p_c + p_o;
    let dp_c = dd_b;
    let dp_o = dd_a - dd_b;
    let alpha = dd_a.dot(f_a).dot(&d_a) + d_a.dot(f_a).dot(dd_a) + d_a.dot(df_a).dot(&d_a);
    let closed = dp_c.dot(f_b).dot(p_c) + p_c.dot(f_b).dot(dp_c) + p_c.dot(df_b).dot(p_c);
    let cross = dp_c.dot(f_b).dot(p_o) + p_c.dot(df_b).dot(p_o) + p_c.dot(f_b).dot(&dp_o);
    alpha + closed + 0.5 * (&cross + &cross.t())
}

/// The antisymmetric MO rotation generator of the ROKS blocks of `z`
/// (κ\[v,c\] = Z_vc, κ\[v,o\] = Z_vo, κ\[o,c\] = Z_oc, κ = −κᵀ).
fn roks_kappa(z: &Blocks3, nc: usize, no: usize, nmo: usize) -> Array2<f64> {
    let na = nc + no;
    let mut k = Array2::<f64>::zeros((nmo, nmo));
    k.slice_mut(s![na.., ..nc]).assign(&z[0]);
    k.slice_mut(s![na.., nc..na]).assign(&z[1]);
    k.slice_mut(s![nc..na, ..nc]).assign(&z[2]);
    &k - &k.t()
}

/// The Cayley rotation (I − tκ/2)⁻¹(I + tκ/2) of an antisymmetric κ: exactly
/// orthogonal, = I + tκ + O(t²), and its inverse is its value at −t.
fn cayley(kappa: &Array2<f64>, tt: f64) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{FactorizeInto, Solve};
    let n = kappa.nrows();
    let eye = Array2::<f64>::eye(n);
    let a = &eye - &(0.5 * tt * kappa);
    let b = &eye + &(0.5 * tt * kappa);
    // One LU of `a`, reused for every column (a fresh `a.solve` per column
    // would refactorize it nmo times).
    let lu = a
        .factorize_into()
        .map_err(|e| FerricError::Lapack(format!("Z-vector relaxation: Cayley LU: {e}")))?;
    let mut u = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let col = lu
            .solve(&b.column(j).to_owned())
            .map_err(|e| FerricError::Lapack(format!("Z-vector relaxation: Cayley solve: {e}")))?;
        u.column_mut(j).assign(&col);
    }
    Ok(u)
}

#[cfg(test)]
mod pcg_curvature_tests {
    use super::*;

    /// Positive curvature proceeds; clearly negative curvature is an error
    /// (the SCF is not a minimum); numerically zero curvature stops PCG and is
    /// accepted only when the residual is already within `CG_FLOOR_TOL`.
    /// The zero-curvature case is the rounding floor that a raw `php > 0`
    /// test misread as an indefinite Hessian (UKS OH, p.Hp = -8.4e-19).
    #[test]
    fn curvature_verdicts() {
        // proceed
        assert!(pcg_curvature_ok(1.0, 1.0, 1.0, 1.0).unwrap());
        // negative curvature with a large residual: indefinite Hessian
        assert!(matches!(
            pcg_curvature_ok(-0.5, 1.0, 1.0, 1.0),
            Err(FerricError::General(_))
        ));
        // the measured rounding floor, residual within CG_FLOOR_TOL: accept
        assert!(!pcg_curvature_ok(-8.4e-19, 1.1e-16, 1e-12, 1e-3).unwrap());
        // zero curvature with a large residual: stall
        assert!(matches!(
            pcg_curvature_ok(0.0, 1.0, 1e-6, 1e-3),
            Err(FerricError::Convergence(_))
        ));
    }
}

/// Issue #265 investigation (measurement, prints): the dense open-shell
/// orbital Hessian of small radicals, its spectrum in the gap metric, and the
/// alignment of its softest eigenvector with the axis-rotation mode κ_L.
/// Pre-registered predictions: tests/HYPOTHESES-degenerate-somo-zvector.md.
#[cfg(test)]
mod degenerate_somo_investigation {
    use super::*;
    use ndarray_linalg::{Eigh, UPLO};

    fn flatten(blocks: &[Array2<f64>]) -> Vec<f64> {
        blocks.iter().flat_map(|b| b.iter().cloned()).collect()
    }

    fn unflatten(v: &[f64], shapes: &[(usize, usize)]) -> Vec<Array2<f64>> {
        let mut off = 0;
        shapes
            .iter()
            .map(|&(r, c)| {
                let a = Array2::from_shape_vec((r, c), v[off..off + r * c].to_vec()).unwrap();
                off += r * c;
                a
            })
            .collect()
    }

    /// (dense H, gap diagonal, κ_L flattened or None) for a converged result.
    fn dense(
        mol: &Molecule,
        bs: &ferric_core::basis::BasisSet,
        prep: &PreparedBasis,
        cfg: &RhfConfig,
        r: &ScfResult,
    ) -> (Array2<f64>, Vec<f64>, Option<Vec<f64>>) {
        let ctx = ParallelContext::default();
        let xc = cfg.xc.as_deref().unwrap();
        let s_ao = ferric_integrals::oneelectron::overlap(prep);
        let gen = linear_axis(mol).map(|ax| axis_rotation_generator(mol, bs, ax).expect("G"));
        match r.spin {
            Spin::Unrestricted => {
                let nelec = mol.nelec() as i64;
                let two_s = mol.multiplicity as i64 - 1;
                let na = ((nelec + two_s) / 2) as usize;
                let nb = ((nelec - two_s) / 2) as usize;
                let db = r.density_beta.as_ref().unwrap();
                let sa =
                    SpinBlock::new("a", &r.mos_alpha, &r.fock_alpha, &r.density_alpha, na).unwrap();
                let sb = SpinBlock::new(
                    "b",
                    r.mos_beta.as_ref().unwrap(),
                    r.fock_beta.as_ref().unwrap(),
                    db,
                    nb,
                )
                .unwrap();
                let resp = ResponseFock::new(mol, prep, cfg, xc, &r.density_alpha, db).unwrap();
                let shapes = [sa.gap.dim(), sb.gap.dim()];
                let dim: usize = shapes.iter().map(|(a, b)| a * b).sum();
                let mut h = Array2::<f64>::zeros((dim, dim));
                for j in 0..dim {
                    let mut e = vec![0.0; dim];
                    e[j] = 1.0;
                    let x = unflatten(&e, &shapes);
                    let (dfa, dfb) = resp
                        .apply_spin(&ctx, &sa.density(&x[0]), &sb.density(&x[1]))
                        .unwrap();
                    let hb = |sp: &SpinBlock, x: &Array2<f64>, df: &Array2<f64>| {
                        let mut hh = sp.f_vv.dot(x) - x.dot(&sp.f_oo);
                        hh += &sp.cv.t().dot(df).dot(&sp.co);
                        hh
                    };
                    let col = flatten(&[hb(&sa, &x[0], &dfa), hb(&sb, &x[1], &dfb)]);
                    for i in 0..dim {
                        h[(i, j)] = col[i];
                    }
                }
                let gap = flatten(&[sa.gap.clone(), sb.gap.clone()]);
                let kl = gen.map(|g| {
                    let k = |sp: &SpinBlock| sp.cv.t().dot(&s_ao).dot(&g.t()).dot(&sp.co);
                    flatten(&[k(&sa), k(&sb)])
                });
                (h, gap, kl)
            }
            _ => unreachable!("UKS only here"),
        }
    }

    fn report(label: &str, h: &Array2<f64>, gap: &[f64], kl: Option<&Vec<f64>>) {
        let dim = gap.len();
        let asym = max_abs(&(h - &h.t()));
        let hs = 0.5 * (h + &h.t());
        // Gap metric: M^{-1/2} H M^{-1/2}.
        let mut hm = hs.clone();
        for i in 0..dim {
            for j in 0..dim {
                hm[(i, j)] /= (gap[i] * gap[j]).sqrt();
            }
        }
        let (w, v) = hm.eigh(UPLO::Lower).unwrap();
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| w[a].abs().total_cmp(&w[b].abs()));
        let soft = order[0];
        println!(
            "{label}: dim {dim}, max|H - H^T| {asym:.2e}; gap-metric eigenvalues (sorted by |.|): \
             {:?}",
            order
                .iter()
                .take(4)
                .map(|&k| format!("{:.3e}", w[k]))
                .collect::<Vec<_>>()
        );
        match kl {
            None => println!("{label}: not linear, no axis-rotation mode"),
            Some(k) => {
                let kn: f64 = k.iter().map(|x| x * x).sum::<f64>().sqrt();
                if kn < 1e-10 {
                    println!("{label}: kappa_L vanishes (|kappa_L| = {kn:.2e})");
                    return;
                }
                // Rayleigh in the gap metric and cosine with the softest
                // eigenvector (both in the M^{1/2}-scaled coordinates).
                let ks: Vec<f64> = k.iter().zip(gap).map(|(x, g)| x * g.sqrt()).collect();
                let ksn: f64 = ks.iter().map(|x| x * x).sum::<f64>().sqrt();
                let hk: Vec<f64> = (0..dim)
                    .map(|i| (0..dim).map(|j| hs[(i, j)] * k[j]).sum())
                    .collect();
                let khk: f64 = k.iter().zip(&hk).map(|(a, b)| a * b).sum();
                let kmk: f64 = k.iter().zip(gap).map(|(a, g)| a * a * g).sum();
                let cos: f64 = (0..dim).map(|i| v[(i, soft)] * ks[i]).sum::<f64>().abs() / ksn;
                let hk_rel = hk.iter().map(|x| x.abs()).fold(0.0, f64::max)
                    / k.iter().map(|x| x.abs()).fold(0.0, f64::max);
                println!(
                    "{label}: |kappa_L| {kn:.3e}; Rayleigh kHk/kMk = {:.3e}; max|H k|/max|k| = \
                     {hk_rel:.3e}; |cos(softest, kappa_L)| = {cos:.6}",
                    khk / kmk
                );
            }
        }
    }

    fn uks(xyz: &str, mult: usize, basis: &str, xc: &str, ang: usize) {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = ferric_core::basis::bundled(basis).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let cfg = RhfConfig {
            xc: Some(xc.into()),
            max_iter: 300,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            dft_grid: Some(ferric_dft::grid::AtomicGridConfig {
                n_radial: 75,
                n_angular: ang,
                prune: None,
            }),
            ..Default::default()
        };
        let r =
            crate::uhf::solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
        assert!(r.converged);
        let (h, gap, kl) = dense(&mol, &bs, &prep, &cfg, &r);
        report(
            &format!(
                "{} {basis} UKS-{xc} (75,{ang})",
                xyz.lines().nth(1).unwrap()
            ),
            &h,
            &gap,
            kl.as_ref(),
        );
    }

    const OH: &str = "2\nOH\nO 0.000000 0.000000 0.000000\nH 0.100000 0.000000 0.970000\n";
    const NH2: &str =
        "3\nNH2\nN 0.000000 0.000000 0.142000\nH 0.000000 0.802000 -0.497000\nH 0.000000 -0.802000 -0.497000\n";
    const O2: &str = "2\nO2\nO 0.000000 0.000000 0.604000\nO 0.000000 0.000000 -0.604000\n";
    const CH: &str = "2\nCH\nC 0.000000 0.000000 0.000000\nH 0.000000 0.300000 1.080000\n";
    const NO: &str = "2\nNO\nN 0.000000 0.000000 0.000000\nO 0.200000 0.000000 1.140000\n";

    /// (Rayleigh quotient of κ_L in the gap metric, |cos(softest eigenvector,
    /// κ_L)|, smallest |eigenvalue| among the OTHER directions), or `None`
    /// when no axis-rotation mode exists.
    fn null_mode_numbers(xyz: &str, mult: usize, basis: &str, xc: &str) -> Option<(f64, f64, f64)> {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = ferric_core::basis::bundled(basis).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let cfg = RhfConfig {
            xc: Some(xc.into()),
            max_iter: 300,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..Default::default()
        };
        let r =
            crate::uhf::solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
        assert!(r.converged);
        let (h, gap, kl) = dense(&mol, &bs, &prep, &cfg, &r);
        let k = kl?;
        if k.iter().fold(0.0_f64, |m, x| m.max(x.abs())) < KAPPA_VANISH {
            return None;
        }
        let dim = gap.len();
        let hs = 0.5 * (&h + &h.t());
        let mut hm = hs.clone();
        for i in 0..dim {
            for j in 0..dim {
                hm[(i, j)] /= (gap[i] * gap[j]).sqrt();
            }
        }
        let (w, v) = hm.eigh(UPLO::Lower).unwrap();
        let ks: Vec<f64> = k.iter().zip(&gap).map(|(x, g)| x * g.sqrt()).collect();
        let ksn: f64 = ks.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cosines: Vec<f64> = (0..dim)
            .map(|e| (0..dim).map(|i| v[(i, e)] * ks[i]).sum::<f64>().abs() / ksn)
            .collect();
        let soft = (0..dim)
            .max_by(|&a, &b| cosines[a].total_cmp(&cosines[b]))
            .unwrap();
        let other = (0..dim)
            .filter(|&e| e != soft)
            .map(|e| w[e].abs())
            .fold(f64::INFINITY, f64::min);
        let khk: f64 = (0..dim)
            .map(|i| k[i] * (0..dim).map(|j| hs[(i, j)] * k[j]).sum::<f64>())
            .sum();
        let kmk: f64 = k.iter().zip(&gap).map(|(a, g)| a * a * g).sum();
        Some((khk / kmk, cosines[soft], other))
    }

    /// INDEPENDENT CONSTRUCTION CHECK of the generator and of the detection
    /// bar: the eigenvector of the dense UKS orbital Hessian (built by unit
    /// matvecs, no symmetry input) that best matches κ_L must match it to
    /// |cos| > 0.9999, its Rayleigh quotient must lie under
    /// `NULL_RAYLEIGH_TOL`, and every OTHER eigenvalue must lie above it, so
    /// the bar separates the two sides on these systems (s/p bases and the
    /// pure-d cc-pVDZ, where the generator acts on solid harmonics). A wrong
    /// generator (sign or ordering convention, wrong axis) gives |cos| ≪ 1.
    /// NH2 (bent) and O2 ³Σg⁻ (both π partners of each spin filled) have no
    /// mode.
    #[test]
    fn axis_mode_is_the_softest_hessian_direction() {
        for (xyz, mult, basis, xc) in [
            (OH, 2, "sto-3g", "PBE"),
            (OH, 2, "sto-3g", "PBE0"),
            (NO, 2, "sto-3g", "PBE"),
            (OH, 2, "cc-pvdz", "PBE"),
        ] {
            let (ray, cos, other) =
                null_mode_numbers(xyz, mult, basis, xc).expect("mode must be detected");
            println!(
                "{basis} {xc}: rayleigh {ray:.3e}, |cos| {cos:.8}, other min |eig| {other:.3e}"
            );
            assert!(cos > 0.9999, "{basis} {xc}: |cos| {cos}");
            assert!(
                ray.abs() < NULL_RAYLEIGH_TOL,
                "{basis} {xc}: rayleigh {ray}"
            );
            assert!(
                other > NULL_RAYLEIGH_TOL,
                "{basis} {xc}: other eigenvalue {other}"
            );
        }
        assert!(null_mode_numbers(NH2, 2, "sto-3g", "PBE").is_none());
        assert!(null_mode_numbers(O2, 3, "sto-3g", "PBE").is_none());
    }

    /// THE PROJECTION ITSELF, both references: with a near-axisymmetric
    /// property (V = the core Hamiltonian + a 1e-6 symmetry-breaking
    /// admixture, see below) the UKS (OH/STO-3G PBE0) and
    /// ROKS (NO/6-31G PBE) solves must report the mode and return a Z with no
    /// component along κ_L. Both modes have NEGATIVE Rayleigh quotients;
    /// without the projection the small κ_L component of the right-hand side
    /// is amplified by 1/λ (or PCG meets the negative curvature) — each half
    /// fails if its reference loses the projection.
    #[test]
    fn projected_z_has_no_null_mode_component() {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            xc: Some("PBE0".into()),
            max_iter: 300,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..Default::default()
        };
        // An axisymmetric property plus a 1e-6 symmetry-breaking admixture:
        // the right-hand side then has a small component along κ_L (the size
        // the MBD lattice's anisotropy gives, rhs overlap ~1e-6..1e-8, under
        // NULL_RHS_TOL), which an unprojected solve amplifies by 1/λ.
        let v =
            ferric_integrals::oneelectron::hcore(&prep) + 1e-6 * generic_symmetric(prep.nbasis());
        let ru = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        let u =
            relaxation_gradient_unrestricted(&ctx, &mol, &prep, &bs, op, &bounds, &cfg, &ru, &v)
                .expect("UKS");
        let nelec = mol.nelec() as usize;
        let (na, nb) = (nelec.div_ceil(2), nelec / 2);
        let c_b = ru.mos_beta.as_ref().unwrap();
        let mode = axis_rotation_mode(
            &mol,
            &bs,
            &prep,
            &[
                (
                    &ru.mos_alpha.slice(s![.., na..]).to_owned(),
                    &ru.mos_alpha.slice(s![.., ..na]).to_owned(),
                ),
                (
                    &c_b.slice(s![.., nb..]).to_owned(),
                    &c_b.slice(s![.., ..nb]).to_owned(),
                ),
            ],
        )
        .unwrap()
        .expect("mode");
        let zu = [u.z_alpha.clone(), u.z_beta.clone()];
        let ovl_u = inner_blocks(&mode, &zu).abs() / inner_blocks(&zu, &zu).sqrt();
        // ROKS on NO/6-31G-PBE, where the mode's Rayleigh quotient is
        // NEGATIVE (−4.6e-3): there the unprojected PCG meets it. (On OH
        // ROKS-PBE0 it is positive and the Krylov space of an axisymmetric
        // right-hand side never acquires a κ_L component, projection or not.)
        let mol = Molecule::parse_xyz(NO, 0, 2).unwrap();
        let bs = ferric_core::basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RhfConfig {
            xc: Some("PBE".into()),
            ..cfg
        };
        let v =
            ferric_integrals::oneelectron::hcore(&prep) + 1e-6 * generic_symmetric(prep.nbasis());
        let nelec = mol.nelec() as usize;
        let (na, nb) = (nelec.div_ceil(2), nelec / 2);
        let ro = crate::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(ro.converged);
        let o = relaxation_gradient_roks(&ctx, &mol, &prep, &bs, op, &bounds, &cfg, &ro, &v)
            .expect("ROKS");
        let c = &ro.mos_alpha;
        let (cc, co, cv) = (
            c.slice(s![.., ..nb]).to_owned(),
            c.slice(s![.., nb..na]).to_owned(),
            c.slice(s![.., na..]).to_owned(),
        );
        let mode_o = axis_rotation_mode(&mol, &bs, &prep, &[(&cv, &cc), (&cv, &co), (&co, &cc)])
            .unwrap()
            .expect("mode");
        let zo = [o.z_vc.clone(), o.z_vo.clone(), o.z_oc.clone()];
        let ovl_o = inner_blocks(&mode_o, &zo).abs() / inner_blocks(&zo, &zo).sqrt();
        println!(
            "UKS: {:?}, |z.k|/|z| = {ovl_u:.2e}; ROKS: {:?}, |z.k|/|z| = {ovl_o:.2e}",
            u.null_mode, o.null_mode
        );
        assert!(u.null_mode.is_some() && o.null_mode.is_some());
        assert!(ovl_u < 1e-10, "UKS Z has a null-mode component {ovl_u:.3e}");
        assert!(
            ovl_o < 1e-10,
            "ROKS Z has a null-mode component {ovl_o:.3e}"
        );
    }

    /// A deterministic symmetric matrix with no symmetry at all.
    fn generic_symmetric(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            (0.37 * a + 1.13 * b + 0.29 * a * b).sin()
        })
    }

    fn oh_uks_pbe0(
        ext: Option<ferric_core::external_potential::ExternalPotential>,
    ) -> (
        Molecule,
        ferric_core::basis::BasisSet,
        PreparedBasis,
        SchwarzBounds,
        RhfConfig,
        ScfResult,
    ) {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let cfg = RhfConfig {
            xc: Some("PBE0".into()),
            max_iter: 300,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            external_potential: ext,
            ..Default::default()
        };
        let r =
            crate::uhf::solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
        assert!(r.converged);
        (mol, bs, prep, bounds, cfg, r)
    }

    /// REFUSED, NOT PROJECTED: a property that is not symmetric about the axis
    /// has a right-hand side along κ_L, so its Z-vector equation has no
    /// solution on the complement and the derivative is not determined. A
    /// generic symmetric V (no symmetry) must be refused with the overlap
    /// named. Fails if the `NULL_RHS_TOL` refusal is dropped.
    #[test]
    fn non_axisymmetric_property_is_refused() {
        let (mol, bs, prep, bounds, cfg, r) = oh_uks_pbe0(None);
        let v = generic_symmetric(prep.nbasis());
        let err = relaxation_gradient_unrestricted(
            &ParallelContext::default(),
            &mol,
            &prep,
            &bs,
            Operator::coulomb(),
            &bounds,
            &cfg,
            &r,
            &v,
        )
        .expect_err("a non-axisymmetric property must be refused");
        println!("{err}");
        assert!(format!("{err}").contains("not orthogonal"), "{err}");
    }

    /// NOT PROJECTED when the symmetry is broken from outside: a point charge
    /// off the molecular axis makes the axis rotation an ordinary, stiff
    /// direction (its Rayleigh quotient rises above `NULL_RAYLEIGH_TOL`), so
    /// the mode must not be removed. Fails if the Rayleigh bar is dropped.
    #[test]
    fn external_symmetry_breaking_is_not_projected() {
        let ext = ferric_core::external_potential::ExternalPotential {
            point_charges: vec![ferric_core::external_potential::PointCharge {
                q: 1.0,
                x: 3.0,
                y: 0.0,
                z: 0.9,
            }],
            ..Default::default()
        };
        let (mol, bs, prep, bounds, cfg, r) = oh_uks_pbe0(Some(ext));
        let v = ferric_integrals::oneelectron::hcore(&prep);
        let u = relaxation_gradient_unrestricted(
            &ParallelContext::default(),
            &mol,
            &prep,
            &bs,
            Operator::coulomb(),
            &bounds,
            &cfg,
            &r,
            &v,
        )
        .expect("solve");
        // The Rayleigh quotient of κ_L itself, for the record.
        let (h, gap, kl) = dense(&mol, &bs, &prep, &cfg, &r);
        let k = kl.expect("linear");
        let dim = gap.len();
        let khk: f64 = (0..dim)
            .map(|i| {
                k[i] * (0..dim)
                    .map(|j| 0.5 * (h[(i, j)] + h[(j, i)]) * k[j])
                    .sum::<f64>()
            })
            .sum();
        let kmk: f64 = k.iter().zip(&gap).map(|(a, g)| a * a * g).sum();
        println!(
            "off-axis charge: Rayleigh {:.3e}, null_mode {:?}",
            khk / kmk,
            u.null_mode
        );
        assert!(
            (khk / kmk).abs() > NULL_RAYLEIGH_TOL,
            "the charge is too weak to test the bar"
        );
        assert_eq!(u.null_mode, None);
    }

    #[test]
    #[ignore = "measurement: issue #265 null-mode investigation"]
    fn measure_open_shell_hessian_null_mode() {
        for ang in [110, 302] {
            uks(OH, 2, "sto-3g", "PBE", ang);
        }
        uks(OH, 2, "6-31g", "PBE", 110);
        uks(OH, 2, "sto-3g", "PBE0", 110);
        uks(CH, 2, "sto-3g", "PBE", 110);
        uks(NO, 2, "sto-3g", "PBE", 110);
        uks(CH, 2, "6-31g", "PBE", 110);
        uks(CH, 2, "sto-3g", "PBE", 302);
        uks(NH2, 2, "sto-3g", "PBE", 110);
        uks(O2, 3, "sto-3g", "PBE", 110);
    }
}
