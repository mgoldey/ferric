//! Kohn–Sham Z-vector (closed-shell RKS and unrestricted UKS references): the
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
//! that are not the aufbau projectors of the MOs (MOM); ROKS is refused.
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
             the SCF is not a minimum, or (open shell) a degenerate SOMO pair makes the \
             orbital Hessian near-singular"
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
        let fxc = crate::rohf::FxcKernelStore::build(mol, prep, &grid, xc, d_a, d_b)?;
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
             ROKS references are not supported"
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
    let rhs = [
        -2.0 * sa.cv.t().dot(v_ao).dot(&sa.co),
        -2.0 * sb.cv.t().dot(v_ao).dot(&sb.co),
    ];
    let rhs_max = max2(&rhs);
    let mut z = precond(&rhs);
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
        let mut p = precond(&r);
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
            let hp = hx(&p)?;
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
            let zr = precond(&r);
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
    })
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
