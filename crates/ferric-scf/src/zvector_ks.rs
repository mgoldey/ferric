//! Closed-shell (RHF-reference Kohn–Sham) Z-vector: the orbital-relaxation
//! term of the nuclear gradient of a post-SCF quantity Q(R, D).
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
    fn new(
        mol: &Molecule,
        prep: &'a PreparedBasis,
        config: &RhfConfig,
        xc: &str,
        d: &Array2<f64>,
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
        let d_half = 0.5 * d;
        let fxc = crate::rohf::FxcKernelStore::build(mol, prep, &grid, xc, &d_half, &d_half)?;
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
/// [`CG_REL_TOL`] in [`CG_MAX_ITER`] iterations; any integral, kernel or
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

    let resp = ResponseFock::new(mol, prep, config, xc, d)?;
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
            if !(php > 0.0) {
                return Err(FerricError::General(format!(
                    "Z-vector relaxation: the orbital Hessian is not positive definite along \
                     the search direction (p.Hp = {php:.3e}); the SCF is not a minimum"
                )));
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
        let co = c.slice(s![.., ..nocc]).to_owned();
        let cv = c.slice(s![.., nocc..]).to_owned();
        let ztz = z.t().dot(&z);
        let rotated_density = |tt: f64| -> Result<Array2<f64>, FerricError> {
            use ndarray_linalg::{Eigh, UPLO};
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
            let ct = (&co + &(tt * &cv.dot(&z))).dot(&m_inv_sqrt);
            Ok(2.0 * ct.dot(&ct.t()))
        };
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
    })
}
