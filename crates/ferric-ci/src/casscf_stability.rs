//! Second-order check of a converged CASSCF point: the lowest eigenvalue of
//! the CI-relaxed orbital Hessian, and a step along its mode when negative.
//!
//! The CASSCF energy depends on the orbital rotation `κ` and the CI vector.
//! With `c(z) = (c + z) / ‖c + z‖`, `z ⊥ c`, around the CI root `c`
//! (`H c = θ c`) of the orbitals `C`, the full Hessian at `κ = 0`, `z = 0` is
//!
//! ```text
//!   [ H_oo   H_oc ]      H_oo x = the fixed-RDM orbital Hessian (crate::casscf_newton)
//!   [ H_co   H_cc ]      H_co x = 2 P dH(x) c,        H_cc = 2 P (H − θ) P
//!                        H_oc z = g(γ_T, Γ_T)
//! ```
//!
//! `P = 1 − c cᵀ`; `dH(x)` is the frame derivative of the active-space
//! Hamiltonian along the rotation `x`, with one-electron part
//! `dh_tu = (dF^I)_tu` and
//! `d(tu|vw) = Σ_r X_rt (ru|vw) + X_ru (tr|vw) + X_rv (tu|rw) + X_rw (tu|rv)`
//! (the active block of the frame derivatives in `crate::casscf_newton`);
//! `γ_T`, `Γ_T` are the symmetrized transition RDMs of `c` and `z`
//! ([`crate::rdm::spin_summed_transition_rdms`]), and `g(γ_T, Γ_T)` is the
//! part of the orbital gradient linear in the RDMs:
//!
//! ```text
//!   F_pi = 2 F^A[γ_T]_pi,   F_pt = Σ_u F^I_pu γ_T,ut + Σ_uvw (pu|vw) Γ_T,tuvw,
//!   g = pack 2 (F − Fᵀ).
//! ```
//!
//! Since the gradient is `⟨c|dH(x)|c⟩`, `x · g(γ_T, Γ_T) = 2 ⟨z|dH(x)|c⟩`, so
//! `H_oc = H_coᵀ`. The energy with the CI vector relaxed at every orbital
//! set, `E_rel(κ) = min_z E(κ, z)`, has the Hessian (the Schur complement)
//!
//! ```text
//!   H_rel x = H_oo x + H_oc z(x),   (H − θ) z = −P dH(x) c,   z ⊥ c,
//! ```
//!
//! one CI linear solve per product, by conjugate gradients on the
//! complement of `c` with the preconditioner `P (diag H − θ)⁻¹ P`
//! (`diag H − θ >= 0` because `θ` is the lowest eigenvalue). `H_rel` is what
//! the finite difference of the CI-relaxed analytic gradient measures
//! (`tests/casscf_vs_pyscf.rs`); the fixed-RDM `H_oo` lacks the CI response
//! and is stiffer (`H_rel <= H_oo`). Its lowest eigenpair comes from the
//! block Davidson of [`ferric_scf::stability::davidson_lowest`] on the
//! approximate diagonal of
//! [`CasScfProblem::approx_hessian_diag`](crate::casscf::CasScfProblem::approx_hessian_diag).
//!
//! A negative eigenvalue below the noise floor makes the converged point a
//! saddle of the CASSCF energy within the active-space partition:
//! [`CasScfStabilityMode::Follow`] rotates the orbitals by [`FOLLOW_STEP`](crate::casscf_stability::FOLLOW_STEP)
//! along the eigenvector (the sign with the lower energy, halving the step
//! when neither lowers it) and converges again, at most [`MAX_FOLLOW`](crate::casscf_stability::MAX_FOLLOW) times.
//! Equations and measurements: `crates/ferric-ci/theory.md`.

use crate::casscf::{CasScfPoint, CasScfProblem};
use crate::casscf_newton::OrbitalHessian;
use crate::hamiltonian::{hamiltonian_diagonal, DeterminantSpace};
use crate::integrals::ActiveSpaceIntegrals;
use crate::rdm::spin_summed_transition_rdms;
use crate::sigma::DirectSigma;
use crate::strings::enumerate_strings;
use ferric_core::FerricError;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_scf::stability::{davidson_lowest, StabilityConfig, StabilityVerdict};
use ndarray::{s, Array2, ArrayView2};
use std::cell::Cell;

/// Rotation (radians, 2-norm of `κ`) of one step along a negative mode.
pub const FOLLOW_STEP: f64 = 0.3;
/// Steps along negative modes in one run, each followed by a full
/// re-optimization.
pub const MAX_FOLLOW: usize = 5;
/// An eigenvalue with `|λ| <=` this (Ha) is not distinguished from zero
/// (the default floor of the SCF stability check,
/// [`ferric_scf::stability::StabilityConfig::noise_floor`]).
pub const STABILITY_NOISE_FLOOR: f64 = 1e-6;
/// Davidson residual of the lowest CI-relaxed eigenpair.
pub const STABILITY_CONV_THRESH: f64 = 1e-5;
/// Relative residual `‖(H − θ) z + P dH c‖ / ‖P dH c‖` of each CI linear
/// solve.
pub const CI_RESPONSE_RTOL: f64 = 1e-10;
/// Conjugate-gradient iterations of one CI linear solve.
pub const CI_RESPONSE_MAX_ITER: usize = 500;
/// Floor on `diag H − θ` in the CI preconditioner (Ha).
const CI_PRECOND_FLOOR: f64 = 1e-4;

/// What a converged CASSCF does with its CI-relaxed orbital Hessian
/// (`[cas] stability`, Python `casscf_stability`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CasScfStabilityMode {
    /// No second-order check (the default: the check costs 0.5 to 3 times
    /// the CASSCF it checks, `crates/ferric-ci/theory.md`).
    #[default]
    Off,
    /// The lowest eigenvalue is computed and reported
    /// ([`crate::CasScfResult::stability`]).
    Check,
    /// As `Check`; at a saddle the orbitals step along the negative mode and
    /// the CASSCF converges again, until the point is not a saddle or
    /// [`MAX_FOLLOW`] steps are taken.
    Follow,
}

impl CasScfStabilityMode {
    /// Strict parse of a config value: `"check"`, `"follow"` or `"off"`.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(CasScfStabilityMode::Off),
            "check" => Ok(CasScfStabilityMode::Check),
            "follow" => Ok(CasScfStabilityMode::Follow),
            other => Err(format!(
                "stability must be \"check\", \"follow\" or \"off\", got {other:?}"
            )),
        }
    }

    /// `"off"`, `"check"` or `"follow"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            CasScfStabilityMode::Off => "off",
            CasScfStabilityMode::Check => "check",
            CasScfStabilityMode::Follow => "follow",
        }
    }
}

/// The lowest eigenpair of the CI-relaxed orbital Hessian at the returned
/// CASSCF orbitals.
#[derive(Debug, Clone)]
pub struct CasScfStability {
    /// Lowest eigenvalue (Ha / rad²). Negative: a saddle.
    pub lowest_eigenvalue: f64,
    /// Its eigenvector over [`CasScfProblem::pairs`] of the returned
    /// orbitals, unit 2-norm.
    pub eigenvector: Vec<f64>,
    /// Davidson residual `‖H v − λ v‖` of the pair.
    pub residual: f64,
    /// Davidson iterations.
    pub iterations: usize,
    /// CI-relaxed Hessian-vector products (one CI linear solve each).
    pub products: usize,
    /// The Davidson block met [`STABILITY_CONV_THRESH`].
    pub converged: bool,
    /// Worst relative residual of the CI linear solves.
    pub ci_response_residual: f64,
    /// `max(STABILITY_NOISE_FLOOR, residual)`: `|λ|` at or below it is not
    /// distinguished from zero.
    pub noise_floor: f64,
    /// Steps taken along negative modes ([`CasScfStabilityMode::Follow`])
    /// before the returned point.
    pub follow_steps: usize,
    /// The energy of the first converged point when `follow_steps > 0`
    /// (Ha).
    pub energy_before_follow: Option<f64>,
}

impl CasScfStability {
    /// The four-way verdict of [`ferric_scf::stability::StabilityVerdict`].
    /// A negative eigenvalue below the floor is a saddle even when the
    /// Davidson did not converge (Rayleigh–Ritz bounds `λ_min` from above);
    /// any other unconverged solve, Davidson or CI response, proves nothing.
    pub fn verdict(&self) -> StabilityVerdict {
        let ci_ok = self.ci_response_residual <= 10.0 * CI_RESPONSE_RTOL;
        if !ci_ok {
            StabilityVerdict::Indeterminate
        } else if self.lowest_eigenvalue < -self.noise_floor {
            StabilityVerdict::Unstable
        } else if !self.converged {
            StabilityVerdict::Indeterminate
        } else if self.lowest_eigenvalue.abs() <= self.noise_floor {
            StabilityVerdict::Marginal
        } else {
            StabilityVerdict::Stable
        }
    }

    /// One line for logs and certificates.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "CI-relaxed orbital Hessian: {}, lambda_min = {:+.6e} Ha (noise floor {:.1e}, \
             Davidson residual {:.1e}, {} products)",
            self.verdict().label(),
            self.lowest_eigenvalue,
            self.noise_floor,
            self.residual,
            self.products
        );
        if let Some(e0) = self.energy_before_follow {
            out.push_str(&format!(
                "; {} step(s) along negative modes from E = {e0:.10} Ha",
                self.follow_steps
            ));
        }
        out
    }
}

/// The CI-relaxed orbital Hessian at one converged point: the fixed-RDM
/// Hessian and the CI space of the root
/// ([`CasScfProblem::ci_relaxed_hessian`]).
pub struct CiRelaxedHessian {
    orbital: OrbitalHessian,
    ints: ActiveSpaceIntegrals,
    space: DeterminantSpace,
    c: Vec<f64>,
    theta: f64,
    precond: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn project_out(v: &mut [f64], c: &[f64]) {
    let o = dot(v, c);
    v.iter_mut().zip(c).for_each(|(a, b)| *a -= o * b);
}

impl CasScfProblem<'_> {
    /// The CI-relaxed orbital Hessian at the point `pt` evaluated at `c`
    /// (module documentation).
    pub fn ci_relaxed_hessian(
        &self,
        c: &Array2<f64>,
        pt: &CasScfPoint,
    ) -> Result<CiRelaxedHessian, FerricError> {
        let orbital = self.orbital_hessian(c, pt)?;
        let ints = pt.ci.integrals.clone();
        let (na_e, nb_e) = self.ci_config.n_elec_active;
        let space = DeterminantSpace {
            alpha_strings: enumerate_strings(self.n_active, na_e),
            beta_strings: enumerate_strings(self.n_active, nb_e),
        };
        let mut cv = pt.ci.ci_vector.clone();
        let nrm = dot(&cv, &cv).sqrt();
        if cv.len() != space.n_det() || !(nrm > 0.0) {
            return Err(FerricError::General(format!(
                "CASSCF stability: CI vector of length {} (norm {nrm:e}) for {} determinants",
                cv.len(),
                space.n_det()
            )));
        }
        cv.iter_mut().for_each(|x| *x /= nrm);
        let direct = DirectSigma::new(&ints, &space);
        let theta = dot(&cv, &direct.apply(&cv));
        drop(direct);
        let precond = hamiltonian_diagonal(&ints, &space)
            .into_iter()
            .map(|d| (d - theta).max(CI_PRECOND_FLOOR))
            .collect();
        Ok(CiRelaxedHessian {
            orbital,
            ints,
            space,
            c: cv,
            theta,
            precond,
        })
    }

    /// `H_rel x`, the CI-relaxed orbital Hessian times `x`.
    pub fn ci_relaxed_hessian_vector(
        &self,
        h: &CiRelaxedHessian,
        x: &[f64],
    ) -> Result<Vec<f64>, FerricError> {
        Ok(self.ci_relaxed_hessian_vector_with(h, x, true)?.0)
    }

    /// `H_rel x` with the CI response (`with_ci`) or `H_oo x` without it
    /// through the same code path, and the relative residual of the CI
    /// linear solve.
    pub fn ci_relaxed_hessian_vector_with(
        &self,
        h: &CiRelaxedHessian,
        x: &[f64],
        with_ci: bool,
    ) -> Result<(Vec<f64>, f64), FerricError> {
        let (dg, xm, dfi) = self.frame_derivative_parts(&h.orbital, x)?;
        let g = &h.orbital.gmat;
        let comm = g.dot(&xm) - xm.dot(g);
        let mut out = self.pack(&(dg - 0.5 * comm));
        if !with_ci {
            return Ok((out, 0.0));
        }
        let direct = DirectSigma::new(&h.ints, &h.space);
        let dints = self.active_hamiltonian_derivative(h, &xm, &dfi)?;
        let mut b = direct.apply_integrals(&dints, &h.c);
        project_out(&mut b, &h.c);
        b.iter_mut().for_each(|v| *v = -*v);
        let (z, rel) = solve_ci_response(&direct, h, &b)?;
        let hoc = self.ci_to_orbital(h, &direct, &z)?;
        out.iter_mut().zip(&hoc).for_each(|(a, b)| *a += b);
        Ok((out, rel))
    }

    /// `dH(x)`: `dh_tu = (dF^I)_tu` (symmetrized) and `d(tu|vw)` from
    /// `(ru|vw)` (module documentation), no constant.
    fn active_hamiltonian_derivative(
        &self,
        h: &CiRelaxedHessian,
        xm: &Array2<f64>,
        dfi: &Array2<f64>,
    ) -> Result<ActiveSpaceIntegrals, FerricError> {
        let (nc, na, nmo) = (self.n_core, self.n_active, self.nmo);
        let na3 = na * na * na;
        let blk = dfi.slice(s![nc..nc + na, nc..nc + na]);
        let dh = 0.5 * (&blk + &blk.t());
        let p = ArrayView2::from_shape((nmo, na3), &h.orbital.paaa)
            .map_err(|e| FerricError::General(format!("CASSCF (pu|vw) shape: {e}")))?;
        // a[t, (u v w)] = Σ_r X_rt (ru|vw).
        let a = with_blas_threads(opt_in_blas_threads(), || {
            xm.slice(s![.., nc..nc + na]).t().dot(&p)
        });
        let at = |t: usize, u: usize, v: usize, w: usize| a[(t, (u * na + v) * na + w)];
        let mut eri = vec![0.0; na * na3];
        for t in 0..na {
            for u in 0..na {
                for v in 0..na {
                    for w in 0..na {
                        eri[((t * na + u) * na + v) * na + w] =
                            at(t, u, v, w) + at(u, t, v, w) + at(v, w, t, u) + at(w, v, t, u);
                    }
                }
            }
        }
        Ok(ActiveSpaceIntegrals {
            n_active: na,
            h: dh,
            eri,
            e_core: 0.0,
        })
    }

    /// `H_oc z = g(γ_T, Γ_T)`, the RDM-linear part of the orbital gradient
    /// at the transition RDMs of `c` and `z`.
    fn ci_to_orbital(
        &self,
        h: &CiRelaxedHessian,
        direct: &DirectSigma<'_>,
        z: &[f64],
    ) -> Result<Vec<f64>, FerricError> {
        let (nc, na, nmo) = (self.n_core, self.n_active, self.nmo);
        let nocc = nc + na;
        let na3 = na * na * na;
        let rd = spin_summed_transition_rdms(direct, &h.c, z)?;
        let gt = &rd.dm1;
        let fa_core = match &h.orbital.occ_kernel {
            Some(l) => {
                // D^a_T = C (Y + Yᵀ) Cᵀ with Y_{nc+u, nc+t} = ½ γ_T,ut.
                let mut y = ndarray::Array1::<f64>::zeros(nmo * nocc);
                for u in 0..na {
                    for t in 0..na {
                        y[(nc + u) * nocc + nc + t] = 0.5 * gt[(u, t)];
                    }
                }
                let ly = with_blas_threads(opt_in_blas_threads(), || l.dot(&y));
                ly.into_shape_with_order((nmo, nocc))
                    .map_err(|e| FerricError::General(format!("CASSCF L vec(Y) shape: {e}")))?
                    .slice(s![.., ..nc])
                    .to_owned()
            }
            None => self
                .active_fock(&h.orbital.c, gt)?
                .slice(s![.., ..nc])
                .to_owned(),
        };
        let mut f = Array2::<f64>::zeros((nmo, nmo));
        f.slice_mut(s![.., ..nc]).assign(&(2.0 * &fa_core));
        let one = h.orbital.fi.slice(s![.., nc..nocc]).dot(gt);
        let p = ArrayView2::from_shape((nmo, na3), &h.orbital.paaa)
            .map_err(|e| FerricError::General(format!("CASSCF (pu|vw) shape: {e}")))?;
        let g2 = rd
            .dm2
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((na, na3))
            .map_err(|e| FerricError::General(format!("CASSCF dm2 shape: {e}")))?;
        let two = p.dot(&g2.t());
        f.slice_mut(s![.., nc..nocc]).assign(&(&one + &two));
        Ok(self.pack(&(2.0 * (&f - &f.t()))))
    }

    /// Lowest eigenpair of the CI-relaxed orbital Hessian at the converged
    /// point `pt` evaluated at `c` (module documentation).
    pub fn stability(
        &self,
        c: &Array2<f64>,
        pt: &CasScfPoint,
    ) -> Result<CasScfStability, FerricError> {
        self.stability_with(c, pt, true)
    }

    /// [`Self::stability`] with the CI response (`with_ci`), or the same
    /// eigensolve on the fixed-RDM Hessian `H_oo` (the comparison of
    /// `theory.md`).
    pub fn stability_with(
        &self,
        c: &Array2<f64>,
        pt: &CasScfPoint,
        with_ci: bool,
    ) -> Result<CasScfStability, FerricError> {
        if self.pairs.is_empty() {
            return Err(FerricError::General(
                "CASSCF stability: no nonredundant orbital rotations".to_string(),
            ));
        }
        let h = self.ci_relaxed_hessian(c, pt)?;
        let diag = self.approx_hessian_diag(pt);
        let products = Cell::new(0usize);
        let worst = Cell::new(0.0f64);
        let cfg = StabilityConfig {
            conv_thresh: STABILITY_CONV_THRESH,
            noise_floor: STABILITY_NOISE_FLOOR,
            ..StabilityConfig::default()
        };
        let r = davidson_lowest(
            self.pairs.len(),
            |x| {
                products.set(products.get() + 1);
                let (v, rel) = self.ci_relaxed_hessian_vector_with(&h, x, with_ci)?;
                worst.set(worst.get().max(rel));
                Ok(v)
            },
            &diag,
            &cfg,
        )?;
        let nv = dot(&r.eigenvector, &r.eigenvector).sqrt();
        Ok(CasScfStability {
            lowest_eigenvalue: r.eigenvalue,
            eigenvector: r.eigenvector.iter().map(|x| x / nv).collect(),
            residual: r.residual,
            iterations: r.iterations,
            products: products.get(),
            converged: r.converged,
            ci_response_residual: worst.get(),
            noise_floor: STABILITY_NOISE_FLOOR.max(r.residual),
            follow_steps: 0,
            energy_before_follow: None,
        })
    }

    /// Orbitals one step down the mode `v` from `c` (energy `e0`): the sign
    /// of `±FOLLOW_STEP v` with the lower energy, halving the step up to
    /// three times while neither lowers it. `None` when no step lowers the
    /// energy.
    pub(crate) fn follow_step(
        &self,
        c: &Array2<f64>,
        e0: f64,
        v: &[f64],
    ) -> Result<Option<Array2<f64>>, FerricError> {
        let mut a = FOLLOW_STEP;
        for _ in 0..4 {
            let mut best: Option<(f64, Array2<f64>)> = None;
            for sign in [1.0, -1.0] {
                let k: Vec<f64> = v.iter().map(|x| sign * a * x).collect();
                let cn = self.rotate(c, &k)?;
                let e = self.energy(&cn)?;
                if best.as_ref().is_none_or(|(eb, _)| e < *eb) {
                    best = Some((e, cn));
                }
            }
            if let Some((e, cn)) = best {
                if e < e0 {
                    return Ok(Some(cn));
                }
            }
            a *= 0.5;
        }
        Ok(None)
    }
}

/// `(H − θ) z = rhs` on the complement of `c` (`rhs ⊥ c`) by
/// preconditioned conjugate gradients; returns `z` and the relative
/// residual reached.
fn solve_ci_response(
    direct: &DirectSigma<'_>,
    h: &CiRelaxedHessian,
    rhs: &[f64],
) -> Result<(Vec<f64>, f64), FerricError> {
    let n = rhs.len();
    let bn = dot(rhs, rhs).sqrt();
    if bn == 0.0 {
        return Ok((vec![0.0; n], 0.0));
    }
    let apply = |p: &[f64]| -> Vec<f64> {
        let mut hp = direct.apply(p);
        hp.iter_mut().zip(p).for_each(|(a, b)| *a -= h.theta * b);
        project_out(&mut hp, &h.c);
        hp
    };
    let precondition = |r: &[f64]| -> Vec<f64> {
        let mut z: Vec<f64> = r.iter().zip(&h.precond).map(|(a, d)| a / d).collect();
        project_out(&mut z, &h.c);
        z
    };
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    project_out(&mut r, &h.c);
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut rel = 1.0;
    for _ in 0..CI_RESPONSE_MAX_ITER {
        let ap = apply(&p);
        let pap = dot(&p, &ap);
        if !(pap > 0.0) {
            return Err(FerricError::General(format!(
                "CASSCF stability: CI response operator H − θ is not positive on the \
                 complement of the root (p·(H − θ)p = {pap:e}); the root is not the lowest \
                 of its sector"
            )));
        }
        let alpha = rz / pap;
        x.iter_mut().zip(&p).for_each(|(a, b)| *a += alpha * b);
        r.iter_mut().zip(&ap).for_each(|(a, b)| *a -= alpha * b);
        rel = dot(&r, &r).sqrt() / bn;
        if rel <= CI_RESPONSE_RTOL {
            break;
        }
        z = precondition(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        p.iter_mut().zip(&z).for_each(|(a, b)| *a = b + beta * *a);
    }
    project_out(&mut x, &h.c);
    Ok((x, rel))
}
