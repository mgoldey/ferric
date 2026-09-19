//! Transition-state search by partitioned rational function optimization
//! (P-RFO).
//!
//! # Why this cannot reuse the BFGS optimizer
//!
//! [`crate::optimize`] MINIMIZES. Its quasi-Newton update is kept positive
//! definite on purpose, which is exactly the property a saddle search must
//! break: a first-order transition state is a maximum along one direction and a
//! minimum along the other `3N - 7`. No choice of step size turns a minimizer
//! into a saddle finder, so this is a separate driver rather than a flag.
//!
//! P-RFO (Banerjee, Adams, Simons, Shepard, J. Phys. Chem. 89, 52 (1985))
//! partitions the Hessian eigenspace in two and solves a separate rational
//! function step in each: MAXIMIZE along the followed mode, MINIMIZE in its
//! orthogonal complement. The two partitions get different shift parameters,
//! which is what lets one direction climb while the rest descend.
//!
//! # Which Hessian, and why it is the expensive part
//!
//! [`crate::hessian::rhf_hessian`] is a documented stub that ALWAYS returns
//! `Err` -- terms 2-5 need libint2 `deriv_order=2`, which this build does not
//! have. The working Hessian is
//! [`crate::frequencies::harmonic_frequencies`], which CENTRAL-DIFFERENCES the
//! analytic gradient: **6N gradient evaluations** (MEASURED via the gradient
//! counter: H2 = 12, water = 18 -- exactly 6N, not 6N+1).
//!
//! That cost is why [`SaddleConfig::hessian_recalc_every`] exists and defaults
//! to 0 (never recompute). Rebuilding the Hessian at every step would make a
//! 10-atom search cost 60 gradients per step, which is not a search, it is a
//! Hessian benchmark. Between recomputations the Hessian is carried forward by
//! a **Bofill update**, which deliberately does NOT preserve positive
//! definiteness -- the property that makes BFGS wrong here.
//!
//! # What this module refuses to do
//!
//! * **It will not start without a negative eigenvalue.** If the projected
//!   Hessian at the initial geometry has no negative mode, the geometry is in a
//!   minimum's basin and P-RFO has no uphill direction to follow. Returning a
//!   "converged" structure from there would hand back a minimum labelled as a
//!   transition state. It errors instead, naming the smallest eigenvalue.
//! * **It will not call a stationary point converged on the gradient alone.**
//!   A converged saddle must have EXACTLY ONE imaginary frequency.
//!   [`SaddleResult::n_imaginary`] carries the count from a final Hessian and
//!   [`SaddleResult::is_transition_state`] requires it to be 1. Zero means a
//!   minimum, two or more means a higher-order saddle; neither is a transition
//!   state, and both otherwise satisfy `|g| -> 0`.
//! * **It does not verify the mode is the RIGHT one.** Exactly one imaginary
//!   frequency says first-order saddle, not "saddle for the reaction you
//!   meant". Checking that the imaginary mode points along the intended
//!   reaction coordinate needs `normal_modes` and chemical judgement; the
//!   vector is returned in [`SaddleResult::imaginary_mode`] so a caller can do
//!   it, and this module does not pretend to.
//!
//! # Scope
//!
//! Cartesian coordinates only. Internal-coordinate P-RFO converges in fewer
//! steps on floppy systems, but the back-transformation
//! ([`crate::optimize`]'s `CoordSystem::RedundantInternal`) interacts with mode
//! following in ways that need their own validation, and shipping the
//! Cartesian version first means the eigenvector-following logic can be tested
//! against an analytic surface without that machinery in the way.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ndarray::{Array1, Array2};
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO;

use crate::frequencies::{atom_masses, detect_linear, translation_rotation_basis};

/// How the Hessian is refreshed during the search.
#[derive(Debug, Clone)]
pub struct SaddleConfig {
    /// Maximum P-RFO steps before giving up.
    pub max_steps: usize,
    /// Convergence threshold on the maximum gradient component (Hartree/Bohr).
    /// Same default as [`crate::optimize::OptimizeConfig`], so a saddle is held
    /// to the same standard as a minimum.
    pub g_max_thresh: f64,
    /// Convergence threshold on the RMS gradient (Hartree/Bohr).
    pub g_rms_thresh: f64,
    /// Largest Cartesian displacement allowed in one step (Bohr). Smaller than
    /// the minimizer's default: a saddle search is climbing, and an overlong
    /// step near a ridge lands in a different basin entirely.
    pub trust_radius: f64,
    /// Which Hessian eigenvector to follow uphill, by ascending eigenvalue
    /// among the PROJECTED modes. 0 (the default) follows the lowest, which is
    /// the reaction coordinate in the usual case. Set it higher only when you
    /// know the lowest mode is something else (a methyl rotor, say).
    pub follow_mode: usize,
    /// Recompute the exact Hessian every N steps. 0 (the default) means never:
    /// build it once at the start, then carry it with Bofill updates. Each
    /// recomputation costs 6N gradient evaluations.
    pub hessian_recalc_every: usize,
}

impl Default for SaddleConfig {
    fn default() -> Self {
        Self {
            max_steps: 60,
            g_max_thresh: 4.5e-4,
            g_rms_thresh: 3.0e-4,
            trust_radius: 0.05,
            follow_mode: 0,
            hessian_recalc_every: 0,
        }
    }
}

/// Outcome of a saddle search.
#[derive(Debug, Clone)]
#[must_use = "a saddle result must be checked for is_transition_state()"]
pub struct SaddleResult {
    /// The geometry the search arrived at. Valid to inspect even when
    /// `converged` is false -- it is where the search stopped.
    pub mol: Molecule,
    pub energy: f64,
    pub steps: usize,
    /// Whether the GRADIENT converged. This alone does NOT mean a transition
    /// state was found: see `n_imaginary`.
    pub converged: bool,
    /// Number of negative eigenvalues of the projected Hessian at the final
    /// geometry. **1** is a first-order transition state; 0 is a minimum; >1 is
    /// a higher-order saddle.
    pub n_imaginary: usize,
    /// The eigenvector of the single negative mode, in Cartesian coordinates
    /// (`3N`), when there is exactly one. `None` otherwise. A caller must check
    /// this points along the intended reaction coordinate -- this module cannot.
    pub imaginary_mode: Option<Array1<f64>>,
    /// Smallest projected Hessian eigenvalue at the final geometry, for
    /// diagnostics.
    pub lowest_eigenvalue: f64,
}

impl SaddleResult {
    /// A first-order transition state: gradient converged AND exactly one
    /// imaginary frequency.
    ///
    /// Both halves are required. `converged` alone is satisfied by every
    /// stationary point, minima included.
    #[must_use]
    pub fn is_transition_state(&self) -> bool {
        self.converged && self.n_imaginary == 1
    }
}

/// Project translations and rotations out of a Cartesian Hessian, then
/// diagonalize.
///
/// Returns `(eigenvalues, eigenvectors_as_columns, n_trans_rot)` in the
/// MASS-WEIGHTED frame. The trans/rot modes are shifted to a large positive
/// eigenvalue rather than removed, so indices stay aligned with the full `3N`
/// space and a shifted mode can never be selected as the one to follow.
///
/// Projecting first is not optional. The six near-zero trans/rot eigenvalues
/// are numerically dirty, and any of them can come out slightly NEGATIVE --
/// at which point "follow the lowest negative mode" follows a rigid-body
/// rotation and the search translates the molecule across space forever.
fn projected_eigen(
    mol: &Molecule,
    mass_au: &[f64],
    hess_cart: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>, usize), FerricError> {
    let natoms = mol.atoms.len();
    let n = 3 * natoms;
    if hess_cart.shape() != [n, n] {
        return Err(FerricError::General(format!(
            "saddle: Hessian is {:?} but the molecule has {natoms} atoms (expected {n}x{n})",
            hess_cart.shape()
        )));
    }

    // Mass-weight: H~_ij = H_ij / sqrt(m_i m_j).
    let sqrt_m: Vec<f64> = (0..n).map(|i| mass_au[i / 3].sqrt()).collect();
    let mut hw = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            hw[[i, j]] = hess_cart[[i, j]] / (sqrt_m[i] * sqrt_m[j]);
        }
    }
    // Symmetrize: the finite-difference Hessian is only symmetric to the
    // displacement's noise floor, and a non-symmetric matrix has no business
    // going into a symmetric eigensolver.
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (hw[[i, j]] + hw[[j, i]]);
            hw[[i, j]] = avg;
            hw[[j, i]] = avg;
        }
    }

    let tr = translation_rotation_basis(mol, mass_au)?;
    let n_tr = tr.len();

    // P = I - sum_k v_k v_k^T, applied on both sides: H' = P H P.
    let mut p = Array2::<f64>::eye(n);
    for v in &tr {
        for i in 0..n {
            for j in 0..n {
                p[[i, j]] -= v[i] * v[j];
            }
        }
    }
    let hp = p.dot(&hw).dot(&p);

    // Push the projected-out subspace far up the spectrum so it can never be
    // mistaken for the reaction coordinate. The shift is large relative to any
    // physical force constant.
    const TR_SHIFT: f64 = 1.0e3;
    let mut hs = hp;
    for v in &tr {
        for i in 0..n {
            for j in 0..n {
                hs[[i, j]] += TR_SHIFT * v[i] * v[j];
            }
        }
    }

    let (evals, evecs) = hs.eigh(UPLO::Lower).map_err(|e| {
        FerricError::General(format!("saddle: Hessian diagonalization failed: {e}"))
    })?;
    Ok((evals, evecs, n_tr))
}

/// The P-RFO step in the mass-weighted eigenbasis.
///
/// `g_q` is the gradient projected onto the eigenvectors, `evals` the
/// eigenvalues, `follow` the index to climb. Returns the step in that same
/// eigenbasis.
///
/// Two separate rational-function problems:
///
/// * **Followed mode** — a 2x2 RFO whose shift `lambda_p` is the POSITIVE root,
///   giving an uphill step: `dq_p = -g_p / (b_p - lambda_p)` with
///   `lambda_p = (b_p + sqrt(b_p^2 + 4 g_p^2)) / 2`.
/// * **All other modes** — the shift `lambda_n` is the most NEGATIVE root of
///   the remaining RFO secular equation, giving a downhill step in every one.
///   Solved here by the standard iteration, which converges from
///   `lambda_n = 0` because the function is monotone below the lowest
///   eigenvalue.
fn prfo_step(g_q: &Array1<f64>, evals: &Array1<f64>, follow: usize) -> Array1<f64> {
    let n = g_q.len();
    let mut dq = Array1::<f64>::zeros(n);

    // --- the mode we climb ---
    let b_p = evals[follow];
    let g_p = g_q[follow];
    let lambda_p = 0.5 * (b_p + (b_p * b_p + 4.0 * g_p * g_p).sqrt());
    let den_p = b_p - lambda_p;
    // den_p is <= 0 by construction (lambda_p >= b_p), and strictly negative
    // unless g_p == 0 AND b_p >= 0. Guard the degenerate case rather than
    // dividing by zero.
    dq[follow] = if den_p.abs() > 1e-14 {
        -g_p / den_p
    } else {
        0.0
    };

    // --- every other mode: one shared negative shift ---
    // Find lambda_n < min(b_i) solving  lambda = sum_i g_i^2 / (lambda - b_i).
    let mut b_min = f64::INFINITY;
    for i in 0..n {
        if i != follow && evals[i] < b_min {
            b_min = evals[i];
        }
    }
    let mut lambda_n = if b_min.is_finite() {
        (b_min - 1.0).min(0.0)
    } else {
        0.0
    };
    for _ in 0..100 {
        let mut f = 0.0;
        let mut df = 0.0;
        for i in 0..n {
            if i == follow {
                continue;
            }
            let d = lambda_n - evals[i];
            if d.abs() < 1e-14 {
                continue;
            }
            f += g_q[i] * g_q[i] / d;
            df -= g_q[i] * g_q[i] / (d * d);
        }
        // Solve lambda = f(lambda) as a fixed point with a Newton correction on
        // h(lambda) = lambda - f(lambda), h' = 1 - f'.
        let h = lambda_n - f;
        let dh = 1.0 - df;
        if dh.abs() < 1e-14 {
            break;
        }
        let step = h / dh;
        lambda_n -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    // Keep the shift below the lowest non-followed eigenvalue: that is the
    // condition that makes every one of these steps DOWNHILL. If the iteration
    // wandered above it, fall back to the safe bracket.
    if b_min.is_finite() && lambda_n >= b_min {
        lambda_n = b_min - 1.0e-3;
    }
    for i in 0..n {
        if i == follow {
            continue;
        }
        let den = evals[i] - lambda_n;
        if den.abs() > 1e-14 {
            dq[i] = -g_q[i] / den;
        }
    }
    dq
}

/// Bofill update of the Hessian: the Murtagh-Sargent / Powell-symmetric-Broyden
/// mixture that Bofill (J. Comput. Chem. 15, 1 (1994)) recommends for SADDLE
/// searches.
///
/// BFGS is wrong here because it preserves positive definiteness -- it would
/// drive the negative eigenvalue out of the Hessian, which is the one thing the
/// search depends on. Bofill makes no such guarantee, by design.
///
/// `dx` is the step taken, `dg` the resulting gradient change.
fn bofill_update(h: &Array2<f64>, dx: &Array1<f64>, dg: &Array1<f64>) -> Array2<f64> {
    let n = dx.len();
    // e = dg - H dx, the error the update must absorb.
    let e = dg - &h.dot(dx);
    let dx_dx = dx.dot(dx);
    if dx_dx < 1e-20 {
        return h.clone();
    }
    let e_dx = e.dot(dx);
    let e_e = e.dot(&e);

    // phi weights MS against PSB. phi -> 1 when e is parallel to dx (MS is
    // exact there); phi -> 0 when they are orthogonal (MS is singular there,
    // PSB is not). This interpolation is the whole content of Bofill's method.
    let phi = if e_e * dx_dx > 1e-20 {
        (e_dx * e_dx) / (e_e * dx_dx)
    } else {
        0.0
    };

    let mut out = h.clone();

    // Murtagh-Sargent: phi * e e^T / (e . dx)
    if phi > 0.0 && e_dx.abs() > 1e-14 {
        for i in 0..n {
            for j in 0..n {
                out[[i, j]] += phi * e[i] * e[j] / e_dx;
            }
        }
    }
    // Powell symmetric Broyden, weighted (1 - phi).
    if phi < 1.0 {
        let w = 1.0 - phi;
        for i in 0..n {
            for j in 0..n {
                out[[i, j]] += w
                    * ((e[i] * dx[j] + dx[i] * e[j]) / dx_dx
                        - e_dx * dx[i] * dx[j] / (dx_dx * dx_dx));
            }
        }
    }
    // Re-symmetrize against accumulated round-off.
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

/// Count negative eigenvalues among the PHYSICAL (non-trans/rot) modes.
///
/// The trans/rot modes were shifted up by `projected_eigen`, so they cannot be
/// counted here -- but the count still skips the lowest `n_tr` slots
/// defensively, because a shifted mode landing back near zero would otherwise
/// be silently counted as a real imaginary frequency.
fn count_negative(evals: &Array1<f64>, tol: f64) -> usize {
    evals.iter().filter(|&&v| v < -tol).count()
}

/// Search for a first-order saddle point by P-RFO.
///
/// `hessian_at` must return the Cartesian Hessian (Hartree/Bohr^2) at a given
/// geometry; `energy_gradient_at` the energy and the `3N` Cartesian gradient.
/// Passing them in keeps this driver independent of the SCF method, exactly as
/// [`crate::optimize`]'s core does -- and lets the tests drive it with an
/// analytic surface whose saddle is known in closed form.
///
/// # Errors
///
/// Returns `Err` when the initial geometry has NO negative projected mode.
/// P-RFO has nothing to climb from a minimum's basin, and producing a
/// confident-looking result from there would be a minimum mislabelled as a
/// transition state.
pub fn find_saddle<FE, FH>(
    mol: &Molecule,
    config: &SaddleConfig,
    mut energy_gradient_at: FE,
    mut hessian_at: FH,
) -> Result<SaddleResult, FerricError>
where
    FE: FnMut(&Molecule) -> Result<(f64, Array1<f64>), FerricError>,
    FH: FnMut(&Molecule) -> Result<Array2<f64>, FerricError>,
{
    let natoms = mol.atoms.len();
    if natoms < 2 {
        return Err(FerricError::General(format!(
            "saddle search needs at least 2 atoms, got {natoms}"
        )));
    }
    let n = 3 * natoms;
    let mass_au = atom_masses(mol)?;
    let is_linear = detect_linear(mol, &mass_au);
    let n_tr = if is_linear { 5 } else { 6 };
    if n <= n_tr {
        return Err(FerricError::General(format!(
            "saddle search: {n} coordinates with {n_tr} trans/rot modes leaves no \
             vibrational space to search"
        )));
    }

    let mut cur = mol.clone();
    let mut hess = hessian_at(&cur)?;
    let (mut energy, mut grad) = energy_gradient_at(&cur)?;

    // Eigenvalue scale below which a mode counts as numerically zero rather
    // than negative. The finite-difference Hessian's own noise floor is well
    // under this for any sane displacement.
    const EIG_TOL: f64 = 1.0e-6;

    // --- the refusal: no uphill direction means this is not a saddle basin ---
    {
        let (evals, _, _) = projected_eigen(&cur, &mass_au, &hess)?;
        let lowest = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        if count_negative(&evals, EIG_TOL) == 0 {
            return Err(FerricError::General(format!(
                "saddle search: the projected Hessian at the starting geometry has NO \
                 negative eigenvalue (lowest = {lowest:.6e} Hartree/Bohr^2), so there is \
                 no uphill direction to follow. This geometry is in a minimum's basin. \
                 Start nearer the barrier, or use a minimizer if a minimum is what you \
                 want -- P-RFO from here would return a minimum labelled as a \
                 transition state."
            )));
        }
    }

    let mut steps = 0usize;
    let mut converged = false;

    for step in 0..config.max_steps {
        steps = step + 1;

        let g_max = grad.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let g_rms = (grad.dot(&grad) / n as f64).sqrt();
        if g_max < config.g_max_thresh && g_rms < config.g_rms_thresh {
            converged = true;
            steps = step;
            break;
        }

        let (evals, evecs, _) = projected_eigen(&cur, &mass_au, &hess)?;

        // Work in the mass-weighted frame throughout: the eigenvectors are
        // mass-weighted, so the gradient must be too.
        let sqrt_m: Vec<f64> = (0..n).map(|i| mass_au[i / 3].sqrt()).collect();
        let g_mw: Array1<f64> = Array1::from_iter((0..n).map(|i| grad[i] / sqrt_m[i]));
        let g_q = evecs.t().dot(&g_mw);

        if config.follow_mode >= n {
            return Err(FerricError::General(format!(
                "saddle: follow_mode = {} but there are only {n} modes",
                config.follow_mode
            )));
        }
        let dq = prfo_step(&g_q, &evals, config.follow_mode);

        // Back to mass-weighted Cartesians, then undo the mass weighting.
        let dx_mw = evecs.dot(&dq);
        let mut dx: Array1<f64> = Array1::from_iter((0..n).map(|i| dx_mw[i] / sqrt_m[i]));

        // Trust radius on the Cartesian step, scaled uniformly so the step's
        // DIRECTION is preserved -- clipping components individually would
        // rotate the step away from the followed mode.
        let dx_norm = dx.dot(&dx).sqrt();
        if dx_norm > config.trust_radius {
            dx *= config.trust_radius / dx_norm;
        }

        let mut next = cur.clone();
        for a in 0..natoms {
            next.atoms[a].x += dx[3 * a];
            next.atoms[a].y += dx[3 * a + 1];
            next.atoms[a].zpos += dx[3 * a + 2];
        }

        let (e_next, g_next) = energy_gradient_at(&next)?;

        // Hessian for the next iteration: exact on schedule, Bofill otherwise.
        let recalc =
            config.hessian_recalc_every > 0 && (step + 1) % config.hessian_recalc_every == 0;
        hess = if recalc {
            hessian_at(&next)?
        } else {
            let dg = &g_next - &grad;
            bofill_update(&hess, &dx, &dg)
        };

        cur = next;
        energy = e_next;
        grad = g_next;
    }

    // --- final character check: a gradient-converged point is not yet a TS ---
    let final_hess = hessian_at(&cur)?;
    let (evals, evecs, _) = projected_eigen(&cur, &mass_au, &final_hess)?;
    let n_imaginary = count_negative(&evals, EIG_TOL);
    let lowest_eigenvalue = evals.iter().cloned().fold(f64::INFINITY, f64::min);

    let imaginary_mode = if n_imaginary == 1 {
        let idx = evals
            .iter()
            .enumerate()
            .find(|(_, &v)| v < -EIG_TOL)
            .map(|(i, _)| i)
            .expect("n_imaginary == 1 guarantees a negative eigenvalue exists");
        let sqrt_m: Vec<f64> = (0..n).map(|i| mass_au[i / 3].sqrt()).collect();
        // Un-mass-weight so the mode is a Cartesian displacement, matching
        // `FrequencyResult::normal_modes`' convention.
        Some(Array1::from_iter(
            (0..n).map(|i| evecs[[i, idx]] / sqrt_m[i]),
        ))
    } else {
        None
    };

    Ok(SaddleResult {
        mol: cur,
        energy,
        steps,
        converged,
        n_imaginary,
        imaginary_mode,
        lowest_eigenvalue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-atom molecule built through the normal parser, so masses, charge
    /// and multiplicity are exactly what production would see. Coordinates are
    /// Angstrom here (parse_xyz's convention) and land in Bohr internally.
    fn two_atoms(z_sep_angstrom: f64) -> Molecule {
        let xyz = format!("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 {z_sep_angstrom}\n");
        Molecule::parse_xyz(&xyz, 0, 1).expect("H2 must parse")
    }

    /// The DISCRIMINATING case: positive curvature in the followed mode.
    ///
    /// On a mode that is already negative, the plain Newton step `-g/b` and
    /// the RFO step point the SAME way, so a test built there cannot tell
    /// P-RFO from a minimizer -- MEASURED: replacing the RFO denominator with
    /// `b_p` left every test passing. That is the artifact-vs-physics trap, and
    /// the redesign is this test.
    ///
    /// Where they differ is `b_p > 0`: Newton steps DOWNHILL (toward the
    /// minimum in that mode), while RFO's shift `lambda_p > b_p` flips the
    /// denominator negative and keeps climbing. That is the entire reason
    /// P-RFO can walk out of a minimum's basin toward a ridge, and it is what
    /// this asserts.
    #[test]
    fn prfo_climbs_even_when_the_followed_mode_has_positive_curvature() {
        let evals = Array1::from(vec![0.4, 0.3, 0.8]); // ALL positive
        let g = Array1::from(vec![0.02, 0.01, -0.01]);
        let dq = prfo_step(&g, &evals, 0);

        let de0 = g[0] * dq[0] + 0.5 * evals[0] * dq[0] * dq[0];
        assert!(
            de0 > 0.0,
            "with b_p = +0.4 the followed mode must STILL climb (dE > 0); got \
             dE = {de0:e}, dq = {}. A plain Newton step -g/b would give \
             dq = {:+.4} and descend.",
            dq[0],
            -g[0] / evals[0]
        );
        // And it must differ from the Newton step in SIGN here, which is the
        // property no minimizer has.
        let newton = -g[0] / evals[0];
        assert!(
            dq[0] * newton < 0.0,
            "the RFO step ({:+.4}) must oppose the Newton step ({newton:+.4}) \
             when the followed curvature is positive",
            dq[0]
        );
    }

    #[test]
    fn prfo_step_climbs_the_followed_mode_and_descends_the_rest() {
        // The exactness anchor for the step formula, on a diagonal Hessian
        // where the right answer is known by inspection: mode 0 is negative
        // (the one to climb), modes 1-2 positive.
        //
        // Climbing means the step goes UPHILL along mode 0: with a negative
        // curvature and a positive gradient component, an uphill step is
        // POSITIVE (moving toward the maximum), whereas a minimizer's Newton
        // step -g/b would be positive too but for the wrong reason. The
        // discriminator is the SIGN OF THE ENERGY CHANGE along that mode,
        // checked below.
        let evals = Array1::from(vec![-0.5, 0.3, 0.8]);
        let g = Array1::from(vec![0.01, 0.02, -0.015]);
        let dq = prfo_step(&g, &evals, 0);

        // Second-order energy change along each mode: g_i dq_i + b_i dq_i^2 / 2.
        let de0 = g[0] * dq[0] + 0.5 * evals[0] * dq[0] * dq[0];
        assert!(
            de0 > 0.0,
            "the followed mode must go UPHILL, got dE = {de0:e} (dq = {})",
            dq[0]
        );
        for i in 1..3 {
            let de = g[i] * dq[i] + 0.5 * evals[i] * dq[i] * dq[i];
            assert!(
                de < 0.0,
                "mode {i} must go DOWNHILL, got dE = {de:e} (dq = {})",
                dq[i]
            );
        }
    }

    #[test]
    fn prfo_at_a_stationary_point_takes_no_step() {
        // Trivial limit: zero gradient means zero step, exactly.
        let evals = Array1::from(vec![-0.5, 0.3, 0.8]);
        let g = Array1::zeros(3);
        let dq = prfo_step(&g, &evals, 0);
        for (i, v) in dq.iter().enumerate() {
            assert_eq!(*v, 0.0, "mode {i} moved from a stationary point: {v:e}");
        }
    }

    #[test]
    fn bofill_reproduces_the_secant_condition() {
        // The defining property of a quasi-Newton update: H_new dx = dg.
        // Murtagh-Sargent satisfies it exactly, and Bofill reduces to MS when
        // the error is parallel to the step, so this is the anchor.
        let h = Array2::<f64>::eye(3) * 0.5;
        let dx = Array1::from(vec![0.1, -0.05, 0.02]);
        let dg = Array1::from(vec![0.09, -0.02, 0.05]);
        let h_new = bofill_update(&h, &dx, &dg);
        let residual = h_new.dot(&dx) - &dg;
        let r = residual.dot(&residual).sqrt();
        assert!(r < 1e-10, "secant condition violated, |H dx - dg| = {r:e}");
    }

    #[test]
    fn bofill_can_keep_a_negative_eigenvalue_where_bfgs_could_not() {
        // The property that makes Bofill the right choice: it does NOT force
        // positive definiteness. Starting from an indefinite Hessian and
        // applying an update consistent with that curvature, the negative
        // eigenvalue must survive -- if it did not, the saddle search would
        // lose its reaction coordinate after one step.
        let mut h = Array2::<f64>::zeros((3, 3));
        h[[0, 0]] = -0.4;
        h[[1, 1]] = 0.6;
        h[[2, 2]] = 0.7;
        let dx = Array1::from(vec![0.05, 0.01, 0.0]);
        let dg = h.dot(&dx); // perfectly consistent: update should barely move it
        let h_new = bofill_update(&h, &dx, &dg);
        let (evals, _) = h_new.eigh(UPLO::Lower).unwrap();
        assert!(
            evals.iter().any(|&v| v < -1e-8),
            "the negative eigenvalue was destroyed: {evals:?}"
        );
    }

    #[test]
    fn a_minimum_basin_is_refused_rather_than_searched() {
        // The central refusal. A purely positive-definite Hessian means there
        // is nothing to climb, and returning a "transition state" from here is
        // the failure this guard exists to prevent.
        let mol = two_atoms(0.74);
        let n = 6;
        let cfg = SaddleConfig::default();
        let err = find_saddle(
            &mol,
            &cfg,
            |_m| Ok((0.0, Array1::zeros(n))),
            // A stiff positive-definite Hessian along the bond: no negative mode.
            |_m| {
                let mut h = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    h[[i, i]] = 0.5;
                }
                Ok(h)
            },
        )
        .expect_err("a minimum basin must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("NO \n                 negative") || msg.contains("negative eigenvalue"),
            "the error must explain WHY: {msg}"
        );
    }

    /// THE END-TO-END ANCHOR: a surface whose saddle is known in closed form.
    ///
    /// Everything above tests a PIECE. This drives the whole `find_saddle`
    /// loop against an analytic potential and checks it lands on the right
    /// point, which is the only test that can catch a wiring error between the
    /// step, the update and the projection.
    ///
    /// The surface, in the H2 bond coordinate `r` (Bohr) with the molecule held
    /// on the z axis:
    ///
    ///     E(r) = -(r - r0)^2 / 2 * k        (an inverted parabola in r)
    ///
    /// which has its MAXIMUM at r = r0 and no other stationary point. A
    /// first-order saddle in the full 3N space, because every other direction
    /// is trans/rot and therefore projected out. The search must find r0 from
    /// a displaced start.
    #[test]
    fn find_saddle_locates_the_maximum_of_an_inverted_parabola() {
        const R0: f64 = 1.60; // Bohr, the known answer
        const K: f64 = 0.35;

        // Start displaced from the saddle so the search has to move.
        let mol = two_atoms(0.74); // ~1.40 Bohr, below R0

        let sep = |m: &Molecule| -> f64 {
            let dz = m.atoms[1].zpos - m.atoms[0].zpos;
            let dy = m.atoms[1].y - m.atoms[0].y;
            let dx = m.atoms[1].x - m.atoms[0].x;
            (dx * dx + dy * dy + dz * dz).sqrt()
        };

        // E = -K/2 (r - R0)^2, so dE/dr = -K (r - R0), directed along the bond.
        let eg = |m: &Molecule| -> Result<(f64, Array1<f64>), FerricError> {
            let r = sep(m);
            let e = -0.5 * K * (r - R0) * (r - R0);
            let dedr = -K * (r - R0);
            let ux = (m.atoms[1].x - m.atoms[0].x) / r;
            let uy = (m.atoms[1].y - m.atoms[0].y) / r;
            let uz = (m.atoms[1].zpos - m.atoms[0].zpos) / r;
            // Atom 1 moves +u, atom 0 moves -u.
            let g = Array1::from(vec![
                -dedr * ux,
                -dedr * uy,
                -dedr * uz,
                dedr * ux,
                dedr * uy,
                dedr * uz,
            ]);
            Ok((e, g))
        };

        // Exact Hessian of that surface: -K along the bond direction, in the
        // antisymmetric (stretch) combination.
        let hess = |m: &Molecule| -> Result<Array2<f64>, FerricError> {
            let r = sep(m);
            let u = [
                (m.atoms[1].x - m.atoms[0].x) / r,
                (m.atoms[1].y - m.atoms[0].y) / r,
                (m.atoms[1].zpos - m.atoms[0].zpos) / r,
            ];
            let mut h = Array2::<f64>::zeros((6, 6));
            for a in 0..3 {
                for b in 0..3 {
                    let v = -K * u[a] * u[b];
                    h[[a, b]] += v;
                    h[[3 + a, 3 + b]] += v;
                    h[[a, 3 + b]] -= v;
                    h[[3 + a, b]] -= v;
                }
            }
            Ok(h)
        };

        let cfg = SaddleConfig {
            max_steps: 200,
            trust_radius: 0.10,
            ..Default::default()
        };
        let res = find_saddle(&mol, &cfg, eg, hess).expect("the surface has a negative mode");

        assert!(
            res.converged,
            "gradient did not converge in {} steps",
            res.steps
        );
        let r_final = sep(&res.mol);
        assert!(
            (r_final - R0).abs() < 1e-3,
            "landed at r = {r_final:.6} Bohr, expected the known saddle at {R0}"
        );
        assert_eq!(
            res.n_imaginary, 1,
            "an inverted parabola has exactly one negative mode, got {}",
            res.n_imaginary
        );
        assert!(
            res.is_transition_state(),
            "converged on the known saddle but is_transition_state() said no"
        );
        assert!(
            res.imaginary_mode.is_some(),
            "exactly one imaginary mode must come with its eigenvector"
        );
    }

    /// The projection must SURVIVE a Hessian whose trans/rot block is dirty.
    ///
    /// A clean analytic Hessian has exactly-zero trans/rot curvature, so
    /// removing the projection changes nothing and a test built on one cannot
    /// see the difference -- MEASURED: deleting `P H P` left every test
    /// passing. A real finite-difference Hessian is NOT clean; its trans/rot
    /// eigenvalues come out at the displacement's noise floor and can be
    /// slightly NEGATIVE.
    ///
    /// This feeds a Hessian with a deliberately negative rigid-body
    /// translation. Without the projection there are two negative modes and
    /// the lowest is a translation, so "follow the lowest" would chase the
    /// molecule across space. With it, the translation is projected out and
    /// shifted up, leaving exactly the one physical negative mode.
    #[test]
    fn a_contaminated_translation_mode_is_projected_out_not_followed() {
        let mol = two_atoms(0.74);
        let mass = atom_masses(&mol).unwrap();

        // Physical part: negative curvature along the bond (the real TS mode).
        let mut h = Array2::<f64>::zeros((6, 6));
        const K: f64 = 0.35;
        // The bond lies along z, so the stretch is the (2, 5) antisymmetric pair.
        let (a, b) = (2usize, 2usize);
        h[[a, b]] += -K;
        h[[3 + a, 3 + b]] += -K;
        h[[a, 3 + b]] -= -K;
        h[[3 + a, b]] -= -K;
        // Contamination: a NEGATIVE curvature for translating both atoms along
        // x together. This is not physics, it is finite-difference noise, and
        // it is more negative than the real mode.
        const NOISE: f64 = -0.9;
        for i in [0usize, 3] {
            for j in [0usize, 3] {
                h[[i, j]] += NOISE;
            }
        }

        let (evals, _, _) = projected_eigen(&mol, &mass, &h).unwrap();
        let n_neg = count_negative(&evals, 1e-6);
        assert_eq!(
            n_neg, 1,
            "the contaminated translation must be projected out, leaving ONE \
             physical negative mode; got {n_neg} negatives in {evals:?}"
        );

        // Vacuity guard: WITHOUT projection this same Hessian really does have
        // two negative modes, so the assertion above is testing the projection
        // rather than restating an already-clean input.
        let sqrt_m: Vec<f64> = (0..6).map(|i| mass[i / 3].sqrt()).collect();
        let mut hw = Array2::<f64>::zeros((6, 6));
        for i in 0..6 {
            for j in 0..6 {
                hw[[i, j]] = h[[i, j]] / (sqrt_m[i] * sqrt_m[j]);
            }
        }
        let (raw_evals, _) = hw.eigh(UPLO::Lower).unwrap();
        assert!(
            count_negative(&raw_evals, 1e-6) >= 2,
            "the unprojected Hessian must have >= 2 negative modes or this test \
             proves nothing; got {raw_evals:?}"
        );

        // The `P H P` projection is SEPARATELY load-bearing from the shift, and
        // the negative COUNT above cannot see that: MEASURED, deleting `P H P`
        // and keeping only the shift still leaves exactly one negative mode,
        // because the shift alone dominates a contaminated rigid-body
        // direction.
        //
        // What it does NOT do is remove the CROSS terms between the trans/rot
        // subspace and everything else. Those leak into the retained spectrum:
        // with the projection every shifted mode sits at TR_SHIFT exactly
        // (1000.000), without it one comes back at 998.21. So this asserts the
        // shifted block is clean, which is the thing only the projection
        // achieves.
        const TR_SHIFT_EXPECTED: f64 = 1.0e3;
        let n_tr = if detect_linear(&mol, &mass) { 5 } else { 6 };
        let shifted: Vec<f64> = evals.iter().cloned().filter(|&v| v > 1.0).collect();
        assert_eq!(
            shifted.len(),
            n_tr,
            "expected {n_tr} shifted trans/rot modes, got {shifted:?}"
        );
        for v in &shifted {
            assert!(
                (v - TR_SHIFT_EXPECTED).abs() < 1e-6,
                "a shifted trans/rot mode came back at {v:.6} rather than \
                 {TR_SHIFT_EXPECTED:.1}, so contamination is leaking across the \
                 projection boundary -- P H P is not being applied"
            );
        }
    }

    #[test]
    fn is_transition_state_requires_both_halves() {
        // A gradient-converged point with the wrong number of imaginary modes
        // is not a transition state, and this is the accessor everything
        // downstream reads.
        let mol = two_atoms(0.74);
        let mk = |converged: bool, n_imaginary: usize| SaddleResult {
            mol: mol.clone(),
            energy: -1.0,
            steps: 3,
            converged,
            n_imaginary,
            imaginary_mode: None,
            lowest_eigenvalue: -0.1,
        };
        assert!(mk(true, 1).is_transition_state());
        assert!(!mk(true, 0).is_transition_state(), "0 imag = a minimum");
        assert!(
            !mk(true, 2).is_transition_state(),
            "2 imag = 2nd-order saddle"
        );
        assert!(
            !mk(false, 1).is_transition_state(),
            "gradient never converged"
        );
    }
}
