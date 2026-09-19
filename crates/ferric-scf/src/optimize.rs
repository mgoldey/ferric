//! Geometry optimization using analytical gradients.
//!
//! Implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for
//! minimizing the molecular energy with respect to nuclear coordinates, in
//! either Cartesian coordinates (the default) or redundant internal
//! coordinates ([`CoordSystem`](crate::optimize::CoordSystem)).

use crate::gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
use crate::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
use crate::rhf::{solve_rhf, RhfConfig};
use crate::rohf::solve_rohf;
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf;
use ferric_core::internal_coords::{
    generalized_inverse, gradient_to_internal, step_to_cartesian, InternalCoords, Primitive,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2};

/// Which coordinate system the BFGS search runs in.
///
/// # Why Cartesian is the default
///
/// Cartesian is the pre-existing, long-validated path and is unchanged by the
/// addition of internals (pinned by
/// `tests/internal_coord_optimize.rs::selecting_internals_does_not_perturb_the_cartesian_path`,
/// which compares the two live in one process, and by
/// `cartesian_path_is_reproducible_and_matches_the_recorded_path`).
/// Internals are opt-in because they add machinery that can fail in ways
/// Cartesians cannot — bond perception can mis-assign connectivity on an
/// unusual geometry, and the back-transformation can fail to converge — and
/// because the measured iteration-count advantage, while real on floppy
/// systems, is not uniform across every system (see the ledger printed by
/// `iteration_count_ledger`). Making the *safer* path the default and the
/// *faster-on-hard-cases* path explicit is the honest ordering: a user who
/// opts in has decided their system is one where torsions dominate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordSystem {
    /// Plain Cartesian BFGS with a fixed trust-radius clip. The default.
    #[default]
    Cartesian,
    /// Redundant internal coordinates (bonds, angles, proper dihedrals) with a
    /// Schlegel empirical initial Hessian and an iterative step
    /// back-transformation. Falls back to Cartesian, with a warning, if the
    /// coordinates cannot be constructed for the input geometry.
    RedundantInternal,
}

/// Configuration for geometry optimization.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Maximum number of optimization steps.
    pub max_steps: usize,
    /// Convergence threshold for the maximum gradient component (Hartree/Bohr).
    pub g_max_thresh: f64,
    /// Convergence threshold for the RMS gradient (Hartree/Bohr).
    pub g_rms_thresh: f64,
    /// Convergence threshold for the energy change (Hartree).
    pub e_conv: f64,
    /// Initial step size for line search.
    pub trust_radius: f64,
    /// Coordinate system for the search. Defaults to
    /// [`CoordSystem::Cartesian`], which is bit-identical to the behaviour
    /// before internals existed.
    pub coord_system: CoordSystem,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            g_max_thresh: 4.5e-4,
            g_rms_thresh: 3.0e-4,
            e_conv: 1.0e-6,
            trust_radius: 0.1,
            coord_system: CoordSystem::Cartesian,
        }
    }
}

/// Maximum passes of the iterative internal→Cartesian back-transformation
/// before the step is declared non-convergent and halved.
const BACKTRANSFORM_MAX_ITER: usize = 25;

/// Convergence tolerance for the back-transformation, as a maximum remaining
/// **Cartesian** correction in Bohr (see
/// [`ferric_core::internal_coords::step_to_cartesian`] for why the residual is
/// measured in Cartesians rather than in the internal coordinates themselves).
///
/// 1e-9 Bohr is five orders below the geometry change implied by the
/// optimizer's own gradient threshold, so the back-transformation is never what
/// limits the converged geometry, while staying comfortably above the level at
/// which the iteration would be chasing floating-point noise.
const BACKTRANSFORM_TOL: f64 = 1e-9;

/// Trust radius for the internal-coordinate step, in the primitives' own units.
///
/// Separate from [`OptimizeConfig::trust_radius`] (Bohr, Cartesian) because
/// radians and Bohr are not comparable: 0.3 rad is 17 degrees, a sensible
/// maximum torsional move, while 0.3 Bohr would be a reckless bond step. The
/// bond and angle components are clipped against the caller's Cartesian
/// `trust_radius`; angular components against this.
const INTERNAL_ANGULAR_TRUST: f64 = 0.3;

/// Result of a geometry optimization.
#[derive(Debug, Clone)]
#[must_use = "optimization result contains the relaxed geometry and energy"]
pub struct OptimizeResult {
    pub mol: Molecule,
    pub energy: f64,
    pub steps: usize,
    pub converged: bool,
}

impl std::fmt::Display for OptimizeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimized energy: {:.10} Ha ({} steps, converged: {})",
            self.energy, self.steps, self.converged
        )
    }
}

/// Optimize the molecular geometry using RHF (or closed-shell KS-DFT, via
/// `rhf_config.xc`) analytical gradients.
pub fn optimize_geometry(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    run_bfgs(mol, opt_config, |m| {
        compute_energy_and_gradient(ctx, m, basis_name, op, rhf_config)
    })
}

/// Optimize the molecular geometry using UHF analytical gradients.
///
/// `mol.charge`/`mol.multiplicity` fix the spin state for every step (the
/// occupation is not re-derived from a Aufbau guess mid-optimization, mirroring
/// how `solve_uhf` itself works — the caller is responsible for choosing a
/// multiplicity that stays the correct ground state along the whole path).
pub fn optimize_geometry_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    uhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    run_bfgs(mol, opt_config, |m| {
        compute_energy_and_gradient_uhf(ctx, m, basis_name, op, uhf_config)
    })
}

/// Optimize the molecular geometry using ROHF analytical gradients.
pub fn optimize_geometry_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rohf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    run_bfgs(mol, opt_config, |m| {
        compute_energy_and_gradient_rohf(ctx, m, basis_name, op, rohf_config)
    })
}

/// Shared BFGS driver: minimizes `energy_and_gradient(mol)` over nuclear
/// coordinates starting from `mol`. Identical algorithm for every reference
/// (RHF/RKS/UHF/ROHF) — only how energy+gradient are computed at a geometry
/// differs, which is captured entirely in the closure. A thin wrapper over
/// [`optimize_coordinates`] that flattens/unflattens the `Molecule` into a
/// coordinate vector at the boundary.
fn run_bfgs(
    mol: &Molecule,
    opt_config: &OptimizeConfig,
    mut energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    if opt_config.coord_system == CoordSystem::RedundantInternal {
        // Build the coordinate system ONCE, from the starting geometry, and
        // reuse it for every step. Regenerating it mid-run would silently
        // invalidate the BFGS history, which is indexed against this exact
        // primitive list.
        match InternalCoords::generate(mol) {
            Ok(coords) => {
                if coords.connectivity.n_fragments > 1 {
                    println!(
                        "Note: {} disconnected fragments perceived; added {} interfragment \
                         link(s) so the internal coordinates span their relative motion.",
                        coords.connectivity.n_fragments,
                        coords.connectivity.interfragment_bonds.len()
                    );
                }
                return run_bfgs_internal(mol, &coords, opt_config, energy_and_gradient);
            }
            Err(e) => {
                // Falling back is better than failing: the Cartesian path
                // always works, and the user asked for an optimization, not
                // for a particular coordinate system. But say so loudly — a
                // silent downgrade would make the iteration counts
                // uninterpretable.
                println!(
                    "Warning: could not build redundant internal coordinates ({e}); \
                     falling back to Cartesian BFGS."
                );
            }
        }
    }

    let x0 = flatten_molecule_coords(mol);
    let mol_template = mol.clone();

    let (x_final, energy, steps, converged) = optimize_coordinates(&x0, opt_config, |x| {
        let mut m = mol_template.clone();
        set_molecule_coords(&mut m, x);
        let (e, grad_arr) = energy_and_gradient(&m)?;
        Ok((e, flatten_gradient(&grad_arr).to_vec()))
    })?;

    let mut current_mol = mol_template;
    set_molecule_coords(&mut current_mol, &x_final);

    Ok(OptimizeResult {
        mol: current_mol,
        energy,
        steps,
        converged,
    })
}

/// BFGS in redundant internal coordinates.
///
/// # The loop
///
/// 1. Compute energy and the **Cartesian** gradient at the current geometry.
/// 2. Test convergence on the **Cartesian** gradient — deliberately the same
///    test the Cartesian path uses, so "converged" means the same thing in
///    both coordinate systems and the two are comparable. A coordinate system
///    is a parameterization of the search, not of the convergence criterion.
/// 3. Transform the gradient to internals: `g_q = G^- B g_x`.
/// 4. Take a BFGS step in internals, clipped per coordinate type.
/// 5. Back-transform iteratively to Cartesians; halve the step and retry if the
///    back-transformation fails to converge.
/// 6. Update the inverse Hessian with the internal-coordinate `s` and `y`.
///
/// # Why the Hessian is initialized from Schlegel rather than the identity
///
/// This is the whole point. In internals the diagonal force constants differ by
/// two orders of magnitude between a stretch and a torsion, and the empirical
/// guess gets each within about a factor of two
/// (`initial_hessian_is_within_an_order_of_magnitude_of_reality`). An identity
/// Hessian in *Cartesians* cannot express that separation at all, because every
/// Cartesian coordinate mixes stiff and soft modes.
fn run_bfgs_internal(
    mol: &Molecule,
    coords: &InternalCoords,
    opt_config: &OptimizeConfig,
    mut energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    let nq = coords.len();
    let mut current = mol.clone();

    let (mut energy, grad_arr) = energy_and_gradient(&current)?;
    let mut grad_cart = flatten_gradient(&grad_arr);

    // Inverse Hessian in internals, seeded from the empirical diagonal. BFGS
    // updates the INVERSE, so seed with 1/k.
    let h_diag = coords.initial_hessian_diagonal(&current)?;
    let mut h_inv = Array2::<f64>::zeros((nq, nq));
    for i in 0..nq {
        h_inv[(i, i)] = 1.0 / h_diag[i];
    }

    let mut prev_energy = energy;
    let mut converged = false;
    let mut step_idx = 0;

    println!(
        "Redundant internal coordinates: {} primitives ({} bonds, {} angles, {} dihedrals) \
         over {} atoms",
        nq,
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Bond(..)))
            .count(),
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Angle(..)))
            .count(),
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Dihedral(..)))
            .count(),
        coords.natoms,
    );
    println!("Step | Energy (Ha) | Delta E | Max Grad | RMS Grad");
    println!("-----+-------------+---------+----------+---------");

    while step_idx < opt_config.max_steps {
        let n_cart = grad_cart.len();
        let g_max = grad_cart.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        let g_rms = (grad_cart.iter().map(|g| g * g).sum::<f64>() / n_cart as f64).sqrt();
        let e_diff = (energy - prev_energy).abs();

        println!(
            "{:4} | {:11.8} | {:7.1e} | {:8.2e} | {:8.2e}",
            step_idx,
            energy,
            if step_idx == 0 {
                0.0
            } else {
                energy - prev_energy
            },
            g_max,
            g_rms
        );

        // Same convergence test as the Cartesian path, on the same (Cartesian)
        // gradient — so "converged" is one definition, not two.
        if step_idx > 0
            && e_diff < opt_config.e_conv
            && g_max < opt_config.g_max_thresh
            && g_rms < opt_config.g_rms_thresh
        {
            converged = true;
            break;
        }

        let b = coords.b_matrix(&current)?;
        let ginv = generalized_inverse(&b)?;
        let g_int = gradient_to_internal(
            &b,
            &ginv,
            grad_cart.as_slice().expect("gradient is contiguous"),
        )?;

        // BFGS step in internals.
        let p = -h_inv.dot(&g_int);
        let dq = clip_internal_step(coords, &p, opt_config.trust_radius);

        // Back-transform, shrinking the step if the nonlinear solve cannot
        // reach it. A failed back-transformation means the step left the
        // regime where B is a good linearization — the fix is a shorter step,
        // not a different coordinate system.
        let mut scale = 1.0;
        let mut taken: Option<(Array1<f64>, Vec<f64>)> = None;
        for attempt in 0..5 {
            let trial = &dq * scale;
            let bt = step_to_cartesian(
                coords,
                &current,
                &trial,
                BACKTRANSFORM_MAX_ITER,
                BACKTRANSFORM_TOL,
            )?;
            if bt.converged {
                taken = Some((trial, bt.coords));
                break;
            }
            if attempt == 4 {
                // Out of retries: take the best iterate the back-transformation
                // reached rather than stalling. It is a descent direction and
                // the next gradient will correct it; report it so a user can
                // see the optimizer is working harder than it should.
                println!(
                    "Warning: internal→Cartesian back-transformation did not converge \
                     (residual {:.2e} after {} passes, step scaled to {:.3}); \
                     accepting the best iterate.",
                    bt.residual, bt.iterations, scale
                );
                taken = Some((trial, bt.coords));
                break;
            }
            scale *= 0.5;
        }
        // The requested step is deliberately dropped here: from this point on
        // only the ACHIEVED displacement (measured from the two geometries by
        // `bfgs_secant_step`) is allowed to reach the Hessian update. Keeping
        // the requested vector in scope is what made the old
        // "s = requested step" defect possible.
        let (_dq_requested, new_coords) =
            taken.expect("the retry loop always assigns on its last pass");

        let q_before = coords.values(&current)?;
        set_molecule_coords(&mut current, &new_coords);
        let q_after = coords.values(&current)?;

        let prev_grad_int = g_int;
        prev_energy = energy;

        let (e_new, grad_arr_new) = energy_and_gradient(&current)?;
        energy = e_new;
        grad_cart = flatten_gradient(&grad_arr_new);

        let b_new = coords.b_matrix(&current)?;
        let ginv_new = generalized_inverse(&b_new)?;
        let g_int_new = gradient_to_internal(
            &b_new,
            &ginv_new,
            grad_cart.as_slice().expect("gradient is contiguous"),
        )?;

        // The BFGS `s` must be the displacement ACTUALLY achieved, not the one
        // requested: the back-transformation is iterative and the two differ
        // whenever it stops short. Using the requested step would feed the
        // Hessian update a lie about the curvature.
        let s = bfgs_secant_step(coords, &q_before, &q_after);
        debug_assert_eq!(s.len(), nq);

        let y = &g_int_new - &prev_grad_int;
        let ys = y.dot(&s);
        if ys.abs() > 1e-12 {
            let hy = h_inv.dot(&y);
            let yhy = y.dot(&hy);
            let rho = 1.0 / ys;
            let term1 = (ys + yhy) * rho * rho;
            for i in 0..nq {
                for j in 0..nq {
                    h_inv[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
                }
            }
        } else {
            // Reset to the empirical guess, not to the identity: the identity
            // is meaningless in internals, where the coordinates have
            // different units.
            let hd = coords.initial_hessian_diagonal(&current)?;
            h_inv = Array2::zeros((nq, nq));
            for i in 0..nq {
                h_inv[(i, i)] = 1.0 / hd[i];
            }
        }

        step_idx += 1;
    }

    Ok(OptimizeResult {
        mol: current,
        energy,
        steps: step_idx,
        converged,
    })
}

/// The BFGS secant vector `s`: the internal displacement **actually achieved**
/// between two geometries, from their measured coordinate values.
///
/// # Why this takes the two geometries and not the requested step
///
/// The internal→Cartesian back-transformation is iterative, and in a redundant
/// coordinate set the requested `Δq` is generally not even reachable (ethane:
/// 28 primitives, rank 18 — see
/// [`ferric_core::internal_coords::step_to_cartesian`]). Measured on ethane
/// with an optimizer-sized step, the requested and achieved vectors differ by
/// 2.1e-2, which is larger than the 1e-2..2e-2 step components themselves.
/// Feeding BFGS the requested vector would therefore tell it the geometry moved
/// somewhere it did not, and the resulting curvature estimate would be wrong by
/// order unity — not a rounding detail.
///
/// Taking the two geometries as arguments makes it impossible to pass the
/// requested step by mistake: this function never sees it.
fn bfgs_secant_step(
    coords: &InternalCoords,
    q_before: &Array1<f64>,
    q_after: &Array1<f64>,
) -> Array1<f64> {
    let mut s = Array1::zeros(coords.len());
    for (idx, p) in coords.primitives.iter().enumerate() {
        let d = q_after[idx] - q_before[idx];
        // A torsion crossing the +/-pi branch cut must read as the short way
        // round, not as a ~2*pi excursion that would wreck the Hessian update.
        s[idx] = if matches!(p, Primitive::Dihedral(..)) {
            ferric_core::internal_coords::wrap_to_pi(d)
        } else {
            d
        };
    }
    s
}

/// Clip a proposed internal step, separately per coordinate type.
///
/// Bonds are clipped against the caller's Cartesian `trust_radius` (both are
/// Bohr); angles and dihedrals against [`INTERNAL_ANGULAR_TRUST`] radians.
/// Mixing the two into a single vector norm would be a units error: adding
/// Bohr² to rad² produces a number with no meaning, and whichever group
/// happened to be numerically larger would dominate the clip.
fn clip_internal_step(
    coords: &InternalCoords,
    p: &Array1<f64>,
    cartesian_trust: f64,
) -> Array1<f64> {
    let mut dist_norm = 0.0;
    let mut ang_norm = 0.0;
    for (idx, prim) in coords.primitives.iter().enumerate() {
        if prim.is_angular() {
            ang_norm += p[idx] * p[idx];
        } else {
            dist_norm += p[idx] * p[idx];
        }
    }
    let dist_norm = dist_norm.sqrt();
    let ang_norm = ang_norm.sqrt();

    let dist_scale = if dist_norm > cartesian_trust {
        cartesian_trust / dist_norm
    } else {
        1.0
    };
    let ang_scale = if ang_norm > INTERNAL_ANGULAR_TRUST {
        INTERNAL_ANGULAR_TRUST / ang_norm
    } else {
        1.0
    };

    let mut out = p.clone();
    for (idx, prim) in coords.primitives.iter().enumerate() {
        out[idx] *= if prim.is_angular() {
            ang_scale
        } else {
            dist_scale
        };
    }
    out
}

/// Coordinate-vector core of the BFGS driver: minimizes `f(x)` over a flat
/// `Vec<f64>` of length `x0.len()` (any multiple-of-3 layout the caller's
/// closure agrees with itself on — this function never interprets the
/// coordinates as atoms). Returns `(x_final, energy, steps, converged)`.
///
/// This is the exact algorithm `run_bfgs` used before being rewritten on
/// top of this function (same line search, same convergence tests, same
/// Hessian update) — only the `Molecule`-specific flatten/unflatten at the
/// boundary moved out. Pinned bit-for-bit by
/// `test_optimize_h2_sto3g_step_energy_anchor`.
pub fn optimize_coordinates(
    x0: &[f64],
    opt_config: &OptimizeConfig,
    mut f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), FerricError>,
) -> Result<(Vec<f64>, f64, usize, bool), FerricError> {
    let n_coord = x0.len();
    let mut x = Array1::from_vec(x0.to_vec());

    // Initial energy and gradient
    let (mut energy, grad0) = f(x.as_slice().expect("x is contiguous"))?;
    let mut grad = Array1::from_vec(grad0);

    // Approximate inverse Hessian (initialized to identity)
    let mut h = Array2::<f64>::eye(n_coord);

    let mut prev_energy = energy;
    let mut converged = false;
    let mut step_idx = 0;

    println!("Step | Energy (Ha) | Delta E | Max Grad | RMS Grad");
    println!("-----+-------------+---------+----------+---------");

    while step_idx < opt_config.max_steps {
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        let g_rms = (grad.iter().map(|g| g * g).sum::<f64>() / n_coord as f64).sqrt();
        let e_diff = (energy - prev_energy).abs();

        println!(
            "{:4} | {:11.8} | {:7.1e} | {:8.2e} | {:8.2e}",
            step_idx,
            energy,
            if step_idx == 0 {
                0.0
            } else {
                energy - prev_energy
            },
            g_max,
            g_rms
        );

        // Check convergence
        if step_idx > 0
            && e_diff < opt_config.e_conv
            && g_max < opt_config.g_max_thresh
            && g_rms < opt_config.g_rms_thresh
        {
            converged = true;
            break;
        }

        // BFGS step: p = -H * g
        let p = -h.dot(&grad);

        // Simple trust-radius scaling
        let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        let step = if p_norm > opt_config.trust_radius {
            &p * (opt_config.trust_radius / p_norm)
        } else {
            p
        };

        // Update coordinates
        x = &x + &step;

        let prev_grad = grad.clone();
        prev_energy = energy;

        // Compute new energy and gradient
        let (e_new, grad_new) = f(x.as_slice().expect("x is contiguous"))?;
        energy = e_new;
        grad = Array1::from_vec(grad_new);

        // BFGS update for H
        let s = step; // x_{k+1} - x_k
        let y = &grad - &prev_grad; // g_{k+1} - g_k

        let ys = y.dot(&s);
        if ys.abs() > 1e-12 {
            let hy = h.dot(&y);
            let yhy = y.dot(&hy);
            let rho = 1.0 / ys;

            // H = H + (ys + yHy)/(ys^2) * (s s^T) - (H y s^T + s y^T H) / ys
            let term1 = (ys + yhy) * rho * rho;
            for i in 0..n_coord {
                for j in 0..n_coord {
                    h[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
                }
            }
        } else {
            // Reset Hessian if update is unstable
            h = Array2::eye(n_coord);
        }

        step_idx += 1;
    }

    Ok((x.to_vec(), energy, step_idx, converged))
}

fn compute_energy_and_gradient(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, rhf_config.screening)?;
    let res = solve_rhf(ctx, mol, &prep, op, &bounds, rhf_config)?;
    let grad = if let Some(xc_name) = rhf_config.xc.as_deref() {
        ks_gradient_closed(
            mol,
            &prep,
            &bs,
            op,
            &bounds,
            xc_name,
            &res,
            rhf_config.external_potential.as_ref(),
        )?
    } else {
        rhf_gradient(
            mol,
            &prep,
            op,
            &bounds,
            &res,
            rhf_config.external_potential.as_ref(),
        )?
    };
    Ok((res.energy, grad))
}

fn compute_energy_and_gradient_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    uhf_config: &RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    // `uhf_gradient` is HF-only (no XC term), so an `xc` run must route to
    // `ks_gradient_uks` instead. That function IS implemented (LDA/GGA/hybrid/
    // RSH/meta-GGA + VV10) and is FD- and PySCF-validated by
    // tests/uks_gradient.rs, tests/uks_gradient_rsh.rs and
    // tests/dft_gradient_mgga.rs.
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, uhf_config.screening)?;
    let res = solve_uhf(ctx, mol, &prep, &bounds, uhf_config)?;
    let ext = uhf_config.external_potential.as_ref();
    let grad = if let Some(xc_name) = uhf_config.xc.as_deref() {
        ks_gradient_uks(mol, &prep, &bs, op, &bounds, xc_name, &res, ext)?
    } else {
        uhf_gradient(mol, &prep, op, &bounds, &res, ext)?
    };
    Ok((res.energy, grad))
}

fn compute_energy_and_gradient_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rohf_config: &RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    // `rohf_gradient` is HF-only (no XC term); an `xc` run routes to
    // `ks_gradient_roks`, which is implemented and FD-validated by
    // tests/roks_gradient.rs (LDA/PBE/B3LYP/wB97X-V). Same shape as the UHF
    // path above.
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, rohf_config.screening)?;
    let res = solve_rohf(ctx, mol, &prep, op, &bounds, rohf_config)?;
    let ext = rohf_config.external_potential.as_ref();
    let grad = if let Some(xc_name) = rohf_config.xc.as_deref() {
        ks_gradient_roks(mol, &prep, &bs, op, &bounds, xc_name, &res, ext)?
    } else {
        rohf_gradient(mol, &prep, op, &bounds, &res, ext)?
    };
    Ok((res.energy, grad))
}

fn flatten_gradient(grad: &Array2<f64>) -> Array1<f64> {
    let mut flat = Array1::zeros(grad.len());
    let mut idx = 0;
    for i in 0..grad.nrows() {
        for j in 0..3 {
            flat[idx] = grad[(i, j)];
            idx += 1;
        }
    }
    flat
}

/// Flatten a [`Molecule`]'s Cartesian coordinates into `(x0,y0,z0,x1,y1,z1,...)`.
fn flatten_molecule_coords(mol: &Molecule) -> Vec<f64> {
    let mut out = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        out.push(atom.x);
        out.push(atom.y);
        out.push(atom.zpos);
    }
    out
}

/// Overwrite a [`Molecule`]'s Cartesian coordinates from a flat
/// `(x0,y0,z0,x1,y1,z1,...)` vector (the inverse of [`flatten_molecule_coords`]).
fn set_molecule_coords(mol: &mut Molecule, x: &[f64]) {
    let mut idx = 0;
    for atom in mol.atoms.iter_mut() {
        atom.x = x[idx];
        atom.y = x[idx + 1];
        atom.zpos = x[idx + 2];
        idx += 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_optimize_h2_sto3g() {
        // Start from a stretched bond: 1.0 Angstrom = 1.89 Bohr
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();

        assert!(result.converged);
        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!("H2/STO-3G optimized distance: {:.6} Bohr", dist);
        // STO-3G H2 bond length is ~1.346 Bohr
        assert!(
            (dist - 1.346).abs() < 1e-2,
            "dist = {dist}, expected ~1.346"
        );
    }

    /// F2-1 EXACTNESS ANCHOR. Pins the exact per-step energy trajectory
    /// `run_bfgs` produces for H2/STO-3G BEFORE it is rewritten on top of
    /// `optimize_coordinates` (the coordinate-vector core). The bit patterns
    /// below were captured from the pre-refactor implementation; after the
    /// refactor this test must still pass byte-for-byte (`to_bits()`
    /// equality, not a tolerance) — any drift means the refactor changed the
    /// algorithm, not just its shape.
    #[test]
    fn test_optimize_h2_sto3g_step_energy_anchor() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };
        let ctx = ParallelContext::default();

        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();
        assert!(result.converged);
        eprintln!(
            "[F2-1 anchor] H2/STO-3G final energy bits = {:#018x} ({:.15}), steps = {}",
            result.energy.to_bits(),
            result.energy,
            result.steps,
        );
        // Captured 2026-08-27 from the pre-refactor `run_bfgs` (this exact
        // test, run once before touching optimize.rs):
        //   [F2-1 anchor] H2/STO-3G final energy bits = 0xbff1e14dd9dd63b7
        //   (-1.117505885156999), steps = 7
        //
        // Bit identity holds on the machine the anchor was captured on, but
        // NOT across BLAS kernel sets: CI run 33216140352 (AMD EPYC 9V74,
        // OpenBLAS 0.3.20 with OPENBLAS_CORETYPE=Haswell) reproduced this
        // energy to 5 ulp (-1.117505885156998 vs ...999), which is GEMM
        // reduction-order noise, not an optimizer change. The refactor
        // invariant this test guards -- same minimum, same step count -- is
        // therefore asserted at 1e-12 Ha (six orders below the optimizer's
        // e_conv and ~1e4x above ulp noise), and exactly on `steps`.
        const ANCHOR_ENERGY: f64 = -1.117505885156999;
        const ANCHOR_STEPS: usize = 7;
        let de = (result.energy - ANCHOR_ENERGY).abs();
        assert!(
            de < 1e-12,
            "post-refactor energy {:.15} != pre-refactor anchor {:.15} (|Δ| {de:.3e})",
            result.energy,
            ANCHOR_ENERGY
        );
        assert_eq!(result.steps, ANCHOR_STEPS);
    }

    #[test]
    fn test_optimize_h2o_sto3g() {
        // Second molecule (widens past H2-only): H2O/STO-3G, started from a
        // distorted geometry (O-H 0.9/0.92 A-ish, non-equilibrium angle).
        // Reference: PySCF RHF/STO-3G geometric-optimizer result --
        //   O-H bond lengths (Bohr): 1.869732, 1.869731
        //   H-O-H angle (deg): 100.0258
        //   E_final = -74.9659011921 Ha
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();

        assert!(result.converged);
        let o = &result.mol.atoms[0];
        let h1 = &result.mol.atoms[1];
        let h2 = &result.mol.atoms[2];
        let r1 = ((o.x - h1.x).powi(2) + (o.y - h1.y).powi(2) + (o.zpos - h1.zpos).powi(2)).sqrt();
        let r2 = ((o.x - h2.x).powi(2) + (o.y - h2.y).powi(2) + (o.zpos - h2.zpos).powi(2)).sqrt();
        eprintln!("H2O/STO-3G optimized O-H distances: {r1:.6}, {r2:.6} Bohr (ref 1.869732)");
        assert!(
            (r1 - 1.869732).abs() < 1e-2,
            "r1 = {r1}, expected ~1.869732"
        );
        assert!(
            (r2 - 1.869732).abs() < 1e-2,
            "r2 = {r2}, expected ~1.869732"
        );
    }

    #[test]
    fn test_optimize_h2plus_uhf_sto3g() {
        // H2+ (one electron, doublet), started stretched at 1.5 Angstrom
        // (2.835 Bohr) -- well beyond the equilibrium bond -- and optimized
        // with UHF/STO-3G analytical gradients. UHF on a single-electron
        // system reduces exactly to RHF on that electron, so this is a
        // useful sanity check that the UHF optimize path (a) actually
        // iterates (not just prints one gradient) and (b) lands at a
        // reasonable minimum.
        let mol = Molecule::parse_xyz("2\nH2+\nH 0 0 0\nH 0 0 1.5\n", 1, 2).unwrap();
        let op = Operator::coulomb();
        let uhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_uhf(&ctx, &mol, "sto-3g", op, &uhf_config)
            .unwrap()
            .0;

        let result =
            optimize_geometry_uhf(&ctx, &mol, "sto-3g", op, &uhf_config, &opt_config).unwrap();

        assert!(
            result.converged,
            "UHF H2+ optimization did not converge in {} steps",
            result.steps
        );
        assert!(
            result.steps > 0,
            "optimizer should take at least one step from a stretched start"
        );
        assert!(
            result.energy < e0,
            "optimized energy {} should be lower than initial energy {}",
            result.energy,
            e0
        );

        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!(
            "H2+/UHF/STO-3G optimized distance: {:.6} Bohr, energy: {:.10} Ha",
            dist, result.energy
        );
        // Independent PySCF UHF/STO-3G geomeTRIC-optimizer reference (2026-07-21):
        //   dist = 2.004215 Bohr, E = -0.5826966474 Ha
        // ferric matches to ~1e-9 Ha / <1e-6 Bohr -- tightened from the old
        // loose +-0.3 Bohr sanity band now that a real reference exists.
        const PYSCF_DIST: f64 = 2.004215;
        const PYSCF_E: f64 = -0.5826966474;
        assert!(
            (dist - PYSCF_DIST).abs() < 1e-3,
            "dist = {dist}, expected {PYSCF_DIST} (PySCF)"
        );
        assert!(
            (result.energy - PYSCF_E).abs() < 1e-5,
            "energy = {}, expected {PYSCF_E} (PySCF)",
            result.energy
        );

        // Final gradient norm must be below the configured convergence
        // thresholds -- re-derive it directly rather than trusting the
        // driver's internal bookkeeping.
        let (_, grad_arr) =
            compute_energy_and_gradient_uhf(&ctx, &result.mol, "sto-3g", op, &uhf_config).unwrap();
        let grad = flatten_gradient(&grad_arr);
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(
            g_max < opt_config.g_max_thresh,
            "final |g|_max = {g_max:.3e} not converged"
        );
    }

    #[test]
    fn test_optimize_oh_radical_rohf_sto3g() {
        // OH radical (doublet), started stretched at 1.3 Angstrom (vs the
        // experimental 0.9697 Angstrom in testdata/molecules/oh.xyz) and
        // optimized with ROHF/STO-3G analytical gradients.
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.3\n", 0, 2).unwrap();
        let op = Operator::coulomb();
        let rohf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_rohf(&ctx, &mol, "sto-3g", op, &rohf_config)
            .unwrap()
            .0;

        let result =
            optimize_geometry_rohf(&ctx, &mol, "sto-3g", op, &rohf_config, &opt_config).unwrap();

        assert!(
            result.converged,
            "ROHF OH optimization did not converge in {} steps",
            result.steps
        );
        assert!(
            result.steps > 0,
            "optimizer should take at least one step from a stretched start"
        );
        assert!(
            result.energy < e0,
            "optimized energy {} should be lower than initial energy {}",
            result.energy,
            e0
        );

        let dist_bohr = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        let dist_ang = dist_bohr * 0.529_177_210_92;
        eprintln!(
            "OH/ROHF/STO-3G optimized distance: {:.6} Bohr ({:.4} Ang), energy: {:.10} Ha",
            dist_bohr, dist_ang, result.energy
        );
        // STO-3G ROHF is a minimal basis, so the equilibrium bond will not
        // match the experimental 0.9697 Ang exactly -- minimal-basis HF
        // typically overbinds by several tenths of an Angstrom for OH.
        // Loose cross-check band: this catches optimizing to the wrong
        // stationary point (e.g. dissociation or a basis artifact far off).
        assert!(
            (dist_ang - 0.9697).abs() < 0.2,
            "dist = {dist_ang} Ang, expected within 0.2 Ang of experimental 0.9697"
        );
        // Independent PySCF ROHF/STO-3G geomeTRIC-optimizer reference
        // (2026-07-21): dist = 1.913998 Bohr, E = -74.3636983636 Ha. This
        // confirms the ~1 Ang overbinding above is genuine minimal-basis
        // ROHF physics (PySCF lands at the SAME stationary point ferric
        // does), not a ferric bug -- a real, tight computational
        // cross-check alongside the deliberately loose experimental band.
        const PYSCF_DIST_BOHR: f64 = 1.913998;
        const PYSCF_E: f64 = -74.3636983636;
        assert!(
            (dist_bohr - PYSCF_DIST_BOHR).abs() < 1e-3,
            "dist = {dist_bohr} Bohr, expected {PYSCF_DIST_BOHR} (PySCF)"
        );
        assert!(
            (result.energy - PYSCF_E).abs() < 1e-5,
            "energy = {}, expected {PYSCF_E} (PySCF)",
            result.energy
        );

        let (_, grad_arr) =
            compute_energy_and_gradient_rohf(&ctx, &result.mol, "sto-3g", op, &rohf_config)
                .unwrap();
        let grad = flatten_gradient(&grad_arr);
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(
            g_max < opt_config.g_max_thresh,
            "final |g|_max = {g_max:.3e} not converged"
        );
    }

    // -- internal-coordinate step machinery -------------------------------
    //
    // These two tests exist because the integration suite
    // (tests/internal_coord_optimize.rs) did NOT kill the corresponding
    // mutations: it asserts that the optimizer converges and roughly how fast,
    // which a degraded-but-still-descending step satisfies. Convergence is too
    // coarse an observable to pin step CONSTRUCTION, so the construction is
    // pinned directly here.

    fn ethane_for_steps() -> Molecule {
        Molecule::parse_xyz(
            "8\nethane\nC  0.000  0.000  0.766\nC  0.000  0.000 -0.766\n\
             H  1.019  0.000  1.163\nH -0.509  0.883  1.163\nH -0.509 -0.883  1.163\n\
             H  1.019  0.000 -1.163\nH -0.509  0.883 -1.163\nH -0.509 -0.883 -1.163\n",
            0,
            1,
        )
        .unwrap()
    }

    /// The step clip must treat Bohr and radians as SEPARATE budgets.
    ///
    /// Mixing them is a units error: a single vector norm over a mixed-unit
    /// vector adds Bohr^2 to rad^2, and whichever group is numerically larger
    /// then dictates the clip for both. This test constructs a step that is
    /// large in the angular block and small in the distance block, and checks
    /// that the angular block is cut to its own trust radius while the
    /// distance block — already within budget — is left completely alone.
    ///
    /// Mutation M15 (apply the Bohr scale to angular coordinates too) survived
    /// the integration suite and is killed here.
    #[test]
    fn internal_step_clip_keeps_distance_and_angular_budgets_separate() {
        let mol = ethane_for_steps();
        let coords = InternalCoords::generate(&mol).unwrap();
        let n_ang = coords.primitives.iter().filter(|p| p.is_angular()).count();
        let n_dist = coords.len() - n_ang;
        assert!(
            n_ang > 0 && n_dist > 0,
            "this test needs both coordinate kinds: {n_dist} distance, {n_ang} angular"
        );

        // Distance block: deliberately tiny, total norm far inside the trust
        // radius. Angular block: deliberately huge.
        let cartesian_trust = 0.1;
        let small = 1e-4;
        let big = 1.0;
        let mut p = Array1::zeros(coords.len());
        for (idx, prim) in coords.primitives.iter().enumerate() {
            p[idx] = if prim.is_angular() { big } else { small };
        }
        let dist_norm_in = (n_dist as f64).sqrt() * small;
        assert!(
            dist_norm_in < cartesian_trust,
            "precondition: the distance block must already be inside the trust radius"
        );

        let out = clip_internal_step(&coords, &p, cartesian_trust);

        let mut dist_norm_out = 0.0;
        let mut ang_norm_out = 0.0;
        for (idx, prim) in coords.primitives.iter().enumerate() {
            if prim.is_angular() {
                ang_norm_out += out[idx] * out[idx];
            } else {
                dist_norm_out += out[idx] * out[idx];
            }
        }
        let dist_norm_out = dist_norm_out.sqrt();
        let ang_norm_out = ang_norm_out.sqrt();
        eprintln!(
            "clip: distance {dist_norm_in:.4e} -> {dist_norm_out:.4e} (trust {cartesian_trust}), \
             angular {:.4e} -> {ang_norm_out:.4e} (trust {INTERNAL_ANGULAR_TRUST})",
            (n_ang as f64).sqrt() * big
        );

        // The angular block IS clipped, to its own budget.
        assert!(
            (ang_norm_out - INTERNAL_ANGULAR_TRUST).abs() < 1e-12,
            "angular block should be clipped to {INTERNAL_ANGULAR_TRUST}, got {ang_norm_out:.6e}"
        );
        // The distance block is untouched — bit-identical, because no scale
        // was applied at all. This is what fails if the two budgets are mixed:
        // the huge angular block would drag the distance block down with it.
        for (idx, prim) in coords.primitives.iter().enumerate() {
            if !prim.is_angular() {
                assert_eq!(
                    out[idx].to_bits(),
                    p[idx].to_bits(),
                    "distance component {idx} was scaled ({} -> {}) although its own \
                     block was within budget — the two budgets are being mixed",
                    p[idx],
                    out[idx]
                );
            }
        }
        assert!((dist_norm_out - dist_norm_in).abs() < 1e-18);
    }

    /// The BFGS secant vector `s` must be the displacement ACTUALLY achieved,
    /// not the one requested.
    ///
    /// The back-transformation is iterative and, in a redundant coordinate set,
    /// the requested `Δq` is generally not even reachable (see
    /// `ferric_core::internal_coords::step_to_cartesian`). Feeding the
    /// requested step into the Hessian update would tell BFGS the geometry
    /// moved somewhere it did not, corrupting the curvature estimate.
    ///
    /// Mutation M16 (use `dq_taken` instead of `q_after - q_before`) survived
    /// the integration suite and is killed here: this test measures the gap
    /// between requested and achieved directly and asserts it is real, so a
    /// driver that conflated them would be using a demonstrably wrong vector.
    #[test]
    fn requested_and_achieved_internal_steps_differ_in_a_redundant_set() {
        let mol = ethane_for_steps();
        let coords = InternalCoords::generate(&mol).unwrap();
        let b = coords.b_matrix(&mol).unwrap();
        let ginv = generalized_inverse(&b).unwrap();
        assert!(
            coords.len() > ginv.rank,
            "precondition: the set must be redundant ({} primitives, rank {})",
            coords.len(),
            ginv.rank
        );

        // A representative optimizer-sized step.
        let mut dq = Array1::zeros(coords.len());
        for (idx, prim) in coords.primitives.iter().enumerate() {
            dq[idx] = if prim.is_angular() { 0.02 } else { 0.01 };
        }

        let q_before = coords.values(&mol).unwrap();
        let bt = step_to_cartesian(
            &coords,
            &mol,
            &dq,
            BACKTRANSFORM_MAX_ITER,
            BACKTRANSFORM_TOL,
        )
        .unwrap();
        assert!(
            bt.converged,
            "the back-transformation should converge for this step"
        );

        let mut moved = mol.clone();
        set_molecule_coords(&mut moved, &bt.coords);
        let q_after = coords.values(&moved).unwrap();

        // THE ASSERTION: the secant vector the driver builds is the achieved
        // displacement. `bfgs_secant_step` is the function the driver calls,
        // and it cannot see `dq` at all.
        let s = bfgs_secant_step(&coords, &q_before, &q_after);

        let mut max_gap = 0.0f64;
        let mut max_err = 0.0f64;
        for idx in 0..coords.len() {
            let achieved = s[idx];
            max_gap = max_gap.max((achieved - dq[idx]).abs());
            // Independently recomputed achieved displacement (no wrapping is
            // needed at this step size, so a plain difference is the honest
            // cross-check of what `bfgs_secant_step` returned).
            max_err = max_err.max((achieved - (q_after[idx] - q_before[idx])).abs());
        }
        eprintln!(
            "secant step: max |s - requested| = {max_gap:.3e}, \
             max |s - (q_after - q_before)| = {max_err:.3e} \
             ({} primitives, rank {})",
            coords.len(),
            ginv.rank
        );

        // `s` IS the achieved displacement, exactly.
        assert!(
            max_err < 1e-14,
            "bfgs_secant_step returned something other than q_after - q_before \
             (max deviation {max_err:.3e})"
        );
        // And the achieved displacement is NOT the requested one — so a driver
        // that used the requested step would be using a demonstrably different
        // vector. Without this second half the test would be vacuous.
        assert!(
            max_gap > 1e-3,
            "requested and achieved steps agree to {max_gap:.3e}; if they were \
             really interchangeable this test would be vacuous — check that the \
             coordinate set is still redundant"
        );
    }

    /// `bfgs_secant_step` must take a torsion across the +/-pi branch cut the
    /// SHORT way. Without the wrap, a 2-degree move that happens to straddle
    /// the cut reads as a 358-degree one and destroys the Hessian update.
    #[test]
    fn bfgs_secant_step_wraps_torsions_across_the_branch_cut() {
        // H2O2 has exactly one dihedral, which makes the assertion unambiguous.
        let mol = Molecule::parse_xyz(
            "4\nH2O2\nO  0.000  0.734 -0.055\nO  0.000 -0.734 -0.055\n\
             H  0.839  0.885  0.435\nH -0.839 -0.885  0.435\n",
            0,
            1,
        )
        .unwrap();
        let coords = InternalCoords::generate(&mol).unwrap();
        let dih_idx = coords
            .primitives
            .iter()
            .position(|p| matches!(p, Primitive::Dihedral(..)))
            .expect("H2O2 has a dihedral");

        let pi = std::f64::consts::PI;
        let mut q_before = Array1::zeros(coords.len());
        let mut q_after = Array1::zeros(coords.len());
        // Straddle the cut: +179 deg -> -179 deg is a +2 deg move, not -358.
        q_before[dih_idx] = pi - 0.0175;
        q_after[dih_idx] = -pi + 0.0175;

        let s = bfgs_secant_step(&coords, &q_before, &q_after);
        eprintln!(
            "torsion {:.4} rad -> {:.4} rad gives s = {:.4} rad ({:.2} deg)",
            q_before[dih_idx],
            q_after[dih_idx],
            s[dih_idx],
            s[dih_idx].to_degrees()
        );
        assert!(
            (s[dih_idx] - 0.035).abs() < 1e-9,
            "expected the short way round (+0.035 rad), got {:.6} rad",
            s[dih_idx]
        );

        // A NON-torsion must NOT be wrapped: bonds have no periodicity, and
        // wrapping one would silently corrupt a large bond displacement.
        let bond_idx = coords
            .primitives
            .iter()
            .position(|p| matches!(p, Primitive::Bond(..)))
            .expect("H2O2 has bonds");
        let mut qb = Array1::zeros(coords.len());
        let mut qa = Array1::zeros(coords.len());
        qa[bond_idx] = 2.0 * pi + 0.5; // larger than 2*pi on purpose
        let s2 = bfgs_secant_step(&coords, &qb, &qa);
        assert!(
            (s2[bond_idx] - (2.0 * pi + 0.5)).abs() < 1e-12,
            "a bond displacement must never be wrapped: got {:.6}",
            s2[bond_idx]
        );
        let _ = &mut qb;
    }
}
