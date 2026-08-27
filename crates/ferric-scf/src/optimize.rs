//! Geometry optimization using analytical gradients.
//!
//! Implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for
//! minimizing the molecular energy with respect to nuclear coordinates.

use crate::gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
use crate::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
use crate::rhf::{solve_rhf, RhfConfig};
use crate::rohf::solve_rohf;
use crate::uhf::solve_uhf;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_core::parallel::ParallelContext;
use ndarray::{Array1, Array2};

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
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            g_max_thresh: 4.5e-4,
            g_rms_thresh: 3.0e-4,
            e_conv: 1.0e-6,
            trust_radius: 0.1,
        }
    }
}

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
        write!(f, "Optimized energy: {:.10} Ha ({} steps, converged: {})",
            self.energy, self.steps, self.converged)
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
    let x0 = flatten_molecule_coords(mol);
    let mol_template = mol.clone();

    let (x_final, energy, steps, converged) =
        optimize_coordinates(&x0, opt_config, |x| {
            let mut m = mol_template.clone();
            set_molecule_coords(&mut m, x);
            let (e, grad_arr) = energy_and_gradient(&m)?;
            Ok((e, flatten_gradient(&grad_arr).to_vec()))
        })?;

    let mut current_mol = mol_template;
    set_molecule_coords(&mut current_mol, &x_final);

    Ok(OptimizeResult { mol: current_mol, energy, steps, converged })
}

/// Coordinate-vector core of the BFGS driver: minimizes `f(x)` over a flat
/// `Vec<f64>` of length `x0.len()` (any multiple-of-3 layout the caller's
/// closure agrees with itself on — this function never interprets the
/// coordinates as atoms). Returns `(x_final, energy, steps, converged)`.
///
/// This is the exact algorithm [`run_bfgs`] used before being rewritten on
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
            step_idx, energy, if step_idx == 0 { 0.0 } else { energy - prev_energy }, g_max, g_rms
        );

        // Check convergence
        if step_idx > 0 && e_diff < opt_config.e_conv && g_max < opt_config.g_max_thresh && g_rms < opt_config.g_rms_thresh {
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
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let res = solve_rhf(ctx, mol, &prep, op, &bounds, rhf_config)?;
    let grad = if let Some(xc_name) = rhf_config.xc.as_deref() {
        ks_gradient_closed(mol, &prep, &bs, op, &bounds, xc_name, &res, rhf_config.external_potential.as_ref())?
    } else {
        rhf_gradient(mol, &prep, op, &bounds, &res, rhf_config.external_potential.as_ref())?
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
    let bounds = SchwarzBounds::compute(op, &prep)?;
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
    let bounds = SchwarzBounds::compute(op, &prep)?;
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
        let rhf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
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
        assert!((dist - 1.346).abs() < 1e-2, "dist = {dist}, expected ~1.346");
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
        let rhf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let opt_config = OptimizeConfig { trust_radius: 0.1, ..Default::default() };
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
        const ANCHOR_ENERGY_BITS: u64 = 0xbff1e14dd9dd63b7;
        const ANCHOR_STEPS: usize = 7;
        assert_eq!(
            result.energy.to_bits(),
            ANCHOR_ENERGY_BITS,
            "post-refactor energy {:.15} != pre-refactor anchor {:.15}",
            result.energy,
            f64::from_bits(ANCHOR_ENERGY_BITS)
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
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1,
        ).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
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
        assert!((r1 - 1.869732).abs() < 1e-2, "r1 = {r1}, expected ~1.869732");
        assert!((r2 - 1.869732).abs() < 1e-2, "r2 = {r2}, expected ~1.869732");
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
        let uhf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_uhf(&ctx, &mol, "sto-3g", op, &uhf_config)
            .unwrap()
            .0;

        let result = optimize_geometry_uhf(&ctx, &mol, "sto-3g", op, &uhf_config, &opt_config).unwrap();

        assert!(result.converged, "UHF H2+ optimization did not converge in {} steps", result.steps);
        assert!(result.steps > 0, "optimizer should take at least one step from a stretched start");
        assert!(
            result.energy < e0,
            "optimized energy {} should be lower than initial energy {}",
            result.energy,
            e0
        );

        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!("H2+/UHF/STO-3G optimized distance: {:.6} Bohr, energy: {:.10} Ha", dist, result.energy);
        // Independent PySCF UHF/STO-3G geomeTRIC-optimizer reference (2026-07-21):
        //   dist = 2.004215 Bohr, E = -0.5826966474 Ha
        // ferric matches to ~1e-9 Ha / <1e-6 Bohr -- tightened from the old
        // loose +-0.3 Bohr sanity band now that a real reference exists.
        const PYSCF_DIST: f64 = 2.004215;
        const PYSCF_E: f64 = -0.5826966474;
        assert!((dist - PYSCF_DIST).abs() < 1e-3, "dist = {dist}, expected {PYSCF_DIST} (PySCF)");
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
        assert!(g_max < opt_config.g_max_thresh, "final |g|_max = {g_max:.3e} not converged");
    }

    #[test]
    fn test_optimize_oh_radical_rohf_sto3g() {
        // OH radical (doublet), started stretched at 1.3 Angstrom (vs the
        // experimental 0.9697 Angstrom in testdata/molecules/oh.xyz) and
        // optimized with ROHF/STO-3G analytical gradients.
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.3\n", 0, 2).unwrap();
        let op = Operator::coulomb();
        let rohf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_rohf(&ctx, &mol, "sto-3g", op, &rohf_config)
            .unwrap()
            .0;

        let result = optimize_geometry_rohf(&ctx, &mol, "sto-3g", op, &rohf_config, &opt_config).unwrap();

        assert!(result.converged, "ROHF OH optimization did not converge in {} steps", result.steps);
        assert!(result.steps > 0, "optimizer should take at least one step from a stretched start");
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
            compute_energy_and_gradient_rohf(&ctx, &result.mol, "sto-3g", op, &rohf_config).unwrap();
        let grad = flatten_gradient(&grad_arr);
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(g_max < opt_config.g_max_thresh, "final |g|_max = {g_max:.3e} not converged");
    }
}
