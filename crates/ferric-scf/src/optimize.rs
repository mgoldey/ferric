//! Geometry optimization using analytical gradients.
//!
//! Implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for
//! minimizing the molecular energy with respect to nuclear coordinates.

use crate::gradient::rhf_gradient;
use crate::rhf::{solve_rhf, RhfConfig};
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
#[derive(Debug)]
pub struct OptimizeResult {
    pub mol: Molecule,
    pub energy: f64,
    pub steps: usize,
    pub converged: bool,
}

/// Optimize the molecular geometry using RHF analytical gradients.
pub fn optimize_geometry(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    let mut current_mol = mol.clone();
    let natoms = current_mol.atoms.len();
    let n_coord = natoms * 3;
    
    // Initial energy and gradient
    let (mut energy, mut grad_arr) = compute_energy_and_gradient(ctx, &current_mol, basis_name, op, rhf_config)?;
    let mut grad = flatten_gradient(&grad_arr);
    
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
        let p_norm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        let step = if p_norm > opt_config.trust_radius {
            &p * (opt_config.trust_radius / p_norm)
        } else {
            p
        };

        // Update geometry
        update_molecule_coords(&mut current_mol, &step);
        
        let prev_grad = grad.clone();
        prev_energy = energy;
        
        // Compute new energy and gradient
        let res = compute_energy_and_gradient(ctx, &current_mol, basis_name, op, rhf_config)?;
        energy = res.0;
        grad_arr = res.1;
        grad = flatten_gradient(&grad_arr);
        
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

    Ok(OptimizeResult {
        mol: current_mol,
        energy,
        steps: step_idx,
        converged,
    })
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
    let grad = rhf_gradient(mol, &prep, op, &bounds, &res)?;
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

fn update_molecule_coords(mol: &mut Molecule, step: &Array1<f64>) {
    let mut idx = 0;
    for atom in mol.atoms.iter_mut() {
        atom.x += step[idx];
        atom.y += step[idx + 1];
        atom.zpos += step[idx + 2];
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
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n").unwrap();
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
}
