//! RPA-driven geometry optimization (BFGS in Cartesian coordinates).
//!
//! Mirrors `ferric_scf::optimize::optimize_geometry` but uses
//! `rpa_correlation_gradient` + `rhf_gradient` as the gradient driver.
//!
//! The BFGS update and trust-radius scaling are copied from the RHF
//! optimizer; only the energy + gradient evaluation differs.  Keeping the
//! optimizer here (rather than extending ferric-scf with an RPA call) avoids
//! a circular dependency (ferric-scf does not currently depend on
//! ferric-rpa).

use crate::gradient::total_rpa_gradient;
use crate::PdepRpaConfig;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::OptimizeConfig;
use ndarray::{Array1, Array2};

/// Result of an RPA geometry optimization.
#[derive(Debug)]
pub struct RpaOptimizeResult {
    pub mol: Molecule,
    pub energy: f64,
    pub steps: usize,
    pub converged: bool,
}

/// Optimize a molecular geometry on the RPA PES using
/// `total_rpa_gradient` (analytic RHF gradient + projection-fixed
/// finite-difference RPA correlation gradient).
///
/// `h_fd` controls the inner FD step for the correlation gradient (Bohr);
/// `5e-4` is a sensible default.
pub fn optimize_geometry_rpa(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rpa_config: &PdepRpaConfig,
    opt_config: &OptimizeConfig,
    h_fd: f64,
) -> Result<RpaOptimizeResult, FerricError> {
    let mut current_mol = mol.clone();
    let natoms = current_mol.atoms.len();
    let n_coord = natoms * 3;

    let (mut energy, mut grad_arr) =
        total_rpa_gradient(&current_mol, obs_basis, aux_basis, op, rpa_config, h_fd)?;
    let mut grad = flatten(&grad_arr);

    let mut h_inv = Array2::<f64>::eye(n_coord);
    let mut prev_energy = energy;
    let mut converged = false;
    let mut step_idx = 0;

    println!("Step | Energy (Ha) | Delta E | Max Grad | RMS Grad");
    println!("-----+--------------+----------+----------+---------");

    while step_idx < opt_config.max_steps {
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        let g_rms = (grad.iter().map(|g| g * g).sum::<f64>() / n_coord as f64).sqrt();
        let de = if step_idx == 0 { 0.0 } else { energy - prev_energy };

        println!(
            "{:4} | {:12.8} | {:+8.1e} | {:8.2e} | {:8.2e}",
            step_idx, energy, de, g_max, g_rms
        );

        if step_idx > 0
            && de.abs() < opt_config.e_conv
            && g_max < opt_config.g_max_thresh
            && g_rms < opt_config.g_rms_thresh
        {
            converged = true;
            break;
        }

        // BFGS step: p = -H_inv * g, trust-radius limited.
        let p = -h_inv.dot(&grad);
        let p_norm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        let step = if p_norm > opt_config.trust_radius {
            &p * (opt_config.trust_radius / p_norm)
        } else {
            p
        };

        apply_step(&mut current_mol, &step);

        let prev_grad = grad.clone();
        prev_energy = energy;

        let (e_new, g_new) =
            total_rpa_gradient(&current_mol, obs_basis, aux_basis, op, rpa_config, h_fd)?;
        energy = e_new;
        grad_arr = g_new;
        grad = flatten(&grad_arr);

        // BFGS inverse-Hessian update.
        let s = step;
        let y = &grad - &prev_grad;
        let ys = y.dot(&s);
        if ys.abs() > 1e-12 {
            let hy = h_inv.dot(&y);
            let yhy = y.dot(&hy);
            let rho = 1.0 / ys;
            let term1 = (ys + yhy) * rho * rho;
            for i in 0..n_coord {
                for j in 0..n_coord {
                    h_inv[(i, j)] +=
                        term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
                }
            }
        } else {
            h_inv = Array2::eye(n_coord);
        }

        step_idx += 1;
    }

    Ok(RpaOptimizeResult {
        mol: current_mol,
        energy,
        steps: step_idx,
        converged,
    })
}

fn flatten(g: &Array2<f64>) -> Array1<f64> {
    let mut flat = Array1::zeros(g.len());
    let mut k = 0;
    for i in 0..g.nrows() {
        for j in 0..3 {
            flat[k] = g[(i, j)];
            k += 1;
        }
    }
    flat
}

fn apply_step(mol: &mut Molecule, step: &Array1<f64>) {
    let mut k = 0;
    for atom in mol.atoms.iter_mut() {
        atom.x += step[k];
        atom.y += step[k + 1];
        atom.zpos += step[k + 2];
        k += 3;
    }
}
