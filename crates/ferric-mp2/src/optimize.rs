//! RI-MP2-driven geometry optimization (BFGS in Cartesian coordinates).
//!
//! Mirrors `ferric_rpa::optimize::optimize_geometry_rpa` (which itself
//! mirrors `ferric_scf::optimize::optimize_geometry`) but uses
//! `total_rimp2_gradient` — the analytical RI-MP2 nuclear gradient — as the
//! energy+gradient driver instead of RPA's finite-difference correlation
//! gradient or RHF's analytical gradient.
//!
//! The BFGS update and trust-radius scaling are copied verbatim from the RPA
//! optimizer; only the energy + gradient evaluation differs. Keeping the
//! optimizer here (rather than extending `ferric-scf::optimize`) avoids
//! entangling the generic RHF-only driver with correlated-method configs,
//! and matches the precedent `ferric-rpa` already set for its own
//! energy/gradient pair.

use crate::gradient::total_rimp2_gradient_warm;
use crate::rimp2::RiMp2Config;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::OptimizeConfig;
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::{Array1, Array2};

/// Result of an RI-MP2 geometry optimization.
#[derive(Debug, Clone)]
#[must_use]
pub struct RiMp2OptimizeResult {
    pub mol: Molecule,
    pub energy: f64,
    pub steps: usize,
    pub converged: bool,
    /// Per-evaluation SCF warm-start records, in order (see
    /// `ferric_scf::warm_start`; every record is cold `Disabled` when
    /// `OptimizeConfig::scf_warm_start` is off).
    pub scf_warm_start: Vec<ferric_scf::warm_start::WarmStartRecord>,
}

/// Optimize a molecular geometry on the RI-MP2 PES using
/// `total_rimp2_gradient` (analytical Z-vector/relaxed-density RI-MP2
/// gradient, RHF + correlation together).
pub fn optimize_geometry_rimp2(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
) -> Result<RiMp2OptimizeResult, FerricError> {
    let mut current_mol = mol.clone();
    let natoms = current_mol.atoms.len();
    let n_coord = natoms * 3;

    // Each step's RHF starts from the previous step's converged orbitals
    // (`opt_config.scf_warm_start`, see `ferric_scf::warm_start`).
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let mut eval = |m: &Molecule| {
        total_rimp2_gradient_warm(m, obs_basis, aux_basis, op, mp2_config, ext, &mut warm)
    };
    let (mut energy, mut grad_arr) = eval(&current_mol)?;
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
        let de = if step_idx == 0 {
            0.0
        } else {
            energy - prev_energy
        };

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

        let (e_new, g_new) = eval(&current_mol)?;
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
                    h_inv[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
                }
            }
        } else {
            h_inv = Array2::eye(n_coord);
        }

        step_idx += 1;
    }

    if let Some(line) = warm.summary() {
        println!("{line}");
    }
    Ok(RiMp2OptimizeResult {
        mol: current_mol,
        energy,
        steps: step_idx,
        converged,
        scf_warm_start: warm.records().to_vec(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_optimize_h2_sto3g_rimp2() {
        // Start from a stretched bond: 1.0 Angstrom.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let obs_basis = basis::bundled("sto-3g").unwrap();
        let aux_basis = basis::bundled("cc-pvdz-ri").unwrap();
        let mp2_config = RiMp2Config {
            frozen_core: 0,
            memory_budget_bytes: None,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let result = optimize_geometry_rimp2(
            &mol,
            &obs_basis,
            &aux_basis,
            op,
            &mp2_config,
            &opt_config,
            None,
        )
        .unwrap();

        assert!(result.converged);
        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!("H2/STO-3G RI-MP2 optimized distance: {:.6} Bohr", dist);
        // RHF/STO-3G optimum is ~1.346 Bohr; RI-MP2 correlation should shift it
        // modestly (H2/STO-3G has only one virtual so MP2 correlation is small,
        // but the geometry must still be physically reasonable, not NaN/wild).
        assert!(dist.is_finite());
        assert!(
            (0.8..2.5).contains(&dist),
            "dist = {dist}, expected a reasonable H2 bond length"
        );
    }

    /// Warm vs cold RI-MP2 optimization: the same optimization within the
    /// optimizer's own tolerances (same step count, final energies within
    /// `e_conv`, final geometries within the optimizer's step resolution),
    /// warm start used after the first evaluation, and fewer SCF iterations.
    #[test]
    fn warm_start_rimp2_optimization_matches_cold() {
        let mol = Molecule::parse_xyz(
            "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n",
            0,
            1,
        )
        .unwrap();
        let obs = basis::bundled("sto-3g").unwrap();
        let aux = basis::bundled("cc-pvdz-ri").unwrap();
        let mp2 = RiMp2Config::default();
        let run = |warm: bool| {
            let opt = OptimizeConfig {
                scf_warm_start: warm,
                ..Default::default()
            };
            optimize_geometry_rimp2(&mol, &obs, &aux, Operator::coulomb(), &mp2, &opt, None)
                .unwrap()
        };
        let (w, c) = (run(true), run(false));
        let it = |r: &RiMp2OptimizeResult| {
            r.scf_warm_start
                .iter()
                .map(|x| x.iterations)
                .collect::<Vec<_>>()
        };
        let dx = w
            .mol
            .atoms
            .iter()
            .zip(&c.mol.atoms)
            .map(|(a, b)| {
                (a.x - b.x)
                    .abs()
                    .max((a.y - b.y).abs())
                    .max((a.zpos - b.zpos).abs())
            })
            .fold(0.0f64, f64::max);
        let de = (w.energy - c.energy).abs();
        eprintln!(
            "RI-MP2 optimize: steps warm {} cold {}; |dE| {de:.2e}; max|dx| {dx:.2e} Bohr; \
             SCF iterations warm {:?} cold {:?}",
            w.steps,
            c.steps,
            it(&w),
            it(&c)
        );
        let opt = OptimizeConfig::default();
        assert!(w.converged && c.converged);
        assert_eq!(w.steps, c.steps);
        assert!(de < opt.e_conv, "|dE| {de:e}");
        // Converged geometries agree to well inside one step of the size the
        // gradient threshold allows (|dx| ~ g_max / k, k ~ 0.5 Ha/Bohr^2).
        assert!(dx < opt.g_max_thresh, "max|dx| {dx:e}");
        assert_eq!(
            w.scf_warm_start[0].cold,
            Some(ferric_scf::warm_start::ColdReason::FirstGeometry)
        );
        assert!(w.scf_warm_start[1..].iter().all(|r| r.cold.is_none()));
        assert!(it(&w).iter().sum::<usize>() < it(&c).iter().sum::<usize>());
    }
}
