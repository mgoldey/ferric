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

use crate::gradient::{exact_scf_jk, total_rimp2_gradient_on};
use crate::rimp2::RiMp2Config;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::OptimizeConfig;
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::{Array1, Array2};

/// Result of an RI-MP2 geometry optimization.
#[derive(Debug, Clone)]
#[must_use]
pub struct RiMp2OptimizeResult {
    /// Final geometry (coordinates in Bohr).
    pub mol: Molecule,
    /// RHF + RI-MP2 total energy at the final geometry (Hartree).
    pub energy: f64,
    /// BFGS steps taken.
    pub steps: usize,
    /// Whether the energy-change, max-gradient and RMS-gradient criteria of
    /// the `OptimizeConfig` were all met within `max_steps`.
    pub converged: bool,
    /// Per-evaluation SCF warm-start records, in order (see
    /// `ferric_scf::warm_start`; every record is cold `Disabled` when
    /// `OptimizeConfig::scf_warm_start` is off).
    pub scf_warm_start: Vec<ferric_scf::warm_start::WarmStartRecord>,
    /// Energy at every geometry the optimizer evaluated, in order (the
    /// starting geometry first).
    pub energy_trace: Vec<f64>,
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
    optimize_geometry_rimp2_on(
        mol,
        obs_basis,
        aux_basis,
        op,
        mp2_config,
        opt_config,
        ext,
        &exact_scf_jk(),
    )
}

/// [`optimize_geometry_rimp2`] with the reference SCF on the J/K route
/// `scf_jk` (`"exact"` or `"rijk"`; see
/// [`crate::gradient::total_rimp2_gradient_on`]).
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    let mut current_mol = mol.clone();
    let natoms = current_mol.atoms.len();
    let n_coord = natoms * 3;

    // Each step's RHF starts from the previous step's converged orbitals
    // (`opt_config.scf_warm_start`, see `ferric_scf::warm_start`).
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let mut eval = |m: &Molecule| {
        total_rimp2_gradient_on(
            m, obs_basis, aux_basis, op, mp2_config, ext, scf_jk, &mut warm,
        )
    };
    let (mut energy, mut grad_arr) = eval(&current_mol)?;
    let mut grad = flatten(&grad_arr);
    let mut energy_trace = vec![energy];

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
        energy_trace.push(e_new);
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
        energy_trace,
    })
}

/// The certificate's `gradient` line of an RI-MP2 optimization whose
/// reference SCF is on `scf_jk` (four-centre, or RI-JK on its aux).
pub fn rimp2_gradient_label(scf_jk: &ScfJk) -> String {
    match scf_jk.aux.as_deref() {
        Some(aux) => format!("analytic (Z-vector), RI-JK reference ({aux})"),
        None => "analytic (Z-vector)".to_string(),
    }
}

/// The certificate's `gradient` line of an OO-RI-MP2 optimization on the
/// reference J/K route of `rhf_config`: four-centre, or RI-JK on one
/// JK-fitting basis.
pub fn oo_rimp2_gradient_label(rhf_config: &ferric_scf::rhf::RhfConfig) -> String {
    let base = "analytic (zero-response: orbital-optimized, no Z-vector)";
    let route = ferric_scf::scf_jk::effective_jk(rhf_config);
    match (route.mode, route.j_aux) {
        ("rijk", Some(aux)) => format!("{base}, RI-JK reference ({aux})"),
        _ => base.to_string(),
    }
}

/// Why there is no SCS-MP2 geometry optimization.
///
/// [`crate::gradient::scs_mp2_gradient_analytical`] scales the whole MP2
/// part of the gradient by `(c_os + c_ss) / 2`, while the SCS-MP2 energy is
/// `E_HF + c_os E_os + c_ss E_ss`. The two agree only when `c_os = c_ss`; for
/// H2 (`E_ss = 0`) at Grimme's 6/5, 1/3 the energy's correlation part carries
/// 6/5 and the gradient's 23/30.
pub const SCS_MP2_NO_GRADIENT: &str =
    "SCS-MP2 has no geometry optimization: ferric's SCS-MP2 gradient scales the whole MP2 \
     gradient by (c_os + c_ss) / 2, which is not the derivative of the SCS-MP2 energy \
     (c_os E_os + c_ss E_ss) unless c_os = c_ss. Optimizable correlated methods: \"rimp2\" \
     and \"oo-rimp2\"";

/// Optimize a closed-shell geometry on the OO-RI-MP2 surface.
///
/// At every geometry: the RHF reference on `rhf_config` (warm-started per
/// `opt_config.scf_warm_start`), the orbital optimisation
/// ([`crate::oo_rimp2::oo_ri_mp2`] with `oo_config`) from it, and the
/// zero-response analytic gradient
/// ([`crate::oo_rimp2_gradient::oo_ri_mp2_gradient`]) at the converged
/// orbitals; the external potential of `rhf_config` enters the energy and the
/// gradient. The BFGS driver is
/// [`ferric_scf::optimize::optimize_geometry_with_energy_gradient`], the one
/// the SCF optimizers use.
///
/// The reference J/K route of `rhf_config` is the route of the whole surface:
/// four-centre, or RI-JK (`df_j_aux == df_k_aux`, the `scf_jk = "rijk"`
/// spelling), whose fitted J/K the orbital steps and the gradient reuse. The
/// RI-J tensor is stored in f64 on both (a geometry task,
/// [`ferric_scf::rhf::RhfConfig::for_geometry_task`]).
///
/// Refused before any integral: `oo_config.frozen_core > 0` (the frozen
/// core-active and core-virtual rotations are not optimised, so the
/// zero-response gradient is not the derivative of that energy), any other
/// fitted reference (RI-J with four-centre or COSX exchange, or RI-J and
/// RI-K on different bases: OO-MP2 has no route for them), and a PCM or
/// COSMO reaction field (OO-MP2 does not include one). An orbital
/// optimisation that does not converge at some geometry is an error: the
/// zero-response gradient holds only at a stationary point.
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_oo_rimp2(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
    opt_config: &OptimizeConfig,
) -> Result<ferric_scf::optimize::OptimizeResult, FerricError> {
    refuse_oo_rimp2_optimize(rhf_config, oo_config)?;
    let rhf_config = rhf_config.for_geometry_task();
    let ext = rhf_config.external_potential.as_ref();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = ferric_scf::optimize::optimize_geometry_with_energy_gradient(mol, opt_config, |m| {
        let obs = ferric_integrals::basis_bridge::PreparedBasis::new(m, obs_basis)?;
        let dfbs = ferric_integrals::basis_bridge::PreparedBasis::new(m, aux_basis)?;
        let bounds = ferric_scf::screening::SchwarzBounds::compute_for_screening(
            op,
            &obs,
            rhf_config.screening,
        )?;
        let rhf = ferric_scf::warm_start::solve_rhf_warm(
            &ctx,
            m,
            &obs,
            op,
            &bounds,
            &rhf_config,
            &mut warm,
        )?;
        if !rhf.converged {
            return Err(FerricError::ScfConvergence {
                iterations: rhf.iterations,
                last_energy: rhf.energy,
            });
        }
        let oo = crate::oo_rimp2::oo_ri_mp2(m, &obs, &dfbs, op, &bounds, &rhf, oo_config, ext)?;
        if !oo.converged {
            return Err(FerricError::General(format!(
                "OO-RI-MP2 optimize: the orbital optimisation did not converge at this \
                 geometry ({} iterations, orbital gradient norm {:.2e}); the zero-response \
                 gradient holds only at a stationary point",
                oo.iterations, oo.grad_norm
            )));
        }
        let g =
            crate::oo_rimp2_gradient::oo_ri_mp2_gradient(m, &obs, &dfbs, op, &bounds, &oo, 0, ext)?;
        Ok((oo.total_energy, g))
    });
    if let Some(line) = warm.summary() {
        println!("{line}");
    }
    out
}

/// What [`optimize_geometry_oo_rimp2`] refuses before any integral.
fn refuse_oo_rimp2_optimize(
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
) -> Result<(), FerricError> {
    let refuse = |why: String| Err(FerricError::General(format!("OO-RI-MP2 optimize: {why}")));
    if oo_config.frozen_core > 0 {
        return refuse(format!(
            "frozen_core = {} is not supported: the core-active and core-virtual rotations \
             are not optimised, so the zero-response gradient is not the derivative of the \
             frozen-core energy; use frozen_core = 0",
            oo_config.frozen_core
        ));
    }
    let route = ferric_scf::scf_jk::effective_jk(rhf_config);
    if !matches!(route.mode, "exact" | "rijk") {
        return refuse(format!(
            "the reference SCF J/K route is \"{}\"; {}",
            route.mode,
            ferric_scf::scf_jk::OO_BUILDS_EXACT_OR_RIJK
        ));
    }
    if rhf_config.pcm.is_some() || rhf_config.cosmo.is_some() {
        return refuse(
            "a PCM/COSMO reaction field is not supported: OO-RI-MP2 has no solvation term"
                .to_string(),
        );
    }
    Ok(())
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

    /// `optimize_geometry_oo_rimp2` refuses, before any integral, the setups
    /// its zero-response gradient does not differentiate; the default
    /// (four-centre) and the RI-JK reference pass the same check.
    #[test]
    fn oo_rimp2_optimize_refuses_what_it_does_not_differentiate() {
        use crate::oo_rimp2::OoRiMp2Config;
        use ferric_scf::rhf::RhfConfig;
        let refused = |rhf: &RhfConfig, oo: &OoRiMp2Config, needle: &str| {
            let r = refuse_oo_rimp2_optimize(rhf, oo);
            assert!(
                matches!(&r, Err(e) if e.to_string().contains(needle)),
                "{needle}: {r:?}"
            );
        };
        let (rhf, oo) = (RhfConfig::default(), OoRiMp2Config::default());
        assert!(refuse_oo_rimp2_optimize(&rhf, &oo).is_ok());
        let frozen = OoRiMp2Config {
            frozen_core: 1,
            ..OoRiMp2Config::default()
        };
        refused(&rhf, &frozen, "frozen_core = 1");
        let rijk = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..RhfConfig::default()
        };
        assert!(refuse_oo_rimp2_optimize(&rijk, &oo).is_ok());
        assert!(oo_rimp2_gradient_label(&rijk).ends_with("RI-JK reference (def2-universal-jkfit)"));
        let rijcosx = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            k_builder: Some("cosx".into()),
            ..RhfConfig::default()
        };
        refused(&rijcosx, &oo, "\"rijcosx\"");
        let split = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("cc-pvtz-jkfit".into()),
            ..RhfConfig::default()
        };
        refused(&split, &oo, "\"mixed\"");
        let cosmo = RhfConfig {
            cosmo: Some(ferric_scf::cosmo::CosmoConfig::default()),
            ..RhfConfig::default()
        };
        refused(&cosmo, &oo, "PCM/COSMO");
    }

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
