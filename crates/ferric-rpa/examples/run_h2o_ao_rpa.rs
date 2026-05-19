use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_rpa::ao_rpa::{ao_rpa_correlation_energy_minimax, pi_mo_dressed};
use ferric_rpa::quadrature::MinimaxJointQuadrature;
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_mp2::mo_transform::transform_3center_ov;
use std::time::Instant;
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;

#[derive(serde::Deserialize)]
struct MinimaxJson {
    tau_points: Vec<f64>,
    omega_points: Vec<f64>,
    omega_weights: Vec<f64>,
    w_transform: Vec<f64>,
}

fn main() {
    let ctx = ParallelContext::new();
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    
    // Use cc-pVDZ basis set
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    
    println!("Solving RHF for H2O/cc-pVDZ...");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
    println!("RHF Energy: {:.10} Hartree", rhf.energy);

    // Build the V^{-1/2} and AO ERI3 tensors
    println!("Building Integrals...");
    let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
    let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
    
    // Orbital energies and MO coefficients
    let nelec = mol.nelec() as usize;
    let nocc = nelec / 2;
    let nvir = obs.nbasis() - nocc;
    
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();
    let eps = rhf.eps_r();
    let eps_occ = eps[..nocc].to_vec();
    let eps_vir = eps[nocc..].to_vec();
    
    let eps_homo = eps_occ.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_lumo = eps_vir.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_min = eps_occ.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps_vir.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = eps_lumo - eps_homo;
    let ymax = eps_max - eps_min;
    let r = ymax / ymin;
    println!("Transition energy ranges for H2O/cc-pVDZ:");
    println!("  HOMO: {:.6}, LUMO: {:.6}", eps_homo, eps_lumo);
    println!("  Min orbital: {:.6}, Max orbital: {:.6}", eps_min, eps_max);
    println!("  ymin (gap): {:.6} Hartree", ymin);
    println!("  ymax:       {:.6} Hartree", ymax);
    println!("  R = ymax/ymin: {:.6}", r);

    // Load the Joint Minimax Quadrature Grid (in-tree data file).
    println!("Loading Joint Minimax Quadrature...");
    let grid_path = "crates/ferric-rpa/data/joint_minimax_N12.json";
    let file_content = std::fs::read_to_string(grid_path)
        .or_else(|_| std::fs::read_to_string("data/joint_minimax_N12.json"))
        .or_else(|_| std::fs::read_to_string("../../crates/ferric-rpa/data/joint_minimax_N12.json"))
        .unwrap_or_else(|_| panic!("could not find joint_minimax_N12.json (tried {grid_path} and relative variants)"));
        
    let grids: MinimaxJson = serde_json::from_str(&file_content).unwrap();
    let joint_grids = MinimaxJointQuadrature {
        tau_points: grids.tau_points.clone(),
        omega_points: grids.omega_points.clone(),
        omega_weights: grids.omega_weights.clone(),
        w_transform: grids.w_transform.clone(),
    };
    
    // 1) Run the Minimax AO-RPA Algorithm (with time-quadrature/AO assumption)
    println!("Running O(N^3) Minimax AO-RPA...");
    let start_ao = Instant::now();
    let (e_corr_ao, n_tau, n_omega) = ao_rpa_correlation_energy_minimax(
        &eri3_ao, &v_inv_sqrt, &c_occ, &c_vir, &eps_occ, &eps_vir, &joint_grids
    ).unwrap();
    let duration_ao = start_ao.elapsed();
    
    // 2) Run the Exact MO-RPA (without time-quadrature assumption, computing exact Pi(iω) in MO basis)
    println!("Running Exact MO-RPA for comparison...");
    let start_mo = Instant::now();
    
    // transform eri3_ao to MO basis: (P|ia)
    let eri3_mo = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
    let naux = dfbs.nbasis();
    let eri3_flat = eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap();
    let b_ov = v_inv_sqrt.dot(&eri3_flat);

    // Let's compare Pi(iomega_0) from AO minimax and MO exact!
    let omega_0 = grids.omega_points[0];
    let pi_mo = pi_mo_dressed(&b_ov, &eps_occ, &eps_vir, omega_0);
    
    // We can also evaluate pi_ao at omega_0 using the minimax stack
    let eri3_dressed = ferric_rpa::ao_rpa::dress_eri3_with_metric(&eri3_ao, &v_inv_sqrt);
    let laplace = ferric_quadrature::LaplaceQuadrature {
        points: joint_grids.tau_points.clone(),
        weights: vec![0.0; joint_grids.tau_points.len()],
        n_quad: joint_grids.tau_points.len(),
    };
    let chi0_stack = ferric_rpa::ao_rpa::chi0_ao_full_time(
        &eri3_dressed, &c_occ, &c_vir, &eps_occ, &eps_vir, &laplace
    ).unwrap();
    let pi_ao = ferric_rpa::ao_rpa::pi_ao_at_omega_minimax(&chi0_stack, &joint_grids, 0);

    println!("\n--- PI MATRIX COMPARISON AT OMEGA_0 = {:.6} ---", omega_0);
    println!("pi_mo (first 3x3):");
    for i in 0..3 {
        println!("  {:.6} {:.6} {:.6}", pi_mo[(i, 0)], pi_mo[(i, 1)], pi_mo[(i, 2)]);
    }
    println!("pi_ao minimax (first 3x3):");
    for i in 0..3 {
        println!("  {:.6} {:.6} {:.6}", pi_ao[(i, 0)], pi_ao[(i, 1)], pi_ao[(i, 2)]);
    }
    println!("Ratio pi_mo / pi_ao (first 3x3):");
    for i in 0..3 {
        println!("  {:.6} {:.6} {:.6}", pi_mo[(i, 0)] / pi_ao[(i, 0)], pi_mo[(i, 1)] / pi_ao[(i, 1)], pi_mo[(i, 2)] / pi_ao[(i, 2)]);
    }
    println!("-------------------------------------------------\n");

    // Integrate trace-log exactly over the same omega grid points
    let contribs: Result<Vec<f64>, ferric_core::FerricError> = grids.omega_points
        .par_iter()
        .zip(grids.omega_weights.par_iter())
        .map(|(&omega, &wk)| {
            let pi = pi_mo_dressed(&b_ov, &eps_occ, &eps_vir, omega);
            let mut eps_mat = pi;
            for p in 0..naux {
                eps_mat[(p, p)] += 1.0;
            }
            let (evals, _) = eps_mat.eigh(UPLO::Upper)
                .map_err(|e| ferric_core::FerricError::General(format!("MO-RPA eigh: {e}")))?;
            let contrib: f64 = evals.iter().map(|&lam| lam.ln() + (1.0 - lam)).sum();
            Ok(wk * contrib)
        })
        .collect();
        
    let e_corr_mo = contribs.unwrap().iter().sum::<f64>() / (2.0 * std::f64::consts::PI);
    let duration_mo = start_mo.elapsed();
    
    let diff = e_corr_ao - e_corr_mo;
    let rel_diff = diff.abs() / e_corr_mo.abs();
    
    println!("==================================================");
    println!("Quadrature Comparison Complete!");
    println!("Grid sizes: N_tau = {}, N_omega = {}", n_tau, n_omega);
    println!("Minimax AO-RPA Energy: {:.12} Hartree (Time: {:.2?})", e_corr_ao, duration_ao);
    println!("Exact MO-RPA Energy:   {:.12} Hartree (Time: {:.2?})", e_corr_mo, duration_mo);
    println!("--------------------------------------------------");
    println!("Quadrature Error (abs):  {:+.6e} Hartree", diff);
    println!("Quadrature Error (rel):  {:.6e} ({:.4}%)", rel_diff, rel_diff * 100.0);
    println!("==================================================");
}
