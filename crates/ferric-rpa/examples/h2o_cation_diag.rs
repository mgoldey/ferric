//! Diagnose ferric UHF on H2O+ vs PySCF. Print E, ⟨S²⟩, MO energies.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess, UhfConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};

fn s_squared(c_a: &ndarray::Array2<f64>, c_b: &ndarray::Array2<f64>, s: &ndarray::Array2<f64>, na: usize, nb: usize) -> f64 {
    let st = 0.5 * (na as f64 - nb as f64);
    let ideal = st * (st + 1.0);
    if na == 0 || nb == 0 { return ideal; }
    let ca = c_a.slice(ndarray::s![.., ..na]);
    let cb = c_b.slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    let ss: f64 = ov.iter().map(|v| v*v).sum();
    ideal + (nb as f64) - ss
}

fn main() {
    let ctx = ParallelContext::default();
    let xyz = "3\nH2O+\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let cation = Molecule::parse_xyz(xyz, 1, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&cation, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    // Neutral RHF first to get directed guess.
    {
        let neutral = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let prep_n = PreparedBasis::new(&neutral, &bs).unwrap();
        let bounds_n = SchwarzBounds::compute(op, &prep_n).unwrap();
        let rhf = solve_rhf(&ctx, &neutral, &prep_n, op, &bounds_n, &RhfConfig::default()).unwrap();
        // Reuse neutral RHF MOs as both α and β starting points for cation UHF.
        let c_rhf = rhf.mos_alpha.clone();
        let cfg = UhfConfig::default();
        let res = solve_uhf_with_guess(&ctx, &cation, &prep, op, &bounds, &cfg, Some((&c_rhf, &c_rhf))).unwrap();
        println!("[neutral-RHF-guess UHF] E={:.8}, conv={}, iters={}", res.energy, res.converged, res.iterations);
    }
    println!();

    // Run with very loose convergence to see iter-1 / iter-2 / etc. trajectory.
    for max_iter in [1usize, 2, 3, 5, 10, 20, 50] {
        let cfg = UhfConfig { max_iter, energy_conv: 1e-20, density_conv: 1e-20, ..Default::default() };
        match solve_uhf(&ctx, &cation, &prep, op, &bounds, &cfg) {
            Ok(r) => println!("[max_iter={:>3}] E={:.6}, conv={}, iters={}", max_iter, r.energy, r.converged, r.iterations),
            Err(e) => match e {
                ferric_core::FerricError::ScfConvergence { iterations, last_energy } => {
                    println!("[max_iter={:>3}] (DNF) last_E={:.6}, iters={}", max_iter, last_energy, iterations);
                }
                _ => println!("[max_iter={:>3}] ERR: {:?}", max_iter, e),
            }
        }
    }
    println!("\nPySCF cycle trajectory (1e/hcore guess):");
    println!("  cycle  1: E = -73.3065");
    println!("  cycle  2: E = -75.0341");
    println!("  cycle  3: E = -75.6134");
    println!("  cycle  4: E = -75.6311");
    println!("  cycle 11: E = -75.6318  (converged)");
    println!();
    println!();
    let cfg = UhfConfig { max_iter: 500, energy_conv: 1e-11, density_conv: 1e-9, ..Default::default() };
    let res = solve_uhf(&ctx, &cation, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("final unwrap: {e:?}"));
    let s_ao = oneelectron::overlap(&prep);
    let s2 = s_squared(&res.mos_alpha, res.mos_beta.as_ref().unwrap(), &s_ao, 5, 4);
    println!("ferric UHF H2O+ cc-pVDZ:");
    println!("  E_tot   = {:.8}", res.energy);
    println!("  conv    = {}, iters = {}", res.converged, res.iterations);
    println!("  <S^2>   = {:.4} (ideal 0.75)", s2);
    println!("  α eps[0..6] = {:?}", &res.eps_alpha[..6.min(res.eps_alpha.len())]);
    println!("  β eps[0..6] = {:?}", &res.eps_beta.as_ref().unwrap()[..6.min(res.eps_beta.as_ref().unwrap().len())]);
    println!("\nPySCF reference: E = -75.631774, <S^2> = 0.7561");
    println!("  α eps[0..6] = [-21.140, -1.914, -1.218, -1.130, -1.098, -0.142]");
    println!("  β eps[0..6] = [-21.095, -1.758, -1.179, -1.047, -0.314, -0.128]");
}
